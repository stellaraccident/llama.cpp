#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct pyre_block_q4_K {
    unsigned short d;
    unsigned short dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

static __device__ __forceinline__ void pyre_get_scale_min_k4(
        int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

template <int WG_SIZE>
static __device__ __forceinline__ float pyre_reduce_wg(float sum, float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (WG_SIZE <= warpSize) {
        return sum;
    }
    if (lane == 0) {
        shared[wave] = sum;
    }
    __syncthreads();

    sum = lane < ((WG_SIZE + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset);
        }
    }
    return sum;
}

template <int WG_SIZE>
static __device__ __forceinline__ void pyre_mul_mat_vec_q4_k_f32_impl(
        const pyre_block_q4_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    __shared__ float sumsh[WG_SIZE / 32];

    const long long blocks_per_row = k / 256;
    const pyre_block_q4_K * row_blocks = src0 + row * blocks_per_row;
    const float * src1_col = src1 + col * k;
    float sum = 0.0f;

    const int block_lane = tid & 63;
    const int block_slot = tid >> 6;
    const int block_stride = WG_SIZE >> 6;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += block_stride) {
        const pyre_block_q4_K * block = row_blocks + block_idx;

        uint8_t sc = 0;
        uint8_t m = 0;
        pyre_get_scale_min_k4(group, block->scales, &sc, &m);

        const float d = __half2float(__ushort_as_half(block->d)) * static_cast<float>(sc);
        const float min = __half2float(__ushort_as_half(block->dmin)) * static_cast<float>(m);
        const long long src_base = block_idx * 256 + group * 32 + lane;
        const int qs_base = (group >> 1) * 32 + lane;

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint8_t packed = block->qs[qs_base + j];
            const float q = (group & 1) ?
                static_cast<float>(packed >> 4) :
                static_cast<float>(packed & 0x0F);
            sum += (d * q - min) * src1_col[src_base + j];
        }
    }

    sum = pyre_reduce_wg<WG_SIZE>(sum, sumsh);

    if (tid == 0) {
        dst[col * rows + row] = sum;
    }
}

extern "C" __global__ void pyre_mul_mat_vec_q4_k_f32(
        const pyre_block_q4_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    pyre_mul_mat_vec_q4_k_f32_impl<256>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void pyre_mul_mat_vec_q4_k_wg128_f32(
        const pyre_block_q4_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    pyre_mul_mat_vec_q4_k_f32_impl<128>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void pyre_mul_mat_vec_q4_k_wg64_f32(
        const pyre_block_q4_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    pyre_mul_mat_vec_q4_k_f32_impl<64>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void pyre_mul_mat_vec_q4_k_packed_wg64_f32(
        const pyre_block_q4_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    __shared__ float sumsh[2];

    const long long blocks_per_row = k / 256;
    const pyre_block_q4_K * row_blocks = src0 + row * blocks_per_row;
    const float * src1_col = src1 + col * k;
    float sum = 0.0f;

    const int itid = tid & 15;
    const int block_slot = tid >> 4;
    const int il = itid >> 2;
    const int ir = itid - 4 * il;
    const int v_im = il >> 1;
    const int v_in = il & 1;
    const int l0 = 4 * (2 * ir + v_in);
    const int q_offset = 32 * v_im + l0;
    const int y_offset = 64 * v_im + l0;
    const int g0 = 2 * v_im;
    const int g1 = g0 + 1;
    const int g2 = g0 + 4;
    const int g3 = g2 + 1;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 4) {
        const pyre_block_q4_K * block = row_blocks + block_idx;

        uint8_t sc0 = 0;
        uint8_t m0 = 0;
        uint8_t sc1 = 0;
        uint8_t m1 = 0;
        uint8_t sc2 = 0;
        uint8_t m2 = 0;
        uint8_t sc3 = 0;
        uint8_t m3 = 0;
        pyre_get_scale_min_k4(g0, block->scales, &sc0, &m0);
        pyre_get_scale_min_k4(g1, block->scales, &sc1, &m1);
        pyre_get_scale_min_k4(g2, block->scales, &sc2, &m2);
        pyre_get_scale_min_k4(g3, block->scales, &sc3, &m3);

        const float d = __half2float(__ushort_as_half(block->d));
        const float dmin = __half2float(__ushort_as_half(block->dmin));
        const float d0 = d * static_cast<float>(sc0);
        const float d1 = d * static_cast<float>(sc1);
        const float d2 = d * static_cast<float>(sc2);
        const float d3 = d * static_cast<float>(sc3);
        const float min0 = dmin * static_cast<float>(m0);
        const float min1 = dmin * static_cast<float>(m1);
        const float min2 = dmin * static_cast<float>(m2);
        const float min3 = dmin * static_cast<float>(m3);

        const long long src_base = block_idx * 256 + y_offset;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint8_t q01 = block->qs[q_offset + j];
            const uint8_t q23 = block->qs[q_offset + 64 + j];
            const float y0 = src1_col[src_base + j];
            const float y1 = src1_col[src_base + 32 + j];
            const float y2 = src1_col[src_base + 128 + j];
            const float y3 = src1_col[src_base + 160 + j];
            sum += (d0 * static_cast<float>(q01 & 0x0F) - min0) * y0;
            sum += (d1 * static_cast<float>(q01 >> 4) - min1) * y1;
            sum += (d2 * static_cast<float>(q23 & 0x0F) - min2) * y2;
            sum += (d3 * static_cast<float>(q23 >> 4) - min3) * y3;
        }
    }

    sum = pyre_reduce_wg<64>(sum, sumsh);

    if (tid == 0) {
        dst[col * rows + row] = sum;
    }
}
