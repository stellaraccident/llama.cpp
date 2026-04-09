#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct pyre_block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    unsigned short d;
};

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

static __device__ __forceinline__ int pyre_q6_k_value(
        const pyre_block_q6_K * block, int in_block) {
    const int half = in_block / 128;
    const int idx = in_block - half * 128;
    const int lane = idx & 31;
    const int ql_base = half * 64;
    const int qh_base = half * 32;

    int q = 0;
    if (idx < 32) {
        q = (block->ql[ql_base + lane] & 0x0F) | (((block->qh[qh_base + lane] >> 0) & 3) << 4);
    } else if (idx < 64) {
        q = (block->ql[ql_base + lane + 32] & 0x0F) | (((block->qh[qh_base + lane] >> 2) & 3) << 4);
    } else if (idx < 96) {
        q = (block->ql[ql_base + lane] >> 4) | (((block->qh[qh_base + lane] >> 4) & 3) << 4);
    } else {
        q = (block->ql[ql_base + lane + 32] >> 4) | (((block->qh[qh_base + lane] >> 6) & 3) << 4);
    }

    return q - 32;
}

static __device__ __forceinline__ int pyre_q6_k_scale(
        const pyre_block_q6_K * block,
        int group,
        int lane) {
    const int half = group >> 2;
    const int group_in_half = group & 3;
    return static_cast<int>(block->scales[half * 8 + group_in_half * 2 + lane / 16]);
}

template <int WG_SIZE>
static __device__ __forceinline__ void pyre_mul_mat_vec_q6_k_f32_impl(
        const pyre_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    __shared__ float sumsh[WG_SIZE / 32];

    const long long blocks_per_row = k / 256;
    const pyre_block_q6_K * row_blocks = src0 + row * blocks_per_row;
    const float * src1_col = src1 + col * k;
    float sum = 0.0f;

    const int block_lane = tid & 63;
    const int block_slot = tid >> 6;
    const int block_stride = WG_SIZE >> 6;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const int in_block_base = group * 32 + lane;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += block_stride) {
        const pyre_block_q6_K * block = row_blocks + block_idx;
        const long long src_base = block_idx * 256 + in_block_base;
        const float d = __half2float(__ushort_as_half(block->d)) *
            static_cast<float>(pyre_q6_k_scale(block, group, lane));

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            sum += d * static_cast<float>(pyre_q6_k_value(block, in_block_base + j)) * src1_col[src_base + j];
        }
    }

    sum = pyre_reduce_wg<WG_SIZE>(sum, sumsh);

    if (tid == 0) {
        dst[col * rows + row] = sum;
    }
}

extern "C" __global__ void pyre_mul_mat_vec_q6_k_f32(
        const pyre_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    pyre_mul_mat_vec_q6_k_f32_impl<256>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void pyre_mul_mat_vec_q6_k_wg128_f32(
        const pyre_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    pyre_mul_mat_vec_q6_k_f32_impl<128>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void pyre_mul_mat_vec_q6_k_wg64_f32(
        const pyre_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    pyre_mul_mat_vec_q6_k_f32_impl<64>(src0, src1, dst, k, rows, cols);
}
