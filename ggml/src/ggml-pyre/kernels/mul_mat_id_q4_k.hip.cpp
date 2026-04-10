#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct pyre_block_q4_K_id {
    unsigned short d;
    unsigned short dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct pyre_mul_mat_id_q4_k_constants {
    long long k;
    long long rows;
    long long n_ids;
    long long n_tokens;
    long long n_experts;
    long long src0_nb1;
    long long src0_nb2;
    long long src1_nb1;
    long long src1_nb2;
    long long ids_nb0;
    long long ids_nb1;
    long long dst_nb1;
    long long dst_nb2;
};

static __device__ __forceinline__ void pyre_get_scale_min_k4_id(
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
static __device__ __forceinline__ void pyre_mul_mat_id_q4_k_f32_impl(
        const pyre_block_q4_K_id * src0, const float * src1, const int * ids, float * dst,
        pyre_mul_mat_id_q4_k_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.rows) {
        return;
    }

    const long long id_pos = outer % c.n_ids;
    const long long token = outer / c.n_ids;
    if (token >= c.n_tokens) {
        return;
    }

    const int expert = *reinterpret_cast<const int *>(
        reinterpret_cast<const char *>(ids) + id_pos * c.ids_nb0 + token * c.ids_nb1);
    if (expert < 0 || expert >= c.n_experts) {
        return;
    }

    __shared__ float sumsh[WG_SIZE / 32];
    const char * src0_row_base = reinterpret_cast<const char *>(src0) + expert * c.src0_nb2 + row * c.src0_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;
    float sum = 0.0f;

    const int block_lane = tid & 63;
    const int block_slot = tid >> 6;
    const int block_stride = WG_SIZE >> 6;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += block_stride) {
        const pyre_block_q4_K_id * block = reinterpret_cast<const pyre_block_q4_K_id *>(
            src0_row_base + block_idx * sizeof(pyre_block_q4_K_id));

        uint8_t sc = 0;
        uint8_t m = 0;
        pyre_get_scale_min_k4_id(group, block->scales, &sc, &m);

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
            const float b = *reinterpret_cast<const float *>(src1_col + (src_base + j) * sizeof(float));
            sum += (d * q - min) * b;
        }
    }

    sum = pyre_reduce_wg<WG_SIZE>(sum, sumsh);

    if (tid == 0) {
        *reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst) + row * sizeof(float) + id_pos * c.dst_nb1 + token * c.dst_nb2) = sum;
    }
}

extern "C" __global__ void pyre_mul_mat_id_q4_k_f32(
        const pyre_block_q4_K_id * src0, const float * src1, const int * ids, float * dst,
        pyre_mul_mat_id_q4_k_constants c) {
    pyre_mul_mat_id_q4_k_f32_impl<256>(src0, src1, ids, dst, c);
}

extern "C" __global__ void pyre_mul_mat_id_q4_k_wg128_f32(
        const pyre_block_q4_K_id * src0, const float * src1, const int * ids, float * dst,
        pyre_mul_mat_id_q4_k_constants c) {
    pyre_mul_mat_id_q4_k_f32_impl<128>(src0, src1, ids, dst, c);
}

extern "C" __global__ void pyre_mul_mat_id_q4_k_wg64_f32(
        const pyre_block_q4_K_id * src0, const float * src1, const int * ids, float * dst,
        pyre_mul_mat_id_q4_k_constants c) {
    pyre_mul_mat_id_q4_k_f32_impl<64>(src0, src1, ids, dst, c);
}

extern "C" __global__ void pyre_mul_mat_id_q4_k_row4_wg64_f32(
        const pyre_block_q4_K_id * src0, const float * src1, const int * ids, float * dst,
        pyre_mul_mat_id_q4_k_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 4;
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 + 3 >= c.rows) {
        return;
    }

    const long long id_pos = outer % c.n_ids;
    const long long token = outer / c.n_ids;
    if (token >= c.n_tokens) {
        return;
    }

    const int expert = *reinterpret_cast<const int *>(
        reinterpret_cast<const char *>(ids) + id_pos * c.ids_nb0 + token * c.ids_nb1);
    if (expert < 0 || expert >= c.n_experts) {
        return;
    }

    __shared__ float sumsh[64 / 32];
    const char * src0_expert_base = reinterpret_cast<const char *>(src0) + expert * c.src0_nb2;
    const char * src0_row0_base = src0_expert_base + row0 * c.src0_nb1;
    const char * src0_row1_base = src0_row0_base + c.src0_nb1;
    const char * src0_row2_base = src0_row1_base + c.src0_nb1;
    const char * src0_row3_base = src0_row2_base + c.src0_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    const int block_lane = tid & 63;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;

    for (long long block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const pyre_block_q4_K_id * block0 = reinterpret_cast<const pyre_block_q4_K_id *>(
            src0_row0_base + block_idx * sizeof(pyre_block_q4_K_id));
        const pyre_block_q4_K_id * block1 = reinterpret_cast<const pyre_block_q4_K_id *>(
            src0_row1_base + block_idx * sizeof(pyre_block_q4_K_id));
        const pyre_block_q4_K_id * block2 = reinterpret_cast<const pyre_block_q4_K_id *>(
            src0_row2_base + block_idx * sizeof(pyre_block_q4_K_id));
        const pyre_block_q4_K_id * block3 = reinterpret_cast<const pyre_block_q4_K_id *>(
            src0_row3_base + block_idx * sizeof(pyre_block_q4_K_id));

        uint8_t sc0 = 0;
        uint8_t sc1 = 0;
        uint8_t sc2 = 0;
        uint8_t sc3 = 0;
        uint8_t m0 = 0;
        uint8_t m1 = 0;
        uint8_t m2 = 0;
        uint8_t m3 = 0;
        pyre_get_scale_min_k4_id(group, block0->scales, &sc0, &m0);
        pyre_get_scale_min_k4_id(group, block1->scales, &sc1, &m1);
        pyre_get_scale_min_k4_id(group, block2->scales, &sc2, &m2);
        pyre_get_scale_min_k4_id(group, block3->scales, &sc3, &m3);

        const float d0 = __half2float(__ushort_as_half(block0->d)) * static_cast<float>(sc0);
        const float d1 = __half2float(__ushort_as_half(block1->d)) * static_cast<float>(sc1);
        const float d2 = __half2float(__ushort_as_half(block2->d)) * static_cast<float>(sc2);
        const float d3 = __half2float(__ushort_as_half(block3->d)) * static_cast<float>(sc3);
        const float min0 = __half2float(__ushort_as_half(block0->dmin)) * static_cast<float>(m0);
        const float min1 = __half2float(__ushort_as_half(block1->dmin)) * static_cast<float>(m1);
        const float min2 = __half2float(__ushort_as_half(block2->dmin)) * static_cast<float>(m2);
        const float min3 = __half2float(__ushort_as_half(block3->dmin)) * static_cast<float>(m3);
        const long long src_base = block_idx * 256 + group * 32 + lane;
        const int qs_base = (group >> 1) * 32 + lane;

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float b = *reinterpret_cast<const float *>(src1_col + (src_base + j) * sizeof(float));
            const uint8_t packed0 = block0->qs[qs_base + j];
            const uint8_t packed1 = block1->qs[qs_base + j];
            const uint8_t packed2 = block2->qs[qs_base + j];
            const uint8_t packed3 = block3->qs[qs_base + j];
            const float q0 = (group & 1) ? static_cast<float>(packed0 >> 4) : static_cast<float>(packed0 & 0x0F);
            const float q1 = (group & 1) ? static_cast<float>(packed1 >> 4) : static_cast<float>(packed1 & 0x0F);
            const float q2 = (group & 1) ? static_cast<float>(packed2 >> 4) : static_cast<float>(packed2 & 0x0F);
            const float q3 = (group & 1) ? static_cast<float>(packed3 >> 4) : static_cast<float>(packed3 & 0x0F);
            sum0 += (d0 * q0 - min0) * b;
            sum1 += (d1 * q1 - min1) * b;
            sum2 += (d2 * q2 - min2) * b;
            sum3 += (d3 * q3 - min3) * b;
        }
    }

    sum0 = pyre_reduce_wg<64>(sum0, sumsh);
    __syncthreads();
    sum1 = pyre_reduce_wg<64>(sum1, sumsh);
    __syncthreads();
    sum2 = pyre_reduce_wg<64>(sum2, sumsh);
    __syncthreads();
    sum3 = pyre_reduce_wg<64>(sum3, sumsh);

    if (tid == 0) {
        char * dst_base = reinterpret_cast<char *>(dst) + id_pos * c.dst_nb1 + token * c.dst_nb2;
        *reinterpret_cast<float *>(dst_base + row0 * sizeof(float)) = sum0;
        *reinterpret_cast<float *>(dst_base + (row0 + 1) * sizeof(float)) = sum1;
        *reinterpret_cast<float *>(dst_base + (row0 + 2) * sizeof(float)) = sum2;
        *reinterpret_cast<float *>(dst_base + (row0 + 3) * sizeof(float)) = sum3;
    }
}

extern "C" __global__ void pyre_mul_mat_id_q4_k_row8_wg64_f32(
        const pyre_block_q4_K_id * src0, const float * src1, const int * ids, float * dst,
        pyre_mul_mat_id_q4_k_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 8;
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 + 7 >= c.rows) {
        return;
    }

    const long long id_pos = outer % c.n_ids;
    const long long token = outer / c.n_ids;
    if (token >= c.n_tokens) {
        return;
    }

    const int expert = *reinterpret_cast<const int *>(
        reinterpret_cast<const char *>(ids) + id_pos * c.ids_nb0 + token * c.ids_nb1);
    if (expert < 0 || expert >= c.n_experts) {
        return;
    }

    __shared__ float sumsh[64 / 32];
    const char * src0_expert_base = reinterpret_cast<const char *>(src0) + expert * c.src0_nb2;
    const char * src0_row0_base = src0_expert_base + row0 * c.src0_nb1;
    const char * src0_row1_base = src0_row0_base + c.src0_nb1;
    const char * src0_row2_base = src0_row1_base + c.src0_nb1;
    const char * src0_row3_base = src0_row2_base + c.src0_nb1;
    const char * src0_row4_base = src0_row3_base + c.src0_nb1;
    const char * src0_row5_base = src0_row4_base + c.src0_nb1;
    const char * src0_row6_base = src0_row5_base + c.src0_nb1;
    const char * src0_row7_base = src0_row6_base + c.src0_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;

    const int block_lane = tid & 63;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;

    for (long long block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const pyre_block_q4_K_id * block0 = reinterpret_cast<const pyre_block_q4_K_id *>(
            src0_row0_base + block_idx * sizeof(pyre_block_q4_K_id));
        const pyre_block_q4_K_id * block1 = reinterpret_cast<const pyre_block_q4_K_id *>(
            src0_row1_base + block_idx * sizeof(pyre_block_q4_K_id));
        const pyre_block_q4_K_id * block2 = reinterpret_cast<const pyre_block_q4_K_id *>(
            src0_row2_base + block_idx * sizeof(pyre_block_q4_K_id));
        const pyre_block_q4_K_id * block3 = reinterpret_cast<const pyre_block_q4_K_id *>(
            src0_row3_base + block_idx * sizeof(pyre_block_q4_K_id));
        const pyre_block_q4_K_id * block4 = reinterpret_cast<const pyre_block_q4_K_id *>(
            src0_row4_base + block_idx * sizeof(pyre_block_q4_K_id));
        const pyre_block_q4_K_id * block5 = reinterpret_cast<const pyre_block_q4_K_id *>(
            src0_row5_base + block_idx * sizeof(pyre_block_q4_K_id));
        const pyre_block_q4_K_id * block6 = reinterpret_cast<const pyre_block_q4_K_id *>(
            src0_row6_base + block_idx * sizeof(pyre_block_q4_K_id));
        const pyre_block_q4_K_id * block7 = reinterpret_cast<const pyre_block_q4_K_id *>(
            src0_row7_base + block_idx * sizeof(pyre_block_q4_K_id));

        uint8_t sc0 = 0;
        uint8_t sc1 = 0;
        uint8_t sc2 = 0;
        uint8_t sc3 = 0;
        uint8_t sc4 = 0;
        uint8_t sc5 = 0;
        uint8_t sc6 = 0;
        uint8_t sc7 = 0;
        uint8_t m0 = 0;
        uint8_t m1 = 0;
        uint8_t m2 = 0;
        uint8_t m3 = 0;
        uint8_t m4 = 0;
        uint8_t m5 = 0;
        uint8_t m6 = 0;
        uint8_t m7 = 0;
        pyre_get_scale_min_k4_id(group, block0->scales, &sc0, &m0);
        pyre_get_scale_min_k4_id(group, block1->scales, &sc1, &m1);
        pyre_get_scale_min_k4_id(group, block2->scales, &sc2, &m2);
        pyre_get_scale_min_k4_id(group, block3->scales, &sc3, &m3);
        pyre_get_scale_min_k4_id(group, block4->scales, &sc4, &m4);
        pyre_get_scale_min_k4_id(group, block5->scales, &sc5, &m5);
        pyre_get_scale_min_k4_id(group, block6->scales, &sc6, &m6);
        pyre_get_scale_min_k4_id(group, block7->scales, &sc7, &m7);

        const float d0 = __half2float(__ushort_as_half(block0->d)) * static_cast<float>(sc0);
        const float d1 = __half2float(__ushort_as_half(block1->d)) * static_cast<float>(sc1);
        const float d2 = __half2float(__ushort_as_half(block2->d)) * static_cast<float>(sc2);
        const float d3 = __half2float(__ushort_as_half(block3->d)) * static_cast<float>(sc3);
        const float d4 = __half2float(__ushort_as_half(block4->d)) * static_cast<float>(sc4);
        const float d5 = __half2float(__ushort_as_half(block5->d)) * static_cast<float>(sc5);
        const float d6 = __half2float(__ushort_as_half(block6->d)) * static_cast<float>(sc6);
        const float d7 = __half2float(__ushort_as_half(block7->d)) * static_cast<float>(sc7);
        const float min0 = __half2float(__ushort_as_half(block0->dmin)) * static_cast<float>(m0);
        const float min1 = __half2float(__ushort_as_half(block1->dmin)) * static_cast<float>(m1);
        const float min2 = __half2float(__ushort_as_half(block2->dmin)) * static_cast<float>(m2);
        const float min3 = __half2float(__ushort_as_half(block3->dmin)) * static_cast<float>(m3);
        const float min4 = __half2float(__ushort_as_half(block4->dmin)) * static_cast<float>(m4);
        const float min5 = __half2float(__ushort_as_half(block5->dmin)) * static_cast<float>(m5);
        const float min6 = __half2float(__ushort_as_half(block6->dmin)) * static_cast<float>(m6);
        const float min7 = __half2float(__ushort_as_half(block7->dmin)) * static_cast<float>(m7);
        const long long src_base = block_idx * 256 + group * 32 + lane;
        const int qs_base = (group >> 1) * 32 + lane;

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float b = *reinterpret_cast<const float *>(src1_col + (src_base + j) * sizeof(float));
#define PYRE_Q4K_ROW8_ACC(N) \
            do { \
                const uint8_t packed = block##N->qs[qs_base + j]; \
                const float q = (group & 1) ? static_cast<float>(packed >> 4) : static_cast<float>(packed & 0x0F); \
                sum##N += (d##N * q - min##N) * b; \
            } while (0)
            PYRE_Q4K_ROW8_ACC(0);
            PYRE_Q4K_ROW8_ACC(1);
            PYRE_Q4K_ROW8_ACC(2);
            PYRE_Q4K_ROW8_ACC(3);
            PYRE_Q4K_ROW8_ACC(4);
            PYRE_Q4K_ROW8_ACC(5);
            PYRE_Q4K_ROW8_ACC(6);
            PYRE_Q4K_ROW8_ACC(7);
#undef PYRE_Q4K_ROW8_ACC
        }
    }

    sum0 = pyre_reduce_wg<64>(sum0, sumsh);
    __syncthreads();
    sum1 = pyre_reduce_wg<64>(sum1, sumsh);
    __syncthreads();
    sum2 = pyre_reduce_wg<64>(sum2, sumsh);
    __syncthreads();
    sum3 = pyre_reduce_wg<64>(sum3, sumsh);
    __syncthreads();
    sum4 = pyre_reduce_wg<64>(sum4, sumsh);
    __syncthreads();
    sum5 = pyre_reduce_wg<64>(sum5, sumsh);
    __syncthreads();
    sum6 = pyre_reduce_wg<64>(sum6, sumsh);
    __syncthreads();
    sum7 = pyre_reduce_wg<64>(sum7, sumsh);

    if (tid == 0) {
        char * dst_base = reinterpret_cast<char *>(dst) + id_pos * c.dst_nb1 + token * c.dst_nb2;
        *reinterpret_cast<float *>(dst_base + row0 * sizeof(float)) = sum0;
        *reinterpret_cast<float *>(dst_base + (row0 + 1) * sizeof(float)) = sum1;
        *reinterpret_cast<float *>(dst_base + (row0 + 2) * sizeof(float)) = sum2;
        *reinterpret_cast<float *>(dst_base + (row0 + 3) * sizeof(float)) = sum3;
        *reinterpret_cast<float *>(dst_base + (row0 + 4) * sizeof(float)) = sum4;
        *reinterpret_cast<float *>(dst_base + (row0 + 5) * sizeof(float)) = sum5;
        *reinterpret_cast<float *>(dst_base + (row0 + 6) * sizeof(float)) = sum6;
        *reinterpret_cast<float *>(dst_base + (row0 + 7) * sizeof(float)) = sum7;
    }
}
