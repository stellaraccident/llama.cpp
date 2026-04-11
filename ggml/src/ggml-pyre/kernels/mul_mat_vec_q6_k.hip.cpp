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

template <int WG_SIZE>
static __device__ __forceinline__ void pyre_reduce_wg4(
        float & sum0,
        float & sum1,
        float & sum2,
        float & sum3,
        float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;
    constexpr int waves = (WG_SIZE + 31) / 32;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset);
        sum1 += __shfl_down(sum1, offset);
        sum2 += __shfl_down(sum2, offset);
        sum3 += __shfl_down(sum3, offset);
    }
    if (lane == 0) {
        shared[wave + 0 * waves] = sum0;
        shared[wave + 1 * waves] = sum1;
        shared[wave + 2 * waves] = sum2;
        shared[wave + 3 * waves] = sum3;
    }
    __syncthreads();

    sum0 = lane < waves ? shared[lane + 0 * waves] : 0.0f;
    sum1 = lane < waves ? shared[lane + 1 * waves] : 0.0f;
    sum2 = lane < waves ? shared[lane + 2 * waves] : 0.0f;
    sum3 = lane < waves ? shared[lane + 3 * waves] : 0.0f;
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum0 += __shfl_down(sum0, offset);
            sum1 += __shfl_down(sum1, offset);
            sum2 += __shfl_down(sum2, offset);
            sum3 += __shfl_down(sum3, offset);
        }
    }
}

template <int WG_SIZE>
static __device__ __forceinline__ void pyre_reduce_wg8(
        float & sum0,
        float & sum1,
        float & sum2,
        float & sum3,
        float & sum4,
        float & sum5,
        float & sum6,
        float & sum7,
        float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;
    constexpr int waves = (WG_SIZE + 31) / 32;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset);
        sum1 += __shfl_down(sum1, offset);
        sum2 += __shfl_down(sum2, offset);
        sum3 += __shfl_down(sum3, offset);
        sum4 += __shfl_down(sum4, offset);
        sum5 += __shfl_down(sum5, offset);
        sum6 += __shfl_down(sum6, offset);
        sum7 += __shfl_down(sum7, offset);
    }
    if (lane == 0) {
        shared[wave + 0 * waves] = sum0;
        shared[wave + 1 * waves] = sum1;
        shared[wave + 2 * waves] = sum2;
        shared[wave + 3 * waves] = sum3;
        shared[wave + 4 * waves] = sum4;
        shared[wave + 5 * waves] = sum5;
        shared[wave + 6 * waves] = sum6;
        shared[wave + 7 * waves] = sum7;
    }
    __syncthreads();

    sum0 = lane < waves ? shared[lane + 0 * waves] : 0.0f;
    sum1 = lane < waves ? shared[lane + 1 * waves] : 0.0f;
    sum2 = lane < waves ? shared[lane + 2 * waves] : 0.0f;
    sum3 = lane < waves ? shared[lane + 3 * waves] : 0.0f;
    sum4 = lane < waves ? shared[lane + 4 * waves] : 0.0f;
    sum5 = lane < waves ? shared[lane + 5 * waves] : 0.0f;
    sum6 = lane < waves ? shared[lane + 6 * waves] : 0.0f;
    sum7 = lane < waves ? shared[lane + 7 * waves] : 0.0f;
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum0 += __shfl_down(sum0, offset);
            sum1 += __shfl_down(sum1, offset);
            sum2 += __shfl_down(sum2, offset);
            sum3 += __shfl_down(sum3, offset);
            sum4 += __shfl_down(sum4, offset);
            sum5 += __shfl_down(sum5, offset);
            sum6 += __shfl_down(sum6, offset);
            sum7 += __shfl_down(sum7, offset);
        }
    }
}

template <int WG_SIZE>
static __device__ __forceinline__ void pyre_reduce_wg16(float (&sum)[16], float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;
    constexpr int waves = (WG_SIZE + 31) / 32;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int col = 0; col < 16; ++col) {
            sum[col] += __shfl_down(sum[col], offset);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int col = 0; col < 16; ++col) {
            shared[wave + col * waves] = sum[col];
        }
    }
    __syncthreads();

    #pragma unroll
    for (int col = 0; col < 16; ++col) {
        sum[col] = lane < waves ? shared[lane + col * waves] : 0.0f;
    }
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int col = 0; col < 16; ++col) {
                sum[col] += __shfl_down(sum[col], offset);
            }
        }
    }
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

static __device__ __forceinline__ float pyre_q6_k_dot4(
        const pyre_block_q6_K * block,
        const float * src,
        float d,
        int group,
        int lane) {
    const int half = group >> 2;
    const int group_in_half = group & 3;
    const int ql_base = half * 64 + ((group_in_half & 1) ? 32 : 0) + lane;
    const int qh_base = half * 32 + lane;
    const int qh_shift = (group_in_half & 3) * 2;
    const bool high_nibble = group_in_half >= 2;

    const uint32_t ql_word =
        static_cast<uint32_t>(block->ql[ql_base]) |
        (static_cast<uint32_t>(block->ql[ql_base + 1]) << 8) |
        (static_cast<uint32_t>(block->ql[ql_base + 2]) << 16) |
        (static_cast<uint32_t>(block->ql[ql_base + 3]) << 24);
    const uint32_t qh_word =
        static_cast<uint32_t>(block->qh[qh_base]) |
        (static_cast<uint32_t>(block->qh[qh_base + 1]) << 8) |
        (static_cast<uint32_t>(block->qh[qh_base + 2]) << 16) |
        (static_cast<uint32_t>(block->qh[qh_base + 3]) << 24);
    float sum = 0.0f;

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int ql_shift = 8 * j + (high_nibble ? 4 : 0);
        const int ql = (ql_word >> ql_shift) & 0x0F;
        const int qh = (qh_word >> (8 * j + qh_shift)) & 0x03;
        const int q = (ql | (qh << 4)) - 32;
        sum += d * static_cast<float>(q) * src[j];
    }

    return sum;
}

static __device__ __forceinline__ void pyre_q6_k_dot4_cols4(
        const pyre_block_q6_K * block,
        const float * src0,
        const float * src1,
        const float * src2,
        const float * src3,
        float d,
        int group,
        int lane,
        float & sum0,
        float & sum1,
        float & sum2,
        float & sum3) {
    const int half = group >> 2;
    const int group_in_half = group & 3;
    const int ql_base = half * 64 + ((group_in_half & 1) ? 32 : 0) + lane;
    const int qh_base = half * 32 + lane;
    const int qh_shift = (group_in_half & 3) * 2;
    const bool high_nibble = group_in_half >= 2;

    const uint32_t ql_word =
        static_cast<uint32_t>(block->ql[ql_base]) |
        (static_cast<uint32_t>(block->ql[ql_base + 1]) << 8) |
        (static_cast<uint32_t>(block->ql[ql_base + 2]) << 16) |
        (static_cast<uint32_t>(block->ql[ql_base + 3]) << 24);
    const uint32_t qh_word =
        static_cast<uint32_t>(block->qh[qh_base]) |
        (static_cast<uint32_t>(block->qh[qh_base + 1]) << 8) |
        (static_cast<uint32_t>(block->qh[qh_base + 2]) << 16) |
        (static_cast<uint32_t>(block->qh[qh_base + 3]) << 24);

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int ql_shift = 8 * j + (high_nibble ? 4 : 0);
        const int ql = (ql_word >> ql_shift) & 0x0F;
        const int qh = (qh_word >> (8 * j + qh_shift)) & 0x03;
        const float q = d * static_cast<float>((ql | (qh << 4)) - 32);
        sum0 += q * src0[j];
        sum1 += q * src1[j];
        sum2 += q * src2[j];
        sum3 += q * src3[j];
    }
}

static __device__ __forceinline__ void pyre_q6_k_dot4_cols8(
        const pyre_block_q6_K * block,
        const float * src0,
        const float * src1,
        const float * src2,
        const float * src3,
        const float * src4,
        const float * src5,
        const float * src6,
        const float * src7,
        float d,
        int group,
        int lane,
        float & sum0,
        float & sum1,
        float & sum2,
        float & sum3,
        float & sum4,
        float & sum5,
        float & sum6,
        float & sum7) {
    const int half = group >> 2;
    const int group_in_half = group & 3;
    const int ql_base = half * 64 + ((group_in_half & 1) ? 32 : 0) + lane;
    const int qh_base = half * 32 + lane;
    const int qh_shift = (group_in_half & 3) * 2;
    const bool high_nibble = group_in_half >= 2;

    const uint32_t ql_word =
        static_cast<uint32_t>(block->ql[ql_base]) |
        (static_cast<uint32_t>(block->ql[ql_base + 1]) << 8) |
        (static_cast<uint32_t>(block->ql[ql_base + 2]) << 16) |
        (static_cast<uint32_t>(block->ql[ql_base + 3]) << 24);
    const uint32_t qh_word =
        static_cast<uint32_t>(block->qh[qh_base]) |
        (static_cast<uint32_t>(block->qh[qh_base + 1]) << 8) |
        (static_cast<uint32_t>(block->qh[qh_base + 2]) << 16) |
        (static_cast<uint32_t>(block->qh[qh_base + 3]) << 24);

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int ql_shift = 8 * j + (high_nibble ? 4 : 0);
        const int ql = (ql_word >> ql_shift) & 0x0F;
        const int qh = (qh_word >> (8 * j + qh_shift)) & 0x03;
        const float q = d * static_cast<float>((ql | (qh << 4)) - 32);
        sum0 += q * src0[j];
        sum1 += q * src1[j];
        sum2 += q * src2[j];
        sum3 += q * src3[j];
        sum4 += q * src4[j];
        sum5 += q * src5[j];
        sum6 += q * src6[j];
        sum7 += q * src7[j];
    }
}

static __device__ __forceinline__ void pyre_q6_k_dot4_cols16(
        const pyre_block_q6_K * block,
        const float * src0,
        long long k,
        float d,
        int group,
        int lane,
        float (&sum)[16]) {
    const int half = group >> 2;
    const int group_in_half = group & 3;
    const int ql_base = half * 64 + ((group_in_half & 1) ? 32 : 0) + lane;
    const int qh_base = half * 32 + lane;
    const int qh_shift = (group_in_half & 3) * 2;
    const bool high_nibble = group_in_half >= 2;

    const uint32_t ql_word =
        static_cast<uint32_t>(block->ql[ql_base]) |
        (static_cast<uint32_t>(block->ql[ql_base + 1]) << 8) |
        (static_cast<uint32_t>(block->ql[ql_base + 2]) << 16) |
        (static_cast<uint32_t>(block->ql[ql_base + 3]) << 24);
    const uint32_t qh_word =
        static_cast<uint32_t>(block->qh[qh_base]) |
        (static_cast<uint32_t>(block->qh[qh_base + 1]) << 8) |
        (static_cast<uint32_t>(block->qh[qh_base + 2]) << 16) |
        (static_cast<uint32_t>(block->qh[qh_base + 3]) << 24);

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int ql_shift = 8 * j + (high_nibble ? 4 : 0);
        const int ql = (ql_word >> ql_shift) & 0x0F;
        const int qh = (qh_word >> (8 * j + qh_shift)) & 0x03;
        const float q = d * static_cast<float>((ql | (qh << 4)) - 32);
        #pragma unroll
        for (int col = 0; col < 16; ++col) {
            sum[col] += q * src0[col * k + j];
        }
    }
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

        sum += pyre_q6_k_dot4(block, src1_col + src_base, d, group, lane);
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

extern "C" __global__ void pyre_mul_mat_vec_q6_k_cols4_wg128_f32(
        const pyre_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * 4;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col0 + 3 >= cols) {
        return;
    }

    __shared__ float sumsh[4 * (128 / 32)];

    const long long blocks_per_row = k / 256;
    const pyre_block_q6_K * row_blocks = src0 + row * blocks_per_row;
    const float * src1_col0 = src1 + col0 * k;
    const float * src1_col1 = src1_col0 + k;
    const float * src1_col2 = src1_col1 + k;
    const float * src1_col3 = src1_col2 + k;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    const int block_lane = tid & 63;
    const int block_slot = tid >> 6;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const int in_block_base = group * 32 + lane;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 2) {
        const pyre_block_q6_K * block = row_blocks + block_idx;
        const long long src_base = block_idx * 256 + in_block_base;
        const float d = __half2float(__ushort_as_half(block->d)) *
            static_cast<float>(pyre_q6_k_scale(block, group, lane));

        pyre_q6_k_dot4_cols4(
            block,
            src1_col0 + src_base,
            src1_col1 + src_base,
            src1_col2 + src_base,
            src1_col3 + src_base,
            d,
            group,
            lane,
            sum0,
            sum1,
            sum2,
            sum3);
    }

    pyre_reduce_wg4<128>(sum0, sum1, sum2, sum3, sumsh);

    if (tid == 0) {
        dst[col0 * rows + row] = sum0;
        dst[(col0 + 1) * rows + row] = sum1;
        dst[(col0 + 2) * rows + row] = sum2;
        dst[(col0 + 3) * rows + row] = sum3;
    }
}

extern "C" __global__ void pyre_mul_mat_vec_q6_k_cols8_wg128_f32(
        const pyre_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * 8;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col0 + 7 >= cols) {
        return;
    }

    __shared__ float sumsh[8 * (128 / 32)];

    const long long blocks_per_row = k / 256;
    const pyre_block_q6_K * row_blocks = src0 + row * blocks_per_row;
    const float * src1_col0 = src1 + col0 * k;
    const float * src1_col1 = src1_col0 + k;
    const float * src1_col2 = src1_col1 + k;
    const float * src1_col3 = src1_col2 + k;
    const float * src1_col4 = src1_col3 + k;
    const float * src1_col5 = src1_col4 + k;
    const float * src1_col6 = src1_col5 + k;
    const float * src1_col7 = src1_col6 + k;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;

    const int block_lane = tid & 63;
    const int block_slot = tid >> 6;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const int in_block_base = group * 32 + lane;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 2) {
        const pyre_block_q6_K * block = row_blocks + block_idx;
        const long long src_base = block_idx * 256 + in_block_base;
        const float d = __half2float(__ushort_as_half(block->d)) *
            static_cast<float>(pyre_q6_k_scale(block, group, lane));

        pyre_q6_k_dot4_cols8(
            block,
            src1_col0 + src_base,
            src1_col1 + src_base,
            src1_col2 + src_base,
            src1_col3 + src_base,
            src1_col4 + src_base,
            src1_col5 + src_base,
            src1_col6 + src_base,
            src1_col7 + src_base,
            d,
            group,
            lane,
            sum0,
            sum1,
            sum2,
            sum3,
            sum4,
            sum5,
            sum6,
            sum7);
    }

    pyre_reduce_wg8<128>(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sumsh);

    if (tid == 0) {
        dst[col0 * rows + row] = sum0;
        dst[(col0 + 1) * rows + row] = sum1;
        dst[(col0 + 2) * rows + row] = sum2;
        dst[(col0 + 3) * rows + row] = sum3;
        dst[(col0 + 4) * rows + row] = sum4;
        dst[(col0 + 5) * rows + row] = sum5;
        dst[(col0 + 6) * rows + row] = sum6;
        dst[(col0 + 7) * rows + row] = sum7;
    }
}

extern "C" __global__ void pyre_mul_mat_vec_q6_k_cols16_wg128_f32(
        const pyre_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * 16;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col0 + 15 >= cols) {
        return;
    }

    __shared__ float sumsh[16 * (128 / 32)];

    const long long blocks_per_row = k / 256;
    const pyre_block_q6_K * row_blocks = src0 + row * blocks_per_row;
    const float * src1_col0 = src1 + col0 * k;
    float sum[16] = {};

    const int block_lane = tid & 63;
    const int block_slot = tid >> 6;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const int in_block_base = group * 32 + lane;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 2) {
        const pyre_block_q6_K * block = row_blocks + block_idx;
        const long long src_base = block_idx * 256 + in_block_base;
        const float d = __half2float(__ushort_as_half(block->d)) *
            static_cast<float>(pyre_q6_k_scale(block, group, lane));

        pyre_q6_k_dot4_cols16(block, src1_col0 + src_base, k, d, group, lane, sum);
    }

    pyre_reduce_wg16<128>(sum, sumsh);

    if (tid == 0) {
        #pragma unroll
        for (int col = 0; col < 16; ++col) {
            dst[(col0 + col) * rows + row] = sum[col];
        }
    }
}
