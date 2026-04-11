#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct pyre_block_q6_K_q8_1_lhs {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    unsigned short d;
};

struct pyre_block_q8_1_rhs_q6 {
    unsigned short d;
    unsigned short s;
    int8_t qs[32];
};

static __device__ __forceinline__ float pyre_reduce_256_q6_q8_1(float sum, float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (lane == 0) {
        shared[wave] = sum;
    }
    __syncthreads();

    sum = lane < (256 / warpSize) ? shared[lane] : 0.0f;
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset);
        }
    }
    return sum;
}

template <int WG_SIZE>
static __device__ __forceinline__ void pyre_reduce_wg8_q6_q8_1(
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

static __device__ __forceinline__ int pyre_q6_k_value(
        const pyre_block_q6_K_q8_1_lhs * block,
        int in_block) {
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
        const pyre_block_q6_K_q8_1_lhs * block,
        int group,
        int lane) {
    const int half = group >> 2;
    const int group_in_half = group & 3;
    return static_cast<int>(block->scales[half * 8 + group_in_half * 2 + lane / 16]);
}

static __device__ __forceinline__ int pyre_sdot4_q6_q8_1(int q0, int q1, int q2, int q3, const int8_t * rhs) {
    const unsigned int qpack =
        (static_cast<unsigned int>(static_cast<unsigned char>(static_cast<int8_t>(q0))) << 0) |
        (static_cast<unsigned int>(static_cast<unsigned char>(static_cast<int8_t>(q1))) << 8) |
        (static_cast<unsigned int>(static_cast<unsigned char>(static_cast<int8_t>(q2))) << 16) |
        (static_cast<unsigned int>(static_cast<unsigned char>(static_cast<int8_t>(q3))) << 24);
    const int rpack = *reinterpret_cast<const int *>(rhs);
    return __builtin_amdgcn_sudot4(true, static_cast<int>(qpack), true, rpack, 0, false);
}

extern "C" __global__ void pyre_mul_mat_vec_q6_k_q8_1_f32(
        const pyre_block_q6_K_q8_1_lhs * src0,
        const pyre_block_q8_1_rhs_q6 * src1,
        float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    __shared__ float sumsh[256];

    const long long blocks_per_row = k / 256;
    const pyre_block_q6_K_q8_1_lhs * row_blocks = src0 + row * blocks_per_row;
    const pyre_block_q8_1_rhs_q6 * src1_col = src1 + col * (k / 32);
    float sum = 0.0f;

    const int block_lane = tid & 63;
    const int block_slot = tid >> 6;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const int in_block_base = group * 32 + lane;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 4) {
        const pyre_block_q6_K_q8_1_lhs * block = row_blocks + block_idx;
        const pyre_block_q8_1_rhs_q6 * rhs = src1_col + block_idx * 8 + group;

        const int qsum = pyre_sdot4_q6_q8_1(
            pyre_q6_k_value(block, in_block_base + 0),
            pyre_q6_k_value(block, in_block_base + 1),
            pyre_q6_k_value(block, in_block_base + 2),
            pyre_q6_k_value(block, in_block_base + 3),
            rhs->qs + lane);

        const float d = __half2float(__ushort_as_half(block->d)) *
            static_cast<float>(pyre_q6_k_scale(block, group, lane));
        const float d8 = __half2float(__ushort_as_half(rhs->d));
        sum += d * d8 * static_cast<float>(qsum);
    }

    sum = pyre_reduce_256_q6_q8_1(sum, sumsh);

    if (tid == 0) {
        dst[col * rows + row] = sum;
    }
}

extern "C" __global__ void pyre_mul_mat_vec_q6_k_q8_1_cols8_wg128_f32(
        const pyre_block_q6_K_q8_1_lhs * src0,
        const pyre_block_q8_1_rhs_q6 * src1,
        float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * 8;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col0 + 7 >= cols) {
        return;
    }

    __shared__ float sumsh[8 * (128 / 32)];

    const long long blocks_per_row = k / 256;
    const long long q8_blocks_per_col = k / 32;
    const pyre_block_q6_K_q8_1_lhs * row_blocks = src0 + row * blocks_per_row;
    const pyre_block_q8_1_rhs_q6 * src1_col0 = src1 + col0 * q8_blocks_per_col;
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
        const pyre_block_q6_K_q8_1_lhs * block = row_blocks + block_idx;
        const long long rhs_base = block_idx * 8 + group;
        const int q0 = pyre_q6_k_value(block, in_block_base + 0);
        const int q1 = pyre_q6_k_value(block, in_block_base + 1);
        const int q2 = pyre_q6_k_value(block, in_block_base + 2);
        const int q3 = pyre_q6_k_value(block, in_block_base + 3);
        const float d = __half2float(__ushort_as_half(block->d)) *
            static_cast<float>(pyre_q6_k_scale(block, group, lane));

        const pyre_block_q8_1_rhs_q6 * rhs0 = src1_col0 + rhs_base;
        const pyre_block_q8_1_rhs_q6 * rhs1 = rhs0 + q8_blocks_per_col;
        const pyre_block_q8_1_rhs_q6 * rhs2 = rhs1 + q8_blocks_per_col;
        const pyre_block_q8_1_rhs_q6 * rhs3 = rhs2 + q8_blocks_per_col;
        const pyre_block_q8_1_rhs_q6 * rhs4 = rhs3 + q8_blocks_per_col;
        const pyre_block_q8_1_rhs_q6 * rhs5 = rhs4 + q8_blocks_per_col;
        const pyre_block_q8_1_rhs_q6 * rhs6 = rhs5 + q8_blocks_per_col;
        const pyre_block_q8_1_rhs_q6 * rhs7 = rhs6 + q8_blocks_per_col;

        const int qsum0 = pyre_sdot4_q6_q8_1(q0, q1, q2, q3, rhs0->qs + lane);
        const int qsum1 = pyre_sdot4_q6_q8_1(q0, q1, q2, q3, rhs1->qs + lane);
        const int qsum2 = pyre_sdot4_q6_q8_1(q0, q1, q2, q3, rhs2->qs + lane);
        const int qsum3 = pyre_sdot4_q6_q8_1(q0, q1, q2, q3, rhs3->qs + lane);
        const int qsum4 = pyre_sdot4_q6_q8_1(q0, q1, q2, q3, rhs4->qs + lane);
        const int qsum5 = pyre_sdot4_q6_q8_1(q0, q1, q2, q3, rhs5->qs + lane);
        const int qsum6 = pyre_sdot4_q6_q8_1(q0, q1, q2, q3, rhs6->qs + lane);
        const int qsum7 = pyre_sdot4_q6_q8_1(q0, q1, q2, q3, rhs7->qs + lane);

        sum0 += d * __half2float(__ushort_as_half(rhs0->d)) * static_cast<float>(qsum0);
        sum1 += d * __half2float(__ushort_as_half(rhs1->d)) * static_cast<float>(qsum1);
        sum2 += d * __half2float(__ushort_as_half(rhs2->d)) * static_cast<float>(qsum2);
        sum3 += d * __half2float(__ushort_as_half(rhs3->d)) * static_cast<float>(qsum3);
        sum4 += d * __half2float(__ushort_as_half(rhs4->d)) * static_cast<float>(qsum4);
        sum5 += d * __half2float(__ushort_as_half(rhs5->d)) * static_cast<float>(qsum5);
        sum6 += d * __half2float(__ushort_as_half(rhs6->d)) * static_cast<float>(qsum6);
        sum7 += d * __half2float(__ushort_as_half(rhs7->d)) * static_cast<float>(qsum7);
    }

    pyre_reduce_wg8_q6_q8_1<128>(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sumsh);

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
