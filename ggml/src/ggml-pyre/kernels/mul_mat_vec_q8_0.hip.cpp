#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct pyre_block_q8_0 {
    unsigned short d;
    int8_t qs[32];
};

static __device__ __forceinline__ float pyre_reduce_256(float sum, float * shared) {
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

static __device__ __forceinline__ void pyre_reduce8_256(
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
    constexpr int waves = 256 / 32;

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

extern "C" __global__ void pyre_mul_mat_vec_q8_0_f32(
        const pyre_block_q8_0 * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    __shared__ float sumsh[256];

    const long long blocks_per_row = k / 32;
    const pyre_block_q8_0 * row_blocks = src0 + row * blocks_per_row;
    const float * src1_col = src1 + col * k;
    float sum = 0.0f;

    const int block_lane = tid & 7;
    const int block_slot = tid >> 3;
    const int in_block_base = block_lane << 2;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 32) {
        const pyre_block_q8_0 * block = row_blocks + block_idx;
        const float d = __half2float(__ushort_as_half(block->d));
        const long long src_base = block_idx * 32 + in_block_base;

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float value = d * static_cast<float>(block->qs[in_block_base + j]);
            sum += value * src1_col[src_base + j];
        }
    }

    sum = pyre_reduce_256(sum, sumsh);

    if (tid == 0) {
        dst[col * rows + row] = sum;
    }
}

extern "C" __global__ void pyre_mul_mat_vec_q8_0_cols8_f32(
        const pyre_block_q8_0 * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col0 = __builtin_amdgcn_workgroup_id_y() * 8;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col0 + 7 >= cols) {
        return;
    }

    __shared__ float sumsh[8 * (256 / 32)];

    const long long blocks_per_row = k / 32;
    const pyre_block_q8_0 * row_blocks = src0 + row * blocks_per_row;
    const float * src1_col0 = src1 + col0 * k;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;

    const int block_lane = tid & 7;
    const int block_slot = tid >> 3;
    const int in_block_base = block_lane << 2;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 32) {
        const pyre_block_q8_0 * block = row_blocks + block_idx;
        const float d = __half2float(__ushort_as_half(block->d));
        const long long src_base = block_idx * 32 + in_block_base;
        const float4 b0 = *reinterpret_cast<const float4 *>(src1_col0 + src_base);
        const float4 b1 = *reinterpret_cast<const float4 *>(src1_col0 + k + src_base);
        const float4 b2 = *reinterpret_cast<const float4 *>(src1_col0 + 2 * k + src_base);
        const float4 b3 = *reinterpret_cast<const float4 *>(src1_col0 + 3 * k + src_base);
        const float4 b4 = *reinterpret_cast<const float4 *>(src1_col0 + 4 * k + src_base);
        const float4 b5 = *reinterpret_cast<const float4 *>(src1_col0 + 5 * k + src_base);
        const float4 b6 = *reinterpret_cast<const float4 *>(src1_col0 + 6 * k + src_base);
        const float4 b7 = *reinterpret_cast<const float4 *>(src1_col0 + 7 * k + src_base);
#define PYRE_Q8_0_COLS8_ACC(J, FIELD) \
        do { \
            const float value = d * static_cast<float>(block->qs[in_block_base + (J)]); \
            sum0 += value * b0.FIELD; \
            sum1 += value * b1.FIELD; \
            sum2 += value * b2.FIELD; \
            sum3 += value * b3.FIELD; \
            sum4 += value * b4.FIELD; \
            sum5 += value * b5.FIELD; \
            sum6 += value * b6.FIELD; \
            sum7 += value * b7.FIELD; \
        } while (0)
        PYRE_Q8_0_COLS8_ACC(0, x);
        PYRE_Q8_0_COLS8_ACC(1, y);
        PYRE_Q8_0_COLS8_ACC(2, z);
        PYRE_Q8_0_COLS8_ACC(3, w);
#undef PYRE_Q8_0_COLS8_ACC
    }

    pyre_reduce8_256(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sumsh);

    if (tid == 0) {
        float * dst_col0 = dst + col0 * rows + row;
        dst_col0[0] = sum0;
        dst_col0[rows] = sum1;
        dst_col0[2 * rows] = sum2;
        dst_col0[3 * rows] = sum3;
        dst_col0[4 * rows] = sum4;
        dst_col0[5 * rows] = sum5;
        dst_col0[6 * rows] = sum6;
        dst_col0[7 * rows] = sum7;
    }
}

extern "C" __global__ void pyre_mul_mat_vec_q8_0_add_f32(
        const pyre_block_q8_0 * src0, const float * src1, const float * bias, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    __shared__ float sumsh[256];

    const long long blocks_per_row = k / 32;
    const pyre_block_q8_0 * row_blocks = src0 + row * blocks_per_row;
    const float * src1_col = src1 + col * k;
    float sum = 0.0f;

    const int block_lane = tid & 7;
    const int block_slot = tid >> 3;
    const int in_block_base = block_lane << 2;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 32) {
        const pyre_block_q8_0 * block = row_blocks + block_idx;
        const float d = __half2float(__ushort_as_half(block->d));
        const long long src_base = block_idx * 32 + in_block_base;

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float value = d * static_cast<float>(block->qs[in_block_base + j]);
            sum += value * src1_col[src_base + j];
        }
    }

    sum = pyre_reduce_256(sum, sumsh);

    if (tid == 0) {
        const long long out_idx = col * rows + row;
        dst[out_idx] = sum + bias[out_idx];
    }
}

extern "C" __global__ void pyre_mul_mat_vec_q8_0_add_cols8_f32(
        const pyre_block_q8_0 * src0, const float * src1, const float * bias, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col0 = __builtin_amdgcn_workgroup_id_y() * 8;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col0 + 7 >= cols) {
        return;
    }

    __shared__ float sumsh[8 * (256 / 32)];

    const long long blocks_per_row = k / 32;
    const pyre_block_q8_0 * row_blocks = src0 + row * blocks_per_row;
    const float * src1_col0 = src1 + col0 * k;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;

    const int block_lane = tid & 7;
    const int block_slot = tid >> 3;
    const int in_block_base = block_lane << 2;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 32) {
        const pyre_block_q8_0 * block = row_blocks + block_idx;
        const float d = __half2float(__ushort_as_half(block->d));
        const long long src_base = block_idx * 32 + in_block_base;
        const float4 b0 = *reinterpret_cast<const float4 *>(src1_col0 + src_base);
        const float4 b1 = *reinterpret_cast<const float4 *>(src1_col0 + k + src_base);
        const float4 b2 = *reinterpret_cast<const float4 *>(src1_col0 + 2 * k + src_base);
        const float4 b3 = *reinterpret_cast<const float4 *>(src1_col0 + 3 * k + src_base);
        const float4 b4 = *reinterpret_cast<const float4 *>(src1_col0 + 4 * k + src_base);
        const float4 b5 = *reinterpret_cast<const float4 *>(src1_col0 + 5 * k + src_base);
        const float4 b6 = *reinterpret_cast<const float4 *>(src1_col0 + 6 * k + src_base);
        const float4 b7 = *reinterpret_cast<const float4 *>(src1_col0 + 7 * k + src_base);
#define PYRE_Q8_0_ADD_COLS8_ACC(J, FIELD) \
        do { \
            const float value = d * static_cast<float>(block->qs[in_block_base + (J)]); \
            sum0 += value * b0.FIELD; \
            sum1 += value * b1.FIELD; \
            sum2 += value * b2.FIELD; \
            sum3 += value * b3.FIELD; \
            sum4 += value * b4.FIELD; \
            sum5 += value * b5.FIELD; \
            sum6 += value * b6.FIELD; \
            sum7 += value * b7.FIELD; \
        } while (0)
        PYRE_Q8_0_ADD_COLS8_ACC(0, x);
        PYRE_Q8_0_ADD_COLS8_ACC(1, y);
        PYRE_Q8_0_ADD_COLS8_ACC(2, z);
        PYRE_Q8_0_ADD_COLS8_ACC(3, w);
#undef PYRE_Q8_0_ADD_COLS8_ACC
    }

    pyre_reduce8_256(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sumsh);

    if (tid == 0) {
        const long long out_idx = col0 * rows + row;
        dst[out_idx] = sum0 + bias[out_idx];
        dst[out_idx + rows] = sum1 + bias[out_idx + rows];
        dst[out_idx + 2 * rows] = sum2 + bias[out_idx + 2 * rows];
        dst[out_idx + 3 * rows] = sum3 + bias[out_idx + 3 * rows];
        dst[out_idx + 4 * rows] = sum4 + bias[out_idx + 4 * rows];
        dst[out_idx + 5 * rows] = sum5 + bias[out_idx + 5 * rows];
        dst[out_idx + 6 * rows] = sum6 + bias[out_idx + 6 * rows];
        dst[out_idx + 7 * rows] = sum7 + bias[out_idx + 7 * rows];
    }
}
