#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ float pyre_bf16_to_f32(uint16_t value) {
    union {
        uint32_t u;
        float f;
    } bits = { static_cast<uint32_t>(value) << 16 };
    return bits.f;
}

template <int WG_SIZE>
static __device__ __forceinline__ float pyre_reduce_bf16(float sum, float * shared) {
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
static __device__ __forceinline__ void pyre_mul_mat_vec_bf16_f32_impl(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    __shared__ float sumsh[(WG_SIZE + 31) / 32];

    const uint16_t * src0_row = src0 + row * k;
    const float * src1_col = src1 + col * k;
    float sum = 0.0f;
    for (long long i = tid; i < k; i += WG_SIZE) {
        sum += pyre_bf16_to_f32(src0_row[i]) * src1_col[i];
    }

    sum = pyre_reduce_bf16<WG_SIZE>(sum, sumsh);

    if (tid == 0) {
        dst[col * rows + row] = sum;
    }
}

extern "C" __global__ void pyre_mul_mat_vec_bf16_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    pyre_mul_mat_vec_bf16_f32_impl<256>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void pyre_mul_mat_vec_bf16_wg128_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    pyre_mul_mat_vec_bf16_f32_impl<128>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void pyre_mul_mat_vec_bf16_wg64_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    pyre_mul_mat_vec_bf16_f32_impl<64>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void pyre_mul_mat_vec_bf16_cols1_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows) {
        return;
    }
    (void) cols;

    __shared__ float sumsh[8];

    const uint16_t * src0_row = src0 + row * k;
    float sum = 0.0f;
    for (long long i = tid; i < k; i += 256) {
        sum += pyre_bf16_to_f32(src0_row[i]) * src1[i];
    }

    sum = pyre_reduce_bf16<256>(sum, sumsh);

    if (tid == 0) {
        dst[row] = sum;
    }
}

extern "C" __global__ void pyre_mul_mat_vec_bf16_cols4_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col0 = __builtin_amdgcn_workgroup_id_y() * 4;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col0 >= cols) {
        return;
    }

    __shared__ float sumsh[8];

    const uint16_t * src0_row = src0 + row * k;
    const float * src1_col0 = src1 + col0 * k;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    for (long long i = tid; i < k; i += 256) {
        const float a = pyre_bf16_to_f32(src0_row[i]);
        sum0 += a * src1_col0[i];
        sum1 += a * src1_col0[k + i];
        sum2 += a * src1_col0[2 * k + i];
        sum3 += a * src1_col0[3 * k + i];
    }

    sum0 = pyre_reduce_bf16<256>(sum0, sumsh);
    __syncthreads();
    sum1 = pyre_reduce_bf16<256>(sum1, sumsh);
    __syncthreads();
    sum2 = pyre_reduce_bf16<256>(sum2, sumsh);
    __syncthreads();
    sum3 = pyre_reduce_bf16<256>(sum3, sumsh);

    if (tid == 0) {
        float * dst_col0 = dst + col0 * rows + row;
        dst_col0[0] = sum0;
        dst_col0[rows] = sum1;
        dst_col0[2 * rows] = sum2;
        dst_col0[3 * rows] = sum3;
    }
}
