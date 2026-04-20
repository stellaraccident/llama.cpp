#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

static __device__ __forceinline__ float hrx_reduce_256(float sum, float * shared) {
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

template <int COLS>
static __device__ __forceinline__ void hrx_reduce_cols_256(float sum[COLS], float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;
    constexpr int waves = 256 / 32;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
#pragma unroll
        for (int col = 0; col < COLS; ++col) {
            sum[col] += __shfl_down(sum[col], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int col = 0; col < COLS; ++col) {
            shared[col * waves + wave] = sum[col];
        }
    }
    __syncthreads();

#pragma unroll
    for (int col = 0; col < COLS; ++col) {
        sum[col] = lane < waves ? shared[col * waves + lane] : 0.0f;
    }
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
#pragma unroll
            for (int col = 0; col < COLS; ++col) {
                sum[col] += __shfl_down(sum[col], offset);
            }
        }
    }
}

template <int COLS>
static __device__ __forceinline__ void hrx_mul_mat_vec_f32_cols_f32_impl(
        const float * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col0 = __builtin_amdgcn_workgroup_id_y() * COLS;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col0 + COLS > cols) {
        return;
    }

    __shared__ float sumsh[COLS * (256 / 32)];

    const float * src0_row = src0 + row * k;
    const float * src1_col0 = src1 + col0 * k;
    float sum[COLS] = {};
    for (long long i = tid; i < k; i += 256) {
        const float a = src0_row[i];
#pragma unroll
        for (int col = 0; col < COLS; ++col) {
            sum[col] += a * src1_col0[col * k + i];
        }
    }

    hrx_reduce_cols_256<COLS>(sum, sumsh);

    if (tid == 0) {
        float * dst_col0 = dst + col0 * rows + row;
#pragma unroll
        for (int col = 0; col < COLS; ++col) {
            dst_col0[col * rows] = sum[col];
        }
    }
}

extern "C" __global__ void hrx_mul_mat_vec_f32_f32(
        const float * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    __shared__ float sumsh[256];

    const float * src0_row = src0 + row * k;
    const float * src1_col = src1 + col * k;
    float sum = 0.0f;
    for (long long i = tid; i < k; i += 256) {
        sum += src0_row[i] * src1_col[i];
    }

    sum = hrx_reduce_256(sum, sumsh);

    if (tid == 0) {
        dst[col * rows + row] = sum;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_f32_cols3_f32(
        const float * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    hrx_mul_mat_vec_f32_cols_f32_impl<3>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void hrx_mul_mat_vec_f32_cols4_f32(
        const float * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    hrx_mul_mat_vec_f32_cols_f32_impl<4>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void hrx_mul_mat_vec_f32_cols5_f32(
        const float * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    hrx_mul_mat_vec_f32_cols_f32_impl<5>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void hrx_mul_mat_vec_f32_cols6_f32(
        const float * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    hrx_mul_mat_vec_f32_cols_f32_impl<6>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void hrx_mul_mat_vec_f32_cols7_f32(
        const float * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    hrx_mul_mat_vec_f32_cols_f32_impl<7>(src0, src1, dst, k, rows, cols);
}
