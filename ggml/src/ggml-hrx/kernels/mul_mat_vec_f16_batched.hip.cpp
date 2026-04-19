#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_mul_mat_vec_f16_batched_constants {
    long long k;
    long long rows;
    long long cols;
    long long dst_ne2;
    long long dst_ne3;
    long long src0_ne2;
    long long src0_ne3;
    long long src0_nb1;
    long long src0_nb2;
    long long src0_nb3;
    long long src1_nb1;
    long long src1_nb2;
    long long src1_nb3;
    long long dst_nb1;
    long long dst_nb2;
    long long dst_nb3;
};

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

static __device__ __forceinline__ void hrx_reduce4_256(
        float & sum0,
        float & sum1,
        float & sum2,
        float & sum3,
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

static __device__ __forceinline__ void hrx_reduce8_256(
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

extern "C" __global__ void hrx_mul_mat_vec_f16_batched_f32(
        const __half * src0, const float * src1, float * dst,
        hrx_mul_mat_vec_f16_batched_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.rows) {
        return;
    }

    const long long i11 = outer % c.cols;
    const long long t = outer / c.cols;
    const long long i12 = t % c.dst_ne2;
    const long long i13 = t / c.dst_ne2;
    if (i13 >= c.dst_ne3) {
        return;
    }

    const long long src0_i02 = c.src0_ne2 == c.dst_ne2 ? i12 : i12 / (c.dst_ne2 / c.src0_ne2);
    const long long src0_i03 = c.src0_ne3 == c.dst_ne3 ? i13 : i13 / (c.dst_ne3 / c.src0_ne3);
    const char * src0_row = reinterpret_cast<const char *>(src0) +
        row * c.src0_nb1 + src0_i02 * c.src0_nb2 + src0_i03 * c.src0_nb3;
    const char * src1_col = reinterpret_cast<const char *>(src1) +
        i11 * c.src1_nb1 + i12 * c.src1_nb2 + i13 * c.src1_nb3;

    __shared__ float sumsh[4 * (256 / 32)];
    float sum = 0.0f;
    for (long long i = tid; i < c.k; i += 256) {
        const float a = __half2float(*reinterpret_cast<const __half *>(src0_row + i * sizeof(__half)));
        const float b = *reinterpret_cast<const float *>(src1_col + i * sizeof(float));
        sum += a * b;
    }

    sum = hrx_reduce_256(sum, sumsh);

    if (tid == 0) {
        *reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst) + row * sizeof(float) + i11 * c.dst_nb1 + i12 * c.dst_nb2 + i13 * c.dst_nb3) =
            sum;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_f16_batched_cols1_f32(
        const __half * src0, const float * src1, float * dst,
        hrx_mul_mat_vec_f16_batched_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long i12 = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.rows || i12 >= c.dst_ne2) {
        return;
    }

    const long long src0_i02 = c.src0_ne2 == c.dst_ne2 ? i12 : i12 / (c.dst_ne2 / c.src0_ne2);
    const char * src0_row = reinterpret_cast<const char *>(src0) +
        row * c.src0_nb1 + src0_i02 * c.src0_nb2;
    const char * src1_col = reinterpret_cast<const char *>(src1) + i12 * c.src1_nb2;

    __shared__ float sumsh[256];
    float sum = 0.0f;
    for (long long i = tid; i < c.k; i += 256) {
        const float a = __half2float(*reinterpret_cast<const __half *>(src0_row + i * sizeof(__half)));
        const float b = *reinterpret_cast<const float *>(src1_col + i * sizeof(float));
        sum += a * b;
    }

    sum = hrx_reduce_256(sum, sumsh);

    if (tid == 0) {
        *reinterpret_cast<float *>(reinterpret_cast<char *>(dst) + row * sizeof(float) + i12 * c.dst_nb2) = sum;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_f16_batched_cols4_f32(
        const __half * src0, const float * src1, float * dst,
        hrx_mul_mat_vec_f16_batched_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.rows) {
        return;
    }

    const long long col_groups = (c.cols + 3) / 4;
    const long long col_group = outer % col_groups;
    const long long i11 = col_group * 4;
    const long long t = outer / col_groups;
    const long long i12 = t % c.dst_ne2;
    const long long i13 = t / c.dst_ne2;
    if (i13 >= c.dst_ne3) {
        return;
    }

    const long long src0_i02 = c.src0_ne2 == c.dst_ne2 ? i12 : i12 / (c.dst_ne2 / c.src0_ne2);
    const long long src0_i03 = c.src0_ne3 == c.dst_ne3 ? i13 : i13 / (c.dst_ne3 / c.src0_ne3);
    const char * src0_row = reinterpret_cast<const char *>(src0) +
        row * c.src0_nb1 + src0_i02 * c.src0_nb2 + src0_i03 * c.src0_nb3;
    const char * src1_col0 = reinterpret_cast<const char *>(src1) +
        i11 * c.src1_nb1 + i12 * c.src1_nb2 + i13 * c.src1_nb3;

    __shared__ float sumsh[256];
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    const long long rem = c.cols - i11;
    if (i11 + 3 < c.cols) {
        for (long long i = tid; i < c.k; i += 256) {
            const float a = __half2float(*reinterpret_cast<const __half *>(src0_row + i * sizeof(__half)));
            const long long rhs = i * sizeof(float);
            sum0 += a * *reinterpret_cast<const float *>(src1_col0 + rhs);
            sum1 += a * *reinterpret_cast<const float *>(src1_col0 + c.src1_nb1 + rhs);
            sum2 += a * *reinterpret_cast<const float *>(src1_col0 + 2 * c.src1_nb1 + rhs);
            sum3 += a * *reinterpret_cast<const float *>(src1_col0 + 3 * c.src1_nb1 + rhs);
        }
    } else if (rem == 1) {
        for (long long i = tid; i < c.k; i += 256) {
            const float a = __half2float(*reinterpret_cast<const __half *>(src0_row + i * sizeof(__half)));
            const long long rhs = i * sizeof(float);
            sum0 += a * *reinterpret_cast<const float *>(src1_col0 + rhs);
        }
    } else {
        for (long long i = tid; i < c.k; i += 256) {
            const float a = __half2float(*reinterpret_cast<const __half *>(src0_row + i * sizeof(__half)));
            const long long rhs = i * sizeof(float);
            if (i11 + 0 < c.cols) { sum0 += a * *reinterpret_cast<const float *>(src1_col0 + rhs); }
            if (i11 + 1 < c.cols) { sum1 += a * *reinterpret_cast<const float *>(src1_col0 + c.src1_nb1 + rhs); }
            if (i11 + 2 < c.cols) { sum2 += a * *reinterpret_cast<const float *>(src1_col0 + 2 * c.src1_nb1 + rhs); }
            if (i11 + 3 < c.cols) { sum3 += a * *reinterpret_cast<const float *>(src1_col0 + 3 * c.src1_nb1 + rhs); }
        }
    }

    if (rem == 1) {
        sum0 = hrx_reduce_256(sum0, sumsh);
    } else {
        hrx_reduce4_256(sum0, sum1, sum2, sum3, sumsh);
    }

    if (tid == 0) {
        char * dst_row = reinterpret_cast<char *>(dst) +
            row * sizeof(float) + i11 * c.dst_nb1 + i12 * c.dst_nb2 + i13 * c.dst_nb3;
        *reinterpret_cast<float *>(dst_row) = sum0;
        if (rem > 1) { *reinterpret_cast<float *>(dst_row + c.dst_nb1) = sum1; }
        if (rem > 2) { *reinterpret_cast<float *>(dst_row + 2 * c.dst_nb1) = sum2; }
        if (rem > 3) { *reinterpret_cast<float *>(dst_row + 3 * c.dst_nb1) = sum3; }
    }
}

extern "C" __global__ void hrx_mul_mat_vec_f16_batched_cols8_f32(
        const __half * src0, const float * src1, float * dst,
        hrx_mul_mat_vec_f16_batched_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.rows) {
        return;
    }

    const long long col_groups = (c.cols + 7) / 8;
    const long long col_group = outer % col_groups;
    const long long i11 = col_group * 8;
    const long long t = outer / col_groups;
    const long long i12 = t % c.dst_ne2;
    const long long i13 = t / c.dst_ne2;
    if (i13 >= c.dst_ne3) {
        return;
    }

    const long long src0_i02 = c.src0_ne2 == c.dst_ne2 ? i12 : i12 / (c.dst_ne2 / c.src0_ne2);
    const long long src0_i03 = c.src0_ne3 == c.dst_ne3 ? i13 : i13 / (c.dst_ne3 / c.src0_ne3);
    const char * src0_row = reinterpret_cast<const char *>(src0) +
        row * c.src0_nb1 + src0_i02 * c.src0_nb2 + src0_i03 * c.src0_nb3;
    const char * src1_col0 = reinterpret_cast<const char *>(src1) +
        i11 * c.src1_nb1 + i12 * c.src1_nb2 + i13 * c.src1_nb3;

    __shared__ float sumsh[8 * (256 / 32)];
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;
    const long long rem = c.cols - i11;
    if (i11 + 7 < c.cols) {
        for (long long i = tid; i < c.k; i += 256) {
            const float a = __half2float(*reinterpret_cast<const __half *>(src0_row + i * sizeof(__half)));
            const long long rhs = i * sizeof(float);
            sum0 += a * *reinterpret_cast<const float *>(src1_col0 + rhs);
            sum1 += a * *reinterpret_cast<const float *>(src1_col0 + c.src1_nb1 + rhs);
            sum2 += a * *reinterpret_cast<const float *>(src1_col0 + 2 * c.src1_nb1 + rhs);
            sum3 += a * *reinterpret_cast<const float *>(src1_col0 + 3 * c.src1_nb1 + rhs);
            sum4 += a * *reinterpret_cast<const float *>(src1_col0 + 4 * c.src1_nb1 + rhs);
            sum5 += a * *reinterpret_cast<const float *>(src1_col0 + 5 * c.src1_nb1 + rhs);
            sum6 += a * *reinterpret_cast<const float *>(src1_col0 + 6 * c.src1_nb1 + rhs);
            sum7 += a * *reinterpret_cast<const float *>(src1_col0 + 7 * c.src1_nb1 + rhs);
        }
    } else if (rem == 1) {
        for (long long i = tid; i < c.k; i += 256) {
            const float a = __half2float(*reinterpret_cast<const __half *>(src0_row + i * sizeof(__half)));
            const long long rhs = i * sizeof(float);
            sum0 += a * *reinterpret_cast<const float *>(src1_col0 + rhs);
        }
    } else if (rem <= 4) {
        for (long long i = tid; i < c.k; i += 256) {
            const float a = __half2float(*reinterpret_cast<const __half *>(src0_row + i * sizeof(__half)));
            const long long rhs = i * sizeof(float);
            sum0 += a * *reinterpret_cast<const float *>(src1_col0 + rhs);
            if (rem > 1) { sum1 += a * *reinterpret_cast<const float *>(src1_col0 + c.src1_nb1 + rhs); }
            if (rem > 2) { sum2 += a * *reinterpret_cast<const float *>(src1_col0 + 2 * c.src1_nb1 + rhs); }
            if (rem > 3) { sum3 += a * *reinterpret_cast<const float *>(src1_col0 + 3 * c.src1_nb1 + rhs); }
        }
    } else {
        for (long long i = tid; i < c.k; i += 256) {
            const float a = __half2float(*reinterpret_cast<const __half *>(src0_row + i * sizeof(__half)));
            const long long rhs = i * sizeof(float);
            if (i11 + 0 < c.cols) { sum0 += a * *reinterpret_cast<const float *>(src1_col0 + rhs); }
            if (i11 + 1 < c.cols) { sum1 += a * *reinterpret_cast<const float *>(src1_col0 + c.src1_nb1 + rhs); }
            if (i11 + 2 < c.cols) { sum2 += a * *reinterpret_cast<const float *>(src1_col0 + 2 * c.src1_nb1 + rhs); }
            if (i11 + 3 < c.cols) { sum3 += a * *reinterpret_cast<const float *>(src1_col0 + 3 * c.src1_nb1 + rhs); }
            if (i11 + 4 < c.cols) { sum4 += a * *reinterpret_cast<const float *>(src1_col0 + 4 * c.src1_nb1 + rhs); }
            if (i11 + 5 < c.cols) { sum5 += a * *reinterpret_cast<const float *>(src1_col0 + 5 * c.src1_nb1 + rhs); }
            if (i11 + 6 < c.cols) { sum6 += a * *reinterpret_cast<const float *>(src1_col0 + 6 * c.src1_nb1 + rhs); }
            if (i11 + 7 < c.cols) { sum7 += a * *reinterpret_cast<const float *>(src1_col0 + 7 * c.src1_nb1 + rhs); }
        }
    }

    if (rem == 1) {
        sum0 = hrx_reduce_256(sum0, sumsh);
    } else if (rem <= 4) {
        hrx_reduce4_256(sum0, sum1, sum2, sum3, sumsh);
    } else {
        hrx_reduce8_256(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sumsh);
    }

    if (tid == 0) {
        char * dst_row = reinterpret_cast<char *>(dst) +
            row * sizeof(float) + i11 * c.dst_nb1 + i12 * c.dst_nb2 + i13 * c.dst_nb3;
        *reinterpret_cast<float *>(dst_row) = sum0;
        if (rem > 1) { *reinterpret_cast<float *>(dst_row + c.dst_nb1) = sum1; }
        if (rem > 2) { *reinterpret_cast<float *>(dst_row + 2 * c.dst_nb1) = sum2; }
        if (rem > 3) { *reinterpret_cast<float *>(dst_row + 3 * c.dst_nb1) = sum3; }
        if (rem > 4) { *reinterpret_cast<float *>(dst_row + 4 * c.dst_nb1) = sum4; }
        if (rem > 5) { *reinterpret_cast<float *>(dst_row + 5 * c.dst_nb1) = sum5; }
        if (rem > 6) { *reinterpret_cast<float *>(dst_row + 6 * c.dst_nb1) = sum6; }
        if (rem > 7) { *reinterpret_cast<float *>(dst_row + 7 * c.dst_nb1) = sum7; }
    }
}

extern "C" __global__ void hrx_mul_mat_vec_f16_batched_cols16_f32(
        const __half * src0, const float * src1, float * dst,
        hrx_mul_mat_vec_f16_batched_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.rows) {
        return;
    }

    const long long col_groups = (c.cols + 15) / 16;
    const long long col_group = outer % col_groups;
    const long long i11 = col_group * 16;
    const long long t = outer / col_groups;
    const long long i12 = t % c.dst_ne2;
    const long long i13 = t / c.dst_ne2;
    if (i13 >= c.dst_ne3) {
        return;
    }

    const long long src0_i02 = c.src0_ne2 == c.dst_ne2 ? i12 : i12 / (c.dst_ne2 / c.src0_ne2);
    const long long src0_i03 = c.src0_ne3 == c.dst_ne3 ? i13 : i13 / (c.dst_ne3 / c.src0_ne3);
    const char * src0_row = reinterpret_cast<const char *>(src0) +
        row * c.src0_nb1 + src0_i02 * c.src0_nb2 + src0_i03 * c.src0_nb3;
    const char * src1_col0 = reinterpret_cast<const char *>(src1) +
        i11 * c.src1_nb1 + i12 * c.src1_nb2 + i13 * c.src1_nb3;

    __shared__ float sumsh0[8 * (256 / 32)];
    __shared__ float sumsh1[8 * (256 / 32)];
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;
    float sum8 = 0.0f;
    float sum9 = 0.0f;
    float sum10 = 0.0f;
    float sum11 = 0.0f;
    float sum12 = 0.0f;
    float sum13 = 0.0f;
    float sum14 = 0.0f;
    float sum15 = 0.0f;
    const long long rem = c.cols - i11;
    if (i11 + 15 < c.cols) {
        for (long long i = tid; i < c.k; i += 256) {
            const float a = __half2float(*reinterpret_cast<const __half *>(src0_row + i * sizeof(__half)));
            const long long rhs = i * sizeof(float);
            sum0 += a * *reinterpret_cast<const float *>(src1_col0 + rhs);
            sum1 += a * *reinterpret_cast<const float *>(src1_col0 + c.src1_nb1 + rhs);
            sum2 += a * *reinterpret_cast<const float *>(src1_col0 + 2 * c.src1_nb1 + rhs);
            sum3 += a * *reinterpret_cast<const float *>(src1_col0 + 3 * c.src1_nb1 + rhs);
            sum4 += a * *reinterpret_cast<const float *>(src1_col0 + 4 * c.src1_nb1 + rhs);
            sum5 += a * *reinterpret_cast<const float *>(src1_col0 + 5 * c.src1_nb1 + rhs);
            sum6 += a * *reinterpret_cast<const float *>(src1_col0 + 6 * c.src1_nb1 + rhs);
            sum7 += a * *reinterpret_cast<const float *>(src1_col0 + 7 * c.src1_nb1 + rhs);
            sum8 += a * *reinterpret_cast<const float *>(src1_col0 + 8 * c.src1_nb1 + rhs);
            sum9 += a * *reinterpret_cast<const float *>(src1_col0 + 9 * c.src1_nb1 + rhs);
            sum10 += a * *reinterpret_cast<const float *>(src1_col0 + 10 * c.src1_nb1 + rhs);
            sum11 += a * *reinterpret_cast<const float *>(src1_col0 + 11 * c.src1_nb1 + rhs);
            sum12 += a * *reinterpret_cast<const float *>(src1_col0 + 12 * c.src1_nb1 + rhs);
            sum13 += a * *reinterpret_cast<const float *>(src1_col0 + 13 * c.src1_nb1 + rhs);
            sum14 += a * *reinterpret_cast<const float *>(src1_col0 + 14 * c.src1_nb1 + rhs);
            sum15 += a * *reinterpret_cast<const float *>(src1_col0 + 15 * c.src1_nb1 + rhs);
        }
    } else if (rem == 1) {
        for (long long i = tid; i < c.k; i += 256) {
            const float a = __half2float(*reinterpret_cast<const __half *>(src0_row + i * sizeof(__half)));
            const long long rhs = i * sizeof(float);
            sum0 += a * *reinterpret_cast<const float *>(src1_col0 + rhs);
        }
    } else if (rem <= 4) {
        for (long long i = tid; i < c.k; i += 256) {
            const float a = __half2float(*reinterpret_cast<const __half *>(src0_row + i * sizeof(__half)));
            const long long rhs = i * sizeof(float);
            sum0 += a * *reinterpret_cast<const float *>(src1_col0 + rhs);
            if (rem > 1) { sum1 += a * *reinterpret_cast<const float *>(src1_col0 + c.src1_nb1 + rhs); }
            if (rem > 2) { sum2 += a * *reinterpret_cast<const float *>(src1_col0 + 2 * c.src1_nb1 + rhs); }
            if (rem > 3) { sum3 += a * *reinterpret_cast<const float *>(src1_col0 + 3 * c.src1_nb1 + rhs); }
        }
    } else if (rem <= 8) {
        for (long long i = tid; i < c.k; i += 256) {
            const float a = __half2float(*reinterpret_cast<const __half *>(src0_row + i * sizeof(__half)));
            const long long rhs = i * sizeof(float);
            sum0 += a * *reinterpret_cast<const float *>(src1_col0 + rhs);
            sum1 += a * *reinterpret_cast<const float *>(src1_col0 + c.src1_nb1 + rhs);
            sum2 += a * *reinterpret_cast<const float *>(src1_col0 + 2 * c.src1_nb1 + rhs);
            sum3 += a * *reinterpret_cast<const float *>(src1_col0 + 3 * c.src1_nb1 + rhs);
            sum4 += a * *reinterpret_cast<const float *>(src1_col0 + 4 * c.src1_nb1 + rhs);
            if (rem > 5) { sum5 += a * *reinterpret_cast<const float *>(src1_col0 + 5 * c.src1_nb1 + rhs); }
            if (rem > 6) { sum6 += a * *reinterpret_cast<const float *>(src1_col0 + 6 * c.src1_nb1 + rhs); }
            if (rem > 7) { sum7 += a * *reinterpret_cast<const float *>(src1_col0 + 7 * c.src1_nb1 + rhs); }
        }
    } else {
        for (long long i = tid; i < c.k; i += 256) {
            const float a = __half2float(*reinterpret_cast<const __half *>(src0_row + i * sizeof(__half)));
            const long long rhs = i * sizeof(float);
            if (i11 + 0 < c.cols) { sum0 += a * *reinterpret_cast<const float *>(src1_col0 + rhs); }
            if (i11 + 1 < c.cols) { sum1 += a * *reinterpret_cast<const float *>(src1_col0 + c.src1_nb1 + rhs); }
            if (i11 + 2 < c.cols) { sum2 += a * *reinterpret_cast<const float *>(src1_col0 + 2 * c.src1_nb1 + rhs); }
            if (i11 + 3 < c.cols) { sum3 += a * *reinterpret_cast<const float *>(src1_col0 + 3 * c.src1_nb1 + rhs); }
            if (i11 + 4 < c.cols) { sum4 += a * *reinterpret_cast<const float *>(src1_col0 + 4 * c.src1_nb1 + rhs); }
            if (i11 + 5 < c.cols) { sum5 += a * *reinterpret_cast<const float *>(src1_col0 + 5 * c.src1_nb1 + rhs); }
            if (i11 + 6 < c.cols) { sum6 += a * *reinterpret_cast<const float *>(src1_col0 + 6 * c.src1_nb1 + rhs); }
            if (i11 + 7 < c.cols) { sum7 += a * *reinterpret_cast<const float *>(src1_col0 + 7 * c.src1_nb1 + rhs); }
            if (i11 + 8 < c.cols) { sum8 += a * *reinterpret_cast<const float *>(src1_col0 + 8 * c.src1_nb1 + rhs); }
            if (i11 + 9 < c.cols) { sum9 += a * *reinterpret_cast<const float *>(src1_col0 + 9 * c.src1_nb1 + rhs); }
            if (i11 + 10 < c.cols) { sum10 += a * *reinterpret_cast<const float *>(src1_col0 + 10 * c.src1_nb1 + rhs); }
            if (i11 + 11 < c.cols) { sum11 += a * *reinterpret_cast<const float *>(src1_col0 + 11 * c.src1_nb1 + rhs); }
            if (i11 + 12 < c.cols) { sum12 += a * *reinterpret_cast<const float *>(src1_col0 + 12 * c.src1_nb1 + rhs); }
            if (i11 + 13 < c.cols) { sum13 += a * *reinterpret_cast<const float *>(src1_col0 + 13 * c.src1_nb1 + rhs); }
            if (i11 + 14 < c.cols) { sum14 += a * *reinterpret_cast<const float *>(src1_col0 + 14 * c.src1_nb1 + rhs); }
            if (i11 + 15 < c.cols) { sum15 += a * *reinterpret_cast<const float *>(src1_col0 + 15 * c.src1_nb1 + rhs); }
        }
    }

    if (rem == 1) {
        sum0 = hrx_reduce_256(sum0, sumsh0);
    } else if (rem <= 4) {
        hrx_reduce4_256(sum0, sum1, sum2, sum3, sumsh0);
    } else if (rem <= 8) {
        hrx_reduce8_256(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sumsh0);
    } else {
        hrx_reduce8_256(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sumsh0);
        hrx_reduce8_256(sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15, sumsh1);
    }

    if (tid == 0) {
        char * dst_row = reinterpret_cast<char *>(dst) +
            row * sizeof(float) + i11 * c.dst_nb1 + i12 * c.dst_nb2 + i13 * c.dst_nb3;
        if (i11 + 0 < c.cols) { *reinterpret_cast<float *>(dst_row) = sum0; }
        if (i11 + 1 < c.cols) { *reinterpret_cast<float *>(dst_row + c.dst_nb1) = sum1; }
        if (i11 + 2 < c.cols) { *reinterpret_cast<float *>(dst_row + 2 * c.dst_nb1) = sum2; }
        if (i11 + 3 < c.cols) { *reinterpret_cast<float *>(dst_row + 3 * c.dst_nb1) = sum3; }
        if (i11 + 4 < c.cols) { *reinterpret_cast<float *>(dst_row + 4 * c.dst_nb1) = sum4; }
        if (i11 + 5 < c.cols) { *reinterpret_cast<float *>(dst_row + 5 * c.dst_nb1) = sum5; }
        if (i11 + 6 < c.cols) { *reinterpret_cast<float *>(dst_row + 6 * c.dst_nb1) = sum6; }
        if (i11 + 7 < c.cols) { *reinterpret_cast<float *>(dst_row + 7 * c.dst_nb1) = sum7; }
        if (i11 + 8 < c.cols) { *reinterpret_cast<float *>(dst_row + 8 * c.dst_nb1) = sum8; }
        if (i11 + 9 < c.cols) { *reinterpret_cast<float *>(dst_row + 9 * c.dst_nb1) = sum9; }
        if (i11 + 10 < c.cols) { *reinterpret_cast<float *>(dst_row + 10 * c.dst_nb1) = sum10; }
        if (i11 + 11 < c.cols) { *reinterpret_cast<float *>(dst_row + 11 * c.dst_nb1) = sum11; }
        if (i11 + 12 < c.cols) { *reinterpret_cast<float *>(dst_row + 12 * c.dst_nb1) = sum12; }
        if (i11 + 13 < c.cols) { *reinterpret_cast<float *>(dst_row + 13 * c.dst_nb1) = sum13; }
        if (i11 + 14 < c.cols) { *reinterpret_cast<float *>(dst_row + 14 * c.dst_nb1) = sum14; }
        if (i11 + 15 < c.cols) { *reinterpret_cast<float *>(dst_row + 15 * c.dst_nb1) = sum15; }
    }
}
