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
static __device__ __forceinline__ void pyre_reduce4_bf16(
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
    if (WG_SIZE <= warpSize) {
        return;
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
static __device__ __forceinline__ void pyre_reduce8_bf16(
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
    if (WG_SIZE <= warpSize) {
        return;
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

    __shared__ float sumsh[4 * (256 / 32)];

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

    pyre_reduce4_bf16<256>(sum0, sum1, sum2, sum3, sumsh);

    if (tid == 0) {
        float * dst_col0 = dst + col0 * rows + row;
        dst_col0[0] = sum0;
        dst_col0[rows] = sum1;
        dst_col0[2 * rows] = sum2;
        dst_col0[3 * rows] = sum3;
    }
}

extern "C" __global__ void pyre_mul_mat_vec_bf16_cols8_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col0 = __builtin_amdgcn_workgroup_id_y() * 8;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col0 + 7 >= cols) {
        return;
    }

    __shared__ float sumsh[8 * ((256 + 31) / 32)];

    const uint16_t * src0_row = src0 + row * k;
    const float * src1_col0 = src1 + col0 * k;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;
    for (long long i = tid; i < k; i += 256) {
        const float a = pyre_bf16_to_f32(src0_row[i]);
        sum0 += a * src1_col0[i];
        sum1 += a * src1_col0[k + i];
        sum2 += a * src1_col0[2 * k + i];
        sum3 += a * src1_col0[3 * k + i];
        sum4 += a * src1_col0[4 * k + i];
        sum5 += a * src1_col0[5 * k + i];
        sum6 += a * src1_col0[6 * k + i];
        sum7 += a * src1_col0[7 * k + i];
    }

    pyre_reduce8_bf16<256>(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sumsh);

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

extern "C" __global__ void pyre_mul_mat_vec_bf16_cols16_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col0 = __builtin_amdgcn_workgroup_id_y() * 16;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col0 + 15 >= cols) {
        return;
    }

    __shared__ float sumsh0[8 * ((256 + 31) / 32)];
    __shared__ float sumsh1[8 * ((256 + 31) / 32)];

    const uint16_t * src0_row = src0 + row * k;
    const float * src1_col0 = src1 + col0 * k;
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
    for (long long i = tid; i < k; i += 256) {
        const float a = pyre_bf16_to_f32(src0_row[i]);
        sum0 += a * src1_col0[i];
        sum1 += a * src1_col0[k + i];
        sum2 += a * src1_col0[2 * k + i];
        sum3 += a * src1_col0[3 * k + i];
        sum4 += a * src1_col0[4 * k + i];
        sum5 += a * src1_col0[5 * k + i];
        sum6 += a * src1_col0[6 * k + i];
        sum7 += a * src1_col0[7 * k + i];
        sum8 += a * src1_col0[8 * k + i];
        sum9 += a * src1_col0[9 * k + i];
        sum10 += a * src1_col0[10 * k + i];
        sum11 += a * src1_col0[11 * k + i];
        sum12 += a * src1_col0[12 * k + i];
        sum13 += a * src1_col0[13 * k + i];
        sum14 += a * src1_col0[14 * k + i];
        sum15 += a * src1_col0[15 * k + i];
    }

    pyre_reduce8_bf16<256>(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sumsh0);
    pyre_reduce8_bf16<256>(sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15, sumsh1);

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
        dst_col0[8 * rows] = sum8;
        dst_col0[9 * rows] = sum9;
        dst_col0[10 * rows] = sum10;
        dst_col0[11 * rows] = sum11;
        dst_col0[12 * rows] = sum12;
        dst_col0[13 * rows] = sum13;
        dst_col0[14 * rows] = sum14;
        dst_col0[15 * rows] = sum15;
    }
}

extern "C" __global__ void pyre_mul_mat_vec_bf16_cols32_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col0 = __builtin_amdgcn_workgroup_id_y() * 32;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col0 + 31 >= cols) {
        return;
    }

    __shared__ float sumsh0[8 * ((256 + 31) / 32)];
    __shared__ float sumsh1[8 * ((256 + 31) / 32)];
    __shared__ float sumsh2[8 * ((256 + 31) / 32)];
    __shared__ float sumsh3[8 * ((256 + 31) / 32)];

    const uint16_t * src0_row = src0 + row * k;
    const float * src1_col0 = src1 + col0 * k;
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
    float sum16 = 0.0f;
    float sum17 = 0.0f;
    float sum18 = 0.0f;
    float sum19 = 0.0f;
    float sum20 = 0.0f;
    float sum21 = 0.0f;
    float sum22 = 0.0f;
    float sum23 = 0.0f;
    float sum24 = 0.0f;
    float sum25 = 0.0f;
    float sum26 = 0.0f;
    float sum27 = 0.0f;
    float sum28 = 0.0f;
    float sum29 = 0.0f;
    float sum30 = 0.0f;
    float sum31 = 0.0f;
    for (long long i = tid; i < k; i += 256) {
        const float a = pyre_bf16_to_f32(src0_row[i]);
        sum0 += a * src1_col0[i];
        sum1 += a * src1_col0[k + i];
        sum2 += a * src1_col0[2 * k + i];
        sum3 += a * src1_col0[3 * k + i];
        sum4 += a * src1_col0[4 * k + i];
        sum5 += a * src1_col0[5 * k + i];
        sum6 += a * src1_col0[6 * k + i];
        sum7 += a * src1_col0[7 * k + i];
        sum8 += a * src1_col0[8 * k + i];
        sum9 += a * src1_col0[9 * k + i];
        sum10 += a * src1_col0[10 * k + i];
        sum11 += a * src1_col0[11 * k + i];
        sum12 += a * src1_col0[12 * k + i];
        sum13 += a * src1_col0[13 * k + i];
        sum14 += a * src1_col0[14 * k + i];
        sum15 += a * src1_col0[15 * k + i];
        sum16 += a * src1_col0[16 * k + i];
        sum17 += a * src1_col0[17 * k + i];
        sum18 += a * src1_col0[18 * k + i];
        sum19 += a * src1_col0[19 * k + i];
        sum20 += a * src1_col0[20 * k + i];
        sum21 += a * src1_col0[21 * k + i];
        sum22 += a * src1_col0[22 * k + i];
        sum23 += a * src1_col0[23 * k + i];
        sum24 += a * src1_col0[24 * k + i];
        sum25 += a * src1_col0[25 * k + i];
        sum26 += a * src1_col0[26 * k + i];
        sum27 += a * src1_col0[27 * k + i];
        sum28 += a * src1_col0[28 * k + i];
        sum29 += a * src1_col0[29 * k + i];
        sum30 += a * src1_col0[30 * k + i];
        sum31 += a * src1_col0[31 * k + i];
    }

    pyre_reduce8_bf16<256>(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sumsh0);
    pyre_reduce8_bf16<256>(sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15, sumsh1);
    pyre_reduce8_bf16<256>(sum16, sum17, sum18, sum19, sum20, sum21, sum22, sum23, sumsh2);
    pyre_reduce8_bf16<256>(sum24, sum25, sum26, sum27, sum28, sum29, sum30, sum31, sumsh3);

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
        dst_col0[8 * rows] = sum8;
        dst_col0[9 * rows] = sum9;
        dst_col0[10 * rows] = sum10;
        dst_col0[11 * rows] = sum11;
        dst_col0[12 * rows] = sum12;
        dst_col0[13 * rows] = sum13;
        dst_col0[14 * rows] = sum14;
        dst_col0[15 * rows] = sum15;
        dst_col0[16 * rows] = sum16;
        dst_col0[17 * rows] = sum17;
        dst_col0[18 * rows] = sum18;
        dst_col0[19 * rows] = sum19;
        dst_col0[20 * rows] = sum20;
        dst_col0[21 * rows] = sum21;
        dst_col0[22 * rows] = sum22;
        dst_col0[23 * rows] = sum23;
        dst_col0[24 * rows] = sum24;
        dst_col0[25 * rows] = sum25;
        dst_col0[26 * rows] = sum26;
        dst_col0[27 * rows] = sum27;
        dst_col0[28 * rows] = sum28;
        dst_col0[29 * rows] = sum29;
        dst_col0[30 * rows] = sum30;
        dst_col0[31 * rows] = sum31;
    }
}
