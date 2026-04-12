#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ float pyre_bf16_swiglu_to_f32(uint16_t value) {
    union {
        uint32_t u;
        float f;
    } bits = { static_cast<uint32_t>(value) << 16 };
    return bits.f;
}

template <int WG_SIZE>
static __device__ __forceinline__ float pyre_reduce_bf16_swiglu(float sum, float * shared) {
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
static __device__ __forceinline__ void pyre_reduce8_bf16_swiglu(
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
static __device__ __forceinline__ void pyre_mul_mat_vec_bf16_swiglu_f32_impl(
        const uint16_t * gate,
        const uint16_t * up,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    __shared__ float gate_sumsh[(WG_SIZE + 31) / 32];
    __shared__ float up_sumsh[(WG_SIZE + 31) / 32];

    const uint16_t * gate_row = gate + row * k;
    const uint16_t * up_row = up + row * k;
    const float * src1_col = src1 + col * k;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (long long i = tid; i < k; i += WG_SIZE) {
        const float b = src1_col[i];
        gate_sum += pyre_bf16_swiglu_to_f32(gate_row[i]) * b;
        up_sum += pyre_bf16_swiglu_to_f32(up_row[i]) * b;
    }

    gate_sum = pyre_reduce_bf16_swiglu<WG_SIZE>(gate_sum, gate_sumsh);
    up_sum = pyre_reduce_bf16_swiglu<WG_SIZE>(up_sum, up_sumsh);

    if (tid == 0) {
        const float silu_gate = gate_sum / (1.0f + __expf(-gate_sum));
        dst[col * rows + row] = up_sum * silu_gate;
    }
}

extern "C" __global__ void pyre_mul_mat_vec_bf16_swiglu_f32(
        const uint16_t * gate,
        const uint16_t * up,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    pyre_mul_mat_vec_bf16_swiglu_f32_impl<256>(gate, up, src1, dst, k, rows, cols);
}

extern "C" __global__ void pyre_mul_mat_vec_bf16_swiglu_wg128_f32(
        const uint16_t * gate,
        const uint16_t * up,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    pyre_mul_mat_vec_bf16_swiglu_f32_impl<128>(gate, up, src1, dst, k, rows, cols);
}

extern "C" __global__ void pyre_mul_mat_vec_bf16_swiglu_wg64_f32(
        const uint16_t * gate,
        const uint16_t * up,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    pyre_mul_mat_vec_bf16_swiglu_f32_impl<64>(gate, up, src1, dst, k, rows, cols);
}

extern "C" __global__ void pyre_mul_mat_vec_bf16_swiglu_cols1_f32(
        const uint16_t * gate,
        const uint16_t * up,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows) {
        return;
    }
    (void) cols;

    __shared__ float gate_sumsh[8];
    __shared__ float up_sumsh[8];

    const uint16_t * gate_row = gate + row * k;
    const uint16_t * up_row = up + row * k;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (long long i = tid; i < k; i += 256) {
        const float b = src1[i];
        gate_sum += pyre_bf16_swiglu_to_f32(gate_row[i]) * b;
        up_sum += pyre_bf16_swiglu_to_f32(up_row[i]) * b;
    }

    gate_sum = pyre_reduce_bf16_swiglu<256>(gate_sum, gate_sumsh);
    up_sum = pyre_reduce_bf16_swiglu<256>(up_sum, up_sumsh);

    if (tid == 0) {
        const float silu_gate = gate_sum / (1.0f + __expf(-gate_sum));
        dst[row] = up_sum * silu_gate;
    }
}

extern "C" __global__ void pyre_mul_mat_vec_bf16_swiglu_cols4_f32(
        const uint16_t * gate,
        const uint16_t * up,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col0 = __builtin_amdgcn_workgroup_id_y() * 4;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col0 >= cols) {
        return;
    }

    __shared__ float sumsh[8 * (256 / 32)];

    const uint16_t * gate_row = gate + row * k;
    const uint16_t * up_row = up + row * k;
    const float * src1_col0 = src1 + col0 * k;
    float gate_sum0 = 0.0f;
    float gate_sum1 = 0.0f;
    float gate_sum2 = 0.0f;
    float gate_sum3 = 0.0f;
    float up_sum0 = 0.0f;
    float up_sum1 = 0.0f;
    float up_sum2 = 0.0f;
    float up_sum3 = 0.0f;
    for (long long i = tid; i < k; i += 256) {
        const float g = pyre_bf16_swiglu_to_f32(gate_row[i]);
        const float u = pyre_bf16_swiglu_to_f32(up_row[i]);
        const float b0 = src1_col0[i];
        const float b1 = src1_col0[k + i];
        const float b2 = src1_col0[2 * k + i];
        const float b3 = src1_col0[3 * k + i];
        gate_sum0 += g * b0;
        gate_sum1 += g * b1;
        gate_sum2 += g * b2;
        gate_sum3 += g * b3;
        up_sum0 += u * b0;
        up_sum1 += u * b1;
        up_sum2 += u * b2;
        up_sum3 += u * b3;
    }

    pyre_reduce8_bf16_swiglu<256>(
        gate_sum0, gate_sum1, gate_sum2, gate_sum3,
        up_sum0, up_sum1, up_sum2, up_sum3, sumsh);

    if (tid == 0) {
        float * dst_col0 = dst + col0 * rows + row;
        const float silu_gate0 = gate_sum0 / (1.0f + __expf(-gate_sum0));
        const float silu_gate1 = gate_sum1 / (1.0f + __expf(-gate_sum1));
        const float silu_gate2 = gate_sum2 / (1.0f + __expf(-gate_sum2));
        const float silu_gate3 = gate_sum3 / (1.0f + __expf(-gate_sum3));
        dst_col0[0] = up_sum0 * silu_gate0;
        dst_col0[rows] = up_sum1 * silu_gate1;
        dst_col0[2 * rows] = up_sum2 * silu_gate2;
        dst_col0[3 * rows] = up_sum3 * silu_gate3;
    }
}

extern "C" __global__ void pyre_mul_mat_vec_bf16_swiglu_cols8_f32(
        const uint16_t * gate,
        const uint16_t * up,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col0 = __builtin_amdgcn_workgroup_id_y() * 8;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col0 + 7 >= cols) {
        return;
    }

    __shared__ float gate_sumsh[8 * (256 / 32)];
    __shared__ float up_sumsh[8 * (256 / 32)];

    const uint16_t * gate_row = gate + row * k;
    const uint16_t * up_row = up + row * k;
    const float * src1_col0 = src1 + col0 * k;
    float gate_sum0 = 0.0f;
    float gate_sum1 = 0.0f;
    float gate_sum2 = 0.0f;
    float gate_sum3 = 0.0f;
    float gate_sum4 = 0.0f;
    float gate_sum5 = 0.0f;
    float gate_sum6 = 0.0f;
    float gate_sum7 = 0.0f;
    float up_sum0 = 0.0f;
    float up_sum1 = 0.0f;
    float up_sum2 = 0.0f;
    float up_sum3 = 0.0f;
    float up_sum4 = 0.0f;
    float up_sum5 = 0.0f;
    float up_sum6 = 0.0f;
    float up_sum7 = 0.0f;
    for (long long i = tid; i < k; i += 256) {
        const float g = pyre_bf16_swiglu_to_f32(gate_row[i]);
        const float u = pyre_bf16_swiglu_to_f32(up_row[i]);
        const float b0 = src1_col0[i];
        const float b1 = src1_col0[k + i];
        const float b2 = src1_col0[2 * k + i];
        const float b3 = src1_col0[3 * k + i];
        const float b4 = src1_col0[4 * k + i];
        const float b5 = src1_col0[5 * k + i];
        const float b6 = src1_col0[6 * k + i];
        const float b7 = src1_col0[7 * k + i];
        gate_sum0 += g * b0;
        gate_sum1 += g * b1;
        gate_sum2 += g * b2;
        gate_sum3 += g * b3;
        gate_sum4 += g * b4;
        gate_sum5 += g * b5;
        gate_sum6 += g * b6;
        gate_sum7 += g * b7;
        up_sum0 += u * b0;
        up_sum1 += u * b1;
        up_sum2 += u * b2;
        up_sum3 += u * b3;
        up_sum4 += u * b4;
        up_sum5 += u * b5;
        up_sum6 += u * b6;
        up_sum7 += u * b7;
    }

    pyre_reduce8_bf16_swiglu<256>(
        gate_sum0, gate_sum1, gate_sum2, gate_sum3,
        gate_sum4, gate_sum5, gate_sum6, gate_sum7, gate_sumsh);
    pyre_reduce8_bf16_swiglu<256>(
        up_sum0, up_sum1, up_sum2, up_sum3,
        up_sum4, up_sum5, up_sum6, up_sum7, up_sumsh);

    if (tid == 0) {
        float * dst_col0 = dst + col0 * rows + row;
        const float silu_gate0 = gate_sum0 / (1.0f + __expf(-gate_sum0));
        const float silu_gate1 = gate_sum1 / (1.0f + __expf(-gate_sum1));
        const float silu_gate2 = gate_sum2 / (1.0f + __expf(-gate_sum2));
        const float silu_gate3 = gate_sum3 / (1.0f + __expf(-gate_sum3));
        const float silu_gate4 = gate_sum4 / (1.0f + __expf(-gate_sum4));
        const float silu_gate5 = gate_sum5 / (1.0f + __expf(-gate_sum5));
        const float silu_gate6 = gate_sum6 / (1.0f + __expf(-gate_sum6));
        const float silu_gate7 = gate_sum7 / (1.0f + __expf(-gate_sum7));
        dst_col0[0] = up_sum0 * silu_gate0;
        dst_col0[rows] = up_sum1 * silu_gate1;
        dst_col0[2 * rows] = up_sum2 * silu_gate2;
        dst_col0[3 * rows] = up_sum3 * silu_gate3;
        dst_col0[4 * rows] = up_sum4 * silu_gate4;
        dst_col0[5 * rows] = up_sum5 * silu_gate5;
        dst_col0[6 * rows] = up_sum6 * silu_gate6;
        dst_col0[7 * rows] = up_sum7 * silu_gate7;
    }
}
