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
