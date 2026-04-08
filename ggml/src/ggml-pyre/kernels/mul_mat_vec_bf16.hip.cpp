#include <hip/hip_fp16.h>
#include <stdint.h>

static __device__ __forceinline__ float pyre_bf16_to_f32(uint16_t value) {
    union {
        uint32_t u;
        float f;
    } bits = { static_cast<uint32_t>(value) << 16 };
    return bits.f;
}

extern "C" __global__ void pyre_mul_mat_vec_bf16_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    __shared__ float sumsh[256];

    const uint16_t * src0_row = src0 + row * k;
    const float * src1_col = src1 + col * k;
    float sum = 0.0f;
    for (long long i = tid; i < k; i += 256) {
        sum += pyre_bf16_to_f32(src0_row[i]) * src1_col[i];
    }

    sumsh[tid] = sum;
    __builtin_amdgcn_s_barrier();

    for (unsigned int step = 128; step > 0; step >>= 1) {
        if (tid < step) {
            sum += sumsh[tid + step];
            sumsh[tid] = sum;
        }
        __builtin_amdgcn_s_barrier();
    }

    if (tid == 0) {
        dst[col * rows + row] = sumsh[0];
    }
}
