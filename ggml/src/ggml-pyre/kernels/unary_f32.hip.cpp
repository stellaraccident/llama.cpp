#include <hip/hip_runtime.h>

static __device__ __forceinline__ float pyre_sigmoid_value_f32(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

static __device__ __forceinline__ float pyre_softplus_value_f32(float x) {
    // Stable enough for the Qwen glue shapes: log(1 + exp(x)) without host libm dependencies.
    return x > 20.0f ? x : __logf(1.0f + __expf(x));
}

extern "C" __global__ void pyre_silu_f32(const float * src, float * dst, long long n) {
    const long long idx = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    if (idx < n) {
        const float x = src[idx];
        dst[idx] = x * pyre_sigmoid_value_f32(x);
    }
}

extern "C" __global__ void pyre_sigmoid_f32(const float * src, float * dst, long long n) {
    const long long idx = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    if (idx < n) {
        dst[idx] = pyre_sigmoid_value_f32(src[idx]);
    }
}

extern "C" __global__ void pyre_softplus_f32(const float * src, float * dst, long long n) {
    const long long idx = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    if (idx < n) {
        dst[idx] = pyre_softplus_value_f32(src[idx]);
    }
}

extern "C" __global__ void pyre_swiglu_f32(
        const float * src0, const float * src1, float * dst, long long n) {
    const long long idx = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    if (idx < n) {
        const float gate = src0[idx];
        dst[idx] = gate * pyre_sigmoid_value_f32(gate) * src1[idx];
    }
}
