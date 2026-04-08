#include <hip/hip_runtime.h>

struct pyre_clamp_f32_constants {
    long long n;
    float min_value;
    float max_value;
};

extern "C" __global__ void pyre_clamp_f32(
        const float * src, float * dst, pyre_clamp_f32_constants c) {
    const long long idx = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    if (idx < c.n) {
        const float x = src[idx];
        dst[idx] = x < c.min_value ? c.min_value : (x > c.max_value ? c.max_value : x);
    }
}
