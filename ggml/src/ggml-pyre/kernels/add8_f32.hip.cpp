#include <hip/hip_runtime.h>
#include <stdint.h>

struct pyre_add8_f32_constants {
    long long n;
};

extern "C" __global__ void pyre_add8_f32(
        const float * src0,
        const float * src1,
        const float * src2,
        const float * src3,
        const float * src4,
        const float * src5,
        const float * src6,
        const float * src7,
        float * dst,
        pyre_add8_f32_constants c) {
    const long long idx = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    if (idx >= c.n) {
        return;
    }

    dst[idx] = src0[idx] + src1[idx] + src2[idx] + src3[idx] +
        src4[idx] + src5[idx] + src6[idx] + src7[idx];
}
