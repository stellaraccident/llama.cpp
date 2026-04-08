#include <hip/hip_runtime.h>
#include <stdint.h>

struct pyre_mul_f32_broadcast_constants {
    long long ne0;
    long long ne1;
    long long src0_nb1;
    long long src1_nb1;
    long long dst_nb1;
};

extern "C" __global__ void pyre_mul_f32_broadcast(
        const float * src0, const float * src1, float * dst,
        pyre_mul_f32_broadcast_constants c) {
    const long long idx = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    const long long n = c.ne0 * c.ne1;
    if (idx >= n) {
        return;
    }

    const long long col = idx % c.ne0;
    const long long row = idx / c.ne0;
    const char * src0_row = reinterpret_cast<const char *>(src0) + row * c.src0_nb1;
    const char * src1_row = reinterpret_cast<const char *>(src1) + (c.src1_nb1 == 0 ? 0 : row * c.src1_nb1);
    char * dst_row = reinterpret_cast<char *>(dst) + row * c.dst_nb1;
    *reinterpret_cast<float *>(dst_row + col * sizeof(float)) =
        *reinterpret_cast<const float *>(src0_row + col * sizeof(float)) *
        *reinterpret_cast<const float *>(src1_row + col * sizeof(float));
}
