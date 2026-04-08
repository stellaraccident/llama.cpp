#include <hip/hip_runtime.h>

struct pyre_row_reduce_f32_constants {
    long long ncols;
    long long nrows;
    float eps;
    int pad;
    long long src_nb1;
    long long dst_nb1;
};

extern "C" __global__ void pyre_sum_rows_f32(
        const float * src, float * dst, pyre_row_reduce_f32_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.nrows) {
        return;
    }

    __shared__ float sumsh[256];
    const char * src_row = reinterpret_cast<const char *>(src) + row * c.src_nb1;
    float sum = 0.0f;
    for (long long col = tid; col < c.ncols; col += 256) {
        sum += *reinterpret_cast<const float *>(src_row + col * sizeof(float));
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
        *reinterpret_cast<float *>(reinterpret_cast<char *>(dst) + row * c.dst_nb1) = sumsh[0];
    }
}

extern "C" __global__ void pyre_l2_norm_f32(
        const float * src, float * dst, pyre_row_reduce_f32_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.nrows) {
        return;
    }

    __shared__ float sumsh[256];
    const char * src_row = reinterpret_cast<const char *>(src) + row * c.src_nb1;
    char * dst_row = reinterpret_cast<char *>(dst) + row * c.dst_nb1;
    float sum = 0.0f;
    for (long long col = tid; col < c.ncols; col += 256) {
        const float value = *reinterpret_cast<const float *>(src_row + col * sizeof(float));
        sum += value * value;
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

    const float denom = __builtin_sqrtf(sumsh[0]);
    const float scale = 1.0f / (denom > c.eps ? denom : c.eps);
    for (long long col = tid; col < c.ncols; col += 256) {
        *reinterpret_cast<float *>(dst_row + col * sizeof(float)) =
            *reinterpret_cast<const float *>(src_row + col * sizeof(float)) * scale;
    }
}
