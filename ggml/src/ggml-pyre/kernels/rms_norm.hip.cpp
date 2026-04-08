struct pyre_rms_norm_constants {
    long long ncols;
    long long nrows;
    float eps;
    int _pad;
    long long src_nb1;
    long long dst_nb1;
};

extern "C" __global__ void pyre_rms_norm_f32(
        const float * src, float * dst, pyre_rms_norm_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.nrows) {
        return;
    }

    __shared__ float sumsh[512];

    const char * src_row = reinterpret_cast<const char *>(src) + row * c.src_nb1;
    char * dst_row = reinterpret_cast<char *>(dst) + row * c.dst_nb1;
    float sum = 0.0f;
    for (long long col = tid; col < c.ncols; col += 512) {
        const float value = *reinterpret_cast<const float *>(src_row + col * sizeof(float));
        sum += value * value;
    }

    sumsh[tid] = sum;
    __builtin_amdgcn_s_barrier();

    for (unsigned int step = 256; step > 0; step >>= 1) {
        if (tid < step) {
            sum += sumsh[tid + step];
            sumsh[tid] = sum;
        }
        __builtin_amdgcn_s_barrier();
    }

    const float scale = 1.0f / __builtin_sqrtf(sumsh[0] / (float) c.ncols + c.eps);
    for (long long col = tid; col < c.ncols; col += 512) {
        *reinterpret_cast<float *>(dst_row + col * sizeof(float)) =
            *reinterpret_cast<const float *>(src_row + col * sizeof(float)) * scale;
    }
}
