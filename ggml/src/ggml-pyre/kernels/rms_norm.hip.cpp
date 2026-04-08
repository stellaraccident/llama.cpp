extern "C" __global__ void pyre_rms_norm_f32(
        const float * src, float * dst, long long ncols, long long nrows, float eps) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= nrows) {
        return;
    }

    __shared__ float sumsh[512];

    const float * src_row = src + row * ncols;
    float * dst_row = dst + row * ncols;
    float sum = 0.0f;
    for (long long col = tid; col < ncols; col += 512) {
        const float value = src_row[col];
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

    const float scale = 1.0f / __builtin_sqrtf(sumsh[0] / (float) ncols + eps);
    for (long long col = tid; col < ncols; col += 512) {
        dst_row[col] = src_row[col] * scale;
    }
}
