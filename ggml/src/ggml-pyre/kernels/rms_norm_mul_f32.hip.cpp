struct pyre_rms_norm_mul_constants {
    long long ncols;
    long long nrows;
    long long ne1;
    long long ne2;
    long long src_nb1;
    long long src_nb2;
    long long src_nb3;
    long long weight_ne0;
    long long weight_ne1;
    long long weight_ne2;
    long long weight_ne3;
    long long weight_nb1;
    long long weight_nb2;
    long long weight_nb3;
    long long dst_nb1;
    long long dst_nb2;
    long long dst_nb3;
    float eps;
    int _pad;
};

extern "C" __global__ void pyre_rms_norm_mul_f32(
        const float * src, const float * weight, float * dst,
        pyre_rms_norm_mul_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.nrows) {
        return;
    }

    __shared__ float sumsh[512];

    const long long i3 = row / (c.ne1 * c.ne2);
    const long long i2 = (row - i3 * c.ne1 * c.ne2) / c.ne1;
    const long long i1 = row - i3 * c.ne1 * c.ne2 - i2 * c.ne1;
    const long long wi1 = c.weight_ne1 == 1 ? 0 : i1;
    const long long wi2 = c.weight_ne2 == 1 ? 0 : i2;
    const long long wi3 = c.weight_ne3 == 1 ? 0 : i3;

    const char * src_row = reinterpret_cast<const char *>(src) +
        i1 * c.src_nb1 + i2 * c.src_nb2 + i3 * c.src_nb3;
    const char * weight_row = reinterpret_cast<const char *>(weight) +
        wi1 * c.weight_nb1 + wi2 * c.weight_nb2 + wi3 * c.weight_nb3;
    char * dst_row = reinterpret_cast<char *>(dst) +
        i1 * c.dst_nb1 + i2 * c.dst_nb2 + i3 * c.dst_nb3;

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
        const long long wcol = c.weight_ne0 == 1 ? 0 : col;
        const float src_value = *reinterpret_cast<const float *>(src_row + col * sizeof(float));
        const float weight_value = *reinterpret_cast<const float *>(weight_row + wcol * sizeof(float));
        *reinterpret_cast<float *>(dst_row + col * sizeof(float)) = src_value * scale * weight_value;
    }
}
