struct pyre_add_rms_norm_mul_f32_broadcast_constants {
    long long ncols;
    long long nrows;
    long long ne1;
    long long ne2;
    long long src1_ne0;
    long long src0_nb1;
    long long src0_nb2;
    long long src0_nb3;
    long long src1_nb1;
    long long src1_nb2;
    long long src1_nb3;
    long long weight_ne0;
    long long weight_ne1;
    long long weight_ne2;
    long long weight_ne3;
    long long weight_nb1;
    long long weight_nb2;
    long long weight_nb3;
    long long add_dst_nb1;
    long long add_dst_nb2;
    long long add_dst_nb3;
    long long dst_nb1;
    long long dst_nb2;
    long long dst_nb3;
    float eps;
    int _pad;
};

extern "C" __global__ void pyre_add_rms_norm_mul_f32_broadcast(
        const float * src0, const float * src1, float * add_dst, const float * weight, float * dst,
        pyre_add_rms_norm_mul_f32_broadcast_constants c) {
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

    const char * src0_row = reinterpret_cast<const char *>(src0) +
        i1 * c.src0_nb1 + i2 * c.src0_nb2 + i3 * c.src0_nb3;
    const char * src1_row = reinterpret_cast<const char *>(src1) +
        (c.src1_nb1 == 0 ? 0 : i1 * c.src1_nb1) +
        (c.src1_nb2 == 0 ? 0 : i2 * c.src1_nb2) +
        (c.src1_nb3 == 0 ? 0 : i3 * c.src1_nb3);
    const char * weight_row = reinterpret_cast<const char *>(weight) +
        wi1 * c.weight_nb1 + wi2 * c.weight_nb2 + wi3 * c.weight_nb3;
    char * dst_row = reinterpret_cast<char *>(dst) +
        i1 * c.dst_nb1 + i2 * c.dst_nb2 + i3 * c.dst_nb3;
    char * add_dst_row = reinterpret_cast<char *>(add_dst) +
        i1 * c.add_dst_nb1 + i2 * c.add_dst_nb2 + i3 * c.add_dst_nb3;

    float sum = 0.0f;
    for (long long col = tid; col < c.ncols; col += 512) {
        const long long src1_col = c.src1_ne0 == 1 ? 0 : col;
        const float value =
            *reinterpret_cast<const float *>(src0_row + col * sizeof(float)) +
            *reinterpret_cast<const float *>(src1_row + src1_col * sizeof(float));
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
        const long long src1_col = c.src1_ne0 == 1 ? 0 : col;
        const long long wcol = c.weight_ne0 == 1 ? 0 : col;
        const float value =
            *reinterpret_cast<const float *>(src0_row + col * sizeof(float)) +
            *reinterpret_cast<const float *>(src1_row + src1_col * sizeof(float));
        *reinterpret_cast<float *>(add_dst_row + col * sizeof(float)) = value;
        const float weight_value = *reinterpret_cast<const float *>(weight_row + wcol * sizeof(float));
        *reinterpret_cast<float *>(dst_row + col * sizeof(float)) = value * scale * weight_value;
    }
}
