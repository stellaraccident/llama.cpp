#include <hip/hip_fp16.h>
#include <stdint.h>

struct pyre_block_q8_0 {
    unsigned short d;
    int8_t qs[32];
};

extern "C" __global__ void pyre_mul_mat_vec_q8_0_f32(
        const pyre_block_q8_0 * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    __shared__ float sumsh[256];

    const long long blocks_per_row = k / 32;
    const pyre_block_q8_0 * row_blocks = src0 + row * blocks_per_row;
    const float * src1_col = src1 + col * k;
    float sum = 0.0f;

    for (long long i = tid; i < k; i += 256) {
        const long long block_idx = i / 32;
        const int in_block = static_cast<int>(i - block_idx * 32);
        const pyre_block_q8_0 * block = row_blocks + block_idx;
        const float value = __half2float(__ushort_as_half(block->d)) *
            static_cast<float>(block->qs[in_block]);
        sum += value * src1_col[i];
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
