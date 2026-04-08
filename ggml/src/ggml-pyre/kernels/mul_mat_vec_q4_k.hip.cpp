#include <hip/hip_fp16.h>
#include <stdint.h>

struct pyre_block_q4_K {
    unsigned short d;
    unsigned short dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

static __device__ __forceinline__ void pyre_get_scale_min_k4(
        int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

extern "C" __global__ void pyre_mul_mat_vec_q4_k_f32(
        const pyre_block_q4_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    __shared__ float sumsh[256];

    const long long blocks_per_row = k / 256;
    const pyre_block_q4_K * row_blocks = src0 + row * blocks_per_row;
    const float * src1_col = src1 + col * k;
    float sum = 0.0f;

    for (long long i = tid; i < k; i += 256) {
        const long long block_idx = i / 256;
        const int in_block = static_cast<int>(i - block_idx * 256);
        const int group = in_block / 32;
        const int lane = in_block & 31;
        const pyre_block_q4_K * block = row_blocks + block_idx;

        uint8_t sc = 0;
        uint8_t m = 0;
        pyre_get_scale_min_k4(group, block->scales, &sc, &m);

        const uint8_t packed = block->qs[(group / 2) * 32 + lane];
        const float q = (group & 1) ? static_cast<float>(packed >> 4) : static_cast<float>(packed & 0x0F);
        const float d = __half2float(__ushort_as_half(block->d)) * static_cast<float>(sc);
        const float min = __half2float(__ushort_as_half(block->dmin)) * static_cast<float>(m);
        sum += (d * q - min) * src1_col[i];
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
