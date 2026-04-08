#include <hip/hip_fp16.h>
#include <stdint.h>

struct pyre_block_q4_K_id {
    unsigned short d;
    unsigned short dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct pyre_mul_mat_id_q4_k_constants {
    long long k;
    long long rows;
    long long n_ids;
    long long n_tokens;
    long long n_experts;
    long long src0_nb1;
    long long src0_nb2;
    long long src1_nb1;
    long long src1_nb2;
    long long ids_nb0;
    long long ids_nb1;
    long long dst_nb1;
    long long dst_nb2;
};

static __device__ __forceinline__ void pyre_get_scale_min_k4_id(
        int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

extern "C" __global__ void pyre_mul_mat_id_q4_k_f32(
        const pyre_block_q4_K_id * src0, const float * src1, const int * ids, float * dst,
        pyre_mul_mat_id_q4_k_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.rows) {
        return;
    }

    const long long id_pos = outer % c.n_ids;
    const long long token = outer / c.n_ids;
    if (token >= c.n_tokens) {
        return;
    }

    const int expert = *reinterpret_cast<const int *>(
        reinterpret_cast<const char *>(ids) + id_pos * c.ids_nb0 + token * c.ids_nb1);
    if (expert < 0 || expert >= c.n_experts) {
        return;
    }

    __shared__ float sumsh[256];
    const char * src0_row_base = reinterpret_cast<const char *>(src0) + expert * c.src0_nb2 + row * c.src0_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;
    float sum = 0.0f;

    for (long long i = tid; i < c.k; i += 256) {
        const long long block_idx = i / 256;
        const int in_block = static_cast<int>(i - block_idx * 256);
        const int group = in_block / 32;
        const int lane = in_block & 31;
        const pyre_block_q4_K_id * block = reinterpret_cast<const pyre_block_q4_K_id *>(
            src0_row_base + block_idx * sizeof(pyre_block_q4_K_id));

        uint8_t sc = 0;
        uint8_t m = 0;
        pyre_get_scale_min_k4_id(group, block->scales, &sc, &m);

        const uint8_t packed = block->qs[(group / 2) * 32 + lane];
        const float q = (group & 1) ? static_cast<float>(packed >> 4) : static_cast<float>(packed & 0x0F);
        const float d = __half2float(__ushort_as_half(block->d)) * static_cast<float>(sc);
        const float min = __half2float(__ushort_as_half(block->dmin)) * static_cast<float>(m);
        const float b = *reinterpret_cast<const float *>(src1_col + i * sizeof(float));
        sum += (d * q - min) * b;
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
        *reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst) + row * sizeof(float) + id_pos * c.dst_nb1 + token * c.dst_nb2) = sumsh[0];
    }
}
