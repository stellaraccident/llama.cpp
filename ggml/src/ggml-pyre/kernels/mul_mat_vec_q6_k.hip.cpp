#include <hip/hip_fp16.h>
#include <stdint.h>

struct pyre_block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    unsigned short d;
};

static __device__ __forceinline__ float pyre_dequant_q6_k(
        const pyre_block_q6_K * block, int in_block) {
    const int half = in_block / 128;
    const int idx = in_block - half * 128;
    const int lane = idx & 31;
    const int ql_base = half * 64;
    const int qh_base = half * 32;
    const int sc_base = half * 8;

    int q = 0;
    int scale = 0;
    if (idx < 32) {
        q = (block->ql[ql_base + lane] & 0x0F) | (((block->qh[qh_base + lane] >> 0) & 3) << 4);
        scale = block->scales[sc_base + lane / 16 + 0];
    } else if (idx < 64) {
        q = (block->ql[ql_base + lane + 32] & 0x0F) | (((block->qh[qh_base + lane] >> 2) & 3) << 4);
        scale = block->scales[sc_base + lane / 16 + 2];
    } else if (idx < 96) {
        q = (block->ql[ql_base + lane] >> 4) | (((block->qh[qh_base + lane] >> 4) & 3) << 4);
        scale = block->scales[sc_base + lane / 16 + 4];
    } else {
        q = (block->ql[ql_base + lane + 32] >> 4) | (((block->qh[qh_base + lane] >> 6) & 3) << 4);
        scale = block->scales[sc_base + lane / 16 + 6];
    }

    return __half2float(__ushort_as_half(block->d)) * static_cast<float>(scale) *
        static_cast<float>(q - 32);
}

extern "C" __global__ void pyre_mul_mat_vec_q6_k_f32(
        const pyre_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    __shared__ float sumsh[256];

    const long long blocks_per_row = k / 256;
    const pyre_block_q6_K * row_blocks = src0 + row * blocks_per_row;
    const float * src1_col = src1 + col * k;
    float sum = 0.0f;

    for (long long i = tid; i < k; i += 256) {
        const long long block_idx = i / 256;
        const int in_block = static_cast<int>(i - block_idx * 256);
        sum += pyre_dequant_q6_k(row_blocks + block_idx, in_block) * src1_col[i];
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
