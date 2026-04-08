#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct pyre_block_q4_K_id_swiglu {
    unsigned short d;
    unsigned short dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct pyre_mul_mat_id_q4_k_swiglu_constants {
    long long k;
    long long rows;
    long long n_ids;
    long long n_tokens;
    long long n_experts;
    long long gate_nb1;
    long long gate_nb2;
    long long up_nb1;
    long long up_nb2;
    long long src1_nb1;
    long long src1_nb2;
    long long ids_nb0;
    long long ids_nb1;
    long long dst_nb1;
    long long dst_nb2;
};

static __device__ __forceinline__ void pyre_get_scale_min_k4_id_swiglu(
        int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static __device__ __forceinline__ float pyre_reduce_256_swiglu(float sum, float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (lane == 0) {
        shared[wave] = sum;
    }
    __syncthreads();

    sum = lane < (256 / warpSize) ? shared[lane] : 0.0f;
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset);
        }
    }
    return sum;
}

extern "C" __global__ void pyre_mul_mat_id_q4_k_swiglu_f32(
        const pyre_block_q4_K_id_swiglu * gate,
        const pyre_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        pyre_mul_mat_id_q4_k_swiglu_constants c) {
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

    __shared__ float gate_sumsh[256];
    __shared__ float up_sumsh[256];
    const char * gate_row_base = reinterpret_cast<const char *>(gate) + expert * c.gate_nb2 + row * c.gate_nb1;
    const char * up_row_base = reinterpret_cast<const char *>(up) + expert * c.up_nb2 + row * c.up_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    const int block_lane = tid & 63;
    const int block_slot = tid >> 6;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 4) {
        const pyre_block_q4_K_id_swiglu * gate_block = reinterpret_cast<const pyre_block_q4_K_id_swiglu *>(
            gate_row_base + block_idx * sizeof(pyre_block_q4_K_id_swiglu));
        const pyre_block_q4_K_id_swiglu * up_block = reinterpret_cast<const pyre_block_q4_K_id_swiglu *>(
            up_row_base + block_idx * sizeof(pyre_block_q4_K_id_swiglu));

        uint8_t gate_sc = 0;
        uint8_t gate_m = 0;
        uint8_t up_sc = 0;
        uint8_t up_m = 0;
        pyre_get_scale_min_k4_id_swiglu(group, gate_block->scales, &gate_sc, &gate_m);
        pyre_get_scale_min_k4_id_swiglu(group, up_block->scales, &up_sc, &up_m);

        const float gate_d = __half2float(__ushort_as_half(gate_block->d)) * static_cast<float>(gate_sc);
        const float gate_min = __half2float(__ushort_as_half(gate_block->dmin)) * static_cast<float>(gate_m);
        const float up_d = __half2float(__ushort_as_half(up_block->d)) * static_cast<float>(up_sc);
        const float up_min = __half2float(__ushort_as_half(up_block->dmin)) * static_cast<float>(up_m);
        const long long src_base = block_idx * 256 + group * 32 + lane;
        const int qs_base = (group >> 1) * 32 + lane;

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint8_t gate_packed = gate_block->qs[qs_base + j];
            const uint8_t up_packed = up_block->qs[qs_base + j];
            const float gate_q = (group & 1) ?
                static_cast<float>(gate_packed >> 4) :
                static_cast<float>(gate_packed & 0x0F);
            const float up_q = (group & 1) ?
                static_cast<float>(up_packed >> 4) :
                static_cast<float>(up_packed & 0x0F);
            const float b = *reinterpret_cast<const float *>(src1_col + (src_base + j) * sizeof(float));
            gate_sum += (gate_d * gate_q - gate_min) * b;
            up_sum += (up_d * up_q - up_min) * b;
        }
    }

    gate_sum = pyre_reduce_256_swiglu(gate_sum, gate_sumsh);
    up_sum = pyre_reduce_256_swiglu(up_sum, up_sumsh);

    if (tid == 0) {
        const float silu_gate = gate_sum / (1.0f + __expf(-gate_sum));
        *reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst) + row * sizeof(float) + id_pos * c.dst_nb1 + token * c.dst_nb2) =
            up_sum * silu_gate;
    }
}
