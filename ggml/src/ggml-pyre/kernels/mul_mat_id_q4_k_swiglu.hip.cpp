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

template <int WG_SIZE>
static __device__ __forceinline__ float pyre_reduce_wg_swiglu(float sum, float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (WG_SIZE <= warpSize) {
        return sum;
    }
    if (lane == 0) {
        shared[wave] = sum;
    }
    __syncthreads();

    sum = lane < ((WG_SIZE + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset);
        }
    }
    return sum;
}

template <int WG_SIZE>
static __device__ __forceinline__ void pyre_mul_mat_id_q4_k_swiglu_f32_impl(
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

    __shared__ float gate_sumsh[WG_SIZE / 32];
    __shared__ float up_sumsh[WG_SIZE / 32];
    const char * gate_row_base = reinterpret_cast<const char *>(gate) + expert * c.gate_nb2 + row * c.gate_nb1;
    const char * up_row_base = reinterpret_cast<const char *>(up) + expert * c.up_nb2 + row * c.up_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    const int block_lane = tid & 63;
    const int block_slot = tid >> 6;
    const int block_stride = WG_SIZE >> 6;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += block_stride) {
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

    gate_sum = pyre_reduce_wg_swiglu<WG_SIZE>(gate_sum, gate_sumsh);
    up_sum = pyre_reduce_wg_swiglu<WG_SIZE>(up_sum, up_sumsh);

    if (tid == 0) {
        const float silu_gate = gate_sum / (1.0f + __expf(-gate_sum));
        *reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst) + row * sizeof(float) + id_pos * c.dst_nb1 + token * c.dst_nb2) =
            up_sum * silu_gate;
    }
}

extern "C" __global__ void pyre_mul_mat_id_q4_k_swiglu_f32(
        const pyre_block_q4_K_id_swiglu * gate,
        const pyre_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        pyre_mul_mat_id_q4_k_swiglu_constants c) {
    pyre_mul_mat_id_q4_k_swiglu_f32_impl<256>(gate, up, src1, ids, dst, c);
}

extern "C" __global__ void pyre_mul_mat_id_q4_k_swiglu_wg128_f32(
        const pyre_block_q4_K_id_swiglu * gate,
        const pyre_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        pyre_mul_mat_id_q4_k_swiglu_constants c) {
    pyre_mul_mat_id_q4_k_swiglu_f32_impl<128>(gate, up, src1, ids, dst, c);
}

extern "C" __global__ void pyre_mul_mat_id_q4_k_swiglu_wg64_f32(
        const pyre_block_q4_K_id_swiglu * gate,
        const pyre_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        pyre_mul_mat_id_q4_k_swiglu_constants c) {
    pyre_mul_mat_id_q4_k_swiglu_f32_impl<64>(gate, up, src1, ids, dst, c);
}

extern "C" __global__ void pyre_mul_mat_id_q4_k_swiglu_row2_wg64_f32(
        const pyre_block_q4_K_id_swiglu * gate,
        const pyre_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        pyre_mul_mat_id_q4_k_swiglu_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 + 1 >= c.rows) {
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

    __shared__ float sumsh[64 / 32];
    const char * gate_expert_base = reinterpret_cast<const char *>(gate) + expert * c.gate_nb2;
    const char * up_expert_base = reinterpret_cast<const char *>(up) + expert * c.up_nb2;
    const char * gate_row0_base = gate_expert_base + row0 * c.gate_nb1;
    const char * gate_row1_base = gate_row0_base + c.gate_nb1;
    const char * up_row0_base = up_expert_base + row0 * c.up_nb1;
    const char * up_row1_base = up_row0_base + c.up_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;
    float gate_sum0 = 0.0f;
    float up_sum0 = 0.0f;
    float gate_sum1 = 0.0f;
    float up_sum1 = 0.0f;

    const int block_lane = tid & 63;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;

    for (long long block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const pyre_block_q4_K_id_swiglu * gate_block0 = reinterpret_cast<const pyre_block_q4_K_id_swiglu *>(
            gate_row0_base + block_idx * sizeof(pyre_block_q4_K_id_swiglu));
        const pyre_block_q4_K_id_swiglu * gate_block1 = reinterpret_cast<const pyre_block_q4_K_id_swiglu *>(
            gate_row1_base + block_idx * sizeof(pyre_block_q4_K_id_swiglu));
        const pyre_block_q4_K_id_swiglu * up_block0 = reinterpret_cast<const pyre_block_q4_K_id_swiglu *>(
            up_row0_base + block_idx * sizeof(pyre_block_q4_K_id_swiglu));
        const pyre_block_q4_K_id_swiglu * up_block1 = reinterpret_cast<const pyre_block_q4_K_id_swiglu *>(
            up_row1_base + block_idx * sizeof(pyre_block_q4_K_id_swiglu));

        uint8_t gate_sc0 = 0;
        uint8_t gate_m0 = 0;
        uint8_t gate_sc1 = 0;
        uint8_t gate_m1 = 0;
        uint8_t up_sc0 = 0;
        uint8_t up_m0 = 0;
        uint8_t up_sc1 = 0;
        uint8_t up_m1 = 0;
        pyre_get_scale_min_k4_id_swiglu(group, gate_block0->scales, &gate_sc0, &gate_m0);
        pyre_get_scale_min_k4_id_swiglu(group, gate_block1->scales, &gate_sc1, &gate_m1);
        pyre_get_scale_min_k4_id_swiglu(group, up_block0->scales, &up_sc0, &up_m0);
        pyre_get_scale_min_k4_id_swiglu(group, up_block1->scales, &up_sc1, &up_m1);

        const float gate_d0 = __half2float(__ushort_as_half(gate_block0->d)) * static_cast<float>(gate_sc0);
        const float gate_d1 = __half2float(__ushort_as_half(gate_block1->d)) * static_cast<float>(gate_sc1);
        const float up_d0 = __half2float(__ushort_as_half(up_block0->d)) * static_cast<float>(up_sc0);
        const float up_d1 = __half2float(__ushort_as_half(up_block1->d)) * static_cast<float>(up_sc1);
        const float gate_min0 = __half2float(__ushort_as_half(gate_block0->dmin)) * static_cast<float>(gate_m0);
        const float gate_min1 = __half2float(__ushort_as_half(gate_block1->dmin)) * static_cast<float>(gate_m1);
        const float up_min0 = __half2float(__ushort_as_half(up_block0->dmin)) * static_cast<float>(up_m0);
        const float up_min1 = __half2float(__ushort_as_half(up_block1->dmin)) * static_cast<float>(up_m1);
        const long long src_base = block_idx * 256 + group * 32 + lane;
        const int qs_base = (group >> 1) * 32 + lane;

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float b = *reinterpret_cast<const float *>(src1_col + (src_base + j) * sizeof(float));
            const uint8_t gate_packed0 = gate_block0->qs[qs_base + j];
            const uint8_t gate_packed1 = gate_block1->qs[qs_base + j];
            const uint8_t up_packed0 = up_block0->qs[qs_base + j];
            const uint8_t up_packed1 = up_block1->qs[qs_base + j];
            const float gate_q0 = (group & 1) ?
                static_cast<float>(gate_packed0 >> 4) :
                static_cast<float>(gate_packed0 & 0x0F);
            const float gate_q1 = (group & 1) ?
                static_cast<float>(gate_packed1 >> 4) :
                static_cast<float>(gate_packed1 & 0x0F);
            const float up_q0 = (group & 1) ?
                static_cast<float>(up_packed0 >> 4) :
                static_cast<float>(up_packed0 & 0x0F);
            const float up_q1 = (group & 1) ?
                static_cast<float>(up_packed1 >> 4) :
                static_cast<float>(up_packed1 & 0x0F);
            gate_sum0 += (gate_d0 * gate_q0 - gate_min0) * b;
            gate_sum1 += (gate_d1 * gate_q1 - gate_min1) * b;
            up_sum0 += (up_d0 * up_q0 - up_min0) * b;
            up_sum1 += (up_d1 * up_q1 - up_min1) * b;
        }
    }

    gate_sum0 = pyre_reduce_wg_swiglu<64>(gate_sum0, sumsh);
    __syncthreads();
    up_sum0 = pyre_reduce_wg_swiglu<64>(up_sum0, sumsh);
    __syncthreads();
    gate_sum1 = pyre_reduce_wg_swiglu<64>(gate_sum1, sumsh);
    __syncthreads();
    up_sum1 = pyre_reduce_wg_swiglu<64>(up_sum1, sumsh);

    if (tid == 0) {
        char * dst_base = reinterpret_cast<char *>(dst) + id_pos * c.dst_nb1 + token * c.dst_nb2;
        const float silu_gate0 = gate_sum0 / (1.0f + __expf(-gate_sum0));
        const float silu_gate1 = gate_sum1 / (1.0f + __expf(-gate_sum1));
        *reinterpret_cast<float *>(dst_base + row0 * sizeof(float)) = up_sum0 * silu_gate0;
        *reinterpret_cast<float *>(dst_base + (row0 + 1) * sizeof(float)) = up_sum1 * silu_gate1;
    }
}

extern "C" __global__ void pyre_mul_mat_id_q4_k_swiglu_packed_wg64_f32(
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

    __shared__ float gate_sumsh[2];
    __shared__ float up_sumsh[2];

    const char * gate_row_base = reinterpret_cast<const char *>(gate) + expert * c.gate_nb2 + row * c.gate_nb1;
    const char * up_row_base = reinterpret_cast<const char *>(up) + expert * c.up_nb2 + row * c.up_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    const long long blocks_per_row = c.k / 256;
    const int itid = tid & 15;
    const int block_slot = tid >> 4;
    const int il = itid >> 2;
    const int ir = itid - 4 * il;
    const int v_im = il >> 1;
    const int v_in = il & 1;
    const int l0 = 4 * (2 * ir + v_in);
    const int q_offset = 32 * v_im + l0;
    const int y_offset = 64 * v_im + l0;
    const int g0 = 2 * v_im;
    const int g1 = g0 + 1;
    const int g2 = g0 + 4;
    const int g3 = g2 + 1;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 4) {
        const pyre_block_q4_K_id_swiglu * gate_block = reinterpret_cast<const pyre_block_q4_K_id_swiglu *>(
            gate_row_base + block_idx * sizeof(pyre_block_q4_K_id_swiglu));
        const pyre_block_q4_K_id_swiglu * up_block = reinterpret_cast<const pyre_block_q4_K_id_swiglu *>(
            up_row_base + block_idx * sizeof(pyre_block_q4_K_id_swiglu));

        uint8_t gate_sc0 = 0;
        uint8_t gate_m0 = 0;
        uint8_t gate_sc1 = 0;
        uint8_t gate_m1 = 0;
        uint8_t gate_sc2 = 0;
        uint8_t gate_m2 = 0;
        uint8_t gate_sc3 = 0;
        uint8_t gate_m3 = 0;
        uint8_t up_sc0 = 0;
        uint8_t up_m0 = 0;
        uint8_t up_sc1 = 0;
        uint8_t up_m1 = 0;
        uint8_t up_sc2 = 0;
        uint8_t up_m2 = 0;
        uint8_t up_sc3 = 0;
        uint8_t up_m3 = 0;
        pyre_get_scale_min_k4_id_swiglu(g0, gate_block->scales, &gate_sc0, &gate_m0);
        pyre_get_scale_min_k4_id_swiglu(g1, gate_block->scales, &gate_sc1, &gate_m1);
        pyre_get_scale_min_k4_id_swiglu(g2, gate_block->scales, &gate_sc2, &gate_m2);
        pyre_get_scale_min_k4_id_swiglu(g3, gate_block->scales, &gate_sc3, &gate_m3);
        pyre_get_scale_min_k4_id_swiglu(g0, up_block->scales, &up_sc0, &up_m0);
        pyre_get_scale_min_k4_id_swiglu(g1, up_block->scales, &up_sc1, &up_m1);
        pyre_get_scale_min_k4_id_swiglu(g2, up_block->scales, &up_sc2, &up_m2);
        pyre_get_scale_min_k4_id_swiglu(g3, up_block->scales, &up_sc3, &up_m3);

        const float gate_d = __half2float(__ushort_as_half(gate_block->d));
        const float gate_dmin = __half2float(__ushort_as_half(gate_block->dmin));
        const float gate_d0 = gate_d * static_cast<float>(gate_sc0);
        const float gate_d1 = gate_d * static_cast<float>(gate_sc1);
        const float gate_d2 = gate_d * static_cast<float>(gate_sc2);
        const float gate_d3 = gate_d * static_cast<float>(gate_sc3);
        const float gate_min0 = gate_dmin * static_cast<float>(gate_m0);
        const float gate_min1 = gate_dmin * static_cast<float>(gate_m1);
        const float gate_min2 = gate_dmin * static_cast<float>(gate_m2);
        const float gate_min3 = gate_dmin * static_cast<float>(gate_m3);

        const float up_d = __half2float(__ushort_as_half(up_block->d));
        const float up_dmin = __half2float(__ushort_as_half(up_block->dmin));
        const float up_d0 = up_d * static_cast<float>(up_sc0);
        const float up_d1 = up_d * static_cast<float>(up_sc1);
        const float up_d2 = up_d * static_cast<float>(up_sc2);
        const float up_d3 = up_d * static_cast<float>(up_sc3);
        const float up_min0 = up_dmin * static_cast<float>(up_m0);
        const float up_min1 = up_dmin * static_cast<float>(up_m1);
        const float up_min2 = up_dmin * static_cast<float>(up_m2);
        const float up_min3 = up_dmin * static_cast<float>(up_m3);

        const long long src_base = block_idx * 256 + y_offset;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint8_t gate_q01 = gate_block->qs[q_offset + j];
            const uint8_t gate_q23 = gate_block->qs[q_offset + 64 + j];
            const uint8_t up_q01 = up_block->qs[q_offset + j];
            const uint8_t up_q23 = up_block->qs[q_offset + 64 + j];
            const float y0 = *reinterpret_cast<const float *>(src1_col + (src_base + j) * sizeof(float));
            const float y1 = *reinterpret_cast<const float *>(src1_col + (src_base + 32 + j) * sizeof(float));
            const float y2 = *reinterpret_cast<const float *>(src1_col + (src_base + 128 + j) * sizeof(float));
            const float y3 = *reinterpret_cast<const float *>(src1_col + (src_base + 160 + j) * sizeof(float));
            gate_sum += (gate_d0 * static_cast<float>(gate_q01 & 0x0F) - gate_min0) * y0;
            gate_sum += (gate_d1 * static_cast<float>(gate_q01 >> 4) - gate_min1) * y1;
            gate_sum += (gate_d2 * static_cast<float>(gate_q23 & 0x0F) - gate_min2) * y2;
            gate_sum += (gate_d3 * static_cast<float>(gate_q23 >> 4) - gate_min3) * y3;
            up_sum += (up_d0 * static_cast<float>(up_q01 & 0x0F) - up_min0) * y0;
            up_sum += (up_d1 * static_cast<float>(up_q01 >> 4) - up_min1) * y1;
            up_sum += (up_d2 * static_cast<float>(up_q23 & 0x0F) - up_min2) * y2;
            up_sum += (up_d3 * static_cast<float>(up_q23 >> 4) - up_min3) * y3;
        }
    }

    gate_sum = pyre_reduce_wg_swiglu<64>(gate_sum, gate_sumsh);
    up_sum = pyre_reduce_wg_swiglu<64>(up_sum, up_sumsh);

    if (tid == 0) {
        const float silu_gate = gate_sum / (1.0f + __expf(-gate_sum));
        *reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst) + row * sizeof(float) + id_pos * c.dst_nb1 + token * c.dst_nb2) =
            up_sum * silu_gate;
    }
}
