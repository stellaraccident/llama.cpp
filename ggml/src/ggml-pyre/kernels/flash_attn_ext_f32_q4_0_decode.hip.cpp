#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

struct pyre_block_q4_0 {
    unsigned short d;
    uint8_t qs[16];
};

struct pyre_flash_attn_ext_f32_q4_0_decode_constants {
    long long D;
    long long KV;
    long long N;
    long long H;
    long long H_KV;
    long long S;
    long long q_nb1;
    long long q_nb2;
    long long q_nb3;
    long long k_nb1;
    long long k_nb2;
    long long k_nb3;
    long long v_nb1;
    long long v_nb2;
    long long v_nb3;
    long long dst_nb1;
    long long dst_nb2;
    long long dst_nb3;
    long long mask_nb0;
    long long mask_nb1;
    long long mask_nb3;
    float scale;
    int has_mask;
    float max_bias;
    float m0;
    float m1;
    float logit_softcap;
    int n_head_log2;
    int has_sinks;
};

static __device__ __forceinline__ float pyre_load_q4_0(const pyre_block_q4_0 * base, long long d) {
    const pyre_block_q4_0 * block = base + d / 32;
    const int in_block = static_cast<int>(d & 31);
    const uint8_t packed = block->qs[in_block & 15];
    const int quant = in_block < 16 ? (packed & 0x0f) - 8 : (packed >> 4) - 8;
    return __half2float(__ushort_as_half(block->d)) * static_cast<float>(quant);
}

static __device__ __forceinline__ float pyre_load_f16(const __half * base, long long byte_offset) {
    return __half2float(*reinterpret_cast<const __half *>(reinterpret_cast<const char *>(base) + byte_offset));
}

static __device__ __forceinline__ double pyre_alibi_slope(
        const pyre_flash_attn_ext_f32_q4_0_decode_constants c,
        long long head) {
    if (c.max_bias <= 0.0f) {
        return 1.0;
    }
    const double base = head < c.n_head_log2 ? c.m0 : c.m1;
    const int exp_h = head < c.n_head_log2 ? static_cast<int>(head + 1) :
        static_cast<int>(2 * (head - c.n_head_log2) + 1);
    return pow(base, exp_h);
}

extern "C" __global__ void pyre_flash_attn_ext_f32_q4_0_decode(
        const float * q, const pyre_block_q4_0 * k, const pyre_block_q4_0 * v,
        const __half * mask, const float * sinks, float * dst,
        pyre_flash_attn_ext_f32_q4_0_decode_constants c) {
    __shared__ float logits[1024];
    __shared__ double partial[256];

    const long long head = __builtin_amdgcn_workgroup_id_x();
    const long long token = __builtin_amdgcn_workgroup_id_y();
    const long long seq = __builtin_amdgcn_workgroup_id_z();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (head >= c.H || token >= c.N || seq >= c.S || c.KV > 1024) {
        return;
    }

    const long long kv_group = c.H / c.H_KV;
    const long long kv_head = head / kv_group;
    const char * q_head = reinterpret_cast<const char *>(q) + token * c.q_nb1 + head * c.q_nb2 + seq * c.q_nb3;
    const char * k_head = reinterpret_cast<const char *>(k) + kv_head * c.k_nb2 + seq * c.k_nb3;
    const char * v_head = reinterpret_cast<const char *>(v) + kv_head * c.v_nb2 + seq * c.v_nb3;
    const char * mask_row = reinterpret_cast<const char *>(mask) + token * c.mask_nb1 + seq * c.mask_nb3;
    const double slope = pyre_alibi_slope(c, head);
    const double sink = c.has_sinks ? static_cast<double>(sinks[head]) : -DBL_MAX;

    double local_max = sink;
    for (long long t = tid; t < c.KV; t += 256) {
        double score = 0.0;
        const char * k_row = k_head + t * c.k_nb1;
        for (long long d = 0; d < c.D; ++d) {
            const float qv = *reinterpret_cast<const float *>(q_head + d * static_cast<long long>(sizeof(float)));
            score += qv * pyre_load_q4_0(reinterpret_cast<const pyre_block_q4_0 *>(k_row), d);
        }
        score *= c.scale;
        if (c.logit_softcap != 0.0f) {
            score = static_cast<double>(c.logit_softcap) * tanh(score);
        }
        if (c.has_mask) {
            score += slope * pyre_load_f16(reinterpret_cast<const __half *>(mask_row), t * c.mask_nb0);
        }
        logits[t] = static_cast<float>(score);
        local_max = fmax(local_max, score);
    }

    partial[tid] = local_max;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < static_cast<unsigned int>(stride)) {
            partial[tid] = fmax(partial[tid], partial[tid + stride]);
        }
        __syncthreads();
    }
    const double max_val = partial[0];

    double local_sum = c.has_sinks && tid == 0 ? exp(sink - max_val) : 0.0;
    for (long long t = tid; t < c.KV; t += 256) {
        const double prob = exp(static_cast<double>(logits[t]) - max_val);
        logits[t] = static_cast<float>(prob);
        local_sum += prob;
    }

    partial[tid] = local_sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < static_cast<unsigned int>(stride)) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const double inv_sum = 1.0 / partial[0];

    char * dst_head = reinterpret_cast<char *>(dst) + head * c.dst_nb1 + token * c.dst_nb2 + seq * c.dst_nb3;
    for (long long d = 0; d < c.D; ++d) {
        double local = 0.0;
        for (long long t = tid; t < c.KV; t += 256) {
            const char * v_row = v_head + t * c.v_nb1;
            local += logits[t] * inv_sum * pyre_load_q4_0(reinterpret_cast<const pyre_block_q4_0 *>(v_row), d);
        }

        partial[tid] = local;
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < static_cast<unsigned int>(stride)) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            *reinterpret_cast<float *>(dst_head + d * static_cast<long long>(sizeof(float))) =
                static_cast<float>(partial[0]);
        }
        __syncthreads();
    }
}
