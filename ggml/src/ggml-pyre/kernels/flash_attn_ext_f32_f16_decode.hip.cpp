#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

struct pyre_flash_attn_ext_f32_f16_decode_constants {
    long long D;
    long long KV;
    long long N;
    long long H;
    long long H_KV;
    long long q_nb1;
    long long q_nb2;
    long long k_nb1;
    long long k_nb2;
    long long v_nb1;
    long long v_nb2;
    long long dst_nb1;
    long long dst_nb2;
    long long mask_nb0;
    long long mask_nb1;
    float scale;
    int has_mask;
};

static __device__ __forceinline__ float pyre_load_f16(const __half * base, long long byte_offset) {
    return __half2float(*reinterpret_cast<const __half *>(reinterpret_cast<const char *>(base) + byte_offset));
}

extern "C" __global__ void pyre_flash_attn_ext_f32_f16_decode(
        const float * q, const __half * k, const __half * v, const __half * mask, float * dst,
        pyre_flash_attn_ext_f32_f16_decode_constants c) {
    __shared__ float logits[1024];
    __shared__ double partial[256];

    const long long head = __builtin_amdgcn_workgroup_id_x();
    const long long token = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (head >= c.H || token >= c.N || c.KV > 1024) {
        return;
    }

    const long long kv_group = c.H / c.H_KV;
    const long long kv_head = head / kv_group;
    const char * q_head = reinterpret_cast<const char *>(q) + token * c.q_nb1 + head * c.q_nb2;
    const char * k_head = reinterpret_cast<const char *>(k) + kv_head * c.k_nb2;
    const char * v_head = reinterpret_cast<const char *>(v) + kv_head * c.v_nb2;
    const char * mask_row = reinterpret_cast<const char *>(mask) + token * c.mask_nb1;

    double local_max = -DBL_MAX;
    for (long long t = tid; t < c.KV; t += 256) {
        double score = 0.0;
        const char * k_row = k_head + t * c.k_nb1;
        for (long long d = 0; d < c.D; ++d) {
            const float qv = *reinterpret_cast<const float *>(q_head + d * static_cast<long long>(sizeof(float)));
            score += qv * pyre_load_f16(reinterpret_cast<const __half *>(k_row), d * static_cast<long long>(sizeof(__half)));
        }
        score *= c.scale;
        if (c.has_mask) {
            score += pyre_load_f16(reinterpret_cast<const __half *>(mask_row), t * c.mask_nb0);
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

    double local_sum = 0.0;
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

    char * dst_head = reinterpret_cast<char *>(dst) + head * c.dst_nb1 + token * c.dst_nb2;
    for (long long d = 0; d < c.D; ++d) {
        double local = 0.0;
        for (long long t = tid; t < c.KV; t += 256) {
            const char * v_row = v_head + t * c.v_nb1;
            local += logits[t] * inv_sum *
                pyre_load_f16(reinterpret_cast<const __half *>(v_row), d * static_cast<long long>(sizeof(__half)));
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
