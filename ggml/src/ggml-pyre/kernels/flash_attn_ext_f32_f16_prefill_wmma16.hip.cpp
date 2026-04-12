#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <float.h>
#include <math.h>
#include <stdint.h>

struct pyre_flash_attn_ext_f32_f16_prefill_wmma16_constants {
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

static __device__ __forceinline__ float pyre_load_f16_wmma16(const __half * base, long long byte_offset) {
    return __half2float(*reinterpret_cast<const __half *>(reinterpret_cast<const char *>(base) + byte_offset));
}

static __device__ __forceinline__ float pyre_wave_reduce_max_wmma16(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down(v, offset, 32));
    }
    return v;
}

static __device__ __forceinline__ float pyre_wave_reduce_sum_wmma16(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset, 32);
    }
    return v;
}

static __device__ __forceinline__ float pyre_alibi_slope_wmma16(
        const pyre_flash_attn_ext_f32_f16_prefill_wmma16_constants c,
        long long head) {
    if (c.max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = head < c.n_head_log2 ? c.m0 : c.m1;
    const int exp_h = head < c.n_head_log2 ? static_cast<int>(head + 1) :
        static_cast<int>(2 * (head - c.n_head_log2) + 1);
    return powf(base, exp_h);
}

extern "C" __global__ void pyre_flash_attn_ext_f32_f16_prefill_wmma16(
        const float * q,
        const __half * k,
        const __half * v,
        const __half * mask,
        const float * sinks,
        float * dst,
        pyre_flash_attn_ext_f32_f16_prefill_wmma16_constants c) {
    constexpr int BR = 16;
    constexpr int BK = 16;
    constexpr int BC = 64;
    constexpr int WG = 128;
    __shared__ _Float16 q_tile[BR][256];
    __shared__ _Float16 p_tile[BR][BK];
    __shared__ float matrix_tile[4][BR][BK];
    __shared__ _Float16 logits[BR][512];
    __shared__ float row_reduce[BR][4];

    const long long tile = __builtin_amdgcn_workgroup_id_x();
    const long long head = __builtin_amdgcn_workgroup_id_y();
    const long long seq = __builtin_amdgcn_workgroup_id_z();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int wave = tid >> 5;
    const long long token_base = tile * BR;

    if (head >= c.H || seq >= c.S || c.D != 256 || c.KV != 512) {
        return;
    }

    const long long kv_group = c.H / c.H_KV;
    const long long kv_head = head / kv_group;
    const char * k_head = reinterpret_cast<const char *>(k) + kv_head * c.k_nb2 + seq * c.k_nb3;
    const char * v_head = reinterpret_cast<const char *>(v) + kv_head * c.v_nb2 + seq * c.v_nb3;
    const float slope = pyre_alibi_slope_wmma16(c, head);
    const float sink = c.has_sinks ? sinks[head] : -FLT_MAX;

    for (int idx = tid; idx < BR * 256; idx += WG) {
        const int r = idx >> 8;
        const int d = idx & 255;
        const long long token = token_base + r;
        const char * q_row = reinterpret_cast<const char *>(q) + token * c.q_nb1 + head * c.q_nb2 + seq * c.q_nb3;
        const float qv = token < c.N ?
            *reinterpret_cast<const float *>(q_row + d * static_cast<long long>(sizeof(float))) : 0.0f;
        q_tile[r][d] = static_cast<_Float16>(qv * c.scale);
    }
    __syncthreads();

    for (int jb = 0; jb < 512; jb += BC) {
        const int kb = jb + static_cast<int>(wave) * BK;
        rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> acc;
        rocwmma::fill_fragment(acc, 0.0f);

        for (int db = 0; db < 256; db += BK) {
            rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, _Float16, rocwmma::row_major> a_frag;
            rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, _Float16, rocwmma::col_major> b_frag;
            const _Float16 * k_block = reinterpret_cast<const _Float16 *>(
                k_head + static_cast<long long>(kb) * c.k_nb1 + db * static_cast<long long>(sizeof(__half)));
            rocwmma::load_matrix_sync(a_frag, &q_tile[0][db], 256);
            rocwmma::load_matrix_sync(b_frag, k_block, static_cast<uint32_t>(c.k_nb1 / sizeof(__half)));
            rocwmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        rocwmma::store_matrix_sync(&matrix_tile[wave][0][0], acc, BK, rocwmma::mem_row_major);
        __syncthreads();

        for (int idx = tid; idx < BR * BC; idx += WG) {
            const int r = idx / BC;
            const int t_inner = idx - r * BC;
            const int w = t_inner >> 4;
            const long long token = token_base + r;
            const long long t = jb + t_inner;
            if (token >= c.N) {
                logits[r][t] = static_cast<_Float16>(-65504.0f);
                continue;
            }

            const char * mask_row = reinterpret_cast<const char *>(mask) + token * c.mask_nb1 + seq * c.mask_nb3;
            float score = matrix_tile[w][r][t_inner & 15];
            if (c.logit_softcap != 0.0f) {
                score = c.logit_softcap * tanhf(score);
            }
            if (c.has_mask) {
                const float mask_value =
                    pyre_load_f16_wmma16(reinterpret_cast<const __half *>(mask_row), t * c.mask_nb0);
                if (mask_value <= -60000.0f) {
                    score = -65504.0f;
                } else {
                    score += slope * mask_value;
                }
            }
            logits[r][t] = static_cast<_Float16>(score);
        }
        __syncthreads();
    }

    float local_max[BR];
#pragma unroll
    for (int r = 0; r < BR; ++r) {
        local_max[r] = sink;
    }

    for (int idx = tid; idx < BR * 512; idx += WG) {
        const int r = idx >> 9;
        const int t = idx & 511;
        if (token_base + r < c.N) {
            local_max[r] = fmaxf(local_max[r], static_cast<float>(logits[r][t]));
        }
    }

    for (int r = 0; r < BR; ++r) {
        const float wave_max = pyre_wave_reduce_max_wmma16(local_max[r]);
        if ((tid & 31) == 0) {
            row_reduce[r][wave] = wave_max;
        }
    }
    __syncthreads();

    if (tid < BR) {
        float m = row_reduce[tid][0];
#pragma unroll
        for (int w = 1; w < 4; ++w) {
            m = fmaxf(m, row_reduce[tid][w]);
        }
        row_reduce[tid][0] = m;
    }
    __syncthreads();

    float local_sum[BR];
#pragma unroll
    for (int r = 0; r < BR; ++r) {
        local_sum[r] = (c.has_sinks && tid == 0 && token_base + r < c.N) ? __expf(sink - row_reduce[r][0]) : 0.0f;
    }

    for (int idx = tid; idx < BR * 512; idx += WG) {
        const int r = idx >> 9;
        const int t = idx & 511;
        if (token_base + r < c.N) {
            const float prob = __expf(static_cast<float>(logits[r][t]) - row_reduce[r][0]);
            logits[r][t] = static_cast<_Float16>(prob);
            local_sum[r] += prob;
        }
    }

    for (int r = 0; r < BR; ++r) {
        const float wave_sum = pyre_wave_reduce_sum_wmma16(local_sum[r]);
        if ((tid & 31) == 0) {
            row_reduce[r][wave] = wave_sum;
        }
    }
    __syncthreads();

    if (tid < BR) {
        float s = row_reduce[tid][0];
#pragma unroll
        for (int w = 1; w < 4; ++w) {
            s += row_reduce[tid][w];
        }
        row_reduce[tid][0] = s;
    }
    __syncthreads();

    for (int d_base = 0; d_base < 256; d_base += BC) {
        const int d_block = d_base + static_cast<int>(wave) * BK;
        rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> acc;
        rocwmma::fill_fragment(acc, 0.0f);

        for (int tb = 0; tb < 512; tb += BK) {
            for (int idx = tid; idx < BR * BK; idx += WG) {
                const int r = idx >> 4;
                const int c_inner = idx & 15;
                const float inv_sum = 1.0f / row_reduce[r][0];
                p_tile[r][c_inner] = static_cast<_Float16>(static_cast<float>(logits[r][tb + c_inner]) * inv_sum);
            }
            __syncthreads();

            rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, _Float16, rocwmma::row_major> a_frag;
            rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, _Float16, rocwmma::row_major> b_frag;
            const _Float16 * v_block = reinterpret_cast<const _Float16 *>(
                v_head + static_cast<long long>(tb) * c.v_nb1 + d_block * static_cast<long long>(sizeof(__half)));
            rocwmma::load_matrix_sync(a_frag, &p_tile[0][0], BK);
            rocwmma::load_matrix_sync(b_frag, v_block, static_cast<uint32_t>(c.v_nb1 / sizeof(__half)));
            rocwmma::mma_sync(acc, a_frag, b_frag, acc);
            __syncthreads();
        }

        rocwmma::store_matrix_sync(&matrix_tile[wave][0][0], acc, BK, rocwmma::mem_row_major);
        __syncthreads();

        for (int idx = tid; idx < BR * BC; idx += WG) {
            const int r = idx / BC;
            const int d_inner = idx - r * BC;
            const long long token = token_base + r;
            const int d = d_base + d_inner;
            char * dst_row = reinterpret_cast<char *>(dst) + head * c.dst_nb1 + token * c.dst_nb2 + seq * c.dst_nb3;
            *reinterpret_cast<float *>(dst_row + d * static_cast<long long>(sizeof(float))) =
                matrix_tile[d_inner >> 4][r][d_inner & 15];
        }
        __syncthreads();
    }
}
