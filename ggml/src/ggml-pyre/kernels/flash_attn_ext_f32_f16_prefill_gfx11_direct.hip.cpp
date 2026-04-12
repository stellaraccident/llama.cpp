#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

struct pyre_flash_attn_ext_f32_f16_prefill_gfx11_direct_constants {
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

static __device__ __forceinline__ float pyre_load_f16_gfx11_direct(const __half * base, long long byte_offset) {
    return __half2float(*reinterpret_cast<const __half *>(reinterpret_cast<const char *>(base) + byte_offset));
}

static __device__ __forceinline__ float pyre_wave_reduce_max_gfx11_direct(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down(v, offset, 32));
    }
    return v;
}

static __device__ __forceinline__ float pyre_wave_reduce_sum_gfx11_direct(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset, 32);
    }
    return v;
}

typedef _Float16 pyre_gfx11_half16_vec __attribute__((ext_vector_type(16)));
typedef float pyre_gfx11_float8_vec __attribute__((ext_vector_type(8)));

static __device__ __forceinline__ uint32_t pyre_pack_f16x2_gfx11_direct(_Float16 lo, _Float16 hi) {
    union {
        _Float16 h[2];
        uint32_t u;
    } pack;
    pack.h[0] = lo;
    pack.h[1] = hi;
    return pack.u;
}

static __device__ __forceinline__ _Float16 pyre_unpack_f16x2_gfx11_direct(uint32_t bits, int idx) {
    union {
        uint32_t u;
        _Float16 h[2];
    } pack;
    pack.u = bits;
    return pack.h[idx];
}

static __device__ __forceinline__ pyre_gfx11_half16_vec pyre_duplicate_wmma_input_gfx11_direct(
        _Float16 x0, _Float16 x1, _Float16 x2, _Float16 x3,
        _Float16 x4, _Float16 x5, _Float16 x6, _Float16 x7) {
    constexpr int SWAP16_CTRL = (16 << 10) | 0x1f;
    const uint32_t p0 = pyre_pack_f16x2_gfx11_direct(x0, x1);
    const uint32_t p1 = pyre_pack_f16x2_gfx11_direct(x2, x3);
    const uint32_t p2 = pyre_pack_f16x2_gfx11_direct(x4, x5);
    const uint32_t p3 = pyre_pack_f16x2_gfx11_direct(x6, x7);
    const uint32_t s0 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p0), SWAP16_CTRL));
    const uint32_t s1 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p1), SWAP16_CTRL));
    const uint32_t s2 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p2), SWAP16_CTRL));
    const uint32_t s3 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p3), SWAP16_CTRL));

    pyre_gfx11_half16_vec result;
    result[0] = x0;
    result[1] = x1;
    result[2] = x2;
    result[3] = x3;
    result[4] = x4;
    result[5] = x5;
    result[6] = x6;
    result[7] = x7;
    result[8] = pyre_unpack_f16x2_gfx11_direct(s0, 0);
    result[9] = pyre_unpack_f16x2_gfx11_direct(s0, 1);
    result[10] = pyre_unpack_f16x2_gfx11_direct(s1, 0);
    result[11] = pyre_unpack_f16x2_gfx11_direct(s1, 1);
    result[12] = pyre_unpack_f16x2_gfx11_direct(s2, 0);
    result[13] = pyre_unpack_f16x2_gfx11_direct(s2, 1);
    result[14] = pyre_unpack_f16x2_gfx11_direct(s3, 0);
    result[15] = pyre_unpack_f16x2_gfx11_direct(s3, 1);
    return result;
}

static __device__ __forceinline__ pyre_gfx11_half16_vec pyre_duplicate_wmma_packed_pairs_gfx11_direct(
        uint32_t p0, uint32_t p1, uint32_t p2, uint32_t p3) {
    constexpr int SWAP16_CTRL = (16 << 10) | 0x1f;
    union {
        uint32_t u[8];
        pyre_gfx11_half16_vec h;
    } result;
    result.u[0] = p0;
    result.u[1] = p1;
    result.u[2] = p2;
    result.u[3] = p3;
    result.u[4] = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p0), SWAP16_CTRL));
    result.u[5] = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p1), SWAP16_CTRL));
    result.u[6] = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p2), SWAP16_CTRL));
    result.u[7] = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p3), SWAP16_CTRL));
    return result.h;
}

static __device__ __forceinline__ uint32_t pyre_load_u16_gfx11_direct(const _Float16 * ptr) {
    return static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(ptr));
}

static __device__ __forceinline__ pyre_gfx11_half16_vec pyre_load_wmma_a_row_major_gfx11_direct(
        const _Float16 * base, int ldm, unsigned int lane) {
    const int row = static_cast<int>(lane & 15);
    const int k_base = static_cast<int>(lane >> 4) * 8;
    const _Float16 * ptr = base + row * ldm + k_base;
    return pyre_duplicate_wmma_input_gfx11_direct(
        ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7]);
}

static __device__ __forceinline__ pyre_gfx11_half16_vec pyre_load_wmma_b_col_major_gfx11_direct(
        const _Float16 * base, int ldm, unsigned int lane) {
    const int col = static_cast<int>(lane & 15);
    const int k_base = static_cast<int>(lane >> 4) * 8;
    const _Float16 * ptr = base + col * ldm + k_base;
    return pyre_duplicate_wmma_input_gfx11_direct(
        ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7]);
}

static __device__ __forceinline__ pyre_gfx11_half16_vec pyre_load_wmma_b_row_major_gfx11_direct(
        const _Float16 * base, int ldm, unsigned int lane) {
    const int col = static_cast<int>(lane & 15);
    const int k_base = static_cast<int>(lane >> 4) * 8;
    const uint32_t p0 =
        pyre_load_u16_gfx11_direct(&base[(k_base + 0) * ldm + col]) |
        (pyre_load_u16_gfx11_direct(&base[(k_base + 1) * ldm + col]) << 16);
    const uint32_t p1 =
        pyre_load_u16_gfx11_direct(&base[(k_base + 2) * ldm + col]) |
        (pyre_load_u16_gfx11_direct(&base[(k_base + 3) * ldm + col]) << 16);
    const uint32_t p2 =
        pyre_load_u16_gfx11_direct(&base[(k_base + 4) * ldm + col]) |
        (pyre_load_u16_gfx11_direct(&base[(k_base + 5) * ldm + col]) << 16);
    const uint32_t p3 =
        pyre_load_u16_gfx11_direct(&base[(k_base + 6) * ldm + col]) |
        (pyre_load_u16_gfx11_direct(&base[(k_base + 7) * ldm + col]) << 16);
    return pyre_duplicate_wmma_packed_pairs_gfx11_direct(p0, p1, p2, p3);
}

static __device__ __forceinline__ pyre_gfx11_float8_vec pyre_wmma_f32_16x16x16_f16_gfx11_direct(
        pyre_gfx11_half16_vec a,
        pyre_gfx11_half16_vec b,
        pyre_gfx11_float8_vec c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

static __device__ __forceinline__ pyre_gfx11_half16_vec pyre_wmma_f16_16x16x16_f16_gfx11_direct(
        pyre_gfx11_half16_vec a,
        pyre_gfx11_half16_vec b,
        pyre_gfx11_half16_vec c) {
    return __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c, false);
}

static __device__ __forceinline__ void pyre_store_wmma_acc_row_major_gfx11_direct(
        float * dst,
        int ldm,
        pyre_gfx11_float8_vec acc,
        unsigned int lane) {
    const int row_base = static_cast<int>(lane >> 4) * 8;
    const int col = static_cast<int>(lane & 15);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        dst[(row_base + i) * ldm + col] = acc[i];
    }
}

static __device__ __forceinline__ void pyre_store_wmma_acc_row_major_f16_gfx11_direct(
        _Float16 * dst,
        int ldm,
        pyre_gfx11_half16_vec acc,
        unsigned int lane) {
    const int row_base = static_cast<int>(lane >> 4) * 8;
    const int col = static_cast<int>(lane & 15);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        dst[(row_base + i) * ldm + col] = acc[i * 2];
    }
}

extern "C" __global__ __launch_bounds__(256) void pyre_flash_attn_ext_f32_f16_prefill_gfx11_direct(
        const float * q,
        const __half * k,
        const __half * v,
        const __half * mask,
        const float * sinks,
        float * dst,
        pyre_flash_attn_ext_f32_f16_prefill_gfx11_direct_constants c) {
    constexpr int BR = 16;
    constexpr int BK = 16;
    constexpr int BC = 64;
    constexpr int WG = 256;
    __shared__ _Float16 q_tile[BR][256];
    __shared__ _Float16 p_tile[BR][BK];
    __shared__ _Float16 prob_tile[BR][BC];
    __shared__ float matrix_tile[8][BR][BK];
    __shared__ _Float16 pv_matrix_tile[8][BR][BK];
    __shared__ float row_reduce[4][4][2];

    const long long tile = __builtin_amdgcn_workgroup_id_x();
    const long long head = __builtin_amdgcn_workgroup_id_y();
    const long long seq = __builtin_amdgcn_workgroup_id_z();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int wave = tid >> 5;
    const unsigned int lane = tid & 31;
    const unsigned int row_group = wave & 3;
    const unsigned int col_half = wave >> 2;
    const long long token_base = tile * BR;

    if (head >= c.H || seq >= c.S || c.D != 256 || c.KV != 512) {
        return;
    }

    const long long kv_group = c.H / c.H_KV;
    const long long kv_head = head / kv_group;
    const char * k_head = reinterpret_cast<const char *>(k) + kv_head * c.k_nb2 + seq * c.k_nb3;
    const char * v_head = reinterpret_cast<const char *>(v) + kv_head * c.v_nb2 + seq * c.v_nb3;

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

    __half2 out_frag[4][2];
    float l_frag[4];
    float m_frag[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        l_frag[r] = 0.0f;
        m_frag[r] = -FLT_MAX * 0.5f;
#pragma unroll
        for (int d = 0; d < 2; ++d) {
            out_frag[r][d] = __float2half2_rn(0.0f);
        }
    }

    for (int jb = 0; jb < 512; jb += BC) {
        if (wave < 4) {
            const int kb = jb + static_cast<int>(wave) * BK;
            pyre_gfx11_float8_vec acc = {};

            for (int db = 0; db < 256; db += BK) {
                const _Float16 * k_block = reinterpret_cast<const _Float16 *>(
                    k_head + static_cast<long long>(kb) * c.k_nb1 + db * static_cast<long long>(sizeof(__half)));
                const pyre_gfx11_half16_vec a_frag = pyre_load_wmma_a_row_major_gfx11_direct(&q_tile[0][db], 256, lane);
                const pyre_gfx11_half16_vec b_frag =
                    pyre_load_wmma_b_col_major_gfx11_direct(k_block, static_cast<int>(c.k_nb1 / sizeof(__half)), lane);
                acc = pyre_wmma_f32_16x16x16_f16_gfx11_direct(a_frag, b_frag, acc);
            }

            pyre_store_wmma_acc_row_major_gfx11_direct(&matrix_tile[wave][0][0], BK, acc, lane);
        }
        __syncthreads();

        float score_cache[4];
        float e_m[4];
#pragma unroll
        for (int r_local = 0; r_local < 4; ++r_local) {
            const int r = static_cast<int>(row_group) * 4 + r_local;
            const long long token = token_base + r;
            const char * mask_row = reinterpret_cast<const char *>(mask) + token * c.mask_nb1 + seq * c.mask_nb3;
            const int t_inner = static_cast<int>(col_half) * 32 + static_cast<int>(lane);
            const long long t = jb + t_inner;
            float score = token < c.N ? matrix_tile[t_inner >> 4][r][t_inner & 15] : -65504.0f;
            if (c.has_mask && token < c.N) {
                const float mask_value =
                    pyre_load_f16_gfx11_direct(reinterpret_cast<const __half *>(mask_row), t * c.mask_nb0);
                if (mask_value <= -60000.0f) {
                    score = -65504.0f;
                } else {
                    score += mask_value;
                }
            }
            score_cache[r_local] = score;

            const float row_max_part = pyre_wave_reduce_max_gfx11_direct(score);
            if (lane == 0) {
                row_reduce[row_group][r_local][col_half] = row_max_part;
            }
        }
        __syncthreads();

#pragma unroll
        for (int r_local = 0; r_local < 4; ++r_local) {
            const float row_max = fmaxf(row_reduce[row_group][r_local][0], row_reduce[row_group][r_local][1]);
            const float old_m = m_frag[r_local];
            const float new_m = fmaxf(old_m, row_max);
            e_m[r_local] = __expf(old_m - new_m);
            m_frag[r_local] = new_m;
            l_frag[r_local] *= e_m[r_local];
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
                out_frag[r_local][pair] =
                    __float2half2_rn(e_m[r_local]) * out_frag[r_local][pair];
            }
        }

        __syncthreads();

#pragma unroll
        for (int r_local = 0; r_local < 4; ++r_local) {
            const int r = static_cast<int>(row_group) * 4 + r_local;
            const long long token = token_base + r;
            const int t_inner = static_cast<int>(col_half) * 32 + static_cast<int>(lane);
            float prob = 0.0f;
            if (token < c.N) {
                prob = __expf(score_cache[r_local] - m_frag[r_local]);
                l_frag[r_local] += prob;
            }
            prob_tile[r][t_inner] = static_cast<_Float16>(prob);
        }
        __syncthreads();

        for (int d_base = 0; d_base < 256; d_base += 2 * BC) {
            const int d_block = d_base + static_cast<int>(wave) * BK;
            pyre_gfx11_half16_vec acc_pv = {};

            for (int tb = 0; tb < BC; tb += BK) {
                for (int idx = tid; idx < BR * BK; idx += WG) {
                    const int r = idx >> 4;
                    const int c_inner = idx & 15;
                    p_tile[r][c_inner] = prob_tile[r][tb + c_inner];
                }
                __syncthreads();

                const _Float16 * v_block = reinterpret_cast<const _Float16 *>(
                    v_head + static_cast<long long>(jb + tb) * c.v_nb1 + d_block * static_cast<long long>(sizeof(__half)));
                const pyre_gfx11_half16_vec a_frag = pyre_load_wmma_a_row_major_gfx11_direct(&p_tile[0][0], BK, lane);
                const pyre_gfx11_half16_vec b_frag =
                    pyre_load_wmma_b_row_major_gfx11_direct(v_block, static_cast<int>(c.v_nb1 / sizeof(__half)), lane);
                acc_pv = pyre_wmma_f16_16x16x16_f16_gfx11_direct(a_frag, b_frag, acc_pv);
                __syncthreads();
            }

            pyre_store_wmma_acc_row_major_f16_gfx11_direct(&pv_matrix_tile[wave][0][0], BK, acc_pv, lane);
            __syncthreads();

#pragma unroll
            for (int r_local = 0; r_local < 4; ++r_local) {
                const int r = static_cast<int>(row_group) * 4 + r_local;
                const int d_vec = static_cast<int>(col_half) * 32 + static_cast<int>(lane);
                const int d_col = d_vec * 4;
                if (d_col >= d_base && d_col < d_base + 2 * BC) {
                    const int local_d = d_col - d_base;
#pragma unroll
                    for (int c4 = 0; c4 < 4; c4 += 2) {
                        const int local_scalar = local_d + c4;
                        const int out_idx = c4 >> 1;
                        const __half2 add_v =
                            *reinterpret_cast<const __half2 *>(&pv_matrix_tile[local_scalar >> 4][r][local_scalar & 15]);
                        out_frag[r_local][out_idx] = out_frag[r_local][out_idx] + add_v;
                    }
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int r_local = 0; r_local < 4; ++r_local) {
        const float l_part = pyre_wave_reduce_sum_gfx11_direct(l_frag[r_local]);
        if (lane == 0) {
            row_reduce[row_group][r_local][col_half] = l_part;
        }
    }
    __syncthreads();

#pragma unroll
    for (int r_local = 0; r_local < 4; ++r_local) {
        const int r = static_cast<int>(row_group) * 4 + r_local;
        const long long token = token_base + r;
        const float l_sum = row_reduce[row_group][r_local][0] + row_reduce[row_group][r_local][1];
        if (token >= c.N || l_sum == 0.0f) {
            continue;
        }
        char * dst_row = reinterpret_cast<char *>(dst) + head * c.dst_nb1 + token * c.dst_nb2 + seq * c.dst_nb3;
        const int d_vec = static_cast<int>(col_half) * 32 + static_cast<int>(lane);
        const int d = d_vec * 4;
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
            const __half2 out_v = out_frag[r_local][pair];
            *reinterpret_cast<float *>(dst_row + (d + pair * 2) * static_cast<long long>(sizeof(float))) =
                __low2float(out_v) / l_sum;
            *reinterpret_cast<float *>(dst_row + (d + pair * 2 + 1) * static_cast<long long>(sizeof(float))) =
                __high2float(out_v) / l_sum;
        }
    }
}
