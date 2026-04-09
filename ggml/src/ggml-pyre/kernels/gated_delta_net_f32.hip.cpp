#include <hip/hip_runtime.h>
#include <stdint.h>

struct pyre_gated_delta_net_f32_constants {
    long long S_v;
    long long H;
    long long n_tokens;
    long long n_seqs;
    long long neq1;
    long long nek1;
    long long rq3;
    long long rk3;
    long long q_nb1;
    long long q_nb2;
    long long q_nb3;
    long long k_nb1;
    long long k_nb2;
    long long k_nb3;
    long long v_nb1;
    long long v_nb2;
    long long v_nb3;
    long long g_ne0;
    long long g_nb1;
    long long g_nb2;
    long long g_nb3;
    long long beta_nb1;
    long long beta_nb2;
    long long beta_nb3;
    long long state_dst_offset;
    float scale;
    int pad;
};

extern "C" __global__ void pyre_gated_delta_net_f32(
        const float * q,
        const float * k,
        const float * v,
        const float * g,
        const float * beta,
        const float * state_in,
        float * dst,
        float * state_dst,
        pyre_gated_delta_net_f32_constants c) {
    const long long col = __builtin_amdgcn_workgroup_id_x();
    const long long head = __builtin_amdgcn_workgroup_id_y();
    const long long seq = __builtin_amdgcn_workgroup_id_z();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();

    if (col >= c.S_v || head >= c.H || seq >= c.n_seqs) {
        return;
    }

    __shared__ float reduce[256];

    const long long iq1 = head % c.neq1;
    const long long ik1 = head % c.nek1;
    const long long iq3 = seq / c.rq3;
    const long long ik3 = seq / c.rk3;
    const bool kda = c.g_ne0 == c.S_v;

    const long long attn_score_elems = c.S_v * c.H * c.n_tokens * c.n_seqs;
    float * attn_out = dst + (seq * c.n_tokens * c.H + head) * c.S_v + col;
    (void) attn_score_elems;
    float * state_out = state_dst + c.state_dst_offset + (seq * c.H + head) * c.S_v * c.S_v + col * c.S_v;
    const float * state_col = state_in + (seq * c.H + head) * c.S_v * c.S_v + col * c.S_v;

    float s = 0.0f;
    if (tid < static_cast<unsigned int>(c.S_v)) {
        s = state_col[tid];
    }

    for (long long token = 0; token < c.n_tokens; ++token) {
        const char * q_base = reinterpret_cast<const char *>(q) + iq3 * c.q_nb3 + token * c.q_nb2 + iq1 * c.q_nb1;
        const char * k_base = reinterpret_cast<const char *>(k) + ik3 * c.k_nb3 + token * c.k_nb2 + ik1 * c.k_nb1;
        const char * v_base = reinterpret_cast<const char *>(v) + seq * c.v_nb3 + token * c.v_nb2 + head * c.v_nb1;
        const char * g_base = reinterpret_cast<const char *>(g) + seq * c.g_nb3 + token * c.g_nb2 + head * c.g_nb1;
        const char * beta_base =
            reinterpret_cast<const char *>(beta) + seq * c.beta_nb3 + token * c.beta_nb2 + head * c.beta_nb1;

        float q_i = 0.0f;
        float k_i = 0.0f;
        if (tid < static_cast<unsigned int>(c.S_v)) {
            q_i = *reinterpret_cast<const float *>(q_base + tid * sizeof(float));
            k_i = *reinterpret_cast<const float *>(k_base + tid * sizeof(float));
        }

        float kv_partial = 0.0f;
        if (tid < static_cast<unsigned int>(c.S_v)) {
            const float g_i = kda ? __builtin_expf(*reinterpret_cast<const float *>(g_base + tid * sizeof(float))) : 1.0f;
            kv_partial = g_i * s * k_i;
        }
        reduce[tid] = kv_partial;
        __builtin_amdgcn_s_barrier();

        for (unsigned int step = 128; step > 0; step >>= 1) {
            if (tid < step) {
                reduce[tid] += reduce[tid + step];
            }
            __builtin_amdgcn_s_barrier();
        }

        const float kv_col = reduce[0];
        const float beta_val = *reinterpret_cast<const float *>(beta_base);
        const float v_col = *reinterpret_cast<const float *>(v_base + col * sizeof(float));
        const float g_scalar = kda ? 1.0f : __builtin_expf(*reinterpret_cast<const float *>(g_base));
        const float delta_col = (v_col - (kda ? kv_col : g_scalar * kv_col)) * beta_val;

        float attn_partial = 0.0f;
        if (tid < static_cast<unsigned int>(c.S_v)) {
            const float g_i = kda ? __builtin_expf(*reinterpret_cast<const float *>(g_base + tid * sizeof(float))) : g_scalar;
            s = g_i * s + k_i * delta_col;
            attn_partial = s * q_i;
        }
        reduce[tid] = attn_partial;
        __builtin_amdgcn_s_barrier();

        for (unsigned int step = 128; step > 0; step >>= 1) {
            if (tid < step) {
                reduce[tid] += reduce[tid + step];
            }
            __builtin_amdgcn_s_barrier();
        }

        if (tid == 0) {
            *attn_out = reduce[0] * c.scale;
        }
        attn_out += c.S_v * c.H;
    }

    if (tid < static_cast<unsigned int>(c.S_v)) {
        state_out[tid] = s;
    }
}
