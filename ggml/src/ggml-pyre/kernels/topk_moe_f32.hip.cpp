#include <hip/hip_runtime.h>
#include <float.h>
#include <stdint.h>

struct pyre_topk_moe_f32_constants {
    long long n_experts;
    long long n_rows;
    long long n_expert_used;
    long long logits_nb1;
    long long weights_nb1;
    long long ids_nb1;
    float scale;
    float clamp_min;
    float clamp_max;
    int with_norm;
};

extern "C" __global__ void pyre_topk_moe_f32(
        const float * logits, float * weights, int * ids,
        pyre_topk_moe_f32_constants c) {
    __shared__ float values[64];
    __shared__ int indices[64];
    __shared__ float top_values[32];

    const int tid = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const long long row = static_cast<long long>(__builtin_amdgcn_workgroup_id_x());
    if (row >= c.n_rows) {
        return;
    }

    const float * logits_row = reinterpret_cast<const float *>(
        reinterpret_cast<const char *>(logits) + row * c.logits_nb1);
    char * weights_row = reinterpret_cast<char *>(weights) + row * c.weights_nb1;
    char * ids_row = reinterpret_cast<char *>(ids) + row * c.ids_nb1;

    float local[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int expert = tid + j * 64;
        local[j] = expert < c.n_experts ? logits_row[expert] * c.scale : -FLT_MAX;
    }

    float local_max = fmaxf(fmaxf(local[0], local[1]), fmaxf(local[2], local[3]));
    values[tid] = local_max;
    __builtin_amdgcn_s_barrier();

    for (int step = 32; step > 0; step >>= 1) {
        if (tid < step) {
            values[tid] = fmaxf(values[tid], values[tid + step]);
        }
        __builtin_amdgcn_s_barrier();
    }
    const float max_val = values[0];

    float local_sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int expert = tid + j * 64;
        local[j] = expert < c.n_experts ? __expf(local[j] - max_val) : 0.0f;
        local_sum += local[j];
    }
    values[tid] = local_sum;
    __builtin_amdgcn_s_barrier();

    for (int step = 32; step > 0; step >>= 1) {
        if (tid < step) {
            values[tid] += values[tid + step];
        }
        __builtin_amdgcn_s_barrier();
    }
    const float inv_sum = 1.0f / values[0];

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int expert = tid + j * 64;
        local[j] = expert < c.n_experts ? local[j] * inv_sum : -FLT_MAX;
    }

    float selected_sum = 0.0f;
    for (int k = 0; k < c.n_expert_used; ++k) {
        float thread_best = local[0];
        int thread_best_idx = tid;
        #pragma unroll
        for (int j = 1; j < 4; ++j) {
            const int expert = tid + j * 64;
            if (local[j] > thread_best || (local[j] == thread_best && expert < thread_best_idx)) {
                thread_best = local[j];
                thread_best_idx = expert;
            }
        }
        values[tid] = thread_best;
        indices[tid] = thread_best_idx;
        __builtin_amdgcn_s_barrier();

        for (int step = 32; step > 0; step >>= 1) {
            if (tid < step) {
                const float rhs_v = values[tid + step];
                const int rhs_i = indices[tid + step];
                const bool take_rhs = rhs_v > values[tid] ||
                    (rhs_v == values[tid] && rhs_i < indices[tid]);
                if (take_rhs) {
                    values[tid] = rhs_v;
                    indices[tid] = rhs_i;
                }
            }
            __builtin_amdgcn_s_barrier();
        }

        const float best_v = values[0];
        const int best_i = indices[0];
        if (tid == 0) {
            *reinterpret_cast<int *>(ids_row + k * sizeof(int)) = best_i;
            top_values[k] = best_v;
            selected_sum += best_v;
        }
        __builtin_amdgcn_s_barrier();

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (tid + j * 64 == best_i) {
                local[j] = -FLT_MAX;
            }
        }
    }

    if (tid == 0) {
        values[0] = selected_sum;
    }
    __builtin_amdgcn_s_barrier();

    float denom = values[0];
    if (c.with_norm) {
        denom = denom < c.clamp_min ? c.clamp_min : (denom > c.clamp_max ? c.clamp_max : denom);
    }
    if (tid < c.n_expert_used) {
        const float out = c.with_norm ? top_values[tid] / denom : top_values[tid];
        *reinterpret_cast<float *>(weights_row + tid * sizeof(float)) = out;
    }
}
