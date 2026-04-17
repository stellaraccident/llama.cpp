#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_ssm_conv_update_gather_constants {
    long long d_conv;
    long long conv_state_width;
    long long d_inner;
    long long n_tokens;
    long long n_seqs;
    long long state_nb1;
    long long state_nb2;
    long long input_nb0;
    long long input_nb1;
    long long weight_nb1;
    long long dst_nb1;
    long long dst_nb2;
    long long gather_state_flat_offset;
    long long gather_src_nb1;
    float gather_scale;
    float gather_bias;
    int apply_silu;
    int pad;
};

static __device__ __forceinline__ float hrx_ssm_conv_update_gather_load(
        const float * conv_state,
        const int * state_idx,
        const float * input,
        const hrx_ssm_conv_update_gather_constants & c,
        long long channel,
        long long seq,
        long long logical_pos) {
    if (logical_pos < c.conv_state_width) {
        const int row_index = *reinterpret_cast<const int *>(state_idx);
        if (row_index < 0) {
            return 0.0f;
        }
        const long long state_byte_offset =
            channel * c.state_nb1 + seq * c.state_nb2 + logical_pos * static_cast<long long>(sizeof(float));
        const long long flat_col =
            c.gather_state_flat_offset + state_byte_offset / static_cast<long long>(sizeof(float));
        const char * state_row =
            reinterpret_cast<const char *>(conv_state) + static_cast<long long>(row_index) * c.gather_src_nb1;
        const float value = *reinterpret_cast<const float *>(
            state_row + flat_col * static_cast<long long>(sizeof(float)));
        return value * c.gather_scale + c.gather_bias;
    }

    const long long input_token = logical_pos - c.conv_state_width;
    const char * input_base = reinterpret_cast<const char *>(input) +
        input_token * c.input_nb0 + channel * c.input_nb1;
    return *reinterpret_cast<const float *>(input_base);
}

extern "C" __global__ void hrx_ssm_conv_update_gather_f32(
        const float * conv_state,
        const int * state_idx,
        const float * input,
        const float * weight,
        float * state_dst,
        float * dst,
        hrx_ssm_conv_update_gather_constants c) {
    const long long linear = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    const long long total = c.d_inner * c.n_tokens * c.n_seqs;
    if (linear >= total) {
        return;
    }

    const long long channel = linear % c.d_inner;
    const long long t1 = linear / c.d_inner;
    const long long token = t1 % c.n_tokens;
    const long long seq = t1 / c.n_tokens;

    const char * weight_base = reinterpret_cast<const char *>(weight) + channel * c.weight_nb1;

    float sum = 0.0f;
    if (c.d_conv == 4) {
        const float4 w = *reinterpret_cast<const float4 *>(weight_base);
        const float x0 = hrx_ssm_conv_update_gather_load(conv_state, state_idx, input, c, channel, seq, token + 0);
        const float x1 = hrx_ssm_conv_update_gather_load(conv_state, state_idx, input, c, channel, seq, token + 1);
        const float x2 = hrx_ssm_conv_update_gather_load(conv_state, state_idx, input, c, channel, seq, token + 2);
        const float x3 = hrx_ssm_conv_update_gather_load(conv_state, state_idx, input, c, channel, seq, token + 3);
        sum = x0 * w.x + x1 * w.y + x2 * w.z + x3 * w.w;
    } else {
        for (long long i = 0; i < c.d_conv; ++i) {
            const float x = hrx_ssm_conv_update_gather_load(conv_state, state_idx, input, c, channel, seq, token + i);
            const float w = *reinterpret_cast<const float *>(weight_base + i * sizeof(float));
            sum += x * w;
        }
    }

    if (c.apply_silu) {
        sum = sum / (1.0f + __builtin_expf(-sum));
    }

    if (token == 0) {
        for (long long i = 0; i < c.conv_state_width; ++i) {
            state_dst[channel * c.conv_state_width + i] =
                hrx_ssm_conv_update_gather_load(conv_state, state_idx, input, c, channel, seq, c.n_tokens + i);
        }
    }

    *reinterpret_cast<float *>(
        reinterpret_cast<char *>(dst) + channel * sizeof(float) + token * c.dst_nb1 + seq * c.dst_nb2) = sum;
}
