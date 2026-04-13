#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-cpp.h>
#include <ggml-pyre.h>
#include <ggml-quants.h>

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

static ggml_context_ptr make_context() {
    ggml_init_params params = {
        /* .mem_size   = */ 1ull << 20,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    return ggml_context_ptr(ggml_init(params));
}

static void expect_near(const std::vector<float> & actual, const std::vector<float> & expected, float tolerance, const char * label) {
    GGML_ASSERT(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        const float delta = std::fabs(actual[i] - expected[i]);
        if (!(delta <= tolerance)) {
            std::fprintf(stderr, "%s[%zu]: got %.9g expected %.9g delta %.9g\n",
                label, i, actual[i], expected[i], delta);
            std::abort();
        }
    }
}

static void expect_near_rel(
        const std::vector<float> & actual, const std::vector<float> & expected,
        float abs_tolerance, float rel_tolerance, const char * label) {
    GGML_ASSERT(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        const float delta = std::fabs(actual[i] - expected[i]);
        const float threshold = abs_tolerance + rel_tolerance * std::fabs(expected[i]);
        if (!(delta <= threshold)) {
            std::fprintf(stderr, "%s[%zu]: got %.9g expected %.9g delta %.9g threshold %.9g\n",
                label, i, actual[i], expected[i], delta, threshold);
            std::abort();
        }
    }
}

static std::vector<float> reference_rms_norm(const std::vector<float> & input, int64_t ncols, float eps) {
    GGML_ASSERT(ncols > 0);
    GGML_ASSERT((input.size() % static_cast<size_t>(ncols)) == 0);
    std::vector<float> output(input.size(), 0.0f);
    const int64_t nrows = static_cast<int64_t>(input.size()) / ncols;
    for (int64_t row = 0; row < nrows; ++row) {
        const float * src = input.data() + row * ncols;
        float * dst = output.data() + row * ncols;
        float sum = 0.0f;
        for (int64_t col = 0; col < ncols; ++col) {
            sum += src[col] * src[col];
        }
        const float scale = 1.0f / std::sqrt(sum / static_cast<float>(ncols) + eps);
        for (int64_t col = 0; col < ncols; ++col) {
            dst[col] = src[col] * scale;
        }
    }
    return output;
}

static std::vector<float> reference_add(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    GGML_ASSERT(lhs.size() == rhs.size());
    std::vector<float> output(lhs.size(), 0.0f);
    for (size_t i = 0; i < lhs.size(); ++i) {
        output[i] = lhs[i] + rhs[i];
    }
    return output;
}

static std::vector<float> reference_mul(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    GGML_ASSERT(lhs.size() == rhs.size());
    std::vector<float> output(lhs.size(), 0.0f);
    for (size_t i = 0; i < lhs.size(); ++i) {
        output[i] = lhs[i] * rhs[i];
    }
    return output;
}

static std::vector<float> reference_scale(const std::vector<float> & input, float scale, float bias) {
    std::vector<float> output(input.size(), 0.0f);
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] * scale + bias;
    }
    return output;
}

static std::vector<float> reference_ssm_conv(
        const std::vector<float> & input, const std::vector<float> & weight,
        int64_t conv_width, int64_t d_conv, int64_t d_inner, int64_t n_tokens, int64_t n_seqs) {
    GGML_ASSERT(static_cast<int64_t>(input.size()) == conv_width * d_inner * n_seqs);
    GGML_ASSERT(static_cast<int64_t>(weight.size()) == d_conv * d_inner);
    std::vector<float> output(static_cast<size_t>(d_inner * n_tokens * n_seqs), 0.0f);
    for (int64_t seq = 0; seq < n_seqs; ++seq) {
        for (int64_t token = 0; token < n_tokens; ++token) {
            for (int64_t channel = 0; channel < d_inner; ++channel) {
                float sum = 0.0f;
                for (int64_t i = 0; i < d_conv; ++i) {
                    sum += input[static_cast<size_t>(seq * conv_width * d_inner + channel * conv_width + token + i)] *
                        weight[static_cast<size_t>(channel * d_conv + i)];
                }
                output[static_cast<size_t>(seq * d_inner * n_tokens + token * d_inner + channel)] = sum;
            }
        }
    }
    return output;
}

static std::vector<float> reference_gated_delta_net(
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        const std::vector<float> & g,
        const std::vector<float> & beta,
        const std::vector<float> & state,
        int64_t S_v,
        int64_t H,
        int64_t q_heads,
        int64_t n_tokens,
        int64_t n_seqs) {
    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    std::vector<float> output(static_cast<size_t>(attn_score_elems + S_v * S_v * H * n_seqs), 0.0f);
    std::vector<float> state_out = state;
    const float scale = 1.0f / std::sqrt(static_cast<float>(S_v));

    for (int64_t seq = 0; seq < n_seqs; ++seq) {
        for (int64_t head = 0; head < H; ++head) {
            const int64_t q_head = head % q_heads;
            float * s = state_out.data() + static_cast<size_t>((seq * H + head) * S_v * S_v);
            for (int64_t token = 0; token < n_tokens; ++token) {
                const float * q_t = q.data() + static_cast<size_t>((seq * n_tokens * q_heads + token * q_heads + q_head) * S_v);
                const float * k_t = k.data() + static_cast<size_t>((seq * n_tokens * q_heads + token * q_heads + q_head) * S_v);
                const float * v_t = v.data() + static_cast<size_t>((seq * n_tokens * H + token * H + head) * S_v);
                const float g_val = std::exp(g[static_cast<size_t>(seq * n_tokens * H + token * H + head)]);
                const float beta_val = beta[static_cast<size_t>(seq * n_tokens * H + token * H + head)];

                std::vector<float> delta(static_cast<size_t>(S_v), 0.0f);
                for (int64_t col = 0; col < S_v; ++col) {
                    float kv = 0.0f;
                    for (int64_t row = 0; row < S_v; ++row) {
                        kv += s[static_cast<size_t>(col * S_v + row)] * k_t[row];
                    }
                    delta[static_cast<size_t>(col)] = (v_t[col] - g_val * kv) * beta_val;
                }

                for (int64_t col = 0; col < S_v; ++col) {
                    for (int64_t row = 0; row < S_v; ++row) {
                        s[static_cast<size_t>(col * S_v + row)] =
                            g_val * s[static_cast<size_t>(col * S_v + row)] + k_t[row] * delta[static_cast<size_t>(col)];
                    }
                }

                for (int64_t col = 0; col < S_v; ++col) {
                    float attn = 0.0f;
                    for (int64_t row = 0; row < S_v; ++row) {
                        attn += s[static_cast<size_t>(col * S_v + row)] * q_t[row];
                    }
                    output[static_cast<size_t>((seq * n_tokens * H + token * H + head) * S_v + col)] = attn * scale;
                }
            }
        }
    }

    std::copy(state_out.begin(), state_out.end(), output.begin() + attn_score_elems);
    return output;
}

static std::vector<float> reference_imrope(
        const std::vector<float> & input,
        const std::vector<int32_t> & positions,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int32_t n_dims,
        const int sections[GGML_MROPE_SECTIONS],
        float freq_base,
        float freq_scale,
        float attn_factor) {
    GGML_ASSERT(static_cast<int64_t>(input.size()) == ne0 * ne1 * ne2);
    GGML_ASSERT(static_cast<int64_t>(positions.size()) == ne2 * 4);
    std::vector<float> output = input;
    const float theta_scale = std::pow(freq_base, -2.0f / static_cast<float>(n_dims));
    const int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];

    for (int64_t i2 = 0; i2 < ne2; ++i2) {
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
            const int64_t row = i2 * ne1 + i1;
            for (int32_t i0 = 0; i0 < n_dims; i0 += 2) {
                const int sector = (i0 / 2) % sect_dims;
                const int pos_idx =
                    (sector % 3 == 1 && sector < 3 * sections[1]) ? 1 :
                    (sector % 3 == 2 && sector < 3 * sections[2]) ? 2 :
                    (sector % 3 == 0 && sector < 3 * sections[0]) ? 0 : 3;
                const float theta = static_cast<float>(positions[static_cast<size_t>(i2 + ne2 * pos_idx)]) *
                    std::pow(theta_scale, static_cast<float>(i0) / 2.0f) * freq_scale;
                const float cos_theta = std::cos(theta) * attn_factor;
                const float sin_theta = std::sin(theta) * attn_factor;
                const int64_t off0 = i0 / 2;
                const int64_t off1 = off0 + n_dims / 2;
                const float x0 = input[static_cast<size_t>(row * ne0 + off0)];
                const float x1 = input[static_cast<size_t>(row * ne0 + off1)];
                output[static_cast<size_t>(row * ne0 + off0)] = x0 * cos_theta - x1 * sin_theta;
                output[static_cast<size_t>(row * ne0 + off1)] = x0 * sin_theta + x1 * cos_theta;
            }
        }
    }
    return output;
}

static std::vector<float> reference_mul_mat(
        const std::vector<float> & lhs, const std::vector<float> & rhs,
        int64_t k, int64_t rows, int64_t cols) {
    GGML_ASSERT(static_cast<int64_t>(lhs.size()) == k * rows);
    GGML_ASSERT(static_cast<int64_t>(rhs.size()) == k * cols);
    std::vector<float> output(static_cast<size_t>(rows * cols), 0.0f);
    for (int64_t col = 0; col < cols; ++col) {
        for (int64_t row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (int64_t i = 0; i < k; ++i) {
                sum += lhs[static_cast<size_t>(row * k + i)] * rhs[static_cast<size_t>(col * k + i)];
            }
            output[static_cast<size_t>(col * rows + row)] = sum;
        }
    }
    return output;
}

static void expect_eq(const std::vector<float> & actual, const std::vector<float> & expected, const char * label) {
    GGML_ASSERT(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "%s[%zu]: got %.9g expected %.9g\n",
                label, i, actual[i], expected[i]);
            std::abort();
        }
    }
}

static void expect_eq_i32(const std::vector<int32_t> & actual, const std::vector<int32_t> & expected, const char * label) {
    GGML_ASSERT(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "%s[%zu]: got %d expected %d\n",
                label, i, actual[i], expected[i]);
            std::abort();
        }
    }
}

static void run_matvec_case(ggml_backend_t backend, ggml_backend_dev_t dev, ggml_type lhs_type, const char * label) {
    constexpr int64_t k = QK_K;
    constexpr int64_t rows = 2;
    constexpr int64_t cols = 1;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), lhs_type, k, rows);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, cols);
    ggml_tensor * dst = ggml_mul_mat(ctx.get(), lhs, rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, dst));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(rows * k);
    std::vector<float> rhs_f32(cols * k);
    std::vector<float> lhs_reference(lhs_f32.size());
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>(i % 67) - 33) / 34.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>(i % 71) - 35) / 36.0f;
    }

    if (lhs_type == GGML_TYPE_F32) {
        lhs_reference = lhs_f32;
        ggml_backend_tensor_set(lhs, lhs_f32.data(), 0, lhs_f32.size() * sizeof(float));
    } else if (lhs_type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> lhs_bf16(lhs_f32.size());
        ggml_fp32_to_bf16_row(lhs_f32.data(), lhs_bf16.data(), static_cast<int64_t>(lhs_bf16.size()));
        for (size_t i = 0; i < lhs_reference.size(); ++i) {
            lhs_reference[i] = ggml_bf16_to_fp32(lhs_bf16[i]);
        }
        ggml_backend_tensor_set(lhs, lhs_bf16.data(), 0, lhs_bf16.size() * sizeof(ggml_bf16_t));
    } else if (lhs_type == GGML_TYPE_Q5_K) {
        std::vector<block_q5_K> lhs_q5(rows);
        for (int row = 0; row < rows; ++row) {
            quantize_row_q5_K_ref(lhs_f32.data() + row * k, lhs_q5.data() + row, k);
            dequantize_row_q5_K(lhs_q5.data() + row, lhs_reference.data() + row * k, k);
        }
        ggml_backend_tensor_set(lhs, lhs_q5.data(), 0, lhs_q5.size() * sizeof(block_q5_K));
    } else if (lhs_type == GGML_TYPE_Q6_K) {
        std::vector<block_q6_K> lhs_q6(rows);
        for (int row = 0; row < rows; ++row) {
            quantize_row_q6_K_ref(lhs_f32.data() + row * k, lhs_q6.data() + row, k);
            dequantize_row_q6_K(lhs_q6.data() + row, lhs_reference.data() + row * k, k);
        }
        ggml_backend_tensor_set(lhs, lhs_q6.data(), 0, lhs_q6.size() * sizeof(block_q6_K));
    } else if (lhs_type == GGML_TYPE_Q8_0) {
        std::vector<block_q8_0> lhs_q8(rows * k / QK8_0);
        for (int row = 0; row < rows; ++row) {
            quantize_row_q8_0_ref(lhs_f32.data() + row * k, lhs_q8.data() + row * k / QK8_0, k);
            dequantize_row_q8_0(lhs_q8.data() + row * k / QK8_0, lhs_reference.data() + row * k, k);
        }
        ggml_backend_tensor_set(lhs, lhs_q8.data(), 0, lhs_q8.size() * sizeof(block_q8_0));
    } else {
        std::abort();
    }

    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output(rows * cols, -1.0f);
    ggml_backend_tensor_get(dst, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, reference_mul_mat(lhs_reference, rhs_f32, k, rows, cols), 1.0e-4f, label);
}

static void run_q6_k_prompt_matvec_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t k = QK_K;
    constexpr int64_t rows = 4;
    constexpr int64_t cols = 512;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q6_K, k, rows);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, cols);
    ggml_tensor * dst = ggml_mul_mat(ctx.get(), lhs, rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, dst));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(rows * k);
    std::vector<float> rhs_f32(cols * k);
    std::vector<float> lhs_reference(lhs_f32.size());
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>(i % 67) - 33) / 34.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>(i % 71) - 35) / 36.0f;
    }

    std::vector<block_q6_K> lhs_q6(rows);
    for (int row = 0; row < rows; ++row) {
        quantize_row_q6_K_ref(lhs_f32.data() + row * k, lhs_q6.data() + row, k);
        dequantize_row_q6_K(lhs_q6.data() + row, lhs_reference.data() + row * k, k);
    }

    ggml_backend_tensor_set(lhs, lhs_q6.data(), 0, lhs_q6.size() * sizeof(block_q6_K));
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output(rows * cols, -1.0f);
    ggml_backend_tensor_get(dst, output.data(), 0, output.size() * sizeof(float));
    expect_near(
        output, reference_mul_mat(lhs_reference, rhs_f32, k, rows, cols),
        1.0e-4f, "q6_prompt_cols512_output");
}

static void run_q6_k_decode_shape_case(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        int64_t k,
        int64_t rows,
        const char * label) {
    constexpr int64_t cols = 1;
    const int64_t blocks_per_row = k / QK_K;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q6_K, k, rows);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, cols);
    ggml_tensor * dst = ggml_mul_mat(ctx.get(), lhs, rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, dst));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(static_cast<size_t>(rows * k));
    std::vector<float> rhs_f32(static_cast<size_t>(cols * k));
    std::vector<float> lhs_reference(lhs_f32.size());
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>(i % 67) - 33) / 34.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>(i % 71) - 35) / 36.0f;
    }

    std::vector<block_q6_K> lhs_q6(static_cast<size_t>(rows * blocks_per_row));
    for (int64_t row = 0; row < rows; ++row) {
        quantize_row_q6_K_ref(
            lhs_f32.data() + row * k,
            lhs_q6.data() + row * blocks_per_row,
            k);
        dequantize_row_q6_K(
            lhs_q6.data() + row * blocks_per_row,
            lhs_reference.data() + row * k,
            k);
    }

    ggml_backend_tensor_set(lhs, lhs_q6.data(), 0, lhs_q6.size() * sizeof(block_q6_K));
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output(static_cast<size_t>(rows * cols), -1.0f);
    ggml_backend_tensor_get(dst, output.data(), 0, output.size() * sizeof(float));
    expect_near_rel(output, reference_mul_mat(lhs_reference, rhs_f32, k, rows, cols), 3.0e-4f, 2.0e-5f, label);
}

static void run_q6_k_decode_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    run_q6_k_decode_shape_case(backend, dev, 2048, 33, "q6_decode_k2048_rows33_output");
    run_q6_k_decode_shape_case(backend, dev, 4096, 17, "q6_decode_k4096_rows17_output");
}

static void run_q5_k_prompt_matvec_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t k = QK_K;
    constexpr int64_t rows = 4;
    constexpr int64_t cols = 512;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q5_K, k, rows);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, cols);
    ggml_tensor * dst = ggml_mul_mat(ctx.get(), lhs, rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, dst));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(rows * k);
    std::vector<float> rhs_f32(cols * k);
    std::vector<float> lhs_reference(lhs_f32.size());
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>(i % 67) - 33) / 34.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>(i % 71) - 35) / 36.0f;
    }

    std::vector<block_q5_K> lhs_q5(rows);
    for (int row = 0; row < rows; ++row) {
        quantize_row_q5_K_ref(lhs_f32.data() + row * k, lhs_q5.data() + row, k);
        dequantize_row_q5_K(lhs_q5.data() + row, lhs_reference.data() + row * k, k);
    }

    ggml_backend_tensor_set(lhs, lhs_q5.data(), 0, lhs_q5.size() * sizeof(block_q5_K));
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output(rows * cols, -1.0f);
    ggml_backend_tensor_get(dst, output.data(), 0, output.size() * sizeof(float));
    expect_near(
        output, reference_mul_mat(lhs_reference, rhs_f32, k, rows, cols),
        1.0e-4f, "q5_prompt_cols512_output");
}

static void run_bf16_prompt_matvec_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t k = 64;
    constexpr int64_t rows = 5;
    constexpr int64_t cols = 512;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, k, rows);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, cols);
    ggml_tensor * dst = ggml_mul_mat(ctx.get(), lhs, rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, dst));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(rows * k);
    std::vector<float> rhs_f32(cols * k);
    std::vector<float> lhs_reference(lhs_f32.size());
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>(i % 73) - 36) / 37.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>(i % 79) - 39) / 40.0f;
    }

    std::vector<ggml_bf16_t> lhs_bf16(lhs_f32.size());
    ggml_fp32_to_bf16_row(lhs_f32.data(), lhs_bf16.data(), static_cast<int64_t>(lhs_bf16.size()));
    for (size_t i = 0; i < lhs_reference.size(); ++i) {
        lhs_reference[i] = ggml_bf16_to_fp32(lhs_bf16[i]);
    }

    ggml_backend_tensor_set(lhs, lhs_bf16.data(), 0, lhs_bf16.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output(rows * cols, -1.0f);
    ggml_backend_tensor_get(dst, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, reference_mul_mat(lhs_reference, rhs_f32, k, rows, cols), 1.0e-4f, "bf16_prompt_cols512_output");
}

static void run_bf16_prompt_swiglu_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t k = 64;
    constexpr int64_t rows = 4;
    constexpr int64_t cols = 512;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * gate_lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, k, rows);
    ggml_tensor * up_lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, k, rows);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, cols);
    ggml_tensor * gate = ggml_mul_mat(ctx.get(), gate_lhs, rhs);
    ggml_tensor * up = ggml_mul_mat(ctx.get(), up_lhs, rhs);
    ggml_tensor * dst = ggml_swiglu_split(ctx.get(), gate, up);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, dst));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> gate_f32(rows * k);
    std::vector<float> up_f32(rows * k);
    std::vector<float> rhs_f32(cols * k);
    std::vector<float> gate_reference(gate_f32.size());
    std::vector<float> up_reference(up_f32.size());
    for (size_t i = 0; i < gate_f32.size(); ++i) {
        gate_f32[i] = static_cast<float>(static_cast<int>(i % 61) - 30) / 31.0f;
        up_f32[i] = static_cast<float>(static_cast<int>(i % 67) - 33) / 34.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>(i % 71) - 35) / 36.0f;
    }

    std::vector<ggml_bf16_t> gate_bf16(gate_f32.size());
    std::vector<ggml_bf16_t> up_bf16(up_f32.size());
    ggml_fp32_to_bf16_row(gate_f32.data(), gate_bf16.data(), static_cast<int64_t>(gate_bf16.size()));
    ggml_fp32_to_bf16_row(up_f32.data(), up_bf16.data(), static_cast<int64_t>(up_bf16.size()));
    for (size_t i = 0; i < gate_reference.size(); ++i) {
        gate_reference[i] = ggml_bf16_to_fp32(gate_bf16[i]);
        up_reference[i] = ggml_bf16_to_fp32(up_bf16[i]);
    }

    ggml_backend_tensor_set(gate_lhs, gate_bf16.data(), 0, gate_bf16.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(up_lhs, up_bf16.data(), 0, up_bf16.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output(rows * cols, -1.0f);
    std::vector<float> expected(output.size(), 0.0f);
    for (int64_t col = 0; col < cols; ++col) {
        for (int64_t row = 0; row < rows; ++row) {
            float gate_sum = 0.0f;
            float up_sum = 0.0f;
            for (int64_t i = 0; i < k; ++i) {
                const float rhs_value = rhs_f32[static_cast<size_t>(col * k + i)];
                gate_sum += gate_reference[static_cast<size_t>(row * k + i)] * rhs_value;
                up_sum += up_reference[static_cast<size_t>(row * k + i)] * rhs_value;
            }
            const float silu_gate = gate_sum / (1.0f + std::exp(-gate_sum));
            expected[static_cast<size_t>(col * rows + row)] = up_sum * silu_gate;
        }
    }

    ggml_backend_tensor_get(dst, output.data(), 0, output.size() * sizeof(float));
    expect_near_rel(output, expected, 1.0e-4f, 1.0e-5f, "bf16_prompt_swiglu_cols512_output");
}

static void run_bf16_decode_shape_case(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        int64_t k,
        int64_t rows,
        const char * label) {
    const int64_t cols = 1;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, k, rows);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, cols);
    ggml_tensor * dst = ggml_mul_mat(ctx.get(), lhs, rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, dst));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(rows * k);
    std::vector<float> rhs_f32(cols * k);
    std::vector<float> lhs_reference(lhs_f32.size());
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>(i % 61) - 30) / 31.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>(i % 67) - 33) / 34.0f;
    }

    std::vector<ggml_bf16_t> lhs_bf16(lhs_f32.size());
    ggml_fp32_to_bf16_row(lhs_f32.data(), lhs_bf16.data(), static_cast<int64_t>(lhs_bf16.size()));
    for (size_t i = 0; i < lhs_reference.size(); ++i) {
        lhs_reference[i] = ggml_bf16_to_fp32(lhs_bf16[i]);
    }

    ggml_backend_tensor_set(lhs, lhs_bf16.data(), 0, lhs_bf16.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output(rows * cols, -1.0f);
    ggml_backend_tensor_get(dst, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, reference_mul_mat(lhs_reference, rhs_f32, k, rows, cols), 1.0e-4f, label);
}

static void run_bf16_decode_rows2_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    run_bf16_decode_shape_case(backend, dev, 512, 3, "bf16_decode_rows2_output");
}

static void run_bf16_decode_k2048_rows32_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    run_bf16_decode_shape_case(backend, dev, 2048, 32, "bf16_decode_k2048_rows32_output");
}

static void run_bf16_decode_k2048_rows512_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    run_bf16_decode_shape_case(backend, dev, 2048, 512, "bf16_decode_k2048_rows512_output");
}

static void run_bf16_decode_k512_rows2048_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    run_bf16_decode_shape_case(backend, dev, 512, 2048, "bf16_decode_k512_rows2048_output");
}

static void run_bf16_decode_swiglu_shape_case(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        int64_t k,
        int64_t rows,
        const char * label,
        float abs_tolerance = 1.0e-4f,
        float rel_tolerance = 1.0e-5f) {
    constexpr int64_t cols = 1;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * gate_lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, k, rows);
    ggml_tensor * up_lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, k, rows);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, cols);
    ggml_tensor * gate = ggml_mul_mat(ctx.get(), gate_lhs, rhs);
    ggml_tensor * up = ggml_mul_mat(ctx.get(), up_lhs, rhs);
    ggml_tensor * dst = ggml_swiglu_split(ctx.get(), gate, up);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, dst));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> gate_f32(rows * k);
    std::vector<float> up_f32(rows * k);
    std::vector<float> rhs_f32(cols * k);
    std::vector<float> gate_reference(gate_f32.size());
    std::vector<float> up_reference(up_f32.size());
    for (size_t i = 0; i < gate_f32.size(); ++i) {
        gate_f32[i] = static_cast<float>(static_cast<int>(i % 61) - 30) / 31.0f;
        up_f32[i] = static_cast<float>(static_cast<int>(i % 67) - 33) / 34.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>(i % 71) - 35) / 36.0f;
    }

    std::vector<ggml_bf16_t> gate_bf16(gate_f32.size());
    std::vector<ggml_bf16_t> up_bf16(up_f32.size());
    ggml_fp32_to_bf16_row(gate_f32.data(), gate_bf16.data(), static_cast<int64_t>(gate_bf16.size()));
    ggml_fp32_to_bf16_row(up_f32.data(), up_bf16.data(), static_cast<int64_t>(up_bf16.size()));
    for (size_t i = 0; i < gate_reference.size(); ++i) {
        gate_reference[i] = ggml_bf16_to_fp32(gate_bf16[i]);
        up_reference[i] = ggml_bf16_to_fp32(up_bf16[i]);
    }

    ggml_backend_tensor_set(gate_lhs, gate_bf16.data(), 0, gate_bf16.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(up_lhs, up_bf16.data(), 0, up_bf16.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output(rows * cols, -1.0f);
    std::vector<float> expected(output.size(), 0.0f);
    for (int64_t row = 0; row < rows; ++row) {
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        for (int64_t i = 0; i < k; ++i) {
            const float rhs_value = rhs_f32[static_cast<size_t>(i)];
            gate_sum += gate_reference[static_cast<size_t>(row * k + i)] * rhs_value;
            up_sum += up_reference[static_cast<size_t>(row * k + i)] * rhs_value;
        }
        const float silu_gate = gate_sum / (1.0f + std::exp(-gate_sum));
        expected[static_cast<size_t>(row)] = up_sum * silu_gate;
    }

    ggml_backend_tensor_get(dst, output.data(), 0, output.size() * sizeof(float));
    expect_near_rel(output, expected, abs_tolerance, rel_tolerance, label);
}

static void run_bf16_decode_swiglu_rows2_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    run_bf16_decode_swiglu_shape_case(backend, dev, 512, 3, "bf16_decode_swiglu_rows2_output");
}

static void run_bf16_decode_swiglu_k2048_rows512_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    run_bf16_decode_swiglu_shape_case(
        backend, dev, 2048, 512, "bf16_decode_swiglu_k2048_rows512_output", 5.0e-4f, 2.0e-5f);
}

static void run_wide_matvec_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t k = 32;
    constexpr int64_t rows = 3;
    constexpr int64_t cols = 32;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, rows);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, cols);
    ggml_tensor * dst = ggml_mul_mat(ctx.get(), lhs, rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, dst));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(rows * k);
    std::vector<float> rhs_f32(cols * k);
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>(i % 19) - 9) / 10.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>(i % 23) - 11) / 12.0f;
    }

    ggml_backend_tensor_set(lhs, lhs_f32.data(), 0, lhs_f32.size() * sizeof(float));
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output(rows * cols, -1.0f);
    ggml_backend_tensor_get(dst, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, reference_mul_mat(lhs_f32, rhs_f32, k, rows, cols), 1.0e-4f, "wide_f32_output");
}

static void run_q8_0_mul_mat_add_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t k = QK8_0 * 2;
    constexpr int64_t rows = 3;
    constexpr int64_t cols = 1;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, k, rows);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, cols);
    ggml_tensor * bias = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, rows, cols);
    ggml_tensor * mm = ggml_mul_mat(ctx.get(), lhs, rhs);
    ggml_tensor * dst = ggml_add(ctx.get(), mm, bias);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, dst));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(rows * k);
    std::vector<float> lhs_dequant(lhs_f32.size());
    std::vector<float> rhs_f32(cols * k);
    std::vector<float> bias_f32(rows * cols);
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>(i % 53) - 26) / 27.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>(i % 47) - 23) / 24.0f;
    }
    for (size_t i = 0; i < bias_f32.size(); ++i) {
        bias_f32[i] = static_cast<float>(static_cast<int>(i % 11) - 5) / 6.0f;
    }

    std::vector<block_q8_0> lhs_q8(rows * k / QK8_0);
    for (int64_t row = 0; row < rows; ++row) {
        quantize_row_q8_0_ref(lhs_f32.data() + row * k, lhs_q8.data() + row * k / QK8_0, k);
        dequantize_row_q8_0(lhs_q8.data() + row * k / QK8_0, lhs_dequant.data() + row * k, k);
    }

    ggml_backend_tensor_set(lhs, lhs_q8.data(), 0, lhs_q8.size() * sizeof(block_q8_0));
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    ggml_backend_tensor_set(bias, bias_f32.data(), 0, bias_f32.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output(rows * cols, -1.0f);
    std::vector<float> expected = reference_mul_mat(lhs_dequant, rhs_f32, k, rows, cols);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] += bias_f32[i];
    }
    ggml_backend_tensor_get(dst, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, expected, 1.0e-4f, "q8_mul_mat_add_output");
}

static void run_batched_f16_matvec_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t k = 4;
    constexpr int64_t rows = 3;
    constexpr int64_t cols = 1;
    constexpr int64_t src0_batches = 2;
    constexpr int64_t dst_batches = 4;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, k, rows, src0_batches);
    ggml_tensor * rhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, cols, dst_batches);
    ggml_tensor * dst = ggml_mul_mat(ctx.get(), lhs, rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, dst));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(src0_batches * rows * k);
    std::vector<float> rhs_f32(dst_batches * cols * k);
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 9.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 7.0f;
    }
    std::vector<ggml_fp16_t> lhs_f16(lhs_f32.size());
    std::vector<float> lhs_reference(lhs_f32.size());
    ggml_fp32_to_fp16_row(lhs_f32.data(), lhs_f16.data(), static_cast<int64_t>(lhs_f16.size()));
    ggml_fp16_to_fp32_row(lhs_f16.data(), lhs_reference.data(), static_cast<int64_t>(lhs_reference.size()));

    ggml_backend_tensor_set(lhs, lhs_f16.data(), 0, lhs_f16.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output(rows * cols * dst_batches, -1.0f);
    std::vector<float> expected(output.size(), 0.0f);
    for (int64_t batch = 0; batch < dst_batches; ++batch) {
        const int64_t src0_batch = batch / (dst_batches / src0_batches);
        for (int64_t row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (int64_t i = 0; i < k; ++i) {
                sum += lhs_reference[static_cast<size_t>(src0_batch * rows * k + row * k + i)] *
                    rhs_f32[static_cast<size_t>(batch * cols * k + i)];
            }
            expected[static_cast<size_t>(batch * rows * cols + row)] = sum;
        }
    }

    ggml_backend_tensor_get(dst, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, expected, 1.0e-4f, "batched_f16_output");
}

static void run_batched_f16_prompt_matvec_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t k = 64;
    constexpr int64_t rows = 5;
    constexpr int64_t cols = 512;
    constexpr int64_t src0_batches = 2;
    constexpr int64_t dst_batches = 4;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, k, rows, src0_batches);
    ggml_tensor * rhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, cols, dst_batches);
    ggml_tensor * dst = ggml_mul_mat(ctx.get(), lhs, rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, dst));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(src0_batches * rows * k);
    std::vector<float> rhs_f32(dst_batches * cols * k);
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 15.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 16.0f;
    }
    std::vector<ggml_fp16_t> lhs_f16(lhs_f32.size());
    std::vector<float> lhs_reference(lhs_f32.size());
    ggml_fp32_to_fp16_row(lhs_f32.data(), lhs_f16.data(), static_cast<int64_t>(lhs_f16.size()));
    ggml_fp16_to_fp32_row(lhs_f16.data(), lhs_reference.data(), static_cast<int64_t>(lhs_reference.size()));

    ggml_backend_tensor_set(lhs, lhs_f16.data(), 0, lhs_f16.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output(rows * cols * dst_batches, -1.0f);
    std::vector<float> expected(output.size(), 0.0f);
    for (int64_t batch = 0; batch < dst_batches; ++batch) {
        const int64_t src0_batch = batch / (dst_batches / src0_batches);
        for (int64_t col = 0; col < cols; ++col) {
            for (int64_t row = 0; row < rows; ++row) {
                float sum = 0.0f;
                for (int64_t i = 0; i < k; ++i) {
                    sum += lhs_reference[static_cast<size_t>(src0_batch * rows * k + row * k + i)] *
                        rhs_f32[static_cast<size_t>(batch * cols * k + col * k + i)];
                }
                expected[static_cast<size_t>(batch * rows * cols + col * rows + row)] = sum;
            }
        }
    }

    ggml_backend_tensor_get(dst, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, expected, 1.0e-4f, "batched_f16_prompt_output");
}

static void run_batched_f32_matvec_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t k = 4;
    constexpr int64_t rows = 3;
    constexpr int64_t cols = 2;
    constexpr int64_t batches = 4;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, k, rows, 1, batches);
    ggml_tensor * rhs = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, k, cols, 1, batches);
    ggml_tensor * dst = ggml_mul_mat(ctx.get(), lhs, rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, dst));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(batches * rows * k);
    std::vector<float> rhs_f32(batches * cols * k);
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 9.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 7.0f;
    }

    ggml_backend_tensor_set(lhs, lhs_f32.data(), 0, lhs_f32.size() * sizeof(float));
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output(rows * cols * batches, -1.0f);
    std::vector<float> expected(output.size(), 0.0f);
    for (int64_t batch = 0; batch < batches; ++batch) {
        for (int64_t col = 0; col < cols; ++col) {
            for (int64_t row = 0; row < rows; ++row) {
                float sum = 0.0f;
                for (int64_t i = 0; i < k; ++i) {
                    sum += lhs_f32[static_cast<size_t>(batch * rows * k + row * k + i)] *
                        rhs_f32[static_cast<size_t>(batch * cols * k + col * k + i)];
                }
                expected[static_cast<size_t>(batch * rows * cols + col * rows + row)] = sum;
            }
        }
    }

    ggml_backend_tensor_get(dst, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, expected, 1.0e-4f, "batched_f32_output");
}

static void run_f32_batched_decode_shape_case(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        int64_t rows,
        const char * label) {
    constexpr int64_t k = 2048;
    constexpr int64_t cols = 1;
    constexpr int64_t batches = 3;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, k, rows, 1, batches);
    ggml_tensor * rhs = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, k, cols, 1, batches);
    ggml_tensor * dst = ggml_mul_mat(ctx.get(), lhs, rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, dst));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(static_cast<size_t>(batches * rows * k));
    std::vector<float> rhs_f32(static_cast<size_t>(batches * cols * k));
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>(i % 61) - 30) / 31.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>(i % 67) - 33) / 34.0f;
    }

    ggml_backend_tensor_set(lhs, lhs_f32.data(), 0, lhs_f32.size() * sizeof(float));
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output(static_cast<size_t>(rows * cols * batches), -1.0f);
    std::vector<float> expected(output.size(), 0.0f);
    for (int64_t batch = 0; batch < batches; ++batch) {
        for (int64_t row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (int64_t i = 0; i < k; ++i) {
                sum += lhs_f32[static_cast<size_t>(batch * rows * k + row * k + i)] *
                    rhs_f32[static_cast<size_t>(batch * cols * k + i)];
            }
            expected[static_cast<size_t>(batch * rows * cols + row)] = sum;
        }
    }

    ggml_backend_tensor_get(dst, output.data(), 0, output.size() * sizeof(float));
    expect_near_rel(output, expected, 2.0e-4f, 2.0e-5f, label);
}

static void run_f32_batched_decode_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    run_f32_batched_decode_shape_case(backend, dev, 1, "f32_batched_decode_rows1_output");
    run_f32_batched_decode_shape_case(backend, dev, 256, "f32_batched_decode_rows256_output");
}

static void run_mul_mat_id_q4_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t k = QK_K;
    constexpr int64_t rows = 4;
    constexpr int64_t experts = 3;
    constexpr int64_t ids = 2;
    constexpr int64_t tokens = 512;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_K, k, rows, experts);
    ggml_tensor * rhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, ids, tokens);
    ggml_tensor * id_tensor = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, ids, tokens);
    ggml_tensor * dst = ggml_mul_mat_id(ctx.get(), lhs, rhs, id_tensor);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, dst));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(experts * rows * k);
    std::vector<float> lhs_reference(lhs_f32.size());
    std::vector<float> rhs_f32(tokens * ids * k);
    std::vector<int32_t> expert_ids(static_cast<size_t>(ids * tokens));
    for (size_t i = 0; i < expert_ids.size(); ++i) {
        expert_ids[i] = static_cast<int32_t>((i * 5 + 1) % experts);
    }
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>(i % 67) - 33) / 34.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>(i % 71) - 35) / 36.0f;
    }

    std::vector<block_q4_K> lhs_q4(static_cast<size_t>(experts * rows));
    for (int64_t expert = 0; expert < experts; ++expert) {
        for (int64_t row = 0; row < rows; ++row) {
            const size_t row_index = static_cast<size_t>(expert * rows + row);
            quantize_row_q4_K_ref(lhs_f32.data() + row_index * k, lhs_q4.data() + row_index, k);
            dequantize_row_q4_K(lhs_q4.data() + row_index, lhs_reference.data() + row_index * k, k);
        }
    }

    ggml_backend_tensor_set(lhs, lhs_q4.data(), 0, lhs_q4.size() * sizeof(block_q4_K));
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    ggml_backend_tensor_set(id_tensor, expert_ids.data(), 0, expert_ids.size() * sizeof(int32_t));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output(rows * ids * tokens, -1.0f);
    std::vector<float> expected(output.size(), 0.0f);
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t id = 0; id < ids; ++id) {
            const int32_t expert = expert_ids[static_cast<size_t>(token * ids + id)];
            for (int64_t row = 0; row < rows; ++row) {
                float sum = 0.0f;
                for (int64_t i = 0; i < k; ++i) {
                    sum += lhs_reference[static_cast<size_t>((expert * rows + row) * k + i)] *
                        rhs_f32[static_cast<size_t>((token * ids + id) * k + i)];
                }
                expected[static_cast<size_t>(token * ids * rows + id * rows + row)] = sum;
            }
        }
    }

    ggml_backend_tensor_get(dst, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, expected, 1.0e-4f, "mul_mat_id_q4_output");
}

static void run_mul_mat_id_q4_mul_decode_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t k = QK_K * 2;
    constexpr int64_t rows = 2048;
    constexpr int64_t experts = 4;
    constexpr int64_t ids = 8;
    constexpr int64_t tokens = 1;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_K, k, rows, experts);
    ggml_tensor * rhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, ids, tokens);
    ggml_tensor * id_tensor = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, ids, tokens);
    ggml_tensor * scale = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, ids);
    ggml_tensor * mmid = ggml_mul_mat_id(ctx.get(), lhs, rhs, id_tensor);
    ggml_tensor * dst = ggml_mul(ctx.get(), mmid, scale);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, dst));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(experts * rows * k);
    std::vector<float> lhs_reference(lhs_f32.size());
    std::vector<float> rhs_f32(tokens * ids * k);
    std::vector<float> scale_f32(ids);
    std::vector<int32_t> expert_ids(static_cast<size_t>(ids * tokens));
    for (size_t i = 0; i < expert_ids.size(); ++i) {
        expert_ids[i] = static_cast<int32_t>((i * 3 + 1) % experts);
    }
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>(i % 67) - 33) / 34.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>(i % 71) - 35) / 36.0f;
    }
    for (size_t i = 0; i < scale_f32.size(); ++i) {
        scale_f32[i] = 0.25f + static_cast<float>(i) * 0.0625f;
    }

    std::vector<block_q4_K> lhs_q4(static_cast<size_t>(experts * rows * (k / QK_K)));
    for (int64_t expert = 0; expert < experts; ++expert) {
        for (int64_t row = 0; row < rows; ++row) {
            const size_t row_index = static_cast<size_t>(expert * rows + row);
            quantize_row_q4_K_ref(
                lhs_f32.data() + row_index * k,
                lhs_q4.data() + row_index * (k / QK_K),
                k);
            dequantize_row_q4_K(
                lhs_q4.data() + row_index * (k / QK_K),
                lhs_reference.data() + row_index * k,
                k);
        }
    }

    ggml_backend_tensor_set(lhs, lhs_q4.data(), 0, lhs_q4.size() * sizeof(block_q4_K));
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    ggml_backend_tensor_set(id_tensor, expert_ids.data(), 0, expert_ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(scale, scale_f32.data(), 0, scale_f32.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output(rows * ids * tokens, -1.0f);
    std::vector<float> expected(output.size(), 0.0f);
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t id = 0; id < ids; ++id) {
            const int32_t expert = expert_ids[static_cast<size_t>(token * ids + id)];
            for (int64_t row = 0; row < rows; ++row) {
                float sum = 0.0f;
                for (int64_t i = 0; i < k; ++i) {
                    sum += lhs_reference[static_cast<size_t>((expert * rows + row) * k + i)] *
                        rhs_f32[static_cast<size_t>((token * ids + id) * k + i)];
                }
                expected[static_cast<size_t>(token * ids * rows + id * rows + row)] = sum * scale_f32[id];
            }
        }
    }

    ggml_backend_tensor_get(dst, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, expected, 1.0e-4f, "mul_mat_id_q4_mul_decode_output");
}

static void run_mul_mat_id_q4_swiglu_shape_case(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        int64_t rows,
        int64_t experts,
        int64_t tokens,
        float abs_tolerance,
        float rel_tolerance,
        const char * label) {
    const int64_t k = QK_K * 8;
    const int64_t ids = 8;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * gate_lhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_K, k, rows, experts);
    ggml_tensor * up_lhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_K, k, rows, experts);
    ggml_tensor * rhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, ids, tokens);
    ggml_tensor * id_tensor = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, ids, tokens);
    ggml_tensor * gate = ggml_mul_mat_id(ctx.get(), gate_lhs, rhs, id_tensor);
    ggml_tensor * up = ggml_mul_mat_id(ctx.get(), up_lhs, rhs, id_tensor);
    ggml_tensor * dst = ggml_swiglu_split(ctx.get(), gate, up);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, dst));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> gate_f32(experts * rows * k);
    std::vector<float> up_f32(gate_f32.size());
    std::vector<float> gate_reference(gate_f32.size());
    std::vector<float> up_reference(up_f32.size());
    std::vector<float> rhs_f32(tokens * ids * k);
    std::vector<int32_t> expert_ids(static_cast<size_t>(ids * tokens));
    for (size_t i = 0; i < gate_f32.size(); ++i) {
        gate_f32[i] = static_cast<float>(static_cast<int>(i % 67) - 33) / 34.0f;
        up_f32[i] = static_cast<float>(static_cast<int>(i % 59) - 29) / 30.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>(i % 71) - 35) / 36.0f;
    }
    for (size_t i = 0; i < expert_ids.size(); ++i) {
        expert_ids[i] = static_cast<int32_t>((i * 5 + 1) % experts);
    }

    std::vector<block_q4_K> gate_q4(static_cast<size_t>(experts * rows * k / QK_K));
    std::vector<block_q4_K> up_q4(gate_q4.size());
    for (int64_t expert = 0; expert < experts; ++expert) {
        for (int64_t row = 0; row < rows; ++row) {
            const size_t row_index = static_cast<size_t>(expert * rows + row);
            quantize_row_q4_K_ref(gate_f32.data() + row_index * k, gate_q4.data() + row_index * (k / QK_K), k);
            dequantize_row_q4_K(gate_q4.data() + row_index * (k / QK_K), gate_reference.data() + row_index * k, k);
            quantize_row_q4_K_ref(up_f32.data() + row_index * k, up_q4.data() + row_index * (k / QK_K), k);
            dequantize_row_q4_K(up_q4.data() + row_index * (k / QK_K), up_reference.data() + row_index * k, k);
        }
    }

    ggml_backend_tensor_set(gate_lhs, gate_q4.data(), 0, gate_q4.size() * sizeof(block_q4_K));
    ggml_backend_tensor_set(up_lhs, up_q4.data(), 0, up_q4.size() * sizeof(block_q4_K));
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    ggml_backend_tensor_set(id_tensor, expert_ids.data(), 0, expert_ids.size() * sizeof(int32_t));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output(rows * ids * tokens, -1.0f);
    std::vector<float> expected(output.size(), 0.0f);
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t id = 0; id < ids; ++id) {
            const int32_t expert = expert_ids[static_cast<size_t>(token * ids + id)];
            for (int64_t row = 0; row < rows; ++row) {
                float gate_sum = 0.0f;
                float up_sum = 0.0f;
                for (int64_t i = 0; i < k; ++i) {
                    const float rhs_value = rhs_f32[static_cast<size_t>((token * ids + id) * k + i)];
                    gate_sum += gate_reference[static_cast<size_t>((expert * rows + row) * k + i)] * rhs_value;
                    up_sum += up_reference[static_cast<size_t>((expert * rows + row) * k + i)] * rhs_value;
                }
                const float silu_gate = gate_sum / (1.0f + std::exp(-gate_sum));
                expected[static_cast<size_t>(token * ids * rows + id * rows + row)] = up_sum * silu_gate;
            }
        }
    }

    ggml_backend_tensor_get(dst, output.data(), 0, output.size() * sizeof(float));
    expect_near_rel(output, expected, abs_tolerance, rel_tolerance, label);
}

static void run_mul_mat_id_q4_swiglu_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    run_mul_mat_id_q4_swiglu_shape_case(
        backend, dev, 2, 3, 512, 1.0e-3f, 1.0e-5f, "mul_mat_id_q4_swiglu_output");
}

static void run_mul_mat_id_q4_swiglu_decode_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    run_mul_mat_id_q4_swiglu_shape_case(
        backend, dev, 512, 4, 1, 2.0e-3f, 2.0e-5f, "mul_mat_id_q4_swiglu_decode_output");
}

static void run_mul_mat_id_q4_broadcast_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t k = QK_K;
    constexpr int64_t rows = 2;
    constexpr int64_t experts = 4;
    constexpr int64_t ids = 3;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_K, k, rows, experts);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, 1);
    ggml_tensor * id_tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, ids);
    ggml_tensor * dst = ggml_mul_mat_id(ctx.get(), lhs, rhs, id_tensor);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, dst));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(experts * rows * k);
    std::vector<float> lhs_reference(lhs_f32.size());
    std::vector<float> rhs_f32(k);
    const std::vector<int32_t> expert_ids = { 3, 1, 2 };
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>(i % 79) - 39) / 40.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>(i % 83) - 41) / 42.0f;
    }

    std::vector<block_q4_K> lhs_q4(static_cast<size_t>(experts * rows));
    for (int64_t expert = 0; expert < experts; ++expert) {
        for (int64_t row = 0; row < rows; ++row) {
            const size_t row_index = static_cast<size_t>(expert * rows + row);
            quantize_row_q4_K_ref(lhs_f32.data() + row_index * k, lhs_q4.data() + row_index, k);
            dequantize_row_q4_K(lhs_q4.data() + row_index, lhs_reference.data() + row_index * k, k);
        }
    }

    ggml_backend_tensor_set(lhs, lhs_q4.data(), 0, lhs_q4.size() * sizeof(block_q4_K));
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    ggml_backend_tensor_set(id_tensor, expert_ids.data(), 0, expert_ids.size() * sizeof(int32_t));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output(rows * ids, -1.0f);
    std::vector<float> expected(output.size(), 0.0f);
    for (int64_t id = 0; id < ids; ++id) {
        const int32_t expert = expert_ids[static_cast<size_t>(id)];
        for (int64_t row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (int64_t i = 0; i < k; ++i) {
                sum += lhs_reference[static_cast<size_t>((expert * rows + row) * k + i)] *
                    rhs_f32[static_cast<size_t>(i)];
            }
            expected[static_cast<size_t>(id * rows + row)] = sum;
        }
    }

    ggml_backend_tensor_get(dst, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, expected, 1.0e-4f, "mul_mat_id_q4_broadcast_output");
}

static void run_strided_rms_norm_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t cols = 4;
    constexpr int64_t rows = 3;
    constexpr float eps = 1.0e-6f;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * src_full = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cols, rows * 2);
    ggml_tensor * src = ggml_view_2d(ctx.get(), src_full, cols, rows, src_full->nb[1] * 2, 0);
    ggml_tensor * dst = ggml_rms_norm(ctx.get(), src, eps);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, dst));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> full(static_cast<size_t>(cols * rows * 2), 0.0f);
    std::vector<float> compact(static_cast<size_t>(cols * rows), 0.0f);
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < cols; ++col) {
            const float value = static_cast<float>(row * cols + col + 1);
            full[static_cast<size_t>((row * 2) * cols + col)] = value;
            compact[static_cast<size_t>(row * cols + col)] = value;
        }
    }

    ggml_backend_tensor_set(src_full, full.data(), 0, full.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output(static_cast<size_t>(cols * rows), -1.0f);
    ggml_backend_tensor_get(dst, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, reference_rms_norm(compact, cols, eps), 1.0e-5f, "strided_rms_norm_output");
}

static void run_broadcast_mul_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t cols = 4;
    constexpr int64_t rows = 3;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cols, rows);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cols, 1);
    ggml_tensor * dst = ggml_mul(ctx.get(), lhs, rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, dst));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_data(static_cast<size_t>(cols * rows), 0.0f);
    const std::vector<float> rhs_data = { 0.5f, 1.5f, -2.0f, 3.0f };
    std::vector<float> expected(lhs_data.size(), 0.0f);
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < cols; ++col) {
            const size_t index = static_cast<size_t>(row * cols + col);
            lhs_data[index] = static_cast<float>(index + 1);
            expected[index] = lhs_data[index] * rhs_data[static_cast<size_t>(col)];
        }
    }

    ggml_backend_tensor_set(lhs, lhs_data.data(), 0, lhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(rhs, rhs_data.data(), 0, rhs_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output(lhs_data.size(), -1.0f);
    ggml_backend_tensor_get(dst, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, expected, 1.0e-6f, "broadcast_mul_output");
}

static void run_glue_ops_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * unary_src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_tensor * silu = ggml_silu(ctx.get(), unary_src);
    ggml_tensor * sigmoid = ggml_sigmoid(ctx.get(), unary_src);
    ggml_tensor * softplus = ggml_softplus(ctx.get(), unary_src);

    ggml_tensor * gate = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 3, 2);
    ggml_tensor * up = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 3, 2);
    ggml_tensor * swiglu = ggml_swiglu_split(ctx.get(), gate, up);

    ggml_tensor * rows = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 2);
    ggml_tensor * sum_rows = ggml_sum_rows(ctx.get(), rows);
    ggml_tensor * l2_norm = ggml_l2_norm(ctx.get(), rows, 1.0e-6f);

    ggml_tensor * div_rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, 2);
    ggml_tensor * div = ggml_div(ctx.get(), rows, div_rhs);
    ggml_tensor * clamp = ggml_clamp(ctx.get(), div_rhs, 0.25f, 0.75f);

    ggml_tensor * gather_src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 2, 4);
    ggml_tensor * gather_idx = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 2);
    ggml_tensor * gather = ggml_get_rows(ctx.get(), gather_src, gather_idx);
    ggml_tensor * gather_q5_src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q5_K, QK_K, 3);
    ggml_tensor * gather_q5_idx = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 2);
    ggml_tensor * gather_q5 = ggml_get_rows(ctx.get(), gather_q5_src, gather_q5_idx);

    ggml_tensor * cat_lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 2, 2);
    ggml_tensor * cat_rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, 2);
    ggml_tensor * concat = ggml_concat(ctx.get(), cat_lhs, cat_rhs, 0);

    GGML_ASSERT(ggml_backend_dev_supports_op(dev, silu));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, sigmoid));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, softplus));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, swiglu));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, sum_rows));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, l2_norm));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, div));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, clamp));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, gather));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, gather_q5));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, concat));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, silu);
    ggml_build_forward_expand(graph, sigmoid);
    ggml_build_forward_expand(graph, softplus);
    ggml_build_forward_expand(graph, swiglu);
    ggml_build_forward_expand(graph, sum_rows);
    ggml_build_forward_expand(graph, l2_norm);
    ggml_build_forward_expand(graph, div);
    ggml_build_forward_expand(graph, clamp);
    ggml_build_forward_expand(graph, gather);
    ggml_build_forward_expand(graph, gather_q5);
    ggml_build_forward_expand(graph, concat);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const std::vector<float> unary_data = { -2.0f, -0.5f, 0.25f, 2.0f };
    const std::vector<float> gate_data = { -1.0f, 0.5f, 2.0f, -0.25f, 1.0f, 3.0f };
    const std::vector<float> up_data = { 2.0f, -1.0f, 0.5f, 4.0f, -2.0f, 1.5f };
    const std::vector<float> rows_data = { 1.0f, 2.0f, 3.0f, 4.0f, -2.0f, 1.0f, -1.0f, 2.0f };
    const std::vector<float> div_rhs_data = { 2.0f, 4.0f };
    const std::vector<float> gather_src_data = { 1.0f, 2.0f, 10.0f, 20.0f, 100.0f, 200.0f, -1.0f, -2.0f };
    const std::vector<int32_t> gather_idx_data = { 2, 1 };
    std::vector<float> gather_q5_input(3 * QK_K);
    for (size_t i = 0; i < gather_q5_input.size(); ++i) {
        gather_q5_input[i] = static_cast<float>(static_cast<int>(i % 83) - 41) / 42.0f;
    }
    std::vector<block_q5_K> gather_q5_data(3);
    std::vector<float> gather_q5_dequant(gather_q5_input.size());
    for (int row = 0; row < 3; ++row) {
        quantize_row_q5_K_ref(gather_q5_input.data() + row * QK_K, gather_q5_data.data() + row, QK_K);
        dequantize_row_q5_K(gather_q5_data.data() + row, gather_q5_dequant.data() + row * QK_K, QK_K);
    }
    const std::vector<int32_t> gather_q5_idx_data = { 2, 0 };
    const std::vector<float> cat_lhs_data = { 1.0f, 2.0f, 3.0f, 4.0f };
    const std::vector<float> cat_rhs_data = { 5.0f, 6.0f };

    ggml_backend_tensor_set(unary_src, unary_data.data(), 0, unary_data.size() * sizeof(float));
    ggml_backend_tensor_set(gate, gate_data.data(), 0, gate_data.size() * sizeof(float));
    ggml_backend_tensor_set(up, up_data.data(), 0, up_data.size() * sizeof(float));
    ggml_backend_tensor_set(rows, rows_data.data(), 0, rows_data.size() * sizeof(float));
    ggml_backend_tensor_set(div_rhs, div_rhs_data.data(), 0, div_rhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(gather_src, gather_src_data.data(), 0, gather_src_data.size() * sizeof(float));
    ggml_backend_tensor_set(gather_idx, gather_idx_data.data(), 0, gather_idx_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(gather_q5_src, gather_q5_data.data(), 0, gather_q5_data.size() * sizeof(block_q5_K));
    ggml_backend_tensor_set(gather_q5_idx, gather_q5_idx_data.data(), 0, gather_q5_idx_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(cat_lhs, cat_lhs_data.data(), 0, cat_lhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(cat_rhs, cat_rhs_data.data(), 0, cat_rhs_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    auto sigmoid_ref = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    std::vector<float> expected_silu(unary_data.size());
    std::vector<float> expected_sigmoid(unary_data.size());
    std::vector<float> expected_softplus(unary_data.size());
    for (size_t i = 0; i < unary_data.size(); ++i) {
        expected_sigmoid[i] = sigmoid_ref(unary_data[i]);
        expected_silu[i] = unary_data[i] * expected_sigmoid[i];
        expected_softplus[i] = std::log(1.0f + std::exp(unary_data[i]));
    }

    std::vector<float> expected_swiglu(gate_data.size());
    for (size_t i = 0; i < gate_data.size(); ++i) {
        expected_swiglu[i] = gate_data[i] * sigmoid_ref(gate_data[i]) * up_data[i];
    }

    const std::vector<float> expected_sum_rows = { 10.0f, 0.0f };
    std::vector<float> expected_l2_norm(rows_data.size());
    for (int row = 0; row < 2; ++row) {
        float sum = 0.0f;
        for (int col = 0; col < 4; ++col) {
            const float value = rows_data[static_cast<size_t>(row * 4 + col)];
            sum += value * value;
        }
        const float scale = 1.0f / std::max(std::sqrt(sum), 1.0e-6f);
        for (int col = 0; col < 4; ++col) {
            expected_l2_norm[static_cast<size_t>(row * 4 + col)] =
                rows_data[static_cast<size_t>(row * 4 + col)] * scale;
        }
    }
    const std::vector<float> expected_div = { 0.5f, 1.0f, 1.5f, 2.0f, -0.5f, 0.25f, -0.25f, 0.5f };
    const std::vector<float> expected_clamp = { 0.75f, 0.75f };
    const std::vector<float> expected_gather = { 100.0f, 200.0f, 10.0f, 20.0f };
    const std::vector<float> expected_concat = { 1.0f, 2.0f, 5.0f, 3.0f, 4.0f, 6.0f };
    std::vector<float> expected_gather_q5(2 * QK_K);
    std::copy(
        gather_q5_dequant.begin() + 2 * QK_K,
        gather_q5_dequant.begin() + 3 * QK_K,
        expected_gather_q5.begin());
    std::copy(
        gather_q5_dequant.begin(),
        gather_q5_dequant.begin() + QK_K,
        expected_gather_q5.begin() + QK_K);

    std::vector<float> output(expected_silu.size(), 0.0f);
    ggml_backend_tensor_get(silu, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, expected_silu, 1.0e-5f, "silu_output");
    ggml_backend_tensor_get(sigmoid, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, expected_sigmoid, 1.0e-5f, "sigmoid_output");
    ggml_backend_tensor_get(softplus, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, expected_softplus, 1.0e-5f, "softplus_output");

    output.assign(expected_swiglu.size(), 0.0f);
    ggml_backend_tensor_get(swiglu, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, expected_swiglu, 1.0e-5f, "swiglu_output");
    output.assign(expected_l2_norm.size(), 0.0f);
    ggml_backend_tensor_get(l2_norm, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, expected_l2_norm, 1.0e-5f, "l2_norm_output");
    output.assign(expected_div.size(), 0.0f);
    ggml_backend_tensor_get(div, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, expected_div, 1.0e-6f, "div_output");

    output.assign(expected_sum_rows.size(), 0.0f);
    ggml_backend_tensor_get(sum_rows, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, expected_sum_rows, 1.0e-6f, "sum_rows_output");
    ggml_backend_tensor_get(clamp, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, expected_clamp, 1.0e-6f, "clamp_output");

    output.assign(expected_gather.size(), 0.0f);
    ggml_backend_tensor_get(gather, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, expected_gather, 1.0e-6f, "get_rows_output");
    output.assign(expected_gather_q5.size(), 0.0f);
    ggml_backend_tensor_get(gather_q5, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, expected_gather_q5, 1.0e-5f, "get_rows_q5_output");

    output.assign(expected_concat.size(), 0.0f);
    ggml_backend_tensor_get(concat, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, expected_concat, 1.0e-6f, "concat_output");
}

static void run_sigmoid_mul_strided_fusion_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t cols = 4;
    constexpr int64_t heads = 2;
    constexpr int64_t rows = 3;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * attn_base = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, cols, rows, heads);
    ggml_tensor * attn_view = ggml_permute(ctx.get(), attn_base, 0, 2, 1, 3);
    ggml_tensor * attn_cont = ggml_cont(ctx.get(), attn_view);
    ggml_set_name(attn_cont, "attn_pregate-test");

    ggml_tensor * gate_base = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cols * 2, heads * rows);
    ggml_tensor * gate_view = ggml_view_3d(
        ctx.get(), gate_base, cols, heads, rows, gate_base->nb[1], gate_base->nb[1] * heads, cols * sizeof(float));
    ggml_tensor * gate_cont = ggml_cont(ctx.get(), gate_view);
    ggml_set_name(gate_cont, "gate_reshaped-test");

    ggml_tensor * sigmoid = ggml_sigmoid(ctx.get(), gate_cont);
    ggml_tensor * mul = ggml_mul(ctx.get(), attn_cont, sigmoid);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, mul));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, mul);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> attn_data(static_cast<size_t>(cols * rows * heads), 0.0f);
    for (size_t i = 0; i < attn_data.size(); ++i) {
        attn_data[i] = static_cast<float>(static_cast<int>(i) - 11) / 7.0f;
    }
    std::vector<float> gate_data(static_cast<size_t>(cols * 2 * heads * rows), 0.0f);
    for (size_t i = 0; i < gate_data.size(); ++i) {
        gate_data[i] = static_cast<float>(static_cast<int>(i % 19) - 9) / 5.0f;
    }

    ggml_backend_tensor_set(attn_base, attn_data.data(), 0, attn_data.size() * sizeof(float));
    ggml_backend_tensor_set(gate_base, gate_data.data(), 0, gate_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    auto sigmoid_ref = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    std::vector<float> expected(static_cast<size_t>(cols * heads * rows), 0.0f);
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t head = 0; head < heads; ++head) {
            for (int64_t col = 0; col < cols; ++col) {
                const size_t dst_idx = static_cast<size_t>((row * heads + head) * cols + col);
                const size_t attn_idx = static_cast<size_t>((head * rows + row) * cols + col);
                const size_t gate_idx = static_cast<size_t>((row * heads + head) * (cols * 2) + cols + col);
                expected[dst_idx] = attn_data[attn_idx] * sigmoid_ref(gate_data[gate_idx]);
            }
        }
    }

    std::vector<float> output(expected.size(), -1.0f);
    ggml_backend_tensor_get(mul, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, expected, 1.0e-5f, "sigmoid_mul_strided_fusion_output");
}

static void run_singleton_stride_concat_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t rows = 5;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 2, rows);
    ggml_tensor * rhs_base = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, rows, 1);
    ggml_tensor * rhs = ggml_transpose(ctx.get(), rhs_base);
    ggml_tensor * concat = ggml_concat(ctx.get(), lhs, rhs, 0);
    GGML_ASSERT(rhs->ne[0] == 1);
    GGML_ASSERT(rhs->nb[0] != sizeof(float));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, concat));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, concat);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const std::vector<float> lhs_data = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
        9.0f, 10.0f,
    };
    const std::vector<float> rhs_data = { 20.0f, 21.0f, 22.0f, 23.0f, 24.0f };
    const std::vector<float> expected = {
        1.0f, 2.0f, 20.0f,
        3.0f, 4.0f, 21.0f,
        5.0f, 6.0f, 22.0f,
        7.0f, 8.0f, 23.0f,
        9.0f, 10.0f, 24.0f,
    };

    ggml_backend_tensor_set(lhs, lhs_data.data(), 0, lhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(rhs_base, rhs_data.data(), 0, rhs_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output(expected.size(), -1.0f);
    ggml_backend_tensor_get(concat, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, expected, 1.0e-6f, "singleton_stride_concat_output");
}

static void run_router_ops_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t cols = 4;
    constexpr int64_t rows = 4;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * logits = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, cols, 2, 2);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cols, 2);
    ggml_tensor * probs = ggml_soft_max_ext(ctx.get(), logits, mask, 0.5f, 0.0f);
    ggml_tensor * sorted = ggml_argsort(ctx.get(), probs, GGML_SORT_ORDER_DESC);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, probs));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, sorted));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, probs);
    ggml_build_forward_expand(graph, sorted);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const std::vector<float> logits_data = {
        1.0f, 3.0f, -1.0f, 0.5f,
        0.0f, 2.0f, 4.0f, -2.0f,
        2.0f, -1.0f, 1.5f, 0.0f,
        -3.0f, 1.0f, 0.25f, 2.5f,
    };
    const std::vector<float> mask_data = {
        0.0f, -1.0f, 0.5f, 0.0f,
        0.25f, 0.0f, -0.5f, 1.0f,
    };
    std::vector<float> expected_probs(logits_data.size(), 0.0f);
    std::vector<int32_t> expected_sorted(logits_data.size(), 0);
    for (int64_t row = 0; row < rows; ++row) {
        const int64_t mask_row = row % 2;
        float max_value = -INFINITY;
        for (int64_t col = 0; col < cols; ++col) {
            const float value = logits_data[static_cast<size_t>(row * cols + col)] * 0.5f +
                mask_data[static_cast<size_t>(mask_row * cols + col)];
            expected_probs[static_cast<size_t>(row * cols + col)] = value;
            max_value = std::max(max_value, value);
        }
        float sum = 0.0f;
        for (int64_t col = 0; col < cols; ++col) {
            float & value = expected_probs[static_cast<size_t>(row * cols + col)];
            value = std::exp(value - max_value);
            sum += value;
        }
        for (int64_t col = 0; col < cols; ++col) {
            expected_probs[static_cast<size_t>(row * cols + col)] /= sum;
            expected_sorted[static_cast<size_t>(row * cols + col)] = static_cast<int32_t>(col);
        }
        std::sort(
            expected_sorted.begin() + row * cols,
            expected_sorted.begin() + (row + 1) * cols,
            [&](int32_t lhs, int32_t rhs) {
                return expected_probs[static_cast<size_t>(row * cols + lhs)] >
                    expected_probs[static_cast<size_t>(row * cols + rhs)];
            });
    }

    ggml_backend_tensor_set(logits, logits_data.data(), 0, logits_data.size() * sizeof(float));
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> probs_output(expected_probs.size(), -1.0f);
    ggml_backend_tensor_get(probs, probs_output.data(), 0, probs_output.size() * sizeof(float));
    expect_near(probs_output, expected_probs, 1.0e-6f, "router_soft_max_output");

    std::vector<int32_t> sorted_output(expected_sorted.size(), -1);
    ggml_backend_tensor_get(sorted, sorted_output.data(), 0, sorted_output.size() * sizeof(int32_t));
    expect_eq_i32(sorted_output, expected_sorted, "router_argsort_output");
}

static void run_large_masked_soft_max_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t cols = 2048;
    constexpr int64_t rows1 = 2;
    constexpr int64_t rows2 = 2;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * logits = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, cols, rows1, rows2);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cols, rows1);
    ggml_tensor * probs = ggml_soft_max_ext(ctx.get(), logits, mask, 0.25f, 0.0f);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, probs));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, probs);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> logits_data(static_cast<size_t>(cols * rows1 * rows2), 0.0f);
    for (size_t i = 0; i < logits_data.size(); ++i) {
        logits_data[i] = static_cast<float>(static_cast<int>(i % 211) - 105) / 17.0f;
    }
    std::vector<float> mask_data(static_cast<size_t>(cols * rows1), 0.0f);
    for (size_t i = 0; i < mask_data.size(); ++i) {
        mask_data[i] = static_cast<float>(static_cast<int>(i % 43) - 21) / 23.0f;
    }

    ggml_backend_tensor_set(logits, logits_data.data(), 0, logits_data.size() * sizeof(float));
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected(logits_data.size(), 0.0f);
    for (int64_t row = 0; row < rows1 * rows2; ++row) {
        float max_value = -INFINITY;
        for (int64_t col = 0; col < cols; ++col) {
            const size_t index = static_cast<size_t>(row * cols + col);
            const size_t mask_index = static_cast<size_t>((row % rows1) * cols + col);
            const float value = logits_data[index] * 0.25f + mask_data[mask_index];
            expected[index] = value;
            max_value = std::max(max_value, value);
        }
        float sum = 0.0f;
        for (int64_t col = 0; col < cols; ++col) {
            float & value = expected[static_cast<size_t>(row * cols + col)];
            value = std::exp(value - max_value);
            sum += value;
        }
        for (int64_t col = 0; col < cols; ++col) {
            expected[static_cast<size_t>(row * cols + col)] /= sum;
        }
    }

    std::vector<float> output(expected.size(), -1.0f);
    ggml_backend_tensor_get(probs, output.data(), 0, output.size() * sizeof(float));
    expect_near(output, expected, 1.0e-5f, "large_masked_soft_max_output");
}

static void run_imrope_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t ne0 = 8;
    constexpr int64_t ne1 = 2;
    constexpr int64_t ne2 = 1;
    constexpr int32_t n_dims = 6;
    int sections[GGML_MROPE_SECTIONS] = { 1, 1, 1, 0 };

    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, ne0, ne1, ne2);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, ne2 * 4);
    ggml_tensor * rope = ggml_rope_multi(
        ctx.get(), src, pos, nullptr, n_dims, sections, GGML_ROPE_TYPE_IMROPE,
        0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, rope));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, rope);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const std::vector<float> input = {
        0.25f, -0.5f, 0.75f, -1.0f, 1.25f, -1.5f, 1.75f, -2.0f,
        2.25f, -2.5f, 2.75f, -3.0f, 3.25f, -3.5f, 3.75f, -4.0f,
    };
    const std::vector<int32_t> positions = { 2, 3, 5, 7 };
    ggml_backend_tensor_set(src, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(pos, positions.data(), 0, positions.size() * sizeof(int32_t));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output(input.size(), -1.0f);
    ggml_backend_tensor_get(rope, output.data(), 0, output.size() * sizeof(float));
    expect_near(
        output,
        reference_imrope(input, positions, ne0, ne1, ne2, n_dims, sections, 10000.0f, 1.0f, 1.0f),
        1.0e-5f,
        "imrope_output");
}

static void run_gated_delta_net_shape_case(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        int64_t S_v,
        int64_t H,
        int64_t q_heads,
        int64_t n_tokens,
        int64_t n_seqs,
        const char * label) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S_v, q_heads, n_tokens, n_seqs);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S_v, q_heads, n_tokens, n_seqs);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S_v, H, n_tokens, n_seqs);
    ggml_tensor * g = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, H, n_tokens, n_seqs);
    ggml_tensor * beta = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, H, n_tokens, n_seqs);
    ggml_tensor * state = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, S_v * S_v, H, n_seqs);
    ggml_tensor * dst = ggml_gated_delta_net(ctx.get(), q, k, v, g, beta, state);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, dst));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> q_data(static_cast<size_t>(S_v * q_heads * n_tokens * n_seqs));
    std::vector<float> k_data(q_data.size());
    std::vector<float> v_data(static_cast<size_t>(S_v * H * n_tokens * n_seqs));
    std::vector<float> g_data(static_cast<size_t>(H * n_tokens * n_seqs));
    std::vector<float> beta_data(g_data.size());
    std::vector<float> state_data(static_cast<size_t>(S_v * S_v * H * n_seqs));
    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = 0.05f * static_cast<float>(static_cast<int>(i % 17) - 8);
        k_data[i] = 0.04f * static_cast<float>(static_cast<int>(i % 5) - 2);
    }
    for (size_t i = 0; i < v_data.size(); ++i) {
        v_data[i] = 0.03f * static_cast<float>(static_cast<int>(i % 11) - 5);
    }
    for (size_t i = 0; i < g_data.size(); ++i) {
        g_data[i] = -0.35f + 0.2f * static_cast<float>(i % 3);
        beta_data[i] = 0.25f + 0.05f * static_cast<float>(i % 4);
    }
    for (size_t i = 0; i < state_data.size(); ++i) {
        state_data[i] = 0.01f * static_cast<float>(static_cast<int>(i % 13) - 6);
    }

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_data.data(), 0, k_data.size() * sizeof(float));
    ggml_backend_tensor_set(v, v_data.data(), 0, v_data.size() * sizeof(float));
    ggml_backend_tensor_set(g, g_data.data(), 0, g_data.size() * sizeof(float));
    ggml_backend_tensor_set(beta, beta_data.data(), 0, beta_data.size() * sizeof(float));
    ggml_backend_tensor_set(state, state_data.data(), 0, state_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> output(static_cast<size_t>(ggml_nelements(dst)), -1.0f);
    ggml_backend_tensor_get(dst, output.data(), 0, output.size() * sizeof(float));
    expect_near(
        output,
        reference_gated_delta_net(q_data, k_data, v_data, g_data, beta_data, state_data, S_v, H, q_heads, n_tokens, n_seqs),
        1.0e-4f,
        label);
}

static void run_gated_delta_net_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    run_gated_delta_net_shape_case(
        backend,
        dev,
        /* S_v = */ 4,
        /* H = */ 2,
        /* q_heads = */ 1,
        /* n_tokens = */ 2,
        /* n_seqs = */ 1,
        "gated_delta_net_output");
}

static void run_gated_delta_net_s128_decode_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    run_gated_delta_net_shape_case(
        backend,
        dev,
        /* S_v = */ 128,
        /* H = */ 4,
        /* q_heads = */ 1,
        /* n_tokens = */ 1,
        /* n_seqs = */ 1,
        "gated_delta_net_s128_decode_output");
}

static void run_gated_delta_net_s128_decode_gqa_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    run_gated_delta_net_shape_case(
        backend,
        dev,
        /* S_v = */ 128,
        /* H = */ 32,
        /* q_heads = */ 16,
        /* n_tokens = */ 1,
        /* n_seqs = */ 1,
        "gated_delta_net_s128_decode_gqa_output");
}

} // namespace

int main() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("PYRE0");
    if (!dev) {
        std::fprintf(stderr, "PYRE0 not available; skipping test-backend-pyre\n");
        return 0;
    }

    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    GGML_ASSERT(backend != nullptr);

    const char * test_filter = std::getenv("GGML_PYRE_TEST_FILTER");
    if (test_filter != nullptr && std::strcmp(test_filter, "bf16_decode") == 0) {
        run_bf16_decode_rows2_case(backend.get(), dev);
        run_bf16_decode_k2048_rows32_case(backend.get(), dev);
        run_bf16_decode_k2048_rows512_case(backend.get(), dev);
        run_bf16_decode_k512_rows2048_case(backend.get(), dev);
        run_bf16_decode_swiglu_rows2_case(backend.get(), dev);
        run_bf16_decode_swiglu_k2048_rows512_case(backend.get(), dev);
        return 0;
    }
    if (test_filter != nullptr && std::strcmp(test_filter, "f32_batched_decode") == 0) {
        run_f32_batched_decode_case(backend.get(), dev);
        return 0;
    }
    if (test_filter != nullptr && std::strcmp(test_filter, "q6_decode") == 0) {
        run_q6_k_decode_case(backend.get(), dev);
        return 0;
    }
    if (test_filter != nullptr && std::strcmp(test_filter, "q4_id_mul_decode") == 0) {
        run_mul_mat_id_q4_mul_decode_case(backend.get(), dev);
        return 0;
    }
    if (test_filter != nullptr && std::strcmp(test_filter, "q4_id_swiglu_decode") == 0) {
        run_mul_mat_id_q4_swiglu_decode_case(backend.get(), dev);
        return 0;
    }
    if (test_filter != nullptr && std::strcmp(test_filter, "gated_delta_net") == 0) {
        run_gated_delta_net_case(backend.get(), dev);
        run_gated_delta_net_s128_decode_case(backend.get(), dev);
        run_gated_delta_net_s128_decode_gqa_case(backend.get(), dev);
        return 0;
    }

    ggml_context_ptr ctx = make_context();
    ggml_tensor * src  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 8);
    ggml_tensor * view = ggml_view_1d(ctx.get(), src, 4, 2 * sizeof(float));
    ggml_tensor * dst  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(src->buffer == buffer.get());
    GGML_ASSERT(view->buffer == buffer.get());
    GGML_ASSERT(dst->buffer == buffer.get());

    const std::vector<float> input = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f };
    ggml_backend_tensor_set(src, input.data(), 0, input.size() * sizeof(float));

    std::vector<float> view_data(4, -1.0f);
    ggml_backend_tensor_get(view, view_data.data(), 0, view_data.size() * sizeof(float));
    expect_eq(view_data, { 2.0f, 3.0f, 4.0f, 5.0f }, "view_data");

    const std::vector<float> replacement = { 20.0f, 21.0f, 22.0f, 23.0f };
    ggml_backend_tensor_set(view, replacement.data(), 0, replacement.size() * sizeof(float));

    std::vector<float> src_after_view_set(8, -1.0f);
    ggml_backend_tensor_get(src, src_after_view_set.data(), 0, src_after_view_set.size() * sizeof(float));
    expect_eq(src_after_view_set, { 0.0f, 1.0f, 20.0f, 21.0f, 22.0f, 23.0f, 6.0f, 7.0f }, "src_after_view_set");

    ggml_backend_tensor_copy(view, dst);
    std::vector<float> dst_data(4, -1.0f);
    ggml_backend_tensor_get(dst, dst_data.data(), 0, dst_data.size() * sizeof(float));
    expect_eq(dst_data, replacement, "dst_data");

    ggml_backend_tensor_memset(view, 0, 0, ggml_nbytes(view));
    std::vector<float> src_after_memset(8, -1.0f);
    ggml_backend_tensor_get(src, src_after_memset.data(), 0, src_after_memset.size() * sizeof(float));
    expect_eq(src_after_memset, { 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 6.0f, 7.0f }, "src_after_memset");

    ggml_backend_buffer_clear(buffer.get(), 0);
    std::vector<float> src_after_clear(8, -1.0f);
    ggml_backend_tensor_get(src, src_after_clear.data(), 0, src_after_clear.size() * sizeof(float));
    expect_eq(src_after_clear, std::vector<float>(8, 0.0f), "src_after_clear");

    ggml_context_ptr graph_ctx = make_context();
    ggml_tensor * rms_src = ggml_new_tensor_2d(graph_ctx.get(), GGML_TYPE_F32, 4, 2);
    ggml_tensor * rms_dst = ggml_rms_norm(graph_ctx.get(), rms_src, 1.0e-6f);
    ggml_tensor * add_dst = ggml_add(graph_ctx.get(), rms_src, rms_dst);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, rms_dst));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, add_dst));

    ggml_cgraph * graph = ggml_new_graph(graph_ctx.get());
    ggml_build_forward_expand(graph, rms_dst);

    ggml_backend_buffer_ptr graph_buffer(ggml_backend_alloc_ctx_tensors(graph_ctx.get(), backend.get()));
    GGML_ASSERT(graph_buffer != nullptr);

    const std::vector<float> rms_input = {
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 4.0f, 6.0f, 8.0f,
    };
    ggml_backend_tensor_set(rms_src, rms_input.data(), 0, rms_input.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);

    std::vector<float> rms_output(rms_input.size(), -1.0f);
    ggml_backend_tensor_get(rms_dst, rms_output.data(), 0, rms_output.size() * sizeof(float));
    expect_near(rms_output, reference_rms_norm(rms_input, 4, 1.0e-6f), 1.0e-5f, "rms_output");

    ggml_context_ptr elem_ctx = make_context();
    ggml_tensor * elem_lhs = ggml_new_tensor_2d(elem_ctx.get(), GGML_TYPE_F32, 4, 2);
    ggml_tensor * elem_rhs = ggml_new_tensor_2d(elem_ctx.get(), GGML_TYPE_F32, 4, 2);
    ggml_tensor * elem_add = ggml_add(elem_ctx.get(), elem_lhs, elem_rhs);
    ggml_tensor * elem_mul = ggml_mul(elem_ctx.get(), elem_lhs, elem_rhs);
    ggml_tensor * elem_scale = ggml_scale_bias(elem_ctx.get(), elem_lhs, 1.5f, -0.25f);
    ggml_tensor * elem_cpy_target = ggml_new_tensor_2d(elem_ctx.get(), GGML_TYPE_F32, 4, 2);
    ggml_tensor * elem_cpy = ggml_cpy(elem_ctx.get(), elem_lhs, elem_cpy_target);
    ggml_tensor * ssm_input = ggml_new_tensor_3d(elem_ctx.get(), GGML_TYPE_F32, 4, 3, 2);
    ggml_tensor * ssm_weight = ggml_new_tensor_2d(elem_ctx.get(), GGML_TYPE_F32, 2, 3);
    ggml_tensor * ssm_dst = ggml_ssm_conv(elem_ctx.get(), ssm_input, ssm_weight);
    ggml_tensor * row_base = ggml_new_tensor_2d(elem_ctx.get(), GGML_TYPE_F32, 6, 2);
    ggml_tensor * row_view = ggml_view_2d(elem_ctx.get(), row_base, 4, 2, row_base->nb[1], 2 * sizeof(float));
    ggml_tensor * row_cpy_target = ggml_new_tensor_1d(elem_ctx.get(), GGML_TYPE_F32, 8);
    ggml_tensor * row_cpy = ggml_cpy(elem_ctx.get(), row_view, row_cpy_target);
    ggml_tensor * row_cont = ggml_cont_1d(elem_ctx.get(), row_view, 8);
    ggml_tensor * set_target_f32 = ggml_new_tensor_2d(elem_ctx.get(), GGML_TYPE_F32, 4, 4);
    ggml_tensor * set_target_f16 = ggml_new_tensor_2d(elem_ctx.get(), GGML_TYPE_F16, 4, 4);
    ggml_tensor * set_src = ggml_new_tensor_2d(elem_ctx.get(), GGML_TYPE_F32, 4, 2);
    ggml_tensor * set_idxs = ggml_new_tensor_1d(elem_ctx.get(), GGML_TYPE_I64, 2);
    ggml_tensor * set_rows_f32 = ggml_set_rows(elem_ctx.get(), set_target_f32, set_src, set_idxs);
    ggml_tensor * set_rows_f16 = ggml_set_rows(elem_ctx.get(), set_target_f16, set_src, set_idxs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, elem_add));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, elem_mul));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, elem_scale));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, elem_cpy));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, ssm_dst));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, row_cpy));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, row_cont));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, set_rows_f32));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, set_rows_f16));

    ggml_cgraph * elem_graph = ggml_new_graph(elem_ctx.get());
    ggml_build_forward_expand(elem_graph, elem_add);
    ggml_build_forward_expand(elem_graph, elem_mul);
    ggml_build_forward_expand(elem_graph, elem_scale);
    ggml_build_forward_expand(elem_graph, elem_cpy);
    ggml_build_forward_expand(elem_graph, ssm_dst);
    ggml_build_forward_expand(elem_graph, row_cpy);
    ggml_build_forward_expand(elem_graph, row_cont);
    ggml_build_forward_expand(elem_graph, set_rows_f32);
    ggml_build_forward_expand(elem_graph, set_rows_f16);

    ggml_backend_buffer_ptr elem_buffer(ggml_backend_alloc_ctx_tensors(elem_ctx.get(), backend.get()));
    GGML_ASSERT(elem_buffer != nullptr);

    const std::vector<float> elem_input_lhs = {
        1.0f, -2.0f, 3.0f, -4.0f,
        5.0f, -6.0f, 7.0f, -8.0f,
    };
    const std::vector<float> elem_input_rhs = {
        0.5f, 2.0f, -1.0f, -3.0f,
        4.0f, -0.5f, 0.25f, -0.125f,
    };
    ggml_backend_tensor_set(elem_lhs, elem_input_lhs.data(), 0, elem_input_lhs.size() * sizeof(float));
    ggml_backend_tensor_set(elem_rhs, elem_input_rhs.data(), 0, elem_input_rhs.size() * sizeof(float));
    const std::vector<float> row_base_input = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
    };
    const std::vector<float> ssm_input_data = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f, 20.0f,
        21.0f, 22.0f, 23.0f, 24.0f,
    };
    const std::vector<float> ssm_weight_data = {
        0.5f, -1.0f,
        1.5f, 0.25f,
        -0.75f, 2.0f,
    };
    ggml_backend_tensor_set(ssm_input, ssm_input_data.data(), 0, ssm_input_data.size() * sizeof(float));
    ggml_backend_tensor_set(ssm_weight, ssm_weight_data.data(), 0, ssm_weight_data.size() * sizeof(float));
    ggml_backend_tensor_set(row_base, row_base_input.data(), 0, row_base_input.size() * sizeof(float));
    const std::vector<float> set_target_input(16, -1.0f);
    const std::vector<float> set_src_input = {
        10.0f, 11.0f, 12.0f, 13.0f,
        20.0f, 21.0f, 22.0f, 23.0f,
    };
    const std::vector<int64_t> set_idx_input = { 3, 1 };
    std::vector<ggml_fp16_t> set_target_input_f16(set_target_input.size());
    ggml_fp32_to_fp16_row(set_target_input.data(), set_target_input_f16.data(), static_cast<int64_t>(set_target_input_f16.size()));
    ggml_backend_tensor_set(set_target_f32, set_target_input.data(), 0, set_target_input.size() * sizeof(float));
    ggml_backend_tensor_set(set_target_f16, set_target_input_f16.data(), 0, set_target_input_f16.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(set_src, set_src_input.data(), 0, set_src_input.size() * sizeof(float));
    ggml_backend_tensor_set(set_idxs, set_idx_input.data(), 0, set_idx_input.size() * sizeof(int64_t));
    GGML_ASSERT(ggml_backend_graph_compute(backend.get(), elem_graph) == GGML_STATUS_SUCCESS);

    std::vector<float> elem_add_output(elem_input_lhs.size(), -1.0f);
    std::vector<float> elem_mul_output(elem_input_lhs.size(), -1.0f);
    std::vector<float> elem_scale_output(elem_input_lhs.size(), -1.0f);
    std::vector<float> elem_cpy_output(elem_input_lhs.size(), -1.0f);
    std::vector<float> ssm_output(18, -1.0f);
    std::vector<float> row_cpy_output(8, -1.0f);
    std::vector<float> row_cont_output(8, -1.0f);
    std::vector<float> set_rows_f32_output(16, -2.0f);
    std::vector<ggml_fp16_t> set_rows_f16_output_raw(16);
    std::vector<float> set_rows_f16_output(16, -2.0f);
    ggml_backend_tensor_get(elem_add, elem_add_output.data(), 0, elem_add_output.size() * sizeof(float));
    ggml_backend_tensor_get(elem_mul, elem_mul_output.data(), 0, elem_mul_output.size() * sizeof(float));
    ggml_backend_tensor_get(elem_scale, elem_scale_output.data(), 0, elem_scale_output.size() * sizeof(float));
    ggml_backend_tensor_get(elem_cpy, elem_cpy_output.data(), 0, elem_cpy_output.size() * sizeof(float));
    ggml_backend_tensor_get(ssm_dst, ssm_output.data(), 0, ssm_output.size() * sizeof(float));
    ggml_backend_tensor_get(row_cpy, row_cpy_output.data(), 0, row_cpy_output.size() * sizeof(float));
    ggml_backend_tensor_get(row_cont, row_cont_output.data(), 0, row_cont_output.size() * sizeof(float));
    ggml_backend_tensor_get(set_rows_f32, set_rows_f32_output.data(), 0, set_rows_f32_output.size() * sizeof(float));
    ggml_backend_tensor_get(set_rows_f16, set_rows_f16_output_raw.data(), 0, set_rows_f16_output_raw.size() * sizeof(ggml_fp16_t));
    ggml_fp16_to_fp32_row(set_rows_f16_output_raw.data(), set_rows_f16_output.data(), static_cast<int64_t>(set_rows_f16_output.size()));
    expect_near(elem_add_output, reference_add(elem_input_lhs, elem_input_rhs), 1.0e-6f, "elem_add_output");
    expect_near(elem_mul_output, reference_mul(elem_input_lhs, elem_input_rhs), 1.0e-6f, "elem_mul_output");
    expect_near(elem_scale_output, reference_scale(elem_input_lhs, 1.5f, -0.25f), 1.0e-6f, "elem_scale_output");
    expect_near(elem_cpy_output, elem_input_lhs, 1.0e-6f, "elem_cpy_output");
    expect_near(ssm_output, reference_ssm_conv(ssm_input_data, ssm_weight_data, 4, 2, 3, 3, 2), 1.0e-6f, "ssm_output");
    expect_near(row_cpy_output, { 2.0f, 3.0f, 4.0f, 5.0f, 8.0f, 9.0f, 10.0f, 11.0f }, 1.0e-6f, "row_cpy_output");
    expect_near(row_cont_output, { 2.0f, 3.0f, 4.0f, 5.0f, 8.0f, 9.0f, 10.0f, 11.0f }, 1.0e-6f, "row_cont_output");
    const std::vector<float> set_rows_expected = {
        -1.0f, -1.0f, -1.0f, -1.0f,
        20.0f, 21.0f, 22.0f, 23.0f,
        -1.0f, -1.0f, -1.0f, -1.0f,
        10.0f, 11.0f, 12.0f, 13.0f,
    };
    expect_near(set_rows_f32_output, set_rows_expected, 1.0e-6f, "set_rows_f32_output");
    expect_near(set_rows_f16_output, set_rows_expected, 1.0e-3f, "set_rows_f16_output");

    ggml_context_ptr mat_ctx = make_context();
    ggml_tensor * mat_lhs = ggml_new_tensor_2d(mat_ctx.get(), GGML_TYPE_F16, 4, 3);
    ggml_tensor * mat_rhs = ggml_new_tensor_2d(mat_ctx.get(), GGML_TYPE_F32, 4, 2);
    ggml_tensor * mat_dst = ggml_mul_mat(mat_ctx.get(), mat_lhs, mat_rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, mat_dst));

    ggml_cgraph * mat_graph = ggml_new_graph(mat_ctx.get());
    ggml_build_forward_expand(mat_graph, mat_dst);

    ggml_backend_buffer_ptr mat_buffer(ggml_backend_alloc_ctx_tensors(mat_ctx.get(), backend.get()));
    GGML_ASSERT(mat_buffer != nullptr);

    const std::vector<float> mat_lhs_f32 = {
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 3.0f, 4.0f, 5.0f,
        3.0f, 4.0f, 5.0f, 6.0f,
    };
    const std::vector<float> mat_rhs_f32 = {
        1.0f, 0.5f, 0.25f, 0.125f,
        -1.0f, 0.25f, -0.5f, 2.0f,
    };
    std::vector<ggml_fp16_t> mat_lhs_f16(mat_lhs_f32.size());
    ggml_fp32_to_fp16_row(mat_lhs_f32.data(), mat_lhs_f16.data(), static_cast<int64_t>(mat_lhs_f16.size()));
    ggml_backend_tensor_set(mat_lhs, mat_lhs_f16.data(), 0, mat_lhs_f16.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(mat_rhs, mat_rhs_f32.data(), 0, mat_rhs_f32.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend.get(), mat_graph) == GGML_STATUS_SUCCESS);

    std::vector<float> mat_output(6, -1.0f);
    ggml_backend_tensor_get(mat_dst, mat_output.data(), 0, mat_output.size() * sizeof(float));
    expect_near(mat_output, reference_mul_mat(mat_lhs_f32, mat_rhs_f32, 4, 3, 2), 1.0e-5f, "mat_output");

    ggml_context_ptr q4_ctx = make_context();
    ggml_tensor * q4_lhs = ggml_new_tensor_2d(q4_ctx.get(), GGML_TYPE_Q4_K, QK_K, 2);
    ggml_tensor * q4_rhs = ggml_new_tensor_2d(q4_ctx.get(), GGML_TYPE_F32, QK_K, 1);
    ggml_tensor * q4_dst = ggml_mul_mat(q4_ctx.get(), q4_lhs, q4_rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, q4_dst));

    ggml_cgraph * q4_graph = ggml_new_graph(q4_ctx.get());
    ggml_build_forward_expand(q4_graph, q4_dst);

    ggml_backend_buffer_ptr q4_buffer(ggml_backend_alloc_ctx_tensors(q4_ctx.get(), backend.get()));
    GGML_ASSERT(q4_buffer != nullptr);

    std::vector<float> q4_lhs_f32(2 * QK_K);
    std::vector<float> q4_rhs_f32(QK_K);
    for (size_t i = 0; i < q4_lhs_f32.size(); ++i) {
        q4_lhs_f32[i] = static_cast<float>(static_cast<int>(i % 67) - 33) / 34.0f;
    }
    for (size_t i = 0; i < q4_rhs_f32.size(); ++i) {
        q4_rhs_f32[i] = static_cast<float>(static_cast<int>(i % 71) - 35) / 36.0f;
    }

    std::vector<block_q4_K> q4_lhs_data(2);
    std::vector<float> q4_lhs_dequant(q4_lhs_f32.size());
    for (int row = 0; row < 2; ++row) {
        quantize_row_q4_K_ref(q4_lhs_f32.data() + row * QK_K, q4_lhs_data.data() + row, QK_K);
        dequantize_row_q4_K(q4_lhs_data.data() + row, q4_lhs_dequant.data() + row * QK_K, QK_K);
    }

    ggml_backend_tensor_set(q4_lhs, q4_lhs_data.data(), 0, q4_lhs_data.size() * sizeof(block_q4_K));
    ggml_backend_tensor_set(q4_rhs, q4_rhs_f32.data(), 0, q4_rhs_f32.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend.get(), q4_graph) == GGML_STATUS_SUCCESS);

    std::vector<float> q4_output(2, -1.0f);
    ggml_backend_tensor_get(q4_dst, q4_output.data(), 0, q4_output.size() * sizeof(float));
    expect_near(q4_output, reference_mul_mat(q4_lhs_dequant, q4_rhs_f32, QK_K, 2, 1), 1.0e-4f, "q4_output");

    run_matvec_case(backend.get(), dev, GGML_TYPE_F32, "f32_output");
    run_matvec_case(backend.get(), dev, GGML_TYPE_BF16, "bf16_output");
    run_bf16_prompt_matvec_case(backend.get(), dev);
    run_bf16_prompt_swiglu_case(backend.get(), dev);
    run_matvec_case(backend.get(), dev, GGML_TYPE_Q5_K, "q5_output");
    run_q5_k_prompt_matvec_case(backend.get(), dev);
    run_matvec_case(backend.get(), dev, GGML_TYPE_Q6_K, "q6_output");
    run_q6_k_prompt_matvec_case(backend.get(), dev);
    run_matvec_case(backend.get(), dev, GGML_TYPE_Q8_0, "q8_output");
    run_wide_matvec_case(backend.get(), dev);
    run_q8_0_mul_mat_add_case(backend.get(), dev);
    run_batched_f16_matvec_case(backend.get(), dev);
    run_batched_f16_prompt_matvec_case(backend.get(), dev);
    run_batched_f32_matvec_case(backend.get(), dev);
    run_mul_mat_id_q4_case(backend.get(), dev);
    run_mul_mat_id_q4_swiglu_case(backend.get(), dev);
    run_mul_mat_id_q4_broadcast_case(backend.get(), dev);
    run_strided_rms_norm_case(backend.get(), dev);
    run_broadcast_mul_case(backend.get(), dev);
    run_glue_ops_case(backend.get(), dev);
    run_sigmoid_mul_strided_fusion_case(backend.get(), dev);
    run_singleton_stride_concat_case(backend.get(), dev);
    run_router_ops_case(backend.get(), dev);
    run_large_masked_soft_max_case(backend.get(), dev);
    run_imrope_case(backend.get(), dev);
    run_gated_delta_net_case(backend.get(), dev);

    ggml_backend_ptr cpu_backend(ggml_backend_cpu_init());
    GGML_ASSERT(cpu_backend != nullptr);

    ggml_context_ptr sched_ctx = make_context();
    ggml_tensor * add_lhs = ggml_new_tensor_2d(sched_ctx.get(), GGML_TYPE_F32, 4, 2);
    ggml_tensor * add_rhs = ggml_new_tensor_2d(sched_ctx.get(), GGML_TYPE_F32, 4, 2);
    ggml_tensor * sum = ggml_add(sched_ctx.get(), add_lhs, add_rhs);
    ggml_tensor * sched_dst = ggml_rms_norm(sched_ctx.get(), sum, 1.0e-6f);

    ggml_cgraph * sched_graph = ggml_new_graph(sched_ctx.get());
    ggml_build_forward_expand(sched_graph, sched_dst);

    ggml_backend_t sched_backends[] = { backend.get(), cpu_backend.get() };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        sched_backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    GGML_ASSERT(sched != nullptr);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, sched_graph));

    const std::vector<float> add_input_lhs = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
    };
    const std::vector<float> add_input_rhs = {
        8.0f, 7.0f, 6.0f, 5.0f,
        4.0f, 3.0f, 2.0f, 1.0f,
    };
    ggml_backend_tensor_set(add_lhs, add_input_lhs.data(), 0, add_input_lhs.size() * sizeof(float));
    ggml_backend_tensor_set(add_rhs, add_input_rhs.data(), 0, add_input_rhs.size() * sizeof(float));

    GGML_ASSERT(ggml_backend_sched_graph_compute(sched, sched_graph) == GGML_STATUS_SUCCESS);

    std::vector<float> sched_output(add_input_lhs.size(), -1.0f);
    ggml_backend_tensor_get(sched_dst, sched_output.data(), 0, sched_output.size() * sizeof(float));
    expect_near(
        sched_output,
        reference_rms_norm(reference_add(add_input_lhs, add_input_rhs), 4, 1.0e-6f),
        1.0e-5f,
        "sched_output");

    ggml_backend_sched_free(sched);

    ggml_backend_synchronize(backend.get());
    return 0;
}
