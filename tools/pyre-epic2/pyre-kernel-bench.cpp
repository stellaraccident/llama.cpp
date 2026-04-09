#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml-pyre.h>
#include <ggml-quants.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cinttypes>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct bench_options {
    std::string op = "rms_norm";
    int64_t ncols = 2048;
    int64_t nrows = 128;
    int64_t cols_dst = 1;
    int64_t n_experts = 256;
    int64_t n_ids = 8;
    int64_t n_tokens = 1;
    int warmup = 5;
    int iterations = 50;
    float eps = 1.0e-6f;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "Usage: %s [--op rms_norm|mul_mat_vec_f32|mul_mat_vec_f16|mul_mat_vec_bf16|mul_mat_vec_q4_k|mul_mat_vec_q5_k|mul_mat_vec_q6_k|mul_mat_vec_q8_0|mul_mat_id_q4_k|mul_mat_id_q4_k_mul|mul_mat_id_q4_k_swiglu] [--ncols N] [--nrows N] [--cols-dst N] [--n-experts N] [--n-ids N] [--n-tokens N] [--warmup N] [--iters N]\n"
        "\n"
        "Benchmarks ggml-pyre providers without a GGUF model.\n",
        argv0);
}

static bool parse_i64(const char * text, int64_t * value) {
    char * end = nullptr;
    const long long parsed = std::strtoll(text, &end, 10);
    if (!end || *end != '\0' || parsed <= 0) {
        return false;
    }
    *value = parsed;
    return true;
}

static bool parse_i32(const char * text, int * value) {
    int64_t parsed = 0;
    if (!parse_i64(text, &parsed) || parsed > 1000000) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

static bool parse_args(int argc, char ** argv, bench_options * options) {
    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else if (std::strcmp(arg, "--ncols") == 0) {
            const char * value = need_value(arg);
            if (!value || !parse_i64(value, &options->ncols)) {
                return false;
            }
        } else if (std::strcmp(arg, "--op") == 0) {
            const char * value = need_value(arg);
            if (!value) {
                return false;
            }
            options->op = value;
        } else if (std::strcmp(arg, "--nrows") == 0) {
            const char * value = need_value(arg);
            if (!value || !parse_i64(value, &options->nrows)) {
                return false;
            }
        } else if (std::strcmp(arg, "--cols-dst") == 0) {
            const char * value = need_value(arg);
            if (!value || !parse_i64(value, &options->cols_dst)) {
                return false;
            }
        } else if (std::strcmp(arg, "--n-experts") == 0) {
            const char * value = need_value(arg);
            if (!value || !parse_i64(value, &options->n_experts)) {
                return false;
            }
        } else if (std::strcmp(arg, "--n-ids") == 0) {
            const char * value = need_value(arg);
            if (!value || !parse_i64(value, &options->n_ids)) {
                return false;
            }
        } else if (std::strcmp(arg, "--n-tokens") == 0) {
            const char * value = need_value(arg);
            if (!value || !parse_i64(value, &options->n_tokens)) {
                return false;
            }
        } else if (std::strcmp(arg, "--warmup") == 0) {
            const char * value = need_value(arg);
            if (!value || !parse_i32(value, &options->warmup)) {
                return false;
            }
        } else if (std::strcmp(arg, "--iters") == 0) {
            const char * value = need_value(arg);
            if (!value || !parse_i32(value, &options->iterations)) {
                return false;
            }
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg);
            return false;
        }
    }
    return true;
}

static ggml_context_ptr make_context(size_t mem_size) {
    ggml_init_params params = {
        /* .mem_size   = */ mem_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    return ggml_context_ptr(ggml_init(params));
}

static std::vector<float> reference_rms_norm(const std::vector<float> & input, int64_t ncols, float eps) {
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

static std::vector<float> reference_mul_mat(
        const std::vector<float> & lhs, const std::vector<float> & rhs,
        int64_t k, int64_t rows, int64_t cols) {
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

static std::vector<float> reference_q8_1_rhs(
        const std::vector<float> & rhs, int64_t k, int64_t cols) {
    std::vector<block_q8_1> quantized(static_cast<size_t>(cols * (k / QK8_1)));
    std::vector<float> dequantized(rhs.size(), 0.0f);
    for (int64_t col = 0; col < cols; ++col) {
        quantize_row_q8_1_ref(
            rhs.data() + col * k,
            quantized.data() + col * (k / QK8_1),
            k);
        for (int64_t block = 0; block < k / QK8_1; ++block) {
            const block_q8_1 & q = quantized[static_cast<size_t>(col * (k / QK8_1) + block)];
            const float d = ggml_fp16_to_fp32(q.d);
            for (int64_t i = 0; i < QK8_1; ++i) {
                dequantized[static_cast<size_t>(col * k + block * QK8_1 + i)] =
                    d * static_cast<float>(q.qs[i]);
            }
        }
    }
    return dequantized;
}

static std::vector<float> reference_mul_mat_id(
        const std::vector<float> & lhs, const std::vector<float> & rhs,
        const std::vector<int32_t> & ids,
        int64_t k, int64_t rows, int64_t n_experts, int64_t n_ids, int64_t n_tokens) {
    std::vector<float> output(static_cast<size_t>(rows * n_ids * n_tokens), 0.0f);
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t id = 0; id < n_ids; ++id) {
            const int32_t expert = ids[static_cast<size_t>(token * n_ids + id)];
            if (expert < 0 || expert >= n_experts) {
                std::fprintf(stderr, "invalid expert id: %d\n", expert);
                std::exit(1);
            }
            for (int64_t row = 0; row < rows; ++row) {
                float sum = 0.0f;
                for (int64_t i = 0; i < k; ++i) {
                    sum += lhs[static_cast<size_t>((expert * rows + row) * k + i)] *
                        rhs[static_cast<size_t>((token * n_ids + id) * k + i)];
                }
                output[static_cast<size_t>(token * n_ids * rows + id * rows + row)] = sum;
            }
        }
    }
    return output;
}

static bool is_mul_mat_id_op(const std::string & op) {
    return op == "mul_mat_id_q4_k" ||
           op == "mul_mat_id_q4_k_mul" ||
           op == "mul_mat_id_q4_k_swiglu";
}

static bool env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] != '\0' &&
        std::strcmp(value, "0") != 0 &&
        std::strcmp(value, "false") != 0 &&
        std::strcmp(value, "FALSE") != 0 &&
        std::strcmp(value, "off") != 0 &&
        std::strcmp(value, "OFF") != 0;
}

static bool uses_q8_1_rhs(const bench_options & options) {
    if (!env_enabled("GGML_PYRE_ENABLE_Q8_1_MMVQ") || env_enabled("GGML_PYRE_DISABLE_Q8_1_MMVQ")) {
        return false;
    }
    const bool force_all = [] {
        const char * policy = std::getenv("GGML_PYRE_Q8_1_MMVQ_POLICY");
        return policy && std::strcmp(policy, "all") == 0;
    }();
    if (force_all) {
        return options.op == "mul_mat_vec_q4_k" ||
               options.op == "mul_mat_vec_q5_k" ||
               options.op == "mul_mat_vec_q6_k" ||
               options.op == "mul_mat_id_q4_k" ||
               options.op == "mul_mat_id_q4_k_mul";
    }
    return (options.op == "mul_mat_vec_q4_k" && options.ncols >= 2048 && options.nrows >= 4096) ||
           (options.op == "mul_mat_vec_q6_k" && options.ncols >= 2048 && options.nrows >= 2048);
}

static void check_close(const std::vector<float> & actual, const std::vector<float> & expected, bool approximate_rhs) {
    for (size_t i = 0; i < actual.size(); ++i) {
        const float delta = std::fabs(actual[i] - expected[i]);
        const float tolerance = approximate_rhs ?
            2.0e-2f + 5.0e-3f * std::fabs(expected[i]) :
            5.0e-3f + 1.0e-4f * std::fabs(expected[i]);
        if (delta > tolerance) {
            std::fprintf(stderr, "mismatch[%zu]: got %.9g expected %.9g delta %.9g\n",
                i, actual[i], expected[i], delta);
            std::exit(1);
        }
    }
}

} // namespace

int main(int argc, char ** argv) {
    bench_options options;
    if (!parse_args(argc, argv, &options)) {
        usage(argv[0]);
        return 2;
    }
    const bool q8_1_rhs = uses_q8_1_rhs(options);

    ggml_backend_dev_t dev = ggml_backend_dev_by_name("PYRE0");
    if (!dev) {
        std::fprintf(stderr, "PYRE0 not available; skipping pyre-kernel-bench\n");
        return 0;
    }

    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    if (!backend) {
        std::fprintf(stderr, "failed to initialize PYRE backend\n");
        return 1;
    }

    ggml_context_ptr ctx = make_context(16ull << 20);
    ggml_tensor * src = nullptr;
    ggml_tensor * src_gate = nullptr;
    ggml_tensor * rhs = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * scale = nullptr;
    ggml_tensor * dst = nullptr;
    size_t output_count = 0;
    std::vector<float> expected;

    std::vector<float> input;
    std::vector<ggml_fp16_t> input_f16;
    std::vector<ggml_bf16_t> input_bf16;
    std::vector<block_q4_K> input_q4_k;
    std::vector<block_q5_K> input_q5_k;
    std::vector<block_q6_K> input_q6_k;
    std::vector<block_q8_0> input_q8_0;
    std::vector<block_q4_K> input_gate_q4_k;
    std::vector<int32_t> expert_ids;
    std::vector<float> scale_f32;
    std::vector<float> rhs_f32;

    if (options.op == "rms_norm") {
        const size_t element_count = static_cast<size_t>(options.ncols * options.nrows);
        src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, options.ncols, options.nrows);
        dst = ggml_rms_norm(ctx.get(), src, options.eps);
        output_count = element_count;

        input.resize(element_count);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<float>(static_cast<int>(i % 251) - 125) / 127.0f;
        }
        expected = reference_rms_norm(input, options.ncols, options.eps);
    } else if (options.op == "mul_mat_vec_f16" || options.op == "mul_mat_vec_bf16" ||
               options.op == "mul_mat_vec_f32") {
        const ggml_type lhs_type = options.op == "mul_mat_vec_f16" ? GGML_TYPE_F16 :
            (options.op == "mul_mat_vec_bf16" ? GGML_TYPE_BF16 : GGML_TYPE_F32);
        src = ggml_new_tensor_2d(ctx.get(), lhs_type, options.ncols, options.nrows);
        rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, options.ncols, options.cols_dst);
        dst = ggml_mul_mat(ctx.get(), src, rhs);
        output_count = static_cast<size_t>(options.nrows * options.cols_dst);

        input.resize(static_cast<size_t>(options.ncols * options.nrows));
        rhs_f32.resize(static_cast<size_t>(options.ncols * options.cols_dst));
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<float>(static_cast<int>(i % 113) - 56) / 57.0f;
        }
        for (size_t i = 0; i < rhs_f32.size(); ++i) {
            rhs_f32[i] = static_cast<float>(static_cast<int>(i % 127) - 63) / 64.0f;
        }
        if (options.op == "mul_mat_vec_f16") {
            input_f16.resize(input.size());
            ggml_fp32_to_fp16_row(input.data(), input_f16.data(), static_cast<int64_t>(input_f16.size()));
            for (size_t i = 0; i < input.size(); ++i) {
                input[i] = ggml_fp16_to_fp32(input_f16[i]);
            }
        } else if (options.op == "mul_mat_vec_bf16") {
            input_bf16.resize(input.size());
            ggml_fp32_to_bf16_row(input.data(), input_bf16.data(), static_cast<int64_t>(input_bf16.size()));
            for (size_t i = 0; i < input.size(); ++i) {
                input[i] = ggml_bf16_to_fp32(input_bf16[i]);
            }
        }
        expected = reference_mul_mat(input, rhs_f32, options.ncols, options.nrows, options.cols_dst);
    } else if (options.op == "mul_mat_vec_q4_k" || options.op == "mul_mat_vec_q5_k" ||
               options.op == "mul_mat_vec_q6_k" || options.op == "mul_mat_vec_q8_0") {
        if (options.ncols % QK_K != 0) {
            std::fprintf(stderr, "%s requires --ncols to be divisible by %d\n", options.op.c_str(), QK_K);
            return 2;
        }
        const ggml_type lhs_type = options.op == "mul_mat_vec_q4_k" ? GGML_TYPE_Q4_K :
            (options.op == "mul_mat_vec_q5_k" ? GGML_TYPE_Q5_K :
             options.op == "mul_mat_vec_q6_k" ? GGML_TYPE_Q6_K : GGML_TYPE_Q8_0);
        src = ggml_new_tensor_2d(ctx.get(), lhs_type, options.ncols, options.nrows);
        rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, options.ncols, options.cols_dst);
        dst = ggml_mul_mat(ctx.get(), src, rhs);
        output_count = static_cast<size_t>(options.nrows * options.cols_dst);

        input.resize(static_cast<size_t>(options.ncols * options.nrows));
        rhs_f32.resize(static_cast<size_t>(options.ncols * options.cols_dst));
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<float>(static_cast<int>(i % 113) - 56) / 57.0f;
        }
        for (size_t i = 0; i < rhs_f32.size(); ++i) {
            rhs_f32[i] = static_cast<float>(static_cast<int>(i % 127) - 63) / 64.0f;
        }

        const int64_t blocks_per_row = options.ncols / QK_K;
        std::vector<float> dequantized(input.size(), 0.0f);
        if (options.op == "mul_mat_vec_q4_k") {
            input_q4_k.resize(static_cast<size_t>(options.nrows * blocks_per_row));
            for (int64_t row = 0; row < options.nrows; ++row) {
                quantize_row_q4_K_ref(
                    input.data() + row * options.ncols,
                    input_q4_k.data() + row * blocks_per_row,
                    options.ncols);
                dequantize_row_q4_K(
                    input_q4_k.data() + row * blocks_per_row,
                    dequantized.data() + row * options.ncols,
                    options.ncols);
            }
        } else if (options.op == "mul_mat_vec_q5_k") {
            input_q5_k.resize(static_cast<size_t>(options.nrows * blocks_per_row));
            for (int64_t row = 0; row < options.nrows; ++row) {
                quantize_row_q5_K_ref(
                    input.data() + row * options.ncols,
                    input_q5_k.data() + row * blocks_per_row,
                    options.ncols);
                dequantize_row_q5_K(
                    input_q5_k.data() + row * blocks_per_row,
                    dequantized.data() + row * options.ncols,
                    options.ncols);
            }
        } else if (options.op == "mul_mat_vec_q6_k") {
            input_q6_k.resize(static_cast<size_t>(options.nrows * blocks_per_row));
            for (int64_t row = 0; row < options.nrows; ++row) {
                quantize_row_q6_K_ref(
                    input.data() + row * options.ncols,
                    input_q6_k.data() + row * blocks_per_row,
                    options.ncols);
                dequantize_row_q6_K(
                    input_q6_k.data() + row * blocks_per_row,
                    dequantized.data() + row * options.ncols,
                    options.ncols);
            }
        } else {
            input_q8_0.resize(static_cast<size_t>(options.nrows * options.ncols / QK8_0));
            for (int64_t row = 0; row < options.nrows; ++row) {
                quantize_row_q8_0_ref(
                    input.data() + row * options.ncols,
                    input_q8_0.data() + row * (options.ncols / QK8_0),
                    options.ncols);
                dequantize_row_q8_0(
                    input_q8_0.data() + row * (options.ncols / QK8_0),
                    dequantized.data() + row * options.ncols,
                    options.ncols);
            }
        }
        const std::vector<float> rhs_expected =
            q8_1_rhs ? reference_q8_1_rhs(rhs_f32, options.ncols, options.cols_dst) : rhs_f32;
        expected = reference_mul_mat(dequantized, rhs_expected, options.ncols, options.nrows, options.cols_dst);
    } else if (is_mul_mat_id_op(options.op)) {
        if (options.ncols % QK_K != 0) {
            std::fprintf(stderr, "%s requires --ncols to be divisible by %d\n", options.op.c_str(), QK_K);
            return 2;
        }
        src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_K, options.ncols, options.nrows, options.n_experts);
        rhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, options.ncols, options.n_ids, options.n_tokens);
        ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, options.n_ids, options.n_tokens);
        dst = ggml_mul_mat_id(ctx.get(), src, rhs, ids);
        if (options.op == "mul_mat_id_q4_k_mul") {
            scale = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, options.n_ids, 1, 1);
            dst = ggml_mul(ctx.get(), dst, scale);
        } else if (options.op == "mul_mat_id_q4_k_swiglu") {
            src_gate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_K, options.ncols, options.nrows, options.n_experts);
            ggml_tensor * gate = ggml_mul_mat_id(ctx.get(), src_gate, rhs, ids);
            dst = ggml_swiglu_split(ctx.get(), gate, dst);
        }
        output_count = static_cast<size_t>(options.nrows * options.n_ids * options.n_tokens);

        input.resize(static_cast<size_t>(options.ncols * options.nrows * options.n_experts));
        std::vector<float> gate_input(input.size());
        rhs_f32.resize(static_cast<size_t>(options.ncols * options.n_ids * options.n_tokens));
        expert_ids.resize(static_cast<size_t>(options.n_ids * options.n_tokens));
        scale_f32.resize(static_cast<size_t>(options.n_ids));
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<float>(static_cast<int>(i % 113) - 56) / 57.0f;
            gate_input[i] = static_cast<float>(static_cast<int>((i + 17) % 109) - 54) / 55.0f;
        }
        for (size_t i = 0; i < rhs_f32.size(); ++i) {
            rhs_f32[i] = static_cast<float>(static_cast<int>(i % 127) - 63) / 64.0f;
        }
        for (int64_t token = 0; token < options.n_tokens; ++token) {
            for (int64_t id = 0; id < options.n_ids; ++id) {
                expert_ids[static_cast<size_t>(token * options.n_ids + id)] =
                    static_cast<int32_t>((token * 13 + id * 7) % options.n_experts);
            }
        }
        for (size_t i = 0; i < scale_f32.size(); ++i) {
            scale_f32[i] = static_cast<float>(i % 11 + 1) / 11.0f;
        }

        const int64_t blocks_per_row = options.ncols / QK_K;
        input_q4_k.resize(static_cast<size_t>(options.n_experts * options.nrows * blocks_per_row));
        input_gate_q4_k.resize(input_q4_k.size());
        std::vector<float> dequantized(input.size(), 0.0f);
        std::vector<float> gate_dequantized(input.size(), 0.0f);
        for (int64_t expert = 0; expert < options.n_experts; ++expert) {
            for (int64_t row = 0; row < options.nrows; ++row) {
                const size_t row_index = static_cast<size_t>(expert * options.nrows + row);
                quantize_row_q4_K_ref(
                    input.data() + row_index * options.ncols,
                    input_q4_k.data() + row_index * blocks_per_row,
                    options.ncols);
                dequantize_row_q4_K(
                    input_q4_k.data() + row_index * blocks_per_row,
                    dequantized.data() + row_index * options.ncols,
                    options.ncols);
                if (src_gate) {
                    quantize_row_q4_K_ref(
                        gate_input.data() + row_index * options.ncols,
                        input_gate_q4_k.data() + row_index * blocks_per_row,
                        options.ncols);
                    dequantize_row_q4_K(
                        input_gate_q4_k.data() + row_index * blocks_per_row,
                        gate_dequantized.data() + row_index * options.ncols,
                        options.ncols);
                }
            }
        }
        const std::vector<float> rhs_expected =
            q8_1_rhs ? reference_q8_1_rhs(rhs_f32, options.ncols, options.n_ids * options.n_tokens) : rhs_f32;
        expected = reference_mul_mat_id(
            dequantized, rhs_expected, expert_ids,
            options.ncols, options.nrows, options.n_experts, options.n_ids, options.n_tokens);
        if (options.op == "mul_mat_id_q4_k_mul") {
            for (int64_t token = 0; token < options.n_tokens; ++token) {
                for (int64_t id = 0; id < options.n_ids; ++id) {
                    for (int64_t row = 0; row < options.nrows; ++row) {
                        expected[static_cast<size_t>(token * options.n_ids * options.nrows + id * options.nrows + row)] *=
                            scale_f32[static_cast<size_t>(id)];
                    }
                }
            }
        } else if (options.op == "mul_mat_id_q4_k_swiglu") {
            std::vector<float> gate_expected = reference_mul_mat_id(
                gate_dequantized, rhs_expected, expert_ids,
                options.ncols, options.nrows, options.n_experts, options.n_ids, options.n_tokens);
            for (size_t i = 0; i < expected.size(); ++i) {
                const float gate = gate_expected[i];
                expected[i] *= gate / (1.0f + std::exp(-gate));
            }
        }
    } else {
        std::fprintf(stderr, "unsupported op: %s\n", options.op.c_str());
        return 2;
    }

    if (!ggml_backend_dev_supports_op(dev, dst)) {
        std::fprintf(stderr, "PYRE provider is unavailable for op %s on this build/device\n", options.op.c_str());
        return 0;
    }

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) {
        std::fprintf(stderr, "failed to allocate PYRE graph buffer\n");
        return 1;
    }

    if (options.op == "rms_norm") {
        ggml_backend_tensor_set(src, input.data(), 0, input.size() * sizeof(float));
    } else if (options.op == "mul_mat_vec_f16") {
        ggml_backend_tensor_set(src, input_f16.data(), 0, input_f16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    } else if (options.op == "mul_mat_vec_bf16") {
        ggml_backend_tensor_set(src, input_bf16.data(), 0, input_bf16.size() * sizeof(ggml_bf16_t));
        ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    } else if (options.op == "mul_mat_vec_f32") {
        ggml_backend_tensor_set(src, input.data(), 0, input.size() * sizeof(float));
        ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    } else if (options.op == "mul_mat_vec_q4_k") {
        ggml_backend_tensor_set(src, input_q4_k.data(), 0, input_q4_k.size() * sizeof(block_q4_K));
        ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    } else if (options.op == "mul_mat_vec_q5_k") {
        ggml_backend_tensor_set(src, input_q5_k.data(), 0, input_q5_k.size() * sizeof(block_q5_K));
        ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    } else if (options.op == "mul_mat_vec_q6_k") {
        ggml_backend_tensor_set(src, input_q6_k.data(), 0, input_q6_k.size() * sizeof(block_q6_K));
        ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    } else if (options.op == "mul_mat_vec_q8_0") {
        ggml_backend_tensor_set(src, input_q8_0.data(), 0, input_q8_0.size() * sizeof(block_q8_0));
        ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    } else if (is_mul_mat_id_op(options.op)) {
        ggml_backend_tensor_set(src, input_q4_k.data(), 0, input_q4_k.size() * sizeof(block_q4_K));
        if (src_gate) {
            ggml_backend_tensor_set(src_gate, input_gate_q4_k.data(), 0, input_gate_q4_k.size() * sizeof(block_q4_K));
        }
        ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
        ggml_backend_tensor_set(ids, expert_ids.data(), 0, expert_ids.size() * sizeof(int32_t));
        if (scale) {
            ggml_backend_tensor_set(scale, scale_f32.data(), 0, scale_f32.size() * sizeof(float));
        }
    }

    for (int i = 0; i < options.warmup; ++i) {
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "warmup graph compute failed\n");
            return 1;
        }
    }

    std::vector<double> samples;
    samples.reserve(options.iterations);
    for (int i = 0; i < options.iterations; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "graph compute failed\n");
            return 1;
        }
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    std::vector<float> output(output_count, 0.0f);
    ggml_backend_tensor_get(dst, output.data(), 0, output.size() * sizeof(float));
    check_close(output, expected, q8_1_rhs);

    std::sort(samples.begin(), samples.end());
    const double median_us = samples[samples.size() / 2];
    const double min_us = samples.front();
    const double bytes = options.op == "rms_norm" ?
        static_cast<double>(output_count * sizeof(float) * 2) :
        static_cast<double>(
            (options.op == "mul_mat_vec_f16" ?
                input_f16.size() * sizeof(ggml_fp16_t) :
             options.op == "mul_mat_vec_bf16" ?
                input_bf16.size() * sizeof(ggml_bf16_t) :
             options.op == "mul_mat_vec_f32" ?
                input.size() * sizeof(float) :
             options.op == "mul_mat_vec_q4_k" ?
                input_q4_k.size() * sizeof(block_q4_K) :
             options.op == "mul_mat_vec_q5_k" ?
                input_q5_k.size() * sizeof(block_q5_K) :
             options.op == "mul_mat_vec_q6_k" ?
                input_q6_k.size() * sizeof(block_q6_K) :
             options.op == "mul_mat_vec_q8_0" ?
                input_q8_0.size() * sizeof(block_q8_0) :
                (input_q4_k.size() + input_gate_q4_k.size()) * sizeof(block_q4_K)) +
            (rhs_f32.size() + output.size()) * sizeof(float));
    const double gbps = bytes / (min_us * 1.0e-6) / 1.0e9;
    std::printf(
        "{\"op\":\"%s\",\"backend\":\"PYRE\",\"device\":\"PYRE0\","
        "\"ncols\":%" PRId64 ",\"nrows\":%" PRId64 ",\"cols_dst\":%" PRId64
        ",\"n_experts\":%" PRId64 ",\"n_ids\":%" PRId64 ",\"n_tokens\":%" PRId64
        ",\"q8_1_rhs\":%s,\"median_us\":%.3f,"
        "\"min_us\":%.3f,\"bandwidth_gbps\":%.3f}\n",
        options.op.c_str(), options.ncols, options.nrows, options.cols_dst,
        options.n_experts, options.n_ids, options.n_tokens, q8_1_rhs ? "true" : "false",
        median_us, min_us, gbps);
    return 0;
}
