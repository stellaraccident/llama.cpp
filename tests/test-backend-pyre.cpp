#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-cpp.h>
#include <ggml-pyre.h>
#include <ggml-quants.h>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstdio>
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

} // namespace

int main() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("PYRE0");
    if (!dev) {
        std::fprintf(stderr, "PYRE0 not available; skipping test-backend-pyre\n");
        return 0;
    }

    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    GGML_ASSERT(backend != nullptr);

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
    ggml_tensor * row_base = ggml_new_tensor_2d(elem_ctx.get(), GGML_TYPE_F32, 6, 2);
    ggml_tensor * row_view = ggml_view_2d(elem_ctx.get(), row_base, 4, 2, row_base->nb[1], 2 * sizeof(float));
    ggml_tensor * row_cpy_target = ggml_new_tensor_1d(elem_ctx.get(), GGML_TYPE_F32, 8);
    ggml_tensor * row_cpy = ggml_cpy(elem_ctx.get(), row_view, row_cpy_target);
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
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, row_cpy));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, set_rows_f32));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, set_rows_f16));

    ggml_cgraph * elem_graph = ggml_new_graph(elem_ctx.get());
    ggml_build_forward_expand(elem_graph, elem_add);
    ggml_build_forward_expand(elem_graph, elem_mul);
    ggml_build_forward_expand(elem_graph, elem_scale);
    ggml_build_forward_expand(elem_graph, elem_cpy);
    ggml_build_forward_expand(elem_graph, row_cpy);
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
    std::vector<float> row_cpy_output(8, -1.0f);
    std::vector<float> set_rows_f32_output(16, -2.0f);
    std::vector<ggml_fp16_t> set_rows_f16_output_raw(16);
    std::vector<float> set_rows_f16_output(16, -2.0f);
    ggml_backend_tensor_get(elem_add, elem_add_output.data(), 0, elem_add_output.size() * sizeof(float));
    ggml_backend_tensor_get(elem_mul, elem_mul_output.data(), 0, elem_mul_output.size() * sizeof(float));
    ggml_backend_tensor_get(elem_scale, elem_scale_output.data(), 0, elem_scale_output.size() * sizeof(float));
    ggml_backend_tensor_get(elem_cpy, elem_cpy_output.data(), 0, elem_cpy_output.size() * sizeof(float));
    ggml_backend_tensor_get(row_cpy, row_cpy_output.data(), 0, row_cpy_output.size() * sizeof(float));
    ggml_backend_tensor_get(set_rows_f32, set_rows_f32_output.data(), 0, set_rows_f32_output.size() * sizeof(float));
    ggml_backend_tensor_get(set_rows_f16, set_rows_f16_output_raw.data(), 0, set_rows_f16_output_raw.size() * sizeof(ggml_fp16_t));
    ggml_fp16_to_fp32_row(set_rows_f16_output_raw.data(), set_rows_f16_output.data(), static_cast<int64_t>(set_rows_f16_output.size()));
    expect_near(elem_add_output, reference_add(elem_input_lhs, elem_input_rhs), 1.0e-6f, "elem_add_output");
    expect_near(elem_mul_output, reference_mul(elem_input_lhs, elem_input_rhs), 1.0e-6f, "elem_mul_output");
    expect_near(elem_scale_output, reference_scale(elem_input_lhs, 1.5f, -0.25f), 1.0e-6f, "elem_scale_output");
    expect_near(elem_cpy_output, elem_input_lhs, 1.0e-6f, "elem_cpy_output");
    expect_near(row_cpy_output, { 2.0f, 3.0f, 4.0f, 5.0f, 8.0f, 9.0f, 10.0f, 11.0f }, 1.0e-6f, "row_cpy_output");
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
    run_matvec_case(backend.get(), dev, GGML_TYPE_Q5_K, "q5_output");
    run_matvec_case(backend.get(), dev, GGML_TYPE_Q6_K, "q6_output");
    run_matvec_case(backend.get(), dev, GGML_TYPE_Q8_0, "q8_output");

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
