#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-cpp.h>
#include <ggml-pyre.h>

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
    GGML_ASSERT(!ggml_backend_dev_supports_op(dev, add_dst));

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
