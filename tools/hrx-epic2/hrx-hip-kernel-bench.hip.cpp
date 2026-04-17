#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define hrx_reduce_wg hrx_q6_reduce_wg
#define hrx_reduce_wg16 hrx_q6_reduce_wg16
#include "../../ggml/src/ggml-hrx/kernels/mul_mat_vec_q6_k.hip.cpp"
#undef hrx_reduce_wg
#undef hrx_reduce_wg16

#define hrx_reduce_wg hrx_q5_reduce_wg
#include "../../ggml/src/ggml-hrx/kernels/mul_mat_vec_q5_k.hip.cpp"
#undef hrx_reduce_wg

namespace {

struct options {
    std::string variant = "q6_rows2_cols1";
    long long k = 2048;
    long long rows = 4096;
    long long cols = 1;
    int warmup = 20;
    int iters = 200;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "Usage: %s [--variant q6_rows2_cols1|q6_rows4_cols1|q6_rows8_cols1|q6_wg64|q6_wg128|q6_wg256|q5_wg64|q5_wg128|q5_wg256] "
        "[--k N] [--rows N] [--cols N] [--warmup N] [--iters N]\n",
        argv0);
}

static bool parse_ll(const char * text, long long * out) {
    char * end = nullptr;
    const long long value = std::strtoll(text, &end, 10);
    if (!end || *end != '\0' || value <= 0) {
        return false;
    }
    *out = value;
    return true;
}

static bool parse_int(const char * text, int * out) {
    long long value = 0;
    if (!parse_ll(text, &value) || value > 10000000) {
        return false;
    }
    *out = static_cast<int>(value);
    return true;
}

static bool parse_args(int argc, char ** argv, options * opts) {
    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else if (std::strcmp(arg, "--variant") == 0) {
            const char * value = need_value(arg);
            if (!value) {
                return false;
            }
            opts->variant = value;
        } else if (std::strcmp(arg, "--k") == 0) {
            const char * value = need_value(arg);
            if (!value || !parse_ll(value, &opts->k)) {
                return false;
            }
        } else if (std::strcmp(arg, "--rows") == 0) {
            const char * value = need_value(arg);
            if (!value || !parse_ll(value, &opts->rows)) {
                return false;
            }
        } else if (std::strcmp(arg, "--cols") == 0) {
            const char * value = need_value(arg);
            if (!value || !parse_ll(value, &opts->cols)) {
                return false;
            }
        } else if (std::strcmp(arg, "--warmup") == 0) {
            const char * value = need_value(arg);
            if (!value || !parse_int(value, &opts->warmup)) {
                return false;
            }
        } else if (std::strcmp(arg, "--iters") == 0) {
            const char * value = need_value(arg);
            if (!value || !parse_int(value, &opts->iters)) {
                return false;
            }
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg);
            return false;
        }
    }
    return true;
}

static void hip_check(hipError_t status, const char * expr, const char * file, int line) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s:%d: %s failed: %s\n", file, line, expr, hipGetErrorString(status));
        std::exit(1);
    }
}

#define HIP_CHECK(expr) hip_check((expr), #expr, __FILE__, __LINE__)

static void fill_q6_blocks(std::vector<hrx_block_q6_K> * blocks) {
    for (size_t i = 0; i < blocks->size(); ++i) {
        hrx_block_q6_K & block = (*blocks)[i];
        for (int j = 0; j < 128; ++j) {
            block.ql[j] = static_cast<uint8_t>((i * 17 + j * 13) & 0xFF);
        }
        for (int j = 0; j < 64; ++j) {
            block.qh[j] = static_cast<uint8_t>((i * 19 + j * 7) & 0xFF);
        }
        for (int j = 0; j < 16; ++j) {
            block.scales[j] = static_cast<int8_t>((static_cast<int>(i + j) % 15) - 7);
        }
        block.d = 0x3800; // fp16 0.5
    }
}

static void fill_q5_blocks(std::vector<hrx_block_q5_K> * blocks) {
    for (size_t i = 0; i < blocks->size(); ++i) {
        hrx_block_q5_K & block = (*blocks)[i];
        block.d = 0x3800;    // fp16 0.5
        block.dmin = 0x3400; // fp16 0.25
        for (int j = 0; j < 12; ++j) {
            block.scales[j] = static_cast<uint8_t>((i * 11 + j * 17) & 0x3F);
        }
        for (int j = 0; j < 32; ++j) {
            block.qh[j] = static_cast<uint8_t>((i * 19 + j * 5) & 0xFF);
        }
        for (int j = 0; j < 128; ++j) {
            block.qs[j] = static_cast<uint8_t>((i * 13 + j * 23) & 0xFF);
        }
    }
}

static void fill_rhs(std::vector<float> * rhs) {
    for (size_t i = 0; i < rhs->size(); ++i) {
        (*rhs)[i] = static_cast<float>(static_cast<int>((i * 23) % 127) - 63) / 64.0f;
    }
}

static void launch_q6(const options & opts, const hrx_block_q6_K * a, const float * x, float * y) {
    if (opts.variant == "q6_rows2_cols1") {
        dim3 grid(static_cast<unsigned int>((opts.rows + 1) / 2), static_cast<unsigned int>(opts.cols), 1);
        dim3 block(32, 1, 1);
        hipLaunchKernelGGL(
            hrx_mul_mat_vec_q6_k_rows2_cols1_wg32_f32,
            grid, block, 0, 0, a, x, y, opts.k, opts.rows, opts.cols);
    } else if (opts.variant == "q6_rows4_cols1") {
        dim3 grid(static_cast<unsigned int>((opts.rows + 3) / 4), static_cast<unsigned int>(opts.cols), 1);
        dim3 block(64, 1, 1);
        hipLaunchKernelGGL(
            hrx_mul_mat_vec_q6_k_rows4_cols1_wg64_f32,
            grid, block, 0, 0, a, x, y, opts.k, opts.rows, opts.cols);
    } else if (opts.variant == "q6_rows8_cols1") {
        dim3 grid(static_cast<unsigned int>((opts.rows + 7) / 8), static_cast<unsigned int>(opts.cols), 1);
        dim3 block(128, 1, 1);
        hipLaunchKernelGGL(
            hrx_mul_mat_vec_q6_k_rows8_cols1_wg128_f32,
            grid, block, 0, 0, a, x, y, opts.k, opts.rows, opts.cols);
    } else if (opts.variant == "q6_wg64") {
        dim3 grid(static_cast<unsigned int>(opts.rows), static_cast<unsigned int>(opts.cols), 1);
        dim3 block(64, 1, 1);
        hipLaunchKernelGGL(
            hrx_mul_mat_vec_q6_k_wg64_f32,
            grid, block, 0, 0, a, x, y, opts.k, opts.rows, opts.cols);
    } else if (opts.variant == "q6_wg128") {
        dim3 grid(static_cast<unsigned int>(opts.rows), static_cast<unsigned int>(opts.cols), 1);
        dim3 block(128, 1, 1);
        hipLaunchKernelGGL(
            hrx_mul_mat_vec_q6_k_wg128_f32,
            grid, block, 0, 0, a, x, y, opts.k, opts.rows, opts.cols);
    } else if (opts.variant == "q6_wg256") {
        dim3 grid(static_cast<unsigned int>(opts.rows), static_cast<unsigned int>(opts.cols), 1);
        dim3 block(256, 1, 1);
        hipLaunchKernelGGL(
            hrx_mul_mat_vec_q6_k_f32,
            grid, block, 0, 0, a, x, y, opts.k, opts.rows, opts.cols);
    } else {
        std::fprintf(stderr, "unsupported variant: %s\n", opts.variant.c_str());
        std::exit(2);
    }
    HIP_CHECK(hipGetLastError());
}

static void launch_q5(const options & opts, const hrx_block_q5_K * a, const float * x, float * y) {
    dim3 grid(static_cast<unsigned int>(opts.rows), static_cast<unsigned int>(opts.cols), 1);
    if (opts.variant == "q5_wg64") {
        dim3 block(64, 1, 1);
        hipLaunchKernelGGL(
            hrx_mul_mat_vec_q5_k_wg64_f32,
            grid, block, 0, 0, a, x, y, opts.k, opts.rows, opts.cols);
    } else if (opts.variant == "q5_wg128") {
        dim3 block(128, 1, 1);
        hipLaunchKernelGGL(
            hrx_mul_mat_vec_q5_k_wg128_f32,
            grid, block, 0, 0, a, x, y, opts.k, opts.rows, opts.cols);
    } else if (opts.variant == "q5_wg256") {
        dim3 block(256, 1, 1);
        hipLaunchKernelGGL(
            hrx_mul_mat_vec_q5_k_f32,
            grid, block, 0, 0, a, x, y, opts.k, opts.rows, opts.cols);
    } else {
        std::fprintf(stderr, "unsupported q5 variant: %s\n", opts.variant.c_str());
        std::exit(2);
    }
    HIP_CHECK(hipGetLastError());
}

} // namespace

int main(int argc, char ** argv) {
    options opts;
    if (!parse_args(argc, argv, &opts)) {
        usage(argv[0]);
        return 2;
    }
    if (opts.k % 256 != 0) {
        std::fprintf(stderr, "--k must be divisible by 256 for Q6_K\n");
        return 2;
    }

    std::vector<float> host_x(static_cast<size_t>(opts.k * opts.cols));
    std::vector<float> host_y(static_cast<size_t>(opts.rows * opts.cols), 0.0f);
    fill_rhs(&host_x);

    const bool is_q5 = opts.variant.rfind("q5_", 0) == 0;
    const long long blocks_per_row = opts.k / 256;
    std::vector<hrx_block_q6_K> host_q6;
    std::vector<hrx_block_q5_K> host_q5;
    size_t lhs_bytes = 0;
    if (is_q5) {
        host_q5.resize(static_cast<size_t>(opts.rows * blocks_per_row));
        fill_q5_blocks(&host_q5);
        lhs_bytes = host_q5.size() * sizeof(host_q5[0]);
    } else {
        host_q6.resize(static_cast<size_t>(opts.rows * blocks_per_row));
        fill_q6_blocks(&host_q6);
        lhs_bytes = host_q6.size() * sizeof(host_q6[0]);
    }

    void * dev_a = nullptr;
    float * dev_x = nullptr;
    float * dev_y = nullptr;
    HIP_CHECK(hipMalloc(&dev_a, lhs_bytes));
    HIP_CHECK(hipMalloc(&dev_x, host_x.size() * sizeof(host_x[0])));
    HIP_CHECK(hipMalloc(&dev_y, host_y.size() * sizeof(host_y[0])));
    HIP_CHECK(hipMemcpy(dev_a, is_q5 ? static_cast<const void *>(host_q5.data()) : static_cast<const void *>(host_q6.data()),
                        lhs_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_x, host_x.data(), host_x.size() * sizeof(host_x[0]), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dev_y, 0, host_y.size() * sizeof(host_y[0])));

    for (int i = 0; i < opts.warmup; ++i) {
        if (is_q5) {
            launch_q5(opts, static_cast<const hrx_block_q5_K *>(dev_a), dev_x, dev_y);
        } else {
            launch_q6(opts, static_cast<const hrx_block_q6_K *>(dev_a), dev_x, dev_y);
        }
    }
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, 0));
    for (int i = 0; i < opts.iters; ++i) {
        if (is_q5) {
            launch_q5(opts, static_cast<const hrx_block_q5_K *>(dev_a), dev_x, dev_y);
        } else {
            launch_q6(opts, static_cast<const hrx_block_q6_K *>(dev_a), dev_x, dev_y);
        }
    }
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));

    HIP_CHECK(hipMemcpy(host_y.data(), dev_y, host_y.size() * sizeof(host_y[0]), hipMemcpyDeviceToHost));
    double checksum = 0.0;
    for (float value : host_y) {
        checksum += static_cast<double>(value);
    }

    const double avg_us = static_cast<double>(elapsed_ms) * 1000.0 / static_cast<double>(opts.iters);
    const double bytes =
        static_cast<double>(lhs_bytes + host_x.size() * sizeof(host_x[0]) + host_y.size() * sizeof(host_y[0]));
    const double gbps = bytes / (avg_us * 1.0e-6) / 1.0e9;
    std::printf(
        "{\"variant\":\"%s\",\"k\":%lld,\"rows\":%lld,\"cols\":%lld,"
        "\"iters\":%d,\"avg_us\":%.6f,\"bandwidth_gbps\":%.3f,\"checksum\":%.9g}\n",
        opts.variant.c_str(), opts.k, opts.rows, opts.cols, opts.iters, avg_us, gbps, checksum);

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(dev_a));
    HIP_CHECK(hipFree(dev_x));
    HIP_CHECK(hipFree(dev_y));
    return 0;
}
