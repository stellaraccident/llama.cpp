#include "ggml-pyre.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "kernels/pyre_kernel_catalog.h"
#include "pyre_runtime.h"

#include <array>
#include <cstdarg>
#include <cmath>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

static constexpr size_t GGML_PYRE_ALIGNMENT = 256;
static constexpr uintptr_t GGML_PYRE_FAKE_PTR_BASE = 0x1000;
static constexpr uint32_t GGML_PYRE_RMS_NORM_WORKGROUP_SIZE = 512;
static constexpr int64_t GGML_PYRE_MUL_MAT_VEC_MAX_COLS = 512;
static constexpr uint32_t GGML_PYRE_TRACE_FALLBACK_LIMIT = 64;
static constexpr uint32_t GGML_PYRE_TRACE_MUL_MAT_DETAIL_LIMIT = 32;

enum class ggml_backend_pyre_provider_kind {
    none,
    direct_executable,
};

enum class ggml_backend_pyre_kernel_provider_mode {
    pure_hip,
    iree,
    fallback,
};

struct ggml_backend_pyre_provider_policy {
    ggml_backend_pyre_kernel_provider_mode kernel_provider = ggml_backend_pyre_kernel_provider_mode::pure_hip;
    bool trace_providers = false;
    bool disable_fusion = false;
    bool disable_rms_norm = false;
    bool disable_mul_mat_vec = false;
    bool disable_mul_mat_id = false;
    bool disable_add_add_fusion = false;
    bool disable_add_rms_norm_mul_fusion = false;
    bool disable_topk_subgroup = false;
    bool enable_q8_1_mmvq = false;
    bool disable_q8_1_mmvq = false;
};

struct ggml_backend_pyre_op_provider {
    ggml_backend_pyre_provider_kind kind = ggml_backend_pyre_provider_kind::none;
    pyre_executable_t executable = nullptr;
    uint32_t export_ordinal = 0;
    pyre_executable_export_info_t export_info = {};

    ~ggml_backend_pyre_op_provider() {
        if (executable) {
            pyre_executable_release(executable);
        }
    }
};

struct ggml_backend_pyre_device_context {
    pyre_device_t device = nullptr;
    std::string name;
    std::string description;
    std::string architecture;
    size_t memory_total = 0;
    ggml_backend_pyre_provider_policy policy;
    ggml_backend_pyre_op_provider rms_norm_provider;
    ggml_backend_pyre_op_provider rms_norm_mul_provider;
    ggml_backend_pyre_op_provider add_rms_norm_mul_broadcast_provider;
    ggml_backend_pyre_op_provider add_provider;
    ggml_backend_pyre_op_provider add_broadcast_provider;
    ggml_backend_pyre_op_provider add_add_broadcast_provider;
    ggml_backend_pyre_op_provider mul_provider;
    ggml_backend_pyre_op_provider mul_broadcast_provider;
    ggml_backend_pyre_op_provider div_broadcast_provider;
    ggml_backend_pyre_op_provider scale_provider;
    ggml_backend_pyre_op_provider set_rows_f32_provider;
    ggml_backend_pyre_op_provider set_rows_f16_provider;
    ggml_backend_pyre_op_provider silu_provider;
    ggml_backend_pyre_op_provider sigmoid_provider;
    ggml_backend_pyre_op_provider softplus_provider;
    ggml_backend_pyre_op_provider swiglu_provider;
    ggml_backend_pyre_op_provider sum_rows_provider;
    ggml_backend_pyre_op_provider l2_norm_provider;
    ggml_backend_pyre_op_provider clamp_provider;
    ggml_backend_pyre_op_provider get_rows_f32_provider;
    ggml_backend_pyre_op_provider get_rows_q5_k_provider;
    ggml_backend_pyre_op_provider concat_f32_provider;
    ggml_backend_pyre_op_provider soft_max_f32_provider;
    ggml_backend_pyre_op_provider soft_max_f32_mask_provider;
    ggml_backend_pyre_op_provider argsort_f32_provider;
    ggml_backend_pyre_op_provider rope_f32_provider;
    ggml_backend_pyre_op_provider topk_moe_f32_provider;
    ggml_backend_pyre_op_provider topk_moe_f32_subgroup_provider;
    ggml_backend_pyre_op_provider ssm_conv_provider;
    ggml_backend_pyre_op_provider gated_delta_net_provider;
    ggml_backend_pyre_op_provider mul_mat_vec_bf16_provider;
    ggml_backend_pyre_op_provider mul_mat_vec_f16_provider;
    ggml_backend_pyre_op_provider mul_mat_vec_f16_batched_provider;
    ggml_backend_pyre_op_provider mul_mat_vec_f32_provider;
    ggml_backend_pyre_op_provider mul_mat_vec_f32_batched_provider;
    ggml_backend_pyre_op_provider mul_mat_id_q4_k_provider;
    ggml_backend_pyre_op_provider mul_mat_id_q4_k_q8_1_provider;
    ggml_backend_pyre_op_provider mul_mat_id_q4_k_mul_provider;
    ggml_backend_pyre_op_provider mul_mat_id_q4_k_mul_q8_1_provider;
    ggml_backend_pyre_op_provider mul_mat_vec_q4_k_provider;
    ggml_backend_pyre_op_provider quantize_q8_1_provider;
    ggml_backend_pyre_op_provider mul_mat_vec_q4_k_q8_1_provider;
    ggml_backend_pyre_op_provider mul_mat_vec_q5_k_provider;
    ggml_backend_pyre_op_provider mul_mat_vec_q6_k_provider;
    ggml_backend_pyre_op_provider mul_mat_vec_q8_0_provider;
    uint32_t fallback_trace_count = 0;
    uint32_t mul_mat_fallback_trace_count = 0;
    uint32_t mul_mat_id_fallback_trace_count = 0;
};

struct ggml_backend_pyre_reg_context {
    bool gpu_initialized = false;
    std::vector<std::unique_ptr<ggml_backend_pyre_device_context>> device_contexts;
    std::vector<ggml_backend_device> devices;

    ~ggml_backend_pyre_reg_context() {
        for (auto & device_context : device_contexts) {
            if (device_context && device_context->device) {
                pyre_device_release(device_context->device);
                device_context->device = nullptr;
            }
        }
        if (gpu_initialized) {
            pyre_status_t status = pyre_gpu_shutdown();
            if (!pyre_status_is_ok(status)) {
                pyre_status_ignore(status);
            }
        }
    }
};

struct ggml_backend_pyre_buffer_type_context {
    ggml_backend_pyre_device_context * device_context;
    std::string name;
    pyre_buffer_params_t params;
};

struct ggml_backend_pyre_buffer_context {
    ggml_backend_pyre_device_context * device_context;
    pyre_buffer_t buffer;
    uint8_t * base;
};

struct ggml_backend_pyre_context {
    ggml_backend_pyre_device_context * device_context;
    pyre_stream_t stream;
    std::string name;
    pyre_buffer_t scratch_q8_1 = nullptr;
    size_t scratch_q8_1_size = 0;
    std::vector<pyre_buffer_t> retired_scratch_q8_1;
    uint64_t dispatch_count = 0;
    uint64_t rms_norm_count = 0;
    uint64_t elementwise_count = 0;
    uint64_t mul_mat_vec_count = 0;
    uint64_t mul_mat_id_count = 0;
    uint64_t copy_count = 0;
    uint64_t set_rows_count = 0;
    uint64_t get_rows_count = 0;
    uint64_t concat_count = 0;
    uint64_t soft_max_count = 0;
    uint64_t argsort_count = 0;
    uint64_t rope_count = 0;
    uint64_t topk_moe_count = 0;
    uint64_t unary_count = 0;
    uint64_t reduction_count = 0;
    uint64_t ssm_conv_count = 0;
    uint64_t gated_delta_net_count = 0;
    uint64_t metadata_count = 0;
    uint64_t synchronize_count = 0;
};

static bool ggml_backend_pyre_log_status(pyre_status_t status, const char * expr, const char * file, int line) {
    if (pyre_status_is_ok(status)) {
        return true;
    }

    char * message = nullptr;
    size_t length = 0;
    pyre_status_to_string(status, &message, &length);
    GGML_LOG_ERROR("%s:%d: %s failed: %s\n", file, line, expr, message ? message : "unknown pyre error");
    pyre_status_free_message(message);
    pyre_status_ignore(status);
    return false;
}

#define GGML_PYRE_CHECK(expr) ggml_backend_pyre_log_status((expr), #expr, __FILE__, __LINE__)

static ggml_guid_t ggml_backend_pyre_guid(void) {
    static ggml_guid guid = { 0x1c, 0x65, 0x79, 0x0a, 0x31, 0x8b, 0x4d, 0xa6, 0x9e, 0x16, 0x6f, 0x13, 0x39, 0xb2, 0xe7, 0x5c };
    return &guid;
}

static ggml_backend_pyre_device_context * ggml_backend_pyre_get_device_context(ggml_backend_dev_t dev) {
    return static_cast<ggml_backend_pyre_device_context *>(dev->context);
}

static ggml_backend_pyre_buffer_type_context * ggml_backend_pyre_get_buft_context(ggml_backend_buffer_type_t buft) {
    return static_cast<ggml_backend_pyre_buffer_type_context *>(buft->context);
}

static ggml_backend_pyre_buffer_context * ggml_backend_pyre_get_buffer_context(ggml_backend_buffer_t buffer) {
    return static_cast<ggml_backend_pyre_buffer_context *>(buffer->context);
}

static void * ggml_backend_pyre_buffer_get_base(ggml_backend_buffer_t buffer);

static size_t ggml_backend_pyre_tensor_offset(const ggml_backend_pyre_buffer_context * context, const ggml_tensor * tensor) {
    return static_cast<size_t>(static_cast<const uint8_t *>(tensor->data) - context->base);
}

static pyre_buffer_t ggml_backend_pyre_tensor_buffer(const ggml_tensor * tensor) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    auto * context = ggml_backend_pyre_get_buffer_context(buffer);
    return context->buffer;
}

static bool ggml_backend_pyre_tensor_buffer_ref(
        const ggml_tensor * tensor, pyre_buffer_ref_t * out_ref) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (!buffer || buffer->iface.get_base != ggml_backend_pyre_buffer_get_base) {
        return false;
    }

    auto * context = ggml_backend_pyre_get_buffer_context(buffer);
    if (!context->buffer) {
        return false;
    }

    *out_ref = {
        /* .buffer = */ context->buffer,
        /* .offset = */ ggml_backend_pyre_tensor_offset(context, tensor),
        /* .length = */ ggml_nbytes(tensor),
    };
    return true;
}

static bool ggml_backend_pyre_ensure_q8_1_scratch(
        ggml_backend_pyre_context * context,
        size_t size,
        pyre_buffer_ref_t * out_ref) {
    if (size == 0) {
        return false;
    }
    if (context->scratch_q8_1_size < size) {
        if (context->scratch_q8_1) {
            context->retired_scratch_q8_1.push_back(context->scratch_q8_1);
            context->scratch_q8_1 = nullptr;
            context->scratch_q8_1_size = 0;
        }
        pyre_buffer_params_t params = {
            /* .type = */ PYRE_MEMORY_TYPE_DEVICE_LOCAL,
            /* .access = */ PYRE_MEMORY_ACCESS_ALL,
            /* .usage = */ PYRE_BUFFER_USAGE_DEFAULT,
            /* .queue_affinity = */ 0,
        };
        if (!GGML_PYRE_CHECK(pyre_allocator_allocate_buffer(
                pyre_device_allocator(context->device_context->device),
                params,
                size,
                &context->scratch_q8_1))) {
            return false;
        }
        context->scratch_q8_1_size = size;
    }
    *out_ref = {
        /* .buffer = */ context->scratch_q8_1,
        /* .offset = */ 0,
        /* .length = */ size,
    };
    return true;
}

static size_t ggml_backend_pyre_total_memory(pyre_device_t device) {
    uint64_t memory_total = 0;
    if (!GGML_PYRE_CHECK(pyre_device_get_property(
            device, PYRE_DEVICE_PROPERTY_TOTAL_MEMORY,
            &memory_total, sizeof(memory_total)))) {
        return 0;
    }
    return static_cast<size_t>(memory_total);
}

static std::string ggml_backend_pyre_device_description(pyre_device_t device) {
    std::array<char, 128> name = {};
    std::array<char, 128> architecture = {};

    if (!GGML_PYRE_CHECK(pyre_device_get_property(
            device, PYRE_DEVICE_PROPERTY_NAME, name.data(), name.size()))) {
        std::snprintf(name.data(), name.size(), "unknown");
    }

    if (!GGML_PYRE_CHECK(pyre_device_get_property(
            device, PYRE_DEVICE_PROPERTY_ARCHITECTURE,
            architecture.data(), architecture.size()))) {
        std::snprintf(architecture.data(), architecture.size(), "unknown");
    }

    std::string description(name.data());
    if (!description.empty() && architecture[0] != '\0') {
        description += " (";
        description += architecture.data();
        description += ")";
    }
    return description.empty() ? std::string("Pyre GPU") : description;
}

static std::string ggml_backend_pyre_device_architecture(pyre_device_t device) {
    std::array<char, 128> architecture = {};
    if (!GGML_PYRE_CHECK(pyre_device_get_property(
            device, PYRE_DEVICE_PROPERTY_ARCHITECTURE,
            architecture.data(), architecture.size()))) {
        return std::string();
    }
    return std::string(architecture.data());
}

static bool ggml_backend_pyre_env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static ggml_backend_pyre_kernel_provider_mode ggml_backend_pyre_kernel_provider_mode_from_env() {
    const char * value = std::getenv("GGML_PYRE_KERNEL_PROVIDER");
    if (!value || value[0] == '\0' || std::strcmp(value, "pure_hip") == 0) {
        return ggml_backend_pyre_kernel_provider_mode::pure_hip;
    }
    if (std::strcmp(value, "iree") == 0) {
        return ggml_backend_pyre_kernel_provider_mode::iree;
    }
    if (std::strcmp(value, "fallback") == 0) {
        return ggml_backend_pyre_kernel_provider_mode::fallback;
    }
    GGML_LOG_WARN("%s: unknown GGML_PYRE_KERNEL_PROVIDER=%s, using pure_hip\n", __func__, value);
    return ggml_backend_pyre_kernel_provider_mode::pure_hip;
}

static const char * ggml_backend_pyre_kernel_provider_mode_name(ggml_backend_pyre_kernel_provider_mode mode) {
    switch (mode) {
        case ggml_backend_pyre_kernel_provider_mode::pure_hip: return "pure_hip";
        case ggml_backend_pyre_kernel_provider_mode::iree: return "iree";
        case ggml_backend_pyre_kernel_provider_mode::fallback: return "fallback";
    }
    return "unknown";
}

static ggml_backend_pyre_provider_policy ggml_backend_pyre_provider_policy_from_env() {
    return {
        /* .kernel_provider     = */ ggml_backend_pyre_kernel_provider_mode_from_env(),
        /* .trace_providers    = */ ggml_backend_pyre_env_enabled("GGML_PYRE_TRACE_PROVIDERS"),
        /* .disable_fusion     = */ ggml_backend_pyre_env_enabled("GGML_PYRE_DISABLE_FUSION"),
        /* .disable_rms_norm   = */ ggml_backend_pyre_env_enabled("GGML_PYRE_DISABLE_RMS_NORM"),
        /* .disable_mul_mat_vec = */ ggml_backend_pyre_env_enabled("GGML_PYRE_DISABLE_MUL_MAT_VEC"),
        /* .disable_mul_mat_id = */ ggml_backend_pyre_env_enabled("GGML_PYRE_DISABLE_MUL_MAT_ID"),
        /* .disable_add_add_fusion = */ ggml_backend_pyre_env_enabled("GGML_PYRE_DISABLE_ADD_ADD_FUSION"),
        /* .disable_add_rms_norm_mul_fusion = */ ggml_backend_pyre_env_enabled("GGML_PYRE_DISABLE_ADD_RMS_NORM_MUL_FUSION"),
        /* .disable_topk_subgroup = */ ggml_backend_pyre_env_enabled("GGML_PYRE_DISABLE_TOPK_SUBGROUP"),
        /* .enable_q8_1_mmvq = */ ggml_backend_pyre_env_enabled("GGML_PYRE_ENABLE_Q8_1_MMVQ"),
        /* .disable_q8_1_mmvq = */ ggml_backend_pyre_env_enabled("GGML_PYRE_DISABLE_Q8_1_MMVQ"),
    };
}

static void ggml_backend_pyre_trace_provider(
        const ggml_backend_pyre_device_context * device_context,
        const char * format,
        ...) {
    if (!device_context->policy.trace_providers) {
        return;
    }

    std::fprintf(stderr, "ggml-pyre[%s]: ", device_context->name.c_str());
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
}

static void ggml_backend_pyre_trace_tensor(
        const ggml_backend_pyre_device_context * device_context,
        const char * label,
        const ggml_tensor * tensor) {
    if (!device_context->policy.trace_providers) {
        return;
    }
    if (!tensor) {
        ggml_backend_pyre_trace_provider(device_context, "%s=null\n", label);
        return;
    }

    ggml_backend_pyre_trace_provider(
        device_context,
        "%s type=%s ne=(%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ") "
        "nb=(%zu,%zu,%zu,%zu) contiguous=%d view=%d\n",
        label,
        ggml_type_name(tensor->type),
        tensor->ne[0],
        tensor->ne[1],
        tensor->ne[2],
        tensor->ne[3],
        static_cast<size_t>(tensor->nb[0]),
        static_cast<size_t>(tensor->nb[1]),
        static_cast<size_t>(tensor->nb[2]),
        static_cast<size_t>(tensor->nb[3]),
        ggml_is_contiguous(tensor) ? 1 : 0,
        tensor->view_src ? 1 : 0);
}

static const ggml_pyre_kernel_entry * ggml_backend_pyre_find_catalog_entry(const char * name) {
    size_t entry_count = 0;
    const ggml_pyre_kernel_entry * entries = ggml_pyre_kernel_catalog_entries(&entry_count);
    for (size_t i = 0; i < entry_count; ++i) {
        if (std::strcmp(entries[i].name, name) == 0) {
            return &entries[i];
        }
    }
    return nullptr;
}

static bool ggml_backend_pyre_load_catalog_provider(
        ggml_backend_pyre_device_context * device_context,
        const ggml_pyre_kernel_entry * entry,
        ggml_backend_pyre_op_provider * provider) {
    if (!entry || !entry->data || entry->data_size == 0) {
        return false;
    }

    pyre_executable_t executable = nullptr;
    if (!GGML_PYRE_CHECK(pyre_executable_load_data(
            device_context->device, entry->data, entry->data_size, entry->format, &executable))) {
        GGML_LOG_WARN("%s: failed to load Pyre catalog kernel %s for %s\n",
            __func__, entry->name, device_context->architecture.c_str());
        return false;
    }

    uint32_t export_ordinal = 0;
    pyre_executable_export_info_t export_info = {};
    bool ok = GGML_PYRE_CHECK(pyre_executable_lookup_export_by_name(
                  executable, entry->name, &export_ordinal)) &&
              GGML_PYRE_CHECK(pyre_executable_export_info(
                  executable, export_ordinal, &export_info)) &&
              export_info.binding_count == entry->binding_count &&
              export_info.constant_count > 0;
    if (!ok) {
        GGML_LOG_WARN(
            "%s: Pyre catalog kernel %s has unsupported ABI "
            "(bindings=%u expected=%u constants=%u constants_size=%u parameters=%u workgroup=%ux%ux%u)\n",
            __func__,
            entry->name,
            export_info.binding_count,
            entry->binding_count,
            export_info.constant_count,
            entry->constants_size,
            export_info.parameter_count,
            export_info.workgroup_size[0],
            export_info.workgroup_size[1],
            export_info.workgroup_size[2]);
        pyre_executable_release(executable);
        return false;
    }

    provider->kind = ggml_backend_pyre_provider_kind::direct_executable;
    provider->executable = executable;
    provider->export_ordinal = export_ordinal;
    provider->export_info = export_info;
    provider->export_info.workgroup_size[0] = entry->workgroup_size[0];
    provider->export_info.workgroup_size[1] = entry->workgroup_size[1];
    provider->export_info.workgroup_size[2] = entry->workgroup_size[2];
    return true;
}

static bool ggml_backend_pyre_load_rms_norm_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_rms_norm_f32"),
        &device_context->rms_norm_provider);
}

static bool ggml_backend_pyre_load_rms_norm_mul_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_rms_norm_mul_f32"),
        &device_context->rms_norm_mul_provider);
}

static bool ggml_backend_pyre_load_add_rms_norm_mul_broadcast_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_add_rms_norm_mul_f32_broadcast"),
        &device_context->add_rms_norm_mul_broadcast_provider);
}

static bool ggml_backend_pyre_load_add_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_add_f32"),
        &device_context->add_provider);
}

static bool ggml_backend_pyre_load_add_broadcast_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_add_f32_broadcast"),
        &device_context->add_broadcast_provider);
}

static bool ggml_backend_pyre_load_add_add_broadcast_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_add_add_f32_broadcast"),
        &device_context->add_add_broadcast_provider);
}

static bool ggml_backend_pyre_load_mul_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_mul_f32"),
        &device_context->mul_provider);
}

static bool ggml_backend_pyre_load_mul_broadcast_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_mul_f32_broadcast"),
        &device_context->mul_broadcast_provider);
}

static bool ggml_backend_pyre_load_div_broadcast_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_div_f32_broadcast"),
        &device_context->div_broadcast_provider);
}

static bool ggml_backend_pyre_load_scale_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_scale_f32"),
        &device_context->scale_provider);
}

static bool ggml_backend_pyre_load_set_rows_f32_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_set_rows_f32_f32"),
        &device_context->set_rows_f32_provider);
}

static bool ggml_backend_pyre_load_set_rows_f16_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_set_rows_f32_f16"),
        &device_context->set_rows_f16_provider);
}

static bool ggml_backend_pyre_load_silu_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_silu_f32"),
        &device_context->silu_provider);
}

static bool ggml_backend_pyre_load_sigmoid_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_sigmoid_f32"),
        &device_context->sigmoid_provider);
}

static bool ggml_backend_pyre_load_softplus_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_softplus_f32"),
        &device_context->softplus_provider);
}

static bool ggml_backend_pyre_load_swiglu_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_swiglu_f32"),
        &device_context->swiglu_provider);
}

static bool ggml_backend_pyre_load_sum_rows_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_sum_rows_f32"),
        &device_context->sum_rows_provider);
}

static bool ggml_backend_pyre_load_l2_norm_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_l2_norm_f32"),
        &device_context->l2_norm_provider);
}

static bool ggml_backend_pyre_load_clamp_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_clamp_f32"),
        &device_context->clamp_provider);
}

static bool ggml_backend_pyre_load_get_rows_f32_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_get_rows_f32"),
        &device_context->get_rows_f32_provider);
}

static bool ggml_backend_pyre_load_get_rows_q5_k_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_get_rows_q5_k_f32"),
        &device_context->get_rows_q5_k_provider);
}

static bool ggml_backend_pyre_load_concat_f32_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_concat_f32"),
        &device_context->concat_f32_provider);
}

static bool ggml_backend_pyre_load_soft_max_f32_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_soft_max_f32"),
        &device_context->soft_max_f32_provider);
}

static bool ggml_backend_pyre_load_soft_max_f32_mask_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_soft_max_f32_mask"),
        &device_context->soft_max_f32_mask_provider);
}

static bool ggml_backend_pyre_load_argsort_f32_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_argsort_f32_i32"),
        &device_context->argsort_f32_provider);
}

static bool ggml_backend_pyre_load_rope_f32_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_rope_f32"),
        &device_context->rope_f32_provider);
}

static bool ggml_backend_pyre_load_topk_moe_f32_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_topk_moe_f32"),
        &device_context->topk_moe_f32_provider);
}

static bool ggml_backend_pyre_load_topk_moe_f32_subgroup_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_topk_moe_f32_subgroup"),
        &device_context->topk_moe_f32_subgroup_provider);
}

static bool ggml_backend_pyre_load_ssm_conv_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_ssm_conv_f32"),
        &device_context->ssm_conv_provider);
}

static bool ggml_backend_pyre_load_gated_delta_net_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_gated_delta_net_f32"),
        &device_context->gated_delta_net_provider);
}

static bool ggml_backend_pyre_load_mul_mat_vec_f16_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_mul_mat_vec_f16_f32"),
        &device_context->mul_mat_vec_f16_provider);
}

static bool ggml_backend_pyre_load_mul_mat_vec_f16_batched_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_mul_mat_vec_f16_batched_f32"),
        &device_context->mul_mat_vec_f16_batched_provider);
}

static bool ggml_backend_pyre_load_mul_mat_vec_bf16_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_mul_mat_vec_bf16_f32"),
        &device_context->mul_mat_vec_bf16_provider);
}

static bool ggml_backend_pyre_load_mul_mat_vec_f32_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_mul_mat_vec_f32_f32"),
        &device_context->mul_mat_vec_f32_provider);
}

static bool ggml_backend_pyre_load_mul_mat_vec_f32_batched_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_mul_mat_vec_f32_batched_f32"),
        &device_context->mul_mat_vec_f32_batched_provider);
}

static bool ggml_backend_pyre_load_mul_mat_id_q4_k_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_mul_mat_id_q4_k_f32"),
        &device_context->mul_mat_id_q4_k_provider);
}

static bool ggml_backend_pyre_load_mul_mat_id_q4_k_q8_1_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_mul_mat_id_q4_k_q8_1_f32"),
        &device_context->mul_mat_id_q4_k_q8_1_provider);
}

static bool ggml_backend_pyre_load_mul_mat_id_q4_k_mul_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_mul_mat_id_q4_k_mul_f32"),
        &device_context->mul_mat_id_q4_k_mul_provider);
}

static bool ggml_backend_pyre_load_mul_mat_id_q4_k_mul_q8_1_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_mul_mat_id_q4_k_mul_q8_1_f32"),
        &device_context->mul_mat_id_q4_k_mul_q8_1_provider);
}

static bool ggml_backend_pyre_load_mul_mat_vec_q4_k_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_mul_mat_vec_q4_k_f32"),
        &device_context->mul_mat_vec_q4_k_provider);
}

static bool ggml_backend_pyre_load_quantize_q8_1_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_quantize_q8_1_f32"),
        &device_context->quantize_q8_1_provider);
}

static bool ggml_backend_pyre_load_mul_mat_vec_q4_k_q8_1_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_mul_mat_vec_q4_k_q8_1_f32"),
        &device_context->mul_mat_vec_q4_k_q8_1_provider);
}

static bool ggml_backend_pyre_load_mul_mat_vec_q5_k_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_mul_mat_vec_q5_k_f32"),
        &device_context->mul_mat_vec_q5_k_provider);
}

static bool ggml_backend_pyre_load_mul_mat_vec_q6_k_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_mul_mat_vec_q6_k_f32"),
        &device_context->mul_mat_vec_q6_k_provider);
}

static bool ggml_backend_pyre_load_mul_mat_vec_q8_0_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_mul_mat_vec_q8_0_f32"),
        &device_context->mul_mat_vec_q8_0_provider);
}

static bool ggml_backend_pyre_supports_rms_norm(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    return device_context->rms_norm_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           !device_context->policy.disable_rms_norm &&
           op->src[0] &&
           op->src[0]->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           op->src[0]->nb[0] == sizeof(float) &&
           op->nb[0] == sizeof(float) &&
           ggml_are_same_shape(op->src[0], op);
}

static bool ggml_backend_pyre_supports_rms_norm_mul(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * rms_norm,
        const ggml_tensor * mul) {
    if (device_context->rms_norm_mul_provider.kind !=
            ggml_backend_pyre_provider_kind::direct_executable ||
        !ggml_backend_pyre_supports_rms_norm(device_context, rms_norm) ||
        !mul ||
        mul->op != GGML_OP_MUL ||
        mul->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(rms_norm, mul) ||
        (mul->src[0] != rms_norm && mul->src[1] != rms_norm)) {
        return false;
    }

    const ggml_tensor * weight = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
    return weight &&
           weight->type == GGML_TYPE_F32 &&
           (weight->ne[0] == rms_norm->ne[0] || weight->ne[0] == 1) &&
           (weight->ne[1] == rms_norm->ne[1] || weight->ne[1] == 1) &&
           (weight->ne[2] == rms_norm->ne[2] || weight->ne[2] == 1) &&
           (weight->ne[3] == rms_norm->ne[3] || weight->ne[3] == 1) &&
           (weight->ne[0] == 1 || weight->nb[0] == sizeof(float)) &&
           mul->nb[0] == sizeof(float);
}

static bool ggml_backend_pyre_supports_binary_elementwise_f32(
        const ggml_backend_pyre_op_provider & provider,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return provider.kind == ggml_backend_pyre_provider_kind::direct_executable &&
           src0 && src1 &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(src0, src1) &&
           ggml_are_same_shape(src0, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op);
}

static bool ggml_backend_pyre_supports_mul_broadcast(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return device_context->mul_broadcast_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           src0 && src1 &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[0] == op->ne[0] &&
           src0->ne[1] == op->ne[1] &&
           src0->ne[2] == op->ne[2] &&
           src0->ne[3] == op->ne[3] &&
           (src1->ne[0] == src0->ne[0] || src1->ne[0] == 1) &&
           (src1->ne[1] == src0->ne[1] || src1->ne[1] == 1) &&
           (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
           (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1) &&
           (src0->ne[0] == 1 || src0->nb[0] == sizeof(float)) &&
           (src1->ne[0] == 1 || src1->nb[0] == sizeof(float)) &&
           op->nb[0] == sizeof(float);
}

static bool ggml_backend_pyre_supports_add_broadcast(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return device_context->add_broadcast_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           src0 && src1 &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[0] == op->ne[0] &&
           src0->ne[1] == op->ne[1] &&
           src0->ne[2] == op->ne[2] &&
           src0->ne[3] == op->ne[3] &&
           (src1->ne[0] == src0->ne[0] || src1->ne[0] == 1) &&
           (src1->ne[1] == src0->ne[1] || src1->ne[1] == 1) &&
           (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
           (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1) &&
           (src0->ne[0] == 1 || src0->nb[0] == sizeof(float)) &&
           (src1->ne[0] == 1 || src1->nb[0] == sizeof(float)) &&
           op->nb[0] == sizeof(float);
}

static bool ggml_backend_pyre_supports_add_add_broadcast(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * first,
        const ggml_tensor * second) {
    if (device_context->add_add_broadcast_provider.kind != ggml_backend_pyre_provider_kind::direct_executable ||
        !ggml_backend_pyre_supports_add_broadcast(device_context, first) ||
        !second ||
        second->op != GGML_OP_ADD ||
        second->type != GGML_TYPE_F32 ||
        (second->src[0] != first && second->src[1] != first) ||
        !ggml_are_same_shape(first, second) ||
        second->nb[0] != sizeof(float)) {
        return false;
    }

    const ggml_tensor * src2 = second->src[0] == first ? second->src[1] : second->src[0];
    return src2 &&
           src2->type == GGML_TYPE_F32 &&
           (src2->ne[0] == first->ne[0] || src2->ne[0] == 1) &&
           (src2->ne[1] == first->ne[1] || src2->ne[1] == 1) &&
           (src2->ne[2] == first->ne[2] || src2->ne[2] == 1) &&
           (src2->ne[3] == first->ne[3] || src2->ne[3] == 1) &&
           (src2->ne[0] == 1 || src2->nb[0] == sizeof(float));
}

static bool ggml_backend_pyre_supports_add_rms_norm_mul_broadcast(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * add,
        const ggml_tensor * rms_norm,
        const ggml_tensor * mul) {
    return device_context->add_rms_norm_mul_broadcast_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           !device_context->policy.disable_add_rms_norm_mul_fusion &&
           add &&
           rms_norm &&
           rms_norm->op == GGML_OP_RMS_NORM &&
           rms_norm->src[0] == add &&
           ggml_backend_pyre_supports_add_broadcast(device_context, add) &&
           ggml_backend_pyre_supports_rms_norm_mul(device_context, rms_norm, mul);
}

static bool ggml_backend_pyre_supports_div_broadcast(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return device_context->div_broadcast_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           src0 && src1 &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[0] == op->ne[0] &&
           src0->ne[1] == op->ne[1] &&
           src0->ne[2] == op->ne[2] &&
           src0->ne[3] == op->ne[3] &&
           (src1->ne[0] == src0->ne[0] || src1->ne[0] == 1) &&
           (src1->ne[1] == src0->ne[1] || src1->ne[1] == 1) &&
           (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
           (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1) &&
           (src0->ne[0] == 1 || src0->nb[0] == sizeof(float)) &&
           (src1->ne[0] == 1 || src1->nb[0] == sizeof(float)) &&
           op->nb[0] == sizeof(float);
}

static bool ggml_backend_pyre_supports_add(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_pyre_supports_binary_elementwise_f32(
        device_context->add_provider, op) ||
        ggml_backend_pyre_supports_add_broadcast(device_context, op);
}

static bool ggml_backend_pyre_supports_mul(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_pyre_supports_binary_elementwise_f32(
               device_context->mul_provider, op) ||
           ggml_backend_pyre_supports_mul_broadcast(device_context, op);
}

static bool ggml_backend_pyre_supports_scale(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    return device_context->scale_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           src0 &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(src0, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(op);
}

static const ggml_backend_pyre_op_provider * ggml_backend_pyre_unary_provider(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    if (op->op != GGML_OP_UNARY) {
        return nullptr;
    }
    switch (ggml_get_unary_op(op)) {
        case GGML_UNARY_OP_SILU:
            return &device_context->silu_provider;
        case GGML_UNARY_OP_SIGMOID:
            return &device_context->sigmoid_provider;
        case GGML_UNARY_OP_SOFTPLUS:
            return &device_context->softplus_provider;
        default:
            return nullptr;
    }
}

static bool ggml_backend_pyre_supports_unary_f32(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_backend_pyre_op_provider * provider =
        ggml_backend_pyre_unary_provider(device_context, op);
    return provider &&
           provider->kind == ggml_backend_pyre_provider_kind::direct_executable &&
           src0 &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(src0, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(op);
}

static bool ggml_backend_pyre_supports_swiglu_f32(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return device_context->swiglu_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           op->op == GGML_OP_GLU &&
           ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU &&
           src0 && src1 &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(src0, src1) &&
           ggml_are_same_shape(src0, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op);
}

static bool ggml_backend_pyre_supports_silu_mul_f32(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * silu,
        const ggml_tensor * mul) {
    if (device_context->swiglu_provider.kind != ggml_backend_pyre_provider_kind::direct_executable ||
        !silu || !mul ||
        silu->op != GGML_OP_UNARY ||
        ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU ||
        mul->op != GGML_OP_MUL ||
        mul->type != GGML_TYPE_F32 ||
        silu->type != GGML_TYPE_F32 ||
        !silu->src[0] ||
        silu->src[0]->type != GGML_TYPE_F32) {
        return false;
    }

    const ggml_tensor * other = mul->src[0] == silu ? mul->src[1] :
        (mul->src[1] == silu ? mul->src[0] : nullptr);
    return other &&
           other->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(silu, other) &&
           ggml_are_same_shape(silu, mul) &&
           ggml_is_contiguous(silu->src[0]) &&
           ggml_is_contiguous(other) &&
           ggml_is_contiguous(mul);
}

static bool ggml_backend_pyre_supports_cpy(
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return src0 && src1 &&
           src0->type == src1->type &&
           src0->type == op->type &&
           ggml_nelements(src0) == ggml_nelements(op) &&
           ggml_row_size(src0->type, src0->ne[0]) * ggml_nrows(src0) == ggml_nbytes(op) &&
           ggml_nbytes(src1) == ggml_nbytes(op) &&
           ggml_is_contiguous(op) &&
           (ggml_is_contiguous(src0) || src0->nb[0] == ggml_type_size(src0->type));
}

static bool ggml_backend_pyre_supports_cont(
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    return src0 &&
           src0->type == op->type &&
           ggml_nelements(src0) == ggml_nelements(op) &&
           ggml_row_size(src0->type, src0->ne[0]) * ggml_nrows(src0) == ggml_nbytes(op) &&
           ggml_is_contiguous(op) &&
           (ggml_is_contiguous(src0) || src0->nb[0] == ggml_type_size(src0->type));
}

static const ggml_backend_pyre_op_provider * ggml_backend_pyre_set_rows_provider(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    switch (op->type) {
        case GGML_TYPE_F32:
            return &device_context->set_rows_f32_provider;
        case GGML_TYPE_F16:
            return &device_context->set_rows_f16_provider;
        default:
            return nullptr;
    }
}

static bool ggml_backend_pyre_supports_set_rows(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];
    const ggml_backend_pyre_op_provider * provider =
        ggml_backend_pyre_set_rows_provider(device_context, op);
    return provider &&
           provider->kind == ggml_backend_pyre_provider_kind::direct_executable &&
           src0 && src1 && src2 &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_I64 &&
           src2->type == op->type &&
           op->ne[0] == src0->ne[0] &&
           op->ne[2] == src0->ne[2] &&
           op->ne[3] == src0->ne[3] &&
           src0->ne[1] == src1->ne[0] &&
           src0->ne[2] % src1->ne[1] == 0 &&
           src0->ne[3] % src1->ne[2] == 0 &&
           src1->ne[3] == 1 &&
           ggml_is_contiguous_rows(src0) &&
           ggml_is_contiguous_rows(op);
}

static bool ggml_backend_pyre_supports_sum_rows(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    return device_context->sum_rows_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           src0 &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           op->ne[0] == 1 &&
           op->ne[1] == src0->ne[1] &&
           op->ne[2] == src0->ne[2] &&
           op->ne[3] == src0->ne[3] &&
           src0->nb[0] == sizeof(float) &&
           op->nb[0] == sizeof(float);
}

static bool ggml_backend_pyre_supports_l2_norm(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    return device_context->l2_norm_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           src0 &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(src0, op) &&
           src0->nb[0] == sizeof(float) &&
           op->nb[0] == sizeof(float);
}

static bool ggml_backend_pyre_supports_clamp(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    return device_context->clamp_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           src0 &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(src0, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(op);
}

static bool ggml_backend_pyre_supports_get_rows_f32(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const bool has_provider =
        (src0 && src0->type == GGML_TYPE_F32 &&
         device_context->get_rows_f32_provider.kind == ggml_backend_pyre_provider_kind::direct_executable) ||
        (src0 && src0->type == GGML_TYPE_Q5_K &&
         device_context->get_rows_q5_k_provider.kind == ggml_backend_pyre_provider_kind::direct_executable);
    return has_provider &&
           src0 && src1 &&
           (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_Q5_K) &&
           src1->type == GGML_TYPE_I32 &&
           op->type == GGML_TYPE_F32 &&
           op->ne[0] == src0->ne[0] &&
           op->ne[2] == src1->ne[1] &&
           op->ne[3] == src1->ne[2] &&
           src0->ne[2] == src1->ne[1] &&
           src0->ne[3] == src1->ne[2] &&
           (src0->type == GGML_TYPE_Q5_K || src0->nb[0] == sizeof(float)) &&
           src1->nb[0] == sizeof(int32_t) &&
           op->nb[0] == sizeof(float);
}

static bool ggml_backend_pyre_supports_concat_f32(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const int32_t dim = ggml_get_op_params_i32(op, 0);
    return device_context->concat_f32_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           dim == 0 &&
           src0 && src1 &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[1] == src1->ne[1] &&
           src0->ne[1] == op->ne[1] &&
           src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src1->ne[2] == 1 && src1->ne[3] == 1 &&
           op->ne[2] == 1 && op->ne[3] == 1 &&
           op->ne[0] == src0->ne[0] + src1->ne[0] &&
           op->nb[0] == sizeof(float);
}

static bool ggml_backend_pyre_supports_soft_max_f32(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];
    float max_bias = 0.0f;
    std::memcpy(&max_bias, reinterpret_cast<const int32_t *>(op->op_params) + 1, sizeof(float));
    const ggml_backend_pyre_op_provider & provider =
        src1 ? device_context->soft_max_f32_mask_provider : device_context->soft_max_f32_provider;
    return provider.kind == ggml_backend_pyre_provider_kind::direct_executable &&
           src0 &&
           !src2 &&
           max_bias == 0.0f &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[0] <= 1024 &&
           ggml_are_same_shape(src0, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(op) &&
           (!src1 ||
            (src1->type == GGML_TYPE_F32 &&
             ggml_is_contiguous(src1) &&
             src1->ne[0] == src0->ne[0] &&
             src1->ne[1] >= src0->ne[1] &&
             src0->ne[2] % src1->ne[2] == 0 &&
             src0->ne[3] % src1->ne[3] == 0));
}

static bool ggml_backend_pyre_supports_argsort_f32(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const int64_t ncols = src0 ? src0->ne[0] : 0;
    return device_context->argsort_f32_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           src0 &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_I32 &&
           ncols > 0 &&
           ncols <= 256 &&
           (ncols & (ncols - 1)) == 0 &&
           ggml_are_same_shape(src0, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(op);
}

static bool ggml_backend_pyre_supports_topk_moe_f32(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * soft_max,
        const ggml_tensor * weights,
        const ggml_tensor * ids) {
    if (device_context->topk_moe_f32_provider.kind !=
            ggml_backend_pyre_provider_kind::direct_executable ||
        !soft_max || !weights || !ids ||
        soft_max->op != GGML_OP_SOFT_MAX ||
        !soft_max->src[0] ||
        soft_max->src[0]->type != GGML_TYPE_F32 ||
        soft_max->type != GGML_TYPE_F32 ||
        weights->type != GGML_TYPE_F32 ||
        ids->type != GGML_TYPE_I32 ||
        soft_max->src[1] || soft_max->src[2] ||
        soft_max->src[0]->ne[0] <= 0 ||
        soft_max->src[0]->ne[0] > 256 ||
        (soft_max->src[0]->ne[0] & (soft_max->src[0]->ne[0] - 1)) != 0 ||
        ggml_nrows(soft_max->src[0]) != 1 ||
        ggml_nelements(weights) > 32 ||
        ggml_nelements(weights) != ggml_nelements(ids) ||
        !ggml_is_contiguous(soft_max->src[0]) ||
        weights->nb[0] != sizeof(float) ||
        ids->nb[0] != sizeof(int32_t)) {
        return false;
    }

    float scale = 1.0f;
    float max_bias = 0.0f;
    std::memcpy(&scale, reinterpret_cast<const float *>(soft_max->op_params) + 0, sizeof(float));
    std::memcpy(&max_bias, reinterpret_cast<const float *>(soft_max->op_params) + 1, sizeof(float));
    return scale == 1.0f && max_bias == 0.0f;
}

static bool ggml_backend_pyre_supports_rope_f32(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];
    float ext_factor = 0.0f;
    std::memcpy(&ext_factor, reinterpret_cast<const int32_t *>(op->op_params) + 7, sizeof(float));
    const int32_t n_dims = ggml_get_op_params_i32(op, 1);
    const int32_t mode = ggml_get_op_params_i32(op, 2);
    const int32_t section0 = ggml_get_op_params_i32(op, 11);
    const int32_t section1 = ggml_get_op_params_i32(op, 12);
    const int32_t section2 = ggml_get_op_params_i32(op, 13);
    const int32_t section3 = ggml_get_op_params_i32(op, 14);
    return device_context->rope_f32_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           src0 &&
           src1 &&
           !src2 &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_I32 &&
           op->type == GGML_TYPE_F32 &&
           mode == GGML_ROPE_TYPE_IMROPE &&
           ext_factor == 0.0f &&
           n_dims > 0 &&
           n_dims <= src0->ne[0] &&
           (n_dims % 2) == 0 &&
           (src0->ne[0] % 2) == 0 &&
           section0 + section1 + section2 + section3 > 0 &&
           src1->ne[0] == src0->ne[2] * 4 &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op) &&
           ggml_are_same_shape(src0, op);
}

static bool ggml_backend_pyre_supports_ssm_conv(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return device_context->ssm_conv_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           src0 && src1 &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[0] == src1->ne[0] - 1 + op->ne[1] &&
           src0->ne[1] == src1->ne[1] &&
           src0->ne[2] == op->ne[2] &&
           src0->ne[3] == 1 &&
           src1->ne[2] == 1 && src1->ne[3] == 1 &&
           op->ne[0] == src1->ne[1] &&
           op->ne[3] == 1 &&
           src0->nb[0] == sizeof(float) &&
           src1->nb[0] == sizeof(float) &&
           op->nb[0] == sizeof(float);
}

static bool ggml_backend_pyre_supports_ssm_conv_silu(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * ssm,
        const ggml_tensor * silu) {
    return ggml_backend_pyre_supports_ssm_conv(device_context, ssm) &&
           silu &&
           silu->op == GGML_OP_UNARY &&
           ggml_get_unary_op(silu) == GGML_UNARY_OP_SILU &&
           silu->src[0] == ssm &&
           silu->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(silu, ssm) &&
           silu->nb[0] == sizeof(float);
}

static bool ggml_backend_pyre_supports_gated_delta_net(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];
    const ggml_tensor * g = op->src[3];
    const ggml_tensor * beta = op->src[4];
    const ggml_tensor * state = op->src[5];
    if (device_context->gated_delta_net_provider.kind != ggml_backend_pyre_provider_kind::direct_executable ||
        !q || !k || !v || !g || !beta || !state) {
        return false;
    }

    const int64_t S_v = v->ne[0];
    const int64_t H = v->ne[1];
    const int64_t n_tokens = v->ne[2];
    const int64_t n_seqs = v->ne[3];
    return q->type == GGML_TYPE_F32 &&
           k->type == GGML_TYPE_F32 &&
           v->type == GGML_TYPE_F32 &&
           g->type == GGML_TYPE_F32 &&
           beta->type == GGML_TYPE_F32 &&
           state->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           S_v > 0 && S_v <= 256 &&
           H > 0 &&
           n_tokens > 0 &&
           n_seqs > 0 &&
           q->ne[0] == S_v &&
           k->ne[0] == S_v &&
           q->ne[1] == k->ne[1] &&
           q->ne[1] > 0 &&
           H % q->ne[1] == 0 &&
           q->ne[2] == n_tokens &&
           k->ne[2] == n_tokens &&
           q->ne[3] > 0 &&
           k->ne[3] > 0 &&
           n_seqs % q->ne[3] == 0 &&
           n_seqs % k->ne[3] == 0 &&
           (g->ne[0] == 1 || g->ne[0] == S_v) &&
           g->ne[1] == H &&
           g->ne[2] == n_tokens &&
           g->ne[3] == n_seqs &&
           beta->ne[0] == 1 &&
           beta->ne[1] == H &&
           beta->ne[2] == n_tokens &&
           beta->ne[3] == n_seqs &&
           ggml_nelements(state) == S_v * S_v * H * n_seqs &&
           ggml_nelements(op) == S_v * H * n_tokens * n_seqs + S_v * S_v * H * n_seqs &&
           ggml_is_contiguous_rows(q) &&
           ggml_is_contiguous_rows(k) &&
           ggml_is_contiguous_rows(v) &&
           ggml_is_contiguous(g) &&
           ggml_is_contiguous(beta) &&
           ggml_is_contiguous(state) &&
           ggml_is_contiguous(op);
}

struct ggml_backend_pyre_rms_norm_constants {
    int64_t ncols;
    int64_t nrows;
    int64_t ne1;
    int64_t ne2;
    int64_t src_nb1;
    int64_t src_nb2;
    int64_t src_nb3;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t dst_nb3;
    float eps;
    int32_t _pad;
};

struct ggml_backend_pyre_rms_norm_mul_constants {
    int64_t ncols;
    int64_t nrows;
    int64_t ne1;
    int64_t ne2;
    int64_t src_nb1;
    int64_t src_nb2;
    int64_t src_nb3;
    int64_t weight_ne0;
    int64_t weight_ne1;
    int64_t weight_ne2;
    int64_t weight_ne3;
    int64_t weight_nb1;
    int64_t weight_nb2;
    int64_t weight_nb3;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t dst_nb3;
    float eps;
    int32_t _pad;
};

struct ggml_backend_pyre_add_rms_norm_mul_broadcast_constants {
    int64_t ncols;
    int64_t nrows;
    int64_t ne1;
    int64_t ne2;
    int64_t src1_ne0;
    int64_t src0_nb1;
    int64_t src0_nb2;
    int64_t src0_nb3;
    int64_t src1_nb1;
    int64_t src1_nb2;
    int64_t src1_nb3;
    int64_t weight_ne0;
    int64_t weight_ne1;
    int64_t weight_ne2;
    int64_t weight_ne3;
    int64_t weight_nb1;
    int64_t weight_nb2;
    int64_t weight_nb3;
    int64_t add_dst_nb1;
    int64_t add_dst_nb2;
    int64_t add_dst_nb3;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t dst_nb3;
    float eps;
    int32_t _pad;
};

static ggml_status ggml_backend_pyre_dispatch_rms_norm(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    pyre_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(dst, &bindings[1])) {
        GGML_LOG_ERROR("%s: RMS_NORM tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    float eps = 0.0f;
    std::memcpy(&eps, dst->op_params, sizeof(eps));

    ggml_backend_pyre_rms_norm_constants constants = {
        /* .ncols   = */ src0->ne[0],
        /* .nrows   = */ ggml_nrows(src0),
        /* .ne1     = */ src0->ne[1],
        /* .ne2     = */ src0->ne[2],
        /* .src_nb1 = */ static_cast<int64_t>(src0->nb[1]),
        /* .src_nb2 = */ static_cast<int64_t>(src0->nb[2]),
        /* .src_nb3 = */ static_cast<int64_t>(src0->nb[3]),
        /* .dst_nb1 = */ static_cast<int64_t>(dst->nb[1]),
        /* .dst_nb2 = */ static_cast<int64_t>(dst->nb[2]),
        /* .dst_nb3 = */ static_cast<int64_t>(dst->nb[3]),
        /* .eps     = */ eps,
        /* ._pad    = */ 0,
    };

    const auto & provider = context->device_context->rms_norm_provider;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>(constants.nrows),
            1,
            1,
        },
        /* .workgroup_size = */ {
            provider.export_info.workgroup_size[0] ?
                provider.export_info.workgroup_size[0] :
                GGML_PYRE_RMS_NORM_WORKGROUP_SIZE,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            2,
            PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->rms_norm_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_rms_norm_mul(
        ggml_backend_pyre_context * context,
        const ggml_tensor * rms_norm,
        const ggml_tensor * mul) {
    const ggml_tensor * src0 = rms_norm->src[0];
    const ggml_tensor * weight = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
    pyre_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(weight, &bindings[1]) ||
        !ggml_backend_pyre_tensor_buffer_ref(mul, &bindings[2])) {
        GGML_LOG_ERROR("%s: RMS_NORM_MUL tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    float eps = 0.0f;
    std::memcpy(&eps, rms_norm->op_params, sizeof(eps));

    ggml_backend_pyre_rms_norm_mul_constants constants = {
        /* .ncols      = */ src0->ne[0],
        /* .nrows      = */ ggml_nrows(src0),
        /* .ne1        = */ src0->ne[1],
        /* .ne2        = */ src0->ne[2],
        /* .src_nb1    = */ static_cast<int64_t>(src0->nb[1]),
        /* .src_nb2    = */ static_cast<int64_t>(src0->nb[2]),
        /* .src_nb3    = */ static_cast<int64_t>(src0->nb[3]),
        /* .weight_ne0 = */ weight->ne[0],
        /* .weight_ne1 = */ weight->ne[1],
        /* .weight_ne2 = */ weight->ne[2],
        /* .weight_ne3 = */ weight->ne[3],
        /* .weight_nb1 = */ static_cast<int64_t>(weight->nb[1]),
        /* .weight_nb2 = */ static_cast<int64_t>(weight->nb[2]),
        /* .weight_nb3 = */ static_cast<int64_t>(weight->nb[3]),
        /* .dst_nb1    = */ static_cast<int64_t>(mul->nb[1]),
        /* .dst_nb2    = */ static_cast<int64_t>(mul->nb[2]),
        /* .dst_nb3    = */ static_cast<int64_t>(mul->nb[3]),
        /* .eps        = */ eps,
        /* ._pad       = */ 0,
    };

    const auto & provider = context->device_context->rms_norm_mul_provider;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>(constants.nrows),
            1,
            1,
        },
        /* .workgroup_size = */ {
            provider.export_info.workgroup_size[0] ?
                provider.export_info.workgroup_size[0] :
                GGML_PYRE_RMS_NORM_WORKGROUP_SIZE,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            3,
            PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->rms_norm_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_add_rms_norm_mul_broadcast(
        ggml_backend_pyre_context * context,
        const ggml_tensor * add,
        const ggml_tensor * rms_norm,
        const ggml_tensor * mul) {
    const ggml_tensor * src0 = add->src[0];
    const ggml_tensor * src1 = add->src[1];
    const ggml_tensor * weight = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
    pyre_buffer_ref_t bindings[5] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_pyre_tensor_buffer_ref(add, &bindings[2]) ||
        !ggml_backend_pyre_tensor_buffer_ref(weight, &bindings[3]) ||
        !ggml_backend_pyre_tensor_buffer_ref(mul, &bindings[4])) {
        GGML_LOG_ERROR("%s: ADD_RMS_NORM_MUL tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    float eps = 0.0f;
    std::memcpy(&eps, rms_norm->op_params, sizeof(eps));

    ggml_backend_pyre_add_rms_norm_mul_broadcast_constants constants = {
        /* .ncols      = */ add->ne[0],
        /* .nrows      = */ ggml_nrows(add),
        /* .ne1        = */ add->ne[1],
        /* .ne2        = */ add->ne[2],
        /* .src1_ne0   = */ src1->ne[0],
        /* .src0_nb1   = */ static_cast<int64_t>(src0->nb[1]),
        /* .src0_nb2   = */ static_cast<int64_t>(src0->nb[2]),
        /* .src0_nb3   = */ static_cast<int64_t>(src0->nb[3]),
        /* .src1_nb1   = */ src1->ne[1] == 1 ? 0 : static_cast<int64_t>(src1->nb[1]),
        /* .src1_nb2   = */ src1->ne[2] == 1 ? 0 : static_cast<int64_t>(src1->nb[2]),
        /* .src1_nb3   = */ src1->ne[3] == 1 ? 0 : static_cast<int64_t>(src1->nb[3]),
        /* .weight_ne0 = */ weight->ne[0],
        /* .weight_ne1 = */ weight->ne[1],
        /* .weight_ne2 = */ weight->ne[2],
        /* .weight_ne3 = */ weight->ne[3],
        /* .weight_nb1 = */ static_cast<int64_t>(weight->nb[1]),
        /* .weight_nb2 = */ static_cast<int64_t>(weight->nb[2]),
        /* .weight_nb3 = */ static_cast<int64_t>(weight->nb[3]),
        /* .add_dst_nb1 = */ static_cast<int64_t>(add->nb[1]),
        /* .add_dst_nb2 = */ static_cast<int64_t>(add->nb[2]),
        /* .add_dst_nb3 = */ static_cast<int64_t>(add->nb[3]),
        /* .dst_nb1    = */ static_cast<int64_t>(mul->nb[1]),
        /* .dst_nb2    = */ static_cast<int64_t>(mul->nb[2]),
        /* .dst_nb3    = */ static_cast<int64_t>(mul->nb[3]),
        /* .eps        = */ eps,
        /* ._pad       = */ 0,
    };

    const auto & provider = context->device_context->add_rms_norm_mul_broadcast_provider;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>(constants.nrows),
            1,
            1,
        },
        /* .workgroup_size = */ {
            provider.export_info.workgroup_size[0] ?
                provider.export_info.workgroup_size[0] :
                GGML_PYRE_RMS_NORM_WORKGROUP_SIZE,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            5,
            PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->rms_norm_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

struct ggml_backend_pyre_elementwise_constants {
    int64_t n;
};

struct ggml_backend_pyre_mul_broadcast_constants {
    int64_t ne0;
    int64_t nrows;
    int64_t ne1;
    int64_t ne2;
    int64_t src1_ne0;
    int64_t src0_nb1;
    int64_t src0_nb2;
    int64_t src0_nb3;
    int64_t src1_nb1;
    int64_t src1_nb2;
    int64_t src1_nb3;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t dst_nb3;
};

struct ggml_backend_pyre_add_add_broadcast_constants {
    int64_t ne0;
    int64_t nrows;
    int64_t ne1;
    int64_t ne2;
    int64_t src1_ne0;
    int64_t src2_ne0;
    int64_t src0_nb1;
    int64_t src0_nb2;
    int64_t src0_nb3;
    int64_t src1_nb1;
    int64_t src1_nb2;
    int64_t src1_nb3;
    int64_t src2_nb1;
    int64_t src2_nb2;
    int64_t src2_nb3;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t dst_nb3;
};

struct ggml_backend_pyre_row_reduce_constants {
    int64_t ncols;
    int64_t nrows;
    int64_t ne1;
    int64_t ne2;
    int64_t src_nb1;
    int64_t src_nb2;
    int64_t src_nb3;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t dst_nb3;
    float eps;
    int32_t _pad;
};

struct ggml_backend_pyre_clamp_constants {
    int64_t n;
    float min_value;
    float max_value;
};

struct ggml_backend_pyre_get_rows_f32_constants {
    int64_t nc;
    int64_t nr;
    int64_t src0_nb1;
    int64_t src0_nb2;
    int64_t src0_nb3;
    int64_t idx_nb0;
    int64_t idx_nb1;
    int64_t idx_nb2;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t dst_nb3;
    int64_t ne10;
    int64_t ne11;
};

struct ggml_backend_pyre_concat_f32_constants {
    int64_t ne0;
    int64_t ne1;
    int64_t src0_ne0;
    int64_t src0_nb0;
    int64_t src0_nb1;
    int64_t src1_nb0;
    int64_t src1_nb1;
    int64_t dst_nb0;
    int64_t dst_nb1;
};

struct ggml_backend_pyre_soft_max_f32_constants {
    int64_t ncols;
    int64_t nrows;
    int64_t ne01;
    int64_t ne02;
    int64_t mask_nb1;
    int64_t mask_nb2;
    int64_t mask_nb3;
    int64_t mask_ne2;
    int64_t mask_ne3;
    float scale;
    int32_t _pad;
};

struct ggml_backend_pyre_argsort_f32_constants {
    int64_t ncols;
    int64_t nrows;
    int32_t order;
    int32_t ncols_pad;
};

struct ggml_backend_pyre_topk_moe_f32_constants {
    int64_t n_experts;
    int64_t n_rows;
    int64_t n_expert_used;
    int64_t logits_nb1;
    int64_t weights_nb1;
    int64_t ids_nb1;
    float scale;
    float clamp_min;
    float clamp_max;
    int32_t with_norm;
};

struct ggml_backend_pyre_rope_f32_constants {
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t nrows;
    int64_t src_s1;
    int64_t src_s2;
    int64_t src_s3;
    int64_t dst_s1;
    int64_t dst_s2;
    int64_t dst_s3;
    int32_t n_dims;
    int32_t mode;
    int32_t section0;
    int32_t section1;
    int32_t section2;
    int32_t section3;
    float freq_base;
    float freq_scale;
    float attn_factor;
    float _pad;
};

struct ggml_backend_pyre_scale_constants {
    int64_t n;
    float scale;
    float bias;
};

struct ggml_backend_pyre_set_rows_constants {
    int64_t nc;
    int64_t nr;
    int64_t ne02;
    int64_t ne03;
    int64_t ne1;
    int64_t ne11;
    int64_t ne12;
    int64_t src0_nb1;
    int64_t src0_nb2;
    int64_t src0_nb3;
    int64_t idx_nb0;
    int64_t idx_nb1;
    int64_t idx_nb2;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t dst_nb3;
};

struct ggml_backend_pyre_ssm_conv_constants {
    int64_t d_conv;
    int64_t conv_width;
    int64_t d_inner;
    int64_t n_tokens;
    int64_t n_seqs;
    int64_t src0_nb1;
    int64_t src0_nb2;
    int64_t weight_nb1;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int32_t apply_silu;
    int32_t pad;
};

struct ggml_backend_pyre_gated_delta_net_constants {
    int64_t S_v;
    int64_t H;
    int64_t n_tokens;
    int64_t n_seqs;
    int64_t neq1;
    int64_t nek1;
    int64_t rq3;
    int64_t rk3;
    int64_t q_nb1;
    int64_t q_nb2;
    int64_t q_nb3;
    int64_t k_nb1;
    int64_t k_nb2;
    int64_t k_nb3;
    int64_t v_nb1;
    int64_t v_nb2;
    int64_t v_nb3;
    int64_t g_ne0;
    int64_t g_nb1;
    int64_t g_nb2;
    int64_t g_nb3;
    int64_t beta_nb1;
    int64_t beta_nb2;
    int64_t beta_nb3;
    float scale;
    int32_t _pad;
};

static ggml_status ggml_backend_pyre_dispatch_binary_elementwise_f32(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst,
        const ggml_backend_pyre_op_provider & provider,
        const char * op_name) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    pyre_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_pyre_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("%s: %s tensor is not backed by a PYRE buffer\n", __func__, op_name);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_pyre_elementwise_constants constants = {
        /* .n = */ ggml_nelements(dst),
    };

    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((constants.n + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ {
            workgroup_size,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            3,
            PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->elementwise_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_mul_broadcast_f32(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    pyre_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_pyre_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("%s: broadcast MUL tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_pyre_mul_broadcast_constants constants = {
        /* .ne0      = */ dst->ne[0],
        /* .nrows    = */ ggml_nrows(dst),
        /* .ne1      = */ dst->ne[1],
        /* .ne2      = */ dst->ne[2],
        /* .src1_ne0 = */ src1->ne[0],
        /* .src0_nb1 = */ static_cast<int64_t>(src0->nb[1]),
        /* .src0_nb2 = */ static_cast<int64_t>(src0->nb[2]),
        /* .src0_nb3 = */ static_cast<int64_t>(src0->nb[3]),
        /* .src1_nb1 = */ src1->ne[1] == 1 ? 0 : static_cast<int64_t>(src1->nb[1]),
        /* .src1_nb2 = */ src1->ne[2] == 1 ? 0 : static_cast<int64_t>(src1->nb[2]),
        /* .src1_nb3 = */ src1->ne[3] == 1 ? 0 : static_cast<int64_t>(src1->nb[3]),
        /* .dst_nb1  = */ static_cast<int64_t>(dst->nb[1]),
        /* .dst_nb2  = */ static_cast<int64_t>(dst->nb[2]),
        /* .dst_nb3  = */ static_cast<int64_t>(dst->nb[3]),
    };

    const auto & provider = context->device_context->mul_broadcast_provider;
    const int64_t n = constants.ne0 * constants.nrows;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((n + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ {
            workgroup_size,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            3,
            PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->elementwise_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_add_broadcast_f32(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    pyre_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_pyre_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("%s: broadcast ADD tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_pyre_mul_broadcast_constants constants = {
        /* .ne0      = */ dst->ne[0],
        /* .nrows    = */ ggml_nrows(dst),
        /* .ne1      = */ dst->ne[1],
        /* .ne2      = */ dst->ne[2],
        /* .src1_ne0 = */ src1->ne[0],
        /* .src0_nb1 = */ static_cast<int64_t>(src0->nb[1]),
        /* .src0_nb2 = */ static_cast<int64_t>(src0->nb[2]),
        /* .src0_nb3 = */ static_cast<int64_t>(src0->nb[3]),
        /* .src1_nb1 = */ src1->ne[1] == 1 ? 0 : static_cast<int64_t>(src1->nb[1]),
        /* .src1_nb2 = */ src1->ne[2] == 1 ? 0 : static_cast<int64_t>(src1->nb[2]),
        /* .src1_nb3 = */ src1->ne[3] == 1 ? 0 : static_cast<int64_t>(src1->nb[3]),
        /* .dst_nb1  = */ static_cast<int64_t>(dst->nb[1]),
        /* .dst_nb2  = */ static_cast<int64_t>(dst->nb[2]),
        /* .dst_nb3  = */ static_cast<int64_t>(dst->nb[3]),
    };

    const auto & provider = context->device_context->add_broadcast_provider;
    const int64_t n = constants.ne0 * constants.nrows;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((n + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ {
            workgroup_size,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            3,
            PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->elementwise_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_add_add_broadcast_f32(
        ggml_backend_pyre_context * context,
        const ggml_tensor * first,
        const ggml_tensor * second) {
    const ggml_tensor * src0 = first->src[0];
    const ggml_tensor * src1 = first->src[1];
    const ggml_tensor * src2 = second->src[0] == first ? second->src[1] : second->src[0];
    pyre_buffer_ref_t bindings[4] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_pyre_tensor_buffer_ref(src2, &bindings[2]) ||
        !ggml_backend_pyre_tensor_buffer_ref(second, &bindings[3])) {
        GGML_LOG_ERROR("%s: fused broadcast ADD tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_pyre_add_add_broadcast_constants constants = {
        /* .ne0      = */ second->ne[0],
        /* .nrows    = */ ggml_nrows(second),
        /* .ne1      = */ second->ne[1],
        /* .ne2      = */ second->ne[2],
        /* .src1_ne0 = */ src1->ne[0],
        /* .src2_ne0 = */ src2->ne[0],
        /* .src0_nb1 = */ static_cast<int64_t>(src0->nb[1]),
        /* .src0_nb2 = */ static_cast<int64_t>(src0->nb[2]),
        /* .src0_nb3 = */ static_cast<int64_t>(src0->nb[3]),
        /* .src1_nb1 = */ src1->ne[1] == 1 ? 0 : static_cast<int64_t>(src1->nb[1]),
        /* .src1_nb2 = */ src1->ne[2] == 1 ? 0 : static_cast<int64_t>(src1->nb[2]),
        /* .src1_nb3 = */ src1->ne[3] == 1 ? 0 : static_cast<int64_t>(src1->nb[3]),
        /* .src2_nb1 = */ src2->ne[1] == 1 ? 0 : static_cast<int64_t>(src2->nb[1]),
        /* .src2_nb2 = */ src2->ne[2] == 1 ? 0 : static_cast<int64_t>(src2->nb[2]),
        /* .src2_nb3 = */ src2->ne[3] == 1 ? 0 : static_cast<int64_t>(src2->nb[3]),
        /* .dst_nb1  = */ static_cast<int64_t>(second->nb[1]),
        /* .dst_nb2  = */ static_cast<int64_t>(second->nb[2]),
        /* .dst_nb3  = */ static_cast<int64_t>(second->nb[3]),
    };

    const auto & provider = context->device_context->add_add_broadcast_provider;
    const int64_t n = constants.ne0 * constants.nrows;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((n + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ {
            workgroup_size,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            4,
            PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->elementwise_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_div_broadcast_f32(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    pyre_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_pyre_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("%s: broadcast DIV tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_pyre_mul_broadcast_constants constants = {
        /* .ne0      = */ dst->ne[0],
        /* .nrows    = */ ggml_nrows(dst),
        /* .ne1      = */ dst->ne[1],
        /* .ne2      = */ dst->ne[2],
        /* .src1_ne0 = */ src1->ne[0],
        /* .src0_nb1 = */ static_cast<int64_t>(src0->nb[1]),
        /* .src0_nb2 = */ static_cast<int64_t>(src0->nb[2]),
        /* .src0_nb3 = */ static_cast<int64_t>(src0->nb[3]),
        /* .src1_nb1 = */ src1->ne[1] == 1 ? 0 : static_cast<int64_t>(src1->nb[1]),
        /* .src1_nb2 = */ src1->ne[2] == 1 ? 0 : static_cast<int64_t>(src1->nb[2]),
        /* .src1_nb3 = */ src1->ne[3] == 1 ? 0 : static_cast<int64_t>(src1->nb[3]),
        /* .dst_nb1  = */ static_cast<int64_t>(dst->nb[1]),
        /* .dst_nb2  = */ static_cast<int64_t>(dst->nb[2]),
        /* .dst_nb3  = */ static_cast<int64_t>(dst->nb[3]),
    };

    const auto & provider = context->device_context->div_broadcast_provider;
    const int64_t n = constants.ne0 * constants.nrows;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((n + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ {
            workgroup_size,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            3,
            PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->elementwise_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_unary_f32(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst,
        const ggml_backend_pyre_op_provider & provider,
        const char * op_name) {
    const ggml_tensor * src0 = dst->src[0];
    pyre_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(dst, &bindings[1])) {
        GGML_LOG_ERROR("%s: %s tensor is not backed by a PYRE buffer\n", __func__, op_name);
        return GGML_STATUS_FAILED;
    }

    const int64_t n = ggml_nelements(dst);
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((n + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ {
            workgroup_size,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &n,
            sizeof(n),
            bindings,
            2,
            PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->unary_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_swiglu_f32(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    pyre_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_pyre_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("%s: SWIGLU tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    const int64_t n = ggml_nelements(dst);
    const auto & provider = context->device_context->swiglu_provider;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((n + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ {
            workgroup_size,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &n,
            sizeof(n),
            bindings,
            3,
            PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->unary_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_silu_mul_f32(
        ggml_backend_pyre_context * context,
        const ggml_tensor * silu,
        const ggml_tensor * mul) {
    const ggml_tensor * src0 = silu->src[0];
    const ggml_tensor * src1 = mul->src[0] == silu ? mul->src[1] : mul->src[0];
    pyre_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_pyre_tensor_buffer_ref(mul, &bindings[2])) {
        GGML_LOG_ERROR("%s: SILU_MUL tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    const int64_t n = ggml_nelements(mul);
    const auto & provider = context->device_context->swiglu_provider;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((n + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ {
            workgroup_size,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &n,
            sizeof(n),
            bindings,
            3,
            PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->unary_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_scale_f32(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    pyre_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(dst, &bindings[1])) {
        GGML_LOG_ERROR("%s: SCALE tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_pyre_scale_constants constants = {
        /* .n     = */ ggml_nelements(dst),
        /* .scale = */ 0.0f,
        /* .bias  = */ 0.0f,
    };
    const uint8_t * op_params = reinterpret_cast<const uint8_t *>(dst->op_params);
    std::memcpy(&constants.scale, op_params, sizeof(float));
    std::memcpy(&constants.bias, op_params + sizeof(float), sizeof(float));

    const auto & provider = context->device_context->scale_provider;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((constants.n + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ {
            workgroup_size,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            2,
            PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->elementwise_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_cpy(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    pyre_buffer_ref_t src_ref = {};
    pyre_buffer_ref_t dst_ref = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &src_ref) ||
        !ggml_backend_pyre_tensor_buffer_ref(dst, &dst_ref)) {
        GGML_LOG_ERROR("%s: CPY tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    const size_t size = ggml_nbytes(dst);
    if (ggml_is_contiguous(src0)) {
        if (src_ref.buffer != dst_ref.buffer || src_ref.offset != dst_ref.offset) {
            if (!GGML_PYRE_CHECK(pyre_stream_copy_buffer(
                    context->stream,
                    src_ref.buffer,
                    src_ref.offset,
                    dst_ref.buffer,
                    dst_ref.offset,
                    size))) {
                return GGML_STATUS_FAILED;
            }
            if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
                return GGML_STATUS_FAILED;
            }
        }
    } else {
        const size_t row_size = ggml_row_size(src0->type, src0->ne[0]);
        size_t dst_offset = dst_ref.offset;
        for (int64_t i3 = 0; i3 < src0->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < src0->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < src0->ne[1]; ++i1) {
                    const size_t src_offset =
                        src_ref.offset +
                        static_cast<size_t>(i1) * src0->nb[1] +
                        static_cast<size_t>(i2) * src0->nb[2] +
                        static_cast<size_t>(i3) * src0->nb[3];
                    if (!GGML_PYRE_CHECK(pyre_stream_copy_buffer(
                            context->stream,
                            src_ref.buffer,
                            src_offset,
                            dst_ref.buffer,
                            dst_offset,
                            row_size))) {
                        return GGML_STATUS_FAILED;
                    }
                    dst_offset += row_size;
                }
            }
        }
        if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
            return GGML_STATUS_FAILED;
        }
    }
    context->copy_count++;
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_set_rows(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    pyre_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_pyre_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("%s: SET_ROWS tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_pyre_set_rows_constants constants = {
        /* .nc       = */ src0->ne[0],
        /* .nr       = */ src0->ne[1],
        /* .ne02     = */ src0->ne[2],
        /* .ne03     = */ src0->ne[3],
        /* .ne1      = */ dst->ne[1],
        /* .ne11     = */ src1->ne[1],
        /* .ne12     = */ src1->ne[2],
        /* .src0_nb1 = */ static_cast<int64_t>(src0->nb[1]),
        /* .src0_nb2 = */ static_cast<int64_t>(src0->nb[2]),
        /* .src0_nb3 = */ static_cast<int64_t>(src0->nb[3]),
        /* .idx_nb0  = */ static_cast<int64_t>(src1->nb[0]),
        /* .idx_nb1  = */ static_cast<int64_t>(src1->nb[1]),
        /* .idx_nb2  = */ static_cast<int64_t>(src1->nb[2]),
        /* .dst_nb1  = */ static_cast<int64_t>(dst->nb[1]),
        /* .dst_nb2  = */ static_cast<int64_t>(dst->nb[2]),
        /* .dst_nb3  = */ static_cast<int64_t>(dst->nb[3]),
    };

    const ggml_backend_pyre_op_provider * provider =
        ggml_backend_pyre_set_rows_provider(context->device_context, dst);
    if (!provider) {
        return GGML_STATUS_FAILED;
    }

    const int64_t total = constants.nc * constants.nr * constants.ne02 * constants.ne03;
    const uint32_t workgroup_size = provider->export_info.workgroup_size[0] ?
        provider->export_info.workgroup_size[0] : 256;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((total + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ {
            workgroup_size,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream,
            provider->executable,
            provider->export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            3,
            PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->set_rows_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_sum_rows(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    pyre_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(dst, &bindings[1])) {
        GGML_LOG_ERROR("%s: SUM_ROWS tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_pyre_row_reduce_constants constants = {
        /* .ncols   = */ src0->ne[0],
        /* .nrows   = */ ggml_nrows(src0),
        /* .ne1     = */ src0->ne[1],
        /* .ne2     = */ src0->ne[2],
        /* .src_nb1 = */ static_cast<int64_t>(src0->nb[1]),
        /* .src_nb2 = */ static_cast<int64_t>(src0->nb[2]),
        /* .src_nb3 = */ static_cast<int64_t>(src0->nb[3]),
        /* .dst_nb1 = */ static_cast<int64_t>(dst->nb[1]),
        /* .dst_nb2 = */ static_cast<int64_t>(dst->nb[2]),
        /* .dst_nb3 = */ static_cast<int64_t>(dst->nb[3]),
        /* .eps     = */ 0.0f,
        /* ._pad    = */ 0,
    };

    const auto & provider = context->device_context->sum_rows_provider;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ { static_cast<uint32_t>(constants.nrows), 1, 1 },
        /* .workgroup_size = */ {
            provider.export_info.workgroup_size[0] ? provider.export_info.workgroup_size[0] : 256,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream, provider.executable, provider.export_ordinal, &config,
            &constants, sizeof(constants), bindings, 2, PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->reduction_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_l2_norm(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    pyre_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(dst, &bindings[1])) {
        GGML_LOG_ERROR("%s: L2_NORM tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    float eps = 0.0f;
    std::memcpy(&eps, dst->op_params, sizeof(eps));
    ggml_backend_pyre_row_reduce_constants constants = {
        /* .ncols   = */ src0->ne[0],
        /* .nrows   = */ ggml_nrows(src0),
        /* .ne1     = */ src0->ne[1],
        /* .ne2     = */ src0->ne[2],
        /* .src_nb1 = */ static_cast<int64_t>(src0->nb[1]),
        /* .src_nb2 = */ static_cast<int64_t>(src0->nb[2]),
        /* .src_nb3 = */ static_cast<int64_t>(src0->nb[3]),
        /* .dst_nb1 = */ static_cast<int64_t>(dst->nb[1]),
        /* .dst_nb2 = */ static_cast<int64_t>(dst->nb[2]),
        /* .dst_nb3 = */ static_cast<int64_t>(dst->nb[3]),
        /* .eps     = */ eps,
        /* ._pad    = */ 0,
    };

    const auto & provider = context->device_context->l2_norm_provider;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ { static_cast<uint32_t>(constants.nrows), 1, 1 },
        /* .workgroup_size = */ {
            provider.export_info.workgroup_size[0] ? provider.export_info.workgroup_size[0] : 256,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream, provider.executable, provider.export_ordinal, &config,
            &constants, sizeof(constants), bindings, 2, PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->reduction_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_clamp(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    pyre_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(dst, &bindings[1])) {
        GGML_LOG_ERROR("%s: CLAMP tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_pyre_clamp_constants constants = {
        /* .n         = */ ggml_nelements(dst),
        /* .min_value = */ 0.0f,
        /* .max_value = */ 0.0f,
    };
    std::memcpy(&constants.min_value, reinterpret_cast<const float *>(dst->op_params) + 0, sizeof(float));
    std::memcpy(&constants.max_value, reinterpret_cast<const float *>(dst->op_params) + 1, sizeof(float));

    const auto & provider = context->device_context->clamp_provider;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((constants.n + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream, provider.executable, provider.export_ordinal, &config,
            &constants, sizeof(constants), bindings, 2, PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->elementwise_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_get_rows_f32(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst) {
    if (ggml_nelements(dst) == 0) {
        context->get_rows_count++;
        return GGML_STATUS_SUCCESS;
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    pyre_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_pyre_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("%s: GET_ROWS tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_pyre_get_rows_f32_constants constants = {
        /* .nc       = */ src0->ne[0],
        /* .nr       = */ ggml_nelements(src1),
        /* .src0_nb1 = */ static_cast<int64_t>(src0->nb[1]),
        /* .src0_nb2 = */ static_cast<int64_t>(src0->nb[2]),
        /* .src0_nb3 = */ static_cast<int64_t>(src0->nb[3]),
        /* .idx_nb0  = */ static_cast<int64_t>(src1->nb[0]),
        /* .idx_nb1  = */ static_cast<int64_t>(src1->nb[1]),
        /* .idx_nb2  = */ static_cast<int64_t>(src1->nb[2]),
        /* .dst_nb1  = */ static_cast<int64_t>(dst->nb[1]),
        /* .dst_nb2  = */ static_cast<int64_t>(dst->nb[2]),
        /* .dst_nb3  = */ static_cast<int64_t>(dst->nb[3]),
        /* .ne10     = */ src1->ne[0],
        /* .ne11     = */ src1->ne[1],
    };

    const auto & provider = src0->type == GGML_TYPE_Q5_K ?
        context->device_context->get_rows_q5_k_provider :
        context->device_context->get_rows_f32_provider;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((constants.nc + workgroup_size - 1) / workgroup_size),
            static_cast<uint32_t>(constants.nr),
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream, provider.executable, provider.export_ordinal, &config,
            &constants, sizeof(constants), bindings, 3, PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->get_rows_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_concat_f32(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    pyre_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_pyre_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("%s: CONCAT tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_pyre_concat_f32_constants constants = {
        /* .ne0      = */ dst->ne[0],
        /* .ne1      = */ dst->ne[1],
        /* .src0_ne0 = */ src0->ne[0],
        /* .src0_nb0 = */ static_cast<int64_t>(src0->nb[0]),
        /* .src0_nb1 = */ static_cast<int64_t>(src0->nb[1]),
        /* .src1_nb0 = */ static_cast<int64_t>(src1->nb[0]),
        /* .src1_nb1 = */ static_cast<int64_t>(src1->nb[1]),
        /* .dst_nb0  = */ static_cast<int64_t>(dst->nb[0]),
        /* .dst_nb1  = */ static_cast<int64_t>(dst->nb[1]),
    };

    const auto & provider = context->device_context->concat_f32_provider;
    const int64_t n = constants.ne0 * constants.ne1;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((n + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream, provider.executable, provider.export_ordinal, &config,
            &constants, sizeof(constants), bindings, 3, PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->concat_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static int32_t ggml_backend_pyre_next_power_of_2(int64_t value) {
    int32_t result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

static ggml_status ggml_backend_pyre_dispatch_soft_max_f32(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    pyre_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(dst, src1 ? &bindings[2] : &bindings[1])) {
        GGML_LOG_ERROR("%s: SOFT_MAX tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }
    if (src1 && !ggml_backend_pyre_tensor_buffer_ref(src1, &bindings[1])) {
        GGML_LOG_ERROR("%s: SOFT_MAX mask tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    float scale = 1.0f;
    std::memcpy(&scale, reinterpret_cast<const int32_t *>(dst->op_params), sizeof(float));
    ggml_backend_pyre_soft_max_f32_constants constants = {
        /* .ncols    = */ src0->ne[0],
        /* .nrows    = */ ggml_nrows(src0),
        /* .ne01     = */ src0->ne[1],
        /* .ne02     = */ src0->ne[2],
        /* .mask_nb1 = */ src1 ? static_cast<int64_t>(src1->nb[1]) : 0,
        /* .mask_nb2 = */ src1 ? static_cast<int64_t>(src1->nb[2]) : 0,
        /* .mask_nb3 = */ src1 ? static_cast<int64_t>(src1->nb[3]) : 0,
        /* .mask_ne2 = */ src1 ? src1->ne[2] : 1,
        /* .mask_ne3 = */ src1 ? src1->ne[3] : 1,
        /* .scale    = */ scale,
        /* ._pad     = */ 0,
    };

    const auto & provider = src1 ?
        context->device_context->soft_max_f32_mask_provider :
        context->device_context->soft_max_f32_provider;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>(constants.nrows),
            1,
            1,
        },
        /* .workgroup_size = */ { 256, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream, provider.executable, provider.export_ordinal, &config,
            &constants, sizeof(constants), bindings, src1 ? 3 : 2, PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->soft_max_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_argsort_f32(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    pyre_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(dst, &bindings[1])) {
        GGML_LOG_ERROR("%s: ARGSORT tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_pyre_argsort_f32_constants constants = {
        /* .ncols     = */ src0->ne[0],
        /* .nrows     = */ ggml_nrows(src0),
        /* .order     = */ ggml_get_op_params_i32(dst, 0),
        /* .ncols_pad = */ ggml_backend_pyre_next_power_of_2(src0->ne[0]),
    };
    const auto & provider = context->device_context->argsort_f32_provider;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>(constants.nrows),
            1,
            1,
        },
        /* .workgroup_size = */ { 256, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream, provider.executable, provider.export_ordinal, &config,
            &constants, sizeof(constants), bindings, 2, PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->argsort_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_topk_moe_f32(
        ggml_backend_pyre_context * context,
        const ggml_tensor * soft_max,
        const ggml_tensor * weights,
        const ggml_tensor * ids,
        const ggml_tensor * clamp) {
    const ggml_tensor * logits = soft_max->src[0];
    pyre_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(logits, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(weights, &bindings[1]) ||
        !ggml_backend_pyre_tensor_buffer_ref(ids, &bindings[2])) {
        GGML_LOG_ERROR("%s: TOPK_MOE tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    float scale = 1.0f;
    std::memcpy(&scale, reinterpret_cast<const float *>(soft_max->op_params) + 0, sizeof(float));
    float clamp_min = -std::numeric_limits<float>::infinity();
    float clamp_max = std::numeric_limits<float>::infinity();
    if (clamp) {
        std::memcpy(&clamp_min, reinterpret_cast<const float *>(clamp->op_params) + 0, sizeof(float));
        std::memcpy(&clamp_max, reinterpret_cast<const float *>(clamp->op_params) + 1, sizeof(float));
    }

    ggml_backend_pyre_topk_moe_f32_constants constants = {
        /* .n_experts     = */ logits->ne[0],
        /* .n_rows        = */ ggml_nrows(logits),
        /* .n_expert_used = */ ggml_nelements(weights) / ggml_nrows(logits),
        /* .logits_nb1    = */ static_cast<int64_t>(logits->nb[1]),
        /* .weights_nb1   = */ static_cast<int64_t>(weights->nb[1]),
        /* .ids_nb1       = */ static_cast<int64_t>(ids->nb[1]),
        /* .scale         = */ scale,
        /* .clamp_min     = */ clamp_min,
        /* .clamp_max     = */ clamp_max,
        /* .with_norm     = */ clamp ? 1 : 0,
    };

    const bool use_subgroup =
        !context->device_context->policy.disable_topk_subgroup &&
        context->device_context->topk_moe_f32_subgroup_provider.kind ==
            ggml_backend_pyre_provider_kind::direct_executable;
    const auto & provider = use_subgroup ?
        context->device_context->topk_moe_f32_subgroup_provider :
        context->device_context->topk_moe_f32_provider;
    const uint32_t workgroup_size_x = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 64;
    const uint32_t workgroup_size_y = provider.export_info.workgroup_size[1] ?
        provider.export_info.workgroup_size[1] : 1;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((constants.n_rows + workgroup_size_y - 1) / workgroup_size_y),
            1,
            1,
        },
        /* .workgroup_size = */ { workgroup_size_x, workgroup_size_y, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream, provider.executable, provider.export_ordinal, &config,
            &constants, sizeof(constants), bindings, 3, PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->topk_moe_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_rope_f32(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    pyre_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_pyre_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("%s: ROPE tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    float freq_base = 0.0f;
    float freq_scale = 0.0f;
    float attn_factor = 0.0f;
    std::memcpy(&freq_base, reinterpret_cast<const int32_t *>(dst->op_params) + 5, sizeof(float));
    std::memcpy(&freq_scale, reinterpret_cast<const int32_t *>(dst->op_params) + 6, sizeof(float));
    std::memcpy(&attn_factor, reinterpret_cast<const int32_t *>(dst->op_params) + 8, sizeof(float));

    ggml_backend_pyre_rope_f32_constants constants = {
        /* .ne00        = */ src0->ne[0],
        /* .ne01        = */ src0->ne[1],
        /* .ne02        = */ src0->ne[2],
        /* .nrows       = */ ggml_nrows(src0),
        /* .src_s1      = */ static_cast<int64_t>(src0->nb[1] / sizeof(float)),
        /* .src_s2      = */ static_cast<int64_t>(src0->nb[2] / sizeof(float)),
        /* .src_s3      = */ static_cast<int64_t>(src0->nb[3] / sizeof(float)),
        /* .dst_s1      = */ static_cast<int64_t>(dst->nb[1] / sizeof(float)),
        /* .dst_s2      = */ static_cast<int64_t>(dst->nb[2] / sizeof(float)),
        /* .dst_s3      = */ static_cast<int64_t>(dst->nb[3] / sizeof(float)),
        /* .n_dims      = */ ggml_get_op_params_i32(dst, 1),
        /* .mode        = */ ggml_get_op_params_i32(dst, 2),
        /* .section0    = */ ggml_get_op_params_i32(dst, 11),
        /* .section1    = */ ggml_get_op_params_i32(dst, 12),
        /* .section2    = */ ggml_get_op_params_i32(dst, 13),
        /* .section3    = */ ggml_get_op_params_i32(dst, 14),
        /* .freq_base   = */ freq_base,
        /* .freq_scale  = */ freq_scale,
        /* .attn_factor = */ attn_factor,
        /* ._pad        = */ 0.0f,
    };

    const auto & provider = context->device_context->rope_f32_provider;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    const uint64_t total_pairs = static_cast<uint64_t>(constants.nrows) *
        static_cast<uint64_t>(constants.ne00 / 2);
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((total_pairs + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream, provider.executable, provider.export_ordinal, &config,
            &constants, sizeof(constants), bindings, 3, PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->rope_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_ssm_conv(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst,
        const ggml_tensor * fused_dst = nullptr,
        bool apply_silu = false) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * out = fused_dst ? fused_dst : dst;
    pyre_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_pyre_tensor_buffer_ref(out, &bindings[2])) {
        GGML_LOG_ERROR("%s: SSM_CONV tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_pyre_ssm_conv_constants constants = {
        /* .d_conv     = */ src1->ne[0],
        /* .conv_width = */ src0->ne[0],
        /* .d_inner    = */ src0->ne[1],
        /* .n_tokens   = */ dst->ne[1],
        /* .n_seqs     = */ dst->ne[2],
        /* .src0_nb1   = */ static_cast<int64_t>(src0->nb[1]),
        /* .src0_nb2   = */ static_cast<int64_t>(src0->nb[2]),
        /* .weight_nb1 = */ static_cast<int64_t>(src1->nb[1]),
        /* .dst_nb1    = */ static_cast<int64_t>(out->nb[1]),
        /* .dst_nb2    = */ static_cast<int64_t>(out->nb[2]),
        /* .apply_silu = */ apply_silu ? 1 : 0,
        /* .pad        = */ 0,
    };

    const auto & provider = context->device_context->ssm_conv_provider;
    const int64_t total = constants.d_inner * constants.n_tokens * constants.n_seqs;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((total + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ {
            workgroup_size,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            3,
            PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->ssm_conv_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_gated_delta_net(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * g = dst->src[3];
    const ggml_tensor * beta = dst->src[4];
    const ggml_tensor * state = dst->src[5];
    pyre_buffer_ref_t bindings[7] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(q, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(k, &bindings[1]) ||
        !ggml_backend_pyre_tensor_buffer_ref(v, &bindings[2]) ||
        !ggml_backend_pyre_tensor_buffer_ref(g, &bindings[3]) ||
        !ggml_backend_pyre_tensor_buffer_ref(beta, &bindings[4]) ||
        !ggml_backend_pyre_tensor_buffer_ref(state, &bindings[5]) ||
        !ggml_backend_pyre_tensor_buffer_ref(dst, &bindings[6])) {
        GGML_LOG_ERROR("%s: GATED_DELTA_NET tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_pyre_gated_delta_net_constants constants = {
        /* .S_v      = */ v->ne[0],
        /* .H        = */ v->ne[1],
        /* .n_tokens = */ v->ne[2],
        /* .n_seqs   = */ v->ne[3],
        /* .neq1     = */ q->ne[1],
        /* .nek1     = */ k->ne[1],
        /* .rq3      = */ v->ne[3] / q->ne[3],
        /* .rk3      = */ v->ne[3] / k->ne[3],
        /* .q_nb1    = */ static_cast<int64_t>(q->nb[1]),
        /* .q_nb2    = */ static_cast<int64_t>(q->nb[2]),
        /* .q_nb3    = */ static_cast<int64_t>(q->nb[3]),
        /* .k_nb1    = */ static_cast<int64_t>(k->nb[1]),
        /* .k_nb2    = */ static_cast<int64_t>(k->nb[2]),
        /* .k_nb3    = */ static_cast<int64_t>(k->nb[3]),
        /* .v_nb1    = */ static_cast<int64_t>(v->nb[1]),
        /* .v_nb2    = */ static_cast<int64_t>(v->nb[2]),
        /* .v_nb3    = */ static_cast<int64_t>(v->nb[3]),
        /* .g_ne0    = */ g->ne[0],
        /* .g_nb1    = */ static_cast<int64_t>(g->nb[1]),
        /* .g_nb2    = */ static_cast<int64_t>(g->nb[2]),
        /* .g_nb3    = */ static_cast<int64_t>(g->nb[3]),
        /* .beta_nb1 = */ static_cast<int64_t>(beta->nb[1]),
        /* .beta_nb2 = */ static_cast<int64_t>(beta->nb[2]),
        /* .beta_nb3 = */ static_cast<int64_t>(beta->nb[3]),
        /* .scale    = */ 1.0f / std::sqrt(static_cast<float>(v->ne[0])),
        /* ._pad     = */ 0,
    };

    const auto & provider = context->device_context->gated_delta_net_provider;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>(constants.S_v),
            static_cast<uint32_t>(constants.H),
            static_cast<uint32_t>(constants.n_seqs),
        },
        /* .workgroup_size = */ {
            provider.export_info.workgroup_size[0] ? provider.export_info.workgroup_size[0] : 256,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            7,
            PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->gated_delta_net_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static bool ggml_backend_pyre_supports_mul_mat_vec_f16(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return device_context->mul_mat_vec_f16_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           !device_context->policy.disable_mul_mat_vec &&
           src0 && src1 &&
           src0->type == GGML_TYPE_F16 &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src1->ne[2] == 1 && src1->ne[3] == 1 &&
           op->ne[2] == 1 && op->ne[3] == 1 &&
           src0->ne[0] == src1->ne[0] &&
           op->ne[0] == src0->ne[1] &&
           op->ne[1] == src1->ne[1] &&
           src1->ne[1] >= 1 && src1->ne[1] <= GGML_PYRE_MUL_MAT_VEC_MAX_COLS &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op);
}

static bool ggml_backend_pyre_supports_mul_mat_vec_f16_batched(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return device_context->mul_mat_vec_f16_batched_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           !device_context->policy.disable_mul_mat_vec &&
           src0 && src1 &&
           src0->type == GGML_TYPE_F16 &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[3] == 1 && src1->ne[3] == 1 && op->ne[3] == 1 &&
           src0->ne[0] == src1->ne[0] &&
           op->ne[0] == src0->ne[1] &&
           op->ne[1] == src1->ne[1] &&
           op->ne[2] == src1->ne[2] &&
           src0->ne[2] >= 1 && op->ne[2] >= src0->ne[2] &&
           (op->ne[2] % src0->ne[2]) == 0 &&
           src1->ne[1] >= 1 && src1->ne[1] <= GGML_PYRE_MUL_MAT_VEC_MAX_COLS &&
           src0->nb[0] == ggml_type_size(src0->type) &&
           src1->nb[0] == ggml_type_size(src1->type) &&
           op->nb[0] == ggml_type_size(op->type);
}

static bool ggml_backend_pyre_supports_mul_mat_vec_f32_batched(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return device_context->mul_mat_vec_f32_batched_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           !device_context->policy.disable_mul_mat_vec &&
           src0 && src1 &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src1->ne[3] >= 1 && op->ne[3] == src1->ne[3] &&
           src0->ne[0] == src1->ne[0] &&
           op->ne[0] == src0->ne[1] &&
           op->ne[1] == src1->ne[1] &&
           op->ne[2] == src1->ne[2] &&
           src0->ne[2] >= 1 && op->ne[2] >= src0->ne[2] &&
           (op->ne[2] % src0->ne[2]) == 0 &&
           src0->ne[3] >= 1 && op->ne[3] >= src0->ne[3] &&
           (op->ne[3] % src0->ne[3]) == 0 &&
           src1->ne[1] >= 1 && src1->ne[1] <= GGML_PYRE_MUL_MAT_VEC_MAX_COLS &&
           src0->nb[0] == ggml_type_size(src0->type) &&
           src1->nb[0] == ggml_type_size(src1->type) &&
           op->nb[0] == ggml_type_size(op->type);
}

static bool ggml_backend_pyre_supports_mul_mat_vec_bf16(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return device_context->mul_mat_vec_bf16_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           !device_context->policy.disable_mul_mat_vec &&
           src0 && src1 &&
           src0->type == GGML_TYPE_BF16 &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src1->ne[2] == 1 && src1->ne[3] == 1 &&
           op->ne[2] == 1 && op->ne[3] == 1 &&
           src0->ne[0] == src1->ne[0] &&
           op->ne[0] == src0->ne[1] &&
           op->ne[1] == src1->ne[1] &&
           src1->ne[1] >= 1 && src1->ne[1] <= GGML_PYRE_MUL_MAT_VEC_MAX_COLS &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op);
}

static bool ggml_backend_pyre_supports_mul_mat_vec_f32(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return device_context->mul_mat_vec_f32_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           !device_context->policy.disable_mul_mat_vec &&
           src0 && src1 &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src1->ne[2] == 1 && src1->ne[3] == 1 &&
           op->ne[2] == 1 && op->ne[3] == 1 &&
           src0->ne[0] == src1->ne[0] &&
           op->ne[0] == src0->ne[1] &&
           op->ne[1] == src1->ne[1] &&
           src1->ne[1] >= 1 && src1->ne[1] <= GGML_PYRE_MUL_MAT_VEC_MAX_COLS &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op);
}

static bool ggml_backend_pyre_supports_mul_mat_vec_k_quant(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op,
        ggml_type type,
        const ggml_backend_pyre_op_provider & provider,
        int64_t k_multiple) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           !device_context->policy.disable_mul_mat_vec &&
           src0 && src1 &&
           src0->type == type &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[0] % k_multiple == 0 &&
           src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src1->ne[2] == 1 && src1->ne[3] == 1 &&
           op->ne[2] == 1 && op->ne[3] == 1 &&
           src0->ne[0] == src1->ne[0] &&
           op->ne[0] == src0->ne[1] &&
           op->ne[1] == src1->ne[1] &&
           src1->ne[1] >= 1 && src1->ne[1] <= GGML_PYRE_MUL_MAT_VEC_MAX_COLS &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op);
}

static bool ggml_backend_pyre_supports_mul_mat_vec_q4_k(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_pyre_supports_mul_mat_vec_k_quant(
        device_context, op, GGML_TYPE_Q4_K, device_context->mul_mat_vec_q4_k_provider, 256);
}

static bool ggml_backend_pyre_supports_mul_mat_vec_q4_k_q8_1(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    return device_context->policy.enable_q8_1_mmvq &&
           !device_context->policy.disable_q8_1_mmvq &&
           device_context->quantize_q8_1_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           ggml_backend_pyre_supports_mul_mat_vec_k_quant(
               device_context, op, GGML_TYPE_Q4_K,
               device_context->mul_mat_vec_q4_k_q8_1_provider, 256);
}

static bool ggml_backend_pyre_supports_mul_mat_vec_q5_k(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_pyre_supports_mul_mat_vec_k_quant(
        device_context, op, GGML_TYPE_Q5_K, device_context->mul_mat_vec_q5_k_provider, 256);
}

static bool ggml_backend_pyre_supports_mul_mat_vec_q6_k(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_pyre_supports_mul_mat_vec_k_quant(
        device_context, op, GGML_TYPE_Q6_K, device_context->mul_mat_vec_q6_k_provider, 256);
}

static bool ggml_backend_pyre_supports_mul_mat_vec_q8_0(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_pyre_supports_mul_mat_vec_k_quant(
        device_context, op, GGML_TYPE_Q8_0, device_context->mul_mat_vec_q8_0_provider, 32);
}

static const char * ggml_backend_pyre_mul_mat_vec_unsupported_reason(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    if (device_context->policy.disable_mul_mat_vec) {
        return "disabled by GGML_PYRE_DISABLE_MUL_MAT_VEC";
    }
    if (!op) {
        return "op is null";
    }

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (!src0 || !src1) {
        return "missing source tensor";
    }
    if (src0->type != GGML_TYPE_F16 &&
        src0->type != GGML_TYPE_BF16 &&
        src0->type != GGML_TYPE_F32 &&
        src0->type != GGML_TYPE_Q4_K &&
        src0->type != GGML_TYPE_Q5_K &&
        src0->type != GGML_TYPE_Q6_K &&
        src0->type != GGML_TYPE_Q8_0) {
        return "src0 type has no Pyre matvec provider";
    }
    if (src0->type == GGML_TYPE_F16 &&
        device_context->mul_mat_vec_f16_provider.kind != ggml_backend_pyre_provider_kind::direct_executable &&
        device_context->mul_mat_vec_f16_batched_provider.kind != ggml_backend_pyre_provider_kind::direct_executable) {
        return "F16 provider unavailable";
    }
    if (src0->type == GGML_TYPE_BF16 &&
        device_context->mul_mat_vec_bf16_provider.kind != ggml_backend_pyre_provider_kind::direct_executable) {
        return "BF16 provider unavailable";
    }
    if (src0->type == GGML_TYPE_F32 &&
        device_context->mul_mat_vec_f32_provider.kind != ggml_backend_pyre_provider_kind::direct_executable &&
        device_context->mul_mat_vec_f32_batched_provider.kind != ggml_backend_pyre_provider_kind::direct_executable) {
        return "F32 provider unavailable";
    }
    if (src0->type == GGML_TYPE_Q4_K &&
        device_context->mul_mat_vec_q4_k_provider.kind != ggml_backend_pyre_provider_kind::direct_executable) {
        return "Q4_K provider unavailable";
    }
    if (src0->type == GGML_TYPE_Q5_K &&
        device_context->mul_mat_vec_q5_k_provider.kind != ggml_backend_pyre_provider_kind::direct_executable) {
        return "Q5_K provider unavailable";
    }
    if (src0->type == GGML_TYPE_Q6_K &&
        device_context->mul_mat_vec_q6_k_provider.kind != ggml_backend_pyre_provider_kind::direct_executable) {
        return "Q6_K provider unavailable";
    }
    if (src0->type == GGML_TYPE_Q8_0 &&
        device_context->mul_mat_vec_q8_0_provider.kind != ggml_backend_pyre_provider_kind::direct_executable) {
        return "Q8_0 provider unavailable";
    }
    if (src1->type != GGML_TYPE_F32) {
        return "src1 type is not F32";
    }
    if (op->type != GGML_TYPE_F32) {
        return "dst type is not F32";
    }
    if ((src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K || src0->type == GGML_TYPE_Q6_K) &&
        src0->ne[0] % 256 != 0) {
        return "K-quant k is not divisible by 256";
    }
    if (src0->type == GGML_TYPE_Q8_0 && src0->ne[0] % 32 != 0) {
        return "Q8_0 k is not divisible by 32";
    }
    if (src0->type == GGML_TYPE_F16 &&
        src0->ne[3] == 1 && src1->ne[3] == 1 && op->ne[3] == 1 &&
        src0->ne[2] >= 1 && src1->ne[2] == op->ne[2] &&
        op->ne[2] >= src0->ne[2] && (op->ne[2] % src0->ne[2]) == 0) {
        // Covered by the strided F16 batched provider if the remaining checks pass.
    } else if (src0->type == GGML_TYPE_F32 &&
               src1->ne[3] >= 1 && op->ne[3] == src1->ne[3] &&
               src0->ne[2] >= 1 && src1->ne[2] == op->ne[2] &&
               op->ne[2] >= src0->ne[2] && (op->ne[2] % src0->ne[2]) == 0 &&
               src0->ne[3] >= 1 && op->ne[3] >= src0->ne[3] && (op->ne[3] % src0->ne[3]) == 0) {
        // Covered by the strided F32 batched provider if the remaining checks pass.
    } else if (src0->ne[2] != 1 || src0->ne[3] != 1 ||
               src1->ne[2] != 1 || src1->ne[3] != 1 ||
               op->ne[2] != 1 || op->ne[3] != 1) {
        return "batched dims are not supported";
    }
    if (src0->ne[0] != src1->ne[0]) {
        return "src0/src1 k mismatch";
    }
    if (op->ne[0] != src0->ne[1]) {
        return "dst rows do not match src0 rows";
    }
    if (op->ne[1] != src1->ne[1]) {
        return "dst cols do not match src1 cols";
    }
    if (src1->ne[1] < 1 || src1->ne[1] > GGML_PYRE_MUL_MAT_VEC_MAX_COLS) {
        return "src1 column count is outside matvec limit";
    }
    if (!ggml_is_contiguous(src0)) {
        return "src0 is not contiguous";
    }
    if (!ggml_is_contiguous(src1)) {
        return "src1 is not contiguous";
    }
    if (!ggml_is_contiguous(op)) {
        return "dst is not contiguous";
    }
    return "unknown";
}

static bool ggml_backend_pyre_supports_mul_mat_vec(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_pyre_supports_mul_mat_vec_f16(device_context, op) ||
           ggml_backend_pyre_supports_mul_mat_vec_f16_batched(device_context, op) ||
           ggml_backend_pyre_supports_mul_mat_vec_bf16(device_context, op) ||
           ggml_backend_pyre_supports_mul_mat_vec_f32(device_context, op) ||
           ggml_backend_pyre_supports_mul_mat_vec_f32_batched(device_context, op) ||
           ggml_backend_pyre_supports_mul_mat_vec_q4_k(device_context, op) ||
           ggml_backend_pyre_supports_mul_mat_vec_q4_k_q8_1(device_context, op) ||
           ggml_backend_pyre_supports_mul_mat_vec_q5_k(device_context, op) ||
           ggml_backend_pyre_supports_mul_mat_vec_q6_k(device_context, op) ||
           ggml_backend_pyre_supports_mul_mat_vec_q8_0(device_context, op);
}

static bool ggml_backend_pyre_supports_mul_mat_id_q4_k(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];
    return device_context->mul_mat_id_q4_k_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           !device_context->policy.disable_mul_mat_id &&
           src0 && src1 && src2 &&
           src0->type == GGML_TYPE_Q4_K &&
           src1->type == GGML_TYPE_F32 &&
           src2->type == GGML_TYPE_I32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[0] % 256 == 0 &&
           src0->ne[3] == 1 &&
           src1->ne[3] == 1 &&
           src2->ne[2] == 1 && src2->ne[3] == 1 &&
           src0->ne[0] == src1->ne[0] &&
           src2->ne[1] == src1->ne[2] &&
           (src2->ne[0] == src1->ne[1] || src1->ne[1] == 1) &&
           op->ne[0] == src0->ne[1] &&
           op->ne[1] == src2->ne[0] &&
           op->ne[2] == src2->ne[1] &&
           op->ne[3] == 1 &&
           src0->nb[0] == ggml_type_size(src0->type) &&
           src1->nb[0] == ggml_type_size(src1->type) &&
           src2->nb[0] == ggml_type_size(src2->type) &&
           op->nb[0] == ggml_type_size(op->type);
}

static bool ggml_backend_pyre_supports_mul_mat_id_q4_k_q8_1(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    return device_context->policy.enable_q8_1_mmvq &&
           !device_context->policy.disable_q8_1_mmvq &&
           device_context->quantize_q8_1_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           device_context->mul_mat_id_q4_k_q8_1_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           ggml_backend_pyre_supports_mul_mat_id_q4_k(device_context, op);
}

static bool ggml_backend_pyre_supports_mul_mat_id_q4_k_mul(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * mmid,
        const ggml_tensor * mul) {
    if (device_context->mul_mat_id_q4_k_mul_provider.kind !=
            ggml_backend_pyre_provider_kind::direct_executable ||
        !ggml_backend_pyre_supports_mul_mat_id_q4_k(device_context, mmid) ||
        !mul ||
        mul->op != GGML_OP_MUL ||
        mul->type != GGML_TYPE_F32 ||
        mul->src[0] != mmid ||
        !mul->src[1] ||
        mul->src[1]->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(mul, mmid) ||
        mul->nb[0] != ggml_type_size(mul->type)) {
        return false;
    }

    const ggml_tensor * scale = mul->src[1];
    return scale->ne[0] == 1 &&
           scale->ne[1] == mmid->ne[1] &&
           scale->ne[2] == 1 &&
           scale->ne[3] == 1 &&
           scale->nb[0] == ggml_type_size(scale->type) &&
           ggml_is_contiguous(scale);
}

static bool ggml_backend_pyre_supports_mul_mat_id_q4_k_mul_q8_1(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * mmid,
        const ggml_tensor * mul) {
    return device_context->policy.enable_q8_1_mmvq &&
           !device_context->policy.disable_q8_1_mmvq &&
           device_context->quantize_q8_1_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           device_context->mul_mat_id_q4_k_mul_q8_1_provider.kind ==
               ggml_backend_pyre_provider_kind::direct_executable &&
           ggml_backend_pyre_supports_mul_mat_id_q4_k_mul(device_context, mmid, mul);
}

struct ggml_backend_pyre_mul_mat_vec_constants {
    int64_t k;
    int64_t rows;
    int64_t cols;
};

struct ggml_backend_pyre_quantize_q8_1_constants {
    int64_t ne00;
    int64_t s01;
    int64_t s02;
    int64_t s03;
    int64_t ne0;
    int64_t ne1;
    int64_t ne2;
};

struct ggml_backend_pyre_mul_mat_vec_f16_batched_constants {
    int64_t k;
    int64_t rows;
    int64_t cols;
    int64_t dst_ne2;
    int64_t dst_ne3;
    int64_t src0_ne2;
    int64_t src0_ne3;
    int64_t src0_nb1;
    int64_t src0_nb2;
    int64_t src0_nb3;
    int64_t src1_nb1;
    int64_t src1_nb2;
    int64_t src1_nb3;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t dst_nb3;
};

struct ggml_backend_pyre_mul_mat_id_q4_k_constants {
    int64_t k;
    int64_t rows;
    int64_t n_ids;
    int64_t n_tokens;
    int64_t n_experts;
    int64_t src0_nb1;
    int64_t src0_nb2;
    int64_t src1_nb1;
    int64_t src1_nb2;
    int64_t ids_nb0;
    int64_t ids_nb1;
    int64_t dst_nb1;
    int64_t dst_nb2;
};

struct ggml_backend_pyre_mul_mat_id_q4_k_q8_1_constants {
    int64_t k;
    int64_t rows;
    int64_t n_ids;
    int64_t n_tokens;
    int64_t n_experts;
    int64_t src0_nb1;
    int64_t src0_nb2;
    int64_t ids_nb0;
    int64_t ids_nb1;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t src1_cols;
};

struct ggml_backend_pyre_mul_mat_id_q4_k_mul_constants {
    int64_t k;
    int64_t rows;
    int64_t n_ids;
    int64_t n_tokens;
    int64_t n_experts;
    int64_t src0_nb1;
    int64_t src0_nb2;
    int64_t src1_nb1;
    int64_t src1_nb2;
    int64_t ids_nb0;
    int64_t ids_nb1;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t scale_nb1;
};

struct ggml_backend_pyre_mul_mat_id_q4_k_mul_q8_1_constants {
    int64_t k;
    int64_t rows;
    int64_t n_ids;
    int64_t n_tokens;
    int64_t n_experts;
    int64_t src0_nb1;
    int64_t src0_nb2;
    int64_t ids_nb0;
    int64_t ids_nb1;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t src1_cols;
    int64_t scale_nb1;
};

static bool ggml_backend_pyre_dispatch_quantize_q8_1(
        ggml_backend_pyre_context * context,
        const ggml_tensor * src,
        pyre_buffer_ref_t * q8_1_ref) {
    pyre_buffer_ref_t src_ref = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src, &src_ref)) {
        GGML_LOG_ERROR("%s: Q8_1 quantize source is not backed by a PYRE buffer\n", __func__);
        return false;
    }

    const size_t q8_1_size = static_cast<size_t>(src->ne[3] * src->ne[2] * src->ne[1] * (src->ne[0] / 32) * 36);
    if (!ggml_backend_pyre_ensure_q8_1_scratch(context, q8_1_size, q8_1_ref)) {
        return false;
    }

    pyre_buffer_ref_t quant_bindings[2] = { src_ref, *q8_1_ref };
    ggml_backend_pyre_quantize_q8_1_constants quant_constants = {
        /* .ne00 = */ src->ne[0],
        /* .s01  = */ static_cast<int64_t>(src->nb[1] / sizeof(float)),
        /* .s02  = */ static_cast<int64_t>(src->nb[2] / sizeof(float)),
        /* .s03  = */ static_cast<int64_t>(src->nb[3] / sizeof(float)),
        /* .ne0  = */ src->ne[0],
        /* .ne1  = */ src->ne[1],
        /* .ne2  = */ src->ne[2],
    };

    const auto & quant_provider = context->device_context->quantize_q8_1_provider;
    pyre_dispatch_config_t quant_config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>(quant_constants.ne0 / 32),
            static_cast<uint32_t>(quant_constants.ne1),
            static_cast<uint32_t>(src->ne[2] * src->ne[3]),
        },
        /* .workgroup_size = */ {
            quant_provider.export_info.workgroup_size[0] ? quant_provider.export_info.workgroup_size[0] : 32,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };
    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream,
            quant_provider.executable,
            quant_provider.export_ordinal,
            &quant_config,
            &quant_constants,
            sizeof(quant_constants),
            quant_bindings,
            2,
            PYRE_DISPATCH_FLAG_NONE))) {
        return false;
    }
    context->dispatch_count++;
    return GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream));
}

static ggml_status ggml_backend_pyre_dispatch_mul_mat_vec_q4_k_q8_1(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    pyre_buffer_ref_t src1_ref = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src1, &src1_ref)) {
        GGML_LOG_ERROR("%s: Q8_1 quantize source is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    const int64_t k_padded = src1->ne[0];
    const size_t q8_1_size = static_cast<size_t>(src1->ne[3] * src1->ne[2] * src1->ne[1] * (k_padded / 32) * 36);
    pyre_buffer_ref_t q8_1_ref = {};
    if (!ggml_backend_pyre_ensure_q8_1_scratch(context, q8_1_size, &q8_1_ref)) {
        return GGML_STATUS_FAILED;
    }

    pyre_buffer_ref_t quant_bindings[2] = { src1_ref, q8_1_ref };
    ggml_backend_pyre_quantize_q8_1_constants quant_constants = {
        /* .ne00 = */ src1->ne[0],
        /* .s01  = */ static_cast<int64_t>(src1->nb[1] / sizeof(float)),
        /* .s02  = */ static_cast<int64_t>(src1->nb[2] / sizeof(float)),
        /* .s03  = */ static_cast<int64_t>(src1->nb[3] / sizeof(float)),
        /* .ne0  = */ k_padded,
        /* .ne1  = */ src1->ne[1],
        /* .ne2  = */ src1->ne[2],
    };

    const auto & quant_provider = context->device_context->quantize_q8_1_provider;
    pyre_dispatch_config_t quant_config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>(quant_constants.ne0 / 32),
            static_cast<uint32_t>(quant_constants.ne1),
            static_cast<uint32_t>(src1->ne[2] * src1->ne[3]),
        },
        /* .workgroup_size = */ {
            quant_provider.export_info.workgroup_size[0] ? quant_provider.export_info.workgroup_size[0] : 32,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };
    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream,
            quant_provider.executable,
            quant_provider.export_ordinal,
            &quant_config,
            &quant_constants,
            sizeof(quant_constants),
            quant_bindings,
            2,
            PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }

    pyre_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("%s: Q4_K x Q8_1 MUL_MAT tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }
    bindings[1] = q8_1_ref;

    ggml_backend_pyre_mul_mat_vec_constants constants = {
        /* .k    = */ src0->ne[0],
        /* .rows = */ src0->ne[1],
        /* .cols = */ src1->ne[1],
    };
    const auto & provider = context->device_context->mul_mat_vec_q4_k_q8_1_provider;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>(constants.rows),
            static_cast<uint32_t>(constants.cols),
            1,
        },
        /* .workgroup_size = */ {
            provider.export_info.workgroup_size[0] ? provider.export_info.workgroup_size[0] : 256,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };
    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            3,
            PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->mul_mat_vec_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_mul_mat_vec_f16(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst) {
    if (ggml_backend_pyre_supports_mul_mat_vec_f16_batched(context->device_context, dst) ||
        ggml_backend_pyre_supports_mul_mat_vec_f32_batched(context->device_context, dst)) {
        const ggml_tensor * src0 = dst->src[0];
        const ggml_tensor * src1 = dst->src[1];
        pyre_buffer_ref_t bindings[3] = {};
        if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
            !ggml_backend_pyre_tensor_buffer_ref(src1, &bindings[1]) ||
            !ggml_backend_pyre_tensor_buffer_ref(dst, &bindings[2])) {
            GGML_LOG_ERROR("%s: batched MUL_MAT tensor is not backed by a PYRE buffer\n", __func__);
            return GGML_STATUS_FAILED;
        }

        ggml_backend_pyre_mul_mat_vec_f16_batched_constants constants = {
            /* .k         = */ src0->ne[0],
            /* .rows      = */ src0->ne[1],
            /* .cols      = */ src1->ne[1],
            /* .dst_ne2   = */ dst->ne[2],
            /* .dst_ne3   = */ dst->ne[3],
            /* .src0_ne2  = */ src0->ne[2],
            /* .src0_ne3  = */ src0->ne[3],
            /* .src0_nb1  = */ static_cast<int64_t>(src0->nb[1]),
            /* .src0_nb2  = */ static_cast<int64_t>(src0->nb[2]),
            /* .src0_nb3  = */ static_cast<int64_t>(src0->nb[3]),
            /* .src1_nb1  = */ static_cast<int64_t>(src1->nb[1]),
            /* .src1_nb2  = */ static_cast<int64_t>(src1->nb[2]),
            /* .src1_nb3  = */ static_cast<int64_t>(src1->nb[3]),
            /* .dst_nb1   = */ static_cast<int64_t>(dst->nb[1]),
            /* .dst_nb2   = */ static_cast<int64_t>(dst->nb[2]),
            /* .dst_nb3   = */ static_cast<int64_t>(dst->nb[3]),
        };

        const auto & provider = src0->type == GGML_TYPE_F16 ?
            context->device_context->mul_mat_vec_f16_batched_provider :
            context->device_context->mul_mat_vec_f32_batched_provider;
        pyre_dispatch_config_t config = {
            /* .workgroup_count = */ {
                static_cast<uint32_t>(constants.rows),
                static_cast<uint32_t>(constants.cols * constants.dst_ne2 * constants.dst_ne3),
                1,
            },
            /* .workgroup_size = */ {
                provider.export_info.workgroup_size[0] ? provider.export_info.workgroup_size[0] : 256,
                1,
                1,
            },
            /* .subgroup_size = */ 0,
        };

        if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
                context->stream,
                provider.executable,
                provider.export_ordinal,
                &config,
                &constants,
                sizeof(constants),
                bindings,
                3,
                PYRE_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        context->dispatch_count++;
        context->mul_mat_vec_count++;

        if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_backend_pyre_supports_mul_mat_vec_q4_k_q8_1(context->device_context, dst)) {
        return ggml_backend_pyre_dispatch_mul_mat_vec_q4_k_q8_1(context, dst);
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    pyre_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_pyre_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("%s: MUL_MAT tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_pyre_mul_mat_vec_constants constants = {
        /* .k    = */ src0->ne[0],
        /* .rows = */ src0->ne[1],
        /* .cols = */ src1->ne[1],
    };

    const ggml_backend_pyre_op_provider * provider = nullptr;
    switch (src0->type) {
        case GGML_TYPE_F16:
            provider = &context->device_context->mul_mat_vec_f16_provider;
            break;
        case GGML_TYPE_BF16:
            provider = &context->device_context->mul_mat_vec_bf16_provider;
            break;
        case GGML_TYPE_F32:
            provider = &context->device_context->mul_mat_vec_f32_provider;
            break;
        case GGML_TYPE_Q4_K:
            provider = &context->device_context->mul_mat_vec_q4_k_provider;
            break;
        case GGML_TYPE_Q5_K:
            provider = &context->device_context->mul_mat_vec_q5_k_provider;
            break;
        case GGML_TYPE_Q6_K:
            provider = &context->device_context->mul_mat_vec_q6_k_provider;
            break;
        case GGML_TYPE_Q8_0:
            provider = &context->device_context->mul_mat_vec_q8_0_provider;
            break;
        default:
            return GGML_STATUS_FAILED;
    }

    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>(constants.rows),
            static_cast<uint32_t>(constants.cols),
            1,
        },
        /* .workgroup_size = */ {
            provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : 256,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream,
            provider->executable,
            provider->export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            3,
            PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->mul_mat_vec_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_mul_mat_id_q4_k(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];
    pyre_buffer_ref_t bindings[4] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_pyre_tensor_buffer_ref(src2, &bindings[2]) ||
        !ggml_backend_pyre_tensor_buffer_ref(dst, &bindings[3])) {
        GGML_LOG_ERROR("%s: MUL_MAT_ID tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_pyre_mul_mat_id_q4_k_constants constants = {
        /* .k          = */ src0->ne[0],
        /* .rows       = */ src0->ne[1],
        /* .n_ids      = */ src2->ne[0],
        /* .n_tokens   = */ src2->ne[1],
        /* .n_experts  = */ src0->ne[2],
        /* .src0_nb1   = */ static_cast<int64_t>(src0->nb[1]),
        /* .src0_nb2   = */ static_cast<int64_t>(src0->nb[2]),
        /* .src1_nb1   = */ src1->ne[1] == 1 ? 0 : static_cast<int64_t>(src1->nb[1]),
        /* .src1_nb2   = */ static_cast<int64_t>(src1->nb[2]),
        /* .ids_nb0    = */ static_cast<int64_t>(src2->nb[0]),
        /* .ids_nb1    = */ static_cast<int64_t>(src2->nb[1]),
        /* .dst_nb1    = */ static_cast<int64_t>(dst->nb[1]),
        /* .dst_nb2    = */ static_cast<int64_t>(dst->nb[2]),
    };

    const auto & provider = context->device_context->mul_mat_id_q4_k_provider;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>(constants.rows),
            static_cast<uint32_t>(constants.n_ids * constants.n_tokens),
            1,
        },
        /* .workgroup_size = */ {
            provider.export_info.workgroup_size[0] ? provider.export_info.workgroup_size[0] : 256,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            4,
            PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->mul_mat_id_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_mul_mat_id_q4_k_mul(
        ggml_backend_pyre_context * context,
        const ggml_tensor * mmid,
        const ggml_tensor * mul) {
    const ggml_tensor * src0 = mmid->src[0];
    const ggml_tensor * src1 = mmid->src[1];
    const ggml_tensor * src2 = mmid->src[2];
    const ggml_tensor * scale = mul->src[1];
    pyre_buffer_ref_t bindings[5] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_pyre_tensor_buffer_ref(src2, &bindings[2]) ||
        !ggml_backend_pyre_tensor_buffer_ref(scale, &bindings[3]) ||
        !ggml_backend_pyre_tensor_buffer_ref(mul, &bindings[4])) {
        GGML_LOG_ERROR("%s: MUL_MAT_ID_MUL tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_pyre_mul_mat_id_q4_k_mul_constants constants = {
        /* .k          = */ src0->ne[0],
        /* .rows       = */ src0->ne[1],
        /* .n_ids      = */ src2->ne[0],
        /* .n_tokens   = */ src2->ne[1],
        /* .n_experts  = */ src0->ne[2],
        /* .src0_nb1   = */ static_cast<int64_t>(src0->nb[1]),
        /* .src0_nb2   = */ static_cast<int64_t>(src0->nb[2]),
        /* .src1_nb1   = */ src1->ne[1] == 1 ? 0 : static_cast<int64_t>(src1->nb[1]),
        /* .src1_nb2   = */ static_cast<int64_t>(src1->nb[2]),
        /* .ids_nb0    = */ static_cast<int64_t>(src2->nb[0]),
        /* .ids_nb1    = */ static_cast<int64_t>(src2->nb[1]),
        /* .dst_nb1    = */ static_cast<int64_t>(mul->nb[1]),
        /* .dst_nb2    = */ static_cast<int64_t>(mul->nb[2]),
        /* .scale_nb1  = */ static_cast<int64_t>(scale->nb[1]),
    };

    const auto & provider = context->device_context->mul_mat_id_q4_k_mul_provider;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>(constants.rows),
            static_cast<uint32_t>(constants.n_ids * constants.n_tokens),
            1,
        },
        /* .workgroup_size = */ {
            provider.export_info.workgroup_size[0] ? provider.export_info.workgroup_size[0] : 256,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            5,
            PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->mul_mat_id_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_mul_mat_id_q4_k_q8_1(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    pyre_buffer_ref_t q8_1_ref = {};
    if (!ggml_backend_pyre_dispatch_quantize_q8_1(context, src1, &q8_1_ref)) {
        return GGML_STATUS_FAILED;
    }

    pyre_buffer_ref_t bindings[4] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(src2, &bindings[2]) ||
        !ggml_backend_pyre_tensor_buffer_ref(dst, &bindings[3])) {
        GGML_LOG_ERROR("%s: Q8_1 MUL_MAT_ID tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }
    bindings[1] = q8_1_ref;

    ggml_backend_pyre_mul_mat_id_q4_k_q8_1_constants constants = {
        /* .k          = */ src0->ne[0],
        /* .rows       = */ src0->ne[1],
        /* .n_ids      = */ src2->ne[0],
        /* .n_tokens   = */ src2->ne[1],
        /* .n_experts  = */ src0->ne[2],
        /* .src0_nb1   = */ static_cast<int64_t>(src0->nb[1]),
        /* .src0_nb2   = */ static_cast<int64_t>(src0->nb[2]),
        /* .ids_nb0    = */ static_cast<int64_t>(src2->nb[0]),
        /* .ids_nb1    = */ static_cast<int64_t>(src2->nb[1]),
        /* .dst_nb1    = */ static_cast<int64_t>(dst->nb[1]),
        /* .dst_nb2    = */ static_cast<int64_t>(dst->nb[2]),
        /* .src1_cols  = */ src1->ne[1],
    };

    const auto & provider = context->device_context->mul_mat_id_q4_k_q8_1_provider;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>(constants.rows),
            static_cast<uint32_t>(constants.n_ids * constants.n_tokens),
            1,
        },
        /* .workgroup_size = */ {
            provider.export_info.workgroup_size[0] ? provider.export_info.workgroup_size[0] : 256,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };
    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            4,
            PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->mul_mat_id_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_pyre_dispatch_mul_mat_id_q4_k_mul_q8_1(
        ggml_backend_pyre_context * context,
        const ggml_tensor * mmid,
        const ggml_tensor * mul) {
    const ggml_tensor * src0 = mmid->src[0];
    const ggml_tensor * src1 = mmid->src[1];
    const ggml_tensor * src2 = mmid->src[2];
    const ggml_tensor * scale = mul->src[1];

    pyre_buffer_ref_t q8_1_ref = {};
    if (!ggml_backend_pyre_dispatch_quantize_q8_1(context, src1, &q8_1_ref)) {
        return GGML_STATUS_FAILED;
    }

    pyre_buffer_ref_t bindings[5] = {};
    if (!ggml_backend_pyre_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_pyre_tensor_buffer_ref(src2, &bindings[2]) ||
        !ggml_backend_pyre_tensor_buffer_ref(scale, &bindings[3]) ||
        !ggml_backend_pyre_tensor_buffer_ref(mul, &bindings[4])) {
        GGML_LOG_ERROR("%s: Q8_1 MUL_MAT_ID_MUL tensor is not backed by a PYRE buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }
    bindings[1] = q8_1_ref;

    ggml_backend_pyre_mul_mat_id_q4_k_mul_q8_1_constants constants = {
        /* .k          = */ src0->ne[0],
        /* .rows       = */ src0->ne[1],
        /* .n_ids      = */ src2->ne[0],
        /* .n_tokens   = */ src2->ne[1],
        /* .n_experts  = */ src0->ne[2],
        /* .src0_nb1   = */ static_cast<int64_t>(src0->nb[1]),
        /* .src0_nb2   = */ static_cast<int64_t>(src0->nb[2]),
        /* .ids_nb0    = */ static_cast<int64_t>(src2->nb[0]),
        /* .ids_nb1    = */ static_cast<int64_t>(src2->nb[1]),
        /* .dst_nb1    = */ static_cast<int64_t>(mul->nb[1]),
        /* .dst_nb2    = */ static_cast<int64_t>(mul->nb[2]),
        /* .src1_cols  = */ src1->ne[1],
        /* .scale_nb1  = */ static_cast<int64_t>(scale->nb[1]),
    };

    const auto & provider = context->device_context->mul_mat_id_q4_k_mul_q8_1_provider;
    pyre_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>(constants.rows),
            static_cast<uint32_t>(constants.n_ids * constants.n_tokens),
            1,
        },
        /* .workgroup_size = */ {
            provider.export_info.workgroup_size[0] ? provider.export_info.workgroup_size[0] : 256,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };
    if (!GGML_PYRE_CHECK(pyre_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            5,
            PYRE_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    context->dispatch_count++;
    context->mul_mat_id_count++;

    if (!GGML_PYRE_CHECK(pyre_stream_execution_barrier(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

// buffer type interface

static const char * ggml_backend_pyre_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return ggml_backend_pyre_get_buft_context(buft)->name.c_str();
}

static void ggml_backend_pyre_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * context = ggml_backend_pyre_get_buffer_context(buffer);
    if (context->buffer) {
        pyre_buffer_release(context->buffer);
    }
    delete context;
}

static void * ggml_backend_pyre_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ggml_backend_pyre_get_buffer_context(buffer)->base;
}

static enum ggml_status ggml_backend_pyre_buffer_init_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    if (tensor->view_src) {
        GGML_ASSERT(tensor->view_src->buffer != nullptr);
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_pyre_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    auto * context = ggml_backend_pyre_get_buffer_context(buffer);
    if (size == 0 || !context->buffer) {
        return;
    }

    const size_t buffer_offset = ggml_backend_pyre_tensor_offset(context, tensor) + offset;
    if (!GGML_PYRE_CHECK(pyre_queue_fill(
            context->device_context->device, 0, nullptr, nullptr,
            context->buffer, buffer_offset, size, &value, sizeof(value)))) {
        return;
    }
    GGML_PYRE_CHECK(pyre_device_synchronize(context->device_context->device));
}

static void ggml_backend_pyre_buffer_set_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * context = ggml_backend_pyre_get_buffer_context(buffer);
    if (size == 0 || !context->buffer) {
        return;
    }

    const size_t buffer_offset = ggml_backend_pyre_tensor_offset(context, tensor) + offset;
    GGML_PYRE_CHECK(pyre_synchronous_h2d(
        context->device_context->device, data,
        context->buffer, buffer_offset, size));
}

static void ggml_backend_pyre_buffer_get_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * context = ggml_backend_pyre_get_buffer_context(buffer);
    if (size == 0 || !context->buffer) {
        return;
    }

    const size_t buffer_offset = ggml_backend_pyre_tensor_offset(context, tensor) + offset;
    GGML_PYRE_CHECK(pyre_synchronous_d2h(
        context->device_context->device, context->buffer,
        buffer_offset, data, size));
}

static bool ggml_backend_pyre_buffer_cpy_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t src_buffer = src->view_src ? src->view_src->buffer : src->buffer;
    if (!src_buffer || src_buffer->iface.get_base != ggml_backend_pyre_buffer_get_base) {
        return false;
    }

    auto * dst_context = ggml_backend_pyre_get_buffer_context(buffer);
    auto * src_context = ggml_backend_pyre_get_buffer_context(src_buffer);
    if (dst_context->device_context != src_context->device_context ||
        !dst_context->buffer || !src_context->buffer) {
        return false;
    }

    const size_t src_offset = ggml_backend_pyre_tensor_offset(src_context, src);
    const size_t dst_offset = ggml_backend_pyre_tensor_offset(dst_context, dst);
    const size_t size = ggml_nbytes(src);
    if (!GGML_PYRE_CHECK(pyre_queue_copy(
            dst_context->device_context->device, 0, nullptr, nullptr,
            src_context->buffer, src_offset,
            dst_context->buffer, dst_offset, size))) {
        return false;
    }
    return GGML_PYRE_CHECK(pyre_device_synchronize(dst_context->device_context->device));
}

static void ggml_backend_pyre_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * context = ggml_backend_pyre_get_buffer_context(buffer);
    if (buffer->size == 0 || !context->buffer) {
        return;
    }

    if (!GGML_PYRE_CHECK(pyre_queue_fill(
            context->device_context->device, 0, nullptr, nullptr,
            context->buffer, 0, buffer->size, &value, sizeof(value)))) {
        return;
    }
    GGML_PYRE_CHECK(pyre_device_synchronize(context->device_context->device));
}

static const ggml_backend_buffer_i ggml_backend_pyre_buffer_i = {
    /* .free_buffer   = */ ggml_backend_pyre_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_pyre_buffer_get_base,
    /* .init_tensor   = */ ggml_backend_pyre_buffer_init_tensor,
    /* .memset_tensor = */ ggml_backend_pyre_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_pyre_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_pyre_buffer_get_tensor,
    /* .cpy_tensor    = */ ggml_backend_pyre_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_pyre_buffer_clear,
    /* .reset         = */ nullptr,
};

static ggml_backend_buffer_t ggml_backend_pyre_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    auto * buft_context = ggml_backend_pyre_get_buft_context(buft);

    pyre_buffer_t pyre_buffer = nullptr;
    if (size > 0 &&
        !GGML_PYRE_CHECK(pyre_allocator_allocate_buffer(
            pyre_device_allocator(buft_context->device_context->device),
            buft_context->params, size, &pyre_buffer))) {
        return nullptr;
    }

    auto * context = new (std::nothrow) ggml_backend_pyre_buffer_context {
        /* .device_context = */ buft_context->device_context,
        /* .buffer         = */ pyre_buffer,
        /* .base           = */ reinterpret_cast<uint8_t *>(GGML_PYRE_FAKE_PTR_BASE),
    };
    if (!context) {
        if (pyre_buffer) {
            pyre_buffer_release(pyre_buffer);
        }
        return nullptr;
    }

    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(
        buft, ggml_backend_pyre_buffer_i, context, size);
    if (!buffer) {
        if (context->buffer) {
            pyre_buffer_release(context->buffer);
        }
        delete context;
    }
    return buffer;
}

static size_t ggml_backend_pyre_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_PYRE_ALIGNMENT;
}

static size_t ggml_backend_pyre_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * buft_context = ggml_backend_pyre_get_buft_context(buft);
    return buft_context->device_context->memory_total > 0 ?
        buft_context->device_context->memory_total :
        std::numeric_limits<size_t>::max();
}

static const ggml_backend_buffer_type_i ggml_backend_pyre_buffer_type_i = {
    /* .get_name      = */ ggml_backend_pyre_buffer_type_get_name,
    /* .alloc_buffer  = */ ggml_backend_pyre_buffer_type_alloc_buffer,
    /* .get_alignment = */ ggml_backend_pyre_buffer_type_get_alignment,
    /* .get_max_size  = */ ggml_backend_pyre_buffer_type_get_max_size,
    /* .get_alloc_size = */ nullptr,
    /* .is_host       = */ nullptr,
};

static ggml_backend_buffer_type_t ggml_backend_pyre_device_buffer_type(ggml_backend_dev_t dev) {
    auto * device_context = ggml_backend_pyre_get_device_context(dev);
    static std::vector<std::unique_ptr<ggml_backend_buffer_type>> buffer_types;
    static std::vector<std::unique_ptr<ggml_backend_pyre_buffer_type_context>> contexts;

    for (const auto & buft : buffer_types) {
        auto * context = ggml_backend_pyre_get_buft_context(buft.get());
        if (context->device_context == device_context) {
            return buft.get();
        }
    }

    auto * context = new ggml_backend_pyre_buffer_type_context {
        /* .device_context = */ device_context,
        /* .name           = */ device_context->name,
        /* .params         = */ {
            /* .type = */ PYRE_MEMORY_TYPE_DEVICE_LOCAL,
            /* .access = */ PYRE_MEMORY_ACCESS_ALL,
            /* .usage = */ PYRE_BUFFER_USAGE_DEFAULT,
            /* .queue_affinity = */ 0,
        },
    };

    auto * buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_pyre_buffer_type_i,
        /* .device  = */ dev,
        /* .context = */ context,
    };

    contexts.emplace_back(context);
    buffer_types.emplace_back(buft);
    return buft;
}

// backend interface

static const char * ggml_backend_pyre_get_name(ggml_backend_t backend) {
    return static_cast<ggml_backend_pyre_context *>(backend->context)->name.c_str();
}

static void ggml_backend_pyre_free(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_pyre_context *>(backend->context);
    if (context->device_context->policy.trace_providers) {
        GGML_LOG_INFO(
            "%s: provider summary dispatch=%" PRIu64 " rms_norm=%" PRIu64
            " elementwise=%" PRIu64 " mul_mat_vec=%" PRIu64
            " mul_mat_id=%" PRIu64 " copy=%" PRIu64
            " set_rows=%" PRIu64 " get_rows=%" PRIu64
            " concat=%" PRIu64 " soft_max=%" PRIu64
            " argsort=%" PRIu64 " rope=%" PRIu64
            " topk_moe=%" PRIu64 " unary=%" PRIu64
            " reduction=%" PRIu64 " ssm_conv=%" PRIu64
            " gated_delta_net=%" PRIu64
            " metadata=%" PRIu64 " synchronize=%" PRIu64 "\n",
            context->name.c_str(),
            context->dispatch_count,
            context->rms_norm_count,
            context->elementwise_count,
            context->mul_mat_vec_count,
            context->mul_mat_id_count,
            context->copy_count,
            context->set_rows_count,
            context->get_rows_count,
            context->concat_count,
            context->soft_max_count,
            context->argsort_count,
            context->rope_count,
            context->topk_moe_count,
            context->unary_count,
            context->reduction_count,
            context->ssm_conv_count,
            context->gated_delta_net_count,
            context->metadata_count,
            context->synchronize_count);
    }
    if (context->stream) {
        if (context->scratch_q8_1 || !context->retired_scratch_q8_1.empty()) {
            GGML_PYRE_CHECK(pyre_stream_synchronize(context->stream));
        }
        if (context->scratch_q8_1) {
            pyre_buffer_release(context->scratch_q8_1);
        }
        for (pyre_buffer_t buffer : context->retired_scratch_q8_1) {
            pyre_buffer_release(buffer);
        }
        pyre_stream_release(context->stream);
    }
    delete context;
    delete backend;
}

static void ggml_backend_pyre_synchronize(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_pyre_context *>(backend->context);
    if (context->stream) {
        GGML_PYRE_CHECK(pyre_stream_synchronize(context->stream));
        context->synchronize_count++;
    }
}

struct ggml_backend_pyre_topk_moe_fusion {
    const ggml_tensor * soft_max = nullptr;
    const ggml_tensor * weights = nullptr;
    const ggml_tensor * ids = nullptr;
    const ggml_tensor * clamp = nullptr;
    int last_idx = -1;
    int ids_idx = -1;
    int weights_idx = -1;
    std::vector<int> idxs;
    std::vector<ggml_op> ops;
};

static bool ggml_backend_pyre_try_topk_moe_fusion(
        const ggml_cgraph * cgraph,
        int start,
        const ggml_backend_pyre_device_context * device_context,
        ggml_backend_pyre_topk_moe_fusion * fusion) {
    if (start >= cgraph->n_nodes || cgraph->nodes[start]->op != GGML_OP_SOFT_MAX) {
        return false;
    }
    auto add = [&](int idx) {
        fusion->idxs.push_back(idx);
        fusion->ops.push_back(cgraph->nodes[idx]->op);
    };
    auto next = [&](int & idx) -> const ggml_tensor * {
        return idx < cgraph->n_nodes ? cgraph->nodes[idx] : nullptr;
    };

    fusion->idxs.clear();
    fusion->ops.clear();
    fusion->soft_max = cgraph->nodes[start];
    add(start);

    int idx = start + 1;
    const ggml_tensor * probs = fusion->soft_max;
    if (next(idx) && next(idx)->op == GGML_OP_RESHAPE && next(idx)->src[0] == probs) {
        probs = next(idx);
        add(idx++);
    }

    const ggml_tensor * argsort = next(idx);
    if (!argsort || argsort->op != GGML_OP_ARGSORT || argsort->src[0] != fusion->soft_max ||
        ggml_get_op_params_i32(argsort, 0) != GGML_SORT_ORDER_DESC) {
        return false;
    }
    add(idx++);

    const ggml_tensor * ids = next(idx);
    if (!ids || ids->op != GGML_OP_VIEW || ids->src[0] != argsort) {
        return false;
    }
    fusion->ids = ids;
    fusion->ids_idx = idx;
    add(idx++);

    const ggml_tensor * selected = next(idx);
    if (!selected || selected->op != GGML_OP_GET_ROWS ||
        selected->src[0] != probs || selected->src[1] != ids) {
        return false;
    }
    add(idx++);

    const ggml_tensor * weights_base = selected;
    if (next(idx) && next(idx)->op == GGML_OP_RESHAPE && next(idx)->src[0] == weights_base) {
        weights_base = next(idx);
        add(idx++);
    }

    const ggml_tensor * sum_rows = next(idx);
    if (sum_rows && sum_rows->op == GGML_OP_SUM_ROWS && sum_rows->src[0] == weights_base) {
        add(idx++);
        const ggml_tensor * clamp = next(idx);
        if (!clamp || clamp->op != GGML_OP_CLAMP || clamp->src[0] != sum_rows) {
            return false;
        }
        fusion->clamp = clamp;
        add(idx++);

        const ggml_tensor * div = next(idx);
        if (!div || div->op != GGML_OP_DIV || div->src[0] != weights_base || div->src[1] != clamp) {
            return false;
        }
        weights_base = div;
        add(idx++);

        if (next(idx) && next(idx)->op == GGML_OP_RESHAPE && next(idx)->src[0] == weights_base) {
            weights_base = next(idx);
            add(idx++);
        }
    }

    fusion->weights = weights_base;
    fusion->weights_idx = fusion->idxs.back();
    fusion->last_idx = fusion->idxs.back();

    if (!ggml_backend_pyre_supports_topk_moe_f32(
            device_context, fusion->soft_max, fusion->weights, fusion->ids)) {
        return false;
    }

    const int outputs[2] = { fusion->ids_idx, fusion->weights_idx };
    return ggml_can_fuse_subgraph_ext(
        cgraph,
        fusion->idxs.data(),
        static_cast<int>(fusion->idxs.size()),
        fusion->ops.data(),
        outputs,
        2);
}

static ggml_status ggml_backend_pyre_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * context = static_cast<ggml_backend_pyre_context *>(backend->context);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        ggml_backend_pyre_topk_moe_fusion fusion;
        if (!context->device_context->policy.disable_fusion &&
            ggml_backend_pyre_try_topk_moe_fusion(cgraph, i, context->device_context, &fusion)) {
            ggml_backend_pyre_trace_provider(
                context->device_context,
                "claim TOPK_MOE provider=pure_hip_f32 experts=%" PRId64 " k=%" PRId64 " nrows=%" PRId64 "\n",
                fusion.soft_max->src[0]->ne[0], ggml_nelements(fusion.weights), ggml_nrows(fusion.soft_max->src[0]));
            if (ggml_backend_pyre_dispatch_topk_moe_f32(
                    context, fusion.soft_max, fusion.weights, fusion.ids, fusion.clamp) != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i = fusion.last_idx;
            continue;
        }
        if (node->op == GGML_OP_RMS_NORM &&
            !context->device_context->policy.disable_fusion &&
            i + 1 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_MUL &&
            ggml_backend_pyre_supports_rms_norm_mul(context->device_context, node, cgraph->nodes[i + 1]) &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, { i + 1 })) {
            const ggml_tensor * mul = cgraph->nodes[i + 1];
            ggml_backend_pyre_trace_provider(
                context->device_context, "claim RMS_NORM_MUL provider=pure_hip_f32 ncols=%" PRId64 " nrows=%" PRId64 "\n",
                node->src[0]->ne[0], ggml_nrows(node->src[0]));
            if (ggml_backend_pyre_dispatch_rms_norm_mul(context, node, mul) != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i++;
            continue;
        }
        if (node->op == GGML_OP_MUL_MAT_ID &&
            !context->device_context->policy.disable_fusion &&
            i + 1 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_MUL &&
            ggml_backend_pyre_supports_mul_mat_id_q4_k_mul(context->device_context, node, cgraph->nodes[i + 1]) &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_MUL }, { i + 1 })) {
            const ggml_tensor * mul = cgraph->nodes[i + 1];
            const bool use_q8_1 =
                ggml_backend_pyre_supports_mul_mat_id_q4_k_mul_q8_1(context->device_context, node, mul);
            ggml_backend_pyre_trace_provider(
                context->device_context,
                "claim MUL_MAT_ID_MUL provider=pure_hip_q4_K%s k=%" PRId64
                " rows=%" PRId64 " ids=%" PRId64 " tokens=%" PRId64 "\n",
                use_q8_1 ? "_q8_1" : "",
                node->src[0]->ne[0], node->src[0]->ne[1], node->src[2]->ne[0], node->src[2]->ne[1]);
            const ggml_status status = use_q8_1 ?
                ggml_backend_pyre_dispatch_mul_mat_id_q4_k_mul_q8_1(context, node, mul) :
                ggml_backend_pyre_dispatch_mul_mat_id_q4_k_mul(context, node, mul);
            if (status != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i++;
            continue;
        }
        if (node->op == GGML_OP_SSM_CONV &&
            !context->device_context->policy.disable_fusion &&
            i + 1 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_UNARY &&
            ggml_backend_pyre_supports_ssm_conv_silu(context->device_context, node, cgraph->nodes[i + 1]) &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { i + 1 })) {
            const ggml_tensor * silu = cgraph->nodes[i + 1];
            ggml_backend_pyre_trace_provider(
                context->device_context,
                "claim SSM_CONV_SILU provider=pure_hip_f32 d_conv=%" PRId64
                " d_inner=%" PRId64 " n_tokens=%" PRId64 " n_seqs=%" PRId64 "\n",
                node->src[1]->ne[0], node->src[0]->ne[1], node->ne[1], node->ne[2]);
            if (ggml_backend_pyre_dispatch_ssm_conv(context, node, silu, true) != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i++;
            continue;
        }
        if (node->op == GGML_OP_UNARY &&
            !context->device_context->policy.disable_fusion &&
            i + 1 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_MUL &&
            ggml_backend_pyre_supports_silu_mul_f32(context->device_context, node, cgraph->nodes[i + 1]) &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { i + 1 })) {
            const ggml_tensor * mul = cgraph->nodes[i + 1];
            ggml_backend_pyre_trace_provider(
                context->device_context, "claim SILU_MUL provider=pure_hip_f32 n=%" PRId64 "\n",
                ggml_nelements(mul));
            if (ggml_backend_pyre_dispatch_silu_mul_f32(context, node, mul) != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i++;
            continue;
        }
        if (node->op == GGML_OP_ADD &&
            !context->device_context->policy.disable_fusion &&
            !context->device_context->policy.disable_add_rms_norm_mul_fusion &&
            i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_RMS_NORM &&
            cgraph->nodes[i + 2]->op == GGML_OP_MUL &&
            ggml_backend_pyre_supports_add_rms_norm_mul_broadcast(
                context->device_context, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]) &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL }, { i, i + 2 })) {
            const ggml_tensor * rms_norm = cgraph->nodes[i + 1];
            const ggml_tensor * mul = cgraph->nodes[i + 2];
            ggml_backend_pyre_trace_provider(
                context->device_context,
                "claim ADD_RMS_NORM_MUL provider=pure_hip_f32_broadcast ncols=%" PRId64 " nrows=%" PRId64 "\n",
                node->ne[0], ggml_nrows(node));
            if (ggml_backend_pyre_dispatch_add_rms_norm_mul_broadcast(context, node, rms_norm, mul) !=
                    GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i += 2;
            continue;
        }
        if (node->op == GGML_OP_ADD &&
            !context->device_context->policy.disable_fusion &&
            !context->device_context->policy.disable_add_add_fusion &&
            i + 1 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_ADD &&
            ggml_backend_pyre_supports_add_add_broadcast(context->device_context, node, cgraph->nodes[i + 1]) &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_ADD, GGML_OP_ADD }, { i + 1 })) {
            const ggml_tensor * second = cgraph->nodes[i + 1];
            ggml_backend_pyre_trace_provider(
                context->device_context, "claim ADD_ADD provider=pure_hip_f32_broadcast n=%" PRId64 "\n",
                ggml_nelements(second));
            if (ggml_backend_pyre_dispatch_add_add_broadcast_f32(context, node, second) != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i++;
            continue;
        }
        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                context->metadata_count++;
                break;
            case GGML_OP_RMS_NORM:
                if (!ggml_backend_pyre_supports_rms_norm(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: RMS_NORM shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context, "claim RMS_NORM provider=pure_hip ncols=%" PRId64 " nrows=%" PRId64 "\n",
                    node->src[0]->ne[0], ggml_nrows(node->src[0]));
                if (ggml_backend_pyre_dispatch_rms_norm(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_ADD: {
                if (!ggml_backend_pyre_supports_add(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: ADD shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context, "claim ADD provider=pure_hip_f32%s n=%" PRId64 "\n",
                    ggml_backend_pyre_supports_add_broadcast(context->device_context, node) ? "_broadcast" : "",
                    ggml_nelements(node));
                ggml_status status = GGML_STATUS_FAILED;
                if (ggml_backend_pyre_supports_add_broadcast(context->device_context, node)) {
                    status = ggml_backend_pyre_dispatch_add_broadcast_f32(context, node);
                } else {
                    status = ggml_backend_pyre_dispatch_binary_elementwise_f32(
                        context, node, context->device_context->add_provider, "ADD");
                }
                if (status != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            }
            case GGML_OP_MUL: {
                if (!ggml_backend_pyre_supports_mul(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: MUL shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context, "claim MUL provider=pure_hip_f32%s n=%" PRId64 "\n",
                    ggml_backend_pyre_supports_mul_broadcast(context->device_context, node) ? "_broadcast" : "",
                    ggml_nelements(node));
                ggml_status status = GGML_STATUS_FAILED;
                if (ggml_backend_pyre_supports_mul_broadcast(context->device_context, node)) {
                    status = ggml_backend_pyre_dispatch_mul_broadcast_f32(context, node);
                } else {
                    status = ggml_backend_pyre_dispatch_binary_elementwise_f32(
                        context, node, context->device_context->mul_provider, "MUL");
                }
                if (status != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            }
            case GGML_OP_DIV:
                if (!ggml_backend_pyre_supports_div_broadcast(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: DIV shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context, "claim DIV provider=pure_hip_f32_broadcast n=%" PRId64 "\n",
                    ggml_nelements(node));
                if (ggml_backend_pyre_dispatch_div_broadcast_f32(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_SCALE:
                if (!ggml_backend_pyre_supports_scale(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: SCALE shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context, "claim SCALE provider=pure_hip_f32 n=%" PRId64 "\n",
                    ggml_nelements(node));
                if (ggml_backend_pyre_dispatch_scale_f32(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_CPY:
                if (!ggml_backend_pyre_supports_cpy(node)) {
                    GGML_LOG_ERROR("%s: CPY shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context, "claim CPY provider=buffer_copy type=%s nbytes=%zu\n",
                    ggml_type_name(node->type), ggml_nbytes(node));
                if (ggml_backend_pyre_dispatch_cpy(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_CONT:
                if (!ggml_backend_pyre_supports_cont(node)) {
                    GGML_LOG_ERROR("%s: CONT shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context, "claim CONT provider=buffer_copy type=%s nbytes=%zu\n",
                    ggml_type_name(node->type), ggml_nbytes(node));
                if (ggml_backend_pyre_dispatch_cpy(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_SET_ROWS:
                if (!ggml_backend_pyre_supports_set_rows(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: SET_ROWS shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context,
                    "claim SET_ROWS provider=pure_hip_f32_%s nc=%" PRId64 " nr=%" PRId64 "\n",
                    ggml_type_name(node->type), node->src[0]->ne[0], node->src[0]->ne[1]);
                if (ggml_backend_pyre_dispatch_set_rows(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_GET_ROWS:
                if (!ggml_backend_pyre_supports_get_rows_f32(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: GET_ROWS shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context, "claim GET_ROWS provider=pure_hip_%s nc=%" PRId64 " nr=%" PRId64 "\n",
                    node->src[0]->type == GGML_TYPE_Q5_K ? "q5_K" : "f32",
                    node->src[0]->ne[0], ggml_nelements(node->src[1]));
                if (ggml_backend_pyre_dispatch_get_rows_f32(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_CONCAT:
                if (!ggml_backend_pyre_supports_concat_f32(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: CONCAT shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context, "claim CONCAT provider=pure_hip_f32 n=%" PRId64 "\n",
                    ggml_nelements(node));
                if (ggml_backend_pyre_dispatch_concat_f32(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_SOFT_MAX:
                if (!ggml_backend_pyre_supports_soft_max_f32(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: SOFT_MAX shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context, "claim SOFT_MAX provider=pure_hip_f32 ncols=%" PRId64 " nrows=%" PRId64 "\n",
                    node->src[0]->ne[0], ggml_nrows(node->src[0]));
                if (ggml_backend_pyre_dispatch_soft_max_f32(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_ARGSORT:
                if (!ggml_backend_pyre_supports_argsort_f32(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: ARGSORT shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context, "claim ARGSORT provider=pure_hip_f32 ncols=%" PRId64 " nrows=%" PRId64 "\n",
                    node->src[0]->ne[0], ggml_nrows(node->src[0]));
                if (ggml_backend_pyre_dispatch_argsort_f32(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_ROPE:
                if (!ggml_backend_pyre_supports_rope_f32(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: ROPE shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context, "claim ROPE provider=pure_hip_f32 n_dims=%d nrows=%" PRId64 "\n",
                    ggml_get_op_params_i32(node, 1), ggml_nrows(node->src[0]));
                if (ggml_backend_pyre_dispatch_rope_f32(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_UNARY: {
                if (!ggml_backend_pyre_supports_unary_f32(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: UNARY shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                const ggml_backend_pyre_op_provider * provider =
                    ggml_backend_pyre_unary_provider(context->device_context, node);
                ggml_backend_pyre_trace_provider(
                    context->device_context, "claim %s provider=pure_hip_f32 n=%" PRId64 "\n",
                    ggml_op_desc(node), ggml_nelements(node));
                if (ggml_backend_pyre_dispatch_unary_f32(
                        context, node, *provider, ggml_op_desc(node)) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            }
            case GGML_OP_GLU:
                if (!ggml_backend_pyre_supports_swiglu_f32(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: GLU shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context, "claim SWIGLU provider=pure_hip_f32 n=%" PRId64 "\n",
                    ggml_nelements(node));
                if (ggml_backend_pyre_dispatch_swiglu_f32(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_SUM_ROWS:
                if (!ggml_backend_pyre_supports_sum_rows(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: SUM_ROWS shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context, "claim SUM_ROWS provider=pure_hip_f32 nrows=%" PRId64 "\n",
                    ggml_nrows(node->src[0]));
                if (ggml_backend_pyre_dispatch_sum_rows(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_L2_NORM:
                if (!ggml_backend_pyre_supports_l2_norm(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: L2_NORM shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context, "claim L2_NORM provider=pure_hip_f32 ncols=%" PRId64 " nrows=%" PRId64 "\n",
                    node->src[0]->ne[0], ggml_nrows(node->src[0]));
                if (ggml_backend_pyre_dispatch_l2_norm(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_CLAMP:
                if (!ggml_backend_pyre_supports_clamp(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: CLAMP shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context, "claim CLAMP provider=pure_hip_f32 n=%" PRId64 "\n",
                    ggml_nelements(node));
                if (ggml_backend_pyre_dispatch_clamp(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_SSM_CONV:
                if (!ggml_backend_pyre_supports_ssm_conv(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: SSM_CONV shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context,
                    "claim SSM_CONV provider=pure_hip_f32 d_conv=%" PRId64
                    " d_inner=%" PRId64 " n_tokens=%" PRId64 " n_seqs=%" PRId64 "\n",
                    node->src[1]->ne[0], node->src[0]->ne[1], node->ne[1], node->ne[2]);
                if (ggml_backend_pyre_dispatch_ssm_conv(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_GATED_DELTA_NET:
                if (!ggml_backend_pyre_supports_gated_delta_net(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: GATED_DELTA_NET shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context,
                    "claim GATED_DELTA_NET provider=pure_hip_f32 S_v=%" PRId64
                    " H=%" PRId64 " tokens=%" PRId64 " seqs=%" PRId64 "\n",
                    node->src[2]->ne[0], node->src[2]->ne[1], node->src[2]->ne[2], node->src[2]->ne[3]);
                if (ggml_backend_pyre_dispatch_gated_delta_net(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_MUL_MAT:
                if (!ggml_backend_pyre_supports_mul_mat_vec(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: MUL_MAT shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context,
                    "claim MUL_MAT provider=pure_hip_%s%s k=%" PRId64
                    " rows=%" PRId64 " cols=%" PRId64 " ne2=%" PRId64 "\n",
                    ggml_type_name(node->src[0]->type),
                    ggml_backend_pyre_supports_mul_mat_vec_q4_k_q8_1(context->device_context, node) ? "_q8_1" :
                        ((ggml_backend_pyre_supports_mul_mat_vec_f16_batched(context->device_context, node) ||
                          ggml_backend_pyre_supports_mul_mat_vec_f32_batched(context->device_context, node)) ? "_batched" : ""),
                    node->src[0]->ne[0], node->src[0]->ne[1], node->src[1]->ne[1], node->ne[2]);
                if (ggml_backend_pyre_dispatch_mul_mat_vec_f16(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_MUL_MAT_ID: {
                if (!ggml_backend_pyre_supports_mul_mat_id_q4_k(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: MUL_MAT_ID shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context,
                    "claim MUL_MAT_ID provider=pure_hip_q4_K%s k=%" PRId64
                    " rows=%" PRId64 " ids=%" PRId64 " tokens=%" PRId64 "\n",
                    ggml_backend_pyre_supports_mul_mat_id_q4_k_q8_1(context->device_context, node) ? "_q8_1" : "",
                    node->src[0]->ne[0], node->src[0]->ne[1], node->src[2]->ne[0], node->src[2]->ne[1]);
                const ggml_status status =
                    ggml_backend_pyre_supports_mul_mat_id_q4_k_q8_1(context->device_context, node) ?
                    ggml_backend_pyre_dispatch_mul_mat_id_q4_k_q8_1(context, node) :
                    ggml_backend_pyre_dispatch_mul_mat_id_q4_k(context, node);
                if (status != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            }
            default:
                GGML_LOG_ERROR("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
                return GGML_STATUS_FAILED;
        }
    }

    ggml_backend_pyre_synchronize(backend);
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_pyre_i = {
    /* .get_name           = */ ggml_backend_pyre_get_name,
    /* .free               = */ ggml_backend_pyre_free,
    /* .set_tensor_async   = */ nullptr,
    /* .get_tensor_async   = */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,
    /* .synchronize        = */ ggml_backend_pyre_synchronize,
    /* .graph_plan_create  = */ nullptr,
    /* .graph_plan_free    = */ nullptr,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute      = */ ggml_backend_pyre_graph_compute,
    /* .event_record       = */ nullptr,
    /* .event_wait         = */ nullptr,
    /* .graph_optimize     = */ nullptr,
};

// device interface

static const char * ggml_backend_pyre_device_get_name(ggml_backend_dev_t dev) {
    return ggml_backend_pyre_get_device_context(dev)->name.c_str();
}

static const char * ggml_backend_pyre_device_get_description(ggml_backend_dev_t dev) {
    return ggml_backend_pyre_get_device_context(dev)->description.c_str();
}

static void ggml_backend_pyre_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * context = ggml_backend_pyre_get_device_context(dev);
    *free = context->memory_total;
    *total = context->memory_total;
}

static enum ggml_backend_dev_type ggml_backend_pyre_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_pyre_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = ggml_backend_pyre_device_get_name(dev);
    props->description = ggml_backend_pyre_device_get_description(dev);
    props->type = ggml_backend_pyre_device_get_type(dev);
    ggml_backend_pyre_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = nullptr;
    props->caps = {
        /* .async = */ true,
        /* .host_buffer = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events = */ false,
    };
}

static ggml_backend_t ggml_backend_pyre_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);

    auto * device_context = ggml_backend_pyre_get_device_context(dev);
    pyre_stream_t stream = nullptr;
    if (!GGML_PYRE_CHECK(pyre_stream_create(device_context->device, 0, &stream))) {
        return nullptr;
    }

    auto * context = new (std::nothrow) ggml_backend_pyre_context {
        /* .device_context = */ device_context,
        /* .stream         = */ stream,
        /* .name           = */ device_context->name,
        /* .scratch_q8_1   = */ nullptr,
        /* .scratch_q8_1_size = */ 0,
        /* .retired_scratch_q8_1 = */ {},
    };
    if (!context) {
        pyre_stream_release(stream);
        return nullptr;
    }

    ggml_backend_t backend = new (std::nothrow) ggml_backend {
        /* .guid    = */ ggml_backend_pyre_guid(),
        /* .iface   = */ ggml_backend_pyre_i,
        /* .device  = */ dev,
        /* .context = */ context,
    };
    if (!backend) {
        pyre_stream_release(stream);
        delete context;
        return nullptr;
    }

    return backend;
}

static bool ggml_backend_pyre_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    auto * context = ggml_backend_pyre_get_device_context(dev);
    bool supported = false;
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            supported = true;
            break;
        case GGML_OP_RMS_NORM:
            supported = ggml_backend_pyre_supports_rms_norm(context, op);
            break;
        case GGML_OP_ADD:
            supported = ggml_backend_pyre_supports_add(context, op);
            break;
        case GGML_OP_MUL:
            supported = ggml_backend_pyre_supports_mul(context, op);
            break;
        case GGML_OP_DIV:
            supported = ggml_backend_pyre_supports_div_broadcast(context, op);
            break;
        case GGML_OP_SCALE:
            supported = ggml_backend_pyre_supports_scale(context, op);
            break;
        case GGML_OP_CPY:
            supported = ggml_backend_pyre_supports_cpy(op);
            break;
        case GGML_OP_CONT:
            supported = ggml_backend_pyre_supports_cont(op);
            break;
        case GGML_OP_SET_ROWS:
            supported = ggml_backend_pyre_supports_set_rows(context, op);
            break;
        case GGML_OP_GET_ROWS:
            supported = ggml_backend_pyre_supports_get_rows_f32(context, op);
            break;
        case GGML_OP_CONCAT:
            supported = ggml_backend_pyre_supports_concat_f32(context, op);
            break;
        case GGML_OP_SOFT_MAX:
            supported = ggml_backend_pyre_supports_soft_max_f32(context, op);
            break;
        case GGML_OP_ARGSORT:
            supported = ggml_backend_pyre_supports_argsort_f32(context, op);
            break;
        case GGML_OP_ROPE:
            supported = ggml_backend_pyre_supports_rope_f32(context, op);
            break;
        case GGML_OP_UNARY:
            supported = ggml_backend_pyre_supports_unary_f32(context, op);
            break;
        case GGML_OP_GLU:
            supported = ggml_backend_pyre_supports_swiglu_f32(context, op);
            break;
        case GGML_OP_SUM_ROWS:
            supported = ggml_backend_pyre_supports_sum_rows(context, op);
            break;
        case GGML_OP_L2_NORM:
            supported = ggml_backend_pyre_supports_l2_norm(context, op);
            break;
        case GGML_OP_CLAMP:
            supported = ggml_backend_pyre_supports_clamp(context, op);
            break;
        case GGML_OP_SSM_CONV:
            supported = ggml_backend_pyre_supports_ssm_conv(context, op);
            break;
        case GGML_OP_GATED_DELTA_NET:
            supported = ggml_backend_pyre_supports_gated_delta_net(context, op);
            break;
        case GGML_OP_MUL_MAT:
            supported = ggml_backend_pyre_supports_mul_mat_vec(context, op);
            break;
        case GGML_OP_MUL_MAT_ID:
            supported = ggml_backend_pyre_supports_mul_mat_id_q4_k(context, op);
            break;
        default:
            supported = false;
            break;
    }
    if (context->policy.trace_providers && !supported && op->op != GGML_OP_NONE) {
        if (op->op == GGML_OP_MUL_MAT && context->mul_mat_fallback_trace_count < GGML_PYRE_TRACE_MUL_MAT_DETAIL_LIMIT) {
            context->mul_mat_fallback_trace_count++;
            ggml_backend_pyre_trace_provider(
                context,
                "fallback MUL_MAT reason=%s\n",
                ggml_backend_pyre_mul_mat_vec_unsupported_reason(context, op));
            ggml_backend_pyre_trace_tensor(context, "  src0", op->src[0]);
            ggml_backend_pyre_trace_tensor(context, "  src1", op->src[1]);
            ggml_backend_pyre_trace_tensor(context, "  dst", op);
        } else if (op->op == GGML_OP_MUL_MAT_ID &&
                   context->mul_mat_id_fallback_trace_count < GGML_PYRE_TRACE_MUL_MAT_DETAIL_LIMIT) {
            context->mul_mat_id_fallback_trace_count++;
            ggml_backend_pyre_trace_provider(context, "fallback MUL_MAT_ID\n");
            ggml_backend_pyre_trace_tensor(context, "  src0", op->src[0]);
            ggml_backend_pyre_trace_tensor(context, "  src1", op->src[1]);
            ggml_backend_pyre_trace_tensor(context, "  src2", op->src[2]);
            ggml_backend_pyre_trace_tensor(context, "  dst", op);
        } else if (context->fallback_trace_count < GGML_PYRE_TRACE_FALLBACK_LIMIT) {
            context->fallback_trace_count++;
            ggml_backend_pyre_trace_provider(context, "fallback %s\n", ggml_op_desc(op));
            ggml_backend_pyre_trace_tensor(context, "  src0", op->src[0]);
            ggml_backend_pyre_trace_tensor(context, "  src1", op->src[1]);
            ggml_backend_pyre_trace_tensor(context, "  src2", op->src[2]);
            ggml_backend_pyre_trace_tensor(context, "  dst", op);
            if (context->fallback_trace_count == GGML_PYRE_TRACE_FALLBACK_LIMIT) {
                ggml_backend_pyre_trace_provider(
                    context,
                    "fallback trace limit reached; suppressing additional non-MUL_MAT fallback logs\n");
            }
        }
    }
    return supported;
}

static bool ggml_backend_pyre_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_pyre_buffer_type_get_name) {
        return false;
    }
    return buft->device == dev;
}

static const ggml_backend_device_i ggml_backend_pyre_device_i = {
    /* .get_name             = */ ggml_backend_pyre_device_get_name,
    /* .get_description      = */ ggml_backend_pyre_device_get_description,
    /* .get_memory           = */ ggml_backend_pyre_device_get_memory,
    /* .get_type             = */ ggml_backend_pyre_device_get_type,
    /* .get_props            = */ ggml_backend_pyre_device_get_props,
    /* .init_backend         = */ ggml_backend_pyre_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_pyre_device_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_pyre_device_supports_op,
    /* .supports_buft        = */ ggml_backend_pyre_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// backend reg interface

static ggml_backend_pyre_reg_context * ggml_backend_pyre_get_reg_context(ggml_backend_reg_t reg) {
    return static_cast<ggml_backend_pyre_reg_context *>(reg->context);
}

static const char * ggml_backend_pyre_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_PYRE_NAME;
}

static size_t ggml_backend_pyre_reg_get_device_count(ggml_backend_reg_t reg) {
    return ggml_backend_pyre_get_reg_context(reg)->devices.size();
}

static ggml_backend_dev_t ggml_backend_pyre_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * context = ggml_backend_pyre_get_reg_context(reg);
    GGML_ASSERT(index < context->devices.size());
    return &context->devices[index];
}

static void * ggml_backend_pyre_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_pyre_reg_i = {
    /* .get_name         = */ ggml_backend_pyre_reg_get_name,
    /* .get_device_count = */ ggml_backend_pyre_reg_get_device_count,
    /* .get_device       = */ ggml_backend_pyre_reg_get_device,
    /* .get_proc_address = */ ggml_backend_pyre_reg_get_proc_address,
};

static std::unique_ptr<ggml_backend_pyre_reg_context> ggml_backend_pyre_create_reg_context() {
    auto context = std::make_unique<ggml_backend_pyre_reg_context>();

    pyre_status_t status = pyre_gpu_initialize(0);
    if (pyre_status_is_ok(status)) {
        context->gpu_initialized = true;
    } else if (pyre_status_code(status) == PYRE_STATUS_ALREADY_EXISTS) {
        pyre_status_ignore(status);
    } else {
        pyre_status_ignore(status);
        return context;
    }

    int device_count = 0;
    if (!GGML_PYRE_CHECK(pyre_gpu_device_count(&device_count)) || device_count <= 0) {
        return context;
    }

    context->device_contexts.reserve(device_count);
    context->devices.reserve(device_count);

    for (int i = 0; i < device_count; ++i) {
        pyre_device_t device = nullptr;
        if (!GGML_PYRE_CHECK(pyre_gpu_device_get(i, &device)) || !device) {
            continue;
        }
        pyre_device_retain(device);

        auto device_context = std::make_unique<ggml_backend_pyre_device_context>();
        device_context->device = device;
        device_context->name = std::string(GGML_PYRE_NAME) + std::to_string(i);
        device_context->description = ggml_backend_pyre_device_description(device);
        device_context->architecture = ggml_backend_pyre_device_architecture(device);
        device_context->memory_total = ggml_backend_pyre_total_memory(device);
        device_context->policy = ggml_backend_pyre_provider_policy_from_env();
        if (device_context->policy.trace_providers) {
            GGML_LOG_INFO(
                "%s: providers kernel=%s rms_norm=%s elementwise=%s mul_mat_vec=%s mul_mat_id=%s\n",
                device_context->name.c_str(),
                ggml_backend_pyre_kernel_provider_mode_name(device_context->policy.kernel_provider),
                device_context->policy.disable_rms_norm ? "disabled" : "enabled",
                device_context->policy.kernel_provider == ggml_backend_pyre_kernel_provider_mode::pure_hip ? "enabled" : "disabled",
                device_context->policy.disable_mul_mat_vec ? "disabled" : "enabled",
                device_context->policy.disable_mul_mat_id ? "disabled" : "enabled");
        }
        if (device_context->policy.kernel_provider == ggml_backend_pyre_kernel_provider_mode::pure_hip &&
            !device_context->policy.disable_rms_norm) {
            (void) ggml_backend_pyre_load_rms_norm_provider(device_context.get());
            (void) ggml_backend_pyre_load_rms_norm_mul_provider(device_context.get());
            (void) ggml_backend_pyre_load_add_rms_norm_mul_broadcast_provider(device_context.get());
        }
        if (device_context->policy.kernel_provider == ggml_backend_pyre_kernel_provider_mode::pure_hip) {
            (void) ggml_backend_pyre_load_add_provider(device_context.get());
            (void) ggml_backend_pyre_load_add_broadcast_provider(device_context.get());
            (void) ggml_backend_pyre_load_add_add_broadcast_provider(device_context.get());
            (void) ggml_backend_pyre_load_mul_provider(device_context.get());
            (void) ggml_backend_pyre_load_mul_broadcast_provider(device_context.get());
            (void) ggml_backend_pyre_load_div_broadcast_provider(device_context.get());
            (void) ggml_backend_pyre_load_scale_provider(device_context.get());
            (void) ggml_backend_pyre_load_set_rows_f32_provider(device_context.get());
            (void) ggml_backend_pyre_load_set_rows_f16_provider(device_context.get());
            (void) ggml_backend_pyre_load_silu_provider(device_context.get());
            (void) ggml_backend_pyre_load_sigmoid_provider(device_context.get());
            (void) ggml_backend_pyre_load_softplus_provider(device_context.get());
            (void) ggml_backend_pyre_load_swiglu_provider(device_context.get());
            (void) ggml_backend_pyre_load_sum_rows_provider(device_context.get());
            (void) ggml_backend_pyre_load_l2_norm_provider(device_context.get());
            (void) ggml_backend_pyre_load_clamp_provider(device_context.get());
            (void) ggml_backend_pyre_load_get_rows_f32_provider(device_context.get());
            (void) ggml_backend_pyre_load_get_rows_q5_k_provider(device_context.get());
            (void) ggml_backend_pyre_load_concat_f32_provider(device_context.get());
            (void) ggml_backend_pyre_load_soft_max_f32_provider(device_context.get());
            (void) ggml_backend_pyre_load_soft_max_f32_mask_provider(device_context.get());
            (void) ggml_backend_pyre_load_argsort_f32_provider(device_context.get());
            (void) ggml_backend_pyre_load_rope_f32_provider(device_context.get());
            (void) ggml_backend_pyre_load_topk_moe_f32_provider(device_context.get());
            (void) ggml_backend_pyre_load_topk_moe_f32_subgroup_provider(device_context.get());
            (void) ggml_backend_pyre_load_ssm_conv_provider(device_context.get());
            (void) ggml_backend_pyre_load_gated_delta_net_provider(device_context.get());
        }
        if (device_context->policy.kernel_provider == ggml_backend_pyre_kernel_provider_mode::pure_hip &&
            !device_context->policy.disable_mul_mat_vec) {
            (void) ggml_backend_pyre_load_mul_mat_vec_bf16_provider(device_context.get());
            (void) ggml_backend_pyre_load_mul_mat_vec_f16_provider(device_context.get());
            (void) ggml_backend_pyre_load_mul_mat_vec_f16_batched_provider(device_context.get());
            (void) ggml_backend_pyre_load_mul_mat_vec_f32_provider(device_context.get());
            (void) ggml_backend_pyre_load_mul_mat_vec_f32_batched_provider(device_context.get());
            (void) ggml_backend_pyre_load_mul_mat_id_q4_k_provider(device_context.get());
            (void) ggml_backend_pyre_load_mul_mat_id_q4_k_q8_1_provider(device_context.get());
            (void) ggml_backend_pyre_load_mul_mat_id_q4_k_mul_provider(device_context.get());
            (void) ggml_backend_pyre_load_mul_mat_id_q4_k_mul_q8_1_provider(device_context.get());
            (void) ggml_backend_pyre_load_mul_mat_vec_q4_k_provider(device_context.get());
            (void) ggml_backend_pyre_load_quantize_q8_1_provider(device_context.get());
            (void) ggml_backend_pyre_load_mul_mat_vec_q4_k_q8_1_provider(device_context.get());
            (void) ggml_backend_pyre_load_mul_mat_vec_q5_k_provider(device_context.get());
            (void) ggml_backend_pyre_load_mul_mat_vec_q6_k_provider(device_context.get());
            (void) ggml_backend_pyre_load_mul_mat_vec_q8_0_provider(device_context.get());
        }

        context->device_contexts.emplace_back(std::move(device_context));
        context->devices.push_back({
            /* .iface   = */ ggml_backend_pyre_device_i,
            /* .reg     = */ nullptr,
            /* .context = */ context->device_contexts.back().get(),
        });
    }

    return context;
}

} // namespace

ggml_backend_t ggml_backend_pyre_init(size_t dev_num) {
    ggml_backend_reg_t reg = ggml_backend_pyre_reg();
    if (!reg || dev_num >= ggml_backend_reg_dev_count(reg)) {
        GGML_LOG_ERROR("%s: invalid PYRE device index %zu\n", __func__, dev_num);
        return nullptr;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, dev_num), nullptr);
}

bool ggml_backend_is_pyre(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_pyre_guid());
}

int ggml_backend_pyre_get_device_count(void) {
    ggml_backend_reg_t reg = ggml_backend_pyre_reg();
    return reg ? static_cast<int>(ggml_backend_reg_dev_count(reg)) : 0;
}

void ggml_backend_pyre_get_device_description(int device, char * description, size_t description_size) {
    if (!description || description_size == 0) {
        return;
    }

    ggml_backend_reg_t reg = ggml_backend_pyre_reg();
    if (!reg || device < 0 || static_cast<size_t>(device) >= ggml_backend_reg_dev_count(reg)) {
        description[0] = '\0';
        return;
    }

    const char * value = ggml_backend_dev_description(
        ggml_backend_reg_dev_get(reg, static_cast<size_t>(device)));
    std::snprintf(description, description_size, "%s", value ? value : "");
}

void ggml_backend_pyre_get_device_memory(int device, size_t * free, size_t * total) {
    if (free) {
        *free = 0;
    }
    if (total) {
        *total = 0;
    }

    ggml_backend_reg_t reg = ggml_backend_pyre_reg();
    if (!reg || device < 0 || static_cast<size_t>(device) >= ggml_backend_reg_dev_count(reg)) {
        return;
    }

    ggml_backend_dev_memory(
        ggml_backend_reg_dev_get(reg, static_cast<size_t>(device)), free, total);
}

ggml_backend_buffer_type_t ggml_backend_pyre_buffer_type(size_t dev_num) {
    ggml_backend_reg_t reg = ggml_backend_pyre_reg();
    if (!reg || dev_num >= ggml_backend_reg_dev_count(reg)) {
        return nullptr;
    }
    return ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(reg, dev_num));
}

ggml_backend_reg_t ggml_backend_pyre_reg(void) {
    static std::unique_ptr<ggml_backend_pyre_reg_context> context =
        ggml_backend_pyre_create_reg_context();

    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_pyre_reg_i,
        /* .context     = */ context.get(),
    };

    if (context) {
        for (auto & device : context->devices) {
            device.reg = &reg;
        }
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_pyre_reg)
