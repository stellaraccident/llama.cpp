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
    bool disable_rms_norm = false;
    bool disable_mul_mat_vec = false;
    bool disable_mul_mat_id = false;
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
    ggml_backend_pyre_op_provider add_provider;
    ggml_backend_pyre_op_provider mul_provider;
    ggml_backend_pyre_op_provider scale_provider;
    ggml_backend_pyre_op_provider set_rows_f32_provider;
    ggml_backend_pyre_op_provider set_rows_f16_provider;
    ggml_backend_pyre_op_provider mul_mat_vec_bf16_provider;
    ggml_backend_pyre_op_provider mul_mat_vec_f16_provider;
    ggml_backend_pyre_op_provider mul_mat_vec_f32_provider;
    ggml_backend_pyre_op_provider mul_mat_vec_q4_k_provider;
    ggml_backend_pyre_op_provider mul_mat_vec_q5_k_provider;
    ggml_backend_pyre_op_provider mul_mat_vec_q6_k_provider;
    ggml_backend_pyre_op_provider mul_mat_vec_q8_0_provider;
    uint32_t fallback_trace_count = 0;
    uint32_t mul_mat_fallback_trace_count = 0;
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
    uint64_t dispatch_count = 0;
    uint64_t rms_norm_count = 0;
    uint64_t elementwise_count = 0;
    uint64_t mul_mat_vec_count = 0;
    uint64_t copy_count = 0;
    uint64_t set_rows_count = 0;
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
        /* .disable_rms_norm   = */ ggml_backend_pyre_env_enabled("GGML_PYRE_DISABLE_RMS_NORM"),
        /* .disable_mul_mat_vec = */ ggml_backend_pyre_env_enabled("GGML_PYRE_DISABLE_MUL_MAT_VEC"),
        /* .disable_mul_mat_id = */ ggml_backend_pyre_env_enabled("GGML_PYRE_DISABLE_MUL_MAT_ID"),
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

static bool ggml_backend_pyre_load_add_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_add_f32"),
        &device_context->add_provider);
}

static bool ggml_backend_pyre_load_mul_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_mul_f32"),
        &device_context->mul_provider);
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

static bool ggml_backend_pyre_load_mul_mat_vec_f16_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_mul_mat_vec_f16_f32"),
        &device_context->mul_mat_vec_f16_provider);
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

static bool ggml_backend_pyre_load_mul_mat_vec_q4_k_provider(
        ggml_backend_pyre_device_context * device_context) {
    return ggml_backend_pyre_load_catalog_provider(
        device_context,
        ggml_backend_pyre_find_catalog_entry("pyre_mul_mat_vec_q4_k_f32"),
        &device_context->mul_mat_vec_q4_k_provider);
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
           ggml_is_contiguous(op->src[0]) &&
           ggml_is_contiguous(op) &&
           ggml_are_same_shape(op->src[0], op);
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

static bool ggml_backend_pyre_supports_add(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_pyre_supports_binary_elementwise_f32(
        device_context->add_provider, op);
}

static bool ggml_backend_pyre_supports_mul(
        const ggml_backend_pyre_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_pyre_supports_binary_elementwise_f32(
        device_context->mul_provider, op);
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

struct ggml_backend_pyre_rms_norm_constants {
    int64_t ncols;
    int64_t nrows;
    float eps;
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
        /* .ncols = */ src0->ne[0],
        /* .nrows = */ ggml_nrows(src0),
        /* .eps   = */ eps,
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

struct ggml_backend_pyre_elementwise_constants {
    int64_t n;
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
           src1->ne[1] >= 1 && src1->ne[1] <= 16 &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op);
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
           src1->ne[1] >= 1 && src1->ne[1] <= 16 &&
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
           src1->ne[1] >= 1 && src1->ne[1] <= 16 &&
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
           src1->ne[1] >= 1 && src1->ne[1] <= 16 &&
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
        device_context->mul_mat_vec_f16_provider.kind != ggml_backend_pyre_provider_kind::direct_executable) {
        return "F16 provider unavailable";
    }
    if (src0->type == GGML_TYPE_BF16 &&
        device_context->mul_mat_vec_bf16_provider.kind != ggml_backend_pyre_provider_kind::direct_executable) {
        return "BF16 provider unavailable";
    }
    if (src0->type == GGML_TYPE_F32 &&
        device_context->mul_mat_vec_f32_provider.kind != ggml_backend_pyre_provider_kind::direct_executable) {
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
    if (src0->ne[2] != 1 || src0->ne[3] != 1 ||
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
    if (src1->ne[1] < 1 || src1->ne[1] > 16) {
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
           ggml_backend_pyre_supports_mul_mat_vec_bf16(device_context, op) ||
           ggml_backend_pyre_supports_mul_mat_vec_f32(device_context, op) ||
           ggml_backend_pyre_supports_mul_mat_vec_q4_k(device_context, op) ||
           ggml_backend_pyre_supports_mul_mat_vec_q5_k(device_context, op) ||
           ggml_backend_pyre_supports_mul_mat_vec_q6_k(device_context, op) ||
           ggml_backend_pyre_supports_mul_mat_vec_q8_0(device_context, op);
}

struct ggml_backend_pyre_mul_mat_vec_constants {
    int64_t k;
    int64_t rows;
    int64_t cols;
};

static ggml_status ggml_backend_pyre_dispatch_mul_mat_vec_f16(
        ggml_backend_pyre_context * context,
        const ggml_tensor * dst) {
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
            " copy=%" PRIu64 " set_rows=%" PRIu64
            " metadata=%" PRIu64 " synchronize=%" PRIu64 "\n",
            context->name.c_str(),
            context->dispatch_count,
            context->rms_norm_count,
            context->elementwise_count,
            context->mul_mat_vec_count,
            context->copy_count,
            context->set_rows_count,
            context->metadata_count,
            context->synchronize_count);
    }
    if (context->stream) {
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

static ggml_status ggml_backend_pyre_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * context = static_cast<ggml_backend_pyre_context *>(backend->context);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
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
            case GGML_OP_ADD:
                if (!ggml_backend_pyre_supports_add(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: ADD shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context, "claim ADD provider=pure_hip_f32 n=%" PRId64 "\n",
                    ggml_nelements(node));
                if (ggml_backend_pyre_dispatch_binary_elementwise_f32(
                        context, node, context->device_context->add_provider, "ADD") != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_MUL:
                if (!ggml_backend_pyre_supports_mul(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: MUL shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context, "claim MUL provider=pure_hip_f32 n=%" PRId64 "\n",
                    ggml_nelements(node));
                if (ggml_backend_pyre_dispatch_binary_elementwise_f32(
                        context, node, context->device_context->mul_provider, "MUL") != GGML_STATUS_SUCCESS) {
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
            case GGML_OP_MUL_MAT:
                if (!ggml_backend_pyre_supports_mul_mat_vec(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: MUL_MAT shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_backend_pyre_trace_provider(
                    context->device_context,
                    "claim MUL_MAT provider=pure_hip_%s k=%" PRId64 " rows=%" PRId64 " cols=%" PRId64 "\n",
                    ggml_type_name(node->src[0]->type),
                    node->src[0]->ne[0], node->src[0]->ne[1], node->src[1]->ne[1]);
                if (ggml_backend_pyre_dispatch_mul_mat_vec_f16(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
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
        case GGML_OP_SCALE:
            supported = ggml_backend_pyre_supports_scale(context, op);
            break;
        case GGML_OP_CPY:
            supported = ggml_backend_pyre_supports_cpy(op);
            break;
        case GGML_OP_SET_ROWS:
            supported = ggml_backend_pyre_supports_set_rows(context, op);
            break;
        case GGML_OP_MUL_MAT:
            supported = ggml_backend_pyre_supports_mul_mat_vec(context, op);
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
        } else if (context->fallback_trace_count < GGML_PYRE_TRACE_FALLBACK_LIMIT) {
            context->fallback_trace_count++;
            ggml_backend_pyre_trace_provider(context, "fallback %s\n", ggml_op_desc(op));
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
        }
        if (device_context->policy.kernel_provider == ggml_backend_pyre_kernel_provider_mode::pure_hip) {
            (void) ggml_backend_pyre_load_add_provider(device_context.get());
            (void) ggml_backend_pyre_load_mul_provider(device_context.get());
            (void) ggml_backend_pyre_load_scale_provider(device_context.get());
            (void) ggml_backend_pyre_load_set_rows_f32_provider(device_context.get());
            (void) ggml_backend_pyre_load_set_rows_f16_provider(device_context.get());
        }
        if (device_context->policy.kernel_provider == ggml_backend_pyre_kernel_provider_mode::pure_hip &&
            !device_context->policy.disable_mul_mat_vec) {
            (void) ggml_backend_pyre_load_mul_mat_vec_bf16_provider(device_context.get());
            (void) ggml_backend_pyre_load_mul_mat_vec_f16_provider(device_context.get());
            (void) ggml_backend_pyre_load_mul_mat_vec_f32_provider(device_context.get());
            (void) ggml_backend_pyre_load_mul_mat_vec_q4_k_provider(device_context.get());
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
