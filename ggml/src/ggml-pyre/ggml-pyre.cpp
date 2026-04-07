#include "ggml-pyre.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "pyre_runtime.h"

#include <array>
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

struct ggml_backend_pyre_device_context {
    pyre_device_t device = nullptr;
    std::string name;
    std::string description;
    size_t memory_total = 0;
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

static size_t ggml_backend_pyre_tensor_offset(const ggml_backend_pyre_buffer_context * context, const ggml_tensor * tensor) {
    return static_cast<size_t>(static_cast<const uint8_t *>(tensor->data) - context->base);
}

static pyre_buffer_t ggml_backend_pyre_tensor_buffer(const ggml_tensor * tensor) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    auto * context = ggml_backend_pyre_get_buffer_context(buffer);
    return context->buffer;
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
    /* .init_tensor   = */ nullptr,
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
    }
}

static ggml_status ggml_backend_pyre_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
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
    GGML_UNUSED(dev);
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
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
        device_context->memory_total = ggml_backend_pyre_total_memory(device);

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
