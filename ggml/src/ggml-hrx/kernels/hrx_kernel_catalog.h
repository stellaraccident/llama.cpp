#pragma once

#include <cstddef>
#include <cstdint>

struct ggml_hrx_kernel_entry {
    const char * name;
    const unsigned char * data;
    size_t data_size;
    const char * format;
    uint32_t binding_count;
    uint32_t constants_size;
    uint32_t workgroup_size[3];
};

const ggml_hrx_kernel_entry * ggml_hrx_kernel_catalog_entries(size_t * count);
