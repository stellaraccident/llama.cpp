message(STATUS "Using HRX backend")

find_package(hrx CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter)

if (NOT GGML_HRX_ROCM_PATH)
    set(GGML_HRX_ROCM_PATH "/srv/vm-shared/projects/pyre-workspace/build/therock/dist/rocm")
endif()

if (NOT GGML_HRX_CLANGXX)
    find_program(GGML_HRX_CLANGXX
                 NAMES clang++
                 HINTS
                     ${GGML_HRX_ROCM_PATH}/lib/llvm
                 PATH_SUFFIXES bin
                )
endif()

if (NOT GGML_HRX_AMDGPU_TARGET)
    if (AMDGPU_TARGETS)
        list(GET AMDGPU_TARGETS 0 GGML_HRX_AMDGPU_TARGET)
    elseif (DEFINED ENV{AMDGPU_TARGETS})
        set(GGML_HRX_AMDGPU_TARGET "$ENV{AMDGPU_TARGETS}")
    else()
        set(GGML_HRX_AMDGPU_TARGET "gfx1100")
    endif()
endif()

set(GGML_HRX_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(GGML_HRX_KERNEL_CATALOG_CPP "${GGML_HRX_GENERATED_DIR}/hrx_kernel_catalog.cpp")
file(MAKE_DIRECTORY "${GGML_HRX_GENERATED_DIR}")

set(GGML_HRX_KERNEL_GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/generate_hrx_kernels.py")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${GGML_HRX_KERNEL_GENERATOR}")

if (Python3_Interpreter_FOUND AND GGML_HRX_CLANGXX AND GGML_HRX_ROCM_PATH AND GGML_HRX_AMDGPU_TARGET)
    set(GGML_HRX_HSACO_DIR "${GGML_HRX_GENERATED_DIR}/hsaco")
    file(MAKE_DIRECTORY "${GGML_HRX_HSACO_DIR}")

    execute_process(
        COMMAND "${Python3_EXECUTABLE}"
            "${GGML_HRX_KERNEL_GENERATOR}"
            --source-dir "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
            --output "${GGML_HRX_KERNEL_CATALOG_CPP}"
            --list-sources
        OUTPUT_VARIABLE GGML_HRX_KERNEL_SOURCE_LIST_TEXT
        RESULT_VARIABLE GGML_HRX_KERNEL_SOURCE_LIST_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if (NOT GGML_HRX_KERNEL_SOURCE_LIST_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to list HRX HIP kernel sources")
    endif()

    string(REPLACE "\n" ";" GGML_HRX_KERNEL_SOURCE_LIST "${GGML_HRX_KERNEL_SOURCE_LIST_TEXT}")
    set(GGML_HRX_HSACO_OUTPUTS)
    foreach(GGML_HRX_KERNEL_SOURCE IN LISTS GGML_HRX_KERNEL_SOURCE_LIST)
        get_filename_component(GGML_HRX_KERNEL_SOURCE_NAME "${GGML_HRX_KERNEL_SOURCE}" NAME)
        string(REPLACE ".hip.cpp" ".hsaco" GGML_HRX_HSACO_NAME "${GGML_HRX_KERNEL_SOURCE_NAME}")
        set(GGML_HRX_HSACO_OUTPUT "${GGML_HRX_HSACO_DIR}/${GGML_HRX_HSACO_NAME}")
        set(GGML_HRX_KERNEL_DEPS "${GGML_HRX_KERNEL_SOURCE}")
        if (GGML_HRX_KERNEL_SOURCE_NAME STREQUAL "flash_attn_ext_f32_f16_prefill_gfx11_direct.hip.cpp")
            list(APPEND GGML_HRX_KERNEL_DEPS
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_f16_prefill_gfx11_direct_generic.inc"
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_f16_prefill_gfx11_direct_gfx11.inc")
        endif()
        if (GGML_HRX_KERNEL_SOURCE_NAME MATCHES "^topk_moe_f32_")
            list(APPEND GGML_HRX_KERNEL_DEPS
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/topk_moe_f32_common.hip.inc")
        endif()
        if (GGML_HRX_KERNEL_SOURCE_NAME MATCHES "^topk_moe_f32_wave")
            list(APPEND GGML_HRX_KERNEL_DEPS
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/topk_moe_f32_wave.inc")
        endif()
        set(GGML_HRX_KERNEL_COMPILE_FLAGS)
        if (GGML_HRX_KERNEL_SOURCE_NAME STREQUAL "mul_mat_vec_q5_k_q8_1.hip.cpp" OR
            GGML_HRX_KERNEL_SOURCE_NAME STREQUAL "mul_mat_vec_q6_k_q8_1.hip.cpp" OR
            GGML_HRX_KERNEL_SOURCE_NAME STREQUAL "mul_mat_id_q4_k_q8_1_x4_mmq.hip.cpp" OR
            GGML_HRX_KERNEL_SOURCE_NAME STREQUAL "ssm_conv_f32.hip.cpp")
            list(APPEND GGML_HRX_KERNEL_COMPILE_FLAGS -mwavefrontsize64)
        endif()
        if (GGML_HRX_KERNEL_SOURCE_NAME STREQUAL "mul_mat_vec_q5_k_q8_1.hip.cpp" OR
            GGML_HRX_KERNEL_SOURCE_NAME STREQUAL "mul_mat_vec_q6_k_q8_1.hip.cpp")
            list(APPEND GGML_HRX_KERNEL_COMPILE_FLAGS -O3)
        endif()
        if (GGML_HRX_KERNEL_SOURCE_NAME STREQUAL "topk_moe_f32_wave32.hip.cpp" OR
            GGML_HRX_KERNEL_SOURCE_NAME STREQUAL "mul_mat_id_q4_k_mul.hip.cpp")
            list(APPEND GGML_HRX_KERNEL_COMPILE_FLAGS -mno-wavefrontsize64)
        endif()
        add_custom_command(
            OUTPUT "${GGML_HRX_HSACO_OUTPUT}"
            COMMAND "${GGML_HRX_CLANGXX}"
                -x hip
                --offload-device-only
                "--offload-arch=${GGML_HRX_AMDGPU_TARGET}"
                "--rocm-path=${GGML_HRX_ROCM_PATH}"
                -O2
                ${GGML_HRX_KERNEL_COMPILE_FLAGS}
                -c "${GGML_HRX_KERNEL_SOURCE}"
                -o "${GGML_HRX_HSACO_OUTPUT}"
            DEPENDS ${GGML_HRX_KERNEL_DEPS}
            VERBATIM
        )
        list(APPEND GGML_HRX_HSACO_OUTPUTS "${GGML_HRX_HSACO_OUTPUT}")
    endforeach()

    add_custom_command(
        OUTPUT "${GGML_HRX_KERNEL_CATALOG_CPP}"
        COMMAND "${Python3_EXECUTABLE}"
            "${GGML_HRX_KERNEL_GENERATOR}"
            --arch "${GGML_HRX_AMDGPU_TARGET}"
            --source-dir "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
            --hsaco-dir "${GGML_HRX_HSACO_DIR}"
            --output "${GGML_HRX_KERNEL_CATALOG_CPP}"
        DEPENDS
            "${GGML_HRX_KERNEL_GENERATOR}"
            ${GGML_HRX_HSACO_OUTPUTS}
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/hrx_kernel_catalog.h"
        VERBATIM
    )
else()
    file(WRITE "${GGML_HRX_KERNEL_CATALOG_CPP}"
        "// Generated fallback: Python3 not found.\n"
        "#include \"kernels/hrx_kernel_catalog.h\"\n"
        "const ggml_hrx_kernel_entry * ggml_hrx_kernel_catalog_entries(size_t * count) { if (count) { *count = 0; } return nullptr; }\n")
endif()

ggml_add_backend_library(ggml-hrx
                         ggml-hrx.cpp
                         "${GGML_HRX_KERNEL_CATALOG_CPP}"
                         ../../include/ggml-hrx.h
                        )

target_link_libraries(ggml-hrx PRIVATE hrx::hrx)
target_include_directories(ggml-hrx PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${GGML_HRX_GENERATED_DIR}")
