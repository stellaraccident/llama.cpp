message(STATUS "Using PYRE backend")

find_package(pyre CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter)

if (NOT GGML_PYRE_ROCM_PATH)
    set(GGML_PYRE_ROCM_PATH "/srv/vm-shared/projects/pyre-workspace/build/therock/dist/rocm")
endif()

if (NOT GGML_PYRE_CLANGXX)
    find_program(GGML_PYRE_CLANGXX
                 NAMES clang++
                 HINTS
                     ${GGML_PYRE_ROCM_PATH}/lib/llvm
                 PATH_SUFFIXES bin
                )
endif()

if (NOT GGML_PYRE_AMDGPU_TARGET)
    if (AMDGPU_TARGETS)
        list(GET AMDGPU_TARGETS 0 GGML_PYRE_AMDGPU_TARGET)
    elseif (DEFINED ENV{AMDGPU_TARGETS})
        set(GGML_PYRE_AMDGPU_TARGET "$ENV{AMDGPU_TARGETS}")
    else()
        set(GGML_PYRE_AMDGPU_TARGET "gfx1100")
    endif()
endif()

set(GGML_PYRE_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(GGML_PYRE_KERNEL_CATALOG_CPP "${GGML_PYRE_GENERATED_DIR}/pyre_kernel_catalog.cpp")
file(MAKE_DIRECTORY "${GGML_PYRE_GENERATED_DIR}")

set(GGML_PYRE_KERNEL_GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/generate_pyre_kernels.py")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${GGML_PYRE_KERNEL_GENERATOR}")

if (Python3_Interpreter_FOUND AND GGML_PYRE_CLANGXX AND GGML_PYRE_ROCM_PATH AND GGML_PYRE_AMDGPU_TARGET)
    set(GGML_PYRE_HSACO_DIR "${GGML_PYRE_GENERATED_DIR}/hsaco")
    file(MAKE_DIRECTORY "${GGML_PYRE_HSACO_DIR}")

    execute_process(
        COMMAND "${Python3_EXECUTABLE}"
            "${GGML_PYRE_KERNEL_GENERATOR}"
            --source-dir "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
            --output "${GGML_PYRE_KERNEL_CATALOG_CPP}"
            --list-sources
        OUTPUT_VARIABLE GGML_PYRE_KERNEL_SOURCE_LIST_TEXT
        RESULT_VARIABLE GGML_PYRE_KERNEL_SOURCE_LIST_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if (NOT GGML_PYRE_KERNEL_SOURCE_LIST_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to list Pyre HIP kernel sources")
    endif()

    string(REPLACE "\n" ";" GGML_PYRE_KERNEL_SOURCE_LIST "${GGML_PYRE_KERNEL_SOURCE_LIST_TEXT}")
    set(GGML_PYRE_HSACO_OUTPUTS)
    foreach(GGML_PYRE_KERNEL_SOURCE IN LISTS GGML_PYRE_KERNEL_SOURCE_LIST)
        get_filename_component(GGML_PYRE_KERNEL_SOURCE_NAME "${GGML_PYRE_KERNEL_SOURCE}" NAME)
        string(REPLACE ".hip.cpp" ".hsaco" GGML_PYRE_HSACO_NAME "${GGML_PYRE_KERNEL_SOURCE_NAME}")
        set(GGML_PYRE_HSACO_OUTPUT "${GGML_PYRE_HSACO_DIR}/${GGML_PYRE_HSACO_NAME}")
        set(GGML_PYRE_KERNEL_DEPS "${GGML_PYRE_KERNEL_SOURCE}")
        if (GGML_PYRE_KERNEL_SOURCE_NAME STREQUAL "flash_attn_ext_f32_f16_prefill_gfx11_direct.hip.cpp")
            list(APPEND GGML_PYRE_KERNEL_DEPS
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_f16_prefill_gfx11_direct_generic.inc"
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_f16_prefill_gfx11_direct_gfx11.inc")
        endif()
        add_custom_command(
            OUTPUT "${GGML_PYRE_HSACO_OUTPUT}"
            COMMAND "${GGML_PYRE_CLANGXX}"
                -x hip
                --offload-device-only
                "--offload-arch=${GGML_PYRE_AMDGPU_TARGET}"
                "--rocm-path=${GGML_PYRE_ROCM_PATH}"
                -O2
                -c "${GGML_PYRE_KERNEL_SOURCE}"
                -o "${GGML_PYRE_HSACO_OUTPUT}"
            DEPENDS ${GGML_PYRE_KERNEL_DEPS}
            VERBATIM
        )
        list(APPEND GGML_PYRE_HSACO_OUTPUTS "${GGML_PYRE_HSACO_OUTPUT}")
    endforeach()

    add_custom_command(
        OUTPUT "${GGML_PYRE_KERNEL_CATALOG_CPP}"
        COMMAND "${Python3_EXECUTABLE}"
            "${GGML_PYRE_KERNEL_GENERATOR}"
            --arch "${GGML_PYRE_AMDGPU_TARGET}"
            --source-dir "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
            --hsaco-dir "${GGML_PYRE_HSACO_DIR}"
            --output "${GGML_PYRE_KERNEL_CATALOG_CPP}"
        DEPENDS
            "${GGML_PYRE_KERNEL_GENERATOR}"
            ${GGML_PYRE_HSACO_OUTPUTS}
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/pyre_kernel_catalog.h"
        VERBATIM
    )
else()
    file(WRITE "${GGML_PYRE_KERNEL_CATALOG_CPP}"
        "// Generated fallback: Python3 not found.\n"
        "#include \"kernels/pyre_kernel_catalog.h\"\n"
        "const ggml_pyre_kernel_entry * ggml_pyre_kernel_catalog_entries(size_t * count) { if (count) { *count = 0; } return nullptr; }\n")
endif()

ggml_add_backend_library(ggml-pyre
                         ggml-pyre.cpp
                         "${GGML_PYRE_KERNEL_CATALOG_CPP}"
                         ../../include/ggml-pyre.h
                        )

target_link_libraries(ggml-pyre PRIVATE pyre::pyre)
target_include_directories(ggml-pyre PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${GGML_PYRE_GENERATED_DIR}")
