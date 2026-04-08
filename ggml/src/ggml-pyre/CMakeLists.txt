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

if (Python3_Interpreter_FOUND)
    add_custom_command(
        OUTPUT "${GGML_PYRE_KERNEL_CATALOG_CPP}"
        COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/generate_pyre_kernels.py"
            --clang "${GGML_PYRE_CLANGXX}"
            --rocm-path "${GGML_PYRE_ROCM_PATH}"
            --arch "${GGML_PYRE_AMDGPU_TARGET}"
            --source-dir "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
            --output "${GGML_PYRE_KERNEL_CATALOG_CPP}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/generate_pyre_kernels.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/add_f32.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/add_add_f32_broadcast.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/add_f32_broadcast.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/argsort_f32.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/clamp_f32.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/concat_f32.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/div_f32_broadcast.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gated_delta_net_f32.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/get_rows_f32.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/get_rows_q5_k.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_f32.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_f32_broadcast.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_bf16.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_f16_batched.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_f16.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_f32_batched.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_f32.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_id_q4_k.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_id_q4_k_q8_1.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_id_q4_k_mul.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_id_q4_k_mul_q8_1.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/quantize_q8_1.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rms_norm_mul_f32.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rms_norm.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rope_f32.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/row_reduce_f32.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/scale_f32.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/set_rows_f32_f16.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/set_rows_f32_f32.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/soft_max_f32.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ssm_conv_f32.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/topk_moe_f32.hip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/unary_f32.hip.cpp"
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
