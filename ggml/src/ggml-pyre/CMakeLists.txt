message(STATUS "Using PYRE backend")

find_package(pyre CONFIG REQUIRED)

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

ggml_add_backend_library(ggml-pyre
                         ggml-pyre.cpp
                         ../../include/ggml-pyre.h
                        )

target_link_libraries(ggml-pyre PRIVATE pyre::pyre)

if (GGML_PYRE_CLANGXX)
    target_compile_definitions(ggml-pyre PRIVATE GGML_PYRE_DEFAULT_CLANGXX="${GGML_PYRE_CLANGXX}")
endif()

if (GGML_PYRE_ROCM_PATH)
    target_compile_definitions(ggml-pyre PRIVATE GGML_PYRE_DEFAULT_ROCM_PATH="${GGML_PYRE_ROCM_PATH}")
endif()
