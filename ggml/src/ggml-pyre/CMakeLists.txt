message(STATUS "Using PYRE backend")

find_package(pyre CONFIG REQUIRED)

ggml_add_backend_library(ggml-pyre
                         ggml-pyre.cpp
                         ../../include/ggml-pyre.h
                        )

target_link_libraries(ggml-pyre PRIVATE pyre::pyre)
