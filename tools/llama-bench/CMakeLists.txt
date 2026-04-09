set(TARGET llama-bench)
add_executable(${TARGET} llama-bench.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
