cmake_minimum_required(VERSION 3.14)
project(mini_transformer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(mini_transformer_warnings INTERFACE)
target_compile_options(mini_transformer_warnings INTERFACE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->
  $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall -Wextra -Wpedantic>
)

# Kernel backend selection lives in scripts/configure.sh, which feeds the
# resolved values below. Run that script first; CMake here just consumes them.
foreach(_var IN ITEMS
    MINI_LLM_KERNEL_BACKEND
    MINI_LLM_GEMM_DRIVER_SRC
    MINI_LLM_USE_ACCELERATE)
  if(NOT DEFINED ${_var})
    message(FATAL_ERROR
      "${_var} is not set. Run ./scripts/configure.sh before invoking cmake.")
  endif()
endforeach()
message(STATUS "mini-LLM kernel backend: ${MINI_LLM_KERNEL_BACKEND}")

add_library(model STATIC
  src/model/mini_llm.cpp
  src/model/executor_forward.cpp
  src/model/executor_backward.cpp
  src/engine/validation.cpp
  src/engine/kv_cache.cpp
  src/engine/io.cpp
  src/kernel/core.cpp
  ${MINI_LLM_GEMM_DRIVER_SRC}
  ${MINI_LLM_GEMM_PRIMS_SRC}
  src/kernel/attention.cpp
  src/kernel/norm_rope.cpp
  src/kernel/backward.cpp
)

if(MINI_LLM_GEMM_PRIMS_SRC AND MINI_LLM_GEMM_PRIMS_FLAGS)
  set_source_files_properties(${MINI_LLM_GEMM_PRIMS_SRC}
    PROPERTIES COMPILE_OPTIONS "${MINI_LLM_GEMM_PRIMS_FLAGS}")
endif()
if(MINI_LLM_USE_ACCELERATE)
  target_link_libraries(model PUBLIC "-framework Accelerate")
endif()
target_include_directories(model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(model PUBLIC mini_transformer_warnings)

add_library(language_model STATIC
  src/engine/embedding.cpp
  src/engine/decode.cpp
  src/train/loss.cpp
  src/train/optimizer.cpp
  src/train/autograd.cpp
  src/task/sample_data.cpp
  src/task/metrics.cpp
  src/task/evaluation.cpp
)
target_include_directories(language_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(language_model PUBLIC model)

add_executable(inference inference.cpp)
target_link_libraries(inference PRIVATE language_model mini_transformer_warnings)

add_executable(train train.cpp)
target_link_libraries(train PRIVATE language_model mini_transformer_warnings)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE language_model mini_transformer_warnings)

option(BUILD_TESTS "Build gtest-based model tests" ON)
enable_testing()
add_subdirectory(tests)
