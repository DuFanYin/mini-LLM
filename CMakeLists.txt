cmake_minimum_required(VERSION 3.14)
project(mini_transformer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(mini_transformer_warnings INTERFACE)
# Restrict warning flags to C++ sources so they are not forwarded to nvcc when
# the CUDA backend mixes .cu sources into the same target.
target_compile_options(mini_transformer_warnings INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4 /permissive->
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:AppleClang,Clang,GNU>>:-Wall -Wextra -Wpedantic>
)

# Kernel backend selection lives in scripts/configure.sh, which feeds the
# resolved values below. Run that script first; CMake here just consumes them.
foreach(_var IN ITEMS
    MINI_LLM_KERNEL_BACKEND
    MINI_LLM_GEMM_DRIVER_SRC
    MINI_LLM_ATTENTION_SRC
    MINI_LLM_CORE_SRC
    MINI_LLM_USE_CUDA)
  if(NOT DEFINED ${_var})
    message(FATAL_ERROR
      "${_var} is not set. Run ./scripts/configure.sh before invoking cmake.")
  endif()
endforeach()
message(STATUS "mini-LLM kernel backend: ${MINI_LLM_KERNEL_BACKEND}")

# The cuda backend feeds .cu sources (gemm/attention) into the model library.
# Enable the CUDA language and pick a sensible default architecture.
if(MINI_LLM_USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  message(STATUS "mini-LLM CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Device foundation library (Phase 0). CUDA-only; the scalar build skips it.
if(MINI_LLM_USE_CUDA)
  add_library(core STATIC
    src/core/allocator.cu
    src/core/tensor.cu
  )
  target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(core PUBLIC CUDA::cudart mini_transformer_warnings)
endif()

add_library(model STATIC
  src/model/mini_llm.cpp
  src/model/executor_forward.cpp
  src/model/executor_backward.cpp
  src/model/cache.cpp
  src/engine/io.cpp
  src/engine/embedding.cpp
  ${MINI_LLM_CORE_SRC}
  ${MINI_LLM_GEMM_DRIVER_SRC}
  ${MINI_LLM_ATTENTION_SRC}
  src/kernel/scalar/norm_rope.cpp
  src/kernel/common/silu.cpp
)

target_include_directories(model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(model PUBLIC mini_transformer_warnings)
if(MINI_LLM_USE_CUDA)
  target_sources(model PRIVATE src/kernel/cuda/norm_rope.cu)
  target_link_libraries(model PUBLIC CUDA::cudart core)
  target_compile_definitions(model PUBLIC MINI_LLM_USE_CUDA)
endif()

add_library(language_model STATIC
  src/engine/decode.cpp
  src/train/optimizer.cpp
  src/train/loss.cpp
  src/train/step.cpp
  src/task/sample_data.cpp
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
