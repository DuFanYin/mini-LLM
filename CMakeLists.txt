cmake_minimum_required(VERSION 3.18)
project(mini_llm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(mini_llm_warnings INTERFACE)
# C++ sources only, so the flags are not forwarded to nvcc.
target_compile_options(mini_llm_warnings INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Clang,GNU>>:-Wall -Wextra -Wpedantic>
)

add_library(core STATIC
  src/core/allocator.cu
  src/core/tensor.cu
)
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(core PUBLIC CUDA::cudart mini_llm_warnings)

option(BUILD_TESTS "Build gtest-based tests" ON)
enable_testing()
if(BUILD_TESTS)
  add_subdirectory(tests)
endif()
