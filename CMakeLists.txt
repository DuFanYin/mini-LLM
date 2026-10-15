cmake_minimum_required(VERSION 3.25)
project(mini_llm LANGUAGES CXX CUDA)

option(MINI_LLM_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(MINI_LLM_BUILD_TESTS "Build the test suites" ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
include(cmake/warnings.cmake)

add_subdirectory(src)
add_subdirectory(apps)

if(MINI_LLM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
