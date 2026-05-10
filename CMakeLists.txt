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

add_library(model STATIC
  src/model/mini_llm.cpp
  src/model/executor_forward.cpp
  src/model/executor_backward.cpp
  src/engine/validation.cpp
  src/engine/kv_cache.cpp
  src/engine/io.cpp
  src/kernel/core.cpp
  src/kernel/gemm.cpp
  src/kernel/attention.cpp
  src/kernel/norm_rope.cpp
  src/kernel/backward.cpp
)
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

add_executable(debug_scores tests/debug_scores.cpp)
target_link_libraries(debug_scores PRIVATE model mini_transformer_warnings)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE language_model mini_transformer_warnings)

option(BUILD_TESTS "Build gtest-based model tests" ON)
if(BUILD_TESTS)
  include(FetchContent)
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()

  enable_testing()
  include(GoogleTest)

  add_executable(tests_runner
    tests/ops_test.cpp
    tests/io_test.cpp
    tests/data_test.cpp
    tests/decode_test.cpp
    tests/loss_test.cpp
    tests/model_test.cpp
  )
  target_link_libraries(tests_runner PRIVATE language_model GTest::gtest_main mini_transformer_warnings)

  gtest_discover_tests(tests_runner)
endif()
