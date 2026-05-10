# Tests subdirectory: builds the debug_scores helper and the gtest-based
# tests_runner. The runner is gated on BUILD_TESTS (declared at the root) so
# you can `cmake -DBUILD_TESTS=OFF` and still get debug_scores.

add_executable(debug_scores debug_scores.cpp)
target_link_libraries(debug_scores PRIVATE model mini_transformer_warnings)
set_target_properties(debug_scores PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

if(NOT BUILD_TESTS)
    return()
endif()

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
include(GoogleTest)

add_executable(tests_runner
    ops_test.cpp
    io_test.cpp
    data_test.cpp
    decode_test.cpp
    loss_test.cpp
    model_test.cpp
)
target_link_libraries(tests_runner PRIVATE
    language_model
    GTest::gtest_main
    mini_transformer_warnings)
set_target_properties(tests_runner PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

gtest_discover_tests(tests_runner)
