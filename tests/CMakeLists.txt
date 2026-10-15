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

add_executable(tests_runner core_test.cpp)
target_link_libraries(tests_runner PRIVATE core GTest::gtest_main mini_llm_warnings)
set_target_properties(tests_runner PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

gtest_discover_tests(tests_runner)
