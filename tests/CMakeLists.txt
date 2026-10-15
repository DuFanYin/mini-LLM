include(FetchContent)
find_package(GTest 1.14 QUIET)
if(NOT GTest_FOUND)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()
include(GoogleTest)

# mini_llm_add_test(<name> LABEL <label> SOURCES <files...> LIBS <targets...>)
# Labels: cpu (host only), gpu (needs a CUDA device), weights (needs a CUDA device
# and the downloaded checkpoint).
function(mini_llm_add_test name)
  cmake_parse_arguments(T "" "LABEL" "SOURCES;LIBS" ${ARGN})
  add_executable(${name} ${T_SOURCES})
  target_link_libraries(${name} PRIVATE ${T_LIBS} GTest::gtest_main)
  mini_llm_set_warnings(${name})
  gtest_discover_tests(${name} PROPERTIES LABELS ${T_LABEL})
endfunction()

mini_llm_add_test(io_tests LABEL cpu
  SOURCES io/json_test.cpp io/safetensors_test.cpp
  LIBS mini_llm::io)

mini_llm_add_test(core_tests LABEL gpu
  SOURCES core/tensor_test.cpp
  LIBS mini_llm::core)

mini_llm_add_test(model_tests LABEL cpu
  SOURCES model/config_test.cpp
  LIBS mini_llm::model mini_llm::io)

mini_llm_add_test(weights_tests LABEL weights
  SOURCES model/weights_test.cpp
  LIBS mini_llm::model mini_llm::io)
target_compile_definitions(weights_tests PRIVATE
  MINI_LLM_DEFAULT_WEIGHTS="${PROJECT_SOURCE_DIR}/weights/qwen3.5-0.8b")
