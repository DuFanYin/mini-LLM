# core: device memory and tensors (CUDA runtime only).
add_library(mini_llm_core STATIC
  core/allocator.cpp
  core/device.cpp
  core/tensor.cpp)
add_library(mini_llm::core ALIAS mini_llm_core)
target_include_directories(mini_llm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mini_llm_core PUBLIC CUDA::cudart)
mini_llm_set_warnings(mini_llm_core)

# io: JSON and safetensors readers (host only).
add_library(mini_llm_io STATIC
  io/json.cpp
  io/safetensors.cpp)
add_library(mini_llm::io ALIAS mini_llm_io)
target_include_directories(mini_llm_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
mini_llm_set_warnings(mini_llm_io)

# model: Qwen3.5 config and device weights.
add_library(mini_llm_model STATIC
  model/config.cpp
  model/weights.cpp)
add_library(mini_llm::model ALIAS mini_llm_model)
target_include_directories(mini_llm_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mini_llm_model PUBLIC mini_llm::core PRIVATE mini_llm::io)
mini_llm_set_warnings(mini_llm_model)
