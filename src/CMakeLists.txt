# core: device memory and tensors (CUDA runtime only).
add_library(mini_llm_core STATIC
  core/allocator.cpp
  core/device.cpp
  core/stream.cpp
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

# kernels: hand-written CUDA kernels behind host-callable launch functions.
add_library(mini_llm_kernels STATIC
  kernels/attention.cu
  kernels/elementwise.cu
  kernels/gdn.cu
  kernels/linear.cu
  kernels/norm.cu
  kernels/sampling.cu)
add_library(mini_llm::kernels ALIAS mini_llm_kernels)
target_include_directories(mini_llm_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mini_llm_kernels PUBLIC mini_llm::core)
mini_llm_set_warnings(mini_llm_kernels)

# runtime: runs a sequence through the model on the GPU.
add_library(mini_llm_runtime STATIC
  runtime/executor.cpp)
add_library(mini_llm::runtime ALIAS mini_llm_runtime)
target_include_directories(mini_llm_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mini_llm_runtime PUBLIC mini_llm::model PRIVATE mini_llm::kernels)
mini_llm_set_warnings(mini_llm_runtime)
