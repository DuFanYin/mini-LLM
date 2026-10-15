add_executable(inspect inspect.cpp)
target_link_libraries(inspect PRIVATE mini_llm::model mini_llm::io)
mini_llm_set_warnings(inspect)
