# mini_llm_set_warnings(<target>): project warning set for C++ and CUDA sources.
function(mini_llm_set_warnings target)
  set(cxx_warnings
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Wold-style-cast
    -Wnon-virtual-dtor -Woverloaded-virtual -Wnull-dereference -Wdouble-promotion
    -Wformat=2 -Wimplicit-fallthrough)
  set(cuda_warnings -Xcompiler=-Wall,-Wextra)
  if(MINI_LLM_WARNINGS_AS_ERRORS)
    list(APPEND cxx_warnings -Werror)
    list(APPEND cuda_warnings --Werror=all-warnings -Xcompiler=-Werror)
  endif()
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${cxx_warnings}>
    $<$<COMPILE_LANGUAGE:CUDA>:${cuda_warnings} -lineinfo>)
endfunction()
