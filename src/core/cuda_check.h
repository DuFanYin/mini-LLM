#pragma once

#include <cuda_runtime_api.h>

#include <source_location>
#include <stdexcept>
#include <string>

namespace core {

// A failed CUDA runtime call, with the call site.
class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t code, const std::source_location& loc)
        : std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(code) + " (" + loc.file_name() + ":" +
                             std::to_string(loc.line()) + ")"),
          code_(code) {}

    cudaError_t code() const noexcept { return code_; }

private:
    cudaError_t code_;
};

inline void cuda_check(cudaError_t err, const std::source_location& loc = std::source_location::current()) {
    if (err != cudaSuccess) {
        throw CudaError(err, loc);
    }
}

} // namespace core
