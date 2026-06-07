#pragma once

// Internal CUDA error checking for the core library. Include only from .cu
// translation units (it pulls in <cuda_runtime.h>); keep it out of public,
// host-includable headers.

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace core {

inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), file, line);
        std::abort();
    }
}

} // namespace core

#define CORE_CUDA_CHECK(expr) ::core::cuda_check((expr), __FILE__, __LINE__)
