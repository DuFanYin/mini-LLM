#pragma once

// Shared device-side plumbing for the CUDA kernel backend.
//
// The kernel boundary keeps host-pointer signatures (see kernel.h): each public
// CUDA kernel uploads its host inputs, launches handwritten kernels on the
// default stream, synchronizes, and copies results back. Device scratch is held
// in growable, reused buffers so we pay cudaMalloc roughly once per buffer for
// the whole process rather than once per call.

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace kernel::cuda {

inline void check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), file, line);
        std::abort();
    }
}

#define CUDA_CHECK(expr) ::kernel::cuda::check((expr), __FILE__, __LINE__)

// Growable device buffer. reserve(n) guarantees capacity for n floats, reusing
// the existing allocation when it is already large enough. Buffers are meant to
// live as function-local statics so allocations persist across calls.
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        // Best-effort free; ignore errors during process teardown.
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    float* reserve(size_t count) {
        if (count > capacity_) {
            if (ptr_ != nullptr) {
                CUDA_CHECK(cudaFree(ptr_));
            }
            CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(float)));
            capacity_ = count;
        }
        return ptr_;
    }

    float* data() const { return ptr_; }

private:
    float* ptr_ = nullptr;
    size_t capacity_ = 0;
};

// Reserve, then copy `count` host floats into the buffer. Returns the device ptr.
inline float* upload(DeviceBuffer& buf, const float* host, size_t count) {
    float* dev = buf.reserve(count);
    if (count != 0) {
        CUDA_CHECK(cudaMemcpy(dev, host, count * sizeof(float), cudaMemcpyHostToDevice));
    }
    return dev;
}

// Reserve a zero-initialized buffer of `count` floats. Returns the device ptr.
inline float* zeros(DeviceBuffer& buf, size_t count) {
    float* dev = buf.reserve(count);
    if (count != 0) {
        CUDA_CHECK(cudaMemset(dev, 0, count * sizeof(float)));
    }
    return dev;
}

inline void download(float* host, const float* dev, size_t count) {
    if (count != 0) {
        CUDA_CHECK(cudaMemcpy(host, dev, count * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

// Launch-site guard: check the launch error and block until the kernels finish
// (each public call is self-contained because neighbouring ops run on the CPU).
inline void sync() {
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace kernel::cuda
