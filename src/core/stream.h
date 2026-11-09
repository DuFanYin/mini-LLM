#pragma once

#include <cuda_runtime_api.h>

namespace core {

// Owning, move-only non-blocking CUDA stream.
class Stream {
public:
    Stream();
    ~Stream();
    Stream(Stream&& other) noexcept : stream_(other.stream_) { other.stream_ = nullptr; }
    Stream& operator=(Stream&&) = delete;
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    cudaStream_t get() const { return stream_; }
    void synchronize() const;

private:
    cudaStream_t stream_ = nullptr;
};

} // namespace core
