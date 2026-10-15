#pragma once

#include "core/dtype.h"

#include <cstddef>
#include <vector>

namespace core {

class DeviceAllocator;

// Owning, move-only, contiguous row-major tensor in CUDA device memory.
// Allocates from a DeviceAllocator on construction and returns the block on
// destruction. numel() is the product of the shape dims (0 for an empty shape).
class Tensor {
public:
    Tensor() = default;
    Tensor(std::vector<std::size_t> shape, DType dtype, DeviceAllocator& alloc);
    ~Tensor();

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    void* raw() { return data_; }
    const void* raw() const { return data_; }
    template <class T> T* data() { return static_cast<T*>(data_); }
    template <class T> const T* data() const { return static_cast<const T*>(data_); }

    DType dtype() const { return dtype_; }
    std::size_t numel() const { return numel_; }
    std::size_t bytes() const { return numel_ * dtype_size(dtype_); }
    const std::vector<std::size_t>& shape() const { return shape_; }

    void copy_from_host(const void* src); // H2D, bytes() bytes
    void copy_to_host(void* dst) const;   // D2H, bytes() bytes
    void zero();

private:
    void release() noexcept;

    void* data_ = nullptr;
    std::size_t numel_ = 0;
    DType dtype_ = DType::f32;
    std::vector<std::size_t> shape_;
    DeviceAllocator* alloc_ = nullptr;
};

} // namespace core
