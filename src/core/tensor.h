#pragma once

#include <cstddef>
#include <vector>

namespace core {

class DeviceAllocator;

// Owning, move-only fp32 tensor in CUDA device memory. Allocates from a
// DeviceAllocator on construction and returns the block on destruction.
// numel() is the product of the shape dims (0 for an empty shape).
class Tensor {
public:
    Tensor() = default;
    Tensor(std::vector<std::size_t> shape, DeviceAllocator& alloc);
    ~Tensor();

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    float* data() { return data_; }
    const float* data() const { return data_; }
    std::size_t numel() const { return numel_; }
    const std::vector<std::size_t>& shape() const { return shape_; }

    void copy_from_host(const float* src); // H2D: copies numel() floats
    void copy_to_host(float* dst) const;   // D2H: copies numel() floats

private:
    void release() noexcept;

    float* data_ = nullptr;
    std::size_t numel_ = 0;
    std::vector<std::size_t> shape_;
    DeviceAllocator* alloc_ = nullptr;
};

} // namespace core
