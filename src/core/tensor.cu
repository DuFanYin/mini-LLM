#include "core/tensor.h"

#include "core/allocator.h"
#include "core/cuda_check.h"

#include <cuda_runtime.h>
#include <utility>

namespace core {

Tensor::Tensor(std::vector<std::size_t> shape, DeviceAllocator& alloc)
    : shape_(std::move(shape)), alloc_(&alloc) {
    numel_ = shape_.empty() ? 0 : 1;
    for (std::size_t d : shape_) {
        numel_ *= d;
    }
    data_ = (numel_ != 0) ? static_cast<float*>(alloc_->allocate(numel_ * sizeof(float))) : nullptr;
}

void Tensor::release() noexcept {
    if (data_ != nullptr && alloc_ != nullptr) {
        alloc_->deallocate(data_, numel_ * sizeof(float));
    }
    data_ = nullptr;
    numel_ = 0;
    shape_.clear();
}

Tensor::~Tensor() { release(); }

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_), numel_(other.numel_), shape_(std::move(other.shape_)), alloc_(other.alloc_) {
    other.data_ = nullptr;
    other.numel_ = 0;
    other.alloc_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        release();
        data_ = other.data_;
        numel_ = other.numel_;
        shape_ = std::move(other.shape_);
        alloc_ = other.alloc_;
        other.data_ = nullptr;
        other.numel_ = 0;
        other.alloc_ = nullptr;
    }
    return *this;
}

void Tensor::copy_from_host(const float* src) {
    if (numel_ != 0) {
        CORE_CUDA_CHECK(cudaMemcpy(data_, src, numel_ * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void Tensor::copy_to_host(float* dst) const {
    if (numel_ != 0) {
        CORE_CUDA_CHECK(cudaMemcpy(dst, data_, numel_ * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

} // namespace core
