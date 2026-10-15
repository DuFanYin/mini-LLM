#include "core/tensor.h"

#include "core/allocator.h"
#include "core/cuda_check.h"

#include <cuda_runtime_api.h>
#include <utility>

namespace core {

Tensor::Tensor(std::vector<std::size_t> shape, DType dtype, DeviceAllocator& alloc)
    : dtype_(dtype), shape_(std::move(shape)), alloc_(&alloc) {
    numel_ = shape_.empty() ? 0 : 1;
    for (std::size_t d : shape_) {
        numel_ *= d;
    }
    data_ = (numel_ != 0) ? alloc_->allocate(bytes()) : nullptr;
}

void Tensor::release() noexcept {
    if (data_ != nullptr && alloc_ != nullptr) {
        alloc_->deallocate(data_, bytes());
    }
    data_ = nullptr;
    numel_ = 0;
    shape_.clear();
}

Tensor::~Tensor() {
    release();
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_), numel_(other.numel_), dtype_(other.dtype_), shape_(std::move(other.shape_)),
      alloc_(other.alloc_) {
    other.data_ = nullptr;
    other.numel_ = 0;
    other.alloc_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        release();
        data_ = other.data_;
        numel_ = other.numel_;
        dtype_ = other.dtype_;
        shape_ = std::move(other.shape_);
        alloc_ = other.alloc_;
        other.data_ = nullptr;
        other.numel_ = 0;
        other.alloc_ = nullptr;
    }
    return *this;
}

void Tensor::copy_from_host(const void* src) {
    if (numel_ != 0) {
        cuda_check(cudaMemcpy(data_, src, bytes(), cudaMemcpyHostToDevice));
    }
}

void Tensor::copy_to_host(void* dst) const {
    if (numel_ != 0) {
        cuda_check(cudaMemcpy(dst, data_, bytes(), cudaMemcpyDeviceToHost));
    }
}

void Tensor::zero() {
    if (numel_ != 0) {
        cuda_check(cudaMemset(data_, 0, bytes()));
    }
}

} // namespace core
