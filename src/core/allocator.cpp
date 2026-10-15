#include "core/allocator.h"

#include "core/cuda_check.h"

#include <cuda_runtime_api.h>

#include <new>

namespace core {

void* PoolAllocator::allocate(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    if (auto it = free_.find(bytes); it != free_.end() && !it->second.empty()) {
        void* ptr = it->second.back();
        it->second.pop_back();
        live_bytes_ += bytes;
        return ptr;
    }
    void* ptr = nullptr;
    cuda_check(cudaMalloc(&ptr, bytes));
    reserved_bytes_ += bytes;
    live_bytes_ += bytes;
    return ptr;
}

void PoolAllocator::deallocate(void* ptr, std::size_t bytes) {
    if (ptr == nullptr) {
        return;
    }
    free_[bytes].push_back(ptr);
    live_bytes_ -= bytes;
}

PoolAllocator::~PoolAllocator() {
    // Frees pooled blocks only; errors at teardown are ignored.
    for (auto& [size, ptrs] : free_) {
        for (void* ptr : ptrs) {
            cudaFree(ptr);
        }
    }
}

Arena::Arena(std::size_t capacity) : capacity_(aligned(capacity)) {
    if (capacity_ != 0) {
        void* ptr = nullptr;
        cuda_check(cudaMalloc(&ptr, capacity_));
        base_ = static_cast<std::byte*>(ptr);
    }
}

Arena::~Arena() {
    if (base_ != nullptr) {
        cudaFree(base_);
    }
}

void* Arena::allocate(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    const std::size_t size = aligned(bytes);
    if (size > capacity_ - used_) {
        throw std::bad_alloc();
    }
    void* ptr = base_ + used_;
    used_ += size;
    return ptr;
}

} // namespace core
