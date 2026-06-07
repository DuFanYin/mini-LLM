#include "core/allocator.h"

#include "core/cuda_check.h"

#include <cuda_runtime.h>

namespace core {

void* PoolAllocator::allocate(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    auto it = free_.find(bytes);
    if (it != free_.end() && !it->second.empty()) {
        void* ptr = it->second.back();
        it->second.pop_back();
        live_bytes_ += bytes;
        return ptr;
    }
    void* ptr = nullptr;
    CORE_CUDA_CHECK(cudaMalloc(&ptr, bytes));
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
    // Free only the pooled (returned) blocks; best-effort, ignore teardown errors.
    for (auto& [size, ptrs] : free_) {
        for (void* ptr : ptrs) {
            cudaFree(ptr);
        }
    }
}

} // namespace core
