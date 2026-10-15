#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace core {

// Device memory source for Tensor. deallocate() receives the same byte count
// that allocate() was called with.
class DeviceAllocator {
public:
    virtual ~DeviceAllocator() = default;
    virtual void* allocate(std::size_t bytes) = 0;
    virtual void deallocate(void* ptr, std::size_t bytes) = 0;
};

// Caching allocator over cudaMalloc with exact-size free lists, so steady-state
// allocation makes no cudaMalloc/cudaFree calls. Not thread-safe.
class PoolAllocator final : public DeviceAllocator {
public:
    PoolAllocator() = default;
    ~PoolAllocator() override;
    PoolAllocator(const PoolAllocator&) = delete;
    PoolAllocator& operator=(const PoolAllocator&) = delete;

    void* allocate(std::size_t bytes) override;
    void deallocate(void* ptr, std::size_t bytes) override;

    std::size_t live_bytes() const { return live_bytes_; }
    std::size_t reserved_bytes() const { return reserved_bytes_; }

private:
    std::unordered_map<std::size_t, std::vector<void*>> free_;
    std::size_t live_bytes_ = 0;
    std::size_t reserved_bytes_ = 0;
};

// One cudaMalloc carved by a bump pointer; every block is kAlignment-aligned.
// deallocate() is a no-op and the whole region is freed with the arena. Used
// for memory that lives as long as the model (weights).
class Arena final : public DeviceAllocator {
public:
    static constexpr std::size_t kAlignment = 256;
    static constexpr std::size_t aligned(std::size_t bytes) {
        return (bytes + kAlignment - 1) / kAlignment * kAlignment;
    }

    explicit Arena(std::size_t capacity);
    ~Arena() override;
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    void* allocate(std::size_t bytes) override; // throws std::bad_alloc when exhausted
    void deallocate(void*, std::size_t) override {}

    std::size_t capacity() const { return capacity_; }
    std::size_t used() const { return used_; }

private:
    std::byte* base_ = nullptr;
    std::size_t capacity_ = 0;
    std::size_t used_ = 0;
};

} // namespace core
