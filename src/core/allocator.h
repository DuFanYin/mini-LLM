#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace core {

// Abstract device allocator. allocate() returns device memory; deallocate()
// returns it for reuse. bytes passed to deallocate must match the allocate call.
class DeviceAllocator {
public:
    virtual ~DeviceAllocator() = default;
    virtual void* allocate(std::size_t bytes) = 0;
    virtual void deallocate(void* ptr, std::size_t bytes) = 0;
};

// Caching free-list allocator over cudaMalloc. Freed blocks are kept in
// exact-size buckets and reused, so steady-state allocation makes no
// cudaMalloc/cudaFree calls. Single-threaded (one stream); not thread-safe.
class PoolAllocator final : public DeviceAllocator {
public:
    PoolAllocator() = default;
    ~PoolAllocator() override;
    PoolAllocator(const PoolAllocator&) = delete;
    PoolAllocator& operator=(const PoolAllocator&) = delete;

    void* allocate(std::size_t bytes) override;
    void deallocate(void* ptr, std::size_t bytes) override;

    // Introspection for tests.
    std::size_t live_bytes() const { return live_bytes_; }         // currently handed out
    std::size_t reserved_bytes() const { return reserved_bytes_; } // total cudaMalloc'd

private:
    std::unordered_map<std::size_t, std::vector<void*>> free_;
    std::size_t live_bytes_ = 0;
    std::size_t reserved_bytes_ = 0;
};

} // namespace core
