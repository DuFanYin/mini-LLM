#include "core/allocator.h"
#include "core/tensor.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace {

TEST(PoolAllocatorTest, ReusesFreedBlockWithoutNewReservation) {
    core::PoolAllocator alloc;

    void* a = alloc.allocate(256);
    ASSERT_NE(a, nullptr);
    EXPECT_EQ(alloc.reserved_bytes(), 256u);
    EXPECT_EQ(alloc.live_bytes(), 256u);

    alloc.deallocate(a, 256);
    EXPECT_EQ(alloc.live_bytes(), 0u);
    EXPECT_EQ(alloc.reserved_bytes(), 256u); // still reserved, now pooled

    void* b = alloc.allocate(256); // should reuse the pooled block
    EXPECT_EQ(b, a);
    EXPECT_EQ(alloc.reserved_bytes(), 256u); // no new cudaMalloc
    EXPECT_EQ(alloc.live_bytes(), 256u);
    alloc.deallocate(b, 256);
}

TEST(PoolAllocatorTest, ZeroBytesReturnsNull) {
    core::PoolAllocator alloc;
    EXPECT_EQ(alloc.allocate(0), nullptr);
    EXPECT_EQ(alloc.reserved_bytes(), 0u);
}

TEST(TensorTest, HostRoundTripPreservesValues) {
    core::PoolAllocator alloc;
    core::Tensor t({2, 3}, alloc);
    EXPECT_EQ(t.numel(), 6u);
    ASSERT_NE(t.data(), nullptr);

    const std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    t.copy_from_host(in.data());

    std::vector<float> out(6, 0.0f);
    t.copy_to_host(out.data());
    EXPECT_EQ(out, in);
}

TEST(TensorTest, MoveTransfersOwnership) {
    core::PoolAllocator alloc;
    core::Tensor a({4}, alloc);
    float* raw = a.data();

    core::Tensor b = std::move(a);
    EXPECT_EQ(b.data(), raw);
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(a.numel(), 0u);
}

} // namespace
