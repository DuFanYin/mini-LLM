#include "core/allocator.h"
#include "core/cuda_check.h"
#include "core/tensor.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>
#include <vector>

namespace {

TEST(PoolAllocator, ReusesFreedBlockWithoutNewReservation) {
    core::PoolAllocator alloc;
    void* a = alloc.allocate(256);
    ASSERT_NE(a, nullptr);
    alloc.deallocate(a, 256);
    EXPECT_EQ(alloc.live_bytes(), 0u);

    void* b = alloc.allocate(256);
    EXPECT_EQ(b, a);
    EXPECT_EQ(alloc.reserved_bytes(), 256u);
    alloc.deallocate(b, 256);
}

TEST(PoolAllocator, ZeroBytesReturnsNull) {
    core::PoolAllocator alloc;
    EXPECT_EQ(alloc.allocate(0), nullptr);
    EXPECT_EQ(alloc.reserved_bytes(), 0u);
}

TEST(Arena, BlocksAreAlignedAndContiguous) {
    core::Arena arena(1024);
    auto* a = static_cast<std::byte*>(arena.allocate(10));
    auto* b = static_cast<std::byte*>(arena.allocate(300));
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a) % core::Arena::kAlignment, 0u);
    EXPECT_EQ(b - a, static_cast<std::ptrdiff_t>(core::Arena::kAlignment));
    EXPECT_EQ(arena.used(), 3 * core::Arena::kAlignment);
}

TEST(Arena, ThrowsWhenExhausted) {
    core::Arena arena(512);
    arena.allocate(512);
    EXPECT_THROW(arena.allocate(1), std::bad_alloc);
}

TEST(Tensor, BytesFollowDtype) {
    core::PoolAllocator alloc;
    const core::Tensor f({2, 3}, core::DType::f32, alloc);
    const core::Tensor h({2, 3}, core::DType::bf16, alloc);
    EXPECT_EQ(f.numel(), 6u);
    EXPECT_EQ(f.bytes(), 24u);
    EXPECT_EQ(h.bytes(), 12u);
}

TEST(Tensor, HostRoundTripPreservesValues) {
    core::PoolAllocator alloc;
    core::Tensor t({2, 3}, core::DType::f32, alloc);
    const std::vector<float> in = {1, 2, 3, 4, 5, 6};
    t.copy_from_host(in.data());
    std::vector<float> out(6);
    t.copy_to_host(out.data());
    EXPECT_EQ(out, in);
}

TEST(Tensor, ZeroClearsContents) {
    core::PoolAllocator alloc;
    core::Tensor t({4}, core::DType::bf16, alloc);
    const std::vector<std::uint16_t> in = {1, 2, 3, 4};
    t.copy_from_host(in.data());
    t.zero();
    std::vector<std::uint16_t> out(4, 7);
    t.copy_to_host(out.data());
    EXPECT_EQ(out, std::vector<std::uint16_t>(4, 0));
}

TEST(Tensor, MoveTransfersOwnership) {
    core::PoolAllocator alloc;
    core::Tensor a({4}, core::DType::f32, alloc);
    void* raw = a.raw();
    core::Tensor b = std::move(a);
    EXPECT_EQ(b.raw(), raw);
    EXPECT_EQ(a.raw(), nullptr); // NOLINT(bugprone-use-after-move): moved-from state is specified
    EXPECT_EQ(a.numel(), 0u);
}

TEST(CudaCheck, ThrowsOnError) {
    EXPECT_THROW(core::cuda_check(cudaErrorInvalidValue), core::CudaError);
    EXPECT_NO_THROW(core::cuda_check(cudaSuccess));
}

} // namespace
