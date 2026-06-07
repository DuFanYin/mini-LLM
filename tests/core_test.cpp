#include "core/allocator.h"
#include "core/tensor.h"
#include "kernel/cuda/launch.h"

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

// Host reference: y[m,n] = bias[n] + sum_k x[m,k] * w[n,k].
TEST(KernelSeamTest, LinearDeviceMatchesHostReference) {
    const std::size_t M = 2, K = 2, N = 3;
    const std::vector<float> x = {1.0f, 2.0f,
                                  3.0f, 4.0f};           // [M,K]
    const std::vector<float> w = {1.0f, 0.0f,
                                  0.0f, 1.0f,
                                  1.0f, 1.0f};           // [N,K]
    const std::vector<float> bias = {0.5f, -0.5f, 1.0f}; // [N]

    std::vector<float> expected(M * N, 0.0f);
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t n = 0; n < N; ++n) {
            float acc = bias[n];
            for (std::size_t k = 0; k < K; ++k) {
                acc += x[m * K + k] * w[n * K + k];
            }
            expected[m * N + n] = acc;
        }
    }

    core::PoolAllocator alloc;
    core::Tensor dx({M, K}, alloc);
    core::Tensor dw({N, K}, alloc);
    core::Tensor dbias({N}, alloc);
    core::Tensor dy({M, N}, alloc);
    dx.copy_from_host(x.data());
    dw.copy_from_host(w.data());
    dbias.copy_from_host(bias.data());

    kernel::cuda::linear_device(dx.data(), dw.data(), dbias.data(), dy.data(), M, N, K);

    std::vector<float> got(M * N, 0.0f);
    dy.copy_to_host(got.data()); // synchronous D2H orders after the kernel on the default stream
    for (std::size_t i = 0; i < M * N; ++i) {
        EXPECT_FLOAT_EQ(got[i], expected[i]) << "at index " << i;
    }
}

TEST(KernelSeamTest, GemmNtDeviceMatchesHostReference) {
    const std::size_t M = 3, K = 4, N = 2;
    std::vector<float> A(M * K), B(N * K);
    for (std::size_t i = 0; i < A.size(); ++i) {
        A[i] = static_cast<float>(i + 1);
    }
    for (std::size_t i = 0; i < B.size(); ++i) {
        B[i] = static_cast<float>((i % 3) + 1);
    }
    std::vector<float> expected(M * N, 0.0f);
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (std::size_t k = 0; k < K; ++k) {
                acc += A[m * K + k] * B[n * K + k];
            }
            expected[m * N + n] = acc;
        }
    }

    core::PoolAllocator alloc;
    core::Tensor dA({M, K}, alloc), dB({N, K}, alloc), dC({M, N}, alloc);
    dA.copy_from_host(A.data());
    dB.copy_from_host(B.data());

    kernel::cuda::gemm_nt_device(dA.data(), dB.data(), dC.data(), M, N, K);

    std::vector<float> got(M * N, 0.0f);
    dC.copy_to_host(got.data());
    for (std::size_t i = 0; i < M * N; ++i) {
        EXPECT_FLOAT_EQ(got[i], expected[i]) << "at index " << i;
    }
}

} // namespace
