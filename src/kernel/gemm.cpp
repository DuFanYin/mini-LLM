#include "kernel/kernel.h"

#include <cstddef>
#include <cstring>

namespace kernel {

// Inner-kernel primitives. Defined by exactly one of gemm_neon.cpp /
// gemm_avx2.cpp / gemm_scalar.cpp, picked at CMake configure time via
// MINI_LLM_KERNEL_BACKEND.
namespace detail {
float gemm_dot(const float* a, const float* b, std::size_t len);
void gemm_axpy(float* y, const float* x, float scale, std::size_t K);
void gemm_dot4(const float* a,
               const float* b0, const float* b1, const float* b2, const float* b3,
               std::size_t K,
               float* c0, float* c1, float* c2, float* c3);
} // namespace detail

namespace {

// C += A * B^T with A [M×K], B [N×K] row-major, C [M×ldc].
// Four output channels are accumulated together so each A row vector is loaded once per N tile.
void gemm_nt_accum(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, size_t ldc) {
    for (size_t m = 0; m < M; ++m) {
        float* crow = C + m * ldc;
        const float* arow = A + m * K;
        size_t n = 0;
        for (; n + 4 <= N; n += 4) {
            detail::gemm_dot4(arow, B + (n + 0u) * K, B + (n + 1u) * K, B + (n + 2u) * K, B + (n + 3u) * K, K,
                              &crow[n + 0u], &crow[n + 1u], &crow[n + 2u], &crow[n + 3u]);
        }
        for (; n < N; ++n) {
            crow[n] += detail::gemm_dot(arow, B + n * K, K);
        }
    }
}

// C[M×K] += A[M×N] * B[N×K], all row-major. Used for dx in linear backward.
void gemm_nn_accum(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, size_t ldc) {
    for (size_t m = 0; m < M; ++m) {
        const float* arow = A + m * N;
        float* crow = C + m * ldc;
        for (size_t n = 0; n < N; ++n) {
            const float scale = arow[n];
            if (scale != 0.0f) {
                detail::gemm_axpy(crow, B + n * K, scale, K);
            }
        }
    }
}

// C[N×K] += A[M×N]^T * B[M×K], all row-major. Used for grad_w in linear backward.
void gemm_tn_accum(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, size_t ldc) {
    for (size_t m = 0; m < M; ++m) {
        const float* arow = A + m * N;
        const float* brow = B + m * K;
        for (size_t n = 0; n < N; ++n) {
            const float scale = arow[n];
            if (scale != 0.0f) {
                detail::gemm_axpy(C + n * ldc, brow, scale, K);
            }
        }
    }
}

} // namespace

void gemm_nt(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
    if (M == 0 || N == 0 || K == 0) {
        return;
    }
    std::memset(C, 0, M * N * sizeof(float));
    gemm_nt_accum(A, B, C, M, N, K, N);
}

void linear(const float* x, const float* w, const float* bias, float* y, const LinearParams& p) {
    const size_t M = p.rows;
    const size_t N = p.out_dim;
    const size_t K = p.in_dim;
    if (M == 0 || N == 0 || K == 0) {
        return;
    }
    if (bias != nullptr) {
        for (size_t m = 0; m < M; ++m) {
            std::memcpy(y + m * N, bias, N * sizeof(float));
        }
    } else {
        std::memset(y, 0, M * N * sizeof(float));
    }
    gemm_nt_accum(x, w, y, M, N, K, N);
}

void linear_backward(const float* x, const float* w, bool has_bias, const float* dy, float* dx, float* grad_w,
                     float* grad_b, const LinearParams& p) {
    const size_t M = p.rows;
    const size_t N = p.out_dim;
    const size_t K = p.in_dim;
    if (M == 0 || N == 0 || K == 0) {
        return;
    }

    if (has_bias && grad_b != nullptr) {
        for (size_t m = 0; m < M; ++m) {
            const float* row = dy + m * N;
            for (size_t n = 0; n < N; ++n) {
                grad_b[n] += row[n];
            }
        }
    }
    gemm_tn_accum(dy, x, grad_w, M, N, K, K);
    gemm_nn_accum(dy, w, dx, M, N, K, K);
}

} // namespace kernel
