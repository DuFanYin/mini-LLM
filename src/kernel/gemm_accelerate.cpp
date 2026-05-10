// Accelerate variant of the public gemm/linear ABI. Selected at CMake
// configure time via MINI_LLM_KERNEL_BACKEND=accelerate. When selected this
// translation unit replaces gemm.cpp wholesale and routes all GEMM-shaped
// work through cblas_sgemm (vecLib in Accelerate.framework). The neon / avx2
// / scalar variants stay available — pick a different backend value to use
// the handwritten driver.

#include "kernel/kernel.h"

// Opt into the post-macOS-13.3 cblas declarations; the legacy ones in
// vecLib's Accelerate.h are marked deprecated. ILP64 is intentionally NOT
// enabled — we pass plain `int` dimensions.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <cstddef>
#include <cstring>

namespace kernel {

void gemm_nt(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
    if (M == 0 || N == 0 || K == 0) {
        return;
    }
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasTrans,
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                /*alpha=*/1.0f,
                A, static_cast<int>(K),
                B, static_cast<int>(K),
                /*beta=*/0.0f,
                C, static_cast<int>(N));
}

void linear(const float* x, const float* w, const float* bias, float* y, const LinearParams& p) {
    const size_t M = p.rows;
    const size_t N = p.out_dim;
    const size_t K = p.in_dim;
    if (M == 0 || N == 0 || K == 0) {
        return;
    }
    float beta = 0.0f;
    if (bias != nullptr) {
        for (size_t m = 0; m < M; ++m) {
            std::memcpy(y + m * N, bias, N * sizeof(float));
        }
        beta = 1.0f;
    }
    // y[M × N] = x[M × K] @ w[N × K]^T  (+ existing bias rows when beta=1)
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasTrans,
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                /*alpha=*/1.0f,
                x, static_cast<int>(K),
                w, static_cast<int>(K),
                beta,
                y, static_cast<int>(N));
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

    // grad_w[N × K] += dy[M × N]^T @ x[M × K]
    cblas_sgemm(CblasRowMajor,
                CblasTrans, CblasNoTrans,
                static_cast<int>(N), static_cast<int>(K), static_cast<int>(M),
                /*alpha=*/1.0f,
                dy, static_cast<int>(N),
                x, static_cast<int>(K),
                /*beta=*/1.0f,
                grad_w, static_cast<int>(K));

    // dx[M × K] += dy[M × N] @ w[N × K]
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                static_cast<int>(M), static_cast<int>(K), static_cast<int>(N),
                /*alpha=*/1.0f,
                dy, static_cast<int>(N),
                w, static_cast<int>(K),
                /*beta=*/1.0f,
                dx, static_cast<int>(K));
}

} // namespace kernel
