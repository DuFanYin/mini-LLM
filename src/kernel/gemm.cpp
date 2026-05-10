#include "kernel/kernel.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define MINI_LLM_HAVE_NEON 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define MINI_LLM_HAVE_AVX2 1
#endif

namespace kernel {
namespace {

#if MINI_LLM_HAVE_NEON
inline float dot_k(const float* a, const float* b, size_t len) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc = vfmaq_f32(acc, va, vb);
    }
    float s = vaddvq_f32(acc);
    for (; i < len; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

inline void axpy_k(float* y, const float* x, float scale, size_t K) {
    const float32x4_t vs = vdupq_n_f32(scale);
    size_t k = 0;
    for (; k + 4 <= K; k += 4) {
        float32x4_t vy = vld1q_f32(y + k);
        float32x4_t vx = vld1q_f32(x + k);
        vy = vfmaq_f32(vy, vx, vs);
        vst1q_f32(y + k, vy);
    }
    for (; k < K; ++k) {
        y[k] += scale * x[k];
    }
}
#elif MINI_LLM_HAVE_AVX2
inline float dot_k(const float* a, const float* b, size_t len) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float s = _mm_cvtss_f32(lo);
    for (; i < len; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

inline void axpy_k(float* y, const float* x, float scale, size_t K) {
    const __m256 vs = _mm256_set1_ps(scale);
    size_t k = 0;
    for (; k + 8 <= K; k += 8) {
        __m256 vy = _mm256_loadu_ps(y + k);
        __m256 vx = _mm256_loadu_ps(x + k);
        vy = _mm256_fmadd_ps(vx, vs, vy);
        _mm256_storeu_ps(y + k, vy);
    }
    for (; k < K; ++k) {
        y[k] += scale * x[k];
    }
}
#else
inline float dot_k(const float* a, const float* b, size_t len) {
    float s = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

inline void axpy_k(float* y, const float* x, float scale, size_t K) {
    for (size_t k = 0; k < K; ++k) {
        y[k] += scale * x[k];
    }
}
#endif

void dot4_k(const float* a, const float* b0, const float* b1, const float* b2, const float* b3, size_t K, float* c0,
            float* c1, float* c2, float* c3) {
#if MINI_LLM_HAVE_NEON
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    size_t k = 0;
    for (; k + 4 <= K; k += 4) {
        const float32x4_t va = vld1q_f32(a + k);
        acc0 = vfmaq_f32(acc0, va, vld1q_f32(b0 + k));
        acc1 = vfmaq_f32(acc1, va, vld1q_f32(b1 + k));
        acc2 = vfmaq_f32(acc2, va, vld1q_f32(b2 + k));
        acc3 = vfmaq_f32(acc3, va, vld1q_f32(b3 + k));
    }
    *c0 += vaddvq_f32(acc0);
    *c1 += vaddvq_f32(acc1);
    *c2 += vaddvq_f32(acc2);
    *c3 += vaddvq_f32(acc3);
    for (; k < K; ++k) {
        const float av = a[k];
        *c0 += av * b0[k];
        *c1 += av * b1[k];
        *c2 += av * b2[k];
        *c3 += av * b3[k];
    }
#elif MINI_LLM_HAVE_AVX2
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    size_t k = 0;
    for (; k + 8 <= K; k += 8) {
        const __m256 va = _mm256_loadu_ps(a + k);
        acc0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b0 + k), acc0);
        acc1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b1 + k), acc1);
        acc2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b2 + k), acc2);
        acc3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b3 + k), acc3);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc0);
    for (float v : tmp) *c0 += v;
    _mm256_store_ps(tmp, acc1);
    for (float v : tmp) *c1 += v;
    _mm256_store_ps(tmp, acc2);
    for (float v : tmp) *c2 += v;
    _mm256_store_ps(tmp, acc3);
    for (float v : tmp) *c3 += v;
    for (; k < K; ++k) {
        const float av = a[k];
        *c0 += av * b0[k];
        *c1 += av * b1[k];
        *c2 += av * b2[k];
        *c3 += av * b3[k];
    }
#else
    for (size_t k = 0; k < K; ++k) {
        const float av = a[k];
        *c0 += av * b0[k];
        *c1 += av * b1[k];
        *c2 += av * b2[k];
        *c3 += av * b3[k];
    }
#endif
}

// C += A * B^T with A [M×K], B [N×K] row-major, C [M×ldc].
// Four output channels are accumulated together so each A row vector is loaded once per N tile.
void gemm_nt_accum(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, size_t ldc) {
    for (size_t m = 0; m < M; ++m) {
        float* crow = C + m * ldc;
        const float* arow = A + m * K;
        size_t n = 0;
        for (; n + 4 <= N; n += 4) {
            dot4_k(arow, B + (n + 0u) * K, B + (n + 1u) * K, B + (n + 2u) * K, B + (n + 3u) * K, K,
                   &crow[n + 0u], &crow[n + 1u], &crow[n + 2u], &crow[n + 3u]);
        }
        for (; n < N; ++n) {
            crow[n] += dot_k(arow, B + n * K, K);
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
                axpy_k(crow, B + n * K, scale, K);
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
                axpy_k(C + n * ldc, brow, scale, K);
            }
        }
    }
}

void reduce_bias_accum(const float* dy, float* grad_b, size_t M, size_t N) {
    for (size_t m = 0; m < M; ++m) {
        const float* row = dy + m * N;
        for (size_t n = 0; n < N; ++n) {
            grad_b[n] += row[n];
        }
    }
}

void init_bias_rows(float* y, const float* bias, size_t M, size_t N, size_t ldc) {
    for (size_t m = 0; m < M; ++m) {
        std::memcpy(y + m * ldc, bias, N * sizeof(float));
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
    const bool has_bias = (bias != nullptr);
    if (has_bias) {
        init_bias_rows(y, bias, M, N, N);
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
        reduce_bias_accum(dy, grad_b, M, N);
    }
    gemm_tn_accum(dy, x, grad_w, M, N, K, K);
    gemm_nn_accum(dy, w, dx, M, N, K, K);
}

} // namespace kernel
