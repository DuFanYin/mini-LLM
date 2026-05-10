#include <immintrin.h>
#include <cstddef>

// AVX2 variant of the gemm inner-kernel primitives. Declarations live at the
// top of src/kernel/gemm.cpp; signature drift surfaces as a link error.
namespace kernel::detail {

float gemm_dot(const float* a, const float* b, std::size_t len) {
    __m256 acc = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    const __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float s = _mm_cvtss_f32(lo);
    for (; i < len; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

void gemm_axpy(float* y, const float* x, float scale, std::size_t K) {
    const __m256 vs = _mm256_set1_ps(scale);
    std::size_t k = 0;
    for (; k + 8 <= K; k += 8) {
        __m256 vy = _mm256_loadu_ps(y + k);
        const __m256 vx = _mm256_loadu_ps(x + k);
        vy = _mm256_fmadd_ps(vx, vs, vy);
        _mm256_storeu_ps(y + k, vy);
    }
    for (; k < K; ++k) {
        y[k] += scale * x[k];
    }
}

void gemm_dot4(const float* a,
               const float* b0, const float* b1, const float* b2, const float* b3,
               std::size_t K,
               float* c0, float* c1, float* c2, float* c3) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    std::size_t k = 0;
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
}

} // namespace kernel::detail
