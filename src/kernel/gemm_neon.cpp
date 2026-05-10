#include <arm_neon.h>
#include <cstddef>

// NEON variant of the gemm inner-kernel primitives. Declarations live at the
// top of src/kernel/gemm.cpp; signature drift surfaces as a link error.
namespace kernel::detail {

float gemm_dot(const float* a, const float* b, std::size_t len) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    for (; i + 4 <= len; i += 4) {
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);
        acc = vfmaq_f32(acc, va, vb);
    }
    float s = vaddvq_f32(acc);
    for (; i < len; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

void gemm_axpy(float* y, const float* x, float scale, std::size_t K) {
    const float32x4_t vs = vdupq_n_f32(scale);
    std::size_t k = 0;
    for (; k + 4 <= K; k += 4) {
        float32x4_t vy = vld1q_f32(y + k);
        const float32x4_t vx = vld1q_f32(x + k);
        vy = vfmaq_f32(vy, vx, vs);
        vst1q_f32(y + k, vy);
    }
    for (; k < K; ++k) {
        y[k] += scale * x[k];
    }
}

void gemm_dot4(const float* a,
               const float* b0, const float* b1, const float* b2, const float* b3,
               std::size_t K,
               float* c0, float* c1, float* c2, float* c3) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    std::size_t k = 0;
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
}

} // namespace kernel::detail
