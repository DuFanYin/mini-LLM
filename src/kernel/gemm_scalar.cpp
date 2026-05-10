#include <cstddef>

// Scalar variant of the gemm inner-kernel primitives. Declarations live at the
// top of src/kernel/gemm.cpp; signature drift surfaces as a link error.
namespace kernel::detail {

float gemm_dot(const float* a, const float* b, std::size_t len) {
    float s = 0.0f;
    for (std::size_t i = 0; i < len; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

void gemm_axpy(float* y, const float* x, float scale, std::size_t K) {
    for (std::size_t k = 0; k < K; ++k) {
        y[k] += scale * x[k];
    }
}

void gemm_dot4(const float* a,
               const float* b0, const float* b1, const float* b2, const float* b3,
               std::size_t K,
               float* c0, float* c1, float* c2, float* c3) {
    for (std::size_t k = 0; k < K; ++k) {
        const float av = a[k];
        *c0 += av * b0[k];
        *c1 += av * b1[k];
        *c2 += av * b2[k];
        *c3 += av * b3[k];
    }
}

} // namespace kernel::detail
