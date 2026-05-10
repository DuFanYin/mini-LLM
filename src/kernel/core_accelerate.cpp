// Accelerate variant of the elementwise / softmax kernels declared in
// kernel.h. Selected by the configure script only for
// MINI_LLM_KERNEL_BACKEND=accelerate.

#include "kernel/kernel.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace kernel {

float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

float silu_derivative(float x) {
    const float s = 1.0f / (1.0f + std::exp(-x));
    return s + x * s * (1.0f - s);
}

void add(const float* a, const float* b, float* y, const AddParams& p) {
    vDSP_vadd(a, 1, b, 1, y, 1, p.size);
}

void softmax_stable(const float* logits, float* probs, const SoftmaxParams& p) {
    if (p.n == 0) {
        return;
    }
    const int n_int = static_cast<int>(p.n);
    float max_logit = -std::numeric_limits<float>::infinity();
    vDSP_maxv(logits, 1, &max_logit, p.n);
    if (max_logit == -std::numeric_limits<float>::infinity()) {
        std::fill_n(probs, p.n, 0.0f);
        return;
    }
    const float neg_max = -max_logit;
    vDSP_vsadd(logits, 1, &neg_max, probs, 1, p.n);
    vvexpf(probs, probs, &n_int);
    float denom = 0.0f;
    vDSP_sve(probs, 1, &denom, p.n);
    const float inv_denom = (denom > 0.0f) ? 1.0f / denom : 0.0f;
    vDSP_vsmul(probs, 1, &inv_denom, probs, 1, p.n);
}

void silu_mul(const float* gate, const float* up, float* hidden, const SiluMulParams& p) {
    if (p.n == 0) {
        return;
    }
    const int n_int = static_cast<int>(p.n);
    // hidden ← exp(-gate)
    const float neg_one = -1.0f;
    vDSP_vsmul(gate, 1, &neg_one, hidden, 1, p.n);
    vvexpf(hidden, hidden, &n_int);
    // hidden ← 1 + exp(-gate)
    const float one = 1.0f;
    vDSP_vsadd(hidden, 1, &one, hidden, 1, p.n);
    // hidden ← gate / hidden = silu(gate). Note: vDSP_vdiv computes C = A / B
    // with arg order (B, IB, A, IA, C, IC, N).
    vDSP_vdiv(hidden, 1, gate, 1, hidden, 1, p.n);
    // hidden ← hidden * up
    vDSP_vmul(hidden, 1, up, 1, hidden, 1, p.n);
}

void softmax_backward_row(const float* prob, const float* grad_prob, float* grad_logits, const SoftmaxParams& p) {
    if (p.n == 0) {
        return;
    }
    float dot = 0.0f;
    vDSP_dotpr(prob, 1, grad_prob, 1, &dot, p.n);
    // grad_logits ← grad_prob - dot
    const float neg_dot = -dot;
    vDSP_vsadd(grad_prob, 1, &neg_dot, grad_logits, 1, p.n);
    // grad_logits ← prob * grad_logits
    vDSP_vmul(prob, 1, grad_logits, 1, grad_logits, 1, p.n);
}

void silu_mul_backward(const float* gate, const float* up, const float* grad_hidden, float* grad_gate, float* grad_up,
                       const SiluMulParams& p) {
    for (size_t i = 0; i < p.n; ++i) {
        grad_gate[i] = silu_derivative(gate[i]) * up[i] * grad_hidden[i];
        grad_up[i] = silu(gate[i]) * grad_hidden[i];
    }
}

} // namespace kernel
