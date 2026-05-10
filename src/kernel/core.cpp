#include "kernel/kernel.h"

#include <cmath>
#include <limits>
#include <vector>

namespace kernel {

float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

float silu_derivative(float x) {
    const float s = 1.0f / (1.0f + std::exp(-x));
    return s + x * s * (1.0f - s);
}

void add(const float* a, const float* b, float* y, const AddParams& p) {
    for (size_t i = 0; i < p.size; ++i) {
        y[i] = a[i] + b[i];
    }
}

void softmax_stable(const float* logits, float* probs, const SoftmaxParams& p) {
    float max_logit = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < p.n; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    float denom = 0.0f;
    for (size_t i = 0; i < p.n; ++i) {
        const float e = std::exp(logits[i] - max_logit);
        probs[i] = e;
        denom += e;
    }
    const float inv_denom = (denom > 0.0f) ? 1.0f / denom : 0.0f;
    for (size_t i = 0; i < p.n; ++i) {
        probs[i] *= inv_denom;
    }
}

void silu_mul(const float* gate, const float* up, float* hidden, const SiluMulParams& p) {
    for (size_t i = 0; i < p.n; ++i) {
        hidden[i] = silu(gate[i]) * up[i];
    }
}

void softmax_backward_row(const float* prob, const float* grad_prob, float* grad_logits, const SoftmaxParams& p) {
    float dot = 0.0f;
    for (size_t i = 0; i < p.n; ++i) {
        dot += prob[i] * grad_prob[i];
    }
    for (size_t i = 0; i < p.n; ++i) {
        grad_logits[i] = prob[i] * (grad_prob[i] - dot);
    }
}

void silu_mul_backward(const float* gate, const float* up, const float* grad_hidden, float* grad_gate, float* grad_up,
                       const SiluMulParams& p) {
    for (size_t i = 0; i < p.n; ++i) {
        grad_gate[i] = silu_derivative(gate[i]) * up[i] * grad_hidden[i];
        grad_up[i] = silu(gate[i]) * grad_hidden[i];
    }
}

} // namespace kernel
