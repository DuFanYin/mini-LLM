#include "kernel/kernel.h"

#include <cmath>
#include <limits>
#include <vector>

namespace kernel {

void add(const float* a, const float* b, float* y, const AddParams& p) {
    for (size_t i = 0; i < p.size; ++i) {
        y[i] = a[i] + b[i];
    }
}

void softmax(const float* logits, float* probs, const SoftmaxParams& p) {
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

void adamw_update(float* param, const float* grad, float* m, float* v, size_t n, const AdamwParams& p) {
    if (n == 0) {
        return;
    }
    const float one_minus_beta1 = 1.0f - p.beta1;
    const float one_minus_beta2 = 1.0f - p.beta2;
    const float inv_bias_c1 = 1.0f / (1.0f - std::pow(p.beta1, static_cast<float>(p.step)));
    const float inv_bias_c2 = 1.0f / (1.0f - std::pow(p.beta2, static_cast<float>(p.step)));

    for (size_t i = 0; i < n; ++i) {
        const float gi = grad[i];
        const float mi = p.beta1 * m[i] + one_minus_beta1 * gi;
        const float vi = p.beta2 * v[i] + one_minus_beta2 * gi * gi;
        m[i] = mi;
        v[i] = vi;
        const float m_hat = mi * inv_bias_c1;
        const float v_hat = vi * inv_bias_c2;
        const float pi = param[i];
        const float denom = std::sqrt(v_hat) + p.eps;
        param[i] = pi - p.lr * (m_hat / denom + p.weight_decay * pi);
    }
}

double sum_squares(const float* x, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double xi = static_cast<double>(x[i]);
        s += xi * xi;
    }
    return s;
}

void scale(float* x, size_t n, float s) {
    for (size_t i = 0; i < n; ++i) {
        x[i] *= s;
    }
}

} // namespace kernel
