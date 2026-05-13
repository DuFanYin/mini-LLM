#include "kernel/kernel.h"

#include <algorithm>
#include <cmath>

namespace kernel {

// ============================================================
// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight   (forward / backward)
// ============================================================

void rms_norm(const float* x, const float* weight, float eps, float* y, const RmsNormParams& p) {
    for (size_t r = 0; r < p.rows; ++r) {
        float mean_sq = 0.0f;
        for (size_t c = 0; c < p.d_model; ++c) {
            const float v = x[idx2(r, c, p.d_model)];
            mean_sq += v * v;
        }
        mean_sq /= static_cast<float>(p.d_model);
        const float inv = 1.0f / std::sqrt(mean_sq + eps);
        for (size_t c = 0; c < p.d_model; ++c) {
            y[idx2(r, c, p.d_model)] = x[idx2(r, c, p.d_model)] * inv * weight[c];
        }
    }
}

void rms_norm_backward(const float* x, const float* weight, float eps, const float* dy, float* dx, float* grad_weight,
                       const RmsNormParams& p) {
    for (size_t r = 0; r < p.rows; ++r) {
        float mean_sq = 0.0f;
        for (size_t c = 0; c < p.d_model; ++c) {
            const float v = x[idx2(r, c, p.d_model)];
            mean_sq += v * v;
        }
        mean_sq /= static_cast<float>(p.d_model);
        const float rms = std::sqrt(mean_sq + eps);
        const float inv_rms = 1.0f / rms;
        const float r3 = inv_rms * inv_rms * inv_rms;

        float inner = 0.0f;
        for (size_t c = 0; c < p.d_model; ++c) {
            inner += dy[idx2(r, c, p.d_model)] * weight[c] * x[idx2(r, c, p.d_model)];
        }
        const float factor = r3 / static_cast<float>(p.d_model);
        for (size_t c = 0; c < p.d_model; ++c) {
            const float v = x[idx2(r, c, p.d_model)];
            dx[idx2(r, c, p.d_model)] += dy[idx2(r, c, p.d_model)] * weight[c] * inv_rms - factor * v * inner;
            grad_weight[c] += dy[idx2(r, c, p.d_model)] * v * inv_rms;
        }
    }
}

// ============================================================
// Rotary Positional Embedding: rotate (x_2i, x_2i+1) by angle theta(pos, i)   (forward / backward)
// `cos_tab` / `sin_tab` are row-major [pos, freq_idx] with `half = rot / 2` columns;
// the cache living in `model::RopeCache` is responsible for sizing them.
// ============================================================

void apply_rope(float* x, const size_t* positions, const float* cos_tab, const float* sin_tab, size_t rot,
                const RopeParams& p) {
    const size_t eff_rot = std::min(rot, p.head_dim - (p.head_dim % 2));
    const size_t half = rot / 2;
    for (size_t si = 0; si < p.seq_len; ++si) {
        const size_t pos = positions[si];
        for (size_t hi = 0; hi < p.num_heads; ++hi) {
            for (size_t i = 0; i < eff_rot; i += 2) {
                const size_t fi = i / 2;
                const float ct = cos_tab[pos * half + fi];
                const float st = sin_tab[pos * half + fi];
                const size_t a = idx3(si, hi, i, p.num_heads, p.head_dim);
                const size_t b = idx3(si, hi, i + 1, p.num_heads, p.head_dim);
                const float x0 = x[a];
                const float x1 = x[b];
                x[a] = x0 * ct - x1 * st;
                x[b] = x0 * st + x1 * ct;
            }
        }
    }
}

void apply_rope_backward(const float* grad_out, float* grad_in, const size_t* positions, const float* cos_tab,
                         const float* sin_tab, size_t rot, const RopeParams& p) {
    std::fill(grad_in, grad_in + (p.seq_len * p.num_heads * p.head_dim), 0.0f);

    const size_t eff_rot = std::min(rot, p.head_dim - (p.head_dim % 2));
    const size_t half = rot / 2;
    for (size_t si = 0; si < p.seq_len; ++si) {
        const size_t pos = positions[si];
        for (size_t hi = 0; hi < p.num_heads; ++hi) {
            for (size_t j = eff_rot; j < p.head_dim; ++j) {
                grad_in[idx3(si, hi, j, p.num_heads, p.head_dim)] =
                    grad_out[idx3(si, hi, j, p.num_heads, p.head_dim)];
            }
            for (size_t i = 0; i < eff_rot; i += 2) {
                const size_t fi = i / 2;
                const float ct = cos_tab[pos * half + fi];
                const float st = sin_tab[pos * half + fi];
                const size_t a = idx3(si, hi, i, p.num_heads, p.head_dim);
                const size_t b = idx3(si, hi, i + 1, p.num_heads, p.head_dim);
                const float gy0 = grad_out[a];
                const float gy1 = grad_out[b];
                grad_in[a] = gy0 * ct + gy1 * st;
                grad_in[b] = -gy0 * st + gy1 * ct;
            }
        }
    }
}

} // namespace kernel
