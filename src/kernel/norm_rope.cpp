#include "kernel/kernel.h"

#include <algorithm>
#include <cmath>

namespace kernel {

RopeCache::RopeCache(float rope_base, size_t head_dim, size_t rope_dim)
    : rope_base_(rope_base), head_dim_(head_dim), rope_dim_(rope_dim) {
    const size_t rotary_dim = (rope_dim_ == 0) ? head_dim_ : std::min(rope_dim_, head_dim_);
    rot_ = rotary_dim - (rotary_dim % 2);
    half_ = rot_ / 2;
}

size_t RopeCache::rot_dim() const noexcept {
    return rot_;
}

void RopeCache::ensure(size_t max_pos_exclusive) {
    if (half_ == 0 || max_pos_exclusive <= max_pos_cached_) {
        return;
    }
    const size_t old = max_pos_cached_;
    max_pos_cached_ = max_pos_exclusive;
    cos_.resize(max_pos_cached_ * half_);
    sin_.resize(max_pos_cached_ * half_);
    for (size_t p = old; p < max_pos_cached_; ++p) {
        for (size_t i = 0; i < half_; ++i) {
            const float inv_freq = std::pow(rope_base_, -static_cast<float>(i) / static_cast<float>(half_));
            const float theta = static_cast<float>(p) * inv_freq;
            cos_[p * half_ + i] = std::cos(theta);
            sin_[p * half_ + i] = std::sin(theta);
        }
    }
}

float RopeCache::cos_at(size_t pos, size_t i) const {
    return cos_[pos * half_ + i];
}

float RopeCache::sin_at(size_t pos, size_t i) const {
    return sin_[pos * half_ + i];
}

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

void apply_rope(float* x, const size_t* positions, RopeCache& cache, const RopeParams& p) {
    const size_t rot = std::min(cache.rot_dim(), p.head_dim - (p.head_dim % 2));
    size_t max_pos = 0;
    for (size_t i = 0; i < p.seq_len; ++i) {
        const size_t pp = positions[i];
        if (pp > max_pos) {
            max_pos = pp;
        }
    }
    cache.ensure(max_pos + 1);

    for (size_t si = 0; si < p.seq_len; ++si) {
        const size_t pos = positions[si];
        for (size_t hi = 0; hi < p.num_heads; ++hi) {
            for (size_t i = 0; i < rot; i += 2) {
                const size_t fi = i / 2;
                const float ct = cache.cos_at(pos, fi);
                const float st = cache.sin_at(pos, fi);
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

void apply_rope_backward(const float* grad_out, float* grad_in, const size_t* positions, RopeCache& cache,
                         const RopeParams& p) {
    std::fill(grad_in, grad_in + (p.seq_len * p.num_heads * p.head_dim), 0.0f);

    const size_t rot = std::min(cache.rot_dim(), p.head_dim - (p.head_dim % 2));
    for (size_t si = 0; si < p.seq_len; ++si) {
        const size_t pos = positions[si];
        for (size_t hi = 0; hi < p.num_heads; ++hi) {
            for (size_t j = rot; j < p.head_dim; ++j) {
                grad_in[idx3(si, hi, j, p.num_heads, p.head_dim)] =
                    grad_out[idx3(si, hi, j, p.num_heads, p.head_dim)];
            }
            for (size_t i = 0; i < rot; i += 2) {
                const size_t fi = i / 2;
                const float ct = cache.cos_at(pos, fi);
                const float st = cache.sin_at(pos, fi);
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

} // namespace kernel
