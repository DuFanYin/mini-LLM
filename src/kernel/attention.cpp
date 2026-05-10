#include "kernel/kernel.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace kernel {
namespace {

[[nodiscard]] size_t map_q_to_kv_head(size_t qh, size_t hq, size_t hkv) {
    const size_t group = hq / hkv;
    return qh / group;
}

[[nodiscard]] float dot_scaled(const float* q, const float* k_all, size_t si, size_t qh, size_t ti, size_t kh,
                               size_t hq, size_t hkv, size_t d_head, float inv_sqrt_d) {
    float dot = 0.0f;
    for (size_t d = 0; d < d_head; ++d) {
        dot += q[idx3(si, qh, d, hq, d_head)] * k_all[idx3(ti, kh, d, hkv, d_head)];
    }
    return dot * inv_sqrt_d;
}

void apply_causal_and_additive_mask(float* logits, size_t visible, const float* mask_row, size_t total_kv_len) {
    for (size_t ti = 0; ti < total_kv_len; ++ti) {
        if (ti >= visible) {
            logits[ti] = -std::numeric_limits<float>::infinity();
            continue;
        }
        if (mask_row != nullptr) {
            logits[ti] += mask_row[ti];
        }
    }
}

[[nodiscard]] float accumulate_weighted_values(const float* probs, const float* v_all, size_t kh, size_t d, size_t hkv,
                                               size_t d_head, size_t total_kv_len) {
    float acc = 0.0f;
    for (size_t ti = 0; ti < total_kv_len; ++ti) {
        acc += probs[ti] * v_all[idx3(ti, kh, d, hkv, d_head)];
    }
    return acc;
}

void softmax_inplace(float* logits, float* probs, size_t n) {
    float max_logit = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < n; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    float denom = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float e = std::exp(logits[i] - max_logit);
        probs[i] = e;
        denom += e;
    }
    const float inv_denom = (denom > 0.0f) ? 1.0f / denom : 0.0f;
    for (size_t i = 0; i < n; ++i) {
        probs[i] *= inv_denom;
    }
}

} // namespace

void gqa_attention_prefill(const float* q, const float* k_all, const float* v_all, const float* attention_mask,
                           float* ctx, float* attn_probs_out,
                           const AttentionParams& p) {
    const size_t s = p.seq_len;
    const size_t hq = p.num_heads;
    const size_t hkv = p.num_kv_heads;
    const size_t d_head = p.head_dim;
    const size_t total_kv_len = p.total_kv_len;
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d_head));
    const bool emit_probs = (attn_probs_out != nullptr);
    if (emit_probs) {
        std::fill(attn_probs_out, attn_probs_out + (s * hq * total_kv_len), 0.0f);
    }
    std::fill(ctx, ctx + (s * hq * d_head), 0.0f);

    const bool have_mask = (attention_mask != nullptr);

    std::vector<float> logits(total_kv_len, 0.0f);
    std::vector<float> probs(total_kv_len, 0.0f);
    for (size_t si = 0; si < s; ++si) {
        const size_t abs_pos = p.use_cache ? (p.past_len + si) : si;
        const size_t visible = p.causal ? std::min(abs_pos + 1, total_kv_len) : total_kv_len;
        const float* mask_row = have_mask ? (attention_mask + si * total_kv_len) : nullptr;
        for (size_t qh = 0; qh < hq; ++qh) {
            const size_t kh = map_q_to_kv_head(qh, hq, hkv);
            for (size_t ti = 0; ti < total_kv_len; ++ti) {
                logits[ti] = dot_scaled(q, k_all, si, qh, ti, kh, hq, hkv, d_head, inv_sqrt_d);
            }
            apply_causal_and_additive_mask(logits.data(), visible, mask_row, total_kv_len);
            softmax_inplace(logits.data(), probs.data(), total_kv_len);
            if (emit_probs) {
                float* row = attn_probs_out + (si * hq + qh) * total_kv_len;
                for (size_t ti = 0; ti < total_kv_len; ++ti) {
                    row[ti] = probs[ti];
                }
            }
            for (size_t d = 0; d < d_head; ++d) {
                ctx[idx3(si, qh, d, hq, d_head)] =
                    accumulate_weighted_values(probs.data(), v_all, kh, d, hkv, d_head, total_kv_len);
            }
        }
    }
}

void gqa_attention_decode(const float* q, const float* k_all, const float* v_all, const float* attention_mask, float* ctx,
                          const AttentionParams& p) {
    if (p.seq_len != 1) {
        gqa_attention_prefill(q, k_all, v_all, attention_mask, ctx, nullptr, p);
        return;
    }
    const size_t hq = p.num_heads;
    const size_t hkv = p.num_kv_heads;
    const size_t d_head = p.head_dim;
    const size_t total_kv_len = p.total_kv_len;
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d_head));
    std::fill(ctx, ctx + (hq * d_head), 0.0f);

    const bool have_mask = (attention_mask != nullptr);

    const size_t visible = p.causal ? std::min(p.past_len + 1, total_kv_len) : total_kv_len;
    const float* mask_row = have_mask ? attention_mask : nullptr;

    std::vector<float> logits(total_kv_len, 0.0f);
    std::vector<float> probs(total_kv_len, 0.0f);
    for (size_t qh = 0; qh < hq; ++qh) {
        const size_t kh = map_q_to_kv_head(qh, hq, hkv);
        for (size_t ti = 0; ti < total_kv_len; ++ti) {
            logits[ti] = dot_scaled(q, k_all, 0, qh, ti, kh, hq, hkv, d_head, inv_sqrt_d);
        }
        apply_causal_and_additive_mask(logits.data(), visible, mask_row, total_kv_len);
        softmax_inplace(logits.data(), probs.data(), total_kv_len);
        for (size_t d = 0; d < d_head; ++d) {
            ctx[idx2(qh, d, d_head)] = accumulate_weighted_values(probs.data(), v_all, kh, d, hkv, d_head, total_kv_len);
        }
    }
}

void gqa_attention_prefill_backward(const float* q, const float* k_all, const float* v_all, const float* attention_mask,
                                    const float* attn_probs_cached, const float* grad_ctx, float* grad_q, float* grad_k_all,
                                    float* grad_v_all, const AttentionParams& p) {
    const size_t s = p.seq_len;
    const size_t hq = p.num_heads;
    const size_t hkv = p.num_kv_heads;
    const size_t d_head = p.head_dim;
    const size_t total_kv_len = p.total_kv_len;
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d_head));
    const bool have_probs_cache = (attn_probs_cached != nullptr);
    const bool have_mask = (attention_mask != nullptr);

    std::fill(grad_q, grad_q + (s * hq * d_head), 0.0f);
    std::fill(grad_k_all, grad_k_all + (total_kv_len * hkv * d_head), 0.0f);
    std::fill(grad_v_all, grad_v_all + (total_kv_len * hkv * d_head), 0.0f);

    std::vector<float> logits(total_kv_len, 0.0f);
    std::vector<float> probs(total_kv_len, 0.0f);
    std::vector<float> d_prob(total_kv_len, 0.0f);
    std::vector<float> d_logit(total_kv_len, 0.0f);

    for (size_t si = 0; si < s; ++si) {
        const size_t abs_pos = p.use_cache ? (p.past_len + si) : si;
        const size_t visible = p.causal ? std::min(abs_pos + 1, total_kv_len) : total_kv_len;
        const float* mask_row = have_mask ? (attention_mask + si * total_kv_len) : nullptr;

        for (size_t qh = 0; qh < hq; ++qh) {
            const size_t kh = map_q_to_kv_head(qh, hq, hkv);
            if (!have_probs_cache) {
                for (size_t ti = 0; ti < total_kv_len; ++ti) {
                    logits[ti] = dot_scaled(q, k_all, si, qh, ti, kh, hq, hkv, d_head, inv_sqrt_d);
                }
                apply_causal_and_additive_mask(logits.data(), visible, mask_row, total_kv_len);
                softmax_inplace(logits.data(), probs.data(), total_kv_len);
            } else {
                const float* row = attn_probs_cached + (si * hq + qh) * total_kv_len;
                for (size_t ti = 0; ti < total_kv_len; ++ti) {
                    probs[ti] = row[ti];
                }
            }

            for (size_t ti = 0; ti < total_kv_len; ++ti) {
                float gp = 0.0f;
                for (size_t d = 0; d < d_head; ++d) {
                    gp += grad_ctx[idx3(si, qh, d, hq, d_head)] * v_all[idx3(ti, kh, d, hkv, d_head)];
                }
                d_prob[ti] = gp;
            }

            const SoftmaxParams sp{total_kv_len};
            softmax_backward_row(probs.data(), d_prob.data(), d_logit.data(), sp);

            for (size_t ti = 0; ti < total_kv_len; ++ti) {
                const float scale = d_logit[ti] * inv_sqrt_d;
                for (size_t d = 0; d < d_head; ++d) {
                    grad_q[idx3(si, qh, d, hq, d_head)] += scale * k_all[idx3(ti, kh, d, hkv, d_head)];
                    grad_k_all[idx3(ti, kh, d, hkv, d_head)] += scale * q[idx3(si, qh, d, hq, d_head)];
                    grad_v_all[idx3(ti, kh, d, hkv, d_head)] += probs[ti] * grad_ctx[idx3(si, qh, d, hq, d_head)];
                }
            }
        }
    }
}

} // namespace kernel
