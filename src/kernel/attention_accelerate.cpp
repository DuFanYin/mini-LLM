// Accelerate variant of the public GQA attention ABI. Selected by the
// configure script only for MINI_LLM_KERNEL_BACKEND=accelerate.

#include "kernel/kernel.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace kernel {
namespace {
// Anonymous helpers — keep order aligned with attention.cpp:
//   apply_causal_and_additive_mask (softmax: kernel::softmax from core_accelerate.cpp).

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

} // namespace

void gqa_attention_forward(const float* q, const float* k_all, const float* v_all, const float* attention_mask,
                           float* ctx, float* attn_probs_out, const AttentionParams& p) {
    const size_t s = p.seq_len;
    const size_t hq = p.num_heads;
    const size_t hkv = p.num_kv_heads;
    const size_t d_head = p.head_dim;
    const size_t total_kv_len = p.total_kv_len;
    const size_t kv_stride = (p.kv_stride == 0) ? total_kv_len : p.kv_stride;
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d_head));
    const bool emit_probs = (attn_probs_out != nullptr);
    const bool have_mask = (attention_mask != nullptr);

    if (emit_probs) {
        std::fill(attn_probs_out, attn_probs_out + (s * hq * total_kv_len), 0.0f);
    }
    std::fill(ctx, ctx + (s * hq * d_head), 0.0f);

    std::vector<float> logits(s * total_kv_len, 0.0f);
    std::vector<float> probs(s * total_kv_len, 0.0f);

    for (size_t qh = 0; qh < hq; ++qh) {
        const size_t kh = qh / (hq / hkv);
        const float* q_head = q + qh * d_head;
        const float* k_head = k_all + kh * kv_stride * d_head;
        float* ctx_head = ctx + qh * d_head;

        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans, CblasTrans,
                    static_cast<int>(s), static_cast<int>(total_kv_len), static_cast<int>(d_head),
                    inv_sqrt_d,
                    q_head, static_cast<int>(hq * d_head),
                    k_head, static_cast<int>(d_head),
                    0.0f,
                    logits.data(), static_cast<int>(total_kv_len));

        for (size_t si = 0; si < s; ++si) {
            const size_t abs_pos = p.use_cache ? (p.past_len + si) : si;
            const size_t visible = p.causal ? std::min(abs_pos + 1, total_kv_len) : total_kv_len;
            const float* mask_row = have_mask ? (attention_mask + si * total_kv_len) : nullptr;
            float* logits_row = logits.data() + si * total_kv_len;
            float* probs_row = probs.data() + si * total_kv_len;
            apply_causal_and_additive_mask(logits_row, visible, mask_row, total_kv_len);
            softmax(logits_row, probs_row, SoftmaxParams{total_kv_len});
            if (emit_probs) {
                float* out_row = attn_probs_out + (si * hq + qh) * total_kv_len;
                std::copy(probs_row, probs_row + total_kv_len, out_row);
            }
        }

        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans, CblasNoTrans,
                    static_cast<int>(s), static_cast<int>(d_head), static_cast<int>(total_kv_len),
                    1.0f,
                    probs.data(), static_cast<int>(total_kv_len),
                    v_all + kh * kv_stride * d_head, static_cast<int>(d_head),
                    0.0f,
                    ctx_head, static_cast<int>(hq * d_head));
    }
}

void gqa_attention_backward(const float* q, const float* k_all, const float* v_all, const float* attention_mask,
                            const float* attn_probs_cached, const float* grad_ctx, float* grad_q, float* grad_k_all,
                            float* grad_v_all, const AttentionParams& p) {
    const size_t s = p.seq_len;
    const size_t hq = p.num_heads;
    const size_t hkv = p.num_kv_heads;
    const size_t d_head = p.head_dim;
    const size_t total_kv_len = p.total_kv_len;
    const size_t kv_stride = (p.kv_stride == 0) ? total_kv_len : p.kv_stride;
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
            const size_t kh = qh / (hq / hkv);
            if (!have_probs_cache) {
                for (size_t ti = 0; ti < total_kv_len; ++ti) {
                    float dot = 0.0f;
                    for (size_t dd = 0; dd < d_head; ++dd) {
                        dot += q[idx3(si, qh, dd, hq, d_head)] *
                               k_all[(kh * kv_stride + ti) * d_head + dd];
                    }
                    logits[ti] = dot * inv_sqrt_d;
                }
                apply_causal_and_additive_mask(logits.data(), visible, mask_row, total_kv_len);
                softmax(logits.data(), probs.data(), SoftmaxParams{total_kv_len});
            } else {
                const float* row = attn_probs_cached + (si * hq + qh) * total_kv_len;
                std::copy(row, row + total_kv_len, probs.data());
            }

            for (size_t ti = 0; ti < total_kv_len; ++ti) {
                float gp = 0.0f;
                for (size_t d = 0; d < d_head; ++d) {
                    gp += grad_ctx[idx3(si, qh, d, hq, d_head)] * v_all[(kh * kv_stride + ti) * d_head + d];
                }
                d_prob[ti] = gp;
            }

            const SoftmaxParams sp{total_kv_len};
            softmax_backward_row(probs.data(), d_prob.data(), d_logit.data(), sp);

            for (size_t ti = 0; ti < total_kv_len; ++ti) {
                const float scale = d_logit[ti] * inv_sqrt_d;
                for (size_t d = 0; d < d_head; ++d) {
                    grad_q[idx3(si, qh, d, hq, d_head)] += scale * k_all[(kh * kv_stride + ti) * d_head + d];
                    grad_k_all[(kh * total_kv_len + ti) * d_head + d] += scale * q[idx3(si, qh, d, hq, d_head)];
                    grad_v_all[(kh * total_kv_len + ti) * d_head + d] +=
                        probs[ti] * grad_ctx[idx3(si, qh, d, hq, d_head)];
                }
            }
        }
    }
}

} // namespace kernel
