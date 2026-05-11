#pragma once

#include <cstdint>
#include <vector>

namespace kernel {

constexpr size_t idx2(size_t i, size_t j, size_t J) noexcept { return i * J + j; }
constexpr size_t idx3(size_t a, size_t b, size_t c, size_t B, size_t C) noexcept {
    return (a * B + b) * C + c;
}

[[nodiscard]] float silu(float x);
[[nodiscard]] float silu_derivative(float x);

class RopeCache {
public:
    RopeCache(float rope_base, size_t head_dim, size_t rope_dim);
    [[nodiscard]] size_t rot_dim() const noexcept;
    void ensure(size_t max_pos_exclusive);
    [[nodiscard]] float cos_at(size_t pos, size_t i) const;
    [[nodiscard]] float sin_at(size_t pos, size_t i) const;

private:
    float rope_base_ = 10000.0f;
    size_t head_dim_ = 0;
    size_t rope_dim_ = 0;
    size_t rot_ = 0;
    size_t half_ = 0;
    size_t max_pos_cached_ = 0;
    std::vector<float> cos_;
    std::vector<float> sin_;
};

struct AddParams {
    size_t size = 0;
};

struct LinearParams {
    size_t rows = 0;
    size_t in_dim = 0;
    size_t out_dim = 0;
};

struct RmsNormParams {
    size_t rows = 0;
    size_t d_model = 0;
};

struct RopeParams {
    size_t seq_len = 0;
    size_t num_heads = 0;
    size_t head_dim = 0;
};

struct AttentionParams {
    size_t seq_len = 0;
    size_t num_heads = 0;
    size_t num_kv_heads = 0;
    size_t head_dim = 0;
    size_t past_len = 0;
    size_t total_kv_len = 0;
    size_t kv_stride = 0;
    bool causal = true;
    bool use_cache = true;
};

struct SoftmaxParams {
    size_t n = 0;
};

struct SiluMulParams {
    size_t n = 0;
};

void add(const float* a, const float* b, float* y, const AddParams& p);
// C[M×N] = A[M×K] * B[N×K]^T (row-major A,B; each row of B is one output channel). Handwritten GEMM + SIMD dot.
void gemm_nt(const float* A, const float* B, float* C, size_t M, size_t N, size_t K);
void linear(const float* x, const float* w, const float* bias, float* y, const LinearParams& p);
void rms_norm(const float* x, const float* weight, float eps, float* y, const RmsNormParams& p);
void silu_mul(const float* gate, const float* up, float* hidden, const SiluMulParams& p);
void softmax(const float* logits, float* probs, const SoftmaxParams& p);
void apply_rope(float* x, const size_t* positions, RopeCache& cache, const RopeParams& p);

// Shared GQA forward: arbitrary seq_len >= 1; ctx is [seq_len * num_heads * head_dim].
// Pass attn_probs_out when training tape needs softmax cache; nullptr for inference decode or prefill without tape.
void gqa_attention_forward(const float* q, const float* k_all, const float* v_all, const float* attention_mask,
                           float* ctx, float* attn_probs_out, const AttentionParams& p);
void gqa_attention_backward(const float* q, const float* k_all, const float* v_all, const float* attention_mask,
                            const float* attn_probs_cached, const float* grad_ctx, float* grad_q, float* grad_k_all,
                            float* grad_v_all, const AttentionParams& p);

void softmax_backward_row(const float* prob, const float* grad_prob, float* grad_logits, const SoftmaxParams& p);
void silu_mul_backward(const float* gate, const float* up, const float* grad_hidden, float* grad_gate, float* grad_up,
                       const SiluMulParams& p);
void linear_backward(const float* x, const float* w, bool has_bias, const float* dy, float* dx, float* grad_w,
                     float* grad_b, const LinearParams& p);
void apply_rope_backward(const float* grad_out, float* grad_in, const size_t* positions, RopeCache& cache,
                         const RopeParams& p);
void rms_norm_backward(const float* x, const float* weight, float eps, const float* dy, float* dx, float* grad_weight,
                       const RmsNormParams& p);

void backward_output_projection(const float* hidden_states, const float* probs, const uint32_t* targets,
                                const size_t* steps, size_t valid_steps, size_t d_model, size_t vocab_size,
                                float* grad_output_projection);
void backward_hidden(const float* probs, const uint32_t* targets, const size_t* steps, size_t valid_steps,
                     const float* output_projection, size_t total_steps, size_t d_model, size_t vocab_size,
                     float* grad_hidden_out);
void backward_embedding(const uint32_t* token_ids, size_t seq_len, const float* grad_hidden, size_t d_model,
                        float* grad_token_embed);

} // namespace kernel

