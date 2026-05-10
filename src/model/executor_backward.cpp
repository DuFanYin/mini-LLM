#include "model/executor.h"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace model {

// ---------------------------------------------------------------------------
// One decoder layer in reverse order (vs forward):
//   (1) MLP + post-attn RMSNorm — dL/d(layer_out) → contribution at post-attention residual
//   (2) Attention — o_proj, GQA backward, RoPE on K
//   (3) QKV + pre-attn RMSNorm — RoPE on Q, projections, norm1
// ---------------------------------------------------------------------------

void backward_decoder_mlp(const std::vector<float>& grad_layer_out, const BlockForwardTape& tape,
                          const ModelConfig& config, const DecoderLayerWeights& weights,
                          DecoderLayerWeights& layer_grad, size_t seq_len, std::vector<float>& grad_post_attn) {
    const kernel::LinearParams down_params{seq_len, config.d_ff, config.d_model};
    std::vector<float> grad_mid(seq_len * config.d_ff, 0.0f);
    kernel::linear_backward(tape.hidden_mid.data(), weights.mlp.down.weight.data(), !weights.mlp.down.bias.empty(),
                            grad_layer_out.data(), grad_mid.data(), layer_grad.mlp.down.weight.data(),
                            layer_grad.mlp.down.bias.empty() ? nullptr : layer_grad.mlp.down.bias.data(), down_params);

    std::vector<float> grad_gate(seq_len * config.d_ff, 0.0f);
    std::vector<float> grad_up(seq_len * config.d_ff, 0.0f);
    const kernel::SiluMulParams silu_mul_params{grad_mid.size()};
    kernel::silu_mul_backward(tape.gate.data(), tape.up.data(), grad_mid.data(), grad_gate.data(), grad_up.data(),
                              silu_mul_params);

    const kernel::LinearParams gate_params{seq_len, config.d_model, config.d_ff};
    const kernel::LinearParams up_params{seq_len, config.d_model, config.d_ff};
    std::vector<float> grad_norm2(seq_len * config.d_model, 0.0f);
    kernel::linear_backward(tape.norm2_out.data(), weights.mlp.gate.weight.data(), !weights.mlp.gate.bias.empty(),
                            grad_gate.data(), grad_norm2.data(), layer_grad.mlp.gate.weight.data(),
                            layer_grad.mlp.gate.bias.empty() ? nullptr : layer_grad.mlp.gate.bias.data(), gate_params);
    kernel::linear_backward(tape.norm2_out.data(), weights.mlp.up.weight.data(), !weights.mlp.up.bias.empty(),
                            grad_up.data(), grad_norm2.data(), layer_grad.mlp.up.weight.data(),
                            layer_grad.mlp.up.bias.empty() ? nullptr : layer_grad.mlp.up.bias.data(), up_params);

    const kernel::RmsNormParams norm_params{seq_len, config.d_model};
    std::vector<float> grad_post_attn_from_mlp(seq_len * config.d_model, 0.0f);
    kernel::rms_norm_backward(tape.hidden_after_attn.data(), weights.norm2.weight.data(), weights.norm2.eps,
                              grad_norm2.data(), grad_post_attn_from_mlp.data(), layer_grad.norm2.weight.data(),
                              norm_params);

    grad_post_attn.assign(seq_len * config.d_model, 0.0f);
    for (size_t i = 0; i < grad_post_attn.size(); ++i) {
        grad_post_attn[i] = grad_layer_out[i] + grad_post_attn_from_mlp[i];
    }
}

void backward_decoder_attention(const std::vector<float>& grad_post_attn, const BlockForwardTape& tape,
                                const ModelConfig& config, const DecoderLayerWeights& weights,
                                DecoderLayerWeights& layer_grad, kernel::RopeCache& rope_k, size_t seq_len,
                                size_t q_proj_dim, size_t kv_proj_dim, size_t total_kv_len, size_t past_len,
                                std::vector<float>& grad_q, std::vector<float>& grad_k_pre_rope,
                                std::vector<float>& grad_v_proj) {
    const kernel::LinearParams o_proj_params{seq_len, q_proj_dim, config.d_model};
    std::vector<float> grad_context(seq_len * q_proj_dim, 0.0f);
    kernel::linear_backward(tape.attn_ctx.data(), weights.attention.o_proj.weight.data(),
                            !weights.attention.o_proj.bias.empty(), grad_post_attn.data(), grad_context.data(),
                            layer_grad.attention.o_proj.weight.data(),
                            layer_grad.attention.o_proj.bias.empty() ? nullptr : layer_grad.attention.o_proj.bias.data(),
                            o_proj_params);

    kernel::AttentionParams attention_params;
    attention_params.seq_len = seq_len;
    attention_params.num_heads = config.num_heads;
    attention_params.num_kv_heads = config.num_kv_heads;
    attention_params.head_dim = config.head_dim;
    attention_params.past_len = past_len;
    attention_params.total_kv_len = total_kv_len;
    attention_params.kv_stride = total_kv_len;
    attention_params.causal = true;
    attention_params.use_cache = true;

    grad_q.assign(tape.q_rope.size(), 0.0f);
    const float* k_all = (tape.k_all_ptr != nullptr) ? tape.k_all_ptr : (tape.k_all.empty() ? tape.k_rope.data() : tape.k_all.data());
    const float* v_all = (tape.v_all_ptr != nullptr) ? tape.v_all_ptr : (tape.v_all.empty() ? tape.v_proj.data() : tape.v_all.data());
    const float* attn_probs_ptr =
        (tape.attn_probs.size() == seq_len * config.num_heads * total_kv_len) ? tape.attn_probs.data() : nullptr;
    std::vector<float> grad_k_all(total_kv_len * kv_proj_dim, 0.0f);
    std::vector<float> grad_v_all(total_kv_len * kv_proj_dim, 0.0f);
    kernel::gqa_attention_prefill_backward(tape.q_rope.data(), k_all, v_all, nullptr, attn_probs_ptr,
                                           grad_context.data(), grad_q.data(), grad_k_all.data(), grad_v_all.data(),
                                           attention_params);

    std::vector<float> grad_k_rope_current;
    grad_k_rope_current.assign(seq_len * kv_proj_dim, 0.0f);
    grad_v_proj.assign(seq_len * kv_proj_dim, 0.0f);

    for (size_t si = 0; si < seq_len; ++si) {
        const size_t ti = past_len + si;
        for (size_t h = 0; h < config.num_kv_heads; ++h) {
            for (size_t d = 0; d < config.head_dim; ++d) {
                grad_k_rope_current[kernel::idx3(si, h, d, config.num_kv_heads, config.head_dim)] +=
                    grad_k_all[(h * total_kv_len + ti) * config.head_dim + d];
            }
        }
    }

    for (size_t ti = past_len; ti < total_kv_len; ++ti) {
        const size_t si = ti - past_len;
        if (si >= seq_len) {
            continue;
        }
        for (size_t h = 0; h < config.num_kv_heads; ++h) {
            for (size_t d = 0; d < config.head_dim; ++d) {
                grad_v_proj[kernel::idx3(si, h, d, config.num_kv_heads, config.head_dim)] +=
                    grad_v_all[(h * total_kv_len + ti) * config.head_dim + d];
            }
        }
    }

    const kernel::RopeParams rope_k_params{seq_len, config.num_kv_heads, config.head_dim};
    grad_k_pre_rope.assign(seq_len * kv_proj_dim, 0.0f);
    kernel::apply_rope_backward(grad_k_rope_current.data(), grad_k_pre_rope.data(), tape.positions.data(), rope_k,
                                rope_k_params);
}

void backward_decoder_qkv_norm(const std::vector<float>& grad_post_attn, const std::vector<float>& grad_q,
                               const std::vector<float>& grad_k_pre_rope, const std::vector<float>& grad_v_proj,
                               const BlockForwardTape& tape, const ModelConfig& config,
                               const DecoderLayerWeights& weights, DecoderLayerWeights& layer_grad,
                               kernel::RopeCache& rope_q, size_t seq_len, size_t q_proj_dim, size_t kv_proj_dim,
                               std::vector<float>& grad_layer_in) {
    const kernel::RopeParams rope_q_params{seq_len, config.num_heads, config.head_dim};
    std::vector<float> grad_q_pre_rope(seq_len * q_proj_dim, 0.0f);
    kernel::apply_rope_backward(grad_q.data(), grad_q_pre_rope.data(), tape.positions.data(), rope_q, rope_q_params);

    const kernel::LinearParams q_proj_params{seq_len, config.d_model, q_proj_dim};
    const kernel::LinearParams k_proj_params{seq_len, config.d_model, kv_proj_dim};
    const kernel::LinearParams v_proj_params{seq_len, config.d_model, kv_proj_dim};
    std::vector<float> grad_norm1(seq_len * config.d_model, 0.0f);
    kernel::linear_backward(tape.norm1_out.data(), weights.attention.q_proj.weight.data(),
                            !weights.attention.q_proj.bias.empty(), grad_q_pre_rope.data(), grad_norm1.data(),
                            layer_grad.attention.q_proj.weight.data(),
                            layer_grad.attention.q_proj.bias.empty() ? nullptr : layer_grad.attention.q_proj.bias.data(),
                            q_proj_params);
    kernel::linear_backward(tape.norm1_out.data(), weights.attention.k_proj.weight.data(),
                            !weights.attention.k_proj.bias.empty(), grad_k_pre_rope.data(), grad_norm1.data(),
                            layer_grad.attention.k_proj.weight.data(),
                            layer_grad.attention.k_proj.bias.empty() ? nullptr : layer_grad.attention.k_proj.bias.data(),
                            k_proj_params);
    kernel::linear_backward(tape.norm1_out.data(), weights.attention.v_proj.weight.data(),
                            !weights.attention.v_proj.bias.empty(), grad_v_proj.data(), grad_norm1.data(),
                            layer_grad.attention.v_proj.weight.data(),
                            layer_grad.attention.v_proj.bias.empty() ? nullptr : layer_grad.attention.v_proj.bias.data(),
                            v_proj_params);

    const kernel::RmsNormParams norm_params{seq_len, config.d_model};
    std::vector<float> grad_from_norm1(seq_len * config.d_model, 0.0f);
    kernel::rms_norm_backward(tape.x_in.data(), weights.norm1.weight.data(), weights.norm1.eps, grad_norm1.data(),
                              grad_from_norm1.data(), layer_grad.norm1.weight.data(), norm_params);

    grad_layer_in.assign(seq_len * config.d_model, 0.0f);
    for (size_t i = 0; i < grad_layer_in.size(); ++i) {
        grad_layer_in[i] = grad_post_attn[i] + grad_from_norm1[i];
    }
}

void backward_model(const ModelConfig& config, const ModelWeights& weights,
                    const std::vector<BlockForwardTape>& tapes, const std::vector<float>& grad_hidden_out,
                    ModelWeights& grad_weights, std::vector<float>& grad_hidden_in) {
    if (tapes.empty()) {
        grad_hidden_in = grad_hidden_out;
        return;
    }

    kernel::RopeCache rope_q(config.rope_base, config.head_dim, config.rope_dim);
    kernel::RopeCache rope_k(config.rope_base, config.head_dim, config.rope_dim);
    size_t rope_max_pos = 0;
    for (const auto& tape : tapes) {
        for (size_t p : tape.positions) {
            rope_max_pos = std::max(rope_max_pos, p);
        }
    }
    rope_q.ensure(rope_max_pos + 1);
    rope_k.ensure(rope_max_pos + 1);

    std::vector<float> grad_chain = grad_hidden_out;
    std::vector<float> grad_into_prev_layer;
    for (int layer = static_cast<int>(weights.layers.size()) - 1; layer >= 0; --layer) {
        const size_t layer_idx = static_cast<size_t>(layer);
        const BlockForwardTape& tape = tapes[layer_idx];
        const size_t kv_proj_dim = config.num_kv_heads * config.head_dim;
        const size_t q_proj_dim = config.num_heads * config.head_dim;
        const size_t kv_elems = (tape.k_all_ptr != nullptr)
                                    ? tape.k_all_size
                                    : (tape.k_all.empty() ? tape.k_rope.size() : tape.k_all.size());
        const size_t total_kv_len = kv_elems / kv_proj_dim;
        const size_t past_len = tape.past_len;
        const size_t seq_len = total_kv_len - past_len;

        std::vector<float> grad_post_attn;
        std::vector<float> grad_q;
        std::vector<float> grad_k_pre_rope;
        std::vector<float> grad_v_proj;
        backward_decoder_mlp(grad_chain, tape, config, weights.layers[layer_idx],
                             grad_weights.layers[layer_idx], seq_len, grad_post_attn);
        backward_decoder_attention(grad_post_attn, tape, config, weights.layers[layer_idx],
                                   grad_weights.layers[layer_idx], rope_k, seq_len, q_proj_dim, kv_proj_dim,
                                   total_kv_len, past_len, grad_q, grad_k_pre_rope, grad_v_proj);
        backward_decoder_qkv_norm(grad_post_attn, grad_q, grad_k_pre_rope, grad_v_proj, tape, config,
                                  weights.layers[layer_idx], grad_weights.layers[layer_idx], rope_q, seq_len,
                                  q_proj_dim, kv_proj_dim, grad_into_prev_layer);
        grad_chain.swap(grad_into_prev_layer);
    }
    grad_hidden_in = std::move(grad_chain);
}

} // namespace model
