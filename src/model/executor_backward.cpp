#include "model/executor.h"

#include "kernel/kernel.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

namespace model {
namespace {

// One instance per `backward_model` call: buffers are sized for the worst layer in this stack, then reused each
// reverse layer step (not per-layer allocation).
struct BackwardWorkspace {
    // Attention path
    std::vector<float> grad_post_attn;
    std::vector<float> grad_q;
    std::vector<float> grad_k_pre_rope;
    std::vector<float> grad_v_proj;
    std::vector<float> grad_context;
    std::vector<float> grad_k_rope_current;
    std::vector<float> grad_k_all;
    std::vector<float> grad_v_all;
    // MLP path (per layer, non-overlapping with attention in pipeline)
    std::vector<float> grad_ff_mid;
    std::vector<float> grad_ff_gate;
    std::vector<float> grad_ff_up;
    std::vector<float> grad_norm2_buf;
    std::vector<float> grad_post_attn_from_mlp_buf;
    // QKV + norm1 path (after attention; grad_context free for reuse as Q pre-RoPE delta)
    std::vector<float> grad_norm1_buf;
    std::vector<float> grad_from_norm1_buf;
    // Ping-pong gradient chain (replaces grad_chain / grad_into_prev locals)
    std::vector<float> grad_chain_a;
    std::vector<float> grad_chain_b;

    void ensure(const ModelConfig& config, const std::vector<BlockForwardTape>& tapes, size_t grad_out_elems) {
        const size_t kv_pd = config.num_kv_heads * config.head_dim;
        const size_t q_pd = config.num_heads * config.head_dim;
        const size_t dm = config.d_model;
        const size_t dff = config.d_ff;
        size_t max_seq = 0;
        size_t max_total_kv = 0;
        for (const auto& tape : tapes) {
            const size_t kv_elems = (tape.k_all_ptr != nullptr)
                                        ? tape.k_all_size
                                        : (tape.k_all.empty() ? tape.k_rope.size() : tape.k_all.size());
            const size_t total_kv_len = kv_elems / kv_pd;
            const size_t seq_len = total_kv_len - tape.past_len;
            max_seq = std::max(max_seq, seq_len);
            max_total_kv = std::max(max_total_kv, total_kv_len);
        }
        if (grad_out_elems > 0) {
            max_seq = std::max(max_seq, grad_out_elems / dm);
        }
        const size_t row_dm = max_seq * dm;
        const size_t row_q = max_seq * q_pd;
        const size_t row_kv = max_seq * kv_pd;
        const size_t row_ff = max_seq * dff;
        grad_post_attn.resize(row_dm);
        grad_q.resize(row_q);
        grad_k_pre_rope.resize(row_kv);
        grad_v_proj.resize(row_kv);
        grad_context.resize(row_q);
        grad_k_rope_current.resize(row_kv);
        grad_k_all.resize(max_total_kv * kv_pd);
        grad_v_all.resize(max_total_kv * kv_pd);
        grad_ff_mid.resize(row_ff);
        grad_ff_gate.resize(row_ff);
        grad_ff_up.resize(row_ff);
        grad_norm2_buf.resize(row_dm);
        grad_post_attn_from_mlp_buf.resize(row_dm);
        grad_norm1_buf.resize(row_dm);
        grad_from_norm1_buf.resize(row_dm);
        grad_chain_a.resize(row_dm);
        grad_chain_b.resize(row_dm);
    }
};

void backward_decoder_mlp(const std::vector<float>& grad_layer_out, const BlockForwardTape& tape,
                            const ModelConfig& config, const DecoderLayerWeights& weights,
                            DecoderLayerWeights& layer_grad, size_t seq_len, BackwardWorkspace& ws) {
    const size_t row_ff = seq_len * config.d_ff;
    const size_t row_dm = seq_len * config.d_model;
    std::fill(ws.grad_ff_mid.begin(), ws.grad_ff_mid.begin() + row_ff, 0.0f);
    const kernel::LinearParams down_params{seq_len, config.d_ff, config.d_model};
    kernel::linear_backward(tape.hidden_mid.data(), weights.mlp.down.weight.data(), !weights.mlp.down.bias.empty(),
                            grad_layer_out.data(), ws.grad_ff_mid.data(), layer_grad.mlp.down.weight.data(),
                            layer_grad.mlp.down.bias.empty() ? nullptr : layer_grad.mlp.down.bias.data(), down_params);

    const kernel::SiluMulParams silu_mul_params{row_ff};
    kernel::silu_mul_backward(tape.gate.data(), tape.up.data(), ws.grad_ff_mid.data(), ws.grad_ff_gate.data(),
                              ws.grad_ff_up.data(), silu_mul_params);

    const kernel::LinearParams gate_params{seq_len, config.d_model, config.d_ff};
    const kernel::LinearParams up_params{seq_len, config.d_model, config.d_ff};
    std::fill(ws.grad_norm2_buf.begin(), ws.grad_norm2_buf.begin() + row_dm, 0.0f);
    kernel::linear_backward(tape.norm2_out.data(), weights.mlp.gate.weight.data(), !weights.mlp.gate.bias.empty(),
                            ws.grad_ff_gate.data(), ws.grad_norm2_buf.data(), layer_grad.mlp.gate.weight.data(),
                            layer_grad.mlp.gate.bias.empty() ? nullptr : layer_grad.mlp.gate.bias.data(), gate_params);
    kernel::linear_backward(tape.norm2_out.data(), weights.mlp.up.weight.data(), !weights.mlp.up.bias.empty(),
                            ws.grad_ff_up.data(), ws.grad_norm2_buf.data(), layer_grad.mlp.up.weight.data(),
                            layer_grad.mlp.up.bias.empty() ? nullptr : layer_grad.mlp.up.bias.data(), up_params);

    const kernel::RmsNormParams norm_params{seq_len, config.d_model};
    kernel::rms_norm_backward(tape.hidden_after_attn.data(), weights.norm2.weight.data(), weights.norm2.eps,
                              ws.grad_norm2_buf.data(), ws.grad_post_attn_from_mlp_buf.data(),
                              layer_grad.norm2.weight.data(), norm_params);

    const kernel::AddParams add_post{row_dm};
    kernel::add(grad_layer_out.data(), ws.grad_post_attn_from_mlp_buf.data(), ws.grad_post_attn.data(), add_post);
}

void backward_decoder_attention(const std::vector<float>& grad_post_attn, const BlockForwardTape& tape,
                                const ModelConfig& config, const DecoderLayerWeights& weights,
                                DecoderLayerWeights& layer_grad, RopeCache& rope_k, size_t seq_len,
                                size_t q_proj_dim, size_t kv_proj_dim, size_t total_kv_len, size_t past_len,
                                BackwardWorkspace& ws) {
    const size_t row_q = seq_len * q_proj_dim;
    std::fill(ws.grad_context.begin(), ws.grad_context.begin() + row_q, 0.0f);
    const kernel::LinearParams o_proj_params{seq_len, q_proj_dim, config.d_model};
    kernel::linear_backward(tape.attn_ctx.data(), weights.attention.o_proj.weight.data(),
                            !weights.attention.o_proj.bias.empty(), grad_post_attn.data(), ws.grad_context.data(),
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

    const float* k_all = (tape.k_all_ptr != nullptr) ? tape.k_all_ptr : (tape.k_all.empty() ? tape.k_rope.data() : tape.k_all.data());
    const float* v_all = (tape.v_all_ptr != nullptr) ? tape.v_all_ptr : (tape.v_all.empty() ? tape.v_proj.data() : tape.v_all.data());
    const float* attn_probs_ptr =
        (tape.attn_probs.size() == seq_len * config.num_heads * total_kv_len) ? tape.attn_probs.data() : nullptr;
    // gqa_attention_backward zeros grad_q / grad_k_all / grad_v_all internally — do not pre-fill (wastes bandwidth).
    kernel::gqa_attention_backward(tape.q_rope.data(), k_all, v_all, nullptr, attn_probs_ptr, ws.grad_context.data(),
                                   ws.grad_q.data(), ws.grad_k_all.data(), ws.grad_v_all.data(), attention_params);

    const size_t nkv = seq_len * kv_proj_dim;
    std::fill(ws.grad_k_rope_current.begin(), ws.grad_k_rope_current.begin() + nkv, 0.0f);
    std::fill(ws.grad_v_proj.begin(), ws.grad_v_proj.begin() + nkv, 0.0f);

    for (size_t si = 0; si < seq_len; ++si) {
        const size_t ti = past_len + si;
        for (size_t h = 0; h < config.num_kv_heads; ++h) {
            for (size_t d = 0; d < config.head_dim; ++d) {
                ws.grad_k_rope_current[kernel::idx3(si, h, d, config.num_kv_heads, config.head_dim)] +=
                    ws.grad_k_all[(h * total_kv_len + ti) * config.head_dim + d];
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
                ws.grad_v_proj[kernel::idx3(si, h, d, config.num_kv_heads, config.head_dim)] +=
                    ws.grad_v_all[(h * total_kv_len + ti) * config.head_dim + d];
            }
        }
    }

    const kernel::RopeParams rope_k_params{seq_len, config.num_kv_heads, config.head_dim};
    // apply_rope_backward clears grad_k_pre_rope before writing rotary dims.
    kernel::apply_rope_backward(ws.grad_k_rope_current.data(), ws.grad_k_pre_rope.data(), tape.positions.data(),
                                rope_k.cos_data(), rope_k.sin_data(), rope_k.rot_dim(), rope_k_params);
}

void backward_decoder_qkv_norm(const std::vector<float>& grad_post_attn, const BlockForwardTape& tape,
                               const ModelConfig& config, const DecoderLayerWeights& weights,
                               DecoderLayerWeights& layer_grad, RopeCache& rope_q, size_t seq_len,
                               size_t q_proj_dim, size_t kv_proj_dim, BackwardWorkspace& ws,
                               std::vector<float>& grad_layer_in) {
    const kernel::RopeParams rope_q_params{seq_len, config.num_heads, config.head_dim};
    // grad_context is dead after attention backward; reuse for d(loss)/d(q_pre_rope).
    float* grad_q_pre_rope = ws.grad_context.data();
    kernel::apply_rope_backward(ws.grad_q.data(), grad_q_pre_rope, tape.positions.data(), rope_q.cos_data(),
                                rope_q.sin_data(), rope_q.rot_dim(), rope_q_params);

    const kernel::LinearParams q_proj_params{seq_len, config.d_model, q_proj_dim};
    const kernel::LinearParams k_proj_params{seq_len, config.d_model, kv_proj_dim};
    const kernel::LinearParams v_proj_params{seq_len, config.d_model, kv_proj_dim};
    const size_t row_dm = seq_len * config.d_model;
    std::fill(ws.grad_norm1_buf.begin(), ws.grad_norm1_buf.begin() + row_dm, 0.0f);
    kernel::linear_backward(tape.norm1_out.data(), weights.attention.q_proj.weight.data(),
                            !weights.attention.q_proj.bias.empty(), grad_q_pre_rope, ws.grad_norm1_buf.data(),
                            layer_grad.attention.q_proj.weight.data(),
                            layer_grad.attention.q_proj.bias.empty() ? nullptr : layer_grad.attention.q_proj.bias.data(),
                            q_proj_params);
    kernel::linear_backward(tape.norm1_out.data(), weights.attention.k_proj.weight.data(),
                            !weights.attention.k_proj.bias.empty(), ws.grad_k_pre_rope.data(), ws.grad_norm1_buf.data(),
                            layer_grad.attention.k_proj.weight.data(),
                            layer_grad.attention.k_proj.bias.empty() ? nullptr : layer_grad.attention.k_proj.bias.data(),
                            k_proj_params);
    kernel::linear_backward(tape.norm1_out.data(), weights.attention.v_proj.weight.data(),
                            !weights.attention.v_proj.bias.empty(), ws.grad_v_proj.data(), ws.grad_norm1_buf.data(),
                            layer_grad.attention.v_proj.weight.data(),
                            layer_grad.attention.v_proj.bias.empty() ? nullptr : layer_grad.attention.v_proj.bias.data(),
                            v_proj_params);

    const kernel::RmsNormParams norm_params{seq_len, config.d_model};
    kernel::rms_norm_backward(tape.x_in.data(), weights.norm1.weight.data(), weights.norm1.eps, ws.grad_norm1_buf.data(),
                              ws.grad_from_norm1_buf.data(), layer_grad.norm1.weight.data(), norm_params);

    const kernel::AddParams add_in{seq_len * config.d_model};
    grad_layer_in.resize(seq_len * config.d_model);
    kernel::add(grad_post_attn.data(), ws.grad_from_norm1_buf.data(), grad_layer_in.data(), add_in);
}

} // namespace

void backward_model(const ModelConfig& config, const ModelWeights& weights,
                    const std::vector<BlockForwardTape>& tapes, const std::vector<float>& grad_hidden_out,
                    ModelWeights& grad_weights, std::vector<float>& grad_hidden_in) {
    if (tapes.empty()) {
        grad_hidden_in = grad_hidden_out;
        return;
    }

    RopeCache rope_q(config.rope_base, config.head_dim, config.rope_dim);
    RopeCache rope_k(config.rope_base, config.head_dim, config.rope_dim);
    size_t rope_max_pos = 0;
    for (const auto& tape : tapes) {
        for (size_t p : tape.positions) {
            rope_max_pos = std::max(rope_max_pos, p);
        }
    }
    rope_q.ensure(rope_max_pos + 1);
    rope_k.ensure(rope_max_pos + 1);

    BackwardWorkspace ws;
    ws.ensure(config, tapes, grad_hidden_out.size());

    std::vector<float>* grad_cur = &ws.grad_chain_a;
    std::vector<float>* grad_next = &ws.grad_chain_b;
    std::memcpy(grad_cur->data(), grad_hidden_out.data(), grad_hidden_out.size() * sizeof(float));
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

        backward_decoder_mlp(*grad_cur, tape, config, weights.layers[layer_idx], grad_weights.layers[layer_idx],
                             seq_len, ws);
        backward_decoder_attention(ws.grad_post_attn, tape, config, weights.layers[layer_idx],
                                   grad_weights.layers[layer_idx], rope_k, seq_len, q_proj_dim, kv_proj_dim,
                                   total_kv_len, past_len, ws);
        backward_decoder_qkv_norm(ws.grad_post_attn, tape, config, weights.layers[layer_idx],
                                  grad_weights.layers[layer_idx], rope_q, seq_len, q_proj_dim, kv_proj_dim, ws,
                                  *grad_next);
        std::swap(grad_cur, grad_next);
    }
    grad_hidden_in = std::move(*grad_cur);
}

} // namespace model
