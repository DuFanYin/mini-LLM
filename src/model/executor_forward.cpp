#include "model/executor.h"

#include "engine/kv_cache.h"
#include "engine/validation.h"

#include <cstddef>
#include <utility>
#include <vector>

namespace model {

// Attention submodule: pre-norm -> QKV -> RoPE -> KV -> GQA -> o_proj -> residual with input. Writes post-attention
// hidden (before second norm / MLP) to `hidden_after_attn_out`. On the inference path (tape == nullptr) intermediate
// tensors are held in function-local vectors; on the training path they are stored in `tape` for backward.
void forward_attention_block(const ModelConfig& config, const DecoderLayerWeights& weights, const ForwardInput& input,
                             const CacheView& cache_view, const CacheBridge& cache, kernel::RopeCache& rope_q,
                             kernel::RopeCache& rope_k, float* hidden_after_attn_out, BlockForwardTape* tape) {
    const size_t seq_len = input.seq_len;
    const size_t q_proj_dim = config.num_heads * config.head_dim;
    const size_t kv_proj_dim = config.num_kv_heads * config.head_dim;
    const size_t past_len = (input.past_len == static_cast<size_t>(-1)) ? cache_view.past_len : input.past_len;
    engine::validate_past_len_cache(input.use_cache, past_len, cache_view.past_len);

    std::vector<size_t> positions_local;
    const size_t* positions_ptr = nullptr;
    if (input.positions.empty()) {
        positions_local.resize(seq_len);
        for (size_t i = 0; i < seq_len; ++i) {
            positions_local[i] = past_len + i;
        }
        positions_ptr = positions_local.data();
    } else {
        engine::validate_positions_size(seq_len, input.positions);
        positions_ptr = input.positions.data();
    }

    const kernel::RmsNormParams norm_params{seq_len, config.d_model};
    std::vector<float> norm1_out_local;
    std::vector<float>& norm1_out = (tape != nullptr) ? tape->norm1_out : norm1_out_local;
    norm1_out.resize(seq_len * config.d_model);
    float* norm1_out_data = norm1_out.data();
    const float* x_in = (input.hidden_states_ptr != nullptr) ? input.hidden_states_ptr : input.hidden_states.data();

    // --- Pre-norm, Q/K/V linear ---
    kernel::rms_norm(x_in, weights.norm1.weight.data(), weights.norm1.eps, norm1_out_data, norm_params);

    const kernel::LinearParams q_linear_params{seq_len, config.d_model, q_proj_dim};
    const kernel::LinearParams kv_linear_params{seq_len, config.d_model, kv_proj_dim};
    std::vector<float> q_pre_local;
    std::vector<float> k_pre_local;
    std::vector<float> v_proj_local;
    std::vector<float>& q_pre = (tape != nullptr) ? tape->q_pre_rope : q_pre_local;
    std::vector<float>& k_pre = (tape != nullptr) ? tape->k_pre_rope : k_pre_local;
    std::vector<float>& v_proj = (tape != nullptr) ? tape->v_proj : v_proj_local;
    q_pre.resize(seq_len * q_proj_dim);
    k_pre.resize(seq_len * kv_proj_dim);
    v_proj.resize(seq_len * kv_proj_dim);
    float* q_pre_data = q_pre.data();
    float* k_pre_data = k_pre.data();
    float* v_proj_data = v_proj.data();
    kernel::linear(norm1_out_data, weights.attention.q_proj.weight.data(),
                   weights.attention.q_proj.bias.empty() ? nullptr : weights.attention.q_proj.bias.data(),
                   q_pre_data, q_linear_params);
    kernel::linear(norm1_out_data, weights.attention.k_proj.weight.data(),
                   weights.attention.k_proj.bias.empty() ? nullptr : weights.attention.k_proj.bias.data(),
                   k_pre_data, kv_linear_params);
    kernel::linear(norm1_out_data, weights.attention.v_proj.weight.data(),
                   weights.attention.v_proj.bias.empty() ? nullptr : weights.attention.v_proj.bias.data(),
                   v_proj_data, kv_linear_params);

    if (tape != nullptr) {
        tape->positions.assign(positions_ptr, positions_ptr + seq_len);
        tape->past_len = past_len;
        tape->x_in.assign(x_in, x_in + seq_len * config.d_model);
        tape->q_rope = tape->q_pre_rope;
        tape->k_rope = tape->k_pre_rope;
    }

    // --- RoPE on current Q/K ---
    const kernel::RopeParams rope_q_params{seq_len, config.num_heads, config.head_dim};
    const kernel::RopeParams rope_k_params{seq_len, config.num_kv_heads, config.head_dim};
    std::vector<float>& q_rope = (tape != nullptr) ? tape->q_rope : q_pre;
    std::vector<float>& k_rope = (tape != nullptr) ? tape->k_rope : k_pre;
    kernel::apply_rope(q_rope.data(), positions_ptr, rope_q, rope_q_params);
    kernel::apply_rope(k_rope.data(), positions_ptr, rope_k, rope_k_params);

    // --- Full K/V span (cached prefix + current step) for attention ---
    const float* k_all_ptr = nullptr;
    const float* v_all_ptr = nullptr;
    size_t total_kv_len = seq_len;
    if (input.use_cache) {
        engine::append_cache(cache, input.layer_id, k_rope.data(), v_proj.data(), seq_len);
        const CacheView updated_cache_view = engine::build_cache_view(cache, input.layer_id);
        total_kv_len = updated_cache_view.total_kv_len;
        k_all_ptr = updated_cache_view.k_cache;
        v_all_ptr = updated_cache_view.v_cache;
    } else {
        k_all_ptr = k_rope.data();
        v_all_ptr = v_proj.data();
    }
    engine::validate_attention_mask_size(seq_len, total_kv_len, input.attention_mask);

    if (tape != nullptr) {
        if (input.use_cache) {
            const size_t cache_elems = total_kv_len * config.num_kv_heads * config.head_dim;
            // The cache owns these buffers. Backward may read through the pointers as long as the
            // cache is not reset or overwritten between this forward pass and its backward pass.
            tape->k_all.clear();
            tape->v_all.clear();
            tape->k_all_ptr = k_all_ptr;
            tape->v_all_ptr = v_all_ptr;
            tape->k_all_size = cache_elems;
            tape->v_all_size = cache_elems;
        } else {
            tape->k_all.clear();
            tape->v_all.clear();
            tape->k_all_ptr = nullptr;
            tape->v_all_ptr = nullptr;
            tape->k_all_size = 0;
            tape->v_all_size = 0;
        }
    }

    kernel::AttentionParams attention_params;
    attention_params.seq_len = seq_len;
    attention_params.num_heads = config.num_heads;
    attention_params.num_kv_heads = config.num_kv_heads;
    attention_params.head_dim = config.head_dim;
    attention_params.past_len = past_len;
    attention_params.total_kv_len = total_kv_len;
    attention_params.causal = input.causal;
    attention_params.use_cache = input.use_cache;

    // --- GQA: decode vs prefill path (prefill + tape stores softmax for backward) ---
    const bool decode_path = input.is_decode || (input.use_cache && seq_len == 1 && !input.is_prefill);
    std::vector<float> attention_context_local;
    std::vector<float>& attention_context = (tape != nullptr) ? tape->attn_ctx : attention_context_local;
    attention_context.resize(seq_len * q_proj_dim);
    float* attention_context_data = attention_context.data();
    if (decode_path) {
        kernel::gqa_attention_decode(q_rope.data(), k_all_ptr, v_all_ptr,
                                     input.attention_mask.empty() ? nullptr : input.attention_mask.data(),
                                     attention_context_data, attention_params);
    } else if (tape != nullptr) {
        tape->attn_probs.assign(seq_len * config.num_heads * total_kv_len, 0.0f);
        kernel::gqa_attention_prefill(q_rope.data(), k_all_ptr, v_all_ptr,
                                      input.attention_mask.empty() ? nullptr : input.attention_mask.data(),
                                      attention_context_data, tape->attn_probs.data(), attention_params);
    } else {
        kernel::gqa_attention_prefill(q_rope.data(), k_all_ptr, v_all_ptr,
                                      input.attention_mask.empty() ? nullptr : input.attention_mask.data(),
                                      attention_context_data, nullptr, attention_params);
    }

    const kernel::LinearParams o_linear_params{seq_len, q_proj_dim, config.d_model};
    std::vector<float> attention_output_local;
    std::vector<float>& attention_output = (tape != nullptr) ? tape->attn_proj_out : attention_output_local;
    attention_output.resize(seq_len * config.d_model);
    float* attention_output_data = attention_output.data();
    kernel::linear(attention_context_data, weights.attention.o_proj.weight.data(),
                   weights.attention.o_proj.bias.empty() ? nullptr : weights.attention.o_proj.bias.data(),
                   attention_output_data, o_linear_params);

    const kernel::AddParams add_params{seq_len * config.d_model};
    kernel::add(x_in, attention_output_data, hidden_after_attn_out, add_params);
}

// Full decoder layer: attention block (above) then post-norm + SiLU MLP + residual.
void forward_decoder_layer(const ModelConfig& config, const DecoderLayerWeights& weights, const ForwardInput& input,
                           const CacheView& cache_view, const CacheBridge& cache, kernel::RopeCache& rope_q,
                           kernel::RopeCache& rope_k, float* hidden_out, BlockForwardTape* tape) {
    const size_t seq_len = input.seq_len;
    const kernel::RmsNormParams norm_params{seq_len, config.d_model};
    std::vector<float> hidden_after_attn_local;
    std::vector<float>& hidden_after_attn = (tape != nullptr) ? tape->hidden_after_attn : hidden_after_attn_local;
    hidden_after_attn.resize(seq_len * config.d_model);
    float* hidden_after_attn_data = hidden_after_attn.data();
    forward_attention_block(config, weights, input, cache_view, cache, rope_q, rope_k, hidden_after_attn_data, tape);

    std::vector<float> normalized_post_attention_local;
    std::vector<float>& normalized_post_attention =
        (tape != nullptr) ? tape->norm2_out : normalized_post_attention_local;
    normalized_post_attention.resize(seq_len * config.d_model);
    float* normalized_post_attention_data = normalized_post_attention.data();
    kernel::rms_norm(hidden_after_attn_data, weights.norm2.weight.data(), weights.norm2.eps,
                     normalized_post_attention_data, norm_params);

    std::vector<float> mlp_output_local;
    std::vector<float>& mlp_output = (tape != nullptr) ? tape->mlp_out : mlp_output_local;
    mlp_output.resize(seq_len * config.d_model);
    float* mlp_output_data = mlp_output.data();
    const kernel::LinearParams gate_params{seq_len, config.d_model, config.d_ff};
    const kernel::LinearParams up_params{seq_len, config.d_model, config.d_ff};
    const kernel::LinearParams down_params{seq_len, config.d_ff, config.d_model};
    const kernel::SiluMulParams silu_mul_params{seq_len * config.d_ff};

    std::vector<float> gate_local;
    std::vector<float> up_local;
    std::vector<float> hidden_mid_local;
    std::vector<float>& gate = (tape != nullptr) ? tape->gate : gate_local;
    std::vector<float>& up = (tape != nullptr) ? tape->up : up_local;
    std::vector<float>& hidden_mid = (tape != nullptr) ? tape->hidden_mid : hidden_mid_local;
    gate.resize(seq_len * config.d_ff);
    up.resize(seq_len * config.d_ff);
    hidden_mid.resize(seq_len * config.d_ff);
    float* gate_data = gate.data();
    float* up_data = up.data();
    float* hidden_mid_data = hidden_mid.data();

    kernel::linear(normalized_post_attention_data, weights.mlp.gate.weight.data(),
                   weights.mlp.gate.bias.empty() ? nullptr : weights.mlp.gate.bias.data(), gate_data, gate_params);
    kernel::linear(normalized_post_attention_data, weights.mlp.up.weight.data(),
                   weights.mlp.up.bias.empty() ? nullptr : weights.mlp.up.bias.data(), up_data, up_params);

    kernel::silu_mul(gate_data, up_data, hidden_mid_data, silu_mul_params);
    kernel::linear(hidden_mid_data, weights.mlp.down.weight.data(),
                   weights.mlp.down.bias.empty() ? nullptr : weights.mlp.down.bias.data(), mlp_output_data,
                   down_params);

    const kernel::AddParams add_params{seq_len * config.d_model};
    kernel::add(hidden_after_attn_data, mlp_output_data, hidden_out, add_params);
}

void forward_model(const ModelConfig& config, const std::vector<DecoderLayerState>& layers, const ForwardInput& input,
                   CacheBridge& cache, float* hidden_out, std::vector<BlockForwardTape>* layer_tapes) {
    std::vector<float> layer_hidden_a(input.seq_len * config.d_model, 0.0f);
    std::vector<float> layer_hidden_b(input.seq_len * config.d_model, 0.0f);
    const float* current_hidden =
        (input.hidden_states_ptr != nullptr) ? input.hidden_states_ptr : input.hidden_states.data();
    bool use_a = true;
    for (size_t layer_id = 0; layer_id < layers.size(); ++layer_id) {
        const bool is_last_layer = (layer_id + 1u == layers.size());

        ForwardInput layer_in;
        layer_in.seq_len = input.seq_len;
        layer_in.layer_id = layer_id;
        layer_in.past_len = input.past_len;
        layer_in.positions = input.positions;
        layer_in.attention_mask = input.attention_mask;
        layer_in.use_cache = input.use_cache;
        layer_in.causal = input.causal;
        layer_in.is_prefill = input.is_prefill;
        layer_in.is_decode = input.is_decode;
        layer_in.hidden_states_ptr = current_hidden;

        CacheView cache_view;
        if (layer_in.use_cache) {
            cache_view = engine::build_cache_view(cache, layer_in.layer_id);
        } else {
            cache_view.layer_id = layer_in.layer_id;
        }

        float* layer_out = hidden_out;
        if (!is_last_layer) {
            layer_out = use_a ? layer_hidden_a.data() : layer_hidden_b.data();
        }
        BlockForwardTape* tape = (layer_tapes != nullptr) ? &(*layer_tapes)[layer_id] : nullptr;
        forward_decoder_layer(config, *layers[layer_id].weights, layer_in, cache_view, cache, layers[layer_id].rope_q,
                              layers[layer_id].rope_k, layer_out, tape);

        if (!is_last_layer) {
            current_hidden = layer_out;
            use_a = !use_a;
        }
    }
}

} // namespace model
