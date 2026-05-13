#pragma once

// Shared structs for config, weights, I/O, and autograd tape — no execution logic.
// Decoder layer weight layout (on-disk save order): norm1 | Q,K,V,o_proj | norm2 | MLP gate,up,down.
// KV-cache types (KVCache, CacheView, KVCacheConfig) live in model/kv_cache.h.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace model {

struct ModelConfig {
    size_t d_model = 0;
    size_t num_heads = 0;
    size_t num_kv_heads = 0;
    size_t head_dim = 0;
    size_t d_ff = 0;
    float rope_base = 10000.0f;
    size_t rope_dim = 0;
    float rms_norm_eps = 1e-5f;
    size_t num_layers = 0;
};

struct RMSNormWeights {
    std::vector<float> weight;
    float eps = 1e-5f;
};

struct LinearWeights {
    size_t in_dim = 0;
    size_t out_dim = 0;
    std::vector<float> weight;
    std::vector<float> bias;
};

// Q/K/V projections + attention output projection (no separate "attention tensor"; RoPE uses ModelConfig).
struct AttentionLinearWeights {
    LinearWeights q_proj;
    LinearWeights k_proj;
    LinearWeights v_proj;
    LinearWeights o_proj;
};

// SwiGLU-style MLP: gate/up SiLU* then down.
struct MlpLinearWeights {
    LinearWeights gate;
    LinearWeights up;
    LinearWeights down;
};

// One transformer decoder block: pre-norm + (attn linears) + pre-MLP norm + (FFN linears).
struct DecoderLayerWeights {
    RMSNormWeights norm1;
    AttentionLinearWeights attention;
    RMSNormWeights norm2;
    MlpLinearWeights mlp;
};

struct ModelWeights {
    std::vector<DecoderLayerWeights> layers;
    std::vector<float> token_embedding;
    std::vector<float> output_projection;
    size_t vocab_size = 0;
};

struct ForwardInput {
    std::vector<float> hidden_states;
    const float* hidden_states_ptr = nullptr;
    size_t seq_len = 0;
    size_t layer_id = 0;
    size_t past_len = static_cast<size_t>(-1);
    std::vector<size_t> positions;
    std::vector<float> attention_mask;
    bool use_cache = true;
    bool causal = true;
    bool is_prefill = false;
    bool is_decode = false;
};

struct BlockForwardTape {
    std::vector<size_t> positions;
    size_t past_len = 0;
    std::vector<float> x_in;
    std::vector<float> norm1_out;
    std::vector<float> q_pre_rope;
    std::vector<float> k_pre_rope;
    std::vector<float> q_rope;
    std::vector<float> k_rope;
    std::vector<float> v_proj;
    std::vector<float> k_all;
    std::vector<float> v_all;
    const float* k_all_ptr = nullptr;
    const float* v_all_ptr = nullptr;
    size_t k_all_size = 0;
    size_t v_all_size = 0;
    std::vector<float> attn_ctx;
    std::vector<float> attn_probs;
    std::vector<float> attn_proj_out;
    std::vector<float> hidden_after_attn;
    std::vector<float> norm2_out;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> hidden_mid;
    std::vector<float> mlp_out;
};

} // namespace model
