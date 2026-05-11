#include "engine/validation.h"

#include <stdexcept>
#include <string>

namespace engine {

void validate_model_config(const model::ModelConfig& cfg) {
    if (cfg.d_model == 0 || cfg.num_heads == 0 || cfg.num_kv_heads == 0 || cfg.head_dim == 0 || cfg.d_ff == 0) {
        throw std::invalid_argument("ModelConfig: invalid dimension");
    }
    if (cfg.num_heads % cfg.num_kv_heads != 0) {
        throw std::invalid_argument("ModelConfig: num_heads must be divisible by num_kv_heads");
    }
    if (cfg.num_heads * cfg.head_dim != cfg.d_model) {
        throw std::invalid_argument("ModelConfig: d_model must equal num_heads*head_dim");
    }
}

void validate_model_weights(const model::ModelConfig& cfg, const model::ModelWeights& weights) {
    if (weights.layers.empty()) {
        throw std::invalid_argument("Model: at least one layer is required");
    }
    if (cfg.num_layers != 0 && cfg.num_layers != weights.layers.size()) {
        throw std::invalid_argument("ModelConfig: num_layers must match weights.layers.size()");
    }
    if (weights.vocab_size > 0) {
        const size_t expected = weights.vocab_size * cfg.d_model;
        if (weights.token_embedding.size() != expected) {
            throw std::invalid_argument("Model: token_embedding size mismatch");
        }
        if (weights.output_projection.size() != expected) {
            throw std::invalid_argument("Model: output_projection size mismatch");
        }
    }
}

void validate_model_forward(const model::ModelConfig& cfg, size_t num_layers, const model::ForwardInput& input,
                            const KVCache& cache) {
    if (input.hidden_states_ptr == nullptr && input.hidden_states.size() != input.seq_len * cfg.d_model) {
        throw std::invalid_argument("ForwardInput: hidden_states size mismatch");
    }
    if (cache.num_layers() != num_layers) {
        throw std::invalid_argument("Model: cache layer count mismatch");
    }
    if (cache.num_kv_heads() != cfg.num_kv_heads || cache.head_dim() != cfg.head_dim) {
        throw std::invalid_argument("Model: cache shape mismatch with model config");
    }
}

void validate_linear_weights(const model::LinearWeights& linear, size_t in_dim, size_t out_dim, const char* name) {
    if (linear.in_dim != in_dim || linear.out_dim != out_dim) {
        throw std::invalid_argument(std::string("Linear shape mismatch: ") + name);
    }
    if (linear.weight.size() != out_dim * in_dim) {
        throw std::invalid_argument(std::string("Linear weight size mismatch: ") + name);
    }
    if (!linear.bias.empty() && linear.bias.size() != out_dim) {
        throw std::invalid_argument(std::string("Linear bias size mismatch: ") + name);
    }
}

void validate_decoder_layer_weights(const model::ModelConfig& cfg, const model::DecoderLayerWeights* weights) {
    if (weights == nullptr) {
        throw std::invalid_argument("DecoderLayer: layer weights pointer is null");
    }
    if (weights->norm1.weight.size() != cfg.d_model || weights->norm2.weight.size() != cfg.d_model) {
        throw std::invalid_argument("RMSNorm weight size mismatch");
    }
    validate_linear_weights(weights->attention.q_proj, cfg.d_model, cfg.num_heads * cfg.head_dim, "attention.q_proj");
    validate_linear_weights(weights->attention.k_proj, cfg.d_model, cfg.num_kv_heads * cfg.head_dim,
                            "attention.k_proj");
    validate_linear_weights(weights->attention.v_proj, cfg.d_model, cfg.num_kv_heads * cfg.head_dim,
                            "attention.v_proj");
    validate_linear_weights(weights->attention.o_proj, cfg.num_heads * cfg.head_dim, cfg.d_model, "attention.o_proj");
    validate_linear_weights(weights->mlp.gate, cfg.d_model, cfg.d_ff, "mlp.gate");
    validate_linear_weights(weights->mlp.up, cfg.d_model, cfg.d_ff, "mlp.up");
    validate_linear_weights(weights->mlp.down, cfg.d_ff, cfg.d_model, "mlp.down");
}

void validate_block_forward_input(const model::ModelConfig& cfg, const model::ForwardInput& input, const KVCache& cache) {
    if (input.hidden_states_ptr == nullptr && input.hidden_states.size() != input.seq_len * cfg.d_model) {
        throw std::invalid_argument("ForwardInput: hidden_states size mismatch");
    }
    if (input.layer_id >= cache.num_layers()) {
        throw std::out_of_range("ForwardInput: layer_id out of range");
    }
    if (cache.num_kv_heads() != cfg.num_kv_heads || cache.head_dim() != cfg.head_dim) {
        throw std::invalid_argument("ForwardInput: cache shape mismatch with model config");
    }
}

void validate_past_len_cache(bool use_cache, size_t past_len, size_t cache_seq_len_at_layer) {
    if (use_cache && past_len != cache_seq_len_at_layer) {
        throw std::invalid_argument("ForwardInput.past_len must match cache seq_len for append path");
    }
}

void validate_positions_size(size_t seq_len, const std::vector<size_t>& positions) {
    if (!positions.empty() && positions.size() != seq_len) {
        throw std::invalid_argument("positions size must equal seq_len");
    }
}

void validate_attention_mask_size(size_t seq_len, size_t total_kv_len, const std::vector<float>& mask) {
    if (!mask.empty() && mask.size() != seq_len * total_kv_len) {
        throw std::invalid_argument("attention_mask size must be seq_len * total_kv_len");
    }
}

void validate_kv_cache_config(const KVCacheConfig& cfg) {
    if (cfg.num_layers == 0 || cfg.num_kv_heads == 0 || cfg.max_seq_len == 0 || cfg.head_dim == 0 ||
        cfg.page_size == 0) {
        throw std::invalid_argument("KVCache: invalid config");
    }
}

void validate_kv_cache_layer(const KVCacheConfig& cfg, size_t layer) {
    if (layer >= cfg.num_layers) {
        throw std::out_of_range("KVCache: layer_id out of range");
    }
}

void validate_kv_cache_access(const KVCacheConfig& cfg, size_t layer, size_t hkv, size_t pos, size_t d) {
    validate_kv_cache_layer(cfg, layer);
    if (hkv >= cfg.num_kv_heads || pos >= cfg.max_seq_len || d >= cfg.head_dim) {
        throw std::out_of_range("KVCache: access out of range");
    }
}

void validate_kv_cache_append_input(const KVCacheConfig& cfg, size_t seq_len, const std::vector<float>& k_new,
                                    const std::vector<float>& v_new) {
    const size_t expect = seq_len * cfg.num_kv_heads * cfg.head_dim;
    if (k_new.size() != expect || v_new.size() != expect) {
        throw std::invalid_argument("KVCache::append: input size mismatch");
    }
}

} // namespace engine
