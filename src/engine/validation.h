#pragma once

#include "engine/kv_cache.h"
#include "model/model_types.h"

#include <vector>

namespace engine {

void validate_model_config(const model::ModelConfig& cfg);
void validate_model_weights(const model::ModelConfig& cfg, const model::ModelWeights& weights);
void validate_model_forward(const model::ModelConfig& cfg, size_t num_layers, const model::ForwardInput& input,
                            const KVCache& cache);

void validate_linear_weights(const model::LinearWeights& linear, size_t in_dim, size_t out_dim, const char* name);
void validate_decoder_layer_weights(const model::ModelConfig& cfg, const model::DecoderLayerWeights* weights);
void validate_block_forward_input(const model::ModelConfig& cfg, const model::ForwardInput& input, const KVCache& cache);
void validate_past_len_cache(bool use_cache, size_t past_len, size_t cache_seq_len_at_layer);
void validate_positions_size(size_t seq_len, const std::vector<size_t>& positions);
void validate_attention_mask_size(size_t seq_len, size_t total_kv_len, const std::vector<float>& mask);

void validate_kv_cache_config(const KVCacheConfig& cfg);
void validate_kv_cache_layer(const KVCacheConfig& cfg, size_t layer);
void validate_kv_cache_access(const KVCacheConfig& cfg, size_t layer, size_t hkv, size_t pos, size_t d);
void validate_kv_cache_append_input(const KVCacheConfig& cfg, size_t seq_len, const std::vector<float>& k_new,
                                    const std::vector<float>& v_new);

} // namespace engine
