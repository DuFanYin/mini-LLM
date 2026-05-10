#pragma once

#include "model/mini_llm.h"

#include <vector>

namespace model {

// Shared declarations for model execution. Implementations are split by direction:
// `executor_forward.cpp` owns forward paths, `executor_backward.cpp` owns backward paths.
//
// Naming:
// - `forward_attention_block` / backward stages: norm1 -> QKV -> RoPE -> KV -> GQA -> o_proj -> first residual only.
// - `forward_decoder_layer`: one full transformer decoder layer (attention submodule + MLP).
// - `backward_model`: full decoder-stack backward (per layer in .cpp: `backward_decoder_mlp` ->
//   `backward_decoder_attention` -> `backward_decoder_qkv_norm`).
// - `forward_model`: loop over layers (trunk only); each step builds CacheView then calls
//   `forward_decoder_layer` → `forward_attention_block` (+ MLP inside decoder layer).

void forward_attention_block(const ModelConfig& config, const DecoderLayerWeights& weights, const ForwardInput& input,
                             const CacheView& cache_view, const CacheBridge& cache, kernel::RopeCache& rope_q,
                             kernel::RopeCache& rope_k, float* hidden_after_attn_out, BlockForwardTape* tape);

void forward_decoder_layer(const ModelConfig& config, const DecoderLayerWeights& weights, const ForwardInput& input,
                           const CacheView& cache_view, const CacheBridge& cache, kernel::RopeCache& rope_q,
                           kernel::RopeCache& rope_k, float* hidden_out, BlockForwardTape* tape);

void forward_model(const ModelConfig& config, const std::vector<DecoderLayerState>& layers, const ForwardInput& input,
                   CacheBridge& cache, float* hidden_out, std::vector<BlockForwardTape>* layer_tapes);

void backward_decoder_mlp(const std::vector<float>& grad_layer_out, const BlockForwardTape& tape,
                          const ModelConfig& config, const DecoderLayerWeights& weights,
                          DecoderLayerWeights& layer_grad, size_t seq_len, std::vector<float>& grad_post_attn);

void backward_decoder_attention(const std::vector<float>& grad_post_attn, const BlockForwardTape& tape,
                                const ModelConfig& config, const DecoderLayerWeights& weights,
                                DecoderLayerWeights& layer_grad, kernel::RopeCache& rope_k, size_t seq_len,
                                size_t q_proj_dim, size_t kv_proj_dim, size_t total_kv_len, size_t past_len,
                                std::vector<float>& grad_q, std::vector<float>& grad_k_pre_rope,
                                std::vector<float>& grad_v_proj);

void backward_decoder_qkv_norm(const std::vector<float>& grad_post_attn, const std::vector<float>& grad_q,
                               const std::vector<float>& grad_k_pre_rope, const std::vector<float>& grad_v_proj,
                               const BlockForwardTape& tape, const ModelConfig& config,
                               const DecoderLayerWeights& weights, DecoderLayerWeights& layer_grad,
                               kernel::RopeCache& rope_q, size_t seq_len, size_t q_proj_dim, size_t kv_proj_dim,
                               std::vector<float>& grad_layer_in);

void backward_model(const ModelConfig& config, const ModelWeights& weights,
                    const std::vector<BlockForwardTape>& tapes, const std::vector<float>& grad_hidden_out,
                    ModelWeights& grad_weights, std::vector<float>& grad_hidden_in);

} // namespace model
