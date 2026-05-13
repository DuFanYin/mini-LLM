#pragma once

#include "model/mini_llm.h"

#include <vector>

namespace model {

// Model execution API: `forward_model` (full decoder stack) and `backward_model` (full stack adjoint over the
// decoder stack only — input/output projection and embedding gradients are driven by the training-step
// orchestrator via `engine::backward_*` / `engine::*_backward_into`).
// Layer-level forward steps are file-local in `executor_forward.cpp`; backward stages in `executor_backward.cpp`.
// `cache` may be null when `input.use_cache == false` (training / one-shot prefill without caching).

void forward_model(const ModelConfig& config, const std::vector<DecoderLayerWeights*>& layer_weights,
                   std::vector<RopeCache>& rope_q, std::vector<RopeCache>& rope_k,
                   const ForwardInput& input,
                   KVCache* cache, float* hidden_out, std::vector<BlockForwardTape>* layer_tapes);

void backward_model(const ModelConfig& config, const ModelWeights& weights,
                    const std::vector<BlockForwardTape>& tapes, const std::vector<float>& grad_hidden_out,
                    ModelWeights& grad_weights, std::vector<float>& grad_hidden_in);

} // namespace model
