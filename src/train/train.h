#pragma once

#include "engine/decode.h"
#include "model/mini_llm.h"
#include "model/model_types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace train {

// --- loss (cross-entropy) ---

// Cached softmax probabilities + targets per prediction step; sized once by
// `cross_entropy` (forward) and consumed by `cross_entropy_backward`.
struct CrossEntropyResult {
    float loss = 0.0f;
    float accuracy = 0.0f;
    std::vector<float> probs;
    std::vector<uint32_t> targets;
    std::vector<size_t> steps;
    size_t valid_steps = 0;
};

// Cross-entropy over all next-token positions (predict step s+1 from step s).
void cross_entropy(const engine::LogitsOutput& logits, const std::vector<uint32_t>& target_ids,
                   CrossEntropyResult& out);

// Cross-entropy at the explicitly requested prediction steps.
void cross_entropy(const engine::LogitsOutput& logits, const std::vector<uint32_t>& target_ids,
                   const std::vector<size_t>& prediction_steps, CrossEntropyResult& out);

// Gradient of cross-entropy w.r.t. the output projection W. Writes vocab_size * d_model entries (overwrites).
void cross_entropy_grad_weight(const float* hidden_states, const float* probs, const uint32_t* targets,
                               const size_t* steps, size_t valid_steps, size_t d_model, size_t vocab_size,
                               float* grad_output_projection);

// Gradient of cross-entropy w.r.t. the hidden input. Writes total_steps * d_model entries (overwrites).
void cross_entropy_grad_hidden(const float* probs, const uint32_t* targets, const size_t* steps, size_t valid_steps,
                               const float* output_projection, size_t total_steps, size_t d_model, size_t vocab_size,
                               float* grad_hidden_out);

// --- optimizer ---
// Zero-initialized ModelWeights mirroring the shape of `ref`. Used to allocate
// gradient and Adam moment buffers.
[[nodiscard]] model::ModelWeights zeros_like(const model::ModelWeights& ref);

// Zero every tensor of `g` in place (shape unchanged).
void zero(model::ModelWeights& g);

// Rescale every tensor in `g` so its global L2 norm is at most `max_norm`.
// No-op when max_norm <= 0 or the current norm already fits.
void clip_grad(model::ModelWeights& g, float max_norm);

// One AdamW step over the whole model. `m` and `v` are first/second moments
// (same shape as `param`), `t` is the 1-based optimizer step index.
void adamw_step(model::ModelWeights& param, const model::ModelWeights& grad, model::ModelWeights& m,
                model::ModelWeights& v, size_t t, float lr, float beta1, float beta2, float eps, float weight_decay);

// --- step ---
// Metrics from one optimizer step (task-agnostic).
struct StepMetrics {
    float loss = 0.0f;
    float accuracy = 0.0f;
    size_t valid_steps = 0;
};

// Standard causal LM: train on every next-token position (steps 0..seq_len-2 predicting token at step+1).
[[nodiscard]] inline std::vector<size_t> next_token_steps(size_t seq_len) {
    std::vector<size_t> steps;
    if (seq_len < 2) {
        return steps;
    }
    steps.reserve(seq_len - 1u);
    for (size_t s = 0; s + 1u < seq_len; ++s) {
        steps.push_back(s);
    }
    return steps;
}

// Reusable per-step buffers (gradients, hidden tapes, logits, etc.) shared across `train::step` calls.
struct Workspace {
    model::ModelWeights grad{};
    std::vector<model::BlockForwardTape> tapes;
    std::vector<size_t> prediction_steps;
    std::vector<float> hidden_states;
    std::vector<float> gathered_hidden;
    engine::LogitsOutput logits;
    CrossEntropyResult cross_entropy;
    std::vector<float> grad_hidden_out;
    std::vector<float> grad_hidden_in;

    void prepare(const model::MiniLlm& m);
};

// Forward + backward + AdamW update for one batch. `t` is the 1-based optimizer step index.
[[nodiscard]] StepMetrics step(model::MiniLlm& m, const std::vector<uint32_t>& token_ids,
                               const std::vector<size_t>& prediction_steps, size_t t, float lr, float beta1,
                               float beta2, float eps, float weight_decay, float max_grad_norm,
                               model::ModelWeights& adam_m, model::ModelWeights& adam_v, Workspace& ws);

} // namespace train
