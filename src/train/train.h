#pragma once

#include "engine/decode.h"
#include "model/mini_llm.h"
#include "model/model_types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace train {

// --- optimizer ---
[[nodiscard]] model::DecoderLayerWeights clone_decoder_layer_weights(const model::DecoderLayerWeights& ref);
[[nodiscard]] model::ModelWeights clone_model(const model::ModelWeights& ref);
void clear(model::LinearWeights& lg);
void clear(model::DecoderLayerWeights& g);
void clear(model::ModelWeights& g);
void vector_add_sq_sum(const std::vector<float>& v, double* acc);
[[nodiscard]] double grad_l2_sq(const model::ModelWeights& g);
void vector_scale(std::vector<float>& v, float scale);
void clip_grad(model::ModelWeights& g, float max_norm);
void adamw_update_vec(std::vector<float>& param, const std::vector<float>& grad, std::vector<float>& m,
                      std::vector<float>& v, size_t t, float lr, float beta1, float beta2, float eps,
                      float weight_decay);
void adamw_update_decoder_layer(model::DecoderLayerWeights& param, const model::DecoderLayerWeights& grad,
                                model::DecoderLayerWeights& m, model::DecoderLayerWeights& v, size_t t, float lr,
                                float beta1, float beta2, float eps, float weight_decay);
void adamw_update_model(model::ModelWeights& param, const model::ModelWeights& grad,
                        model::ModelWeights& m, model::ModelWeights& v, size_t t, float lr, float beta1,
                        float beta2, float eps, float weight_decay);

// --- autograd + train_step ---
// Metrics from one optimizer step (task-agnostic).
struct StepMetrics {
    float loss = 0.0f;
    float accuracy = 0.0f;
    size_t valid_steps = 0;
};

// Standard causal LM: train on every next-token position (steps 0..seq_len-2 predicting token at step+1).
[[nodiscard]] inline std::vector<size_t> all_next_token_prediction_steps(size_t seq_len) {
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

struct TrainWorkspace {
    model::ModelWeights grad{};
    std::vector<model::BlockForwardTape> tapes;
    std::vector<size_t> prediction_steps;
    std::vector<float> hidden_states;
    std::vector<float> gathered_hidden;
    engine::LogitsOutput logits;
    engine::CrossEntropyResult cross_entropy;
    std::vector<float> grad_hidden_out;
    std::vector<float> grad_hidden_in;

    void prepare(const model::MiniLlm& m);
};

[[nodiscard]] StepMetrics train_step(model::MiniLlm& m, const std::vector<uint32_t>& token_ids,
                                   const std::vector<size_t>& prediction_steps, size_t adam_time_step, float lr,
                                   float beta1, float beta2, float eps, float weight_decay, float max_grad_norm,
                                   model::ModelWeights& adam_m, model::ModelWeights& adam_v,
                                   TrainWorkspace& workspace);

} // namespace train
