#include "train/train.h"

#include "engine/decode.h"
#include "kernel/kernel.h"
#include "task/task.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace train {

void TrainWorkspace::prepare(const model::MiniLlm& m) {
    const model::ModelWeights& weights = m.weights();
    const size_t num_layers = m.num_layers();
    if (grad.layers.size() != num_layers || grad.vocab_size != weights.vocab_size ||
        grad.token_embedding.size() != weights.token_embedding.size() || grad.lm_head.size() != weights.lm_head.size()) {
        grad = clone_model(weights);
    } else {
        clear(grad);
    }
    tapes.resize(num_layers);

    const size_t d = m.d_model();
    const size_t cap = task::token_len_max() * d;
    hidden_states.reserve(cap);
    gathered_hidden.reserve(cap);
    grad_hidden_out.reserve(cap);
    grad_hidden_in.reserve(cap);
    prediction_steps.reserve(task::token_len_max());
}

StepMetrics train_step(model::MiniLlm& m, const std::vector<uint32_t>& token_ids,
                       const std::vector<size_t>& prediction_steps, size_t adam_time_step, float lr, float beta1,
                       float beta2, float eps, float weight_decay, float max_grad_norm, model::ModelWeights& adam_m,
                       model::ModelWeights& adam_v, TrainWorkspace& workspace) {
    if (token_ids.size() < 2) {
        throw std::invalid_argument("train_step: need at least 2 tokens");
    }
    if (adam_time_step == 0) {
        throw std::invalid_argument("train_step: adam_time_step must be >= 1");
    }
    if (prediction_steps.empty()) {
        throw std::invalid_argument("train_step: prediction_steps must be non-empty");
    }
    for (size_t step : prediction_steps) {
        if (step + 1u >= token_ids.size()) {
            throw std::invalid_argument("train_step: prediction_steps index out of range for token_ids");
        }
    }

    workspace.prepare(m);
    model::ModelWeights& weights = m.mutable_weights();
    model::ModelWeights& grad = workspace.grad;

    workspace.hidden_states.resize(token_ids.size() * m.d_model());
    m.forward_for_training(token_ids, workspace.tapes, workspace.hidden_states.data());

    engine::project_logits_steps_into(workspace.hidden_states.data(), token_ids.size(), m.d_model(),
                                      std::span<const size_t>(prediction_steps.data(), prediction_steps.size()),
                                      weights.lm_head, weights.vocab_size, workspace.logits, workspace.gathered_hidden);
    engine::cross_entropy_steps_into(workspace.logits, token_ids, prediction_steps, workspace.cross_entropy);
    const engine::CrossEntropyResult& ce = workspace.cross_entropy;

    kernel::backward_lm_head(workspace.hidden_states.data(), ce.probs.data(), ce.targets.data(), ce.steps.data(),
                             ce.valid_steps, m.d_model(), weights.vocab_size, grad.lm_head.data());

    workspace.grad_hidden_out.resize(token_ids.size() * m.d_model());
    kernel::backward_hidden(ce.probs.data(), ce.targets.data(), ce.steps.data(), ce.valid_steps, weights.lm_head.data(),
                            token_ids.size(), m.d_model(), weights.vocab_size, workspace.grad_hidden_out.data());

    m.backward(workspace.tapes, workspace.grad_hidden_out, grad, workspace.grad_hidden_in);

    kernel::backward_embedding(token_ids.data(), token_ids.size(), workspace.grad_hidden_in.data(), m.d_model(),
                               grad.token_embedding.data());

    clip_grad(grad, max_grad_norm);
    adamw_update_model(weights, grad, adam_m, adam_v, adam_time_step, lr, beta1, beta2, eps, weight_decay);

    StepMetrics metrics;
    metrics.loss = ce.loss;
    metrics.accuracy = ce.accuracy;
    metrics.valid_steps = ce.valid_steps;
    return metrics;
}

} // namespace train
