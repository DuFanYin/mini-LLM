#include "train/train.h"

#include "engine/decode.h"
#include "engine/embedding.h"
#include "task/task.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace train {

void Workspace::prepare(const model::MiniLlm& m) {
    const model::ModelWeights& weights = m.weights();
    const size_t num_layers = m.num_layers();
    if (grad.layers.size() != num_layers || grad.vocab_size != weights.vocab_size ||
        grad.token_embedding.size() != weights.token_embedding.size() ||
        grad.output_projection.size() != weights.output_projection.size()) {
        grad = zeros_like(weights);
    } else {
        zero(grad);
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

StepMetrics step(model::MiniLlm& m, const std::vector<uint32_t>& token_ids,
                 const std::vector<size_t>& prediction_steps, size_t t, float lr, float beta1, float beta2, float eps,
                 float weight_decay, float max_grad_norm, model::ModelWeights& adam_m, model::ModelWeights& adam_v,
                 Workspace& ws) {
    if (token_ids.size() < 2) {
        throw std::invalid_argument("train::step: need at least 2 tokens");
    }
    if (t == 0) {
        throw std::invalid_argument("train::step: t must be >= 1");
    }
    if (prediction_steps.empty()) {
        throw std::invalid_argument("train::step: prediction_steps must be non-empty");
    }
    for (size_t s : prediction_steps) {
        if (s + 1u >= token_ids.size()) {
            throw std::invalid_argument("train::step: prediction_steps index out of range for token_ids");
        }
    }

    ws.prepare(m);
    model::ModelWeights& weights = m.mutable_weights();
    model::ModelWeights& grad = ws.grad;

    ws.hidden_states.resize(token_ids.size() * m.d_model());
    m.forward_train(token_ids, ws.tapes, ws.hidden_states.data());

    engine::project_logits(ws.hidden_states.data(), token_ids.size(), m.d_model(),
                           std::span<const size_t>(prediction_steps.data(), prediction_steps.size()),
                           weights.output_projection, weights.vocab_size, ws.logits, ws.gathered_hidden);
    cross_entropy(ws.logits, token_ids, prediction_steps, ws.cross_entropy);
    const CrossEntropyResult& ce = ws.cross_entropy;

    cross_entropy_grad_weight(ws.hidden_states.data(), ce.probs.data(), ce.targets.data(), ce.steps.data(),
                              ce.valid_steps, m.d_model(), weights.vocab_size, grad.output_projection.data());
    ws.grad_hidden_out.resize(token_ids.size() * m.d_model());
    cross_entropy_grad_hidden(ce.probs.data(), ce.targets.data(), ce.steps.data(), ce.valid_steps,
                              weights.output_projection.data(), token_ids.size(), m.d_model(), weights.vocab_size,
                              ws.grad_hidden_out.data());

    m.backward(ws.tapes, ws.grad_hidden_out, grad, ws.grad_hidden_in);

    engine::embed_tokens_grad(token_ids, ws.grad_hidden_in, m.d_model(), grad.token_embedding);

    clip_grad(grad, max_grad_norm);
    adamw_step(weights, grad, adam_m, adam_v, t, lr, beta1, beta2, eps, weight_decay);

    StepMetrics metrics;
    metrics.loss = ce.loss;
    metrics.accuracy = ce.accuracy;
    metrics.valid_steps = ce.valid_steps;
    return metrics;
}

} // namespace train
