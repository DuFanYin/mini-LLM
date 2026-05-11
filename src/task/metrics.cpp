#include "task/task.h"
#include "engine/decode.h"

#include <span>

namespace task {

TrainBatchMetrics compute_batch_metrics(model::MiniLlm& m, const std::vector<uint32_t>& token_ids) {
    const model::ModelWeights& weights = m.weights();
    std::vector<float> hidden_out(token_ids.size() * m.d_model(), 0.0f);
    m.prefill(token_ids, hidden_out.data());
    std::vector<size_t> steps;
    answer_prediction_steps_into(token_ids, steps);
    std::vector<float> gathered;
    engine::LogitsOutput logits;
    engine::project_logits_steps_into(hidden_out.data(), token_ids.size(), m.d_model(),
                                      std::span<const size_t>(steps.data(), steps.size()), weights.output_projection,
                                      weights.vocab_size, logits, gathered);
    engine::CrossEntropyResult ce;
    engine::cross_entropy_steps_into(logits, token_ids, steps, ce);

    TrainBatchMetrics out;
    out.loss = ce.loss;
    out.accuracy = ce.accuracy;
    out.valid_steps = ce.valid_steps;
    return out;
}

bool is_next_prediction_correct(model::MiniLlm& m, const std::vector<uint32_t>& token_ids, size_t step) {
    const model::ModelWeights& weights = m.weights();
    const size_t d_model = m.d_model();
    std::vector<float> hidden_out(token_ids.size() * d_model, 0.0f);
    m.prefill(token_ids, hidden_out.data());

    const size_t vocab_size = weights.vocab_size;
    std::vector<float> logits_row;
    engine::logits_from_hidden_row(std::span<const float>(hidden_out.data() + step * d_model, d_model), weights.output_projection,
                                   vocab_size, d_model, logits_row);
    size_t argmax = 0;
    float best = logits_row[0];
    for (size_t v = 1; v < vocab_size; ++v) {
        if (logits_row[v] > best) {
            best = logits_row[v];
            argmax = v;
        }
    }
    return argmax == token_ids[step + 1];
}

} // namespace task
