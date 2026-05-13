#include "task/task.h"
#include "engine/decode.h"
#include "kernel/kernel.h"
#include "train/train.h"

#include <cstring>
#include <random>
#include <span>
#include <vector>

namespace task {

// ------------------------------------------------------------
// Training-time metrics on a single batch (forward + CE).
// ------------------------------------------------------------

TrainBatchMetrics batch_metrics(model::MiniLlm& m, const std::vector<uint32_t>& token_ids) {
    const model::ModelWeights& weights = m.weights();
    std::vector<float> hidden_out(token_ids.size() * m.d_model(), 0.0f);
    m.prefill(token_ids, hidden_out.data());
    std::vector<size_t> steps;
    answer_steps(token_ids, steps);
    std::vector<float> gathered;
    engine::LogitsOutput logits;
    engine::project_logits(hidden_out.data(), token_ids.size(), m.d_model(),
                           std::span<const size_t>(steps.data(), steps.size()), weights.output_projection,
                           weights.vocab_size, logits, gathered);
    train::CrossEntropyResult ce;
    train::cross_entropy(logits, token_ids, steps, ce);

    TrainBatchMetrics out;
    out.loss = ce.loss;
    out.accuracy = ce.accuracy;
    out.valid_steps = ce.valid_steps;
    return out;
}

// ------------------------------------------------------------
// Inference-time correctness probes (argmax / greedy decode).
// ------------------------------------------------------------

uint32_t last_argmax(model::MiniLlm& m, const std::vector<uint32_t>& prompt) {
    const model::ModelWeights& weights = m.weights();
    const size_t vocab_size = weights.vocab_size;
    const size_t d_model = m.d_model();

    std::vector<float> hidden(prompt.size() * d_model, 0.0f);
    m.prefill(prompt, hidden.data());
    const float* last_row = hidden.data() + (prompt.size() - 1u) * d_model;
    std::vector<float> logits(vocab_size);
    kernel::gemm_nt(last_row, weights.output_projection.data(), logits.data(), 1, vocab_size, d_model);
    return engine::argmax(logits);
}

namespace {

// True iff every answer-span position's argmax token matches the gold token.
// Only used by `answer_accuracy` below.
bool is_answer_correct(model::MiniLlm& m, const std::vector<uint32_t>& token_ids) {
    Layout layout;
    if (!infer_layout(token_ids, layout)) {
        return false;
    }
    if (!is_valid(token_ids)) {
        return false;
    }

    const model::ModelWeights& weights = m.weights();
    const size_t d_model = m.d_model();
    std::vector<float> hidden_out(token_ids.size() * d_model, 0.0f);
    m.prefill(token_ids, hidden_out.data());
    std::vector<size_t> pred_steps;
    answer_steps(token_ids, pred_steps);
    std::vector<float> gathered;
    engine::LogitsOutput logits;
    engine::project_logits(hidden_out.data(), token_ids.size(), d_model,
                           std::span<const size_t>(pred_steps.data(), pred_steps.size()), weights.output_projection,
                           weights.vocab_size, logits, gathered);

    const size_t vocab_size = weights.vocab_size;
    for (size_t i = 0; i < layout.answer_len; ++i) {
        const float* row = &logits.logits[i * vocab_size];
        size_t argmax_idx = 0;
        float best = row[0];
        for (size_t v = 1; v < vocab_size; ++v) {
            if (row[v] > best) {
                best = row[v];
                argmax_idx = v;
            }
        }
        if (argmax_idx != token_ids[layout.answer_start + i]) {
            return false;
        }
    }
    return true;
}

} // anonymous namespace

float answer_accuracy(model::MiniLlm& m, const std::vector<std::vector<uint32_t>>& val_set) {
    size_t correct = 0;
    size_t total = 0;
    for (const auto& seq : val_set) {
        if (!is_valid(seq)) {
            continue;
        }
        if (is_answer_correct(m, seq)) {
            ++correct;
        }
        ++total;
    }
    return (total == 0) ? 0.0f : (static_cast<float>(correct) / static_cast<float>(total));
}

std::pair<size_t, size_t> probe_hits(model::MiniLlm& m, size_t trials, uint32_t rng_seed) {
    if (trials == 0 || m.weights().vocab_size < n_vocab()) {
        return {0u, 0u};
    }
    Sampler sampler(rng_seed);

    size_t hits = 0;
    for (size_t t = 0; t < trials; ++t) {
        const std::vector<uint32_t> sample = sampler.sample_sequence();
        Layout layout;
        if (!infer_layout(sample, layout)) {
            continue;
        }
        std::vector<uint32_t> prompt(sample.begin(),
                                     sample.begin() + static_cast<std::ptrdiff_t>(layout.idx_query + 1u));
        std::vector<uint32_t> gold(sample.begin() + static_cast<std::ptrdiff_t>(layout.answer_start),
                                   sample.begin() + static_cast<std::ptrdiff_t>(layout.answer_start + layout.answer_len));

        bool exact = true;
        uint32_t token = last_argmax(m, prompt);
        if (gold.empty() || token != gold[0]) {
            exact = false;
        }
        const size_t d_model = m.d_model();
        const size_t vocab_size = m.weights().vocab_size;
        std::vector<float> row(d_model, 0.0f);
        std::vector<float> logits(vocab_size);
        for (size_t i = 1; i < gold.size() && exact; ++i) {
            m.decode(token, row.data());
            kernel::gemm_nt(row.data(), m.weights().output_projection.data(), logits.data(), 1, vocab_size, d_model);
            token = engine::argmax(logits);
            if (token != gold[i]) {
                exact = false;
            }
        }
        if (exact) {
            ++hits;
        }
    }
    return {hits, trials};
}

} // namespace task
