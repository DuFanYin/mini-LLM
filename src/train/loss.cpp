#include "engine/decode.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace engine {

void cross_entropy_next_token_into(const LogitsOutput& logits, const std::vector<uint32_t>& target_ids,
                                   CrossEntropyResult& out) {
    std::vector<size_t> steps;
    if (logits.seq_len > 1) {
        steps.resize(logits.seq_len - 1u);
        for (size_t step = 0; step + 1u < logits.seq_len; ++step) {
            steps[step] = step;
        }
    }
    cross_entropy_steps_into(logits, target_ids, steps, out);
}

void cross_entropy_steps_into(const LogitsOutput& logits, const std::vector<uint32_t>& target_ids,
                              const std::vector<size_t>& prediction_steps, CrossEntropyResult& out) {
    out = CrossEntropyResult{};
    if (prediction_steps.empty()) {
        return;
    }

    const size_t valid_steps = prediction_steps.size();
    const size_t vocab_size = logits.vocab_size;
    out.valid_steps = valid_steps;
    out.targets.resize(valid_steps);
    out.steps = prediction_steps;
    out.probs.assign(valid_steps * vocab_size, 0.0f);

    float total_loss = 0.0f;
    size_t correct = 0;
    for (size_t i = 0; i < valid_steps; ++i) {
        const size_t step = prediction_steps[i];
        const float* row = logits.packed_prediction_rows ? &logits.logits[i * vocab_size]
                                                         : &logits.logits[step * vocab_size];

        float max_logit = -std::numeric_limits<float>::infinity();
        for (size_t v = 0; v < vocab_size; ++v) {
            max_logit = std::max(max_logit, row[v]);
        }

        float denom = 0.0f;
        for (size_t v = 0; v < vocab_size; ++v) {
            const float e = std::exp(row[v] - max_logit);
            out.probs[i * vocab_size + v] = e;
            denom += e;
        }
        const float inv_denom = 1.0f / denom;
        for (size_t v = 0; v < vocab_size; ++v) {
            out.probs[i * vocab_size + v] *= inv_denom;
        }

        const uint32_t target = target_ids[step + 1];
        out.targets[i] = target;

        const float p = std::max(out.probs[i * vocab_size + target], 1e-12f);
        total_loss += -std::log(p);

        size_t argmax = 0;
        float best = row[0];
        for (size_t v = 1; v < vocab_size; ++v) {
            if (row[v] > best) {
                best = row[v];
                argmax = v;
            }
        }
        if (argmax == target) {
            ++correct;
        }
    }

    out.loss = total_loss / static_cast<float>(valid_steps);
    out.accuracy = static_cast<float>(correct) / static_cast<float>(valid_steps);
}

} // namespace engine
