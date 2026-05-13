// Cross-entropy loss, its fused backward, and the training-side batch_metrics
// diagnostic. Pure training-time concern (inference never computes a loss); kept
// out of `engine::` deliberately so the layering reads: kernel < engine < train.

#include "train/train.h"

#include "engine/decode.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace train {

// ------------------------------------------------------------
// Forward cross-entropy
// ------------------------------------------------------------

void cross_entropy(const engine::LogitsOutput& logits, const std::vector<uint32_t>& target_ids,
                   CrossEntropyResult& out) {
    std::vector<size_t> steps;
    if (logits.seq_len > 1) {
        steps.resize(logits.seq_len - 1u);
        for (size_t step = 0; step + 1u < logits.seq_len; ++step) {
            steps[step] = step;
        }
    }
    cross_entropy(logits, target_ids, steps, out);
}

void cross_entropy(const engine::LogitsOutput& logits, const std::vector<uint32_t>& target_ids,
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

        size_t argmax_idx = 0;
        float best = row[0];
        for (size_t v = 1; v < vocab_size; ++v) {
            if (row[v] > best) {
                best = row[v];
                argmax_idx = v;
            }
        }
        if (argmax_idx == target) {
            ++correct;
        }
    }

    out.loss = total_loss / static_cast<float>(valid_steps);
    out.accuracy = static_cast<float>(correct) / static_cast<float>(valid_steps);
}

// ------------------------------------------------------------
// Backward of (softmax-CE + output projection)
// ------------------------------------------------------------

void cross_entropy_grad_weight(const float* hidden_states, const float* probs, const uint32_t* targets,
                               const size_t* steps, size_t valid_steps, size_t d_model, size_t vocab_size,
                               float* grad_output_projection) {
    std::fill(grad_output_projection, grad_output_projection + vocab_size * d_model, 0.0f);
    if (valid_steps == 0) {
        return;
    }
    const float inv_steps = 1.0f / static_cast<float>(valid_steps);
    for (size_t i = 0; i < valid_steps; ++i) {
        const uint32_t target = targets[i];
        const float* hidden_row = hidden_states + steps[i] * d_model;
        for (size_t v = 0; v < vocab_size; ++v) {
            float dlogit = probs[i * vocab_size + v];
            if (v == target) {
                dlogit -= 1.0f;
            }
            dlogit *= inv_steps;
            const size_t woff = v * d_model;
            for (size_t d = 0; d < d_model; ++d) {
                grad_output_projection[woff + d] += dlogit * hidden_row[d];
            }
        }
    }
}

void cross_entropy_grad_hidden(const float* probs, const uint32_t* targets, const size_t* steps, size_t valid_steps,
                               const float* output_projection, size_t total_steps, size_t d_model, size_t vocab_size,
                               float* grad_hidden_out) {
    std::fill(grad_hidden_out, grad_hidden_out + total_steps * d_model, 0.0f);
    if (valid_steps == 0) {
        return;
    }
    const float inv_steps = 1.0f / static_cast<float>(valid_steps);
    for (size_t i = 0; i < valid_steps; ++i) {
        const uint32_t target = targets[i];
        const size_t step = steps[i];
        for (size_t v = 0; v < vocab_size; ++v) {
            float dlogit = probs[i * vocab_size + v];
            if (v == target) {
                dlogit -= 1.0f;
            }
            dlogit *= inv_steps;
            const size_t woff = v * d_model;
            for (size_t d = 0; d < d_model; ++d) {
                grad_hidden_out[step * d_model + d] += dlogit * output_projection[woff + d];
            }
        }
    }
}

} // namespace train
