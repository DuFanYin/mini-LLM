#include "kernel/kernel.h"

#include <algorithm>

namespace kernel {

void backward_output_projection(const float* hidden_states, const float* probs, const uint32_t* targets,
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

void backward_hidden(const float* probs, const uint32_t* targets, const size_t* steps, size_t valid_steps,
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

void backward_embedding(const uint32_t* token_ids, size_t seq_len, const float* grad_hidden, size_t d_model,
                        float* grad_token_embed) {
    for (size_t step = 0; step < seq_len; ++step) {
        const uint32_t token = token_ids[step];
        const size_t row_off = static_cast<size_t>(token) * d_model;
        const size_t gh_off = step * d_model;
        for (size_t d = 0; d < d_model; ++d) {
            grad_token_embed[row_off + d] += grad_hidden[gh_off + d];
        }
    }
}

} // namespace kernel
