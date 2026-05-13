#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace engine {

struct LogitsOutput {
    std::vector<float> logits;
    size_t seq_len = 0;
    size_t vocab_size = 0;
    // If false: logits has seq_len rows (row s = position s). If true: logits has one row per prediction step i.
    bool packed_prediction_rows = false;
};

// ------------------------------------------------------------
// Forward: hidden -> logits
// ------------------------------------------------------------

// Project hidden states to logits at every position.
void project_logits(const std::vector<float>& hidden_states, size_t seq_len, size_t d_model,
                    const std::vector<float>& output_projection, size_t vocab_size, LogitsOutput& out);

// Project hidden states to logits only at the requested prediction steps (packed rows).
void project_logits(const float* hidden_states, size_t seq_len, size_t d_model,
                    std::span<const size_t> prediction_steps, const std::vector<float>& output_projection,
                    size_t vocab_size, LogitsOutput& out, std::vector<float>& gathered_workspace);

// ------------------------------------------------------------
// Decode-time helpers (operate on logits; callers project hidden -> logits first)
// ------------------------------------------------------------

[[nodiscard]] uint32_t argmax(const std::vector<float>& logits);

void softmax(std::vector<float>& logits);

[[nodiscard]] uint32_t sample(std::vector<float>& logits, float temperature, std::mt19937& rng,
                              float temperature_argmax_eps = 1e-6f);

} // namespace engine
