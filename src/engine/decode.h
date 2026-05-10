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
    // If false: logits has seq_len rows (row s = position s). If true: logits has one row per prediction step i,
    // aligned with prediction_steps order (same layout as cross_entropy_steps_into row index i).
    bool packed_prediction_rows = false;
};

struct CrossEntropyResult {
    float loss = 0.0f;
    float accuracy = 0.0f;
    std::vector<float> probs;
    std::vector<uint32_t> targets;
    std::vector<size_t> steps;
    size_t valid_steps = 0;
};

void cross_entropy_next_token_into(const LogitsOutput& logits, const std::vector<uint32_t>& target_ids,
                                   CrossEntropyResult& out);
void cross_entropy_steps_into(const LogitsOutput& logits, const std::vector<uint32_t>& target_ids,
                              const std::vector<size_t>& prediction_steps, CrossEntropyResult& out);
void project_logits_into(const std::vector<float>& hidden_states, size_t seq_len, size_t d_model,
                         const std::vector<float>& lm_head, size_t vocab_size, LogitsOutput& out);
void project_logits_steps_into(const float* hidden_states, size_t seq_len, size_t d_model,
                               std::span<const size_t> prediction_steps, const std::vector<float>& lm_head,
                               size_t vocab_size, LogitsOutput& out, std::vector<float>& gathered_workspace);
void logits_from_hidden_row(std::span<const float> hidden_row, const std::vector<float>& lm_head, size_t vocab_size,
                            size_t d_model, std::vector<float>& logits_out);
[[nodiscard]] uint32_t argmax_from_logits(const std::vector<float>& logits);
[[nodiscard]] uint32_t argmax_from_hidden(std::span<const float> hidden_row, const std::vector<float>& lm_head,
                                          size_t vocab_size, size_t d_model);
void softmax_stable_inplace(std::vector<float>& logits);
[[nodiscard]] uint32_t sample_from_logits(std::vector<float>& logits, float temperature, std::mt19937& rng,
                                          float temperature_argmax_eps = 1e-6f);
[[nodiscard]] uint32_t sample_from_hidden_row(std::span<const float> hidden_row, const std::vector<float>& lm_head,
                                              size_t vocab_size, size_t d_model, float temperature, std::mt19937& rng);

} // namespace engine
