#pragma once

#include "model/mini_llm.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace task {

// Which-Span only: PREFIX [SA] SPAN_A [EA] MIDDLE [SB] SPAN_B [EB] SUFFIX [Q_A|Q_B] GOLD_ANSWER

// --- vocab + length bounds ---

inline constexpr uint32_t tok_sa() noexcept { return 26u; }
inline constexpr uint32_t tok_ea() noexcept { return 27u; }
inline constexpr uint32_t tok_sb() noexcept { return 28u; }
inline constexpr uint32_t tok_eb() noexcept { return 29u; }
inline constexpr uint32_t tok_qa() noexcept { return 30u; }
inline constexpr uint32_t tok_qb() noexcept { return 31u; }

inline constexpr size_t n_vocab() noexcept { return 32u; }

inline constexpr size_t prefix_len_min() noexcept { return 2u; }
inline constexpr size_t prefix_len_max() noexcept { return 10u; }
inline constexpr size_t middle_len_min() noexcept { return 2u; }
inline constexpr size_t middle_len_max() noexcept { return 10u; }
inline constexpr size_t suffix_len_min() noexcept { return 2u; }
inline constexpr size_t suffix_len_max() noexcept { return 10u; }
inline constexpr size_t span_a_len_min() noexcept { return 2u; }
inline constexpr size_t span_a_len_max() noexcept { return 8u; }
inline constexpr size_t span_b_len_min() noexcept { return 2u; }
inline constexpr size_t span_b_len_max() noexcept { return 8u; }

inline constexpr size_t token_len_min() noexcept {
    return prefix_len_min() + 1u + span_a_len_min() + 1u + middle_len_min() + 1u + span_b_len_min() + 1u +
           suffix_len_min() + 1u + span_a_len_min();
}
inline constexpr size_t token_len_max() noexcept {
    return prefix_len_max() + 1u + span_a_len_max() + 1u + middle_len_max() + 1u + span_b_len_max() + 1u +
           suffix_len_max() + 1u + span_a_len_max();
}

[[nodiscard]] inline constexpr char to_char(uint32_t letter_id) noexcept {
    return (letter_id < 26u) ? static_cast<char>('A' + letter_id) : '?';
}

// --- data + layout ---

struct Layout {
    size_t idx_sa = 0;
    size_t idx_ea = 0;
    size_t idx_sb = 0;
    size_t idx_eb = 0;
    size_t idx_query = 0;
    bool query_is_a = true;
    size_t answer_start = 0;
    size_t answer_len = 0;
};

class Sampler {
public:
    explicit Sampler(uint32_t seed);
    [[nodiscard]] std::vector<uint32_t> sample_sequence();
    [[nodiscard]] std::vector<std::vector<uint32_t>> sample_batch(size_t batch_size);

private:
    std::mt19937 rng_;
};

[[nodiscard]] bool is_valid(const std::vector<uint32_t>& t) noexcept;
[[nodiscard]] bool infer_layout(const std::vector<uint32_t>& t, Layout& out) noexcept;

// Prediction-step indices (positions where the model predicts the next answer token).
void answer_steps(const std::vector<uint32_t>& t, std::vector<size_t>& out);
[[nodiscard]] std::vector<size_t> answer_steps(const std::vector<uint32_t>& t);

// Deterministic batch of `count` sequences drawn from a fresh Sampler at `seed`.
[[nodiscard]] std::vector<std::vector<uint32_t>> batch_from_seed(size_t count, uint32_t seed);

// --- evaluation ---

struct TrainBatchMetrics {
    float loss = 0.0f;
    float accuracy = 0.0f;
    size_t valid_steps = 0;
};

// Forward + CE on a single sequence; loss/accuracy/valid-step count.
[[nodiscard]] TrainBatchMetrics batch_metrics(model::MiniLlm& m, const std::vector<uint32_t>& token_ids);

// argmax of the next-token logits at the last position of `prompt`.
[[nodiscard]] uint32_t last_argmax(model::MiniLlm& m, const std::vector<uint32_t>& prompt);

// Fraction of validation sequences for which the model's argmax answer span matches gold.
[[nodiscard]] float answer_accuracy(model::MiniLlm& m, const std::vector<std::vector<uint32_t>>& val_set);

// Sample `trials` fresh sequences, prefill+decode the answer span, return (hits, trials)
// where `hits` counts sequences whose decoded span exactly matches gold.
[[nodiscard]] std::pair<size_t, size_t> probe_hits(model::MiniLlm& m, size_t trials, uint32_t rng_seed = 9001);

} // namespace task
