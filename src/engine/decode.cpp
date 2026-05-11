#include "engine/decode.h"
#include "kernel/kernel.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>

namespace engine {

void project_logits_into(const std::vector<float>& hidden_states, size_t seq_len, size_t d_model,
                         const std::vector<float>& output_projection, size_t vocab_size, LogitsOutput& out) {
    out.seq_len = seq_len;
    out.vocab_size = vocab_size;
    out.packed_prediction_rows = false;
    out.logits.resize(seq_len * vocab_size);
    kernel::gemm_nt(hidden_states.data(), output_projection.data(), out.logits.data(), seq_len, vocab_size, d_model);
}

void project_logits_steps_into(const float* hidden_states, size_t seq_len, size_t d_model,
                               std::span<const size_t> prediction_steps, const std::vector<float>& output_projection,
                               size_t vocab_size, LogitsOutput& out, std::vector<float>& gathered_workspace) {
    out.seq_len = seq_len;
    out.vocab_size = vocab_size;
    out.packed_prediction_rows = true;
    const size_t n = prediction_steps.size();
    gathered_workspace.resize(n * d_model);
    for (size_t i = 0; i < n; ++i) {
        const size_t step = prediction_steps[i];
        if (step >= seq_len) {
            throw std::invalid_argument("project_logits_steps_into: prediction step out of range");
        }
        std::memcpy(gathered_workspace.data() + i * d_model, hidden_states + step * d_model, d_model * sizeof(float));
    }
    out.logits.resize(n * vocab_size);
    if (n == 0) {
        return;
    }
    kernel::gemm_nt(gathered_workspace.data(), output_projection.data(), out.logits.data(), n, vocab_size, d_model);
}

void logits_from_hidden_row(std::span<const float> hidden_row, const std::vector<float>& output_projection, size_t vocab_size,
                            size_t d_model, std::vector<float>& logits_out) {
    logits_out.resize(vocab_size);
    kernel::gemm_nt(hidden_row.data(), output_projection.data(), logits_out.data(), 1, vocab_size, d_model);
}

uint32_t argmax_from_logits(const std::vector<float>& logits) {
    uint32_t best = 0;
    float best_v = logits[0];
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = static_cast<uint32_t>(i);
        }
    }
    return best;
}

uint32_t argmax_from_hidden(std::span<const float> hidden_row, const std::vector<float>& output_projection, size_t vocab_size,
                            size_t d_model) {
    std::vector<float> logits;
    logits_from_hidden_row(hidden_row, output_projection, vocab_size, d_model, logits);
    return argmax_from_logits(logits);
}

void softmax_inplace(std::vector<float>& logits) {
    float max_logit = -std::numeric_limits<float>::infinity();
    for (float z : logits) {
        max_logit = std::max(max_logit, z);
    }
    float denom = 0.0f;
    for (float& z : logits) {
        const float e = std::exp(z - max_logit);
        z = e;
        denom += e;
    }
    const float inv = 1.0f / denom;
    for (float& z : logits) {
        z *= inv;
    }
}

uint32_t sample_from_logits(std::vector<float>& logits, float temperature, std::mt19937& rng,
                            float temperature_argmax_eps) {
    if (temperature < temperature_argmax_eps) {
        return argmax_from_logits(logits);
    }
    for (float& z : logits) {
        z /= temperature;
    }
    softmax_inplace(logits);
    std::discrete_distribution<size_t> dist(logits.begin(), logits.end());
    return static_cast<uint32_t>(dist(rng));
}

uint32_t sample_from_hidden_row(std::span<const float> hidden_row, const std::vector<float>& output_projection,
                                size_t vocab_size, size_t d_model, float temperature, std::mt19937& rng) {
    std::vector<float> logits;
    logits_from_hidden_row(hidden_row, output_projection, vocab_size, d_model, logits);
    return sample_from_logits(logits, temperature, rng);
}

} // namespace engine
