#include "engine/decode.h"
#include "kernel/kernel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>

namespace engine {

// ------------------------------------------------------------
// Forward: hidden -> logits
// ------------------------------------------------------------

void project_logits(const std::vector<float>& hidden_states, size_t seq_len, size_t d_model,
                    const std::vector<float>& output_projection, size_t vocab_size, LogitsOutput& out) {
    out.seq_len = seq_len;
    out.vocab_size = vocab_size;
    out.packed_prediction_rows = false;
    out.logits.resize(seq_len * vocab_size);
    kernel::gemm_nt(hidden_states.data(), output_projection.data(), out.logits.data(), seq_len, vocab_size, d_model);
}

void project_logits(const float* hidden_states, size_t seq_len, size_t d_model,
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
            throw std::invalid_argument("project_logits: prediction step out of range");
        }
        std::memcpy(gathered_workspace.data() + i * d_model, hidden_states + step * d_model, d_model * sizeof(float));
    }
    out.logits.resize(n * vocab_size);
    if (n == 0) {
        return;
    }
    kernel::gemm_nt(gathered_workspace.data(), output_projection.data(), out.logits.data(), n, vocab_size, d_model);
}

// ------------------------------------------------------------
// Decode-time helpers (operate on logits; callers project hidden -> logits first)
// ------------------------------------------------------------

uint32_t argmax(const std::vector<float>& logits) {
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

void softmax(std::vector<float>& logits) {
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

uint32_t sample(std::vector<float>& logits, float temperature, std::mt19937& rng, float temperature_argmax_eps) {
    if (temperature < temperature_argmax_eps) {
        return argmax(logits);
    }
    for (float& z : logits) {
        z /= temperature;
    }
    softmax(logits);
    std::discrete_distribution<size_t> dist(logits.begin(), logits.end());
    return static_cast<uint32_t>(dist(rng));
}

} // namespace engine
