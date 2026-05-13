#include "engine/embedding.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace engine {

void embed_tokens(const std::vector<uint32_t>& token_ids, const std::vector<float>& token_embedding, size_t d_model,
                  std::vector<float>& hidden_out) {
    hidden_out.resize(token_ids.size() * d_model);
    for (size_t i = 0; i < token_ids.size(); ++i) {
        const uint32_t token_id = token_ids[i];
        const size_t src_off = static_cast<size_t>(token_id) * d_model;
        const size_t dst_off = i * d_model;
        std::memcpy(hidden_out.data() + dst_off, token_embedding.data() + src_off, d_model * sizeof(float));
    }
}

void embed_token(uint32_t token_id, const model::ModelWeights& weights, size_t d_model, std::vector<float>& out) {
    out.assign(d_model, 0.0f);
    const size_t off = static_cast<size_t>(token_id) * d_model;
    for (size_t d = 0; d < d_model; ++d) {
        out[d] = weights.token_embedding[off + d];
    }
}

void embed_tokens_grad(const std::vector<uint32_t>& token_ids, const std::vector<float>& grad_hidden, size_t d_model,
                       std::vector<float>& grad_token_embedding) {
    for (size_t step = 0; step < token_ids.size(); ++step) {
        const uint32_t token = token_ids[step];
        const size_t row_off = static_cast<size_t>(token) * d_model;
        const size_t gh_off = step * d_model;
        for (size_t d = 0; d < d_model; ++d) {
            grad_token_embedding[row_off + d] += grad_hidden[gh_off + d];
        }
    }
}

} // namespace engine
