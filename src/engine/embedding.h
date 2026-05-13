#pragma once

#include "model/model_types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace engine {

// Gather token embedding rows into a hidden_states tensor.
void embed_tokens(const std::vector<uint32_t>& token_ids, const std::vector<float>& token_embedding, size_t d_model,
                  std::vector<float>& hidden_out);

// Gather a single token embedding row.
void embed_token(uint32_t token_id, const model::ModelWeights& weights, size_t d_model, std::vector<float>& out);

// Adjoint of embed_tokens: scatter-add grad_hidden rows back into the rows pointed to by token_ids.
// Accumulates into grad_token_embedding; does not clear it.
void embed_tokens_grad(const std::vector<uint32_t>& token_ids, const std::vector<float>& grad_hidden, size_t d_model,
                       std::vector<float>& grad_token_embedding);

} // namespace engine
