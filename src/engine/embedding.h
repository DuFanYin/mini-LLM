#pragma once

#include "model/model_types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace engine {

void embed_token_ids_into(const std::vector<uint32_t>& token_ids, const std::vector<float>& token_embedding,
                          size_t d_model, std::vector<float>& hidden_out);
void embed_one_into(uint32_t token_id, const model::ModelWeights& w, size_t d_model, std::vector<float>& out);

} // namespace engine
