#pragma once

#include "model/model_types.h"

#include <string>

namespace engine {

// On-disk representation of a trained mini-LLM: model config + parameter tensors.
// Not a "training checkpoint" — there is no optimizer state, RNG state, or step count here.
struct SavedModel {
    model::ModelConfig config;
    model::ModelWeights weights;
};

void save_model(const std::string& path, const model::ModelConfig& config, const model::ModelWeights& weights);
[[nodiscard]] SavedModel load_model(const std::string& path);

} // namespace engine
