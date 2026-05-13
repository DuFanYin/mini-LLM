#pragma once

#include "kernel/kernel.h"
#include "model/kv_cache.h"
#include "model/model_types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace model {

class MiniLlm {
public:
    MiniLlm(ModelConfig config, ModelWeights weights);

    /// Entry 1: random-initialized model with fixed in-repo hyperparams.
    [[nodiscard]] static MiniLlm init_random(size_t vocab_size, uint32_t seed);

    /// Entry 2: load config/weights from checkpoint and configure cache.
    [[nodiscard]] static std::unique_ptr<MiniLlm> init_load(const std::string& path, size_t max_seq_len = 128);

    [[nodiscard]] size_t num_layers() const noexcept;
    [[nodiscard]] size_t d_model() const noexcept;
    [[nodiscard]] const ModelConfig& config() const noexcept;
    [[nodiscard]] const ModelWeights& weights() const noexcept;
    [[nodiscard]] ModelWeights& mutable_weights() noexcept;
    void configure_cache(size_t max_seq_len, size_t page_size = 16);
    void reset_cache();
    
    void prefill(const std::vector<uint32_t>& token_ids, float* hidden_out);
    void decode(uint32_t token_id, float* hidden_out);
    void forward_train(const std::vector<uint32_t>& token_ids, std::vector<BlockForwardTape>& layer_tapes,
                              float* hidden_out) const;
    void backward(const std::vector<BlockForwardTape>& tapes,
                  const std::vector<float>& grad_hidden_out,
                  ModelWeights& grad_weights,
                  std::vector<float>& grad_hidden_in) const;

private:
    void ensure_cache(size_t min_seq_len);

    ModelConfig config_;
    ModelWeights weights_;
    std::vector<DecoderLayerWeights*> layer_weights_;
    mutable std::vector<kernel::RopeCache> rope_q_;
    mutable std::vector<kernel::RopeCache> rope_k_;
    std::unique_ptr<KVCache> cache_;
    size_t cache_page_size_ = 16;
};

} // namespace model
