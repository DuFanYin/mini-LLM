#pragma once

#include "engine/kv_cache.h"
#include "kernel/kernel.h"
#include "model/model_types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace model {

// Per-layer resources for the decoder stack: block weights + RoPE caches (mutable during forward).
// `MiniLlm` owns these; `forward_model` reads them while calling `forward_decoder_layer` →
// `forward_attention_block`.
struct DecoderLayerState {
    DecoderLayerWeights* weights = nullptr;
    mutable kernel::RopeCache rope_q;
    mutable kernel::RopeCache rope_k;
};

struct MiniLlmArchitecture {
    ModelConfig config{};
    size_t default_num_layers = 0;
};

[[nodiscard]] DecoderLayerWeights make_decoder_layer_weights(const ModelConfig& cfg, std::mt19937& rng);

class MiniLlm {
public:
    MiniLlm(ModelConfig config, ModelWeights weights);

    [[nodiscard]] static MiniLlmArchitecture architecture();
    [[nodiscard]] static MiniLlm build(size_t vocab_size, uint32_t seed, size_t num_layers = 0);

    [[nodiscard]] size_t num_layers() const noexcept;
    [[nodiscard]] size_t d_model() const noexcept;
    [[nodiscard]] const ModelConfig& config() const noexcept;
    [[nodiscard]] const ModelWeights& weights() const noexcept;
    [[nodiscard]] ModelWeights& mutable_weights() noexcept;
    void configure_cache(size_t max_seq_len, size_t page_size = 16);
    void reset_cache();
    void prefill(const std::vector<uint32_t>& token_ids, float* hidden_out);
    void decode(uint32_t token_id, float* hidden_out);
    void forward_for_training(const std::vector<uint32_t>& token_ids, std::vector<BlockForwardTape>& layer_tapes,
                              float* hidden_out) const;
    void backward(const std::vector<BlockForwardTape>& tapes,
                  const std::vector<float>& grad_hidden_out,
                  ModelWeights& grad_weights,
                  std::vector<float>& grad_hidden_in) const;

    [[nodiscard]] const std::vector<DecoderLayerState>& decoder_layers() const noexcept { return layers_; }

private:
    void ensure_cache(size_t min_seq_len);

    ModelConfig config_;
    ModelWeights weights_;
    std::vector<DecoderLayerState> layers_;
    std::unique_ptr<engine::KVCache> cache_;
    size_t cache_page_size_ = 16;
};

[[nodiscard]] std::unique_ptr<MiniLlm> load_mini_llm(const std::string& path, size_t max_seq_len = 128);

} // namespace model
