#pragma once

#include "core/allocator.h"
#include "core/stream.h"
#include "core/tensor.h"
#include "model/config.h"
#include "model/weights.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

namespace runtime {

// Receives a host copy of an intermediate [rows, cols] activation (tests and debugging).
using Trace =
    std::function<void(std::string_view name, std::span<const float> data, std::size_t rows, std::size_t cols)>;

// Runs one sequence through the model on the GPU. Prefill and decode are the
// same call: forward() appends tokens to the sequence state (KV cache for the
// attention layers, conv rows and recurrent state for the GatedDeltaNet layers).
class Executor {
public:
    struct Options {
        std::size_t max_seq = 4096;  // tokens of context the KV cache holds
        std::size_t max_chunk = 256; // tokens per forward pass; longer inputs are split
    };

    Executor(const model::Config& config, const model::Weights& weights, Options options);

    // Appends `ids` and returns the greedy next token. When given, `last_logits`
    // receives the logits at the last position and `trace` every layer's output.
    std::int32_t forward(std::span<const std::int32_t> ids, std::vector<float>* last_logits = nullptr,
                         const Trace* trace = nullptr);

    void reset();
    std::size_t position() const { return position_; }

private:
    struct LayerState {
        core::Tensor k_cache; // attention: [kv_heads, max_seq, head_dim]
        core::Tensor v_cache;
        core::Tensor conv; // GatedDeltaNet: [conv_kernel - 1, conv_dim]
        core::Tensor s;    // GatedDeltaNet: [v_heads, k_dim, v_dim]
    };

    std::int32_t run_chunk(std::span<const std::int32_t> ids, std::vector<float>* last_logits, const Trace* trace);
    void attention_layer(const model::FullAttentionWeights& w, LayerState& st, std::size_t tokens);
    void delta_net_layer(const model::GatedDeltaNetWeights& w, LayerState& st, std::size_t tokens);
    void emit(const Trace* trace, std::string_view name, const core::Tensor& t, std::size_t rows,
              std::size_t cols) const;

    const model::Config& config_;
    const model::Weights& weights_;
    Options options_;
    core::Stream stream_;
    core::PoolAllocator alloc_; // declared before every tensor below
    std::vector<LayerState> layers_;
    std::size_t position_ = 0;

    // Scratch, sized for options_.max_chunk tokens.
    core::Tensor ids_, next_;
    core::Tensor hidden_, normed_, mixed_;
    core::Tensor qg_, k_, v_, attn_;                        // attention
    core::Tensor qkv_, conv_, z_, a_, b_, g_, beta_, core_; // GatedDeltaNet
    core::Tensor gate_, up_;                                // MLP
    core::Tensor logits_;
};

} // namespace runtime
