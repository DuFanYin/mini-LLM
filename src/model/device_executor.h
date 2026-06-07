#pragma once

// Device-resident inference forward for the cuda backend (Phase 0b). Holds model
// weights, RoPE tables, and the KV cache in GPU memory; runs prefill/decode with a
// single H2D (input embeddings) and D2H (output hidden) per call. Host-clean header
// (only core::Tensor / std types) so mini_llm.cpp can include it.

#include "core/allocator.h"
#include "core/tensor.h"
#include "model/model_types.h"

#include <cstddef>
#include <vector>

namespace model {

class DeviceModel {
public:
    // Snapshots `weights` onto the device; sized for up to max_seq_len total tokens.
    DeviceModel(const ModelConfig& config, const ModelWeights& weights, std::size_t max_seq_len);

    void reset() { current_len_ = 0; } // clear KV context (new request)

    // Run `seq_len` tokens whose embeddings are in hidden_in (host, [seq_len, d_model]).
    // Appends to the KV cache at the current context length, writes hidden_out (host,
    // [seq_len, d_model]), and advances the context length.
    void forward(const float* hidden_in, std::size_t seq_len, float* hidden_out);

    std::size_t current_len() const { return current_len_; }
    std::size_t max_seq_len() const { return max_seq_len_; }

private:
    struct DeviceLinear {
        core::Tensor weight; // [out_dim, in_dim]
        core::Tensor bias;   // [out_dim] (empty if absent)
        bool has_bias = false;
    };
    struct DeviceLayer {
        core::Tensor norm1; // [d_model]
        DeviceLinear q, k, v, o;
        core::Tensor norm2; // [d_model]
        DeviceLinear gate, up, down;
    };

    DeviceLinear upload_linear(const LinearWeights& src);

    ModelConfig config_;
    std::size_t max_seq_len_ = 0;
    std::size_t current_len_ = 0;
    std::size_t rot_ = 0;
    std::size_t half_ = 0;
    std::size_t eff_rot_ = 0;

    core::PoolAllocator alloc_;
    std::vector<DeviceLayer> layers_;
    std::vector<core::Tensor> k_cache_; // per layer, head-major [num_kv_heads, max_seq_len, head_dim]
    std::vector<core::Tensor> v_cache_;
    core::Tensor cos_; // [max_seq_len, half_]
    core::Tensor sin_;
};

} // namespace model
