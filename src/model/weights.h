#pragma once

#include "core/allocator.h"
#include "core/tensor.h"
#include "model/config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

namespace io {
class Checkpoint;
}

namespace model {

// Linear weights are [out, in]. All tensors are bf16 except GatedDeltaNet's
// a_log and norm, which the checkpoint stores as f32.
struct FullAttentionWeights {
    core::Tensor q_proj; // [num_heads * 2 * head_dim, hidden], per head: query | gate
    core::Tensor k_proj;
    core::Tensor v_proj;
    core::Tensor o_proj;
    core::Tensor q_norm; // [head_dim]
    core::Tensor k_norm;
};

struct GatedDeltaNetWeights {
    core::Tensor in_proj_qkv; // [conv_dim, hidden]
    core::Tensor in_proj_z;   // [value_dim, hidden]
    core::Tensor in_proj_a;   // [num_v_heads, hidden]
    core::Tensor in_proj_b;
    core::Tensor conv1d;  // [conv_dim, 1, conv_kernel]
    core::Tensor dt_bias; // [num_v_heads]
    core::Tensor a_log;   // [num_v_heads], f32
    core::Tensor norm;    // [v_head_dim], f32
    core::Tensor out_proj;
};

struct MlpWeights {
    core::Tensor gate_proj;
    core::Tensor up_proj;
    core::Tensor down_proj;
};

struct LayerWeights {
    core::Tensor input_norm;
    core::Tensor post_attention_norm;
    std::variant<GatedDeltaNetWeights, FullAttentionWeights> mixer;
    MlpWeights mlp;
};

enum class Part : std::uint8_t { embedding, norm, full_attention, gated_delta_net, mlp };
inline constexpr std::size_t kNumParts = 5;

struct LoadReport {
    std::array<std::size_t, kNumParts> params_by_part{};
    std::size_t tensors = 0;
    std::size_t params = 0;
    std::size_t bytes = 0;
    std::size_t skipped_tensors = 0; // vision encoder and MTP head
    std::size_t skipped_params = 0;
    double seconds = 0.0;
};

// The text model's weights in one device arena. Move-only.
class Weights {
public:
    static Weights load(const Config& config, const io::Checkpoint& checkpoint);

    const core::Tensor& embed_tokens() const { return embed_tokens_; }
    const core::Tensor& output_projection() const { return lm_head_.numel() != 0 ? lm_head_ : embed_tokens_; }
    const core::Tensor& final_norm() const { return final_norm_; }
    const std::vector<LayerWeights>& layers() const { return layers_; }
    const LoadReport& report() const { return report_; }

private:
    Weights() = default;

    std::unique_ptr<core::Arena> arena_; // declared first: outlives the tensors
    core::Tensor embed_tokens_;
    core::Tensor lm_head_; // empty when tied to embed_tokens
    core::Tensor final_norm_;
    std::vector<LayerWeights> layers_;
    LoadReport report_;
};

} // namespace model
