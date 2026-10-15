#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace io {
class Json;
}

namespace model {

enum class LayerKind : std::uint8_t { gated_delta_net, full_attention };

struct FullAttentionConfig {
    std::size_t num_heads = 0;
    std::size_t num_kv_heads = 0;
    std::size_t head_dim = 0;
    std::size_t rotary_dim = 0; // leading dims of each head that RoPE rotates
    double rope_theta = 0.0;
};

struct GatedDeltaNetConfig {
    std::size_t num_k_heads = 0;
    std::size_t num_v_heads = 0;
    std::size_t k_head_dim = 0;
    std::size_t v_head_dim = 0;
    std::size_t conv_kernel = 0;

    std::size_t key_dim() const { return num_k_heads * k_head_dim; }
    std::size_t value_dim() const { return num_v_heads * v_head_dim; }
    std::size_t conv_dim() const { return 2 * key_dim() + value_dim(); } // q | k | v channels
};

// Text-model configuration of a Qwen3.5 checkpoint. from_json validates every
// field the engine relies on and throws std::runtime_error otherwise.
struct Config {
    std::size_t hidden_size = 0;
    std::size_t intermediate_size = 0;
    std::size_t vocab_size = 0;
    double rms_norm_eps = 0.0;
    bool tie_word_embeddings = true;
    std::int64_t eos_token_id = -1;
    std::vector<LayerKind> layers;
    FullAttentionConfig attn;
    GatedDeltaNetConfig gdn;

    std::size_t num_layers() const { return layers.size(); }
    std::size_t count(LayerKind kind) const;

    static Config from_json(const io::Json& root); // the whole config.json
    static Config load(const std::filesystem::path& checkpoint_dir);
};

} // namespace model
