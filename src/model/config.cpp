#include "model/config.h"

#include "io/json.h"

#include <algorithm>
#include <format>
#include <stdexcept>
#include <string_view>

namespace model {

namespace {

[[noreturn]] void invalid(std::string_view what) {
    throw std::runtime_error(std::format("config: {}", what));
}

void require(bool ok, std::string_view what) {
    if (!ok) {
        invalid(what);
    }
}

std::size_t positive(const io::Json& obj, std::string_view key) {
    const std::int64_t v = obj[key].as_int();
    require(v > 0, std::format("{} must be positive", key));
    return static_cast<std::size_t>(v);
}

LayerKind parse_layer_kind(const std::string& s) {
    if (s == "linear_attention") {
        return LayerKind::gated_delta_net;
    }
    if (s == "full_attention") {
        return LayerKind::full_attention;
    }
    invalid(std::format("unknown layer type '{}'", s));
}

} // namespace

std::size_t Config::count(LayerKind kind) const {
    return static_cast<std::size_t>(std::ranges::count(layers, kind));
}

Config Config::from_json(const io::Json& root) {
    const io::Json& t = root["text_config"];
    require(t["model_type"].as_string() == "qwen3_5_text", "text_config.model_type must be qwen3_5_text");
    require(t["hidden_act"].as_string() == "silu", "hidden_act must be silu");
    require(!t["attention_bias"].as_bool(), "attention_bias is not supported");
    require(t["attn_output_gate"].as_bool(), "attn_output_gate must be true");

    Config c;
    c.hidden_size = positive(t, "hidden_size");
    c.intermediate_size = positive(t, "intermediate_size");
    c.vocab_size = positive(t, "vocab_size");
    c.rms_norm_eps = t["rms_norm_eps"].as_number();
    require(c.rms_norm_eps > 0.0, "rms_norm_eps must be positive");
    c.tie_word_embeddings = t["tie_word_embeddings"].as_bool();
    c.eos_token_id = t["eos_token_id"].as_int();

    for (const io::Json& kind : t["layer_types"].as_array()) {
        c.layers.push_back(parse_layer_kind(kind.as_string()));
    }
    require(c.layers.size() == positive(t, "num_hidden_layers"), "layer_types length must equal num_hidden_layers");

    FullAttentionConfig& a = c.attn;
    a.num_heads = positive(t, "num_attention_heads");
    a.num_kv_heads = positive(t, "num_key_value_heads");
    a.head_dim = positive(t, "head_dim");
    require(a.num_heads % a.num_kv_heads == 0, "num_attention_heads must be a multiple of num_key_value_heads");

    const io::Json& rope = t["rope_parameters"];
    require(rope["rope_type"].as_string() == "default", "only rope_type 'default' is supported");
    a.rope_theta = rope["rope_theta"].as_number();
    const double partial = rope.find("partial_rotary_factor") ? rope["partial_rotary_factor"].as_number() : 1.0;
    a.rotary_dim = static_cast<std::size_t>(static_cast<double>(a.head_dim) * partial);
    require(a.rotary_dim > 0 && a.rotary_dim % 2 == 0 && a.rotary_dim <= a.head_dim,
            "rotary dim must be even and within head_dim");
    // M-RoPE sections split the rotary frequencies across (t, h, w) positions.
    // Text uses one position for all three, so the sections only need to cover
    // the frequencies exactly.
    if (const io::Json* sections = rope.find("mrope_section")) {
        std::size_t freqs = 0;
        for (const io::Json& s : sections->as_array()) {
            freqs += static_cast<std::size_t>(s.as_int());
        }
        require(2 * freqs == a.rotary_dim, "mrope_section must cover rotary_dim / 2 frequencies");
    }

    GatedDeltaNetConfig& g = c.gdn;
    g.num_k_heads = positive(t, "linear_num_key_heads");
    g.num_v_heads = positive(t, "linear_num_value_heads");
    g.k_head_dim = positive(t, "linear_key_head_dim");
    g.v_head_dim = positive(t, "linear_value_head_dim");
    g.conv_kernel = positive(t, "linear_conv_kernel_dim");
    require(g.num_v_heads % g.num_k_heads == 0, "linear value heads must be a multiple of key heads");

    return c;
}

Config Config::load(const std::filesystem::path& checkpoint_dir) {
    const std::filesystem::path path = checkpoint_dir / "config.json";
    try {
        return from_json(io::Json::parse_file(path.string()));
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(std::format("{}: {}", path.string(), e.what()));
    }
}

} // namespace model
