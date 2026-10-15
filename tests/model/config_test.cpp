#include "model/config.h"

#include "io/json.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

namespace {

// text_config of Qwen/Qwen3.5-0.8B (vision_config omitted).
constexpr const char* kConfig = R"({
  "model_type": "qwen3_5",
  "text_config": {
    "attention_bias": false, "attn_output_gate": true, "eos_token_id": 248044, "head_dim": 256,
    "hidden_act": "silu", "hidden_size": 1024, "intermediate_size": 3584,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
    "linear_num_value_heads": 16, "linear_value_head_dim": 128, "model_type": "qwen3_5_text",
    "num_attention_heads": 8, "num_hidden_layers": 24, "num_key_value_heads": 2, "rms_norm_eps": 1e-06,
    "tie_word_embeddings": true, "vocab_size": 248320,
    "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "default",
                        "rope_theta": 10000000, "partial_rotary_factor": 0.25}
  }
})";

model::Config parse(const std::string& text) {
    return model::Config::from_json(io::Json::parse(text));
}

std::string with(const std::string& from, const std::string& to) {
    std::string s = kConfig;
    const auto pos = s.find(from);
    EXPECT_NE(pos, std::string::npos) << from;
    return s.replace(pos, from.size(), to);
}

TEST(Config, ParsesQwen35TextConfig) {
    const model::Config c = parse(kConfig);
    EXPECT_EQ(c.hidden_size, 1024u);
    EXPECT_EQ(c.intermediate_size, 3584u);
    EXPECT_EQ(c.vocab_size, 248320u);
    EXPECT_TRUE(c.tie_word_embeddings);
    EXPECT_EQ(c.eos_token_id, 248044);
    ASSERT_EQ(c.num_layers(), 24u);
    EXPECT_EQ(c.count(model::LayerKind::full_attention), 6u);
    for (std::size_t i = 0; i < c.num_layers(); ++i) {
        const auto want = i % 4 == 3 ? model::LayerKind::full_attention : model::LayerKind::gated_delta_net;
        EXPECT_EQ(c.layers[i], want) << "layer " << i;
    }
}

TEST(Config, DerivesAttentionAndDeltaNetDims) {
    const model::Config c = parse(kConfig);
    EXPECT_EQ(c.attn.num_heads, 8u);
    EXPECT_EQ(c.attn.num_kv_heads, 2u);
    EXPECT_EQ(c.attn.head_dim, 256u);
    EXPECT_EQ(c.attn.rotary_dim, 64u);
    EXPECT_DOUBLE_EQ(c.attn.rope_theta, 1e7);
    EXPECT_EQ(c.gdn.key_dim(), 2048u);
    EXPECT_EQ(c.gdn.value_dim(), 2048u);
    EXPECT_EQ(c.gdn.conv_dim(), 6144u);
    EXPECT_EQ(c.gdn.conv_kernel, 4u);
}

TEST(Config, RejectsUnsupportedVariants) {
    const std::pair<const char*, const char*> cases[] = {
        {R"("num_hidden_layers": 24)", R"("num_hidden_layers": 25)"},
        {R"("attn_output_gate": true)", R"("attn_output_gate": false)"},
        {R"("attention_bias": false)", R"("attention_bias": true)"},
        {R"("rope_type": "default")", R"("rope_type": "yarn")"},
        {R"("mrope_section": [11, 11, 10])", R"("mrope_section": [11, 11, 11])"},
        {R"("num_key_value_heads": 2)", R"("num_key_value_heads": 3)"},
        {R"("hidden_size": 1024)", R"("hidden_size": 0)"},
        {R"("model_type": "qwen3_5_text")", R"("model_type": "qwen3")"},
        {R"("linear_attention", "linear_attention", "linear_attention", "full_attention",)",
         R"("linear_attention", "mamba", "linear_attention", "full_attention",)"},
    };
    for (const auto& [from, to] : cases) {
        EXPECT_THROW(parse(with(from, to)), std::runtime_error) << to;
    }
}

} // namespace
