#include "engine/io.h"
#include "model/mini_llm.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <random>
#include <string>

namespace {

using namespace model;
using namespace engine;

TEST(IoTest, SaveAndLoadRoundTrip) {
    ModelConfig cfg;
    cfg.d_model = 8;
    cfg.num_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.head_dim = 4;
    cfg.d_ff = 16;
    cfg.rope_base = 10000.0f;
    cfg.rope_dim = 4;
    cfg.rms_norm_eps = 1e-5f;

    std::mt19937 rng(42);
    ModelWeights w;
    w.vocab_size = 26;
    w.layers.push_back(make_decoder_layer_weights(cfg, rng));
    w.layers.push_back(make_decoder_layer_weights(cfg, rng));
    w.token_embedding.resize(w.vocab_size * cfg.d_model);
    w.lm_head.resize(w.vocab_size * cfg.d_model);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (float& v : w.token_embedding) v = dist(rng);
    for (float& v : w.lm_head) v = dist(rng);

    const auto path =
        std::filesystem::temp_directory_path() / "mini_transformer_io_test_v1.ckpt";
    save_model(path.string(), cfg, w);
    const SavedModel loaded = load_model(path.string());

    EXPECT_EQ(loaded.config.d_model, cfg.d_model);
    EXPECT_EQ(loaded.config.num_heads, cfg.num_heads);
    EXPECT_EQ(loaded.config.num_kv_heads, cfg.num_kv_heads);
    EXPECT_EQ(loaded.weights.vocab_size, w.vocab_size);
    EXPECT_EQ(loaded.weights.layers.size(), w.layers.size());
    EXPECT_EQ(loaded.weights.token_embedding.size(), w.token_embedding.size());
    EXPECT_EQ(loaded.weights.lm_head.size(), w.lm_head.size());
    EXPECT_FLOAT_EQ(loaded.weights.token_embedding[0], w.token_embedding[0]);
    EXPECT_FLOAT_EQ(loaded.weights.lm_head[0], w.lm_head[0]);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

} // namespace
