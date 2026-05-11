#include "engine/io.h"
#include "model/mini_llm.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

namespace {

using namespace model;
using namespace engine;

TEST(IoTest, SaveAndLoadRoundTrip) {
    MiniLlm model = MiniLlm::init_random(26, 42u);
    const ModelWeights& w = model.weights();
    const ModelConfig& cfg = model.config();

    const auto path =
        std::filesystem::temp_directory_path() / "mini_transformer_io_test_v1.ckpt";
    save_model(path.string(), cfg, w);
    const SavedModel loaded = load_model(path.string());

    EXPECT_EQ(loaded.config.d_model, cfg.d_model);
    EXPECT_EQ(loaded.config.num_heads, cfg.num_heads);
    EXPECT_EQ(loaded.config.num_kv_heads, cfg.num_kv_heads);
    EXPECT_EQ(loaded.config.num_layers, cfg.num_layers);
    EXPECT_EQ(loaded.weights.vocab_size, w.vocab_size);
    EXPECT_EQ(loaded.weights.layers.size(), w.layers.size());
    EXPECT_EQ(loaded.weights.token_embedding.size(), w.token_embedding.size());
    EXPECT_EQ(loaded.weights.output_projection.size(), w.output_projection.size());
    EXPECT_FLOAT_EQ(loaded.weights.token_embedding[0], w.token_embedding[0]);
    EXPECT_FLOAT_EQ(loaded.weights.output_projection[0], w.output_projection[0]);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

} // namespace
