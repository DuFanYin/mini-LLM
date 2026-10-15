// Loads the real checkpoint; skipped when it has not been downloaded.
// MINI_LLM_WEIGHTS overrides the checkpoint directory.

#include "io/safetensors.h"
#include "model/config.h"
#include "model/weights.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <variant>

namespace {

std::filesystem::path checkpoint_dir() {
    if (const char* env = std::getenv("MINI_LLM_WEIGHTS")) {
        return env;
    }
    return MINI_LLM_DEFAULT_WEIGHTS;
}

class Qwen35Weights : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(checkpoint_dir() / "config.json")) {
            GTEST_SKIP() << "no checkpoint at " << checkpoint_dir() << "; run scripts/download_weights.sh";
        }
    }
};

TEST_F(Qwen35Weights, LoadsEveryTextTensor) {
    const model::Config config = model::Config::load(checkpoint_dir());
    const io::Checkpoint ckpt(checkpoint_dir().string());
    const model::Weights w = model::Weights::load(config, ckpt);

    const model::LoadReport& r = w.report();
    EXPECT_EQ(r.tensors, 320u);
    EXPECT_EQ(r.params, 752'393'024u);
    EXPECT_EQ(r.bytes, 1'504'791'232u);
    EXPECT_EQ(r.skipped_tensors, 168u);
    EXPECT_EQ(r.skipped_params, 121'045'760u);

    ASSERT_EQ(w.layers().size(), config.num_layers());
    for (std::size_t i = 0; i < config.num_layers(); ++i) {
        const bool attention = std::holds_alternative<model::FullAttentionWeights>(w.layers()[i].mixer);
        EXPECT_EQ(attention, config.layers[i] == model::LayerKind::full_attention) << "layer " << i;
    }
    EXPECT_EQ(&w.output_projection(), &w.embed_tokens());
}

} // namespace
