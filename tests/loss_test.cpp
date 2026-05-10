#include "engine/decode.h"

#include <gtest/gtest.h>

#include <cmath>
#include <span>
#include <vector>

namespace {

using namespace engine;

TEST(LossTest, ProjectLogitsComputesRowWiseDotProducts) {
    const size_t seq_len = 2;
    const size_t d_model = 2;
    const size_t vocab_size = 3;

    // hidden:
    // row0 = [1, 2]
    // row1 = [3, 4]
    std::vector<float> hidden = {1.0f, 2.0f, 3.0f, 4.0f};

    // lm_head rows:
    // v0 = [1, 0]
    // v1 = [0, 1]
    // v2 = [1, 1]
    std::vector<float> lm_head = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};

    LogitsOutput logits;
    project_logits_into(hidden, seq_len, d_model, lm_head, vocab_size, logits);
    ASSERT_EQ(logits.logits.size(), seq_len * vocab_size);
    EXPECT_FLOAT_EQ(logits.logits[0], 1.0f);
    EXPECT_FLOAT_EQ(logits.logits[1], 2.0f);
    EXPECT_FLOAT_EQ(logits.logits[2], 3.0f);
    EXPECT_FLOAT_EQ(logits.logits[3], 3.0f);
    EXPECT_FLOAT_EQ(logits.logits[4], 4.0f);
    EXPECT_FLOAT_EQ(logits.logits[5], 7.0f);
}

TEST(LossTest, CrossEntropyNextTokenReturnsAverageLossAndAccuracy) {
    LogitsOutput logits;
    logits.seq_len = 3;
    logits.vocab_size = 3;
    logits.logits = {
        // step 0 predicts token_ids[1] = 1 (correct argmax=1)
        0.0f, 2.0f, 0.0f,
        // step 1 predicts token_ids[2] = 2 (incorrect argmax=0)
        3.0f, 1.0f, 0.0f,
        // step 2 ignored in next-token objective
        0.0f, 0.0f, 0.0f,
    };
    std::vector<uint32_t> token_ids = {0, 1, 2};

    CrossEntropyResult ce;
    cross_entropy_next_token_into(logits, token_ids, ce);
    EXPECT_EQ(ce.valid_steps, 2u);
    ASSERT_EQ(ce.steps.size(), 2u);
    EXPECT_EQ(ce.steps[0], 0u);
    EXPECT_EQ(ce.steps[1], 1u);
    EXPECT_FLOAT_EQ(ce.accuracy, 0.5f);

    const float expected_step0 = -std::log(std::exp(2.0f) / (std::exp(0.0f) + std::exp(2.0f) + std::exp(0.0f)));
    const float expected_step1 = -std::log(std::exp(0.0f) / (std::exp(3.0f) + std::exp(1.0f) + std::exp(0.0f)));
    const float expected_avg = 0.5f * (expected_step0 + expected_step1);
    EXPECT_NEAR(ce.loss, expected_avg, 1e-6f);
}

TEST(LossTest, CrossEntropyStepsUsesSelectedPredictionRows) {
    LogitsOutput logits;
    logits.seq_len = 5;
    logits.vocab_size = 3;
    logits.logits = {
        // ignored
        10.0f, 0.0f, 0.0f,
        // step 1 predicts token_ids[2] = 2
        0.0f, 0.0f, 2.0f,
        // ignored
        10.0f, 0.0f, 0.0f,
        // step 3 predicts token_ids[4] = 1
        0.0f, 3.0f, 0.0f,
        // ignored
        10.0f, 0.0f, 0.0f,
    };
    const std::vector<uint32_t> token_ids = {0, 1, 2, 0, 1};
    const std::vector<size_t> steps = {1u, 3u};

    CrossEntropyResult ce;
    cross_entropy_steps_into(logits, token_ids, steps, ce);

    EXPECT_EQ(ce.valid_steps, 2u);
    EXPECT_EQ(ce.steps, steps);
    ASSERT_EQ(ce.targets.size(), 2u);
    EXPECT_EQ(ce.targets[0], 2u);
    EXPECT_EQ(ce.targets[1], 1u);
    EXPECT_FLOAT_EQ(ce.accuracy, 1.0f);

    const float expected_step1 = -std::log(std::exp(2.0f) / (std::exp(0.0f) + std::exp(0.0f) + std::exp(2.0f)));
    const float expected_step3 = -std::log(std::exp(3.0f) / (std::exp(0.0f) + std::exp(3.0f) + std::exp(0.0f)));
    EXPECT_NEAR(ce.loss, 0.5f * (expected_step1 + expected_step3), 1e-6f);
}

TEST(LossTest, ProjectLogitsStepsPackedMatchesDenseForSameSteps) {
    const size_t seq_len = 4;
    const size_t d_model = 2;
    const size_t vocab_size = 3;
    const std::vector<float> hidden = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const std::vector<float> lm_head = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    const std::vector<uint32_t> token_ids = {0, 1, 2, 0};
    const std::vector<size_t> steps = {0u, 2u};

    LogitsOutput dense;
    project_logits_into(hidden, seq_len, d_model, lm_head, vocab_size, dense);
    ASSERT_FALSE(dense.packed_prediction_rows);

    std::vector<float> gathered;
    LogitsOutput packed;
    project_logits_steps_into(hidden.data(), seq_len, d_model, std::span<const size_t>(steps.data(), steps.size()),
                              lm_head, vocab_size, packed, gathered);
    ASSERT_TRUE(packed.packed_prediction_rows);
    ASSERT_EQ(packed.logits.size(), steps.size() * vocab_size);
    for (size_t i = 0; i < steps.size(); ++i) {
        for (size_t v = 0; v < vocab_size; ++v) {
            EXPECT_FLOAT_EQ(packed.logits[i * vocab_size + v], dense.logits[steps[i] * vocab_size + v]);
        }
    }

    CrossEntropyResult ce_dense;
    cross_entropy_steps_into(dense, token_ids, steps, ce_dense);
    CrossEntropyResult ce_packed;
    cross_entropy_steps_into(packed, token_ids, steps, ce_packed);
    EXPECT_FLOAT_EQ(ce_dense.loss, ce_packed.loss);
    EXPECT_FLOAT_EQ(ce_dense.accuracy, ce_packed.accuracy);
}

} // namespace

