#include "engine/decode.h"

#include <gtest/gtest.h>

#include <random>

namespace {

using namespace engine;

TEST(DecodeTest, SoftmaxIsNormalized) {
    std::vector<float> x = {1.0f, 2.0f, 3.0f};
    softmax_inplace(x);
    float s = 0.0f;
    for (float v : x) {
        s += v;
        EXPECT_GE(v, 0.0f);
        EXPECT_LE(v, 1.0f);
    }
    EXPECT_NEAR(s, 1.0f, 1e-5f);
}

TEST(DecodeTest, SampleFromLogitsIsDeterministicForSameSeed) {
    std::vector<float> logits_a = {0.1f, 2.0f, -0.5f};
    std::vector<float> logits_b = {0.1f, 2.0f, -0.5f};
    std::mt19937 r1(999), r2(999);
    const uint32_t id_a = sample_from_logits(logits_a, 1.0f, r1);
    const uint32_t id_b = sample_from_logits(logits_b, 1.0f, r2);
    EXPECT_EQ(id_a, id_b);
}

TEST(DecodeTest, SampleVeryLowTemperatureMatchesArgmax) {
    std::vector<float> logits = {0.1f, 5.0f, 0.2f};
    std::vector<float> logits_saved = logits;
    std::mt19937 rng(1);
    const uint32_t sampled = sample_from_logits(logits, 1e-8f, rng);
    const uint32_t greedy = argmax_from_logits(logits_saved);
    EXPECT_EQ(sampled, greedy);
}

} // namespace
