#include "task/task.h"

#include <gtest/gtest.h>

#include <vector>

namespace {

using namespace task;

TEST(DataTest, WhichSpanTokenIds) {
    EXPECT_EQ(tok_sa(), 26u);
    EXPECT_EQ(tok_ea(), 27u);
    EXPECT_EQ(tok_sb(), 28u);
    EXPECT_EQ(tok_eb(), 29u);
    EXPECT_EQ(tok_qa(), 30u);
    EXPECT_EQ(tok_qb(), 31u);
    EXPECT_EQ(prefix_len_min(), 2u);
    EXPECT_EQ(prefix_len_max(), 10u);
    EXPECT_EQ(middle_len_min(), 2u);
    EXPECT_EQ(middle_len_max(), 10u);
    EXPECT_EQ(suffix_len_min(), 2u);
    EXPECT_EQ(suffix_len_max(), 10u);
    EXPECT_EQ(span_a_len_min(), 2u);
    EXPECT_EQ(span_a_len_max(), 8u);
    EXPECT_EQ(span_b_len_min(), 2u);
    EXPECT_EQ(span_b_len_max(), 8u);
    EXPECT_EQ(token_len_min(), 17u);
    EXPECT_EQ(token_len_max(), 59u);
    EXPECT_EQ(to_char(2u), 'C');
}

TEST(DataTest, SamplerOutput) {
    Sampler sampler(999);
    for (int i = 0; i < 50; ++i) {
        const auto seq = sampler.sample_sequence();
        EXPECT_TRUE(is_valid(seq));
        EXPECT_GE(seq.size(), token_len_min());
        EXPECT_LE(seq.size(), token_len_max());
    }
}

TEST(DataTest, FixedBatchRepeatable) {
    const auto a = batch_at_seed(64, 12345);
    const auto b = batch_at_seed(64, 12345);
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i], b[i]);
    }
}

TEST(DataTest, InvalidSeq) {
    EXPECT_FALSE(is_valid({}));
    EXPECT_FALSE(is_valid(std::vector<uint32_t>(token_len_min(), 0)));
    // Structure parses but gold tail does not match queried span.
    std::vector<uint32_t> bad = {
        0u, 1u, 26u, 3u, 4u, 27u, 5u, 6u, 28u, 7u, 8u, 29u, 9u, 10u, 30u, 3u, 5u};
    EXPECT_FALSE(is_valid(bad));
}

} // namespace
