#include "kernel/kernel.h"
#include "engine/kv_cache.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using namespace model;

constexpr float kGuard = -12345.0f;

TEST(KernelPointerTest, LinearWritesOnlyOutputSpan) {
    const std::vector<float> x = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    const std::vector<float> w = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
    };
    const std::vector<float> bias = {0.5f, -0.5f, 1.0f};
    std::vector<float> y(2u * 3u + 2u, kGuard);

    kernel::linear(x.data(), w.data(), bias.data(), y.data() + 1u, kernel::LinearParams{2u, 2u, 3u});

    EXPECT_FLOAT_EQ(y.front(), kGuard);
    EXPECT_FLOAT_EQ(y.back(), kGuard);
    EXPECT_FLOAT_EQ(y[1], 1.5f);
    EXPECT_FLOAT_EQ(y[2], 1.5f);
    EXPECT_FLOAT_EQ(y[3], 4.0f);
    EXPECT_FLOAT_EQ(y[4], 3.5f);
    EXPECT_FLOAT_EQ(y[5], 3.5f);
    EXPECT_FLOAT_EQ(y[6], 8.0f);
}

TEST(KernelPointerTest, LinearBackwardAccumulatesDxGradWeightAndBias) {
    const std::vector<float> x = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    const std::vector<float> w = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
    };
    const std::vector<float> dy = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };
    std::vector<float> dx(4, 0.5f);
    std::vector<float> grad_w(6, 1.0f);
    std::vector<float> grad_b(3, -1.0f);

    kernel::linear_backward(x.data(), w.data(), true, dy.data(), dx.data(), grad_w.data(), grad_b.data(),
                            kernel::LinearParams{2u, 2u, 3u});

    EXPECT_FLOAT_EQ(dx[0], 4.5f);
    EXPECT_FLOAT_EQ(dx[1], 5.5f);
    EXPECT_FLOAT_EQ(dx[2], 10.5f);
    EXPECT_FLOAT_EQ(dx[3], 11.5f);

    EXPECT_FLOAT_EQ(grad_w[0], 14.0f);
    EXPECT_FLOAT_EQ(grad_w[1], 19.0f);
    EXPECT_FLOAT_EQ(grad_w[2], 18.0f);
    EXPECT_FLOAT_EQ(grad_w[3], 25.0f);
    EXPECT_FLOAT_EQ(grad_w[4], 22.0f);
    EXPECT_FLOAT_EQ(grad_w[5], 31.0f);

    EXPECT_FLOAT_EQ(grad_b[0], 4.0f);
    EXPECT_FLOAT_EQ(grad_b[1], 6.0f);
    EXPECT_FLOAT_EQ(grad_b[2], 8.0f);
}

TEST(AttentionOpsTest, CausalMaskBlocksFutureTokens) {
    // [S=2, H=1, D=1]
    std::vector<float> q = {1.0f, 1.0f};
    std::vector<float> k = {1.0f, 1.0f};
    std::vector<float> v = {1.0f, 3.0f};

    kernel::AttentionParams attention_params;
    attention_params.seq_len = 2;
    attention_params.num_heads = 1;
    attention_params.num_kv_heads = 1;
    attention_params.head_dim = 1;
    attention_params.past_len = 0;
    attention_params.total_kv_len = 2;
    attention_params.causal = true;
    attention_params.use_cache = false;

    std::vector<float> ctx(2, 0.0f);
    kernel::gqa_attention_prefill(q.data(), k.data(), v.data(), nullptr, ctx.data(), nullptr, attention_params);
    ASSERT_EQ(ctx.size(), 2u);
    EXPECT_NEAR(ctx[0], 1.0f, 1e-5f); // first token can only see token 0
    EXPECT_NEAR(ctx[1], 2.0f, 1e-5f); // second token sees both, equal logits => mean
}

TEST(AttentionOpsTest, GQAMappingUsesGroupedKvHeads) {
    // [S=1,Hq=4,D=1]
    std::vector<float> q = {1, 1, 1, 1};
    // [T=2,Hkv=2,D=1], all keys equal => equal probs over T
    std::vector<float> k = {1, 1, 1, 1};
    // token0: kv0=10, kv1=20; token1: kv0=30, kv1=40
    std::vector<float> v = {10, 20, 30, 40};

    kernel::AttentionParams attention_params;
    attention_params.seq_len = 1;
    attention_params.num_heads = 4;
    attention_params.num_kv_heads = 2;
    attention_params.head_dim = 1;
    attention_params.past_len = 0;
    attention_params.total_kv_len = 2;
    attention_params.causal = false;
    attention_params.use_cache = false;

    std::vector<float> ctx(4, 0.0f);
    kernel::gqa_attention_prefill(q.data(), k.data(), v.data(), nullptr, ctx.data(), nullptr, attention_params);
    ASSERT_EQ(ctx.size(), 4u);
    // group_size = 2 => qh 0,1 -> kv0 mean(10,30)=20; qh 2,3 -> kv1 mean(20,40)=30
    EXPECT_NEAR(ctx[0], 20.0f, 1e-5f);
    EXPECT_NEAR(ctx[1], 20.0f, 1e-5f);
    EXPECT_NEAR(ctx[2], 30.0f, 1e-5f);
    EXPECT_NEAR(ctx[3], 30.0f, 1e-5f);
}

TEST(RopeTest, PositionChangesRotation) {
    kernel::RopeCache rc(10000.0f, 4, 4);
    std::vector<float> base = {1.0f, 0.0f, 1.0f, 0.0f}; // [S=1,H=1,D=4]

    std::vector<float> pos0 = base;
    std::vector<float> pos5 = base;
    std::vector<float> pos5_b = base;

    const kernel::RopeParams rope_params{1, 1, 4};
    const std::vector<size_t> p0{0};
    const std::vector<size_t> p5{5};
    kernel::apply_rope(pos0.data(), p0.data(), rc, rope_params);
    kernel::apply_rope(pos5.data(), p5.data(), rc, rope_params);
    kernel::apply_rope(pos5_b.data(), p5.data(), rc, rope_params);

    // position 0 keeps the pair unchanged in this setup
    EXPECT_NEAR(pos0[0], 1.0f, 1e-6f);
    EXPECT_NEAR(pos0[1], 0.0f, 1e-6f);

    // same position gives same rotation
    for (size_t i = 0; i < pos5.size(); ++i) {
        EXPECT_NEAR(pos5[i], pos5_b[i], 1e-6f);
    }

    // different position should differ
    bool any_diff = false;
    for (size_t i = 0; i < pos5.size(); ++i) {
        if (std::fabs(pos5[i] - pos0[i]) > 1e-6f) {
            any_diff = true;
            break;
        }
    }
    EXPECT_TRUE(any_diff);
}

TEST(KVCacheTest, AppendPreservesOrderAcrossPages) {
    engine::KVCacheConfig cfg;
    cfg.num_layers = 1;
    cfg.num_kv_heads = 1;
    cfg.max_seq_len = 8;
    cfg.head_dim = 2;
    cfg.page_size = 2;
    engine::KVCache cache(cfg);

    cache.append(0, std::vector<float>{1, 2}, std::vector<float>{11, 12}, 1);
    cache.append(0, std::vector<float>{3, 4, 5, 6}, std::vector<float>{13, 14, 15, 16}, 2);

    EXPECT_EQ(cache.seq_len(0), 3u);
    EXPECT_NEAR(cache.k_at(0, 0, 0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(cache.k_at(0, 0, 0, 1), 2.0f, 1e-6f);
    EXPECT_NEAR(cache.k_at(0, 0, 1, 0), 3.0f, 1e-6f);
    EXPECT_NEAR(cache.k_at(0, 0, 1, 1), 4.0f, 1e-6f);
    EXPECT_NEAR(cache.k_at(0, 0, 2, 0), 5.0f, 1e-6f);
    EXPECT_NEAR(cache.k_at(0, 0, 2, 1), 6.0f, 1e-6f);
}

TEST(KVCacheTest, AppendRawPreservesOrderAcrossPages) {
    engine::KVCacheConfig cfg;
    cfg.num_layers = 1;
    cfg.num_kv_heads = 2;
    cfg.max_seq_len = 4;
    cfg.head_dim = 2;
    cfg.page_size = 2;
    engine::KVCache cache(cfg);

    const std::vector<float> k = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
    };
    const std::vector<float> v = {
        101.0f, 102.0f, 103.0f, 104.0f,
        105.0f, 106.0f, 107.0f, 108.0f,
        109.0f, 110.0f, 111.0f, 112.0f,
    };

    cache.append_raw(0, k.data(), v.data(), 3);

    EXPECT_EQ(cache.seq_len(0), 3u);
    EXPECT_NEAR(cache.k_at(0, 0, 0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(cache.k_at(0, 1, 0, 1), 4.0f, 1e-6f);
    EXPECT_NEAR(cache.k_at(0, 0, 1, 0), 5.0f, 1e-6f);
    EXPECT_NEAR(cache.k_at(0, 1, 1, 1), 8.0f, 1e-6f);
    EXPECT_NEAR(cache.k_at(0, 0, 2, 0), 9.0f, 1e-6f);
    EXPECT_NEAR(cache.k_at(0, 1, 2, 1), 12.0f, 1e-6f);
    EXPECT_NEAR(cache.v_at(0, 0, 0, 0), 101.0f, 1e-6f);
    EXPECT_NEAR(cache.v_at(0, 1, 2, 1), 112.0f, 1e-6f);
}

} // namespace
