#include "model/mini_llm.h"
#include "model/executor.h"
#include "engine/kv_cache.h"
#include "task/task.h"
#include "train/train.h"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>
#include <vector>

namespace {

using namespace model;
using namespace engine;
using namespace train;

constexpr float kGuard = -12345.0f;

TEST(ModelTest, MultiLayerForwardUpdatesAllLayerCaches) {
    ModelConfig cfg;
    cfg.d_model = 16;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 4;
    cfg.d_ff = 32;
    cfg.rope_dim = 4;
    cfg.rms_norm_eps = 1e-5f;

    std::mt19937 rng(42);
    ModelWeights model_weights;
    model_weights.layers.push_back(make_decoder_layer_weights(cfg, rng));
    model_weights.layers.push_back(make_decoder_layer_weights(cfg, rng));
    model_weights.vocab_size = task::n_vocab();
    model_weights.token_embedding.assign(model_weights.vocab_size * cfg.d_model, 0.0f);
    model_weights.lm_head.assign(model_weights.vocab_size * cfg.d_model, 0.0f);

    MiniLlm llm(cfg, std::move(model_weights));

    engine::KVCacheConfig cache_cfg;
    cache_cfg.num_layers = llm.num_layers();
    cache_cfg.num_kv_heads = cfg.num_kv_heads;
    cache_cfg.max_seq_len = 64;
    cache_cfg.head_dim = cfg.head_dim;
    cache_cfg.page_size = 8;
    engine::KVCache cache(cache_cfg);

    ForwardInput in;
    in.seq_len = 3;
    in.use_cache = true;
    in.causal = true;
    in.is_prefill = true;
    in.hidden_states.resize(in.seq_len * cfg.d_model);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (float& v : in.hidden_states) v = dist(rng);

    model::CacheBridge cache_bridge = engine::make_cache_bridge(cache);
    std::vector<float> hidden_out(in.seq_len * cfg.d_model, 0.0f);
    model::forward_model(llm.config(), llm.decoder_layers(), in, cache_bridge, hidden_out.data(), nullptr);
    EXPECT_EQ(hidden_out.size(), in.seq_len * cfg.d_model);

    for (size_t layer = 0; layer < llm.num_layers(); ++layer) {
        EXPECT_EQ(cache.seq_len(layer), in.seq_len);
    }
}

TEST(ModelPointerTest, PrefillAndTrainingForwardWriteOnlyProvidedHiddenSpan) {
    constexpr uint32_t kSeed = 31415;
    auto prefill_model = std::make_unique<model::MiniLlm>(model::MiniLlm::build(task::n_vocab(), kSeed, 1));
    auto training_model = std::make_unique<model::MiniLlm>(model::MiniLlm::build(task::n_vocab(), kSeed, 1));
    prefill_model->configure_cache(64);

    const std::vector<uint32_t> token_ids = {
        0u, 1u, 26u, 3u, 4u, 27u, 5u, 6u, 28u, 7u, 8u, 29u, 9u, 10u, 30u, 3u, 4u};
    ASSERT_TRUE(task::is_valid(token_ids));

    const size_t hidden_size = token_ids.size() * prefill_model->d_model();
    std::vector<float> prefill_guarded(hidden_size + 2u, kGuard);
    std::vector<float> training_guarded(hidden_size + 2u, kGuard);

    prefill_model->prefill(token_ids, prefill_guarded.data() + 1u);
    std::vector<model::BlockForwardTape> tapes;
    training_model->forward_for_training(token_ids, tapes, training_guarded.data() + 1u);

    EXPECT_FLOAT_EQ(prefill_guarded.front(), kGuard);
    EXPECT_FLOAT_EQ(prefill_guarded.back(), kGuard);
    EXPECT_FLOAT_EQ(training_guarded.front(), kGuard);
    EXPECT_FLOAT_EQ(training_guarded.back(), kGuard);
    ASSERT_EQ(tapes.size(), training_model->num_layers());

    for (size_t i = 0; i < hidden_size; ++i) {
        ASSERT_TRUE(std::isfinite(prefill_guarded[i + 1u]));
        ASSERT_TRUE(std::isfinite(training_guarded[i + 1u]));
        EXPECT_FLOAT_EQ(prefill_guarded[i + 1u], training_guarded[i + 1u]);
    }
}

TEST(ModelPointerTest, DecodeWritesOneHiddenRowAndMatchesFullPrefillLastRow) {
    constexpr uint32_t kSeed = 27182;
    auto decode_model = std::make_unique<model::MiniLlm>(model::MiniLlm::build(task::n_vocab(), kSeed, 1));
    auto full_model = std::make_unique<model::MiniLlm>(model::MiniLlm::build(task::n_vocab(), kSeed, 1));
    decode_model->configure_cache(64);
    full_model->configure_cache(64);

    const std::vector<uint32_t> prompt = {
        0u, 1u, 26u, 3u, 4u, 27u, 5u, 6u, 28u, 7u, 8u, 29u, 9u, 10u, 30u};
    const uint32_t next_token = 3u;
    std::vector<uint32_t> full_sequence = prompt;
    full_sequence.push_back(next_token);

    const size_t d_model = decode_model->d_model();
    std::vector<float> prompt_hidden(prompt.size() * d_model, 0.0f);
    std::vector<float> decode_guarded(d_model + 2u, kGuard);
    std::vector<float> full_guarded(full_sequence.size() * d_model + 2u, kGuard);

    decode_model->prefill(prompt, prompt_hidden.data());
    decode_model->decode(next_token, decode_guarded.data() + 1u);
    full_model->prefill(full_sequence, full_guarded.data() + 1u);

    EXPECT_FLOAT_EQ(decode_guarded.front(), kGuard);
    EXPECT_FLOAT_EQ(decode_guarded.back(), kGuard);
    EXPECT_FLOAT_EQ(full_guarded.front(), kGuard);
    EXPECT_FLOAT_EQ(full_guarded.back(), kGuard);

    const size_t last_row = (full_sequence.size() - 1u) * d_model;
    for (size_t d = 0; d < d_model; ++d) {
        ASSERT_TRUE(std::isfinite(decode_guarded[d + 1u]));
        ASSERT_TRUE(std::isfinite(full_guarded[last_row + d + 1u]));
        EXPECT_NEAR(decode_guarded[d + 1u], full_guarded[last_row + d + 1u], 1e-5f);
    }
}

TEST(ModelTrainTest, BatchMetricsReturnsLossAndAccuracy) {
    ModelConfig cfg;
    cfg.d_model = 8;
    cfg.num_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.head_dim = 4;
    cfg.d_ff = 16;
    cfg.rope_dim = 4;
    cfg.rms_norm_eps = 1e-5f;

    std::mt19937 rng(123);
    ModelWeights model_weights;
    model_weights.layers.push_back(make_decoder_layer_weights(cfg, rng));
    model_weights.vocab_size = task::n_vocab();
    model_weights.token_embedding.resize(model_weights.vocab_size * cfg.d_model);
    model_weights.lm_head.resize(model_weights.vocab_size * cfg.d_model);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (float& v : model_weights.token_embedding) v = dist(rng);
    for (float& v : model_weights.lm_head) v = dist(rng);

    auto m = std::make_unique<model::MiniLlm>(cfg, model_weights);

    const std::vector<uint32_t> token_ids = {
        0u, 1u, 26u, 3u, 4u, 27u, 5u, 6u, 28u, 7u, 8u, 29u, 9u, 10u, 30u, 3u, 4u};
    ASSERT_TRUE(task::is_valid(token_ids));
    const task::TrainBatchMetrics metrics = task::compute_batch_metrics(*m, token_ids);
    EXPECT_EQ(metrics.valid_steps, task::span_a_len_min());
    EXPECT_GE(metrics.loss, 0.0f);
    EXPECT_GE(metrics.accuracy, 0.0f);
    EXPECT_LE(metrics.accuracy, 1.0f);
}

TEST(ModelTrainTest, AdamWReducesLossOnFixedSample) {
    ModelConfig cfg;
    cfg.d_model = 8;
    cfg.num_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.head_dim = 4;
    cfg.d_ff = 16;
    cfg.rope_dim = 4;
    cfg.rms_norm_eps = 1e-5f;

    std::mt19937 rng(5678);
    constexpr size_t k_vocab = task::n_vocab();
    ModelWeights model_weights;
    model_weights.vocab_size = k_vocab;
    model_weights.layers.push_back(make_decoder_layer_weights(cfg, rng));
    model_weights.token_embedding.resize(model_weights.vocab_size * cfg.d_model);
    model_weights.lm_head.resize(model_weights.vocab_size * cfg.d_model);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (float& v : model_weights.token_embedding) v = dist(rng);
    for (float& v : model_weights.lm_head) v = dist(rng);

    auto m = std::make_unique<model::MiniLlm>(cfg, model_weights);
    model::ModelWeights adam_m = clone_model(m->weights());
    model::ModelWeights adam_v = clone_model(m->weights());

    const std::vector<uint32_t> token_ids = {
        0u, 1u, 26u, 3u, 4u, 27u, 5u, 6u, 28u, 7u, 8u, 29u, 9u, 10u, 30u, 3u, 4u};
    ASSERT_TRUE(task::is_valid(token_ids));

    const float loss_before = task::compute_batch_metrics(*m, token_ids).loss;
    constexpr float k_beta1 = 0.9f;
    constexpr float k_beta2 = 0.95f;
    constexpr float k_eps = 1e-8f;
    const std::vector<size_t> prediction_steps = task::answer_prediction_steps(token_ids);
    TrainWorkspace workspace{};
    for (size_t i = 0; i < 40; ++i) {
        (void)train_step(*m, token_ids, prediction_steps, i + 1, 0.02f, k_beta1, k_beta2, k_eps, 0.02f, 1.0f,
                         adam_m, adam_v, workspace);
    }
    const float loss_after = task::compute_batch_metrics(*m, token_ids).loss;
    EXPECT_LT(loss_after, loss_before);
}

} // namespace
