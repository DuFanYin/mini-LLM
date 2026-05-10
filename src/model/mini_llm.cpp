#include "model/mini_llm.h"

#include "engine/io.h"
#include "engine/embedding.h"
#include "engine/validation.h"
#include "model/executor.h"

#include <algorithm>
#include <random>
#include <utility>

namespace model {

static LinearWeights make_linear(size_t in_dim, size_t out_dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    LinearWeights l;
    l.in_dim = in_dim;
    l.out_dim = out_dim;
    l.weight.resize(out_dim * in_dim);
    l.bias.resize(out_dim);
    for (float& v : l.weight) {
        v = dist(rng);
    }
    for (float& v : l.bias) {
        v = dist(rng);
    }
    return l;
}

DecoderLayerWeights make_decoder_layer_weights(const ModelConfig& cfg, std::mt19937& rng) {
    DecoderLayerWeights w;
    w.norm1.weight.assign(cfg.d_model, 1.0f);
    w.norm1.eps = cfg.rms_norm_eps;
    w.norm2.weight.assign(cfg.d_model, 1.0f);
    w.norm2.eps = cfg.rms_norm_eps;
    w.attention.q_proj = make_linear(cfg.d_model, cfg.num_heads * cfg.head_dim, rng);
    w.attention.k_proj = make_linear(cfg.d_model, cfg.num_kv_heads * cfg.head_dim, rng);
    w.attention.v_proj = make_linear(cfg.d_model, cfg.num_kv_heads * cfg.head_dim, rng);
    w.attention.o_proj = make_linear(cfg.num_heads * cfg.head_dim, cfg.d_model, rng);
    w.mlp.gate = make_linear(cfg.d_model, cfg.d_ff, rng);
    w.mlp.up = make_linear(cfg.d_model, cfg.d_ff, rng);
    w.mlp.down = make_linear(cfg.d_ff, cfg.d_model, rng);
    return w;
}

MiniLlm::MiniLlm(ModelConfig config, ModelWeights weights) : config_(std::move(config)), weights_(std::move(weights)) {
    engine::validate_model_config(config_);
    engine::validate_model_weights(config_, weights_);
    layers_.clear();
    layers_.reserve(weights_.layers.size());
    for (auto& layer_w : weights_.layers) {
        engine::validate_decoder_layer_weights(config_, &layer_w);
        layers_.push_back(DecoderLayerState{
            &layer_w,
            kernel::RopeCache(config_.rope_base, config_.head_dim, config_.rope_dim),
            kernel::RopeCache(config_.rope_base, config_.head_dim, config_.rope_dim),
        });
    }
}

// Trainable parameter count (weights + biases). Let V = vocab_size, L = num_layers,
// D = d_model, H = num_heads, Hkv = num_kv_heads, Dh = head_dim, F = d_ff.
//
// Per decoder layer (see make_decoder_layer_weights):
//   RMSNorm norm1+norm2:                    2*D
//   attn q_proj:                            D*(H*Dh) + H*Dh
//   attn k_proj + v_proj:                   2 * ( D*(Hkv*Dh) + Hkv*Dh )
//   attn o_proj:                            (H*Dh)*D + D
//   MLP gate + up + down (SwiGLU):          2*( D*F + F ) + ( F*D + D )
//
// With the defaults below (D=256, H=8, Hkv=4, Dh=32 => H*Dh=256, Hkv*Dh=128, F=512):
//   per_layer = 592384 trainable scalars.
//
// Embeddings (not shared with lm_head in this repo):
//   token_embedding + lm_head:              2 * V * D
//
// Totals for default L=4 (MiniLlmArchitecture::default_num_layers):
//   layers only:     4 * 592384 = 2369536  (~2.37M)
//   + I/O for V=32:  + 2*32*256 = +16384   (Which-Span task vocabulary)
//   ≈ 2385920 params (~2.39M fp32 weights ≈ 9.1MiB).
//
// Capacity note: this is a deliberate micro-model (tiny vocab, short spans). It can learn
// the conditioned copy task when training is stable; if metrics plateau early, raise L / D / F
// before chasing optimizer tweaks. KV-cache memory at inference grows with L * max_seq_len *
// (2 * Hkv * Dh) floats per layer — unrelated to param count above.

MiniLlmArchitecture MiniLlm::architecture() {
    MiniLlmArchitecture arch;
    arch.config.d_model = 256;
    arch.config.num_heads = 8;
    arch.config.num_kv_heads = 4;
    arch.config.head_dim = 32;
    arch.config.d_ff = 512;
    arch.config.rope_base = 10000.0f;
    arch.config.rope_dim = 16;
    arch.config.rms_norm_eps = 1e-5f;
    arch.default_num_layers = 4;
    return arch;
}

MiniLlm MiniLlm::build(size_t vocab_size, uint32_t seed, size_t num_layers) {
    const MiniLlmArchitecture arch = architecture();
    const size_t layer_count = (num_layers == 0) ? arch.default_num_layers : num_layers;

    std::mt19937 rng(seed);
    ModelWeights w;
    w.vocab_size = vocab_size;
    w.layers.reserve(layer_count);
    for (size_t i = 0; i < layer_count; ++i) {
        w.layers.push_back(make_decoder_layer_weights(arch.config, rng));
    }

    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    w.token_embedding.resize(vocab_size * arch.config.d_model);
    w.lm_head.resize(vocab_size * arch.config.d_model);
    for (float& v : w.token_embedding) {
        v = dist(rng);
    }
    for (float& v : w.lm_head) {
        v = dist(rng);
    }

    ModelConfig cfg = arch.config;
    return MiniLlm(std::move(cfg), std::move(w));
}

size_t MiniLlm::num_layers() const noexcept {
    return layers_.size();
}

size_t MiniLlm::d_model() const noexcept {
    return config_.d_model;
}

const ModelConfig& MiniLlm::config() const noexcept {
    return config_;
}

const ModelWeights& MiniLlm::weights() const noexcept {
    return weights_;
}

ModelWeights& MiniLlm::mutable_weights() noexcept {
    return weights_;
}

void MiniLlm::configure_cache(size_t max_seq_len, size_t page_size) {
    cache_page_size_ = page_size;
    engine::KVCacheConfig cfg;
    cfg.num_layers = num_layers();
    cfg.num_kv_heads = config().num_kv_heads;
    cfg.max_seq_len = max_seq_len;
    cfg.head_dim = config().head_dim;
    cfg.page_size = page_size;
    cache_ = std::make_unique<engine::KVCache>(cfg);
}

void MiniLlm::ensure_cache(size_t min_seq_len) {
    if (cache_ != nullptr && cache_->max_seq_len() >= min_seq_len) {
        return;
    }
    configure_cache(std::max<size_t>(128, min_seq_len), cache_page_size_);
}

void MiniLlm::reset_cache() {
    ensure_cache(1);
    cache_->reset();
}

void MiniLlm::prefill(const std::vector<uint32_t>& token_ids, float* hidden_out) {
    ensure_cache(token_ids.size());
    cache_->reset();

    ForwardInput in;
    in.seq_len = token_ids.size();
    in.use_cache = true;
    in.causal = true;
    in.is_prefill = true;
    engine::embed_token_ids_into(token_ids, weights().token_embedding, d_model(), in.hidden_states);

    CacheBridge cache_bridge = engine::make_cache_bridge(*cache_);
    model::forward_model(config_, layers_, in, cache_bridge, hidden_out, nullptr);
}

void MiniLlm::decode(uint32_t token_id, float* hidden_out) {
    ensure_cache(1);

    ForwardInput in;
    in.seq_len = 1;
    in.use_cache = true;
    in.causal = true;
    in.is_decode = true;
    const std::vector<uint32_t> token_ids{token_id};
    engine::embed_token_ids_into(token_ids, weights().token_embedding, d_model(), in.hidden_states);

    CacheBridge cache_bridge = engine::make_cache_bridge(*cache_);
    model::forward_model(config_, layers_, in, cache_bridge, hidden_out, nullptr);
}

void MiniLlm::forward_for_training(const std::vector<uint32_t>& token_ids,
                                   std::vector<BlockForwardTape>& layer_tapes,
                                   float* hidden_out) const {
    ForwardInput in;
    in.seq_len = token_ids.size();
    in.use_cache = false;
    in.causal = true;
    in.is_prefill = true;
    engine::embed_token_ids_into(token_ids, weights().token_embedding, d_model(), in.hidden_states);

    layer_tapes.resize(num_layers());
    CacheBridge cache_bridge;
    model::forward_model(config_, layers_, in, cache_bridge, hidden_out, &layer_tapes);
}

void MiniLlm::backward(const std::vector<BlockForwardTape>& tapes,
                       const std::vector<float>& grad_hidden_out,
                       ModelWeights& grad_weights,
                       std::vector<float>& grad_hidden_in) const {
    model::backward_model(config(), weights(), tapes, grad_hidden_out, grad_weights, grad_hidden_in);
}

std::unique_ptr<MiniLlm> load_mini_llm(const std::string& path, size_t max_seq_len) {
    engine::SavedModel sm = engine::load_model(path);
    auto model = std::make_unique<MiniLlm>(std::move(sm.config), std::move(sm.weights));
    model->configure_cache(max_seq_len);
    return model;
}

} // namespace model
