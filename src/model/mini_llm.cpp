#include "model/mini_llm.h"

#include "engine/embedding.h"
#include "engine/io.h"
#include "model/executor.h"

#include <algorithm>
#include <random>
#include <utility>

namespace model {

namespace {

ModelConfig default_train_config() {
    ModelConfig c;
    c.d_model = 256;
    c.num_heads = 8;
    c.num_kv_heads = 4;
    c.head_dim = 32;
    c.d_ff = 512;
    c.rope_base = 10000.0f;
    c.rope_dim = 16;
    c.rms_norm_eps = 1e-5f;
    c.num_layers = 4;
    return c;
}

ModelWeights make_zero_weights(const ModelConfig& cfg, size_t vocab_size) {
    auto make_linear = [](size_t in_dim, size_t out_dim) {
        LinearWeights l;
        l.in_dim = in_dim;
        l.out_dim = out_dim;
        l.weight.assign(out_dim * in_dim, 0.0f);
        l.bias.assign(out_dim, 0.0f);
        return l;
    };

    ModelWeights w;
    w.vocab_size = vocab_size;
    w.layers.reserve(cfg.num_layers);
    for (size_t i = 0; i < cfg.num_layers; ++i) {
        DecoderLayerWeights layer;
        layer.norm1.weight.assign(cfg.d_model, 1.0f);
        layer.norm1.eps = cfg.rms_norm_eps;
        layer.norm2.weight.assign(cfg.d_model, 1.0f);
        layer.norm2.eps = cfg.rms_norm_eps;
        layer.attention.q_proj = make_linear(cfg.d_model, cfg.num_heads * cfg.head_dim);
        layer.attention.k_proj = make_linear(cfg.d_model, cfg.num_kv_heads * cfg.head_dim);
        layer.attention.v_proj = make_linear(cfg.d_model, cfg.num_kv_heads * cfg.head_dim);
        layer.attention.o_proj = make_linear(cfg.num_heads * cfg.head_dim, cfg.d_model);
        layer.mlp.gate = make_linear(cfg.d_model, cfg.d_ff);
        layer.mlp.up = make_linear(cfg.d_model, cfg.d_ff);
        layer.mlp.down = make_linear(cfg.d_ff, cfg.d_model);
        w.layers.push_back(std::move(layer));
    }
    w.token_embedding.assign(vocab_size * cfg.d_model, 0.0f);
    w.output_projection.assign(vocab_size * cfg.d_model, 0.0f);
    return w;
}

void init_random_weights(ModelWeights& weights, std::mt19937& rng) {
    auto fill_linear = [&rng](LinearWeights& l, float lo, float hi) {
        std::uniform_real_distribution<float> dist(lo, hi);
        for (float& v : l.weight) {
            v = dist(rng);
        }
        for (float& v : l.bias) {
            v = dist(rng);
        }
    };

    constexpr float k_linear_lo = -0.05f;
    constexpr float k_linear_hi = 0.05f;
    for (DecoderLayerWeights& layer : weights.layers) {
        fill_linear(layer.attention.q_proj, k_linear_lo, k_linear_hi);
        fill_linear(layer.attention.k_proj, k_linear_lo, k_linear_hi);
        fill_linear(layer.attention.v_proj, k_linear_lo, k_linear_hi);
        fill_linear(layer.attention.o_proj, k_linear_lo, k_linear_hi);
        fill_linear(layer.mlp.gate, k_linear_lo, k_linear_hi);
        fill_linear(layer.mlp.up, k_linear_lo, k_linear_hi);
        fill_linear(layer.mlp.down, k_linear_lo, k_linear_hi);
    }

    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (float& v : weights.token_embedding) {
        v = dist(rng);
    }
    for (float& v : weights.output_projection) {
        v = dist(rng);
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Default micro-model: tensor layout and param count (weights + biases).
// V = vocab_size, L = num_layers, D = d_model, H = num_heads, Hkv = num_kv_heads,
// Dh = head_dim, F = d_ff.
//
// Per decoder layer (`allocate_decoder_block`):
//   RMSNorm norm1+norm2:                    2*D
//   attn q_proj:                            D*(H*Dh) + H*Dh
//   attn k_proj + v_proj:                   2 * ( D*(Hkv*Dh) + Hkv*Dh )
//   attn o_proj:                            (H*Dh)*D + D
//   MLP gate + up + down (SwiGLU):          2*( D*F + F ) + ( F*D + D )
//
// With default train hyperparams (D=256, H=8, Hkv=4, Dh=32 => H*Dh=256, Hkv*Dh=128, F=512):
//   per_layer = 592384 trainable scalars.
//
// Embeddings (not tied to output_projection here):
//   token_embedding + output_projection:    2 * V * D
//
// Default L=4 layers:
//   layers: 4 * 592384 = 2369536; + I/O for V=32: +16384 → ~2.39M fp32 (~9.1MiB).
//
// Deliberate micro-model: if metrics plateau early, raise L / D / F before chasing optimizers.
// KV-cache memory at inference scales with L * max_seq_len * (2 * Hkv * Dh) floats/layer
// (orthogonal to param count above).
// ---------------------------------------------------------------------------

// --- Static factory ------------------------------------------------------------

MiniLlm MiniLlm::init_random(size_t vocab_size, uint32_t seed) {
    ModelConfig cfg = default_train_config();
    ModelWeights w = make_zero_weights(cfg, vocab_size);

    std::mt19937 rng(seed);
    init_random_weights(w, rng);

    return MiniLlm(std::move(cfg), std::move(w));
}

std::unique_ptr<MiniLlm> MiniLlm::init_load(const std::string& path, size_t max_seq_len) {
    engine::SavedModel sm = engine::load_model(path);
    auto model = std::make_unique<MiniLlm>(std::move(sm.config), std::move(sm.weights));
    model->configure_cache(max_seq_len);
    return model;
}

// --- Construction -------------------------------------------------------------

MiniLlm::MiniLlm(ModelConfig config, ModelWeights weights) : config_(std::move(config)), weights_(std::move(weights)) {
    if (config_.num_layers == 0) {
        config_.num_layers = weights_.layers.size();
    }
    layer_weights_.clear();
    layer_weights_.reserve(weights_.layers.size());
    rope_q_.clear();
    rope_q_.reserve(weights_.layers.size());
    rope_k_.clear();
    rope_k_.reserve(weights_.layers.size());
    for (auto& layer_w : weights_.layers) {
        layer_weights_.push_back(&layer_w);
        rope_q_.emplace_back(config_.rope_base, config_.head_dim, config_.rope_dim);
        rope_k_.emplace_back(config_.rope_base, config_.head_dim, config_.rope_dim);
    }
}

// --- Accessors ----------------------------------------------------------------

size_t MiniLlm::num_layers() const noexcept {
    return layer_weights_.size();
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

// --- KV cache -----------------------------------------------------------------

void MiniLlm::configure_cache(size_t max_seq_len, size_t page_size) {
    cache_page_size_ = page_size;
    KVCacheConfig cfg;
    cfg.num_layers = num_layers();
    cfg.num_kv_heads = config().num_kv_heads;
    cfg.max_seq_len = max_seq_len;
    cfg.head_dim = config().head_dim;
    cfg.page_size = page_size;
    cache_ = std::make_unique<KVCache>(cfg);
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

// --- Forward ------------------------------------------------------------------

void MiniLlm::prefill(const std::vector<uint32_t>& token_ids, float* hidden_out) {
    ensure_cache(token_ids.size());
    cache_->reset();

    ForwardInput in;
    in.seq_len = token_ids.size();
    in.use_cache = true;
    in.causal = true;
    in.is_prefill = true;
    engine::embed_tokens(token_ids, weights().token_embedding, d_model(), in.hidden_states);

    model::forward_model(config_, layer_weights_, rope_q_, rope_k_, in, cache_.get(), hidden_out, nullptr);
}

void MiniLlm::decode(uint32_t token_id, float* hidden_out) {
    ensure_cache(1);

    ForwardInput in;
    in.seq_len = 1;
    in.use_cache = true;
    in.causal = true;
    in.is_decode = true;
    const std::vector<uint32_t> token_ids{token_id};
    engine::embed_tokens(token_ids, weights().token_embedding, d_model(), in.hidden_states);

    model::forward_model(config_, layer_weights_, rope_q_, rope_k_, in, cache_.get(), hidden_out, nullptr);
}

void MiniLlm::forward_train(const std::vector<uint32_t>& token_ids,
                                   std::vector<BlockForwardTape>& layer_tapes,
                                   float* hidden_out) const {
    ForwardInput in;
    in.seq_len = token_ids.size();
    in.use_cache = false;
    in.causal = true;
    in.is_prefill = true;
    engine::embed_tokens(token_ids, weights().token_embedding, d_model(), in.hidden_states);

    layer_tapes.resize(num_layers());
    model::forward_model(config_, layer_weights_, rope_q_, rope_k_, in, nullptr, hidden_out, &layer_tapes);
}

// --- Backward -----------------------------------------------------------------

void MiniLlm::backward(const std::vector<BlockForwardTape>& tapes,
                       const std::vector<float>& grad_hidden_out,
                       ModelWeights& grad_weights,
                       std::vector<float>& grad_hidden_in) const {
    model::backward_model(config(), weights(), tapes, grad_hidden_out, grad_weights, grad_hidden_in);
}

} // namespace model
