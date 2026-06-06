#include "train/train.h"

#include "kernel/kernel.h"

#include <algorithm>
#include <cmath>
#include <vector>

// Backend-agnostic optimizer. This file owns only the structural work — building
// and zeroing gradient/moment buffers and walking ModelWeights tensor by tensor.
// The per-tensor math (AdamW update, sum-of-squares, scale) is delegated to the
// kernel layer (kernel::adamw_update / sum_squares / scale), whose scalar or CUDA
// implementation is selected at build time. There is no per-backend copy here.

namespace train {

namespace {

model::DecoderLayerWeights clone_decoder_layer_weights(const model::DecoderLayerWeights& ref) {
    model::DecoderLayerWeights g;
    g.norm1.weight.assign(ref.norm1.weight.size(), 0.0f);
    g.norm1.eps = ref.norm1.eps;
    g.norm2.weight.assign(ref.norm2.weight.size(), 0.0f);
    g.norm2.eps = ref.norm2.eps;

    auto zero_linear = [](const model::LinearWeights& src) {
        model::LinearWeights lg;
        lg.in_dim = src.in_dim;
        lg.out_dim = src.out_dim;
        lg.weight.assign(src.weight.size(), 0.0f);
        if (!src.bias.empty()) {
            lg.bias.assign(src.bias.size(), 0.0f);
        }
        return lg;
    };
    g.attention.q_proj = zero_linear(ref.attention.q_proj);
    g.attention.k_proj = zero_linear(ref.attention.k_proj);
    g.attention.v_proj = zero_linear(ref.attention.v_proj);
    g.attention.o_proj = zero_linear(ref.attention.o_proj);
    g.mlp.gate = zero_linear(ref.mlp.gate);
    g.mlp.up = zero_linear(ref.mlp.up);
    g.mlp.down = zero_linear(ref.mlp.down);
    return g;
}

void clear_linear(model::LinearWeights& lg) {
    std::fill(lg.weight.begin(), lg.weight.end(), 0.0f);
    if (!lg.bias.empty()) {
        std::fill(lg.bias.begin(), lg.bias.end(), 0.0f);
    }
}

void clear_decoder_layer(model::DecoderLayerWeights& g) {
    std::fill(g.norm1.weight.begin(), g.norm1.weight.end(), 0.0f);
    std::fill(g.norm2.weight.begin(), g.norm2.weight.end(), 0.0f);
    clear_linear(g.attention.q_proj);
    clear_linear(g.attention.k_proj);
    clear_linear(g.attention.v_proj);
    clear_linear(g.attention.o_proj);
    clear_linear(g.mlp.gate);
    clear_linear(g.mlp.up);
    clear_linear(g.mlp.down);
}

// --- per-tensor leaves, delegating raw math to the kernel layer ---

double grad_l2_sq(const model::ModelWeights& g) {
    double s = 0.0;
    auto add = [&](const std::vector<float>& v) { s += kernel::sum_squares(v.data(), v.size()); };
    for (const auto& layer : g.layers) {
        add(layer.norm1.weight);
        add(layer.norm2.weight);
        add(layer.attention.q_proj.weight);
        add(layer.attention.k_proj.weight);
        add(layer.attention.v_proj.weight);
        add(layer.attention.o_proj.weight);
        add(layer.mlp.gate.weight);
        add(layer.mlp.up.weight);
        add(layer.mlp.down.weight);
        if (!layer.attention.q_proj.bias.empty()) {
            add(layer.attention.q_proj.bias);
            add(layer.attention.k_proj.bias);
            add(layer.attention.v_proj.bias);
            add(layer.attention.o_proj.bias);
            add(layer.mlp.gate.bias);
            add(layer.mlp.up.bias);
            add(layer.mlp.down.bias);
        }
    }
    add(g.token_embedding);
    add(g.output_projection);
    return s;
}

void scale_vec(std::vector<float>& v, float s) { kernel::scale(v.data(), v.size(), s); }

void update_vec(std::vector<float>& param, const std::vector<float>& grad, std::vector<float>& m,
                std::vector<float>& v, const kernel::AdamwParams& p) {
    kernel::adamw_update(param.data(), grad.data(), m.data(), v.data(), param.size(), p);
}

void adamw_update_decoder_layer(model::DecoderLayerWeights& param, const model::DecoderLayerWeights& grad,
                                model::DecoderLayerWeights& m, model::DecoderLayerWeights& v,
                                const kernel::AdamwParams& p) {
    update_vec(param.norm1.weight, grad.norm1.weight, m.norm1.weight, v.norm1.weight, p);
    update_vec(param.norm2.weight, grad.norm2.weight, m.norm2.weight, v.norm2.weight, p);

    update_vec(param.attention.q_proj.weight, grad.attention.q_proj.weight, m.attention.q_proj.weight,
               v.attention.q_proj.weight, p);
    update_vec(param.attention.k_proj.weight, grad.attention.k_proj.weight, m.attention.k_proj.weight,
               v.attention.k_proj.weight, p);
    update_vec(param.attention.v_proj.weight, grad.attention.v_proj.weight, m.attention.v_proj.weight,
               v.attention.v_proj.weight, p);
    update_vec(param.attention.o_proj.weight, grad.attention.o_proj.weight, m.attention.o_proj.weight,
               v.attention.o_proj.weight, p);
    update_vec(param.mlp.gate.weight, grad.mlp.gate.weight, m.mlp.gate.weight, v.mlp.gate.weight, p);
    update_vec(param.mlp.up.weight, grad.mlp.up.weight, m.mlp.up.weight, v.mlp.up.weight, p);
    update_vec(param.mlp.down.weight, grad.mlp.down.weight, m.mlp.down.weight, v.mlp.down.weight, p);

    if (!param.attention.q_proj.bias.empty()) {
        update_vec(param.attention.q_proj.bias, grad.attention.q_proj.bias, m.attention.q_proj.bias,
                   v.attention.q_proj.bias, p);
        update_vec(param.attention.k_proj.bias, grad.attention.k_proj.bias, m.attention.k_proj.bias,
                   v.attention.k_proj.bias, p);
        update_vec(param.attention.v_proj.bias, grad.attention.v_proj.bias, m.attention.v_proj.bias,
                   v.attention.v_proj.bias, p);
        update_vec(param.attention.o_proj.bias, grad.attention.o_proj.bias, m.attention.o_proj.bias,
                   v.attention.o_proj.bias, p);
        update_vec(param.mlp.gate.bias, grad.mlp.gate.bias, m.mlp.gate.bias, v.mlp.gate.bias, p);
        update_vec(param.mlp.up.bias, grad.mlp.up.bias, m.mlp.up.bias, v.mlp.up.bias, p);
        update_vec(param.mlp.down.bias, grad.mlp.down.bias, m.mlp.down.bias, v.mlp.down.bias, p);
    }
}

} // namespace

// ------------------------------------------------------------
// Public API (declared in train/train.h).
// ------------------------------------------------------------

model::ModelWeights zeros_like(const model::ModelWeights& ref) {
    model::ModelWeights g;
    g.vocab_size = ref.vocab_size;
    g.layers.reserve(ref.layers.size());
    for (const auto& layer : ref.layers) {
        g.layers.push_back(clone_decoder_layer_weights(layer));
    }
    g.token_embedding.assign(ref.token_embedding.size(), 0.0f);
    g.output_projection.assign(ref.output_projection.size(), 0.0f);
    return g;
}

void zero(model::ModelWeights& g) {
    for (auto& layer : g.layers) {
        clear_decoder_layer(layer);
    }
    std::fill(g.token_embedding.begin(), g.token_embedding.end(), 0.0f);
    std::fill(g.output_projection.begin(), g.output_projection.end(), 0.0f);
}

void clip_grad(model::ModelWeights& g, float max_norm) {
    if (max_norm <= 0.0f) {
        return;
    }
    const double sq = grad_l2_sq(g);
    if (sq <= 0.0) {
        return;
    }
    const double n = std::sqrt(sq);
    if (n <= static_cast<double>(max_norm)) {
        return;
    }
    const float scale = static_cast<float>(max_norm / n);
    for (auto& layer : g.layers) {
        scale_vec(layer.norm1.weight, scale);
        scale_vec(layer.norm2.weight, scale);
        scale_vec(layer.attention.q_proj.weight, scale);
        scale_vec(layer.attention.k_proj.weight, scale);
        scale_vec(layer.attention.v_proj.weight, scale);
        scale_vec(layer.attention.o_proj.weight, scale);
        scale_vec(layer.mlp.gate.weight, scale);
        scale_vec(layer.mlp.up.weight, scale);
        scale_vec(layer.mlp.down.weight, scale);
        if (!layer.attention.q_proj.bias.empty()) {
            scale_vec(layer.attention.q_proj.bias, scale);
            scale_vec(layer.attention.k_proj.bias, scale);
            scale_vec(layer.attention.v_proj.bias, scale);
            scale_vec(layer.attention.o_proj.bias, scale);
            scale_vec(layer.mlp.gate.bias, scale);
            scale_vec(layer.mlp.up.bias, scale);
            scale_vec(layer.mlp.down.bias, scale);
        }
    }
    scale_vec(g.token_embedding, scale);
    scale_vec(g.output_projection, scale);
}

void adamw_step(model::ModelWeights& param, const model::ModelWeights& grad, model::ModelWeights& m,
                model::ModelWeights& v, size_t t, float lr, float beta1, float beta2, float eps, float weight_decay) {
    const kernel::AdamwParams p{t, lr, beta1, beta2, eps, weight_decay};
    for (size_t i = 0; i < param.layers.size(); ++i) {
        adamw_update_decoder_layer(param.layers[i], grad.layers[i], m.layers[i], v.layers[i], p);
    }
    update_vec(param.token_embedding, grad.token_embedding, m.token_embedding, v.token_embedding, p);
    update_vec(param.output_projection, grad.output_projection, m.output_projection, v.output_projection, p);
}

} // namespace train
