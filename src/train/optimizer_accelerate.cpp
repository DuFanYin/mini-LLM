// Accelerate variant of the AdamW optimizer + grad-norm utilities. Selected by
// the configure script only for MINI_LLM_KERNEL_BACKEND=accelerate. The public
// ABI matches src/train/optimizer.cpp exactly; we just swap the per-tensor
// primitives for vDSP / vForce calls so the bandwidth-bound passes hit closer
// to the memory-system limit.

#include "train/train.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace train {

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

model::ModelWeights clone_model(const model::ModelWeights& ref) {
    model::ModelWeights g;
    g.vocab_size = ref.vocab_size;
    g.layers.reserve(ref.layers.size());
    for (const auto& layer : ref.layers) {
        g.layers.push_back(clone_decoder_layer_weights(layer));
    }
    g.token_embedding.assign(ref.token_embedding.size(), 0.0f);
    g.lm_head.assign(ref.lm_head.size(), 0.0f);
    return g;
}

void clear(model::LinearWeights& lg) {
    std::fill(lg.weight.begin(), lg.weight.end(), 0.0f);
    if (!lg.bias.empty()) {
        std::fill(lg.bias.begin(), lg.bias.end(), 0.0f);
    }
}

void clear(model::DecoderLayerWeights& g) {
    std::fill(g.norm1.weight.begin(), g.norm1.weight.end(), 0.0f);
    std::fill(g.norm2.weight.begin(), g.norm2.weight.end(), 0.0f);
    clear(g.attention.q_proj);
    clear(g.attention.k_proj);
    clear(g.attention.v_proj);
    clear(g.attention.o_proj);
    clear(g.mlp.gate);
    clear(g.mlp.up);
    clear(g.mlp.down);
}

void clear(model::ModelWeights& g) {
    for (auto& layer : g.layers) {
        clear(layer);
    }
    std::fill(g.token_embedding.begin(), g.token_embedding.end(), 0.0f);
    std::fill(g.lm_head.begin(), g.lm_head.end(), 0.0f);
}

void vector_add_sq_sum(const std::vector<float>& v, double* acc) {
    if (v.empty()) {
        return;
    }
    float sum_sq = 0.0f;
    vDSP_svesq(v.data(), 1, &sum_sq, v.size());
    *acc += static_cast<double>(sum_sq);
}

double grad_l2_sq(const model::ModelWeights& g) {
    double s = 0.0;
    for (const auto& layer : g.layers) {
        vector_add_sq_sum(layer.norm1.weight, &s);
        vector_add_sq_sum(layer.norm2.weight, &s);
        vector_add_sq_sum(layer.attention.q_proj.weight, &s);
        vector_add_sq_sum(layer.attention.k_proj.weight, &s);
        vector_add_sq_sum(layer.attention.v_proj.weight, &s);
        vector_add_sq_sum(layer.attention.o_proj.weight, &s);
        vector_add_sq_sum(layer.mlp.gate.weight, &s);
        vector_add_sq_sum(layer.mlp.up.weight, &s);
        vector_add_sq_sum(layer.mlp.down.weight, &s);
        if (!layer.attention.q_proj.bias.empty()) {
            vector_add_sq_sum(layer.attention.q_proj.bias, &s);
            vector_add_sq_sum(layer.attention.k_proj.bias, &s);
            vector_add_sq_sum(layer.attention.v_proj.bias, &s);
            vector_add_sq_sum(layer.attention.o_proj.bias, &s);
            vector_add_sq_sum(layer.mlp.gate.bias, &s);
            vector_add_sq_sum(layer.mlp.up.bias, &s);
            vector_add_sq_sum(layer.mlp.down.bias, &s);
        }
    }
    vector_add_sq_sum(g.token_embedding, &s);
    vector_add_sq_sum(g.lm_head, &s);
    return s;
}

void vector_scale(std::vector<float>& v, float scale) {
    if (v.empty()) {
        return;
    }
    vDSP_vsmul(v.data(), 1, &scale, v.data(), 1, v.size());
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
        vector_scale(layer.norm1.weight, scale);
        vector_scale(layer.norm2.weight, scale);
        vector_scale(layer.attention.q_proj.weight, scale);
        vector_scale(layer.attention.k_proj.weight, scale);
        vector_scale(layer.attention.v_proj.weight, scale);
        vector_scale(layer.attention.o_proj.weight, scale);
        vector_scale(layer.mlp.gate.weight, scale);
        vector_scale(layer.mlp.up.weight, scale);
        vector_scale(layer.mlp.down.weight, scale);
        if (!layer.attention.q_proj.bias.empty()) {
            vector_scale(layer.attention.q_proj.bias, scale);
            vector_scale(layer.attention.k_proj.bias, scale);
            vector_scale(layer.attention.v_proj.bias, scale);
            vector_scale(layer.attention.o_proj.bias, scale);
            vector_scale(layer.mlp.gate.bias, scale);
            vector_scale(layer.mlp.up.bias, scale);
            vector_scale(layer.mlp.down.bias, scale);
        }
    }
    vector_scale(g.token_embedding, scale);
    vector_scale(g.lm_head, scale);
}

// AdamW is memory-bandwidth bound: each element does a fused FMA + sqrt + div
// over 4 streams (param, grad, m, v) with no reuse. Splitting it into a
// sequence of vDSP calls multiplies the memory traffic by the number of passes
// and ends up slower than the hand-written single-pass loop. The scalar fused
// version below is what we ship even on the Accelerate backend.
void adamw_update_vec(std::vector<float>& param, const std::vector<float>& grad, std::vector<float>& m,
                      std::vector<float>& v, size_t t, float lr, float beta1, float beta2, float eps,
                      float weight_decay) {
    const size_t n = param.size();
    if (n == 0) {
        return;
    }
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;
    const float inv_bias_c1 = 1.0f / (1.0f - std::pow(beta1, static_cast<float>(t)));
    const float inv_bias_c2 = 1.0f / (1.0f - std::pow(beta2, static_cast<float>(t)));

    float* p_param = param.data();
    const float* p_grad = grad.data();
    float* p_m = m.data();
    float* p_v = v.data();

    for (size_t i = 0; i < n; ++i) {
        const float gi = p_grad[i];
        const float mi = beta1 * p_m[i] + one_minus_beta1 * gi;
        const float vi = beta2 * p_v[i] + one_minus_beta2 * gi * gi;
        p_m[i] = mi;
        p_v[i] = vi;
        const float m_hat = mi * inv_bias_c1;
        const float v_hat = vi * inv_bias_c2;
        const float pi = p_param[i];
        const float denom = std::sqrt(v_hat) + eps;
        p_param[i] = pi - lr * (m_hat / denom + weight_decay * pi);
    }
}

void adamw_update_decoder_layer(model::DecoderLayerWeights& param, const model::DecoderLayerWeights& grad,
                                model::DecoderLayerWeights& m, model::DecoderLayerWeights& v, size_t t, float lr,
                                float beta1, float beta2, float eps, float weight_decay) {
    adamw_update_vec(param.norm1.weight, grad.norm1.weight, m.norm1.weight, v.norm1.weight, t, lr, beta1, beta2, eps,
                     weight_decay);
    adamw_update_vec(param.norm2.weight, grad.norm2.weight, m.norm2.weight, v.norm2.weight, t, lr, beta1, beta2, eps,
                     weight_decay);

    adamw_update_vec(param.attention.q_proj.weight, grad.attention.q_proj.weight, m.attention.q_proj.weight,
                     v.attention.q_proj.weight, t, lr, beta1, beta2, eps, weight_decay);
    adamw_update_vec(param.attention.k_proj.weight, grad.attention.k_proj.weight, m.attention.k_proj.weight,
                     v.attention.k_proj.weight, t, lr, beta1, beta2, eps, weight_decay);
    adamw_update_vec(param.attention.v_proj.weight, grad.attention.v_proj.weight, m.attention.v_proj.weight,
                     v.attention.v_proj.weight, t, lr, beta1, beta2, eps, weight_decay);
    adamw_update_vec(param.attention.o_proj.weight, grad.attention.o_proj.weight, m.attention.o_proj.weight,
                     v.attention.o_proj.weight, t, lr, beta1, beta2, eps, weight_decay);
    adamw_update_vec(param.mlp.gate.weight, grad.mlp.gate.weight, m.mlp.gate.weight, v.mlp.gate.weight, t, lr, beta1,
                     beta2, eps, weight_decay);
    adamw_update_vec(param.mlp.up.weight, grad.mlp.up.weight, m.mlp.up.weight, v.mlp.up.weight, t, lr, beta1, beta2,
                     eps, weight_decay);
    adamw_update_vec(param.mlp.down.weight, grad.mlp.down.weight, m.mlp.down.weight, v.mlp.down.weight, t, lr, beta1,
                     beta2, eps, weight_decay);

    if (!param.attention.q_proj.bias.empty()) {
        adamw_update_vec(param.attention.q_proj.bias, grad.attention.q_proj.bias, m.attention.q_proj.bias,
                         v.attention.q_proj.bias, t, lr, beta1, beta2, eps, weight_decay);
        adamw_update_vec(param.attention.k_proj.bias, grad.attention.k_proj.bias, m.attention.k_proj.bias,
                         v.attention.k_proj.bias, t, lr, beta1, beta2, eps, weight_decay);
        adamw_update_vec(param.attention.v_proj.bias, grad.attention.v_proj.bias, m.attention.v_proj.bias,
                         v.attention.v_proj.bias, t, lr, beta1, beta2, eps, weight_decay);
        adamw_update_vec(param.attention.o_proj.bias, grad.attention.o_proj.bias, m.attention.o_proj.bias,
                         v.attention.o_proj.bias, t, lr, beta1, beta2, eps, weight_decay);
        adamw_update_vec(param.mlp.gate.bias, grad.mlp.gate.bias, m.mlp.gate.bias, v.mlp.gate.bias, t, lr, beta1,
                         beta2, eps, weight_decay);
        adamw_update_vec(param.mlp.up.bias, grad.mlp.up.bias, m.mlp.up.bias, v.mlp.up.bias, t, lr, beta1, beta2, eps,
                         weight_decay);
        adamw_update_vec(param.mlp.down.bias, grad.mlp.down.bias, m.mlp.down.bias, v.mlp.down.bias, t, lr, beta1,
                         beta2, eps, weight_decay);
    }
}

void adamw_update_model(model::ModelWeights& param, const model::ModelWeights& grad, model::ModelWeights& m,
                        model::ModelWeights& v, size_t t, float lr, float beta1, float beta2, float eps,
                        float weight_decay) {
    for (size_t i = 0; i < param.layers.size(); ++i) {
        adamw_update_decoder_layer(param.layers[i], grad.layers[i], m.layers[i], v.layers[i], t, lr, beta1, beta2, eps,
                                   weight_decay);
    }
    adamw_update_vec(param.token_embedding, grad.token_embedding, m.token_embedding, v.token_embedding, t, lr, beta1,
                     beta2, eps, weight_decay);
    adamw_update_vec(param.lm_head, grad.lm_head, m.lm_head, v.lm_head, t, lr, beta1, beta2, eps, weight_decay);
}

} // namespace train
