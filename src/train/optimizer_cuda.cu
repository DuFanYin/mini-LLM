#include "train/train.h"

#include "kernel/cuda_util.cuh"

#include <algorithm>
#include <cmath>
#include <vector>

// CUDA backend for the optimizer math declared in train/train.h. Selected by the
// configure script for MINI_LLM_KERNEL_BACKEND=cuda. The structural code (buffer
// allocation, layer iteration) stays on the host exactly as in optimizer.cpp;
// only the per-tensor math leaves (sum-of-squares, scale, AdamW update) run on
// the GPU. Each leaf uploads the tensor(s), launches a kernel, and copies back.

namespace train {

namespace {

using kernel::cuda::DeviceBuffer;

constexpr int kBlock = 256;

// ------------------------------------------------------------
// Device kernels.
// ------------------------------------------------------------

__global__ void sumsq_kernel(const float* x, int n, double* acc) {
    __shared__ double sdata[kBlock];
    const int tid = threadIdx.x;
    double s = 0.0;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        const double xi = static_cast<double>(x[i]);
        s += xi * xi;
    }
    sdata[tid] = s;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) {
            sdata[tid] += sdata[tid + st];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(acc, sdata[0]);
    }
}

__global__ void scale_kernel(float* x, int n, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] *= scale;
    }
}

__global__ void adamw_kernel(float* param, const float* grad, float* m, float* v, int n, float lr, float beta1,
                             float beta2, float one_minus_beta1, float one_minus_beta2, float inv_bias_c1,
                             float inv_bias_c2, float eps, float weight_decay) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float gi = grad[i];
    const float mi = beta1 * m[i] + one_minus_beta1 * gi;
    const float vi = beta2 * v[i] + one_minus_beta2 * gi * gi;
    m[i] = mi;
    v[i] = vi;
    const float m_hat = mi * inv_bias_c1;
    const float v_hat = vi * inv_bias_c2;
    const float pi = param[i];
    const float denom = sqrtf(v_hat) + eps;
    param[i] = pi - lr * (m_hat / denom + weight_decay * pi);
}

int blocks_for(size_t n) { return static_cast<int>((n + kBlock - 1) / kBlock); }

// Single persistent device accumulator for grad_l2_sq (process lifetime).
double* sumsq_accumulator() {
    static double* ptr = [] {
        double* p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, sizeof(double)));
        return p;
    }();
    return ptr;
}

// ------------------------------------------------------------
// Structural host helpers (identical to the scalar optimizer).
// ------------------------------------------------------------

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

// ------------------------------------------------------------
// CUDA math leaves.
// ------------------------------------------------------------

// Upload `v` and accumulate sum(v[i]^2) into the device accumulator `d_acc`.
void accumulate_sumsq(const std::vector<float>& v, DeviceBuffer& buf, double* d_acc) {
    const size_t n = v.size();
    if (n == 0) {
        return;
    }
    const float* d_x = kernel::cuda::upload(buf, v.data(), n);
    const int blocks = std::min(blocks_for(n), 256);
    sumsq_kernel<<<blocks, kBlock>>>(d_x, static_cast<int>(n), d_acc);
}

double grad_l2_sq(const model::ModelWeights& g) {
    double* acc = sumsq_accumulator();
    CUDA_CHECK(cudaMemset(acc, 0, sizeof(double)));
    static DeviceBuffer buf;

    auto add = [&](const std::vector<float>& v) { accumulate_sumsq(v, buf, acc); };
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

    kernel::cuda::sync();
    double host = 0.0;
    CUDA_CHECK(cudaMemcpy(&host, acc, sizeof(double), cudaMemcpyDeviceToHost));
    return host;
}

void vector_scale(std::vector<float>& v, float scale) {
    const size_t n = v.size();
    if (n == 0) {
        return;
    }
    static DeviceBuffer buf;
    float* d_x = kernel::cuda::upload(buf, v.data(), n);
    scale_kernel<<<blocks_for(n), kBlock>>>(d_x, static_cast<int>(n), scale);
    kernel::cuda::sync();
    kernel::cuda::download(v.data(), d_x, n);
}

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

    static DeviceBuffer dParam, dGrad, dM, dV;
    float* d_param = kernel::cuda::upload(dParam, param.data(), n);
    const float* d_grad = kernel::cuda::upload(dGrad, grad.data(), n);
    float* d_m = kernel::cuda::upload(dM, m.data(), n);
    float* d_v = kernel::cuda::upload(dV, v.data(), n);

    adamw_kernel<<<blocks_for(n), kBlock>>>(d_param, d_grad, d_m, d_v, static_cast<int>(n), lr, beta1, beta2,
                                            one_minus_beta1, one_minus_beta2, inv_bias_c1, inv_bias_c2, eps,
                                            weight_decay);
    kernel::cuda::sync();
    kernel::cuda::download(param.data(), d_param, n);
    kernel::cuda::download(m.data(), d_m, n);
    kernel::cuda::download(v.data(), d_v, n);
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
    vector_scale(g.output_projection, scale);
}

void adamw_step(model::ModelWeights& param, const model::ModelWeights& grad, model::ModelWeights& m,
                model::ModelWeights& v, size_t t, float lr, float beta1, float beta2, float eps, float weight_decay) {
    for (size_t i = 0; i < param.layers.size(); ++i) {
        adamw_update_decoder_layer(param.layers[i], grad.layers[i], m.layers[i], v.layers[i], t, lr, beta1, beta2, eps,
                                   weight_decay);
    }
    adamw_update_vec(param.token_embedding, grad.token_embedding, m.token_embedding, v.token_embedding, t, lr, beta1,
                     beta2, eps, weight_decay);
    adamw_update_vec(param.output_projection, grad.output_projection, m.output_projection, v.output_projection, t, lr,
                     beta1, beta2, eps, weight_decay);
}

} // namespace train
