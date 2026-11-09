#include "kernels/common.cuh"
#include "kernels/ops.h"

#include <stdexcept>

namespace kernels {

using namespace detail;

namespace {

constexpr unsigned kBlock = 256;
constexpr std::size_t kMaxConvKernel = 8;

// One thread per channel, walking the tokens in order.
__global__ void causal_conv1d_silu_kernel(const float* x, float* state, const bf16* w, std::size_t tokens,
                                          std::size_t channels, std::size_t kernel, float* out) {
    const std::size_t c = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (c >= channels) {
        return;
    }
    float window[kMaxConvKernel]; // previous inputs, oldest first
    float wc[kMaxConvKernel];
    for (std::size_t j = 0; j < kernel; ++j) {
        wc[j] = to_float(w[c * kernel + j]);
    }
    for (std::size_t j = 0; j + 1 < kernel; ++j) {
        window[j] = state[j * channels + c];
    }
    for (std::size_t t = 0; t < tokens; ++t) {
        const float xt = x[t * channels + c];
        float acc = 0.0f;
        for (std::size_t j = 0; j + 1 < kernel; ++j) {
            acc += window[j] * wc[j];
        }
        acc += xt * wc[kernel - 1];
        out[t * channels + c] = silu(acc);
        for (std::size_t j = 0; j + 2 < kernel; ++j) {
            window[j] = window[j + 1];
        }
        window[kernel - 2] = xt;
    }
    for (std::size_t j = 0; j + 1 < kernel; ++j) {
        state[j * channels + c] = window[j];
    }
}

__global__ void gdn_gates_kernel(const float* a, const float* b, const float* a_log, const bf16* dt_bias, std::size_t n,
                                 std::size_t heads, float* g, float* beta) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const std::size_t h = i % heads;
    const float x = a[i] + to_float(dt_bias[h]);
    const float softplus = x > 20.0f ? x : log1pf(expf(x));
    g[i] = -expf(a_log[h]) * softplus;
    beta[i] = sigmoid(b[i]);
}

// One block per value head, one thread per value column j; thread j owns column S[:, j].
// Shared memory: q[k_dim], k[k_dim].
__global__ void gdn_recurrence_kernel(const float* qkv, const float* g, const float* beta, float* state, float* out,
                                      std::size_t tokens, std::size_t k_heads, std::size_t v_heads, std::size_t k_dim,
                                      std::size_t v_dim) {
    extern __shared__ float smem[];
    float* qs = smem;
    float* ks = smem + k_dim;
    const std::size_t h = blockIdx.x;
    const std::size_t kh = h / (v_heads / k_heads);
    const std::size_t j = threadIdx.x;
    const std::size_t key_dim = k_heads * k_dim;
    const std::size_t row = 2 * key_dim + v_heads * v_dim;
    float* s = state + h * k_dim * v_dim;
    const float q_scale = 1.0f / sqrtf(static_cast<float>(k_dim));

    for (std::size_t t = 0; t < tokens; ++t) {
        const float* r = qkv + t * row;
        float qsq = 0.0f;
        float ksq = 0.0f;
        for (std::size_t i = threadIdx.x; i < k_dim; i += blockDim.x) {
            const float qv = r[kh * k_dim + i];
            const float kv = r[key_dim + kh * k_dim + i];
            qs[i] = qv;
            ks[i] = kv;
            qsq += qv * qv;
            ksq += kv * kv;
        }
        const float q_inv = 1.0f / sqrtf(block_sum(qsq) + 1e-6f);
        const float k_inv = 1.0f / sqrtf(block_sum(ksq) + 1e-6f);
        // block_sum synchronises, so qs / ks are complete.
        if (j < v_dim) {
            const float decay = expf(g[t * v_heads + h]);
            const float b = beta[t * v_heads + h];
            const float v = r[2 * key_dim + h * v_dim + j];
            float kv_mem = 0.0f;
            for (std::size_t i = 0; i < k_dim; ++i) {
                const float si = s[i * v_dim + j] * decay;
                s[i * v_dim + j] = si;
                kv_mem += si * (ks[i] * k_inv);
            }
            const float delta = (v - kv_mem) * b;
            float o = 0.0f;
            for (std::size_t i = 0; i < k_dim; ++i) {
                const float si = s[i * v_dim + j] + (ks[i] * k_inv) * delta;
                s[i * v_dim + j] = si;
                o += si * (qs[i] * q_inv * q_scale);
            }
            out[(t * v_heads + h) * v_dim + j] = o;
        }
        __syncthreads(); // qs / ks are rewritten for the next token
    }
}

} // namespace

void causal_conv1d_silu(const float* x, float* state, const bf16* w, std::size_t tokens, std::size_t channels,
                        std::size_t kernel, float* out, cudaStream_t stream) {
    if (kernel < 2 || kernel > kMaxConvKernel) {
        throw std::invalid_argument("causal_conv1d_silu: kernel size must be in [2, 8]");
    }
    causal_conv1d_silu_kernel<<<grid_for(channels, kBlock), kBlock, 0, stream>>>(x, state, w, tokens, channels, kernel,
                                                                                 out);
    check_launch();
}

void gdn_gates(const float* a, const float* b, const float* a_log, const bf16* dt_bias, std::size_t tokens,
               std::size_t heads, float* g, float* beta, cudaStream_t stream) {
    const std::size_t n = tokens * heads;
    gdn_gates_kernel<<<grid_for(n, kBlock), kBlock, 0, stream>>>(a, b, a_log, dt_bias, n, heads, g, beta);
    check_launch();
}

void gdn_recurrence(const float* qkv, const float* g, const float* beta, float* state, float* out, std::size_t tokens,
                    std::size_t k_heads, std::size_t v_heads, std::size_t k_dim, std::size_t v_dim,
                    cudaStream_t stream) {
    const unsigned block = static_cast<unsigned>((v_dim + kWarp - 1) / kWarp * kWarp);
    gdn_recurrence_kernel<<<static_cast<unsigned>(v_heads), block, 2 * k_dim * sizeof(float), stream>>>(
        qkv, g, beta, state, out, tokens, k_heads, v_heads, k_dim, v_dim);
    check_launch();
}

} // namespace kernels
