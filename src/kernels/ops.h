#pragma once

// Launch functions for the engine's kernels. All pointers are device memory;
// every call is asynchronous on `stream` and checks the launch. Activations are
// f32, weights bf16 unless noted. Shapes are row-major.

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace kernels {

using bf16 = __nv_bfloat16;

// out[t, :] = table[ids[t], :]
void embedding(const std::int32_t* ids, std::size_t tokens, const bf16* table, std::size_t dim, float* out,
               cudaStream_t stream);

// Zero-centred RMSNorm per row: out = x / sqrt(mean(x²) + eps) · (1 + w). x, out: [rows, dim].
void rms_norm(const float* x, std::size_t rows, std::size_t dim, const bf16* w, float eps, float* out,
              cudaStream_t stream);

// y[r, n] = Σ_k x[r, k] · w[n, k]. x: [rows, in], w: [out, in], y: [rows, out].
void linear(const float* x, std::size_t rows, std::size_t in, const bf16* w, std::size_t out, float* y,
            cudaStream_t stream);

// x += y over n elements.
void add_inplace(float* x, const float* y, std::size_t n, cudaStream_t stream);

// out = silu(gate) · up over n elements.
void silu_mul(const float* gate, const float* up, float* out, std::size_t n, cudaStream_t stream);

// Strided view of per-head vectors: element (t, h, d) is at base[t * token_stride + h * head_stride + d].
struct HeadView {
    float* base;
    std::size_t token_stride;
    std::size_t head_stride;
};

// In place, per (token, head): zero-centred RMSNorm over head_dim, then RoPE
// (rotate-half) on the first rotary_dim dims at position pos0 + t.
void head_norm_rope(HeadView x, std::size_t tokens, std::size_t heads, std::size_t head_dim, const bf16* w, float eps,
                    std::size_t rotary_dim, double rope_theta, std::size_t pos0, cudaStream_t stream);

// Copies k, v ([tokens, kv_heads, head_dim]) into caches [kv_heads, max_seq, head_dim] at pos0.
void kv_append(const float* k, const float* v, float* k_cache, float* v_cache, std::size_t tokens, std::size_t kv_heads,
               std::size_t head_dim, std::size_t pos0, std::size_t max_seq, cudaStream_t stream);

// Causal GQA over the cache; query t sits at position pos0 + t. out: [tokens, heads * head_dim].
// Query head h reads kv head h / (heads / kv_heads).
void attention(HeadView q, const float* k_cache, const float* v_cache, std::size_t max_seq, float* out,
               std::size_t tokens, std::size_t heads, std::size_t kv_heads, std::size_t head_dim, std::size_t pos0,
               cudaStream_t stream);

// x[t, h·d + i] *= sigmoid(gate(t, h, i)). x: [tokens, heads * head_dim].
void sigmoid_gate(float* x, HeadView gate, std::size_t tokens, std::size_t heads, std::size_t head_dim,
                  cudaStream_t stream);

// Depthwise causal conv over time, then SiLU. x, out: [tokens, channels]; w: [channels, kernel];
// state: [kernel - 1, channels], the previous inputs (oldest first), updated in place.
void causal_conv1d_silu(const float* x, float* state, const bf16* w, std::size_t tokens, std::size_t channels,
                        std::size_t kernel, float* out, cudaStream_t stream);

// beta = sigmoid(b), g = −exp(a_log) · softplus(a + dt_bias). a, b, g, beta: [tokens, heads]; a_log is f32.
void gdn_gates(const float* a, const float* b, const float* a_log, const bf16* dt_bias, std::size_t tokens,
               std::size_t heads, float* g, float* beta, cudaStream_t stream);

// Gated delta rule, token by token. qkv: [tokens, (2·k_heads·k_dim + v_heads·v_dim)] after the conv,
// q | k | v. q and k are L2-normalised here and q is scaled by 1/sqrt(k_dim).
// state: [v_heads, k_dim, v_dim], updated in place; out: [tokens, v_heads, v_dim].
void gdn_recurrence(const float* qkv, const float* g, const float* beta, float* state, float* out, std::size_t tokens,
                    std::size_t k_heads, std::size_t v_heads, std::size_t k_dim, std::size_t v_dim,
                    cudaStream_t stream);

// Per row: out = x / sqrt(mean(x²) + eps) · w · silu(z). x, z, out: [rows, dim]; w is f32 [dim].
void gated_rms_norm(const float* x, const float* z, const float* w, float eps, std::size_t rows, std::size_t dim,
                    float* out, cudaStream_t stream);

// out = index of the largest of x[0..n); ties go to the lowest index.
void argmax(const float* x, std::size_t n, std::int32_t* out, cudaStream_t stream);

} // namespace kernels
