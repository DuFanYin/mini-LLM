#pragma once

#include <cstddef>

// Device-pointer kernel entry points: all pointers are ALREADY in device memory.
// These launch on the default stream and do NOT copy host<->device and do NOT
// synchronize — the caller owns residency and ordering. This is the Phase 0
// building block for the device-resident inference path. Host-clean header (no
// CUDA types) so plain .cpp callers can include it.

namespace kernel::cuda {

// C[M,N] = A[M,K] * B[N,K]^T (row-major; B read transposed). Overwrites C.
void gemm_nt_device(const float* A, const float* B, float* C, std::size_t M, std::size_t N, std::size_t K);

// y[M,N] = x[M,K] * w[N,K]^T + (bias ? bias[N] : 0). Overwrites y. bias may be null.
void linear_device(const float* x, const float* w, const float* bias, float* y, std::size_t M, std::size_t N,
                   std::size_t K);

// y[n] = a[n] + b[n].
void add_device(const float* a, const float* b, float* y, std::size_t n);

// hidden[n] = silu(gate[n]) * up[n].
void silu_mul_device(const float* gate, const float* up, float* hidden, std::size_t n);

// RMSNorm per row: y = x / sqrt(mean(x^2) + eps) * weight. x,y are [rows, d_model].
void rms_norm_device(const float* x, const float* weight, float eps, float* y, std::size_t rows,
                     std::size_t d_model);

// In-place RoPE on x = [seq_len, num_heads, head_dim], positions = past_len + si (contiguous).
// cos_tab/sin_tab are device [pos, freq] tables with `half` columns; eff_rot pairs are rotated.
void apply_rope_device(float* x, std::size_t past_len, const float* cos_tab, const float* sin_tab, std::size_t half,
                       std::size_t eff_rot, std::size_t seq_len, std::size_t num_heads, std::size_t head_dim);

// GQA attention forward (inference: no probs output). q is [seq, num_heads, head_dim];
// k_all/v_all are head-major [num_kv_heads, kv_stride, head_dim]; ctx is [seq, num_heads, head_dim].
void gqa_attention_forward_device(const float* q, const float* k_all, const float* v_all, const float* mask,
                                  float* ctx, std::size_t seq_len, std::size_t num_heads, std::size_t num_kv_heads,
                                  std::size_t head_dim, std::size_t past_len, std::size_t total_kv_len,
                                  std::size_t kv_stride, bool causal);

// Append seq_len new rows of K/V (laid out [seq, num_kv_heads, head_dim]) into head-major
// caches [num_kv_heads, kv_stride, head_dim] starting at position past_len.
void kv_append_device(const float* k_src, const float* v_src, float* k_cache, float* v_cache, std::size_t seq_len,
                      std::size_t num_kv_heads, std::size_t head_dim, std::size_t past_len, std::size_t kv_stride);

} // namespace kernel::cuda
