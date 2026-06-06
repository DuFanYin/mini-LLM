#include "kernel/kernel.h"

#include "kernel/cuda_util.cuh"

#include <cmath>

// CUDA backend for grouped-query attention forward/backward (handwritten).
//
// Public signatures match the scalar attention.cpp exactly. One thread block
// handles one (query position si, query head qh) pair; softmax and its backward
// are inlined here (the scalar path delegates to core.cpp). KV *inputs* are read
// with stride `kv_stride`; KV *gradients* are written with stride `total_kv_len`,
// matching the scalar layout. Multiple query heads/positions accumulate into the
// same KV-gradient slots across blocks, so those writes use atomicAdd.

namespace kernel {

namespace {

using cuda::DeviceBuffer;

constexpr int kThreads = 128;

// q/ctx/grad layout: idx3(si, h, d, hq, d_head) == (si * hq + h) * d_head + d.
__device__ inline int qidx(int si, int h, int d, int hq, int d_head) {
    return (si * hq + h) * d_head + d;
}

__device__ float block_reduce_max(float v, float* red) {
    const int tid = threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] = fmaxf(red[tid], red[tid + s]);
        }
        __syncthreads();
    }
    const float result = red[0];
    __syncthreads();
    return result;
}

__device__ float block_reduce_sum(float v, float* red) {
    const int tid = threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] += red[tid + s];
        }
        __syncthreads();
    }
    const float result = red[0];
    __syncthreads();
    return result;
}

// Fill sh_prob[0..total_kv_len) with the softmax row for (si, qh). Mirrors the
// scalar dot -> scale -> causal/additive mask -> numerically-stable softmax.
__device__ void compute_probs_row(float* sh_prob, float* red, const float* q, const float* k_all, const float* mask,
                                   bool have_mask, int si, int qh, int kh, int hq, int d_head, int total_kv_len,
                                   int kv_stride, int visible, float inv_sqrt_d) {
    for (int ti = threadIdx.x; ti < total_kv_len; ti += blockDim.x) {
        if (ti >= visible) {
            sh_prob[ti] = -INFINITY;
            continue;
        }
        float dot = 0.0f;
        for (int d = 0; d < d_head; ++d) {
            dot += q[qidx(si, qh, d, hq, d_head)] * k_all[(kh * kv_stride + ti) * d_head + d];
        }
        float logit = dot * inv_sqrt_d;
        if (have_mask) {
            logit += mask[si * total_kv_len + ti];
        }
        sh_prob[ti] = logit;
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int ti = threadIdx.x; ti < total_kv_len; ti += blockDim.x) {
        local_max = fmaxf(local_max, sh_prob[ti]);
    }
    const float max_logit = block_reduce_max(local_max, red);

    float local_sum = 0.0f;
    for (int ti = threadIdx.x; ti < total_kv_len; ti += blockDim.x) {
        const float e = expf(sh_prob[ti] - max_logit);
        sh_prob[ti] = e;
        local_sum += e;
    }
    const float denom = block_reduce_sum(local_sum, red);
    const float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;

    for (int ti = threadIdx.x; ti < total_kv_len; ti += blockDim.x) {
        sh_prob[ti] *= inv_denom;
    }
    __syncthreads();
}

__device__ inline int visible_for(int si, int past_len, int total_kv_len, bool use_cache, bool causal) {
    const int abs_pos = use_cache ? (past_len + si) : si;
    if (!causal) {
        return total_kv_len;
    }
    const int v = abs_pos + 1;
    return v < total_kv_len ? v : total_kv_len;
}

__global__ void forward_kernel(const float* q, const float* k_all, const float* v_all, const float* mask,
                               bool have_mask, float* ctx, float* probs_out, bool emit_probs, int hq, int hkv,
                               int d_head, int total_kv_len, int kv_stride, int past_len, bool use_cache, bool causal,
                               float inv_sqrt_d) {
    extern __shared__ float sh_prob[]; // total_kv_len floats
    __shared__ float red[kThreads];

    const int blk = blockIdx.x; // si * hq + qh
    const int si = blk / hq;
    const int qh = blk % hq;
    const int kh = qh / (hq / hkv);
    const int visible = visible_for(si, past_len, total_kv_len, use_cache, causal);

    compute_probs_row(sh_prob, red, q, k_all, mask, have_mask, si, qh, kh, hq, d_head, total_kv_len, kv_stride, visible,
                      inv_sqrt_d);

    if (emit_probs) {
        float* row = probs_out + (si * hq + qh) * total_kv_len;
        for (int ti = threadIdx.x; ti < total_kv_len; ti += blockDim.x) {
            row[ti] = sh_prob[ti];
        }
    }

    for (int d = threadIdx.x; d < d_head; d += blockDim.x) {
        float acc = 0.0f;
        for (int ti = 0; ti < total_kv_len; ++ti) {
            acc += sh_prob[ti] * v_all[(kh * kv_stride + ti) * d_head + d];
        }
        ctx[qidx(si, qh, d, hq, d_head)] = acc;
    }
}

__global__ void backward_kernel(const float* q, const float* k_all, const float* v_all, const float* mask,
                                bool have_mask, const float* probs_cached, bool have_probs, const float* grad_ctx,
                                float* grad_q, float* grad_k_all, float* grad_v_all, int hq, int hkv, int d_head,
                                int total_kv_len, int kv_stride, int past_len, bool use_cache, bool causal,
                                float inv_sqrt_d) {
    extern __shared__ float sh[]; // 3 * total_kv_len floats: prob | dprob | dlogit
    __shared__ float red[kThreads];
    float* sh_prob = sh;
    float* sh_dprob = sh + total_kv_len;
    float* sh_dlogit = sh + 2 * total_kv_len;

    const int blk = blockIdx.x; // si * hq + qh
    const int si = blk / hq;
    const int qh = blk % hq;
    const int kh = qh / (hq / hkv);

    if (have_probs) {
        const float* row = probs_cached + (si * hq + qh) * total_kv_len;
        for (int ti = threadIdx.x; ti < total_kv_len; ti += blockDim.x) {
            sh_prob[ti] = row[ti];
        }
        __syncthreads();
    } else {
        const int visible = visible_for(si, past_len, total_kv_len, use_cache, causal);
        compute_probs_row(sh_prob, red, q, k_all, mask, have_mask, si, qh, kh, hq, d_head, total_kv_len, kv_stride,
                          visible, inv_sqrt_d);
    }

    // d_prob[ti] = sum_d grad_ctx[si,qh,d] * v_all[kh,ti,d]
    for (int ti = threadIdx.x; ti < total_kv_len; ti += blockDim.x) {
        float gp = 0.0f;
        for (int d = 0; d < d_head; ++d) {
            gp += grad_ctx[qidx(si, qh, d, hq, d_head)] * v_all[(kh * kv_stride + ti) * d_head + d];
        }
        sh_dprob[ti] = gp;
    }
    __syncthreads();

    // softmax backward: dot = sum_ti prob*dprob; dlogit = prob*(dprob - dot)
    float local_dot = 0.0f;
    for (int ti = threadIdx.x; ti < total_kv_len; ti += blockDim.x) {
        local_dot += sh_prob[ti] * sh_dprob[ti];
    }
    const float dot = block_reduce_sum(local_dot, red);
    for (int ti = threadIdx.x; ti < total_kv_len; ti += blockDim.x) {
        sh_dlogit[ti] = sh_prob[ti] * (sh_dprob[ti] - dot);
    }
    __syncthreads();

    // grad_q[si,qh,d] is unique to this block: plain write of the full sum over ti.
    for (int d = threadIdx.x; d < d_head; d += blockDim.x) {
        float acc = 0.0f;
        for (int ti = 0; ti < total_kv_len; ++ti) {
            acc += sh_dlogit[ti] * inv_sqrt_d * k_all[(kh * kv_stride + ti) * d_head + d];
        }
        grad_q[qidx(si, qh, d, hq, d_head)] = acc;
    }

    // grad_k_all / grad_v_all accumulate across blocks (shared kh, si) -> atomic.
    // Note: KV-grad stride is total_kv_len, not kv_stride.
    for (int idx = threadIdx.x; idx < total_kv_len * d_head; idx += blockDim.x) {
        const int ti = idx / d_head;
        const int d = idx % d_head;
        const float scale = sh_dlogit[ti] * inv_sqrt_d;
        atomicAdd(&grad_k_all[(kh * total_kv_len + ti) * d_head + d], scale * q[qidx(si, qh, d, hq, d_head)]);
        atomicAdd(&grad_v_all[(kh * total_kv_len + ti) * d_head + d],
                  sh_prob[ti] * grad_ctx[qidx(si, qh, d, hq, d_head)]);
    }
}

} // namespace

void gqa_attention_forward(const float* q, const float* k_all, const float* v_all, const float* attention_mask,
                           float* ctx, float* attn_probs_out, const AttentionParams& p) {
    const size_t s = p.seq_len;
    const size_t hq = p.num_heads;
    const size_t hkv = p.num_kv_heads;
    const size_t d_head = p.head_dim;
    const size_t total_kv_len = p.total_kv_len;
    const size_t kv_stride = (p.kv_stride == 0) ? total_kv_len : p.kv_stride;
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d_head));
    const bool emit_probs = (attn_probs_out != nullptr);
    const bool have_mask = (attention_mask != nullptr);

    static DeviceBuffer dQ, dK, dV, dMask, dCtx, dProbs;
    float* d_ctx = cuda::zeros(dCtx, s * hq * d_head);
    float* d_probs = emit_probs ? cuda::zeros(dProbs, s * hq * total_kv_len) : nullptr;

    if (s != 0 && total_kv_len != 0) {
        const float* d_q = cuda::upload(dQ, q, s * hq * d_head);
        const float* d_k = cuda::upload(dK, k_all, hkv * kv_stride * d_head);
        const float* d_v = cuda::upload(dV, v_all, hkv * kv_stride * d_head);
        const float* d_mask = have_mask ? cuda::upload(dMask, attention_mask, s * total_kv_len) : nullptr;

        const size_t shmem = total_kv_len * sizeof(float);
        forward_kernel<<<static_cast<unsigned int>(s * hq), kThreads, shmem>>>(
            d_q, d_k, d_v, d_mask, have_mask, d_ctx, d_probs, emit_probs, static_cast<int>(hq), static_cast<int>(hkv),
            static_cast<int>(d_head), static_cast<int>(total_kv_len), static_cast<int>(kv_stride),
            static_cast<int>(p.past_len), p.use_cache, p.causal, inv_sqrt_d);
        cuda::sync();
    }

    cuda::download(ctx, d_ctx, s * hq * d_head);
    if (emit_probs) {
        cuda::download(attn_probs_out, d_probs, s * hq * total_kv_len);
    }
}

void gqa_attention_backward(const float* q, const float* k_all, const float* v_all, const float* attention_mask,
                            const float* attn_probs_cached, const float* grad_ctx, float* grad_q, float* grad_k_all,
                            float* grad_v_all, const AttentionParams& p) {
    const size_t s = p.seq_len;
    const size_t hq = p.num_heads;
    const size_t hkv = p.num_kv_heads;
    const size_t d_head = p.head_dim;
    const size_t total_kv_len = p.total_kv_len;
    const size_t kv_stride = (p.kv_stride == 0) ? total_kv_len : p.kv_stride;
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d_head));
    const bool have_probs = (attn_probs_cached != nullptr);
    const bool have_mask = (attention_mask != nullptr);

    static DeviceBuffer dQ, dK, dV, dMask, dProbs, dGradCtx, dGradQ, dGradK, dGradV;
    // Outputs are zeroed then filled (matches scalar std::fill at entry).
    float* d_gq = cuda::zeros(dGradQ, s * hq * d_head);
    float* d_gk = cuda::zeros(dGradK, total_kv_len * hkv * d_head);
    float* d_gv = cuda::zeros(dGradV, total_kv_len * hkv * d_head);

    if (s != 0 && total_kv_len != 0) {
        const float* d_q = cuda::upload(dQ, q, s * hq * d_head);
        const float* d_k = cuda::upload(dK, k_all, hkv * kv_stride * d_head);
        const float* d_v = cuda::upload(dV, v_all, hkv * kv_stride * d_head);
        const float* d_mask = have_mask ? cuda::upload(dMask, attention_mask, s * total_kv_len) : nullptr;
        const float* d_probs = have_probs ? cuda::upload(dProbs, attn_probs_cached, s * hq * total_kv_len) : nullptr;
        const float* d_gctx = cuda::upload(dGradCtx, grad_ctx, s * hq * d_head);

        const size_t shmem = 3 * total_kv_len * sizeof(float);
        backward_kernel<<<static_cast<unsigned int>(s * hq), kThreads, shmem>>>(
            d_q, d_k, d_v, d_mask, have_mask, d_probs, have_probs, d_gctx, d_gq, d_gk, d_gv, static_cast<int>(hq),
            static_cast<int>(hkv), static_cast<int>(d_head), static_cast<int>(total_kv_len),
            static_cast<int>(kv_stride), static_cast<int>(p.past_len), p.use_cache, p.causal, inv_sqrt_d);
        cuda::sync();
    }

    cuda::download(grad_q, d_gq, s * hq * d_head);
    cuda::download(grad_k_all, d_gk, total_kv_len * hkv * d_head);
    cuda::download(grad_v_all, d_gv, total_kv_len * hkv * d_head);
}

} // namespace kernel
