#include "kernel/kernel.h"

#include "kernel/cuda_util.cuh"

#include <cmath>

// CUDA backend for the element-wise / softmax core kernels declared in kernel.h.
// Selected by the configure script for MINI_LLM_KERNEL_BACKEND=cuda. Mirrors the
// scalar core.cpp math exactly; each public call uploads inputs, launches a
// handwritten kernel, and copies the result back (matching the host-pointer
// kernel boundary). silu / silu_derivative remain host scalar helpers.

namespace kernel {

namespace {

using cuda::DeviceBuffer;

constexpr int kBlock = 256;

__device__ inline float dev_silu(float x) { return x / (1.0f + expf(-x)); }
__device__ inline float dev_silu_derivative(float x) {
    const float s = 1.0f / (1.0f + expf(-x));
    return s + x * s * (1.0f - s);
}

__global__ void add_kernel(const float* a, const float* b, float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a[i] + b[i];
    }
}

__global__ void silu_mul_kernel(const float* gate, const float* up, float* hidden, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        hidden[i] = dev_silu(gate[i]) * up[i];
    }
}

__global__ void silu_mul_backward_kernel(const float* gate, const float* up, const float* grad_hidden,
                                         float* grad_gate, float* grad_up, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_gate[i] = dev_silu_derivative(gate[i]) * up[i] * grad_hidden[i];
        grad_up[i] = dev_silu(gate[i]) * grad_hidden[i];
    }
}

// Numerically-stable softmax over one row of n elements (single block).
__global__ void softmax_kernel(const float* logits, float* probs, int n) {
    __shared__ float red[kBlock];
    const int tid = threadIdx.x;

    float local_max = -INFINITY;
    for (int i = tid; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, logits[i]);
    }
    red[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] = fmaxf(red[tid], red[tid + s]);
        }
        __syncthreads();
    }
    const float max_logit = red[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float e = expf(logits[i] - max_logit);
        probs[i] = e;
        local_sum += e;
    }
    red[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] += red[tid + s];
        }
        __syncthreads();
    }
    const float denom = red[0];
    __syncthreads();
    const float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;

    for (int i = tid; i < n; i += blockDim.x) {
        probs[i] *= inv_denom;
    }
}

// grad_logits = prob * (grad_prob - sum_i prob_i * grad_prob_i) over one row.
__global__ void softmax_backward_kernel(const float* prob, const float* grad_prob, float* grad_logits, int n) {
    __shared__ float red[kBlock];
    const int tid = threadIdx.x;

    float local_dot = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local_dot += prob[i] * grad_prob[i];
    }
    red[tid] = local_dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] += red[tid + s];
        }
        __syncthreads();
    }
    const float dot = red[0];
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        grad_logits[i] = prob[i] * (grad_prob[i] - dot);
    }
}

int blocks_for(size_t n) { return static_cast<int>((n + kBlock - 1) / kBlock); }

} // namespace

float silu(float x) { return x / (1.0f + std::exp(-x)); }

float silu_derivative(float x) {
    const float s = 1.0f / (1.0f + std::exp(-x));
    return s + x * s * (1.0f - s);
}

void add(const float* a, const float* b, float* y, const AddParams& p) {
    const size_t n = p.size;
    if (n == 0) {
        return;
    }
    static DeviceBuffer dA, dB, dY;
    const float* d_a = cuda::upload(dA, a, n);
    const float* d_b = cuda::upload(dB, b, n);
    float* d_y = dY.reserve(n);
    add_kernel<<<blocks_for(n), kBlock>>>(d_a, d_b, d_y, static_cast<int>(n));
    cuda::sync();
    cuda::download(y, d_y, n);
}

void softmax(const float* logits, float* probs, const SoftmaxParams& p) {
    const size_t n = p.n;
    if (n == 0) {
        return;
    }
    static DeviceBuffer dL, dP;
    const float* d_l = cuda::upload(dL, logits, n);
    float* d_p = dP.reserve(n);
    softmax_kernel<<<1, kBlock>>>(d_l, d_p, static_cast<int>(n));
    cuda::sync();
    cuda::download(probs, d_p, n);
}

void silu_mul(const float* gate, const float* up, float* hidden, const SiluMulParams& p) {
    const size_t n = p.n;
    if (n == 0) {
        return;
    }
    static DeviceBuffer dGate, dUp, dHidden;
    const float* d_gate = cuda::upload(dGate, gate, n);
    const float* d_up = cuda::upload(dUp, up, n);
    float* d_hidden = dHidden.reserve(n);
    silu_mul_kernel<<<blocks_for(n), kBlock>>>(d_gate, d_up, d_hidden, static_cast<int>(n));
    cuda::sync();
    cuda::download(hidden, d_hidden, n);
}

void softmax_backward_row(const float* prob, const float* grad_prob, float* grad_logits, const SoftmaxParams& p) {
    const size_t n = p.n;
    if (n == 0) {
        return;
    }
    static DeviceBuffer dProb, dGradProb, dGradLogits;
    const float* d_prob = cuda::upload(dProb, prob, n);
    const float* d_grad_prob = cuda::upload(dGradProb, grad_prob, n);
    float* d_grad_logits = dGradLogits.reserve(n);
    softmax_backward_kernel<<<1, kBlock>>>(d_prob, d_grad_prob, d_grad_logits, static_cast<int>(n));
    cuda::sync();
    cuda::download(grad_logits, d_grad_logits, n);
}

void silu_mul_backward(const float* gate, const float* up, const float* grad_hidden, float* grad_gate, float* grad_up,
                       const SiluMulParams& p) {
    const size_t n = p.n;
    if (n == 0) {
        return;
    }
    static DeviceBuffer dGate, dUp, dGradHidden, dGradGate, dGradUp;
    const float* d_gate = cuda::upload(dGate, gate, n);
    const float* d_up = cuda::upload(dUp, up, n);
    const float* d_grad_hidden = cuda::upload(dGradHidden, grad_hidden, n);
    float* d_grad_gate = dGradGate.reserve(n);
    float* d_grad_up = dGradUp.reserve(n);
    silu_mul_backward_kernel<<<blocks_for(n), kBlock>>>(d_gate, d_up, d_grad_hidden, d_grad_gate, d_grad_up,
                                                        static_cast<int>(n));
    cuda::sync();
    cuda::download(grad_gate, d_grad_gate, n);
    cuda::download(grad_up, d_grad_up, n);
}

} // namespace kernel
