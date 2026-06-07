#include "kernel/cuda/device.cuh"
#include "kernel/cuda/launch.h"

// Device RMSNorm + RoPE for the resident inference path. Mirrors the scalar
// math in src/kernel/scalar/norm_rope.cpp. Compiled into the cuda backend only.

namespace kernel {
namespace {

constexpr int kBlock = 256;

// One block per row. y = x / sqrt(mean(x^2) + eps) * weight.
__global__ void rms_norm_kernel(const float* x, const float* weight, float eps, float* y, int d_model) {
    __shared__ float red[kBlock];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* xr = x + static_cast<long>(row) * d_model;
    float* yr = y + static_cast<long>(row) * d_model;

    float local = 0.0f;
    for (int c = tid; c < d_model; c += blockDim.x) {
        const float v = xr[c];
        local += v * v;
    }
    red[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] += red[tid + s];
        }
        __syncthreads();
    }
    const float mean_sq = red[0] / static_cast<float>(d_model);
    const float inv = 1.0f / sqrtf(mean_sq + eps);
    __syncthreads();

    for (int c = tid; c < d_model; c += blockDim.x) {
        yr[c] = xr[c] * inv * weight[c];
    }
}

// One thread per (si, head, freq-pair). Positions are contiguous: pos = past_len + si.
// `total` is the exact thread count (seq_len * num_heads * pairs); tail threads in the
// last block are masked out.
__global__ void rope_kernel(float* x, int past_len, const float* cos_tab, const float* sin_tab, int half, int pairs,
                            int num_heads, int head_dim, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int per_seq = num_heads * pairs;
    const int si = idx / per_seq;
    const int rem = idx % per_seq;
    const int hi = rem / pairs;
    const int fi = rem % pairs;
    const int pos = past_len + si;
    const float ct = cos_tab[pos * half + fi];
    const float st = sin_tab[pos * half + fi];
    const int i = fi * 2;
    const long base = (static_cast<long>(si) * num_heads + hi) * head_dim + i;
    const float x0 = x[base];
    const float x1 = x[base + 1];
    x[base] = x0 * ct - x1 * st;
    x[base + 1] = x0 * st + x1 * ct;
}

} // namespace

namespace cuda {

void rms_norm_device(const float* x, const float* weight, float eps, float* y, size_t rows, size_t d_model) {
    if (rows == 0 || d_model == 0) {
        return;
    }
    rms_norm_kernel<<<static_cast<unsigned int>(rows), kBlock>>>(x, weight, eps, y, static_cast<int>(d_model));
}

void apply_rope_device(float* x, size_t past_len, const float* cos_tab, const float* sin_tab, size_t half,
                       size_t eff_rot, size_t seq_len, size_t num_heads, size_t head_dim) {
    const size_t pairs = eff_rot / 2;
    const size_t total = seq_len * num_heads * pairs;
    if (total == 0) {
        return;
    }
    const unsigned int blocks = static_cast<unsigned int>((total + kBlock - 1) / kBlock);
    rope_kernel<<<blocks, kBlock>>>(x, static_cast<int>(past_len), cos_tab, sin_tab, static_cast<int>(half),
                                    static_cast<int>(pairs), static_cast<int>(num_heads), static_cast<int>(head_dim),
                                    static_cast<int>(total));
}

} // namespace cuda

} // namespace kernel
