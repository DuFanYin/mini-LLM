#include "kernels/common.cuh"
#include "kernels/ops.h"

namespace kernels {

using namespace detail;

namespace {

constexpr unsigned kBlock = 256;

// One block per row.
__global__ void rms_norm_kernel(const float* x, std::size_t dim, const bf16* w, float eps, float* out) {
    const float* row = x + std::size_t{blockIdx.x} * dim;
    float* dst = out + std::size_t{blockIdx.x} * dim;
    float ss = 0.0f;
    for (std::size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        ss += row[i] * row[i];
    }
    const float inv = 1.0f / sqrtf(block_sum(ss) / static_cast<float>(dim) + eps);
    for (std::size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        dst[i] = row[i] * inv * (1.0f + to_float(w[i]));
    }
}

// One block per (token, head); blockDim.x == head_dim rounded up to a warp multiple.
__global__ void head_norm_rope_kernel(HeadView x, std::size_t heads, std::size_t head_dim, const bf16* w, float eps,
                                      std::size_t rotary_dim, double rope_theta, std::size_t pos0) {
    extern __shared__ float v[];
    const std::size_t t = blockIdx.x / heads;
    const std::size_t h = blockIdx.x % heads;
    float* base = x.base + t * x.token_stride + h * x.head_stride;
    const std::size_t d = threadIdx.x;
    const float val = d < head_dim ? base[d] : 0.0f;
    const float inv = 1.0f / sqrtf(block_sum(val * val) / static_cast<float>(head_dim) + eps);
    if (d < head_dim) {
        v[d] = val * inv * (1.0f + to_float(w[d]));
    }
    __syncthreads();
    if (d >= head_dim) {
        return;
    }
    if (d < rotary_dim) {
        // rotate-half: pair d with d ± rotary_dim/2; frequency index is d mod rotary_dim/2.
        const std::size_t half = rotary_dim / 2;
        const std::size_t f = d % half;
        const double inv_freq = 1.0 / pow(rope_theta, static_cast<double>(2 * f) / static_cast<double>(rotary_dim));
        const double angle = static_cast<double>(pos0 + t) * inv_freq;
        const float c = static_cast<float>(cos(angle));
        const float s = static_cast<float>(sin(angle));
        const float rotated = d < half ? -v[d + half] : v[d - half];
        base[d] = v[d] * c + rotated * s;
    } else {
        base[d] = v[d];
    }
}

// One block per row: out = x̂ · w · silu(z).
__global__ void gated_rms_norm_kernel(const float* x, const float* z, const float* w, float eps, std::size_t dim,
                                      float* out) {
    const std::size_t off = std::size_t{blockIdx.x} * dim;
    float ss = 0.0f;
    for (std::size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        ss += x[off + i] * x[off + i];
    }
    const float inv = 1.0f / sqrtf(block_sum(ss) / static_cast<float>(dim) + eps);
    for (std::size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        out[off + i] = x[off + i] * inv * w[i] * silu(z[off + i]);
    }
}

unsigned warp_multiple(std::size_t n) {
    return static_cast<unsigned>((n + kWarp - 1) / kWarp * kWarp);
}

} // namespace

void rms_norm(const float* x, std::size_t rows, std::size_t dim, const bf16* w, float eps, float* out,
              cudaStream_t stream) {
    rms_norm_kernel<<<static_cast<unsigned>(rows), kBlock, 0, stream>>>(x, dim, w, eps, out);
    check_launch();
}

void head_norm_rope(HeadView x, std::size_t tokens, std::size_t heads, std::size_t head_dim, const bf16* w, float eps,
                    std::size_t rotary_dim, double rope_theta, std::size_t pos0, cudaStream_t stream) {
    const unsigned block = warp_multiple(head_dim);
    head_norm_rope_kernel<<<static_cast<unsigned>(tokens * heads), block, head_dim * sizeof(float), stream>>>(
        x, heads, head_dim, w, eps, rotary_dim, rope_theta, pos0);
    check_launch();
}

void gated_rms_norm(const float* x, const float* z, const float* w, float eps, std::size_t rows, std::size_t dim,
                    float* out, cudaStream_t stream) {
    gated_rms_norm_kernel<<<static_cast<unsigned>(rows), warp_multiple(dim), 0, stream>>>(x, z, w, eps, dim, out);
    check_launch();
}

} // namespace kernels
