#include "kernels/common.cuh"
#include "kernels/ops.h"

namespace kernels {

using namespace detail;

namespace {

constexpr unsigned kBlock = 256;

__global__ void embedding_kernel(const std::int32_t* ids, std::size_t tokens, const bf16* table, std::size_t dim,
                                 float* out) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i >= tokens * dim) {
        return;
    }
    const std::size_t t = i / dim;
    const std::size_t d = i % dim;
    out[i] = to_float(table[static_cast<std::size_t>(ids[t]) * dim + d]);
}

__global__ void add_kernel(float* x, const float* y, std::size_t n) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += y[i];
    }
}

__global__ void silu_mul_kernel(const float* gate, const float* up, float* out, std::size_t n) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = silu(gate[i]) * up[i];
    }
}

__global__ void sigmoid_gate_kernel(float* x, HeadView gate, std::size_t tokens, std::size_t heads,
                                    std::size_t head_dim) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i >= tokens * heads * head_dim) {
        return;
    }
    const std::size_t t = i / (heads * head_dim);
    const std::size_t h = (i / head_dim) % heads;
    const std::size_t d = i % head_dim;
    x[i] *= sigmoid(gate.base[t * gate.token_stride + h * gate.head_stride + d]);
}

__global__ void kv_append_kernel(const float* k, const float* v, float* k_cache, float* v_cache, std::size_t tokens,
                                 std::size_t kv_heads, std::size_t head_dim, std::size_t pos0, std::size_t max_seq) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i >= tokens * kv_heads * head_dim) {
        return;
    }
    const std::size_t t = i / (kv_heads * head_dim);
    const std::size_t h = (i / head_dim) % kv_heads;
    const std::size_t d = i % head_dim;
    const std::size_t dst = (h * max_seq + pos0 + t) * head_dim + d;
    k_cache[dst] = k[i];
    v_cache[dst] = v[i];
}

} // namespace

void embedding(const std::int32_t* ids, std::size_t tokens, const bf16* table, std::size_t dim, float* out,
               cudaStream_t stream) {
    embedding_kernel<<<grid_for(tokens * dim, kBlock), kBlock, 0, stream>>>(ids, tokens, table, dim, out);
    check_launch();
}

void add_inplace(float* x, const float* y, std::size_t n, cudaStream_t stream) {
    add_kernel<<<grid_for(n, kBlock), kBlock, 0, stream>>>(x, y, n);
    check_launch();
}

void silu_mul(const float* gate, const float* up, float* out, std::size_t n, cudaStream_t stream) {
    silu_mul_kernel<<<grid_for(n, kBlock), kBlock, 0, stream>>>(gate, up, out, n);
    check_launch();
}

void sigmoid_gate(float* x, HeadView gate, std::size_t tokens, std::size_t heads, std::size_t head_dim,
                  cudaStream_t stream) {
    sigmoid_gate_kernel<<<grid_for(tokens * heads * head_dim, kBlock), kBlock, 0, stream>>>(x, gate, tokens, heads,
                                                                                            head_dim);
    check_launch();
}

void kv_append(const float* k, const float* v, float* k_cache, float* v_cache, std::size_t tokens, std::size_t kv_heads,
               std::size_t head_dim, std::size_t pos0, std::size_t max_seq, cudaStream_t stream) {
    kv_append_kernel<<<grid_for(tokens * kv_heads * head_dim, kBlock), kBlock, 0, stream>>>(
        k, v, k_cache, v_cache, tokens, kv_heads, head_dim, pos0, max_seq);
    check_launch();
}

} // namespace kernels
