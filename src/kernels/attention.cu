#include "kernels/common.cuh"
#include "kernels/ops.h"

#include <stdexcept>

namespace kernels {

using namespace detail;

namespace {

constexpr unsigned kBlock = 256;

// One block per (token, query head). Shared memory: q[head_dim] then scores[visible keys].
__global__ void attention_kernel(HeadView q, const float* k_cache, const float* v_cache, std::size_t max_seq,
                                 float* out, std::size_t heads, std::size_t kv_heads, std::size_t head_dim,
                                 std::size_t pos0) {
    extern __shared__ float smem[];
    float* qs = smem;
    float* scores = smem + head_dim;
    const std::size_t t = blockIdx.x / heads;
    const std::size_t h = blockIdx.x % heads;
    const std::size_t kvh = h / (heads / kv_heads);
    const std::size_t visible = pos0 + t + 1;
    const float* qp = q.base + t * q.token_stride + h * q.head_stride;
    const float* kc = k_cache + kvh * max_seq * head_dim;
    const float* vc = v_cache + kvh * max_seq * head_dim;

    for (std::size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        qs[d] = qp[d];
    }
    __syncthreads();

    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    float local_max = -INFINITY;
    for (std::size_t l = threadIdx.x; l < visible; l += blockDim.x) {
        float dot = 0.0f;
        for (std::size_t d = 0; d < head_dim; ++d) {
            dot += qs[d] * kc[l * head_dim + d];
        }
        scores[l] = dot * scale;
        local_max = fmaxf(local_max, scores[l]);
    }
    const float m = block_max(local_max);
    float local_sum = 0.0f;
    for (std::size_t l = threadIdx.x; l < visible; l += blockDim.x) {
        scores[l] = expf(scores[l] - m);
        local_sum += scores[l];
    }
    const float inv_sum = 1.0f / block_sum(local_sum); // block_sum syncs, so all scores are written
    float* o = out + (t * heads + h) * head_dim;
    for (std::size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (std::size_t l = 0; l < visible; ++l) {
            acc += scores[l] * vc[l * head_dim + d];
        }
        o[d] = acc * inv_sum;
    }
}

} // namespace

void attention(HeadView q, const float* k_cache, const float* v_cache, std::size_t max_seq, float* out,
               std::size_t tokens, std::size_t heads, std::size_t kv_heads, std::size_t head_dim, std::size_t pos0,
               cudaStream_t stream) {
    const std::size_t smem = (head_dim + pos0 + tokens) * sizeof(float);
    if (smem > 48 * 1024) {
        throw std::length_error("attention: context exceeds the naive kernel's shared-memory score buffer");
    }
    attention_kernel<<<static_cast<unsigned>(tokens * heads), kBlock, smem, stream>>>(q, k_cache, v_cache, max_seq, out,
                                                                                      heads, kv_heads, head_dim, pos0);
    check_launch();
}

} // namespace kernels
