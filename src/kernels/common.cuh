#pragma once

// Device helpers shared by the kernel translation units.

#include "core/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace kernels::detail {

constexpr unsigned kWarp = 32;

__device__ __forceinline__ float to_float(float v) {
    return v;
}
__device__ __forceinline__ float to_float(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (unsigned offset = kWarp / 2; offset > 0; offset /= 2) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (unsigned offset = kWarp / 2; offset > 0; offset /= 2) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

// Sum over the block; every thread gets the result. blockDim.x must be a multiple of 32.
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float partial[kWarp];
    const unsigned lane = threadIdx.x % kWarp;
    const unsigned warp = threadIdx.x / kWarp;
    v = warp_sum(v);
    __syncthreads(); // partial may still be read by a previous call
    if (lane == 0) {
        partial[warp] = v;
    }
    __syncthreads();
    const unsigned warps = blockDim.x / kWarp;
    v = lane < warps ? partial[lane] : 0.0f;
    return warp_sum(v);
}

__device__ __forceinline__ float block_max(float v) {
    __shared__ float partial[kWarp];
    const unsigned lane = threadIdx.x % kWarp;
    const unsigned warp = threadIdx.x / kWarp;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) {
        partial[warp] = v;
    }
    __syncthreads();
    const unsigned warps = blockDim.x / kWarp;
    v = lane < warps ? partial[lane] : -INFINITY;
    return warp_max(v);
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}
__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

inline unsigned grid_for(std::size_t n, unsigned block) {
    return static_cast<unsigned>((n + block - 1) / block);
}

inline void check_launch() {
    core::cuda_check(cudaGetLastError());
}

} // namespace kernels::detail
