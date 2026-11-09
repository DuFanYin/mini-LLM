#include "kernels/common.cuh"
#include "kernels/ops.h"

namespace kernels {

using namespace detail;

namespace {

constexpr unsigned kBlock = 1024;

struct Best {
    float value;
    std::int32_t index;
};

__device__ __forceinline__ Best better(Best a, Best b) {
    // Larger value wins; on a tie the lower index wins, matching numpy.argmax.
    if (b.value > a.value || (b.value == a.value && b.index < a.index)) {
        return b;
    }
    return a;
}

// Single block.
__global__ void argmax_kernel(const float* x, std::size_t n, std::int32_t* out) {
    __shared__ Best partial[kWarp];
    Best best{-INFINITY, INT32_MAX};
    for (std::size_t i = threadIdx.x; i < n; i += blockDim.x) {
        best = better(best, Best{x[i], static_cast<std::int32_t>(i)});
    }
    for (unsigned offset = kWarp / 2; offset > 0; offset /= 2) {
        const Best other{__shfl_xor_sync(0xffffffffu, best.value, offset),
                         __shfl_xor_sync(0xffffffffu, best.index, offset)};
        best = better(best, other);
    }
    const unsigned lane = threadIdx.x % kWarp;
    const unsigned warp = threadIdx.x / kWarp;
    if (lane == 0) {
        partial[warp] = best;
    }
    __syncthreads();
    if (warp == 0) {
        best = lane < blockDim.x / kWarp ? partial[lane] : Best{-INFINITY, INT32_MAX};
        for (unsigned offset = kWarp / 2; offset > 0; offset /= 2) {
            const Best other{__shfl_xor_sync(0xffffffffu, best.value, offset),
                             __shfl_xor_sync(0xffffffffu, best.index, offset)};
            best = better(best, other);
        }
        if (lane == 0) {
            *out = best.index;
        }
    }
}

} // namespace

void argmax(const float* x, std::size_t n, std::int32_t* out, cudaStream_t stream) {
    argmax_kernel<<<1, kBlock, 0, stream>>>(x, n, out);
    check_launch();
}

} // namespace kernels
