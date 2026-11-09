#include "kernels/common.cuh"
#include "kernels/ops.h"

namespace kernels {

using namespace detail;

namespace {

constexpr unsigned kBlock = 256; // 8 warps

// One warp per output element; lanes stride over the reduction dimension.
__global__ void linear_kernel(const float* x, std::size_t rows, std::size_t in, const bf16* w, std::size_t out,
                              float* y) {
    const std::size_t warp = (std::size_t{blockIdx.x} * blockDim.x + threadIdx.x) / kWarp;
    const unsigned lane = threadIdx.x % kWarp;
    if (warp >= rows * out) {
        return;
    }
    const std::size_t r = warp / out;
    const std::size_t n = warp % out;
    const float* xr = x + r * in;
    const bf16* wn = w + n * in;
    float acc = 0.0f;
    for (std::size_t k = lane; k < in; k += kWarp) {
        acc += xr[k] * to_float(wn[k]);
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        y[warp] = acc;
    }
}

} // namespace

void linear(const float* x, std::size_t rows, std::size_t in, const bf16* w, std::size_t out, float* y,
            cudaStream_t stream) {
    const std::size_t threads = rows * out * kWarp;
    linear_kernel<<<grid_for(threads, kBlock), kBlock, 0, stream>>>(x, rows, in, w, out, y);
    check_launch();
}

} // namespace kernels
