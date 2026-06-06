#include "kernel/kernel.h"

#include <cmath>

// Device-agnostic scalar helpers shared by every backend (compiled
// unconditionally). The SiLU activation and its derivative are plain host
// functions; backends that run on the GPU keep their own __device__ copies for
// in-kernel use.

namespace kernel {

float silu(float x) { return x / (1.0f + std::exp(-x)); }

float silu_derivative(float x) {
    const float s = 1.0f / (1.0f + std::exp(-x));
    return s + x * s * (1.0f - s);
}

} // namespace kernel
