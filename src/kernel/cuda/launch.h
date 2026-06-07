#pragma once

#include <cstddef>

// Device-pointer kernel entry points: all pointers are ALREADY in device memory.
// These launch on the default stream and do NOT copy host<->device and do NOT
// synchronize — the caller owns residency and ordering. This is the Phase 0
// building block for the device-resident inference path. Host-clean header (no
// CUDA types) so plain .cpp callers can include it.

namespace kernel::cuda {

// C[M,N] = A[M,K] * B[N,K]^T (row-major; B read transposed). Overwrites C.
void gemm_nt_device(const float* A, const float* B, float* C, std::size_t M, std::size_t N, std::size_t K);

// y[M,N] = x[M,K] * w[N,K]^T + (bias ? bias[N] : 0). Overwrites y. bias may be null.
void linear_device(const float* x, const float* w, const float* bias, float* y, std::size_t M, std::size_t N,
                   std::size_t K);

} // namespace kernel::cuda
