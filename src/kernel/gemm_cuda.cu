#include "kernel/kernel.h"

#include "kernel/cuda_util.cuh"

// CUDA backend for the GEMM-family kernels (handwritten, no cuBLAS).
//
// Public signatures match the scalar gemm.cpp exactly (host pointers in/out):
//   - gemm_nt        : C[M,N]  = A[M,K] * B[N,K]^T          (overwrites C)
//   - linear         : y[M,N]  = x[M,K] * w[N,K]^T + bias   (overwrites y)
//   - linear_backward: grad_w, dx, grad_b are ACCUMULATED   (+= contributions)
//
// All output matrices here are contiguous (leading dim == width), matching how
// the scalar path is invoked from the executor.

namespace kernel {

namespace {

using cuda::DeviceBuffer;

constexpr int kTile = 16;

// C[m,n] = (bias ? bias[n] : 0) + sum_k A[m,k] * B[n,k].  A is [M,K], B is [N,K]
// row-major; B is read transposed. Tiled over the contraction dim K.
__global__ void nt_kernel(const float* A, const float* B, const float* bias, float* C, int M, int N, int K) {
    __shared__ float as[kTile][kTile];
    __shared__ float bs[kTile][kTile];

    const int row = blockIdx.y * kTile + threadIdx.y; // m
    const int col = blockIdx.x * kTile + threadIdx.x; // n

    float acc = 0.0f;
    for (int kt = 0; kt < K; kt += kTile) {
        const int ak = kt + threadIdx.x;
        as[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
        const int bk = kt + threadIdx.y;
        bs[threadIdx.y][threadIdx.x] = (col < N && bk < K) ? B[col * K + bk] : 0.0f;
        __syncthreads();

        for (int i = 0; i < kTile; ++i) {
            acc += as[threadIdx.y][i] * bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = (bias != nullptr ? bias[col] : 0.0f) + acc;
    }
}

// grad_w[N,K] += sum_m dy[m,n] * x[m,k].  dy is [M,N], x is [M,K] row-major.
// Tiled over the contraction dim M. (A^T * B style.)
__global__ void grad_w_kernel(const float* dy, const float* x, float* grad_w, int M, int N, int K) {
    __shared__ float ds[kTile][kTile];
    __shared__ float xs[kTile][kTile];

    const int row = blockIdx.y * kTile + threadIdx.y; // n
    const int col = blockIdx.x * kTile + threadIdx.x; // k

    float acc = 0.0f;
    for (int mt = 0; mt < M; mt += kTile) {
        const int dm = mt + threadIdx.x;
        ds[threadIdx.y][threadIdx.x] = (dm < M && row < N) ? dy[dm * N + row] : 0.0f;
        const int xm = mt + threadIdx.y;
        xs[threadIdx.y][threadIdx.x] = (xm < M && col < K) ? x[xm * K + col] : 0.0f;
        __syncthreads();

        for (int i = 0; i < kTile; ++i) {
            acc += ds[threadIdx.y][i] * xs[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < K) {
        grad_w[row * K + col] += acc;
    }
}

// dx[M,K] += sum_n dy[m,n] * w[n,k].  dy is [M,N], w is [N,K] row-major.
// Tiled over the contraction dim N.
__global__ void dx_kernel(const float* dy, const float* w, float* dx, int M, int N, int K) {
    __shared__ float ds[kTile][kTile];
    __shared__ float ws[kTile][kTile];

    const int row = blockIdx.y * kTile + threadIdx.y; // m
    const int col = blockIdx.x * kTile + threadIdx.x; // k

    float acc = 0.0f;
    for (int nt = 0; nt < N; nt += kTile) {
        const int dn = nt + threadIdx.x;
        ds[threadIdx.y][threadIdx.x] = (row < M && dn < N) ? dy[row * N + dn] : 0.0f;
        const int wn = nt + threadIdx.y;
        ws[threadIdx.y][threadIdx.x] = (wn < N && col < K) ? w[wn * K + col] : 0.0f;
        __syncthreads();

        for (int i = 0; i < kTile; ++i) {
            acc += ds[threadIdx.y][i] * ws[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        dx[row * K + col] += acc;
    }
}

// grad_b[n] += sum_m dy[m,n]. One thread per output channel n.
__global__ void grad_b_kernel(const float* dy, float* grad_b, int M, int N) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) {
        return;
    }
    float acc = 0.0f;
    for (int m = 0; m < M; ++m) {
        acc += dy[m * N + n];
    }
    grad_b[n] += acc;
}

dim3 grid_for(int rows, int cols) {
    return dim3((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
}

// Run the forward NT matmul: C[M,N] = A[M,K] * B[N,K]^T + optional bias.
void launch_nt(const float* A, const float* B, const float* bias, float* C, size_t M, size_t N, size_t K) {
    static DeviceBuffer dA, dB, dBias, dC;
    const float* d_a = cuda::upload(dA, A, M * K);
    const float* d_b = cuda::upload(dB, B, N * K);
    const float* d_bias = (bias != nullptr) ? cuda::upload(dBias, bias, N) : nullptr;
    float* d_c = dC.reserve(M * N);

    const dim3 block(kTile, kTile);
    nt_kernel<<<grid_for(static_cast<int>(M), static_cast<int>(N)), block>>>(
        d_a, d_b, d_bias, d_c, static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
    cuda::sync();
    cuda::download(C, d_c, M * N);
}

} // namespace

void gemm_nt(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
    if (M == 0 || N == 0 || K == 0) {
        return;
    }
    launch_nt(A, B, nullptr, C, M, N, K);
}

void linear(const float* x, const float* w, const float* bias, float* y, const LinearParams& p) {
    const size_t M = p.rows;
    const size_t N = p.out_dim;
    const size_t K = p.in_dim;
    if (M == 0 || N == 0 || K == 0) {
        return;
    }
    launch_nt(x, w, bias, y, M, N, K);
}

void linear_backward(const float* x, const float* w, bool has_bias, const float* dy, float* dx, float* grad_w,
                     float* grad_b, const LinearParams& p) {
    const size_t M = p.rows;
    const size_t N = p.out_dim;
    const size_t K = p.in_dim;
    if (M == 0 || N == 0 || K == 0) {
        return;
    }

    static DeviceBuffer dX, dW, dDy, dDx, dGw, dGb;
    const float* d_x = cuda::upload(dX, x, M * K);
    const float* d_w = cuda::upload(dW, w, N * K);
    const float* d_dy = cuda::upload(dDy, dy, M * N);
    // Outputs accumulate, so seed device buffers with the caller's current values.
    float* d_dx = cuda::upload(dDx, dx, M * K);
    float* d_gw = cuda::upload(dGw, grad_w, N * K);

    const dim3 block(kTile, kTile);
    grad_w_kernel<<<grid_for(static_cast<int>(N), static_cast<int>(K)), block>>>(
        d_dy, d_x, d_gw, static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
    dx_kernel<<<grid_for(static_cast<int>(M), static_cast<int>(K)), block>>>(
        d_dy, d_w, d_dx, static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));

    float* d_gb = nullptr;
    if (has_bias && grad_b != nullptr) {
        constexpr int kThreads = 128;
        d_gb = cuda::upload(dGb, grad_b, N);
        grad_b_kernel<<<(static_cast<int>(N) + kThreads - 1) / kThreads, kThreads>>>(
            d_dy, d_gb, static_cast<int>(M), static_cast<int>(N));
    }

    cuda::sync();
    cuda::download(dx, d_dx, M * K);
    cuda::download(grad_w, d_gw, N * K);
    if (d_gb != nullptr) {
        cuda::download(grad_b, d_gb, N);
    }
}

} // namespace kernel
