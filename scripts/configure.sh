#!/usr/bin/env bash
# Usage: ./scripts/configure.sh [scalar|cuda]
#
# Configures CMake and builds everything into ./build. Default backend is the
# CUDA path when nvcc is available, otherwise the handwritten scalar path. Build
# type is always Release with -O2 -DNDEBUG.
#
# Backends:
#   scalar : all kernels on the CPU (handwritten scalar).
#   cuda   : GEMM + attention on the GPU (handwritten CUDA kernels); norm, rope,
#            core element-wise ops, and the optimizer stay on the scalar path.

set -euo pipefail

if [[ $# -gt 1 ]]; then
    echo "usage: ./scripts/configure.sh [scalar|cuda]" >&2
    exit 2
fi

backend="${1:-}"

# Default backend: prefer CUDA when nvcc is on PATH, otherwise scalar.
if [[ -z "$backend" ]]; then
    if command -v nvcc >/dev/null 2>&1; then
        backend="cuda"
    else
        backend="scalar"
    fi
fi

# Validate + map backend → selected source files. The cuda backend routes GEMM,
# attention, the core element-wise/softmax kernels, and the optimizer onto the
# GPU; the scalar backend keeps everything on the CPU.
gemm_driver=""
attention_src="src/kernel/attention.cpp"
core_src="src/kernel/core.cpp"
optimizer_src="src/train/optimizer.cpp"
use_cuda="OFF"

case "$backend" in
    scalar)
        gemm_driver="src/kernel/gemm.cpp"
        ;;
    cuda)
        if ! command -v nvcc >/dev/null 2>&1; then
            echo "configure.sh: backend=cuda requires nvcc on PATH" >&2
            exit 1
        fi
        gemm_driver="src/kernel/gemm_cuda.cu"
        attention_src="src/kernel/attention_cuda.cu"
        core_src="src/kernel/core_cuda.cu"
        optimizer_src="src/train/optimizer_cuda.cu"
        use_cuda="ON"
        ;;
    *)
        echo "configure.sh: unknown backend '$backend' (use scalar|cuda)" >&2
        exit 2
        ;;
esac

echo "configure.sh: backend=$backend build_type=Release cxx_flags='-O2 -DNDEBUG' build_dir=build"

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_CXX_FLAGS_RELEASE=-O2 -DNDEBUG" \
    -DMINI_LLM_KERNEL_BACKEND="$backend" \
    -DMINI_LLM_GEMM_DRIVER_SRC="$gemm_driver" \
    -DMINI_LLM_ATTENTION_SRC="$attention_src" \
    -DMINI_LLM_CORE_SRC="$core_src" \
    -DMINI_LLM_OPTIMIZER_SRC="$optimizer_src" \
    -DMINI_LLM_USE_CUDA="$use_cuda"

cmake --build build -j
