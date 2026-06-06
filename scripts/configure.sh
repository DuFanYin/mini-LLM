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
# attention, and the core element-wise/softmax/optimizer-leaf kernels onto the
# GPU; the scalar backend keeps everything on the CPU. The optimizer driver
# (src/train/optimizer.cpp) is backend-agnostic and lives in CMakeLists directly.
gemm_driver=""
attention_src="src/kernel/scalar/attention.cpp"
core_src="src/kernel/scalar/core.cpp"
use_cuda="OFF"

case "$backend" in
    scalar)
        gemm_driver="src/kernel/scalar/gemm.cpp"
        ;;
    cuda)
        if ! command -v nvcc >/dev/null 2>&1; then
            echo "configure.sh: backend=cuda requires nvcc on PATH" >&2
            exit 1
        fi
        gemm_driver="src/kernel/cuda/gemm.cu"
        attention_src="src/kernel/cuda/attention.cu"
        core_src="src/kernel/cuda/core.cu"
        use_cuda="ON"
        ;;
    *)
        echo "configure.sh: unknown backend '$backend' (use scalar|cuda)" >&2
        exit 2
        ;;
esac

# The apps use C++23 <print>, which needs libstdc++ >= 14. Prefer the newest
# such GCC available; nvcc's host compiler is pointed at the same one. If only an
# older default g++ exists, fall back to it (the libraries/tests still build; the
# <print>-using apps will not).
compiler_args=()
for cand in g++-15 g++-14; do
    if command -v "$cand" >/dev/null 2>&1; then
        cc="${cand/g++/gcc}"
        compiler_args+=("-DCMAKE_CXX_COMPILER=$cand" "-DCMAKE_C_COMPILER=$cc")
        if [[ "$use_cuda" == "ON" ]]; then
            compiler_args+=("-DCMAKE_CUDA_HOST_COMPILER=$cand")
        fi
        echo "configure.sh: using host compiler $cand"
        break
    fi
done

echo "configure.sh: backend=$backend build_type=Release cxx_flags='-O2 -DNDEBUG' build_dir=build"

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_CXX_FLAGS_RELEASE=-O2 -DNDEBUG" \
    "${compiler_args[@]}" \
    -DMINI_LLM_KERNEL_BACKEND="$backend" \
    -DMINI_LLM_GEMM_DRIVER_SRC="$gemm_driver" \
    -DMINI_LLM_ATTENTION_SRC="$attention_src" \
    -DMINI_LLM_CORE_SRC="$core_src" \
    -DMINI_LLM_USE_CUDA="$use_cuda"

cmake --build build -j
