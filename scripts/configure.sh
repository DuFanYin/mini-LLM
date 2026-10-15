#!/usr/bin/env bash
# Usage: ./scripts/configure.sh
#
# Configures CMake (Release, -O2 -DNDEBUG) and builds everything into ./build.
# Needs nvcc; /usr/local/cuda/bin is added to PATH when nvcc is not on it.

set -euo pipefail

if ! command -v nvcc >/dev/null 2>&1 && [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "configure.sh: nvcc not found" >&2
    exit 1
fi

# C++23 <print> needs libstdc++ >= 14. Prefer the newest such GCC on PATH and
# point nvcc's host compiler at the same one.
compiler_args=()
for cand in g++-15 g++-14; do
    if command -v "$cand" >/dev/null 2>&1; then
        compiler_args+=("-DCMAKE_CXX_COMPILER=$cand" "-DCMAKE_C_COMPILER=${cand/g++/gcc}"
                        "-DCMAKE_CUDA_HOST_COMPILER=$cand")
        echo "configure.sh: using host compiler $cand"
        break
    fi
done

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_CXX_FLAGS_RELEASE=-O2 -DNDEBUG" \
    "${compiler_args[@]}"

cmake --build build -j
