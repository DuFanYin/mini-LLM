#!/usr/bin/env bash
# Usage: ./scripts/configure.sh [scalar|accelerate]
#
# Configures CMake and builds everything into ./build. Default backend is
# Apple Accelerate on Darwin, otherwise the handwritten scalar path. Build type
# is always Release with -O2 -DNDEBUG.

set -euo pipefail

if [[ $# -gt 1 ]]; then
    echo "usage: ./scripts/configure.sh [scalar|accelerate]" >&2
    exit 2
fi

uname_s="$(uname -s)"
backend="${1:-}"

# Default backend: prefer Apple Accelerate when available, otherwise use the
# single handwritten scalar implementation.
if [[ -z "$backend" ]]; then
    if [[ "$uname_s" == "Darwin" ]]; then
        backend="accelerate"
    else
        backend="scalar"
    fi
fi

# Validate + map backend → selected source files.
gemm_driver=""
attention_src="src/kernel/attention.cpp"
core_src="src/kernel/core.cpp"
optimizer_src="src/train/optimizer.cpp"
use_accelerate="OFF"

case "$backend" in
    scalar)
        gemm_driver="src/kernel/gemm.cpp"
        ;;
    accelerate)
        if [[ "$uname_s" != "Darwin" ]]; then
            echo "configure.sh: backend=accelerate requires Darwin (Apple), got '$uname_s'" >&2
            exit 1
        fi
        gemm_driver="src/kernel/gemm_accelerate.cpp"
        attention_src="src/kernel/attention_accelerate.cpp"
        core_src="src/kernel/core_accelerate.cpp"
        optimizer_src="src/train/optimizer_accelerate.cpp"
        use_accelerate="ON"
        ;;
    *)
        echo "configure.sh: unknown backend '$backend' (use scalar|accelerate)" >&2
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
    -DMINI_LLM_USE_ACCELERATE="$use_accelerate"

cmake --build build -j
