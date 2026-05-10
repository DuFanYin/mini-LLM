#!/usr/bin/env bash
# Usage: ./scripts/configure.sh [neon|avx2|scalar|accelerate]
#
# Configures CMake and builds everything into ./build. Default backend is
# Apple Accelerate on Darwin, otherwise the best handwritten variant for the
# host arch. Build type is always Release with -O2 -DNDEBUG.

set -euo pipefail

if [[ $# -gt 1 ]]; then
    echo "usage: ./scripts/configure.sh [neon|avx2|scalar|accelerate]" >&2
    exit 2
fi

uname_s="$(uname -s)"
uname_m="$(uname -m)"
backend="${1:-}"

# Default backend: prefer Apple Accelerate when available, otherwise pick the
# best handwritten variant for the host arch.
if [[ -z "$backend" ]]; then
    if [[ "$uname_s" == "Darwin" ]]; then
        backend="accelerate"
    else
        case "$uname_m" in
            arm64|aarch64)        backend="neon" ;;
            x86_64|amd64|AMD64)   backend="avx2" ;;
            *)                    backend="scalar" ;;
        esac
    fi
fi

# Validate + map backend → (driver src, prims src, prims compile flags, accelerate?).
gemm_driver=""
gemm_prims=""
gemm_prims_flags=""
attention_src="src/kernel/attention.cpp"
core_src="src/kernel/core.cpp"
use_accelerate="OFF"

case "$backend" in
    neon)
        gemm_driver="src/kernel/gemm.cpp"
        gemm_prims="src/kernel/gemm_neon.cpp"
        ;;
    avx2)
        gemm_driver="src/kernel/gemm.cpp"
        gemm_prims="src/kernel/gemm_avx2.cpp"
        gemm_prims_flags="-mavx2;-mfma"
        ;;
    scalar)
        gemm_driver="src/kernel/gemm.cpp"
        gemm_prims="src/kernel/gemm_scalar.cpp"
        ;;
    accelerate)
        if [[ "$uname_s" != "Darwin" ]]; then
            echo "configure.sh: backend=accelerate requires Darwin (Apple), got '$uname_s'" >&2
            exit 1
        fi
        gemm_driver="src/kernel/gemm_accelerate.cpp"
        gemm_prims=""
        attention_src="src/kernel/attention_accelerate.cpp"
        core_src="src/kernel/core_accelerate.cpp"
        use_accelerate="ON"
        ;;
    *)
        echo "configure.sh: unknown backend '$backend' (use neon|avx2|scalar|accelerate)" >&2
        exit 2
        ;;
esac

echo "configure.sh: backend=$backend build_type=Release cxx_flags='-O2 -DNDEBUG' build_dir=build"

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_CXX_FLAGS_RELEASE=-O2 -DNDEBUG" \
    -DMINI_LLM_KERNEL_BACKEND="$backend" \
    -DMINI_LLM_GEMM_DRIVER_SRC="$gemm_driver" \
    -DMINI_LLM_GEMM_PRIMS_SRC="$gemm_prims" \
    -DMINI_LLM_GEMM_PRIMS_FLAGS="$gemm_prims_flags" \
    -DMINI_LLM_ATTENTION_SRC="$attention_src" \
    -DMINI_LLM_CORE_SRC="$core_src" \
    -DMINI_LLM_USE_ACCELERATE="$use_accelerate"

cmake --build build -j
