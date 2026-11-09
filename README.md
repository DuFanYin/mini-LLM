# mini-LLM

A from-scratch, single-GPU inference engine for Qwen3.5-0.8B. No PyTorch, Triton, cuBLAS or CUTLASS: the engine is C++ and hand-written CUDA kernels on the CUDA runtime alone.

Plan and progress: `ROADMAP.md`. Performance log, one row per optimisation step: `docs/performance.md`. Architecture: `docs/architecture.html` (a 3D view of the model and a diagram of the code). Toolchain versions: `version.toml`.

## Learning goals

This is a learning project first. It exists so that its author learns, by building every layer by hand:

1. **GPU architecture and CUDA, from first principles.** The execution model (grids, blocks, warps), the memory hierarchy (registers, shared memory, L2, HBM/GDDR), coalescing, reductions, synchronisation, and later tensor cores (`mma.sync`, `ldmatrix`, `cp.async`, TMA) on Blackwell (sm_120).
2. **Performance engineering against a measured floor.** Every kernel is judged against what the hardware allows: decode against memory bandwidth, prefill against tensor-core throughput. Profile with Nsight, find the bottleneck, change one thing, measure again.
3. **How LLM inference really works.** The transformer block, RoPE, grouped-query attention and the KV cache, plus a modern hybrid: GatedDeltaNet linear attention with a fixed-size recurrent state.
4. **The techniques serving systems are built on.** Paged KV cache, continuous batching, CUDA graphs, speculative decoding (with the checkpoint's own MTP head) and weight quantization (fp8 / int4), each measured against the same floor.
5. **Engineering that keeps all of the above honest.** A NumPy reference verified against transformers, golden per-layer tests for every kernel, compute-sanitizer, warnings as errors, one CMake target per module.

Non-goals: training, multi-GPU, supporting many model families, competing with vLLM on breadth.

## Status

| Phase | State |
| --- | --- |
| 0 · strip the old training code, download script | done |
| 1 · load the checkpoint onto the GPU, every tensor checked | done |
| 2 · NumPy reference, verified against transformers | done |
| 3 · correct engine on naive kernels (every layer within 3.6e-6 of the reference) | done |
| 4 · bandwidth-bound decode | next — baseline 3.63 ms/token, 24 % of the 0.86 ms floor |

## Build and test

Needs nvcc (`/usr/local/cuda`) and `g++-15` on `PATH` (see `version.toml`).

```bash
cmake --preset release && cmake --build --preset release
ctest --preset release          # all tests
ctest --preset no-weights       # skip tests that need the checkpoint
```

There is also a `debug` preset; outputs go to `build/<preset>/`. Test labels: `cpu` (host only), `gpu` (needs a CUDA device), `weights` (also needs the checkpoint and golden files). Format with the repository's `.clang-format`. After changing a kernel, run compute-sanitizer:

```bash
compute-sanitizer --tool memcheck ./build/release/tests/kernel_tests   # also: racecheck, synccheck
```

## Weights and running

```bash
bash scripts/download_weights.sh     # into weights/qwen3.5-0.8b (git-ignored), pinned revision, sha256-checked
./build/release/apps/inspect         # load onto the GPU and print the config and weight statistics
./build/release/apps/generate        # greedy generation with timing; takes token ids, defaults to "The capital of France is"
```

`HF_TOKEN` adds an auth header to the download; `HF_ENDPOINT` selects a mirror.

## NumPy reference

`tools/reference` is the Qwen3.5 text forward pass in NumPy (f32, no torch), the engine's correctness oracle. It was checked once against transformers: every layer within 2.2e-6, identical greedy output.

```bash
cd tools/reference
uv run qwen35-ref generate "The capital of France is"   # greedy generation
uv run qwen35-ref dump                                   # writes weights/golden/*.safetensors (per-layer hidden states)
uv run pytest && uv run ruff check .
```

## Layout

```
src/core          device Tensor (f32 / bf16 / i32), Pool and Arena allocators, Stream, CudaError
src/io            JSON parser, safetensors (mmap) and multi-shard Checkpoint
src/model         Qwen3.5 Config and Weights (every shape and dtype checked at load)
src/kernels       hand-written CUDA kernels (the only .cu files); launch functions in ops.h
src/runtime       Executor: prefill / decode of one sequence and its state (KV cache, conv rows, recurrent state)
apps              inspect, generate
tests             gtest suites per module; runtime/golden_test.cpp checks every layer against the reference
tools/reference   NumPy reference and golden export (uv project)
docs              architecture.html, performance.md
```
