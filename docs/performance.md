# Performance log

Every optimisation step adds one row to the log, with the same measurements under the same conditions, and is judged against the bandwidth floor. Detail for a step goes in its own section below the log.

## The floor

Single-stream decode reads every bf16 text weight once per token (1.505 GB, the tied lm_head included) and reads and writes the GatedDeltaNet state (~38 MB). On the RTX PRO 6000 at ~1.8 TB/s that is **≈ 0.86 ms/token (≈ 1160 tok/s)**. `% floor` below is 0.86 ms divided by the measured time. Prefill is compute-bound and will be judged against tensor-core throughput once Phase 5 starts.

## How to measure

```bash
cmake --preset release && cmake --build --preset release
./build/release/apps/generate --max-new 128                    # decode: ms/token over 127 steps; prefill of 5 tokens
./build/release/apps/generate --max-new 1 $(python3 -c "print(' '.join(str(760 + i % 500) for i in range(128)))")   # prefill 128
# same with range(512) for prefill 512
nsys profile -o decode --trace=cuda ./build/release/apps/generate --max-new 128
nsys stats --report cuda_gpu_kern_sum decode.nsys-rep           # per-kernel totals
nsys stats --report cuda_gpu_trace decode.nsys-rep              # per launch; group linear_kernel by grid size for shapes
```

Report the median of three runs. Decode numbers come from the 127 steps after the prompt, at contexts of 6 to 133 tokens.

**Conditions.** RTX PRO 6000 Blackwell Max-Q (sm_120, 188 SMs), driver 610.43.02, CUDA 13.0.88, GCC 15.3, `release` preset. SM clock 2137 MHz and memory clock 13365 MHz observed during runs. The GPU is shared: another process holds ~88 GB and shows ~17 % utilisation while idle, so numbers carry a few percent of noise.

## Log

| Date | Step | Decode ms/token | tok/s | % floor | Prefill 5 / 128 / 512 tokens (ms) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-15 | Phase 3 baseline: naive kernels | 3.65 | 274 | 24 % | 7.4 / 98 / 390 | correctness-first; details below |

## Baseline: naive kernels (Phase 3)

One decode token runs 412 kernel launches (187 of them `linear`). GPU time per token from nsys is 3.63 ms, the same as the wall-clock measurement, so launch overhead is hidden behind kernel time at this speed.

**Where a decode token's time goes** (nsys over 128 decode steps):

| Kernel | Share | Per token | Note |
| --- | --- | --- | --- |
| `linear` | 47 % | 1.70 ms | 187 launches; 0.89 TB/s averaged over 1.505 GB of weights (49 % of peak) |
| `gdn_recurrence` | 33 % | 1.20 ms | 18 × 66.5 µs; only 16 blocks (one per head) on 188 SMs, two serial passes over S in global memory, ~63 GB/s |
| `attention` | 5 % | 0.17 ms | 6 × 29 µs at short context; 8 blocks, serial dot products per thread |
| `rms_norm` | 5 % | 0.17 ms | 49 launches × 3.4 µs |
| `head_norm_rope` | 2 % | 0.08 ms | |
| `argmax` | 2 % | 0.08 ms | one block over 248 320 logits |
| everything else | 6 % | 0.22 ms | conv1d, gates, adds, SiLU·mul, gated norm, KV append, embedding |

**`linear` by shape** (decode, one row; median per launch):

| Projection | Weight | Launches / token | µs each | Achieved | % of 1.8 TB/s |
| --- | --- | --- | --- | --- | --- |
| lm_head | 248320 × 1024 | 1 | 314 | 1.62 TB/s | 90 % |
| in_proj_qkv | 6144 × 1024 | 18 | 9.5 | 1.32 TB/s | 74 % |
| gate, up | 3584 × 1024 | 48 | 6.3 | 1.17 TB/s | 65 % |
| q_proj | 4096 × 1024 | 6 | 7.2 | 1.17 TB/s | 65 % |
| in_proj_z | 2048 × 1024 | 18 | 5.3 | 0.79 TB/s | 44 % |
| down, out_proj, o_proj | 1024 × 3584 / 2048 | 48 | 13.4 | 0.43 TB/s | 24 % |
| k_proj, v_proj | 512 × 1024 | 12 | 4.9 | 0.21 TB/s | 12 % |
| in_proj_a, in_proj_b | 16 × 1024 | 36 | 4.9 | — | latency only |

**What Phase 4 should attack first, by size of the win:**

1. `gdn_recurrence` (1.2 ms → the state traffic alone is ~40 MB, ~25 µs at peak): spread each head over many blocks, keep S in registers / shared memory, one pass.
2. Projections with 1024 outputs (0.64 ms): 128 blocks of long dot products cannot fill the GPU; split K or give each output more warps.
3. Tiny projections (a, b, k, v: ~0.24 ms of pure latency): fuse them into their neighbours (`qkv | z | a | b`, `q | k | v`).
4. `argmax`: one block over 248 320 values; a two-stage reduction, later fused into the lm_head.
