# Roadmap

Status: 2026-10-15. The plan for turning mini-LLM into a from-scratch, single-GPU inference engine for **Qwen3.5-0.8B**. Each phase gets its own detailed plan under `docs/` when it starts; this file is the map and the checklist.

## 1. Goal and ground rules

**North star.** Load the official Qwen3.5-0.8B bf16 checkpoint, tokenize real text, and generate with single-stream decode close to the GPU's memory-bandwidth floor, on an engine whose every kernel is hand-written.

- No PyTorch, Triton, cuBLAS, cuDNN or CUTLASS in the engine: CUDA runtime plus hand-written kernels, inline PTX allowed. cuBLAS may appear only in benchmarks as a baseline.
- Inference only. Text only until Phase 9; the vision encoder and the MTP head are skipped at load.
- Correctness is checked against a NumPy reference of Qwen3.5, never against the engine's own code.
- Non-goals: multi-GPU, training, serving many model families.

## 2. The model

Checkpoint: `Qwen/Qwen3.5-0.8B` at revision `2fc0636`, fetched by `scripts/download_weights.sh` into `weights/qwen3.5-0.8b/`. One safetensors file holding the text model, the vision encoder (`model.visual.*`) and one MTP layer (`mtp.*`).

**Text config.** hidden 1024 · 24 layers · MLP intermediate 3584 · vocab 248320 · `rms_norm_eps` 1e-6 · tied embedding · no biases · bf16 except `A_log` and the GatedDeltaNet norm weight (f32).

**Layer pattern.** `6 × (3 × GatedDeltaNet + 1 × full attention)`: layers 3, 7, 11, 15, 19, 23 are full attention, the other 18 are GatedDeltaNet. Every layer is pre-norm with an MLP:

```
h = x + mixer(norm(x))          mixer = GatedDeltaNet or full attention
y = h + down(silu(gate(norm(h))) * up(norm(h)))
logits = norm(y_last) @ embed_tokens^T
```

**RMSNorm is zero-centred**: `x * rsqrt(mean(x²) + eps) * (1 + w)`, computed in f32. This applies to the layer norms, the final norm and QK-norm. The one exception is the GatedDeltaNet output norm (below).

**Full attention** (6 layers): 8 query heads, 2 KV heads, head_dim 256.
- `q_proj` 1024→4096 holds, per head, `[query(256) | gate(256)]`; `k_proj`, `v_proj` 1024→512.
- `q_norm`, `k_norm`: zero-centred RMSNorm over head_dim.
- RoPE on the first 64 of 256 dims (`partial_rotary_factor` 0.25), rotate-half layout, `rope_theta` 1e7. The checkpoint declares M-RoPE (`mrope_section` [11, 11, 10]); for text all three position streams are equal, so it reduces to plain RoPE.
- Causal softmax attention, scale `1/sqrt(256)`; output `* sigmoid(gate)`; `o_proj` 2048→1024.

**GatedDeltaNet** (18 layers): 16 heads, key and value head_dim 128.
- `in_proj_qkv` 1024→6144 (q, k, v each 2048), `in_proj_z` 1024→2048, `in_proj_a` and `in_proj_b` 1024→16.
- Depthwise causal `conv1d` (kernel 4, no bias) over the 6144 qkv channels, then SiLU.
- q, k: L2-normalised per head (eps 1e-6); q scaled by `1/sqrt(128)`.
- Per head and token: `β = sigmoid(b)`, `g = -exp(A_log) * softplus(a + dt_bias)`.
- Recurrence on a per-head state `S ∈ R^{128×128}` (f32):
  `S ← exp(g)·S` · `δ = β·(v − Sᵀk)` · `S ← S + k δᵀ` · `o = Sᵀq`.
- Output norm: `rmsnorm(o) * w * silu(z)` per head (plain `w`, not `1 + w`); `out_proj` 2048→1024.

**Per-sequence decode state.**
- Full attention: KV cache, 6 layers × 2 heads × 256 × (K, V) = 12 KB per token in bf16.
- GatedDeltaNet: fixed size regardless of context. Recurrent state 18 × 16 × 128 × 128 × 4 B ≈ 18.9 MB, plus conv state 18 × 6144 × 3 values.

## 3. The performance floor

Text weights are 752M parameters, 1.51 GB in bf16:

| Part | Params | bf16 bytes |
| --- | --- | --- |
| MLP | 264M | 529 MB |
| embed_tokens (= lm_head) | 254M | 509 MB |
| GatedDeltaNet | 190M | 380 MB |
| Full attention | 44M | 88 MB |

Single-stream decode reads every text weight once per token (the tied lm_head reads the whole embedding table), and reads and writes the GatedDeltaNet state: about 1.55 GB per token at short context.

| Device | Bandwidth | Decode floor |
| --- | --- | --- |
| RTX PRO 6000 Blackwell (sm_120) | ~1.8 TB/s | ~0.86 ms/token (~1150 tok/s) |
| DGX Spark GB10 (sm_121) | ~273 GB/s | ~5.7 ms/token (~175 tok/s) |

The lm_head alone is a third of the bytes, so it is the first kernel worth tuning. Every decode result is reported as a percentage of this floor; prefill is judged against tensor-core peak.

## 4. Correctness strategy

- `tools/reference` (`qwen35_ref.model`): the text forward in NumPy, f32. It reads safetensors directly (8-byte header length, JSON header, bf16 as `uint16 << 16` viewed as f32) and gets token ids from the `tokenizers` package. The recurrence is a plain per-token loop. No torch.
- It dumps golden tensors for fixed prompts: embeddings, each layer's output after the mixer and after the MLP, final logits, and the greedy continuation.
- Engine tests compare layer by layer, so a mismatch names its layer. The f32 path must reproduce greedy tokens exactly; the bf16 path within tolerance and on the first N greedy tokens.
- Each optimised kernel keeps its naive predecessor as an in-test reference.

## 5. Phases

Legend: `[ ]` todo · `[~]` in progress · `[x]` done.

### Phase 0 — Strip to the core `[x]`

- [x] Remove training, the Which-Span task, the toy MiniLlm model, the scalar backend and the per-call CUDA kernels; keep `src/core` (allocator, device Tensor).
- [x] `scripts/download_weights.sh`: pinned revision, sha256 check, resumable.

### Phase 1 — Load the checkpoint `[x]`

- [x] 1.1 Hand-written JSON reader; `Config` from `config.json` → `text_config`, with the derived dims above.
- [x] 1.2 Tensor dtype in `core::Tensor` (f32, bf16).
- [x] 1.3 safetensors reader: mmap, parse the header, look tensors up through `model.safetensors.index.json`.
- [x] 1.4 Weight loader: map `model.language_model.*` onto per-layer structs, check every shape against the config, upload once; skip `model.visual.*` and `mtp.*`, and fail on any other unconsumed tensor.
- [x] 1.5 `inspect` binary: config, layer pattern, per-part parameter counts, device bytes, upload time.
- [x] 1.6 Build with CMake presets (`release`, `debug`, `-Werror`), one target per module, tests labelled `cpu` / `gpu` / `weights`; architecture diagram in `docs/architecture.html`.

**Done when** the whole text model is on the GPU and every tensor is accounted for.

### Phase 2 — NumPy reference `[x]`

- [x] 2.1 `tools/reference` (uv project, numpy + tokenizers, no torch): one `forward` for prefill and decode over a `State` (KV cache, conv rows, recurrent S); f32 throughout, asserted.
- [x] 2.2 Validation: greedy continuations are correct on the built-in prompts ("Paris.", "6, 7, 8, 9,", `fibonacci(n-1) + fibonacci(n-2)`); decode and split prefill reproduce full-prefill logits (pytest). Checked once against transformers 5.19.0 (Qwen3_5ForConditionalGeneration, f32, eager attention, its chunked GDN path) in a throwaway venv on 2026-10-15: every layer's output and the last-position logits agree within 2.2e-6 relative error on all three prompts, and the 12-token greedy continuations are identical. torch is not a dependency of this repo.
- [x] 2.3 `qwen35-ref dump` writes `weights/golden/<prompt>.safetensors`: `input_ids`, `greedy_ids` (I32), `embed`, `layerNN.mixer`, `layerNN.out`, `final_norm` (F32, all positions) and the last row of `logits`. The C++ reader needs I32 support to read the ids (Phase 3).

### Phase 3 — Correct engine, naive kernels `[ ]`

f32 activations, bf16 weights converted in-kernel; correctness only.
- [ ] 3.1 Embedding gather, zero-centred RMSNorm, linear (bf16 weight × f32 activation), SiLU·mul, residual add.
- [ ] 3.2 Full attention: q/gate split, QK-norm, partial RoPE, KV-cache append, causal GQA (8/2), sigmoid gate.
- [ ] 3.3 GatedDeltaNet: causal conv1d + SiLU with conv state, L2 norm, β and g, per-token recurrence, gated RMSNorm.
- [ ] 3.4 Prefill and decode through one executor; prefill runs the recurrence token by token for now.
- [ ] 3.5 Tied lm_head, argmax.

**Done when** logits match the reference per layer and greedy text matches on the golden prompts.

### Phase 4 — Bandwidth-bound decode `[ ]`

- [ ] 4.1 bf16 GEMV for M = 1: 128-bit loads, warp-per-row-group, split-K where rows are few. Report achieved GB/s per projection.
- [ ] 4.2 lm_head with fused argmax / top-k over 248320 rows.
- [ ] 4.3 GatedDeltaNet decode step: one block per head, the 128×128 state streamed through registers and shared memory, conv state update fused in.
- [ ] 4.4 Decode attention for the 6 full layers, split over the sequence (flash-decoding).

**Done when** decode reaches ≥ 70 % of the floor on the RTX PRO 6000.

### Phase 5 — Tensor-core prefill `[ ]`

- [ ] 5.1 Tiled GEMM with `mma.sync` bf16 (m16n8k16), `ldmatrix`, swizzled shared memory, `cp.async` multi-stage pipeline; try TMA for the load stage. Note what sm_120 lacks relative to sm_100 (`tcgen05`, TMEM).
- [ ] 5.2 Chunked gated delta rule (chunk 64, WY form) on tensor cores, replacing the per-token recurrence in prefill.
- [ ] 5.3 FlashAttention-2 style causal prefill for the full-attention layers.
- [ ] 5.4 cuBLAS baseline in `benchmark` only; report % of cuBLAS and of tensor-core peak.

**Done when** prefill GEMMs reach ≥ 70 % of cuBLAS at the model's shapes and the chunked GDN beats the recurrence beyond a few dozen tokens.

### Phase 6 — Fusion and launch overhead `[ ]`

- [ ] 6.1 One projection for `in_proj_qkv | z | a | b`; one for `gate | up` with SiLU·mul in the epilogue; one for `q | k | v`.
- [ ] 6.2 RMSNorm into the following projection's prologue, residual add into the epilogue.
- [ ] 6.3 conv1d + SiLU + L2 norm + β/g in one kernel; QK-norm + RoPE + KV append in one kernel.
- [ ] 6.4 CUDA graph for the whole decode step.
- [ ] 6.5 Nsight Systems / Compute; keep a roofline table in `perf.md`.

**Done when** decode is ≥ 85 % of the floor and launches no longer show in the timeline.

### Phase 7 — Text in, text out `[ ]`

- [ ] 7.1 Byte-level BPE tokenizer from `tokenizer.json` (pre-tokenizer regex, merges, special tokens), checked against `tokenizers` on a corpus.
- [ ] 7.2 Chat template from `chat_template.jinja`, including the thinking switch; stop on the EOS id from the config.
- [ ] 7.3 On-device sampler: temperature, top-k, top-p, repetition penalty, seeded RNG.
- [ ] 7.4 Streaming CLI chat.

**Done when** the binary chats with Qwen3.5-0.8B with no Python anywhere.

### Phase 8 — Throughput `[ ]`

- [ ] 8.1 Paged KV cache for the full-attention layers; fixed per-sequence slots for the GatedDeltaNet state.
- [ ] 8.2 Continuous batching: a scheduler that mixes prefill chunks with in-flight decodes.
- [ ] 8.3 Batched decode turns GEMV into skinny GEMM, and the GDN step into a batched state update; revisit kernels for M = 2…64.

### Phase 9 — Beyond `[ ]`

- [ ] 9.1 Speculative decoding with the checkpoint's own MTP layer.
- [ ] 9.2 Weight-only fp8 / int4 (Blackwell `mma` supports fp8 and fp4); measure the new floor.
- [ ] 9.3 Vision encoder (12-layer ViT) for image input.

## 6. Sequencing

```
0 ─► 1 ─► 3 ─► 4 ─► 5 ─► 6 ─► 8 ─► 9
     2 ─┘         7 (any time after 3)
```

Phase 2 can be written alongside Phase 1. The first milestone is Phase 3: correct Qwen3.5 logits from our own kernels. Speed starts at Phase 4.

## 7. How to use this file

- Tick boxes as steps land; put every measured number next to the §3 floor.
- Before starting a phase, write `docs/phase-N-<name>.md` with interfaces, file changes and the test plan.
