# mini-LLM → Inference Engine Roadmap

Status: draft (2026-06-07). This is the living plan for turning mini-LLM from a
clean teaching implementation into a single-GPU inference engine. Each phase gets
its own detailed implementation plan (a separate `docs/phase-N-*.md`) when we pick
it up; this file is the big picture and the step checklist.

---

## 1. Goal & scope

**North star.** A clean **single-GPU inference engine** for Llama-family models:
device-resident, bf16 (fp8-ready), loads GGUF/safetensors weights, batched, with
CUDA-graph decode and a real sampler.

**Why this is reachable.** `model::MiniLlm` is already architecturally a Llama:
RMSNorm + grouped-query attention + RoPE + SwiGLU. "Serve a real small model" is a
generalization, not a rewrite.

**Non-goals (explicitly out of scope).**
- Multi-GPU / distributed inference (tensor/pipeline parallel). Single GB10 only.
- Production-grade *training*. Training stays on the existing host path; see §7.
- Being vLLM/TensorRT-LLM at scale. We borrow their patterns, not their breadth.

**Definition of done for the north star.** Load an open-weights small model
(e.g. Qwen2.5-0.5B / Llama-3.2-1B), tokenize real text, and serve batched,
streaming generations in bf16 with decode-step latency dominated by compute, not
launch/transfer overhead.

---

## 2. Current state (what we build on)

- `model::MiniLlm` — `prefill` / `decode` / `forward_train` / `backward`; Llama-shaped block.
- `model::KVCache`, `RopeCache` (`src/model/cache.{h,cpp}`) — contiguous KV cache.
- `src/model/executor_forward.cpp` / `executor_backward.cpp` — layer-stack orchestration.
- `engine` — `embedding`, `decode` (logits projection, argmax + temperature sampling), `io` (bespoke `.ckpt`).
- Kernel layer with a backend split: `src/kernel/{scalar,cuda,common}` selected by `scripts/configure.sh`.
- CUDA backend exists but is **per-call host↔device transfer** → correct but slow (see `perf.md`).
- `task` — the toy "Which-Span" task with integer "tokens" (placeholder tokenizer).
- Build: CMake, gcc-15 host, CUDA 13, `CMAKE_CUDA_ARCHITECTURES=native` (sm_121).

**The core problem to fix:** a "tensor" is `std::vector<float>` on the host; the
GPU is something kernels visit and leave. Every phase below depends on changing that.

---

## 3. Design principle: device residency

The single idea the whole engine is built on: **tensors live on the GPU across the
entire prefill/decode; host↔device copies happen only at the token-in / logits-out
boundary, and ops are enqueued on a stream instead of synchronizing per call.**

This is what removes the per-call-transfer tax behind today's ~5 ms decode, and it
is the foundation for kernels (Phase 1), dtypes (Phase 2), paged cache and CUDA
graphs (Phases 4–5).

---

## 4. Phases

Legend: `[ ]` todo · `[~]` in progress · `[x]` done.

### Phase 0 — Device-resident Tensor + inference path  `[ ]`

**Objective.** `prefill`/`decode` keep weights, activations, and KV on the GPU
end-to-end. One H2D (tokens) and one D2H (logits) per step.

> Split into two plans: **0a — device foundation** (`Tensor`, allocator, kernel
> seam) at [docs/superpowers/plans/2026-06-07-phase-0a-device-tensor.md](superpowers/plans/2026-06-07-phase-0a-device-tensor.md);
> **0b — resident executor** (steps 0.3, 0.4, 0.6, 0.7, 0.8) gets its own plan once 0a lands.

Steps:
- [ ] 0.1 `core::Tensor` — owns `{void* data, Shape, Strides, DType, Device}`; non-owning `TensorView`. fp32 only for now. Decide ownership/view semantics.
- [ ] 0.2 `DeviceAllocator` — pool/arena over `cudaMalloc`: bump allocator for activations (reset per request), persistent slab for weights + KV. Retire the scattered `static DeviceBuffer` instances.
- [ ] 0.3 `CudaContext` — owns the stream (+ later cuBLAS handle); ops enqueue on it; exactly one sync at the boundary.
- [ ] 0.4 Upload weights once at model load into device `Tensor`s held by `MiniLlm`; never re-uploaded.
- [ ] 0.5 Split each cuda kernel into a device-pointer launch core (no copy) + keep the host-pointer facade for scalar/tests. Inference calls the device-ptr form.
- [ ] 0.6 Move `KVCache` storage to device tensors; append on device.
- [ ] 0.7 Device `rms_norm` + `rope` (naive is fine; fused in Phase 1) so the inference path is fully resident — no CPU hops mid-stack.
- [ ] 0.8 Rewire `executor_forward` prefill/decode to thread device tensors through the stack; embed→device hidden in, final hidden→logits→host for sampling.

**Deliverable.** Resident inference path: 1 H2D + 1 D2H per step.
**Verification.** Cross-backend diff: device-resident logits match the scalar backend within tolerance; benchmark shows a large decode-latency drop vs `perf.md` baseline.
**Open decisions.** Tensor ownership/view model; allocator policy; how the device path and the host scalar/test path coexist (precursor to the Phase-1 dispatch seam).

### Phase 1 — Real kernels behind a (device, dtype) dispatch seam  `[ ]`

**Objective.** Replace handwritten kernels with vendor/fused ones, behind one seam.

Steps:
- [ ] 1.1 Introduce the `KernelBackend` dispatch (the deferred "Move 2"), keyed on `(device, dtype)`, operating on device `Tensor`s. Callers stay unchanged via a thin facade.
- [ ] 1.2 cuBLAS (cublasLt) GEMM replacing the handwritten tiled GEMM; benchmark both.
- [ ] 1.3 FlashAttention-style fused attention (streaming softmax, **no materialized probs matrix** — today's attention materializes `[s, hq, total_kv_len]`). Keep the naive kernel as the diff reference.
- [ ] 1.4 Fused RMSNorm(+residual) and fused SwiGLU.
- [ ] 1.5 Retain scalar-CPU + naive-cuda as reference backends purely for numerical diffing.

**Deliverable.** Inference on vendor/fused kernels, correctness gated by cross-backend diff.
**Verification.** Numerical diff within tol; throughput up; attention memory no longer O(seq·kv).

### Phase 2 — bf16 numerics  `[ ]`

**Objective.** bf16 weights/compute with fp32 accumulation.

Steps:
- [ ] 2.1 Add `bf16` to `DType`; dtype-aware allocation + Tensor.
- [ ] 2.2 bf16 weight storage + load-time conversion; fp32 accumulators in kernels.
- [ ] 2.3 cuBLAS bf16 GEMM; bf16 attention path.
- [ ] 2.4 Tolerance tests bf16-vs-fp32 (looser, sanity not bit-exact).

**Deliverable.** bf16 inference path (~2× memory + speed). fp8 (Blackwell) deferred to Phase 5.

### Phase 3 — Real models, weights, tokenizer  `[ ]`

**Objective.** Stop being a toy task; load and run a real open-weights model.

Steps:
- [ ] 3.1 Generic `ModelConfig` loaded from file (HF `config.json`-style); remove the hardcoded shape from `mini_llm.cpp`.
- [ ] 3.2 Weight loader: **safetensors** and/or **GGUF** → device tensors, with name-mapping onto the Llama-shaped blocks.
- [ ] 3.3 Real tokenizer: integrate BPE/SentencePiece (existing lib, e.g. sentencepiece / tokenizers-cpp). Retire the integer toy tokens for real models.
- [ ] 3.4 Validation: load a small real model (Qwen2.5-0.5B / TinyLlama / Llama-3.2-1B) and match reference logits / greedy generations.

**Deliverable.** Serve a real open-weights small model end-to-end.

### Phase 4 — Serving (throughput + API)  `[ ]`

**Objective.** Concurrent, batched, streaming serving with a usable surface.

Steps:
- [ ] 4.1 Full sampler: top-k, top-p, temperature, repetition/frequency penalty, stop sequences, seeded RNG (extend `engine/decode`).
- [ ] 4.2 Paged KV cache (block-based, PagedAttention-style) replacing the contiguous cache; enables many sequences + memory efficiency.
- [ ] 4.3 Dynamic/continuous batching: scheduler that co-batches prefill + in-flight decodes.
- [ ] 4.4 Streaming output + API: in-process C++ API → C ABI → optional pybind11 bindings and/or OpenAI-compatible HTTP server.

**Deliverable.** Concurrent, batched, streaming inference.

### Phase 5 — Production polish  `[ ]`

**Objective.** The last 20% that makes it real.

Steps:
- [ ] 5.1 **CUDA Graphs** for the decode step (capture the per-token graph; eliminates launch overhead — the exact cost the benchmark exposed).
- [ ] 5.2 Quantization: int8/int4 weight-only (AWQ/GPTQ-style) load + dequant-in-kernel or int kernels.
- [ ] 5.3 CI: cross-backend + cross-dtype numerical diff; perf-regression tracking layered on `perf.md`.
- [ ] 5.4 Observability: nsys/CUPTI profiling hooks, structured logging, metrics; replace `abort()`-on-CUDA-error with status propagation.
- [ ] 5.5 (Stretch) speculative decoding; fp8 compute on Blackwell.

---

## 5. Cross-cutting concerns (apply across phases)

- **Testing.** Keep a reference backend (scalar CPU) and assert every device/dtype
  kernel against it within tolerance — the cheapest, strongest correctness net.
  Add golden-output generation tests once real models load (Phase 3).
- **Config as data.** Model shape and runtime knobs come from files, not code.
- **Serialization.** Standard formats (safetensors/GGUF), versioned.
- **Error handling.** Status/exception propagation, not `abort()`, anywhere user
  input or model files reach.
- **Docs.** Each phase ships with notes; ARCHITECTURE.md updated as boundaries move.

---

## 6. Sequencing & dependencies

```
Phase 0 ─► Phase 1 ─► Phase 2 ─► Phase 4 ─► Phase 5
                 └─► Phase 3 (config/loader/tokenizer can start once 0 lands;
                              real-model validation wants 1–2 for speed/parity)
```

- **0 → 1 → 2** are strictly ordered: the foundation and the kernel/dtype stack.
- **3** can begin in parallel after 0 (config + loader + tokenizer don't need fused
  kernels), but its validation step wants 1–2.
- **4** needs 0–2 (and 3 for real models).
- **5** is partly ongoing (CI/profiling) and partly final (graphs/quant).

**First milestone = Phase 0** (device-resident inference path). Self-contained,
immediately visible as a decode-latency drop, and the platform everything else
stands on.

---

## 7. Training: maintained, not the focus

Training already works and passes tests; we keep it on the existing host scalar
path and use it to **produce checkpoints for testing inference**. We do not invest
"industry" effort there (no AMP/distributed/etc.). The Phase-0 `Tensor` foundation
is reusable if training is modernized later, but that's out of scope for this
roadmap.

---

## 8. How to use this doc

- Tick the `[ ]`/`[~]`/`[x]` boxes as steps land.
- When starting a phase, write `docs/phase-N-<name>.md` with the detailed
  implementation plan (file-by-file changes, interfaces, test plan) before coding.
- Keep §2 (current state) and the sequencing diagram honest as the codebase moves.
