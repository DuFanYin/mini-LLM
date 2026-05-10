# mini-LLM

A from-scratch decoder-only language model in C++23. No deep-learning framework — every kernel, the autograd tape, the optimizer, the KV cache, and the on-disk format are written by hand so the whole stack stays readable end-to-end.

The repo trains a 4-layer GQA transformer (~2.4M params) on a synthetic **Which-Span** retrieval task and ships interactive `inference`, batch `train`, micro-`benchmark`, and a `tests_runner`.

For layout, dependency rules, and per-component design notes, see `ARCHITECTURE.md`.

## What's in the box

- Decoder block: pre-norm RMSNorm → QKV → RoPE on a head-dim prefix → grouped-query attention (causal) → `o_proj` → residual → RMSNorm → SwiGLU MLP → residual.
- Paged KV cache with explicit `prefill` (whole prompt) and `decode` (single step) on `model::MiniLlm`.
- Manual backward pass: per-layer `BlockForwardTape` + reverse-order `backward_decoder_*` calls, AdamW with global-L2 grad clip.
- Synthetic retrieval task in `src/task/`.
- Layered code: `kernel` (pure math) ← `model` (MiniLlm + executor) ← `engine` (KV cache, I/O, decode helpers); `train` and `task` consume `model`.

## Commands

All commands run from the repo root.

```bash
# Build everything + run gtest suite.
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure

# Train (six positional args + optional save_path + optional --seed N).
./build/train <max_steps> <log_every> <eval_every> <lr> <weight_decay> <max_grad_norm> [save_path] [--seed N]
./build/train 20000 500 500 0.0002 0.001 0.5 weights/model.ckpt --seed 42

# Inference (interactive; defaults: model=weights/model.ckpt, temperature=1.0, RNG=random).
./build/inference [model.ckpt] [--temperature T] [--seed N]
./build/inference weights/model.ckpt --temperature 0.8 --seed 7

# Micro-benchmarks (defaults: warmup=8 iters=50 train_warmup=1 train_iters=30).
./build/benchmark [--warmup N] [--iters N] [--train-warmup N] [--train-iters N]
```

Build outputs land in `build/`: `train`, `inference`, `benchmark`, `tests_runner`.

## Default model

Defined by `model::MiniLlm::architecture()` and used by `MiniLlm::build(vocab, seed, num_layers)`:

| field | value |
|---|---|
| `d_model` | 256 |
| `num_heads` | 8 |
| `num_kv_heads` | 4 (GQA group = 2) |
| `head_dim` | 32 |
| `rope_dim` | 16 (rotary applied to head-dim prefix only) |
| `d_ff` | 512 (SwiGLU) |
| `num_layers` | 4 |
| `vocab_size` | 32 (= `task::n_vocab()`) |

Param count formula (see `src/model/mini_llm.cpp`):
`total = num_layers * 592_384 + 512 * vocab_size` → ~2,384,384 fp32 params at the defaults.

## Task: Which-Span

Each sample is one variable-length sequence of token ids:

```
PREFIX  [SA]  SPAN_A  [EA]  MIDDLE  [SB]  SPAN_B  [EB]  SUFFIX  [Q]  GOLD
```

- Letter tokens `0..25` map to `A..Z`.
- Control tokens: `[SA]=26 [EA]=27 [SB]=28 [EB]=29 [QA]=30 [QB]=31`.
- `[Q]` is either `[QA]` or `[QB]`; `GOLD` is `SPAN_A` if `[QA]`, else `SPAN_B`.
- The training loss only scores the `GOLD` positions (`task::answer_prediction_steps`).
- Length ranges (random per sample): `prefix/middle/suffix ∈ [2..10]`, `span_a/span_b ∈ [2..8]`.

Code lives in `src/task/`: `sample_data.cpp` (sampling + validity), `metrics.cpp` (teacher-forced batch loss/acc), `evaluation.cpp` (answer accuracy + probes).

## Train notes

- `save_path` — written via `engine::save_model` on exit (success **or** timeout).
- `--seed N` — seeds model init plus the train/val/probe samplers (deterministic from one knob).
- Early-stop when `compute_answer_accuracy ≥ 0.995` **and** the probe hit-rate `≥ 0.995`.

## Inference notes

Type **6 whitespace-separated fields**: `PREFIX SPAN_A MIDDLE SPAN_B SUFFIX Q`, where each field is letters `A-Z` (case-insensitive) and `Q` is the single letter `A` or `B`. `inference.cpp` builds the full prompt up through `[Q]`, runs `model::prefill` once over the whole prompt, then loops `model::decode` for the rest of the answer span — both stages share one KV cache.

```text
> AB CD EF GH IJ A
target=CD | pred=CD | exact=yes
> HELLO CAT MID DOG ZZ B
target=DOG | pred=DOG | exact=yes
```

Type `exit` or `quit` to leave.

## Benchmark notes

Output is grouped console text — `Inference {Kernel, Pipeline, End-to-End}` then `Train {Kernel, Pipeline, End-to-End}`. Units are `us/iter` or `us/token` depending on the metric. Model init for the run uses `vocab=32 layers=4 seed=42`.

## Project map

- `src/engine/` — KV cache, model save/load (`engine::save_model` / `engine::load_model`), forward-input validation, decode + embedding helpers.
- `src/model/` — `model::MiniLlm` (config + weights + per-layer state + optional cache) plus `executor.h` and the forward/backward executors.
- `src/kernel/` — pure math kernels (`gemm`, `attention`, `norm_rope`, `core`, `backward`); no model-family dependency.
- `src/train/` — `TrainWorkspace`, `train_step`, AdamW + grad utilities.
- `src/task/` — Which-Span sampler, metrics, and evaluation.
- `tests/` — gtest-based unit tests (ops, model, IO, decode, data, loss).
