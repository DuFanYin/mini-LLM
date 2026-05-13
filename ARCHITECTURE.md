# Architecture

## Project Boundary

`mini-LLM` has one model family and one task:

- model: `model::MiniLlm`
- task: Which-Span

There is no abstract `Model` interface. All callers use `model::MiniLlm` directly. Add an interface only if a second model family is introduced.

The default model is built as:

```cpp
auto m = model::MiniLlm::init_random(task::n_vocab(), seed);
```

Default shape:

- `d_model = 256`
- `num_heads = 8`
- `num_kv_heads = 4`
- `head_dim = 32`
- `rope_dim = 16`
- `d_ff = 512`
- `rms_norm_eps = 1e-5`
- `num_layers = 4`
- `vocab_size = 32`

Saved model files contain only `ModelConfig` and `ModelWeights`. They do not contain optimizer state, RNG state, current step, loss history, or eval metrics.

## Layer Order

The code is split by ownership:

```
apps
  ├─ training app
  ├─ inference app
  └─ benchmark app

task  ─────┐
train ─────┼──► model ───► kernel
apps  ─────┘      │
                  ▼
                engine
```

Layer roles:

- `apps` wire command-line arguments, logging, save/load calls, and top-level loops.
- `task` owns Which-Span data generation, parsing, prediction positions, and evaluation.
- `train` owns one-step training, reusable training buffers, gradient utilities, and AdamW.
- `model` owns `MiniLlm`, model weights/config, two-step random init (`MiniLlm::build` then `init_random`), optional `init_load`, one-shot `load_mini_llm`, per-layer runtime state, forward/backward execution, and cache use.
- `engine` owns runtime utilities shared by callers: trained-model I/O, decode helpers, embedding helpers. (KV cache lives in the `model` layer alongside the executor.)
- `kernel` owns shape-based math functions. It has no model/task/train object dependency.

Allowed dependencies:

- apps -> `task`, `train`, `model`, selected `engine`
- `task` -> `model`
- `train` -> `model`, `engine`, `kernel`
- `model` -> `engine`, `kernel`
- `engine` -> model types only for serialization
- `kernel` -> standard library only

## Model State

`model::MiniLlm` owns the runnable model state:

- `ModelConfig config_`
- `ModelWeights weights_`
- `std::vector<DecoderLayerState> layers_`
- optional `std::unique_ptr<model::KVCache> cache_`

`ModelWeights` contains token embeddings, decoder-layer weights, and output projection weights (`output_projection`). `DecoderLayerState` stores per-layer weight pointers plus RoPE cache state used by execution.

Construction paths:

- `MiniLlm(config, weights)` validates and stores caller-provided config/weights.
- `MiniLlm::init_random(vocab, seed)` uses the hard-coded `ModelConfig` in `mini_llm.cpp`, allocates matching tensors, and fills linears/embeddings with the built-in random scheme. `init_load(path)` loads config/weights from disk and configures cache.
- `MiniLlm::init_load(path, max_seq_len)` is a convenience: load from `engine` + construct `MiniLlm` + `configure_cache` (no prior `build`).

Public runtime methods:

- `forward_train(token_ids, output, tape)` embeds the full sequence, runs the decoder stack without cache, and fills training tapes.
- `prefill(token_ids, output)` resets cache state for the request, runs the full prompt, and writes all prompt hidden rows.
- `decode(token_id, output)` appends one token to the existing cache context and writes one hidden row.
- `backward(tapes, grad_hidden_out, grad, grad_hidden_in)` runs decoder-stack backward.
- `configure_cache(max_seq_len)` preallocates inference cache storage.

Executor-only types such as `ForwardInput`, `BlockForwardTape`, and `ModelForwardTape` stay inside the model layer. Runtime cache types (`KVCache`, `KVCacheConfig`, `CacheView`, `RopeCache`) live next to the cache implementation in `model/cache.h`. Apps, task code, and train code should call `MiniLlm` methods instead of constructing executor inputs or touching the cache directly.

## Forward Execution

Forward execution starts from public `MiniLlm` methods and then enters the executor:

1. The public method prepares token embeddings or a single-token hidden row.
2. `forward_model` iterates decoder layers in order.
3. Each layer runs attention-side work first, then MLP-side work.
4. The final hidden rows return to the caller.

Attention-side order:

1. Normalize input into `norm1_out`.
2. Project Q, K, and V.
3. Apply RoPE to Q and K.
4. If cache is enabled, append K/V and build the visible cache view.
5. Run attention.
6. Project attention output.
7. Add the first residual into `attn_residual`.

MLP-side order:

1. Normalize `attn_residual` into `norm2_out`.
2. Compute gate projection.
3. Compute up projection.
4. Combine gate/up activation.
5. Compute down projection.
6. Add the second residual into the layer output.

Training forward passes run with cache disabled and tapes enabled. Inference prefill/decode passes run with cache enabled and no training tape.

## Backward Execution

Training backward starts above the decoder stack:

1. Selected hidden rows are projected through the output projection (vocabulary logits).
2. Cross entropy produces loss, accuracy, and gradient rows.
3. Output-projection backward accumulates `grad.output_projection`.
4. Hidden-row backward scatters top gradients into the full hidden tensor.
5. `MiniLlm::backward` walks decoder layers in reverse.
6. Token-embedding backward scatters final input gradients into `grad.token_embedding`.

Within each decoder layer, backward order is:

1. MLP backward.
2. Attention output projection backward.
3. Attention backward.
4. RoPE backward for Q/K.
5. Q/K/V projection backward.
6. First normalization backward.
7. Residual gradient accumulation into the previous layer.

`BlockForwardTape` stores only forward values needed by the corresponding backward calls. The tape is allocated by the training workspace and filled during `forward_for_training`.

## Engine Utilities

The engine layer provides shared utilities that are not owned by the model object:

- Trained-model save/load.
- Logit projection helpers.
- Packed prediction-step projection helpers.
- Argmax and temperature sampling.
- Embedding lookup helpers.

I/O path:

1. `engine::save_model(path, config, weights)` writes config followed by tensors in `ModelWeights` order.
2. `engine::load_model(path)` returns `SavedModel { config, weights }`.
3. `model::MiniLlm::init_load(path, max_seq_len)` turns `SavedModel` into a runnable `MiniLlm`.

Use `engine::save_model` for writes and `MiniLlm::init_load` for executable-model reads.

## Kernel Boundary

Kernel functions use explicit pointers, sizes, strides, and shape values. They do not own model buffers and do not allocate model-level scratch state.

Kernel groups currently cover:

- linear projection and its backward path
- residual/pointwise helpers
- RMSNorm and RoPE forward/backward
- attention: `gqa_attention_forward` (optional `attn_probs_out` for training tape) + `gqa_attention_backward`
- cross entropy and selected-row gradient helpers
- embedding backward
- gradient norm, clearing, clipping, and optimizer helpers where they operate on raw tensor storage

Multi-step model blocks stay in the model executor. Do not add fused block-level kernels unless the caller boundary remains clear.

## Training Flow

`TrainWorkspace` owns reusable per-step buffers:

- gradient aggregate
- forward tapes
- packed hidden rows
- logits and loss buffers
- top-gradient buffers
- decoder input/output gradient buffers

One `train_step`:

1. Resizes workspace buffers for the current model and sequence.
2. Runs `MiniLlm::forward_for_training`.
3. Projects only `prediction_steps` into logits.
4. Computes loss and accuracy.
5. Backprops through output projection and selected hidden rows.
6. Calls `MiniLlm::backward`.
7. Backprops into token embeddings.
8. Clips gradients.
9. Applies AdamW to all tensors in `ModelWeights`.

The training app owns the outer loop: sample sequence, choose prediction positions, call `train_step`, log, evaluate, early-stop, and optionally save trained weights.

## Task Flow

Which-Span sequences have this layout:

```
PREFIX [SA] SPAN_A [EA] MIDDLE [SB] SPAN_B [EB] SUFFIX [Q_A|Q_B] GOLD
```

Token ids:

- letters: `0..25`
- `[SA] = 26`
- `[EA] = 27`
- `[SB] = 28`
- `[EB] = 29`
- `[QA] = 30`
- `[QB] = 31`
- vocab size: `32`

Length ranges:

- prefix: `[2..10]`
- middle: `[2..10]`
- suffix: `[2..10]`
- span A: `[2..8]`
- span B: `[2..8]`

Task responsibilities:

- Generate valid sequences.
- Parse delimiter, query, and answer positions.
- Return answer-token prediction positions for training/eval.
- Compute teacher-forced batch metrics.
- Run generated-answer accuracy and probe checks.

## App Flows

Training app:

1. Build a fresh `MiniLlm`.
2. Configure cache capacity.
3. Create AdamW moment buffers.
4. Sample one sequence per step.
5. Evaluate on fixed validation/probe seeds.
6. Save trained weights if `save_path` is provided.

Inference app:

1. Load `MiniLlm` with `MiniLlm::init_load`.
2. Parse six fields: `PREFIX SPAN_A MIDDLE SPAN_B SUFFIX Q`.
3. Build the prompt through the query token.
4. Call `prefill` once.
5. Sample the first answer token from the final prompt hidden row.
6. Call `decode` for remaining answer tokens.

Benchmark app:

1. Build a default `MiniLlm`.
2. Run grouped inference timings.
3. Run grouped training timings.
4. Print kernel, pipeline, and end-to-end measurements.

## Change Rules

Keep these boundaries intact:

- Apps should not call executor functions directly.
- Task code should not construct `ForwardInput`, `KVCache`, or `CacheView`.
- Kernel code should not include model, train, or task object headers.
- Saved model reads/writes should go through the engine/model I/O path.
- Do not add a generic model interface while `MiniLlm` is the only model family.

When adding behavior:

- Put new raw math in `kernel`.
- Put new runtime utility behavior in `engine`.
- Put new model execution behavior on `MiniLlm` or inside the model executor.
- Put optimizer/training-buffer behavior in `train`.
- Put Which-Span generation/evaluation behavior in `task`.

## Naming

Namespaces are fixed to `engine`, `model`, `kernel`, `train`, and `task`.

Use these naming patterns:

- value construction helpers: `make_*`
- loading helpers: `load_*`
- metrics returning values: `compute_*`
- predicates: `is_*`
- counters: `count_*`

Use anonymous namespaces for file-local app helpers. Use a narrow `detail` namespace only when helper symbols must appear in a header.
