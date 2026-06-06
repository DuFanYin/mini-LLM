# mini-LLM

一个从零手写的 C++23 decoder-only 小语言模型。

这个项目不是为了追求大模型规模，而是为了把一条完整的 LLM 工程链路摊开：模型结构、前向、反向、KV cache、RoPE cache、训练步、优化器、checkpoint、推理、benchmark 和测试都在仓库里，尽量少依赖黑盒框架。

当前默认模型是一个 4 层 GQA Transformer，约 2.4M 个 fp32 参数，训练在合成的 Which-Span 检索任务上。

## 能做什么

- 训练一个小型 decoder-only Transformer。
- 用 `prefill + decode` 跑带 KV cache 的自回归推理。
- 对模型核心路径做手写 backward 和 AdamW 更新。
- 保存 / 加载 checkpoint。
- 运行 kernel、pipeline、end-to-end 级别的 benchmark。
- 用 gtest 覆盖基础算子、decode、IO、data、loss、model 行为。

## 快速开始

所有命令都从仓库根目录运行。

```bash
# 配置并构建。检测到 nvcc 时默认使用 cuda backend，否则默认 scalar。
bash scripts/configure.sh

# 显式选择 backend。
bash scripts/configure.sh scalar
bash scripts/configure.sh cuda

# 运行测试。
ctest --test-dir build --output-on-failure
```

构建产物位于 `build/`：

- `build/train`
- `build/inference`
- `build/benchmark`
- `build/tests_runner`
- `build/debug_scores`

`scripts/configure.sh` 会给 CMake 注入当前 backend 需要的源文件：

- `scalar`：全部 kernel / optimizer 跑在 CPU 上（手写 scalar）。
- `cuda`：GEMM、attention、core 逐元素/softmax kernel 以及 optimizer 的逐张量数学（AdamW、梯度范数、scale）跑在 GPU 上（手写 CUDA kernel）；norm、rope 仍走 scalar 路径。

## 训练

```bash
./build/train <max_steps> <log_every> <eval_every> <lr> <weight_decay> <max_grad_norm> [save_path] [--seed N]
```

示例：

```bash
./build/train 20000 500 500 0.0002 0.001 0.5 weights/model.ckpt --seed 42
```

训练说明：

- `save_path` 存在时，程序退出前会通过 `engine::save_model` 写 checkpoint。
- `--seed N` 同时控制模型初始化、训练采样、验证采样和 probe 采样。
- 训练目标是 Which-Span 的答案 token，不是对整段序列做全位置 next-token loss。
- 默认 early stop 条件是 answer accuracy 和 probe hit-rate 都达到 `0.995`。

## 推理

```bash
./build/inference [model.ckpt] [--temperature T] [--seed N]
```

示例：

```bash
./build/inference weights/model.ckpt --temperature 0.8 --seed 7
```

交互输入格式是 6 个空格分隔字段：

```text
PREFIX SPAN_A MIDDLE SPAN_B SUFFIX Q
```

其中：

- `PREFIX` / `SPAN_A` / `MIDDLE` / `SPAN_B` / `SUFFIX` 是 `A-Z` 字母串，大小写均可。
- `Q` 是单个字母 `A` 或 `B`。
- 程序会构造 prompt 到 query token 为止，先跑一次 `MiniLlm::prefill`，再用 `MiniLlm::decode` 逐 token 生成答案。
- `prefill` 和 `decode` 共享同一个模型 KV cache。

示例输入输出：

```text
> AB CD EF GH IJ A
target=CD | pred=CD | exact=yes

> HELLO CAT MID DOG ZZ B
target=DOG | pred=DOG | exact=yes
```

输入 `exit` 或 `quit` 退出。

## Benchmark

```bash
./build/benchmark [--warmup N] [--iters N] [--train-warmup N] [--train-iters N]
```

默认参数：

- `--warmup 8`
- `--iters 50`
- `--train-warmup 1`
- `--train-iters 30`

输出按几个层次组织：

- Inference / Train。
- Individual Kernels：单个 kernel 或 engine 端融合算子耗时。
- Pipeline Breakdown：训练或推理路径中各阶段耗时。
- End-to-End：完整路径耗时。

## 默认模型

默认超参写在 `src/model/mini_llm.cpp` 的 `default_train_config()` 中：

- `d_model = 256`
- `num_heads = 8`
- `num_kv_heads = 4`
- `head_dim = 32`
- `rope_dim = 16`
- `d_ff = 512`
- `num_layers = 4`
- `vocab_size = 32`

结构是标准 pre-norm decoder block：

```text
RMSNorm
  -> Q/K/V projection
  -> RoPE on Q/K
  -> causal grouped-query attention
  -> output projection
  -> residual
  -> RMSNorm
  -> SwiGLU MLP
  -> residual
```

参数量公式：

```text
total = num_layers * 592384 + 512 * vocab_size
```

默认 `vocab_size = 32` 时约为 2.38M fp32 参数。

## Which-Span 任务

每条样本是一个 token 序列：

```text
PREFIX [SA] SPAN_A [EA] MIDDLE [SB] SPAN_B [EB] SUFFIX [Q] GOLD
```

词表：

- 字母 token `0..25` 对应 `A..Z`。
- 控制 token：`[SA]=26`、`[EA]=27`、`[SB]=28`、`[EB]=29`、`[QA]=30`、`[QB]=31`。
- `[Q]` 是 `[QA]` 或 `[QB]`。
- 如果 query 是 `[QA]`，`GOLD = SPAN_A`；如果 query 是 `[QB]`，`GOLD = SPAN_B`。

长度范围：

- `prefix / middle / suffix`: `2..10`
- `span_a / span_b`: `2..8`

训练时只在答案位置计算 loss。相关代码在 `src/task/`：

- `sample_data.cpp`：采样和合法性检查。
- `metrics.cpp`：teacher-forced loss / accuracy。
- `evaluation.cpp`：答案准确率和 probe。

## 代码分层

### 应用入口

- `train.cpp`：训练循环、日志、保存 checkpoint。
- `inference.cpp`：交互式 Which-Span 推理。
- `benchmark.cpp`：微基准和路径分解。

### `src/task/`

Which-Span 任务层。负责数据采样、任务布局解析、答案位置选择、评估指标。

### `src/train/`

训练层。负责一次训练步、梯度工作区、梯度裁剪、AdamW。

核心文件：

- `train_step.cpp`
- `optimizer.cpp`
- `loss.cpp`
- `train.h`

### `src/model/`

模型层。负责 Transformer 的结构、权重、运行时 cache、forward / backward 编排。

核心文件：

- `mini_llm.h` / `mini_llm.cpp`：对外门面，提供 `prefill`、`decode`、`forward_train`、`backward`。
- `model_types.h`：配置、权重、tape、输入结构。
- `cache.h` / `cache.cpp`：模型运行时缓存，包括 `KVCache` 和 `RopeCache`。
- `executor_forward.cpp`：decoder stack forward 编排。
- `executor_backward.cpp`：decoder stack backward 编排。
- `executor.h`：executor 内部 API。

### `src/engine/`

语言模型 IO 端语义层。这里不是 Transformer block 本体，而是 embedding、logits、cross-entropy、采样和 checkpoint 这类模型边界逻辑。

核心文件：

- `embedding.h` / `embedding.cpp`：token embedding forward / backward。
- `decode.h` / `decode.cpp`：logits projection、cross entropy、采样、对应 backward。
- `io.h` / `io.cpp`：checkpoint save / load。

### `src/kernel/`

纯数学 kernel 层。原则是无模型状态、无 cache ownership，只吃指针和参数结构，写输出指针。

核心文件：

- `kernel.h`：kernel 参数结构和函数声明（所有 backend 的公共 seam）。
- `common/silu.cpp`：与设备无关的标量 helper（silu / silu_derivative），始终编译。
- `scalar/`：CPU 实现。`gemm.cpp`、`attention.cpp`、`core.cpp`（逐元素 / softmax / optimizer 逐张量数学），以及始终走 CPU 的 `norm_rope.cpp`（RMSNorm / RoPE forward / backward）。
- `cuda/`：手写 CUDA 实现。`device.cuh`（错误检查、可复用 device buffer）、`gemm.cu`、`attention.cu`、`core.cu`。
- backend 选择由 `scripts/configure.sh` 把 `scalar/` 或 `cuda/` 对应的源文件喂给 CMake；两者实现同一组 `kernel.h` 符号。

## 当前依赖方向

```text
应用入口
  -> task / train
  -> engine
  -> model
  -> kernel
```

更具体地说：

```text
train.cpp / inference.cpp / benchmark.cpp
  -> src/task
  -> src/train
  -> src/engine
  -> src/model
  -> src/kernel
```

`model` 库里也编译了 `engine/io.cpp`，因为 checkpoint 格式直接序列化 `model::ModelConfig` 和 `model::ModelWeights`。除此之外，Transformer block 的核心编排都在 `model`，纯算子都在 `kernel`。

## Cache 设计

`src/model/cache.h` 统一放模型运行时缓存：

- `KVCache`：推理用的 per-layer K/V 存储，支持 `append()`、`view()`、`reset()`。
- `RopeCache`：RoPE 的 cos/sin 表，按最大 position 懒扩容。

`kernel::apply_rope` 不持有 cache，只接收 `cos_data()`、`sin_data()` 和 `rot_dim()`。这样 kernel 仍然是无状态数学，cache 生命周期由 `model::MiniLlm` 管理。

## 测试

```bash
ctest --test-dir build --output-on-failure
```

测试覆盖：

- 基础 kernel 行为。
- RoPE 和 KV cache 行为。
- logits / cross-entropy / sampling。
- checkpoint IO。
- Which-Span data validity。
- `MiniLlm` prefill / decode / train forward / AdamW loss 下降。

## 设计目标

这个仓库优先追求：

- 可读性：从 token 到 loss / logits 的路径能直接跟进源码。
- 分层清晰：kernel 无状态，model 管 Transformer，engine 管语言模型边界，train/task 管任务和优化。
- 可验证：每次重构后能快速用 `ctest` 确认行为。
- 小而完整：规模小，但链路完整。