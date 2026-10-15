# mini-LLM

一个从零手写的单 GPU 推理引擎，目标模型是 Qwen3.5-0.8B。不依赖 PyTorch / Triton / cuBLAS，kernel 全部手写 CUDA。计划见 `ROADMAP.md`，架构图见 `docs/architecture.html`，工具链版本见 `version.toml`。

## 构建与测试

需要 nvcc（`/usr/local/cuda`）和 PATH 上的 `g++-15`。

```bash
cmake --preset release && cmake --build --preset release
ctest --preset release          # 全部测试
ctest --preset no-weights       # 跳过需要 checkpoint 的测试
```

另有 `debug` preset。产物在 `build/<preset>/`。测试标签：`cpu`（只用主机）、`gpu`（需要 CUDA 设备）、`weights`（还需要下载好的 checkpoint）。格式化用仓库里的 `.clang-format`。

## 权重

```bash
bash scripts/download_weights.sh     # 下到 weights/qwen3.5-0.8b（已被 git 忽略）
./build/release/apps/inspect         # 加载到 GPU 并打印配置与权重统计
```

下载脚本固定在 `Qwen/Qwen3.5-0.8B` 的一个 revision，并校验 safetensors 和 tokenizer 的 sha256。设置 `HF_TOKEN` 会带认证头，设置 `HF_ENDPOINT` 可用镜像。

## NumPy 参考实现

`tools/reference` 是 Qwen3.5 文本模型的 NumPy 前向（f32，不依赖 torch），作为引擎的正确性基准。

```bash
cd tools/reference
uv run qwen35-ref generate "The capital of France is"   # 贪心生成
uv run qwen35-ref dump                                   # 写 weights/golden/*.safetensors（逐层隐藏状态）
uv run pytest && uv run ruff check .
```

## 目录

```
src/core    设备 Tensor（f32 / bf16）、Pool 与 Arena 分配器、CudaError
src/io      JSON 解析、safetensors（mmap）与多分片 Checkpoint
src/model   Qwen3.5 Config 与 Weights（加载时逐项校验形状和 dtype）
apps        可执行程序（inspect）
tests       按模块划分的 gtest 测试
tools/reference  NumPy 参考实现与 golden 导出（uv 项目）
```
