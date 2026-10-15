# mini-LLM

一个从零手写的单 GPU 推理引擎，目标模型是 Qwen3.5-0.8B。不依赖 PyTorch / Triton / cuBLAS，kernel 全部手写 CUDA。

训练相关代码已删除；当前只保留 `src/core`（显存分配器和设备 Tensor）。计划见 `ROADMAP.md`。

## 构建

```bash
bash scripts/configure.sh            # 需要 nvcc 和 g++-14 及以上
ctest --test-dir build --output-on-failure
```

## 下载权重

```bash
bash scripts/download_weights.sh     # 默认下到 weights/qwen3.5-0.8b（已被 git 忽略）
```

脚本固定在 `Qwen/Qwen3.5-0.8B` 的一个 revision，并校验 safetensors 和 tokenizer 的 sha256。设置 `HF_TOKEN` 会带认证头，设置 `HF_ENDPOINT` 可用镜像。
