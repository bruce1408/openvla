# OpenVLA 在 Jetson AGX Thor 上的 TensorRT 部署方案 Review

审查对象：`bruce1408/openvla` 的 `cdd_dev` 分支，提交 `6d76469`。

## 结论

原方案的主架构正确：不要把整个 OpenVLA 当成一个普通 ONNX 一次性转换，而应按视觉编码、投影、LLM prefill/decode、动作解码分阶段替换，并始终保留 PyTorch 版本作为 Golden Reference。

建议落地顺序：

1. 固定 BF16 PyTorch 的性能与动作输出。
2. 保存 processor、视觉、projector、多模态 embedding、token 和连续动作的 Golden 数据。
3. 先将 DINOv2 + SigLIP + projector 转为固定 shape 的 FP16 TensorRT engine。
4. 使用“TensorRT 视觉 + PyTorch Llama + 手动 KV cache”的 Hybrid runtime 验证完整 token。
5. 只有 Hybrid runtime 达到 token exact match 后，才开始替换 LLM 后端。
6. FP8、Action-only LM head、NVFP4 必须逐项引入，每次重新做 token 与机器人 rollout 验证。

## 必须修正的三项假设

### 1. Edge-LLM 不能按“Llama 2 已正式支持”处理

截至本次审查，TensorRT Edge-LLM 的公开支持矩阵明确列出 Llama 3.x，没有明确列出 Llama 2。OpenVLA 内部是微调后的 Llama 2，因此正确做法是：

- 从 OpenVLA checkpoint 提取内部 language model 和 tokenizer；
- 对默认 `CausalLM` 执行 exporter/build 兼容性探测；
- 若失败，按 Edge-LLM customization guide 注册 Llama 2 模型和 checkpoint 映射；
- 不能用 base Llama 2 或 Llama 3 权重替换 OpenVLA 内部权重。

官方文档：[Supported Models](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/supported-models.html)、[Customization Guide](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/customization/customization-guide.html)。

### 2. 256 类 Action Head 不是天然等价优化

当前模型的 LM head 输出 padded vocabulary；OpenVLA 的有效词表为 `text_config.vocab_size - pad_to_multiple_of`，动作 token 位于有效词表末端的 256 个 token。只计算这 256 行权重，等价于“把输出强制约束在动作 token 集合中”。

只有同时满足以下条件，才可以认为它与 reference greedy decode 等价：

- reference 每一步 argmax 都落在动作 token 区间；
- 256 行权重和 bias 与原 LM head 对应行完全相同；
- token ID 映射未改变；
- 全验证集 token exact match，且真实机器人 rollout 无回退。

因此此优化应放在 FP16/FP8 正确性验证之后。

### 3. “256 bins”实际对应 256 个 token、255 个中心值

当前 `ActionTokenizer` 使用 `np.linspace(-1, 1, 256)` 产生 bin 边界，再取相邻边界中心，因此 `bin_centers` 长度为 255。第 256 个离散索引会被 clip 到最后一个中心值。C++ 代码必须读取导出的真实数组，不能创建长度为 256 的中心表。

## 对现有 cdd_dev benchmark 的 Review

- `tools/bench_e2e.py` 适合作为正式 E2E 指标，因为它只在阶段边界同步。
- `tools/bench_stages.py` 的多个 forward hook 会反复 `torch.cuda.synchronize()`，适合定位瓶颈，不适合作为最终生产 E2E 数字。
- `other_ms` 是推理总耗时减去已打点 GPU 阶段后的残差，包含 embedding、拼接、HF generate 调度、采样和后处理等，不能标成纯 postprocess。
- 7 维动作通常表现为一次 prefill 生成第一个 token，再执行 6 次 cached decode。
- 原 benchmark 的 decode 平均值用全部生成 token 数作为分母，但 TTFT 已包含首 token；实现已修正为只用后续 decode token 数。
- `--measure-generate` 的默认 token 数原为 64，不代表 7-DoF 动作路径；实现已改为默认读取 `model.get_action_dim(unnorm_key)`。

## 已提供的代码

补丁新增：

- Golden tensor dump；
- 合并或拆分式 Vision + Projector ONNX 导出；
- Thor 上的 `trtexec` engine build 脚本；
- 基于 TensorRT 10 tensor API、直接使用 Torch CUDA buffer 的 Python runner；
- TensorRT Vision + PyTorch Llama 手动 prefill/KV-cache/decode Hybrid runtime；
- Vision embedding 数值验证；
- OpenVLA 内部 Llama checkpoint 提取；
- action metadata 导出；
- C++17 FP32 action decoder 与测试；
- 完整分阶段操作手册。

## 关键 Gate

| Gate | 检查内容 | 通过标准 |
|---|---|---|
| 1 | Processor 输入 | token、attention mask、pixel tensor 与 Golden 一致 |
| 2 | TRT projected embedding | 记录 max/mean error；初筛 cosine ≥ 0.999 |
| 3 | Multimodal embedding | BOS/视觉/文本顺序与长度完全一致 |
| 4 | 第一个 action token | exact match |
| 5 | 完整 action tokens | FP16 目标 100% exact match |
| 6 | 连续动作 | token 相同时应一致；记录最大误差 |
| 7 | 机器人 rollout | 成功率、稳定性和安全约束无回退 |

Cosine similarity 只能通过 Gate 2，不能代替 token 或 rollout 验证。

## 本地验证状态

已完成：

- 所有新增 Python 文件通过 `compileall`；
- shell build 脚本通过 `bash -n`；
- patch 通过 `git diff --check`；
- C++ 工程成功完成 CMake configure。

未在本机完成：

- 模型加载、ONNX 导出、TensorRT build、CUDA 推理；当前执行环境不是 Jetson，且没有 OpenVLA 的 PyTorch/Transformers/TensorRT 运行栈。
- 本机 AppleClang 15 与已安装的 macOS 26 SDK 标准库不兼容，C++ 编译在系统头文件内失败；这是本机工具链问题。C++17 decoder 仍需在 Thor 的 GCC/CUDA 工具链上运行随附的 CTest。

## 推荐的下一次实机执行

在 Thor 上先只完成以下四条命令，并保留完整日志：

```bash
python deploy/tensorrt/export/00_dump_golden.py --dtype bf16

python deploy/tensorrt/export/01_export_vision_projector_onnx.py \
  --mode combined --opset 17

bash deploy/tensorrt/build/build_vision_engine.sh

python deploy/tensorrt/runtime/hybrid_runtime.py \
  --engine deploy/tensorrt/artifacts/engines/vision_projector_fp16.plan \
  --compare-reference
```

如果合并 ONNX 导出失败，立即改用 `--mode split` 定位 DINOv2、SigLIP 或 projector，而不是先引入自定义 TensorRT plugin。
