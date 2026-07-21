# OpenVLA-7B 当前测试方法与实测报告

> 报告日期：2026-07-06（Asia/Shanghai）  
> 数据来源：本目录测试代码、`../logs/` 中现有 JSON/JSONL 结果、当前本地模型配置和当前运行环境。  
> 原则：下文只把日志中真实存在的数据标为“实测”；代码默认值、推断项和未覆盖项会单独说明。

## 1. 结论摘要

- 当前推理不是 FP32，也不是 FP16，而是 **BF16（`torch.bfloat16`）GPU 推理**。
- 当前设备为 `cuda:0`，实测 GPU 是 NVIDIA GeForce RTX 5090（32,607 MiB，compute capability 12.0）。
- 当前模型为本地 `/share_data/public/models/openvla-7b`，使用 PyTorch eager + Transformers，注意力配置为 `sdpa`。
- 模型输入为单张图、batch size 1；动作通过 `predict_action()` 自回归生成 7 个离散 token，再按 `bridge_orig` 统计量反归一化为 7 维动作。
- 最新稳定的 100 张真实图分段测试中：
  - processor 到动作输出平均 **161.996 ms**，约 **6.17 action/s**；
  - 纯推理段平均 **153.793 ms**；
  - LLM 总计平均 **137.772 ms，占 E2E 85.05%**；
  - 视觉骨干平均 **11.876 ms，占 7.33%**；
  - projector 平均 **0.257 ms，占 0.16%**；
  - 峰值 PyTorch allocated / reserved 显存为 **14,757.37 / 14,810.00 MiB**。
- 算子 profiler 的一次完整动作预测记录到 **83 种 ATen 算子、50,378 次 ATen 调用**。线性层/GEMM 是主要 GPU 计算路径。

## 2. 测试对象与环境

### 2.1 硬件与软件

| 项目 | 当前值 |
|---|---:|
| GPU | NVIDIA GeForce RTX 5090 |
| GPU 数量 | 8 |
| 本次使用 | `cuda:0`，单卡、单进程 |
| GPU 显存 | 32,607 MiB |
| GPU compute capability | 12.0 |
| NVIDIA driver | 595.58.03 |
| GPU power limit | 575 W |
| CPU | 2 × Intel Xeon Gold 6530，合计 64 核 / 128 线程 |
| 系统内存 | 503 GiB |
| OS kernel | Linux 6.8.0-124-generic x86_64 |
| Python | 3.10（`openvla5090` Conda 环境） |
| PyTorch | 2.7.0+cu128 |
| CUDA runtime | 12.8 |
| Transformers | 4.40.1 |
| Tokenizers | 0.19.1 |
| timm | 0.9.10 |

### 2.2 模型配置

| 项目 | 当前值 / 观察值 |
|---|---|
| 模型目录 | `/share_data/public/models/openvla-7b` |
| 模型类型 | OpenVLA for action prediction |
| 视觉骨干 | fused DINOv2 ViT-L/14 + SigLIP ViT-SO400M |
| 图像输入 | 224 × 224，`resize-naive` |
| projector | 3 个 Linear 的 GELU MLP：2176 → 8704 → 4096 → 4096 |
| LLM | Llama-2-7B，hidden size 4096，MLP size 11008（由运行时 shape 观察） |
| LLM 最大长度 | 2048 |
| 词表大小 | 32,064 |
| 动作离散 bins | 256 |
| 动作维数 | 7 |
| 反归一化 key | `bridge_orig` |
| attention 配置 | PyTorch SDPA (`sdpa`) |
| 采样方式 | `do_sample=False`，贪心/确定性生成 |

一次 profiler 样例中，文本 prompt 产生 19 个 token，融合视觉产生 256 个 token，首次 LLM forward 的序列长度为 275。随后以序列长度 1 自回归解码。7 维动作一共生成 7 个 token：第一个动作 token 在 prefill 中产生，后续有 6 次 decode forward。

## 3. 当前到底用什么精度推理

答案是 **BF16**。

代码中的 `model_dtype()` / `dtype()` 在设备名以 `cuda` 开头时返回：

```python
torch.bfloat16
```

该 dtype 同时用于：

1. `from_pretrained(..., torch_dtype=torch.bfloat16)`：以 BF16 加载模型浮点权重；
2. `inputs.to(cuda, dtype=torch.bfloat16)`：把图像浮点输入转为 BF16；
3. profiler 中实际出现 `__nv_bfloat16`、BF16 GEMM、BF16 convolution 和 BF16 Flash-SDPA kernel。

当前代码没有使用：

- `torch.float16` / FP16；
- CUDA autocast；
- INT8 / INT4 / FP8 量化；
- TensorRT / TensorRT-LLM；
- `torch.compile`。

如果切到 CPU，代码 fallback 为 FP32；本报告的所有现有日志均为 `cuda:0`，因此都是 BF16 数据。部分归一化、归约、softmax 或 NumPy 后处理可能内部使用 FP32 累加，这不改变“模型权重和主要矩阵计算为 BF16”的结论。

## 4. 当前代码如何完成一次推理

当前工具的公共流程如下：

1. `runtime_env.py` 自动 source `env.sh`；
2. 强制 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`；
3. 从本地目录加载 processor 和 OpenVLA 模型，`local_files_only=True`；
4. 读取 RGB 图像，构造 prompt：

   ```text
   In: What action should the robot take to <instruction>?
   Out:
   ```

5. processor 完成 resize、normalize 和 tokenizer；
6. 输入搬到 `cuda:0` 并转换为 BF16；
7. `torch.inference_mode()` 下调用：

   ```python
   model.predict_action(
       **inputs,
       unnorm_key="bridge_orig",
       do_sample=False,
   )
   ```

8. 模型依次执行 fused vision backbone、projector、LLM prefill 和 6 次 LLM decode；
9. 7 个动作 token 反离散化、按 `bridge_orig` 反归一化，得到 7 维 NumPy action。

## 5. 各测试脚本的真实测试方法

| 脚本 | 用途 | 当前计时边界 | 默认预热/样本 |
|---|---|---|---|
| `check_model.py` | 冒烟测试 | 不统计延迟，只验证本地加载和动作输出 | 1 张 224×224 灰图 |
| `bench_e2e.py` | 基础端到端 benchmark | processor、H2D、`predict_action` 分开；图像在循环前创建/加载 | 默认 warmup 10、iters 100 |
| `infer_images.py` | 多张真实图批量推理 | 只统计 `predict_action`；processor/H2D 不计入；没有预热 | 默认全部匹配图片 |
| `bench_stages.py` | 真实图分阶段延迟 | processor 开始到 action 返回；用同步 hook 拆 vision/projector/LLM | 默认 warmup 5、全部图片 |
| `eval_operators.py` | 算子、模块、shape 清单 | 一次 shape 收集、一次预热、一次 profiler 推理 | profiler 样本 1 次 |

重要边界：

- 所有延迟测试都 **不包含模型加载时间**；
- `bench_stages.py` 在计时开始前已经执行 `Image.open(...).convert("RGB")`，所以 **不包含 JPEG 磁盘读取和 RGB 解码时间**；
- 当前结果不包含相机采集、REST/网络、请求排队、机器人控制器和机械臂执行时间；
- `bench_stages.py` 在各模块 hook 前后调用 `torch.cuda.synchronize()`，以避免把异步 kernel 提交时间误当成执行时间；同步/hook 本身也带来少量扰动；
- `other_ms` 是 `inference - vision - projector - LLM` 的残差，包含 embedding/拼接、token 选择、反离散化/反归一化、Python/hook 和同步开销，不能严格称为纯后处理。

结果文件没有持久化完整命令行和 warmup 参数；下文的样本数、图片名、指令和模型配置可由日志直接确认，warmup 数量只能确认代码默认值，无法排除运行时被命令行覆盖。

## 6. 基础端到端实测

来源：`../logs/openvla_thor_benchmark_1783328778.summary.json`，2026-07-06 17:06（本地时间）。

- 样本：100 次；
- 输入：结果日志未持久化 `--image` 参数；脚本默认值是内存中创建的 224×224 灰图，日志中的 action 也与当前默认灰图冒烟测试输出一致；
- instruction：`move the robot arm forward`；
- batch：1；
- device：`cuda:0`；
- dtype：BF16；
- attention：SDPA。

| 指标 | mean (ms) | p50 | p95 | p99 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| processor | 3.449 | 1.615 | 1.767 | 71.008 | 1.283 | 99.326 |
| H2D | 0.342 | 0.287 | 0.310 | 0.318 | 0.227 | 5.948 |
| `predict_action` | 151.897 | 151.639 | 153.738 | 154.336 | 150.742 | 154.345 |
| model E2E | 155.689 | 153.565 | 155.878 | 225.502 | 152.672 | 256.690 |

`predict_action` 平均吞吐约 **6.58 action/s**；按 model E2E 计算约 **6.42 action/s**。processor 和 E2E 的 mean/p99 被少量 CPU 侧离群点拉高，而 GPU `predict_action` 分布相对稳定。

显存峰值：

- allocated：14,757.44 MiB；
- reserved：14,810.00 MiB。

## 7. 真实图片分阶段延迟

三份结果都使用 100 张 `bridge_sample_0001.jpg` 至 `bridge_sample_0100.jpg`，instruction 为 `pick up the object`，batch size 1，BF16，SDPA，`bridge_orig`。

### 7.1 三次重复运行

| 结果文件（时间） | E2E mean | E2E p50 | E2E p95 | E2E p99 | inference mean | LLM decode mean |
|---|---:|---:|---:|---:|---:|---:|
| `1783332045`（18:01） | 173.710 | 173.501 | 173.966 | 174.889 | 165.282 | 110.039 |
| `1783332834`（18:14） | 164.247 | 162.402 | 163.115 | 250.005 | 154.232 | 100.022 |
| `1783332910`（18:15） | **161.996** | **161.897** | **162.529** | **163.681** | **153.793** | **99.690** |

第二次运行有两个明显 CPU/H2D 离群点：preprocess 最大 99.381 ms，H2D 最大 5.934 ms，导致 E2E p99 250.005 ms。第一次运行的 GPU decode 稳定但整体比后两次慢约 10 ms。日志没有记录 GPU 时钟、温度、并发负载，因此不能武断归因。

### 7.2 最新稳定运行的详细拆分

来源：`../logs/latency_breakdown_1783332910.summary.json`。

| 阶段 | mean (ms) | p50 | p95 | p99 | min | max | E2E 占比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| preprocess | 7.973 | 7.860 | 8.065 | 9.779 | 7.693 | 16.110 | 4.92% |
| H2D | 0.230 | 0.229 | 0.246 | 0.258 | 0.215 | 0.267 | 0.14% |
| vision backbone | 11.876 | 11.878 | 11.924 | 12.075 | 11.782 | 12.141 | 7.33% |
| projector | 0.257 | 0.256 | 0.260 | 0.262 | 0.253 | 0.262 | 0.16% |
| LLM prefill | 38.082 | 38.085 | 38.158 | 38.170 | 37.990 | 38.178 | 23.51% |
| LLM decode（6 token） | 99.690 | 99.658 | 100.103 | 100.455 | 99.164 | 100.715 | 61.54% |
| LLM decode / token | 16.615 | 16.610 | 16.684 | 16.742 | 16.527 | 16.786 | — |
| LLM total | 137.772 | 137.767 | 138.165 | 138.594 | 137.168 | 138.718 | 85.05% |
| other/residual | 3.889 | 3.886 | 3.945 | 4.034 | 3.819 | 4.055 | 2.40% |
| inference total | 153.793 | 153.764 | 154.221 | 154.601 | 153.271 | 154.909 | 94.94% |
| E2E | **161.996** | **161.897** | **162.529** | **163.681** | **161.253** | **170.365** | 100% |

平均 E2E 对应约 **6.17 action/s**。如果控制目标为 5 Hz（周期 200 ms），当前单次本地 processor→action 延迟在这组测试中满足 200 ms 周期；这不代表完整机器人闭环已经满足，因为尚未计入相机、网络、队列、安全过滤和执行器控制。

## 8. 冷启动与无预热批量推理

`infer_images.py` 没有 warmup。100 张日志 `openvla_infer_1783331395.jsonl` 中：

| 指标 | 值 |
|---|---:|
| mean | 170.492 ms |
| p50 | 162.663 ms |
| p95 | 167.034 ms |
| min | 160.671 ms |
| max（第一张） | 905.764 ms |

第一张图片出现约 906 ms 的首次调用开销，后续大部分样本回落到约 161–167 ms。该离群值通常会包含 CUDA 库和 kernel 的 lazy initialization，但当前日志没有进一步拆分原因。因此部署服务应在接收真实机器人请求前执行若干次 warmup，并设置 readiness，而不能把“进程已启动”直接视为“已具备稳定实时性能”。

## 9. 算子分布

来源：`../logs/operator_inventory.json` 和 `../logs/operator_inventory.report.txt`。方法是一次 shape 采集、一次 warmup、一次完整 `predict_action` profiler。

### 9.1 静态叶子 Module 分布

| Module | 数量 | 占 1,309 个叶子模块比例 |
|---|---:|---:|
| Linear | 437 | 33.38% |
| Identity | 322 | 24.60% |
| Dropout | 211 | 16.12% |
| LayerNorm | 105 | 8.02% |
| LlamaRMSNorm | 65 | 4.97% |
| GELU | 54 | 4.13% |
| LayerScale | 48 | 3.67% |
| LlamaRotaryEmbedding | 32 | 2.44% |
| SiLU | 32 | 2.44% |
| Conv2d | 2 | 0.15% |
| Embedding | 1 | 0.08% |

这是结构数量，不是耗时占比。模型处于 `eval()`，因此 Dropout/Identity 数量多并不表示它们消耗了同等计算时间。

### 9.2 ATen 动态调用次数

一次完整动作预测包含 83 种 ATen 类型，共 50,378 次 ATen 调用。调用次数最多的类型如下：

| ATen 算子 | 调用次数 | 调用占比 |
|---|---:|---:|
| `as_strided` | 7,746 | 15.38% |
| `transpose` | 4,055 | 8.05% |
| `view` | 3,672 | 7.29% |
| `to` | 2,738 | 5.43% |
| `reshape` | 2,341 | 4.65% |
| `empty` | 2,264 | 4.49% |
| `mul` | 2,092 | 4.15% |
| `empty_strided` | 1,875 | 3.72% |
| `matmul` | 1,799 | 3.57% |
| `_unsafe_view` | 1,799 | 3.57% |
| `linear` | 1,782 | 3.54% |
| `t` | 1,782 | 3.54% |
| `copy_` | 1,669 | 3.31% |
| `slice` | 1,636 | 3.25% |
| `_to_copy` | 1,600 | 3.18% |
| `mm` | 1,575 | 3.13% |

大量 view/transpose/as_strided 是张量元数据或布局操作；真正的主要 GPU 数值计算是 Linear/GEMM、attention、norm、激活和视觉 patch convolution。

### 9.3 profiler 中累计 CUDA 时间最高的 ATen 事件

| ATen 事件 | count | 累计 CUDA 时间 |
|---|---:|---:|
| `linear` | 1,782 | 90.074 ms |
| `matmul` | 1,799 | 86.614 ms |
| `mm` | 1,575 | 86.348 ms |
| `addmm` | 207 | 3.726 ms |
| `mul` | 2,092 | 3.299 ms |
| `cat` | 1,075 | 3.138 ms |
| `copy_` | 1,669 | 2.929 ms |
| `to` / `_to_copy` | 2,738 / 1,600 | 2.704 ms（各自 inclusive） |
| `scaled_dot_product_attention` | 275 | 2.633 ms |
| `add` | 1,469 | 1.701 ms |
| `mean` | 455 | 1.146 ms |
| `layer_norm` | 102 | 0.564 ms |

这些是 profiler 的 **inclusive 累计时间**：例如 `linear` 内部调用 `matmul`，`matmul` 又落到 `mm`，三者时间重叠，绝对不能把 90.074 + 86.614 + 86.348 ms 相加。该表用于判断算子热点和 NPU 覆盖优先级，不用于替代端到端计时。

Profiler 同时确认 `sdpa` 路径实际触发了 PyTorch Flash-SDPA kernel，但这不等同于代码配置 `flash_attention_2`；当前配置仍然是 `attn_implementation="sdpa"`。

## 10. 关键张量 shape

| 位置/算子 | 典型输入 → 输出 |
|---|---|
| DINO patch Conv2d | `[1,3,224,224] → [1,1024,16,16]`，kernel/stride 14 |
| SigLIP patch Conv2d | `[1,3,224,224] → [1,1152,16,16]`，kernel/stride 14 |
| DINO token stream | `[1,261,1024]` |
| SigLIP token stream | `[1,256,1152]` |
| fused vision feature | `[1,256,2176]` |
| projector Linear 1 | `[1,256,2176] → [1,256,8704]` |
| projector Linear 2 | `[1,256,8704] → [1,256,4096]` |
| projector Linear 3 | `[1,256,4096] → [1,256,4096]` |
| prompt embedding | `[1,19] → [1,19,4096]` |
| LLM prefill hidden | `[1,275,4096]` |
| LLM decode hidden | `[1,1,4096]` |
| Llama MLP / SiLU | `[1,275,11008]` 或 decode `[1,1,11008]` |

从硬件优化角度看，优先级应是：BF16 Linear/GEMM/GEMV → KV-cache 下的小 batch decode → RMSNorm/LayerNorm 与 elementwise fusion → SDPA/Flash Attention → vision Conv2d/ViT → concat/copy/layout 开销。

## 11. 数据解释与限制

当前结果可以客观回答：

- 这份 OpenVLA-7B 能否在 RTX 5090 上本地运行；
- batch=1、BF16、SDPA 时，processor→action 的本地延迟；
- vision/projector/LLM 的分段耗时；
- 一次动作生成涉及的模块、ATen 算子和典型 shape；
- 当前显存峰值。

当前结果不能回答：

- 模型加载/服务冷启动耗时；
- FP32、FP16、BF16 的严格对照；
- batch > 1 的吞吐曲线；
- 多 GPU 并行或多实例吞吐；
- REST、网络和并发请求延迟；
- GPU 利用率、功耗、能耗/动作；
- Nsight kernel 级 self time、memory bandwidth、Tensor Core 利用率；
- 在真实机器人上的任务成功率、动作安全性或轨迹质量；
- `bridge_orig` 动作语义是否适配目标机器人。

测试图片来自 BridgeData V2 图片样例，但日志没有动作真值，因此只能测性能和输出稳定性，不能计算动作误差或成功率。

## 12. 复现实验命令

### 12.1 基础端到端 benchmark

```bash
/home/bruce/miniconda3/envs/openvla5090/bin/python -u \
  /workspace/openvla/tools/bench_e2e.py \
  --warmup 10 \
  --iters 100
```

### 12.2 100 张真实图批量推理

```bash
/home/bruce/miniconda3/envs/openvla5090/bin/python -u \
  /workspace/openvla/tools/infer_images.py \
  --image-dir /workspace/openvla/test_data \
  --glob '*.jpg' \
  --instruction 'pick up the object' \
  --limit 100
```

### 12.3 模型分段延迟

```bash
/home/bruce/miniconda3/envs/openvla5090/bin/python -u \
  /workspace/openvla/tools/bench_stages.py \
  --image-dir /workspace/openvla/test_data \
  --glob '*.jpg' \
  --instruction 'pick up the object' \
  --limit 100 \
  --warmup 5
```

### 12.4 算子与 shape 清单

```bash
/home/bruce/miniconda3/envs/openvla5090/bin/python -u \
  /workspace/openvla/tools/eval_operators.py \
  --image /workspace/openvla/test_data/bridge_sample_0001.jpg \
  --instruction 'pick up the object'
```

## 13. 建议的下一轮测试

为了把当前报告扩展成可用于部署决策的报告，建议依次补充：

1. 在脚本中持久化完整 CLI、git/hash、驱动、GPU 时钟、温度和功耗；
2. 测量模型加载时间与 warmup readiness；
3. 使用同一组图片做 BF16/FP16/FP32 输出误差与延迟对照；
4. 做 REST 单并发和并发压力测试，记录排队时延；
5. 用 Nsight Systems/Compute 获取不重叠的 kernel self time 和带宽；
6. 采集 GPU 功耗，报告平均/峰值功耗与 joule/action；
7. 接入目标机器人数据，验证动作坐标、单位、夹爪方向、限幅和成功率。

现阶段最可信的部署性能基线是：**单卡 RTX 5090、batch=1、BF16、SDPA、本地 processor→7维动作，稳定 E2E 约 162 ms，约 6.17 action/s，PyTorch reserved 显存约 14.46 GiB。**
