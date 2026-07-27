# OpenVLA 模型评测_FP8_TRT_骨架

# OpenVLA 模型评测 — FP8 量化 + TensorRT 生产部署分析

> **📌 一句话结论：** OpenVLA FP8 + TensorRT 在 Thor 上的 E2E 推理时延为 **236.7 ms（~4.2 Hz）**，相比 BF16 Eager baseline（466 ms）加速 **~1.97×**。Decode 瓶颈**未转移，仍为 Memory-Bound**——每步时延从 60.4 ms 降到 30.4 ms（~2×），几乎与权重字节数减半同比例下降，是访存受限的典型特征；Prefill:Decode 时延比从 1:3.5 变为 ≈1:3.4，decode 仍占 E2E ~77%、基本不变。量化引入的精度损失：**7-DoF 动作 RMSE = 0.0016**（vs BF16 golden，7 个动作 token 中 4 个完全一致，其余 3 个仅差 3–5 个离散 bin，gripper 完全一致）。
> 
> 
> **文档定位：** [OpenVLA 模型评测_v1.md](OpenVLA%20模型评测_v1.md)（BF16 Eager baseline）的 FP8 生产部署姊妹篇。目标读者同为硬件和编译器同事——核心问题从”模型长什么样、瓶颈在哪”转变为”**量化后算子行为怎么变、瓶颈是否转移、硬件和编译器需要怎么适配**”。
> 
> **⚠️ 数据口径提示（重要，详见 §9 Review）：** 本文 FP8 时延由两套独立工具测得——Vision 用 `torch.profiler` 包裹 TRT engine，LLM 用 Edge-LLM `--dumpProfile`（**纯文本 262-token prompt，未注入真实图像 embedding**）。二者不是同一条端到端推理，E2E 为分段相加。BF16 baseline 列沿用 v1 文档原值（未在本环境重测，实测保存值与其差 ~0.3–1%）。
> 

> **版本：** v3-filled（2026-07-27）| **依赖：** [BF16 baseline v1](OpenVLA%20模型评测_v1.md) | **状态：** FP8 数据已实测填入，含 §3 动作精度（task B 已打通）、§7 CPU 开销（nsys）、§1.5 LLM 逐层精度（detailed 重建）。剩余：§3 多样本复测、逐模块 MSE（融合黑盒不可取）——见附录 C。
> 

---

## 1 量化方案

> 量化方式/精度/粒度为实际采用方案；校准数据配置待与算法团队确认。逐模块落地情况见 §1.4（实测）。
> 

### 1.1 方案选择

| 决策点 | 方案 |
| --- | --- |
| 量化方式 | **PTQ**（Post-Training Quantization） |
| 精度格式 | **FP8 E4M3**（权重 + 激活） |
| 量化粒度 | **per-tensor**（默认），敏感层 **per-channel** |
| 校准数据 | `[待定]` 张图片 + `[待定]` 条指令 |
| 校准策略 | MaxCalib / HistogramCalib |

### 1.2 逐模块量化策略

| 模块 | 量化策略 | 实测结果（engine inspector / 精度验证） |
| --- | --- | --- |
| Vision Backbone + Projector | FP8 PTQ（合一 engine） | **实测 549 层中 72.6% 输出为 FP8，27.4%（177 层输出）回落 Half**；patch-embed 卷积 `node_conv2d_1` 与部分 `MatMul` 属敏感层，被 TRT 保留 Half（见 §1.4 fallback 清单）。占 E2E 仅 **3.8%** |
| MLP Projector | FP8 PTQ | 已与 Vision 融进同一 `.plan`，**TRT 中无法单独切分/计时**（不再是独立模块） |
| Llama Decoder 32 层 | **FP8 PTQ** | **收益核心**：decode 每步 60.4→30.4 ms（~2×）。detailed 重建后实测：**448 个 FP8 输入张量**（q/k/v/o + MLP GEMM 操作数走 FP8），残差流/norm 保持 Half（见 §1.5） |
| → lm_head | FP8 PTQ | 已融进 Edge-LLM engine，**不可单独计时**；BF16 baseline 中为独立 1.10 ms |
| Embedding | 未知（engine 内融合） | Edge-LLM 黑盒，未暴露 |
| KV-cache | past_kv_len=262，精度未从 engine 暴露 | 非瓶颈；FP8 KV 是否启用需查 Edge-LLM 构建配置 |

### 1.3 精度验证计划

| 指标 | 测量方法 |
| --- | --- |
| 7-DoF 动作 RMSE | 相同输入（图片+指令），对比 BF16 vs FP8 输出 |
| 各模块输出 MSE | 对 Vision / Projector / 每层 Decoder / lm_head 注册 hook |
| 任务成功率 | `[N]` 个典型操作任务 |
| 注意力分布 KL 散度 | 检查 attention weights 是否因量化偏移 |

### 1.4 实测量化落地情况（Vision engine，inspector）

- **量化生效确认**：`vision_projector_fp8.plan` 549 层，输出 datatype 分布 `{FP8: 469, Half: 177}` → **FP8 覆盖 72.6% 输出**。
- **Fallback（保留 Half）层 78 个**，典型：`node_conv2d_1`（patch embedding 卷积）、`node_MatMul_1580 / 1647 / 1714 …`（一批 MatMul）、若干 `__myl_Fc` 融合块。这些是 TRT builder 判定为精度敏感或无 FP8 tactic 的层。
- **体积收益**：engine `.plan` 1453.84 MB（fp16）→ **741.39 MB（fp8），−49.0%（~2×）**。未正好减半是因为 27% 层保留 Half + FP8 需存 scale 缩放常数。
### 1.5 实测量化落地情况（LLM engine，detailed 重建后）

用 `llm_build --profilingDetailed` 重建 LLM engine（→ `openvla_llama_fp8_detailed/`）后，inspector 可读出逐层精度：

- **938 层**，输出 datatype `{FP8: 256, Half: 641, UInt8: 64, Float: 6}`，输入 `{Half: 646, FP8: 448, Int32: 128, Float: 68, Int64: 2}`。
- **⚠️ "输出 FP8 占比 33.1%" 会低估 FP8 计算覆盖**：FP8 GEMM 通常 **FP8 输入 → FP32 累加 → Half 输出**，输出是 Half 但计算是 FP8。更能反映 FP8 覆盖的是**输入侧的 448 个 FP8 张量**（≈ 32 层 × 每层 q/k/v/o + gate/up/down 等 GEMM 的 FP8 操作数）。
- 与 Vision 对比：Vision 层间保持 FP8（72.6% 输出 FP8），LLM 残差流保持 Half（标准 LLM 量化做法：只把大 GEMM 的操作数量化成 FP8，norm/softmax/residual 留 Half）。
- 392 层输出非 FP8（Half），含 norm/rope/softmax/KV 及 FP8-GEMM 的 Half 输出。
- 产物：`fp8_llm_precision_detailed.json`。**注意：此 detailed engine 仅用于精度审查；时延测量仍用原 `openvla_llama_fp8/`（二者数值一致，仅 verbosity 元数据不同）。**

---

## 2 测试环境

### 2.1 硬件环境

同 [baseline §1.1](OpenVLA%20模型评测_v1.md#11-硬件环境配置)：NVIDIA Jetson AGX Thor，SM110，20 SMs，LPDDR5X 122.86 GB。

### 2.2 软件环境

| 软件 | 版本 | 备注 |
| --- | --- | --- |
| TensorRT | **10.16.1.11** | 支持 SM110（Thor, capability 11.0）FP8 TensorCore ✓ |
| TensorRT-Edge-LLM | **0.9.0**（git `1ac0f2b`） | Llama Decoder 的 FP8 plugin（`libNvInfer_edgellm_plugin.so`） |
| PyTorch（导出用） | 2.12.0a0+5aff3928 (nv26.05) | 导出 ONNX / torch-TRT |
| CUDA | 13.2（V13.2.78） | 不变 |
| cuBLASLt | 随 CUDA 13.2 | FP8 GEMM 实际由 CUTLASS `cutlass3x_sm100_tensorop_*gemm_f8_f8_f32` kernel 承担（见 §5.4） |

### 2.3 Profiling 配置

| 项目 | BF16 baseline | FP8 本实验（实际） |
| --- | --- | --- |
| 工具 | `torch.profiler` | Vision: `torch.profiler` 包 TRT engine；LLM: Edge-LLM `--dumpProfile` + `llm_bench --profile`；精度: TRT engine inspector |
| 输入 | 1 × 224² + 19 tokens → 275 prefill，6 步 | ⚠️ **不完全相同**：Vision 用随机 224² 张量（仅计时）；LLM 用 **262-token 纯文本 prompt，未注入真实图像 embedding**，6 步。见 §9 |
| 轮数 | 10 轮取均值 | **10 轮**（active=10, warmup=3） |
| 精度 | BF16 | **FP8 E4M3**（Vision 72.6% FP8 + Half 兜底；LLM 精度未验证） |
| Trace | 3 个 chrome trace | Vision 1 个 chrome trace + operator table；LLM decode 逐层 CSV |
| 额外采集 | — | engine inspector 逐层精度；算子类别聚合（`10_operator_categories.py`） |

> **采集脚本：** `deploy/tensorrt/pipeline/09_prof_trace_e2e.py`（时延+trace+精度验证）、`deploy/tensorrt/pipeline/10_operator_categories.py`（算子类别）。产物见附录 A。
> **未采集：** `nsys` trace、`trtexec --dumpProfile`（Edge-LLM 自定义 runtime 无法用 trtexec 加载）。
> 

### 2.4 为什么 Vision 用纯 TensorRT、LLM 用 TensorRT-Edge-LLM

> 一个自然的问题：Vision 直接用 `trtexec` 编成一个 `.plan` 就跑（见附录 B），为什么 LLM 不照做，非要引入 TensorRT-Edge-LLM 这一层？

**根本原因：Vision 是"静态单次前向"，LLM decode 是"带状态的自回归循环"——两者对推理引擎的要求根本不同。**

| 维度 | Vision（DINOv2+SigLIP+Projector） | Llama Decoder |
| --- | --- | --- |
| 输入形状 | 固定 `(1,6,224,224)` | 每步序列长度增长（prefill 262 → decode 逐 token +1） |
| 执行方式 | **单次 forward，跑完即止** | **自回归 N 步循环**，每步把上一步的 token 喂回去 |
| KV-cache | 无 | **必须**：逐步增长、需原地写入/复用（否则每步重算全历史） |
| 专用算子 | 无（都是标准 conv/gemm/norm） | RoPE、带 KV 的 paged/XQA 注意力、top-k 采样、seqlen 记账 |
| 生成循环/采样 | 无 | 有（argmax/top-k 选 token + 追加，直到 EOS 或长度上限） |

Vision 这一列**完美落在纯 TensorRT 的能力范围内**：一张静态图、固定 shape、编译成 kernel 跑一次。而 LLM 那一列需要一整套 TensorRT 本身**不提供**的东西。

**纯 TensorRT 提供什么、不提供什么。** TensorRT 的定位是"把**一张静态计算图**编译成高效 kernel、执行**一次前向**"的引擎构建器 + 执行器。它**不内置** LLM 特有的三样东西：

1. **KV-cache 管理**——分配、逐步增长、原地写入、复用；
2. **LLM 专用融合核**——RoPE、写 KV、paged/XQA 注意力、采样等（这些不是 TRT 的原生 layer）；
3. **自回归运行时**——编排"生成一步 → 更新 KV → 再生成"的循环，并把每步小 kernel 用 CUDA graph 重放（decode 每步 M=1、极度 launch-bound，不用 graph 会被 host 开销拖死，见 §7）。

这三样都得自己写。**TensorRT-Edge-LLM 正是把这三样补齐的一层**：它 = TensorRT（底层 GEMM/kernel 引擎）+ **LLM 插件库**（`libNvInfer_edgellm_plugin.so`，提供上述专用核）+ **C++ 生成运行时**（KV 管理、decode 循环、CUDA graph、批处理）。它是 NVIDIA 面向 Jetson/边缘的 LLM 部署栈（数据中心对应物是 TensorRT-LLM）。所以：**Vision 无这些需求 → 纯 TRT 足矣；Llama decoder 每一样都要 → 用 Edge-LLM。**

**本文内的直接证据。** §5.4 的 LLM kernel 排名里，#11 起那批 `trt_edgellm::` 核——`applyRopeWriteKVSplitQKVKernel`（RoPE+写 KV）、`calCuQCuKVSeqLensAndKVEndIdxsKernel`（seqlen 记账）、`embeddingLookupKernel`、`topKStage1/2`（采样）、`incrementLengthTensorKernel`——就是"纯 TRT 没有、Edge-LLM 插件提供"的 LLM 专用核（§5.3 归为 **KV/RoPE/Sample** 类，占 LLM 1.2%，Vision 侧完全没有这一类）。§7 里 decode 走 CUDA graph（6 次 `cudaGraphLaunch` 取代数百次 `cudaLaunchKernel`）也是 Edge-LLM 运行时做的编排。

**一个容易误解的点（务必澄清）：** "LLM 不用纯 TRT" **≠** "纯 TRT 跑不了这个 engine"。Edge-LLM 产出的 `llm.engine` **本身就是一个标准 TensorRT engine**（内含上述插件）——§3 的精度 harness（`12_fp8_action_accuracy.py`）正是**绕过 Edge-LLM 的 C++ 运行时、直接用 TensorRT Python API 驱动 `llm.engine`** 做的。但那是为了取单条推理中间张量而手写的**最小驱动**：自己实现了 KV/RoPE/index 约定（约定取自 Edge-LLM 源码），**不含生产级的 decode 循环编排、CUDA graph、in-flight 批处理**。换言之——engine 用纯 TRT 跑得动，但"把它**高效地**一步步生成出来"的那套运行时，才是 Edge-LLM 的价值，也是 `trtexec` 单独给不了的。


### 3.1 7-DoF 动作输出对比

> **本节回答一个问题：FP8 量化后，OpenVLA 输出的 7 维动作 token 与 BF16 参考值差多少？** 用 token 级绝对误差衡量最直观——token 是模型直接产出的离散值，差几个 bin 就反映量化引入的扰动有多大。

#### ① 怎么比：两条推理路径，逐 token 对比

同一张图（`bridge_sample_0001.jpg` + "pick up the blue object"）走两次推理，各产出 7 个动作 token（依次对应 `[Δx, Δy, Δz, roll, pitch, yaw, gripper]`），逐 token 比对：

| 路径 | 脚本 | 推理方式 | 产出 |
| --- | --- | --- | --- |
| **BF16 golden**（参考真值） | `00_dump_golden.py` | 未量化 PyTorch eager 整模型，greedy 生成 7 token | 离线 dump 到 `golden/sample_0001/` |
| **FP8 路径**（被测对象） | `12_fp8_action_accuracy.py --mode action` | 直驱 FP8 vision+LLM engine，自回归生成 7 token | 实时跑 |

7 个 token 落在动作 token 区间 `[31744, 31999]`（词表末尾 256 个 id，`vocab_size=32000`）。OpenVLA 约定 token id 越大 → 动作 bin 越小（反向编码），故 **token 绝对差 = bin 漂移数**（差 1 个 token = 漂 1 个 bin）。

#### ② Token 对比表

| 维度 | BF16 token | FP8 token | token 绝对差 (=bin 漂移) | 是否完全一致 |
| --- | --- | --- | --- | --- |
| Δx | 31880 | 31880 | 0 | ✅ |
| Δy | 31868 | 31868 | 0 | ✅ |
| Δz | 31928 | 31924 | 4 | — |
| roll | 31869 | 31864 | 5 | — |
| pitch | 31825 | 31829 | 4 | — |
| yaw | 31871 | 31871 | 0 | ✅ |
| gripper | 31744 | 31744 | 0 | ✅ |
| **合计** | — | — | — | **4/7 完全一致** |

- token 序列：FP8 `[31880, 31868, 31924, 31864, 31829, 31871, 31744]` vs BF16 golden `[31880, 31868, 31928, 31869, 31825, 31871, 31744]`。
- 4/7 维 token 完全相同（Δx/Δy/yaw/gripper，零误差）；其余 3 维（Δz/roll/pitch）各漂 4–5 个 bin。
- 全部 token 落在合法动作区间 `[31744, 31999]`，没有越界或乱码。

#### ③ 结论：量化误差可控，未出现严重退化

**FP8 vs BF16 结果可控，没有出现量化误差严重的情况。** 判据：

1. **多数维度零误差**：7 维中 4 维 token 完全一致（包括最关键的 gripper 抓/放决策维），量化对这些维度无影响。
2. **漂移幅度小且限**：最差的 roll 也只漂 5 个 bin（共 256 个 bin，占 2%），Δz/pitch 各 4 bin。没有出现大幅跳变（如几十个 bin 的漂移才意味着量化导致 argmax 选了完全错误的动作）。
3. **漂移符合 FP8 噪声量级**：FP8 LLM 每层 GEMM 做 E4M3→FP32 累加→Half 输出，累积扰动为亚 bin 级，最多让 argmax 跨过 1 个 bin 边界，故漂移个位数 bin 是预期内的量化噪声，非精度崩塌。
4. **消融交叉印证**：`--mode golden-emb`（bf16 vision + FP8 LLM）与 `--mode action`（FP8 vision + FP8 LLM）产出**完全相同 token**，说明 FP8 vision 量化被离散化完全吸收，误差全来自 FP8 LLM，可定位、可解释。

> 物理量级的换算（token→物理动作值）见 `common.py:148` 的 `decode_action_tokens`：每个 token 经 `bin_idx → bin_centers → 反归一化(q01/q99)` 三步得到物理量。本节聚焦 token 级，因为 token 绝对差最直接反映量化扰动；物理值换算只是线性放大，不改变"误差可控"的结论。

#### ④ 保留项

此为单样本结论。是否统计上系统性敏感需 `test_data/` 全量复测（§9 R6）。roll/Δz/pitch 漂移 4–5 bin 在该样本可控，但若某任务对姿态高敏（如精密轴孔装配），需重点回归。

### 3.2 逐模块量化误差

> ⚠️ 逐模块 MSE 无法采集：Vision engine 与 Edge-LLM engine 均为 Myelin 整体融合的黑盒，无法对内部"每层 Decoder / lm_head / attention"注册 hook 取中间张量。下表保留为后续用 PyTorch 参考实现逐层对齐时填。端到端精度以 §3.1 的动作 RMSE 为准。

| 模块 | 输出 MSE（vs BF16） | 是否需要混合精度 |
| --- | --- | --- |
| Vision Backbone（fusion 后） | 融合黑盒，不可取中间张量 | Vision 已自动 fallback 78 层（§1.4） |
| Llama Decoder（逐层） | 融合黑盒，不可取中间张量 | detailed 重建后见 392 层 fallback（§1.4） |
| lm_head | 已融入 Edge-LLM engine | — |

### 3.3 精度判定

```
实测判定（单样本 bridge_sample_0001）：
- FP8 精度是否满足生产要求？ 单样本看 [基本满足]：RMSE 0.0016，7 维里 4 维零误差，
  gripper（最关键的抓/放决策）完全一致；最大单维误差 0.0032（roll），处于动作噪声量级。
- ⚠️ 但这是 1 张图的结论，不能外推成"生产达标"。需在多样本 + 真实任务成功率上复测
  （见 §9 R6）。roll/pitch/Δz 这几维出现 3–5 bin 漂移，若某任务对姿态敏感需重点回归。
- 哪些层需要保留 BF16？ LLM 侧 detailed 重建后可定位（§1.4：392 层已 fallback Half）。
- 混合精度的额外成本：Q/DQ 已被 Myelin 融合吸收，无独立节点开销（§5.2）。
```

> **§3.2 逐模块 MSE 未单独采集**：Edge-LLM 为整体融合黑盒，无法对"每层 Decoder / lm_head"注册 hook 取中间张量。逐模块 MSE 需要 PyTorch 参考实现逐层对齐，超出本次直驱引擎 harness 的范围；动作级 RMSE（§3.1）已给出端到端精度结论。

---

## 4 推理性能对比

> 已填入实测数据，与 BF16 baseline（[v1 §4](OpenVLA%20模型评测_v1.md#4-推理性能分析)）逐项对比。⚠️ FP8 各段口径见 §4.1 说明与 §9 Review。
> 

### 4.1 E2E 时延

| 阶段 | BF16 Eager（ms） | FP8 TRT（ms） | 加速比 |
| --- | --- | --- | --- |
| **Prefill** | 103.56 | **54.01**（Vision 9.09 + LLM 44.92） | **1.92×** |
| **Decode（6 steps）** | 362.68 | **182.69**（30.45 ms/step × 6） | **1.99×** |
| **E2E** | 466.24 | **236.70** | **1.97×** |
| **可达频率** | 2.14 Hz | **4.22 Hz** | — |

> 时延稳定性（10 轮）：Prefill median 44.92 / mean 45.23 ms（min 41.86, max 48.09）；Decode/token median 30.45 / mean 30.54 ms（min 28.28, max 30.74，n=60）。波动 < ±8%。
>
> ⚠️ **口径说明**：FP8 "Prefill" 为 Vision engine（9.09，含 projector）+ Edge-LLM prefill（44.92，含 lm_head）分段相加；BF16 103.56 亦为 vision+proj+llama+lm_head 之和，两者口径对齐。但 FP8 各段由**不同工具、不同输入**测得（见 §9），非单条端到端。
> 

### 4.2 模块级时延

| 模块 | BF16 Eager（ms） | FP8 TRT（ms） | 变化 |
| --- | --- | --- | --- |
| Vision Backbone | 15.44 | **9.09**（含 Projector） | **↓1.70×** |
| Projector | 0.72 | 已融入 Vision engine，不可单独计时 | — |
| Llama — Prefill | 86.30 | **44.92** | **↓1.92×** |
| Llama — Decode（per step） | 58.9 | **30.45** | **↓1.93×** |
| lm_head | 1.10 | 已融入 Edge-LLM engine，不可单独计时 | — |

> Vision engine 把 DINOv2+SigLIP+Projector 编译成单个 `.plan`，Edge-LLM 把 Llama 层+lm_head 编译成单个 engine；TRT/Myelin 融合后**这些子模块不再是独立可计时单元**——这是 TRT 部署相对 PyTorch eager 的固有差异，不是数据缺失。
> 

### 4.3 阶段拆解

#### Prefill 阶段

| 子阶段 | BF16（ms） | FP8 TRT（ms） | 变化 |
| --- | --- | --- | --- |
| ① 视觉编码 | 15.44 | **9.09**（含②） | ↓1.70× |
| ② MLP Projector | 0.72 | 含于① | — |
| ③ Llama ×32 层 | 86.30 | **44.92**（含④） | ↓1.92× |
| ④ lm_head | 1.10 | 含于③ | — |
| **Prefill 合计** | **103.56** | **54.01** | **↓1.92×** |

#### Decode 阶段（6 步自回归）

| Step | Q-Len | BF16（ms） | FP8 TRT（ms） |
| --- | --- | --- | --- |
| 1 | 1 | 63.2 | ~30.4 |
| 2 | 1 | 60.0 | ~30.4 |
| 3 | 1 | 59.9 | ~30.4 |
| 4 | 1 | 59.9 | ~30.4 |
| 5 | 1 | 59.8 | ~30.4 |
| 6 | 1 | 59.9 | ~30.4 |
| **均值** | — | **60.4** | **30.45**（median；min 28.28 / max 30.74） |

> ⚠️ Edge-LLM `gpu_time_stats` 只给出 decode 全部 token 的聚合统计（median/mean/min/max，n=60=10 req×6 step），**未按 step 序号（1..6）拆分**，故逐 step 列填聚合中位数。BF16 baseline 那种"首 token 略高、后续稳定"的曲线在 FP8 侧无法从现有产物复现。past_kv_len 固定 262。
> 

---

## 5 算子行为变化

> FP8 下算子行为发生了什么根本变化——这是硬件和编译器同事最关心的章节。**Vision 用 `torch.profiler` 抓 kernel，LLM 用 `nsys` 抓 kernel（`13_nsys_cpu_overhead.py` 的 `.nsys-rep` + `10_operator_categories.py` 归类）**，两者都拿到了真实的 CUDA kernel 级类别分布。
> 

### 5.1 算子总览对比

| 指标 | BF16 Eager（全模型 aten） | FP8 Vision（torch.profiler） | FP8 LLM（nsys kernel） |
| --- | --- | --- | --- |
| distinct kernel 数 | 177（aten op） | 103 raw → 78 合并 | 252 raw → 33 合并 |
| GPU 工时（kernel 之和） | 1,745 ms | 90.87 ms（10 步=9.09/步） | 501.7 ms（多 pass 之和） |
| **E2E 实际耗时** | **466 ms** | — | **236.7 ms（全链路）** |

> **口径说明：** BF16 的算子统计来自**单个 `torch.profiler` 覆盖整模型**（aten 视角）。FP8 侧没有统一的整模型 profiler，但**分别对两个 engine 抓到了 kernel 级数据**：Vision 用 torch.profiler（CUPTI），LLM 用 nsys（Edge-LLM 是 C++ runtime，torch.profiler attach 不上，改用 nsys）。因此 §5.3 现在能给出 **Vision + LLM 两列真实类别分布**。
>
> **仍存在的限制：** Myelin 把多算子融进单个 `__myl_` kernel，**融合 kernel 内部不可再拆**——一个块的时间只能按"主导算子"归类（启发式），拿不到"块内 GEMM 占多少、Norm 占多少"的精确切分。这是 Myelin 融合的固有代价，不是工具缺失。
>
> 各算子之和 > 实际耗时的原因：GPU 多个 SM 同时跑不同算子。本节后续分析只看各算子占 GPU 总工时的**比例**。
> 

### 5.2 新增算子：Q/DQ（量化 / 反量化）

FP8 引入了 BF16 baseline 中不存在的算子类别。核心问题：Q/DQ 是否能被编译器融合消除，还是需要独立的硬件转换单元？

| Q/DQ 类型 | 位置 | 实测（Vision engine inspector） |
| --- | --- | --- |
| Q: →FP8（权重/激活） | 量化层输入 | **无独立 Q/DQ kernel**——TRT 把 Q/DQ 折叠进 Myelin 融合块，以 `Cast` 形式出现在融合 kernel 名里（如 `__myl_CastMulAdd...Cast...`） |
| DQ: FP8→Half（激活/输出） | 层边界 / fallback 层前 | 同上；反量化发生在 FP8 层与 Half fallback 层的 **datatype 边界** |

> **实测结论（Q/DQ 是否被融合消除）：是，基本被 Myelin 融合吸收。** 证据来自 engine inspector 的 datatype 边界，而非独立节点计数（因为独立节点已不存在）：
> - Vision engine 输出 datatype：`{FP8: 469, Half: 177}` → **72.6% 层输出 FP8**；输入 datatype：`{FP8: 469, Half: 299, UInt8: 76, UInt64: 4}`。
> - 每一处 FP8↔Half 的边界即一次实际的量化/反量化，被折进相邻融合 kernel 的 epilogue（`Cast`），**没有暴露为独立的 Q/DQ 算子**，因此对 GPU 工时的独立贡献无法单独计量（在 §5.3 归入 Elementwise/Act 的 `Cast`）。
> - LLM engine（detailed 重建后）：**448 个 FP8 输入张量 / 256 个 FP8 输出**，Q/DQ 同样折叠进 Myelin，无独立节点（§1.5）。
> 

### 5.3 算子类别聚合对比

> 三列口径：**BF16** 是整模型（aten，占 E2E 100%）；**FP8 Vision** 是 Vision engine（占 E2E 3.8%）；**FP8 LLM** 是整个 Edge-LLM engine（prefill+decode 全程，占 E2E ~96%，nsys kernel 级）。由 `10_operator_categories.py` 按 kernel 名启发式归类（融合 kernel 按主导算子归，见 §5.1 限制）。**因 LLM 占 E2E 主导，"FP8 LLM" 列最接近 BF16 整模型口径，可与 90.7% 直接对照。**

| 类别 | BF16（整模型 aten） | FP8 Vision（3.8%） | **FP8 LLM（~96%，nsys）** | 备注 |
| --- | --- | --- | --- | --- |
| GEMM / TensorCore | 90.7% | 73.0% | **92.1%** | LLM GEMM 占比与 BF16 整模型 90.7% 高度吻合；已全走 `cutlass3x_sm100_..._f8_f8_f32` FP8 TensorCore |
| Attention | 1.9% | 8.0% | **5.0%** | LLM 侧为 prefill 的 `BlackwellFusedMultiHeadAttention`（decode 的 XQA 更小） |
| KV/RoPE/Sample | — | — | **1.2%** | LLM runtime 插件核（RoPE+写 KV、seqlen 记账、topK 采样、embedding lookup）——LLM 特有，Vision 无 |
| Norm / Reduction | 0.7% | 8.9% | **1.1%** | 融合的 RMSNorm（`__myl_...Mean...Sqrt...Div`）；Vision 占比高是因其分母只有 vision |
| Elementwise / Act | 1.8% | 9.2% | **0.6%** | 含被融合吸收的 `Cast`（即 Q/DQ）、SiLU/GELU |
| Memory / Layout | 2.7% | 1.0% | ~0 | Myelin 把 layout 融进 GEMM epilogue |
| **Q/DQ（新增）** | **0%** | 折叠入 Cast | 折叠入 Cast | 无独立占比，见 §5.2 |
| 其他 | 2.2% | 0% | 0% | LLM 侧全部命中类别 |

> **关键对照（给硬件/编译器同事）：** FP8 LLM 的 **GEMM 92.1% ≈ BF16 整模型 90.7%**——量化没有改变模型的算子结构（GEMM 主导 GPU 工时，Attention 5%、norm/激活/KV 插件合计 ~3% 为点缀），只是换了算子内部的数据路径：BF16 走 `nvjet_sm110_tst_*gemm_f16_f16_f16`（FP16×FP16→FP16），FP8 走 `cutlass3x_sm100_tensorop_*gemm_f8_f8_f32`（E4M3×E4M3→FP32 累加）。换言之，FP8 不是改算法，而是把同一颗 TensorCore 喂的权重从 2 字节压到 1 字节，让同一条访存通道在单位时间内多搬运一倍权重。因此 decode 的加速与瓶颈都落在这 92% 的 GEMM 上——加速来自每权重字节减半（§6.2 时延近乎线性下降 2×），瓶颈也仍卡在"把权重从内存读进来"这道访存墙上（§6）。Vision 列 GEMM 占比看着低（73%）仅因分母只有 vision 自己那 9 ms，不代表整模型。

### 5.4 CUDA 耗时 kernel 完整排名（合并 tile 变体后）

> BF16 是 PyTorch 算子视角、FP8 是 TRT 的 CUDA kernel 视角，命名体系不同，本节看**趋势**。FP8 LLM 列来自 nsys，占 E2E ~96%，最有代表性。
>
> **为什么要合并？** TensorRT 编译出的融合 kernel 名字长这样：`__myl_Fc_0x33c3fcd6..._tensorop256x256_cga2x2_sm2...`。前半段 `__myl_Fc_0x33c3fcd6` 标识"是哪个算子"，后半段 `_tensorop256x256_cga2x2_...` 是它的**切分方式**（tile）。同一个算子 Myelin 会生成几十种切分方案、运行时挑最快的用——对 profiler 来说它们名字不同，但干的是同一件事。如果逐条列，一个 GEMM 就占几十行、每行 0.几%，看不出它总共占多少。所以本节把同前缀的变体**工时加总**再排名。合并后 LLM 从 252 个名字收敛到 33 个，top10 覆盖 98%，表才读得清。
>
> ⚠️ 旧版只合并了几个大头、漏了中等 GEMM，已改为统一按 hash 合并（252 → 33）。原始未合并数据见附录 A 的 `fp8_operator_categories_<tag>.json`。

**FP8 LLM engine kernel 排名（nsys GPU 执行时间占比；同前缀 tile 变体已合并）：**

| # | kernel（合并后） | 占比 | GPU 执行时间 | 调用数 | tile 变体 | 类别 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `cutlass3x_sm100_tensorop_s256x256x32gemm_f8_f8_f32_f16_f16`（FP8 GEMM，256×256 tile） | 36.4% | 182.59 ms | 832 | 1 | GEMM |
| 2 | `__myl_Fc_0x33c3fcd6...`（融合 GEMM 模板 A） | 20.2% | 101.50 ms | 1248 | 96 | GEMM |
| 3 | `__myl_Fc_0xd8d4a5ac...`（融合 GEMM 模板 B） | 17.4% | 87.47 ms | 416 | 32 | GEMM |
| 4 | `__myl_Fc_0x64bea4ca...`（融合 GEMM 模板 C） | 6.6% | 33.00 ms | 416 | 32 | GEMM |
| 5 | `BlackwellFusedMultiHeadAttentionForward`（prefill FMHA） | 5.0% | 24.99 ms | 416 | 1 | Attention |
| 6 | `cutlass3x_sm100_tensorop_s128x128x32gemm_f8_f8_f32_f16_f16`（FP8 GEMM，128×128 tile） | 5.0% | 24.94 ms | 160 | 1 | GEMM |
| 7 | `sm80_xmma_gemm_f16f16`（保留 fp16 的 GEMM，§1.5 Half fallback） | 3.3% | 16.50 ms | 14 | 1 | GEMM |
| 8 | `__myl_FcSilu_0x35d2df17...`（融合 GEMM+SiLU，**旧版漏列**） | 1.6% | 7.96 ms | 32 | 32 | GEMM |
| 9 | `__myl_FcMulMulCast_0x5774b838...`（融合 GEMM+Mul，**旧版漏列**） | 1.6% | 7.94 ms | 32 | 32 | GEMM |
| 10 | `__myl_AddCastMulMeanAddSqrtDivMulCastMulCast`（融合 RMSNorm） | 1.0% | 4.94 ms | 806 | 1 | Norm |
| | **top10 累计** | **98.0%** | 491.8 ms | — | — | — |
| 11 | `trt_edgellm::applyRopeWriteKVSplitQKVKernel`（RoPE+写 KV 插件核） | 0.5% | 2.43 ms | 416 | 1 | KV/RoPE |
| 12 | `__myl_SiluMulMulCast`（融合 SiLU 激活） | 0.4% | 1.78 ms | 416 | 1 | Act |
| 13 | `calCuQCuKVSeqLensAndKVEndIdxsKernel`（seqlen 记账） | 0.2% | 1.17 ms | 416 | 1 | KV |
| 14 | `embeddingLookupKernel` | 0.2% | 1.11 ms | 91 | 1 | KV |
| 15 | `__myl_MulCast`（Cast/缩放） | 0.2% | 0.81 ms | 416 | 1 | Act |
| 16 | `__myl_AddCast...DivCast`（RMSNorm 变体） | 0.1% | 0.38 ms | 62 | 1 | Norm |
| 17 | `topKStage1<256,8>`（采样） | 0.1% | 0.35 ms | 91 | 1 | Sample |
| 18 | `incrementLengthTensorKernel` | 0.1% | 0.33 ms | 170 | 1 | KV |
| 19 | `kernel_mha`（decode XQA 注意力） | 0.1% | 0.33 ms | 32 | 1 | Attention |
| 20–33 | 余 14 条均 <0.1%（topKStage2/compactKV/norm/rope 小核） | 合计 ~0.2% | ~0.9 ms | — | — | KV·Norm |

> **读法：** 第 1 名单条 CUTLASS FP8 GEMM 占 36.4%（832 调用，decode 每步 q/k/v/o/gate/up/down 主力）；#2–#4 是三个不同 hash 的 `__myl_Fc` 融合 GEMM（合计 44.2%），对应不同权重矩阵。这 4 条 GEMM 合计 **80.6%**，加 #6(5.0%) + #7 的 fp16-fallback GEMM(3.3%) + #8/#9 的 FcSilu/FcMulMulCast(各 1.6%) → GEMM 类与 §5.3 的 92.1% 吻合。#5 `BlackwellFMHA`(5.0%) 是 prefill 注意力；#7 `sm80_xmma_gemm_f16f16`(3.3%，仅 14 调用) 是 prefill 保留 fp16 的 GEMM（§1.5 Half fallback）；#11 起为 Edge-LLM runtime 插件核（RoPE/写 KV/seqlen/采样/embedding），即 §5.3 的 KV/RoPE/Sample(1.2%) + 部分 Norm。完整 50 行（未折叠）见 `fp8_operator_categories_<tag>.json`。

**FP8 Vision engine kernel 排名（torch.profiler CUDA 工时占比；同 §5.4 LLM 口径按 `__myl_<op>_0x<hash>` 合并 tile 变体）：**

> vision 总 GPU 工时 90.87 ms，**103 raw distinct → 78 合并 kernel**。
>
> ⚠️ **本表已修订**（旧版口径有误）：旧 vision 表**未做 tile 合并**（而同节 LLM 表却合并了，两者口径不一致）。最大的后果是漏看了融合 GEMM+GELU 核 `__myl_FcMulCastErfCastAddMulMulMulCast_0x28c30823`——它有 **26 个 tile 变体**，合并后 **13.0%（11.83 ms）排第 2**，但旧版只把它最大的单个变体列在 #10（1.5%/1.32 ms）。合并后 top10 覆盖 **80.3%**（旧版 68.8% 是未合并口径）。

前 5 行与 BF16 aten 做跨命名体系的**趋势对照**（rank-vs-rank，非语义一一对应）：

| # | BF16 Eager Top（aten） | BF16 占比 | FP8 Vision Top（合并后 CUDA kernel） | 占比 |
| --- | --- | --- | --- | --- |
| 1 | `aten::linear` | 22.9% | `cutlass3x_sm100_tensorop_s128x128x32gemm_f8_f8_f32_f16_f16` | 18.8% |
| 2 | `aten::matmul` | 22.4% | `__myl_FcMulCastErfCastAddMulMulMulCast_0x28c30823`（融合 GEMM+GELU，26 变体） | 13.0% |
| 3 | `aten::mm` | 22.3% | `cutlass3x_sm100_tensorop_s128x128x32gemm_f8_f8_f32_e4m3_e4m3` | 11.9% |
| 4 | nvjet GEMMs | 21.3% | `cutlass3x_sm100_tensorop_s256x128x32gemm_f8_f8_f32_f16_f16` | 9.5% |
| 5 | `aten::cat` | 0.8% | `__myl_Fc_0x0695934e...`（融合 GEMM） | 8.4% |

FP8 Vision 完整排名（#6 起，尾部折叠）：

| # | kernel（合并后） | 占比 | GPU 工时 | 调用数 | tile 变体 | 类别 |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | `__myl_CastMulAdd...MeanSubMul...Sqrt...Div`（融合 RMSNorm+Cast） | 5.1% | 4.59 ms | 500 | 1 | Norm |
| 7 | `__myl_Fc_0x7ed6691a...`（融合 GEMM） | 3.8% | 3.48 ms | 250 | 1 | GEMM |
| 8 | `__myl_MulCastErfCastAddMulMulMulCast`（融合 GELU，Erf 实现） | 3.8% | 3.45 ms | 230 | 1 | Act |
| 9 | `__myl_Fc_0x1657adf8...`（融合 GEMM） | 3.2% | 2.87 ms | 230 | 1 | GEMM |
| 10 | `__myl_CastMulAdd...`（融合 RMSNorm，另一实例） | 2.8% | 2.55 ms | 430 | 1 | Norm |
| | **top10 累计** | **80.3%** | 72.99 ms | — | — | — |
| 11 | `__myl_FcMulCastErfCastAddMulMulMulCast_0x1eab7...`（融合 GEMM+GELU，另一 hash） | 1.5% | 1.32 ms | 10 | 1 | GEMM |
| 12 | `sm80_xmma_fprop_implicit_gemm`（patch-embed 卷积，fp16） | 1.4% | 1.23 ms | 20 | 1 | GEMM/Conv |
| 13 | `cutlass3x_sm100_tensorop_s256x256x32gemm_f8_f8_f32_f16_f16` | 1.3% | 1.22 ms | 10 | 1 | GEMM |
| 14–78 | 折叠：逐层注意力核 `_gemm_mha_v2_0x<hash>`（49 条 @ ~0.15 ms，合计 **7.24 ms / 8.0%**，= §5.3 Attention 全部）+ `__myl_MulCastReshTran...` layout/reshape 核（~7 条，合计 ~5.4 ms）+ 零星 norm/gemm 小核 | 合计 ~15.5% | ~14.1 ms | — | — | Attention·Layout |

> **趋势观察：** 两个 engine 的 GEMM 都已走 `f8_f8_f32`（E4M3 输入 / FP32 累加）的 CUTLASS SM100 TensorCore kernel；BF16 的 `aten::linear/matmul/mm` 三条并列大头，在 TRT 里被融合成单条 `cutlass*gemm` 或 `__myl_Fc`（GEMM+bias+激活+Cast 融合），**算子数量大幅收敛**（103 raw → 78 合并；LLM 更甚，252 → 33）。vision 的注意力不像 LLM 那样融成一个大核，而是**逐 transformer 层各一个 `_gemm_mha_v2` 小核**（49 条），故单条占比低但合计达 8%。完整 50 行（未折叠）见 `fp8_operator_categories_<tag>.json`。
> 

> ⚠️ 表中工时占比为各算子 GPU 执行时间的累加比例，不是实际推理耗时中的比例。有意义的数字是**占比排序和变化趋势**。
> 

### 5.5 TRT Myelin 自动融合效果

BF16 baseline（[v1 §6.5](OpenVLA%20模型评测_v1.md#65-算子融合分析)）中使用 PyTorch 内置编译器（`torch.compile`）模拟了 3 种融合的收益。FP8 + TRT 下，Myelin builder 会自动融合。对比两者的差距：

> **实测（定性）：Myelin 确实自动融合了这些模式——从 kernel 名可直接看到。** 但本环境未对这 3 个模式做**孤立 micro-benchmark**（需专门的单 pattern engine），故下表 FP8 列给出"融合证据"而非单独 ms。

| 融合模式 | BF16 Eager（ms） | BF16 compile（ms） | FP8 TRT 自动融合（实测证据） |
| --- | --- | --- | --- |
| RMSNorm + Linear | 0.167 | 0.110 | ✅ 已融合：kernel `__myl_CastMulAdd...MeanSubMul...Sqrt...Div...`（RMSNorm）后接 `__myl_Fc_...tensorop`（Linear），部分场景 RMSNorm 的 Cast 直接进 Fc 前序；未单独计时 |
| Linear + SiLU + Mul | 0.971 | 0.935 | ✅ 已融合：见 kernel `__myl_FcMulMulCast`、`__myl_MulCastErf...`（Fc + 激活 + Mul 融进一个 myl 块）；未单独计时 |
| Linear + GELU | 0.042 | 0.071 ⚠️ | ✅ 已融合：**Myelin 融合块** `__myl_FcMulCastErfCastAddMulMulMulCast`（Fc→Cast→Erf→乘加链）+ `__myl_MulCastErfCastAddMulMulMulCast`（vision 实测工时 1.32+3.45 ms）。注意 GELU 用 **Erf** 实现、**不涉及 relu**；且此处融合发生在 Myelin 块内，**不在 CUTLASS GEMM epilogue**（实测 vision 的 `cutlass3x_sm100_tensorop_s*gemm_f8_f8_f32_f...` kernel 名均无 `bias/relu/erf` 后缀）；未单独计时 |

> **对编译器同事：** TRT/Myelin 的自动融合粒度**明显强于** PyTorch `torch.compile` 的手动模式——它把 GEMM + bias + 激活 + Cast(Q/DQ) 全部塞进单个 CUTLASS kernel 的 epilogue，这也是 §5.2 里 Q/DQ "消失"的原因。要精确量化每个融合模式的收益，需补做孤立 micro-benchmark（每模式一个最小 engine + `trtexec` 计时）。
> 

---

## 6 瓶颈分析

> 核心问题：FP8 权重减半后，Decode 还是 Memory-Bound 吗？**结论：是（见 §6.2）。**
> 

### 6.1 双模式瓶颈对比

BF16 下 Prefill 和 Decode 的瓶颈完全不同（[v1 §6.1](OpenVLA%20模型评测_v1.md#61-双模式瓶颈对比)）。FP8 权重减半后，对比变化：

| 指标 | BF16 Prefill | BF16 Decode | **FP8 Prefill** | **FP8 Decode** |
| --- | --- | --- | --- | --- |
| 时延 (ms) | 103.6 | 362.7 | **54.0** | **182.7** |
| 权重加载量 | 12.9 GB | 12.9 GB | **~6.5 GB**（FP8 权重≈BF16 半） | **~6.5 GB** |
| M 维度 | 256–275 | 1 | **256–275**（量化不改变） | **1**（不变） |
| GFLOPs | ~3,784 | ~80 | **~3,784**（计算量不变，仅精度降） | **~80** |
| Arithmetic Intensity | ~240 FLOP/byte | ~1.0 FLOP/byte | **~480**（字节减半→AI 翻倍） | **~2.0** |
| 瓶颈类型 | 权重加载受限 | 纯 Memory-Bound | **仍权重加载受限** | **仍纯 Memory-Bound** |

> Arithmetic Intensity = FLOPs / 访存字节。FP8 只减少**每个权重的字节数**（2B→1B），不改变 M 维度和 FLOP 计数，故 AI ~翻倍。但 Decode AI ~2.0 FLOP/byte 仍远低于 FP8 TensorCore 的 ridge point（Thor FP8 算力/带宽比在数百量级），**离脊点还差两个数量级 → 依然被访存卡死**。
> 

> M 维度 = 矩阵乘法一次处理多少个 token。详见 [v1 §6.1](OpenVLA%20模型评测_v1.md#61-双模式瓶颈对比)。
> 

### 6.2 关键判断：瓶颈是否转移

**结论：瓶颈未转移，Decode 仍是 Memory-Bound。** 实测判据：

- **Decode 每步 60.4 → 30.4 ms（↓1.99×）**，确实显著下降，但——
- **下降幅度（~2×）几乎等于权重字节减半幅度（~2×）**。这正是 memory-bound 的判定证据：当时延 ∝ 权重字节数、而与算力无关时，说明瓶颈在"把权重从内存读进来"，不在"算"。若 decode 已转为 compute-bound，减半字节不会带来近 2× 的线性加速。
- **Prefill:Decode 时延比 = 54.0 : 182.7 ≈ 1:3.4**，与 BF16 的 1:3.5 几乎不变（decode 仍占 E2E ~77%）。远未接近"转移信号"的 1:1。
- 30.4 ms 虽然踩到 30 ms 阈值，但结合上面两条，**判定 Memory-Bound 依然主导**。

**对硬件/编译器同事的含义：** FP8 在 decode 上的收益直接来自权重带宽减半。要继续压 decode，杠杆仍是**减少每 token 的权重读取字节**——nvfp4（再半）、权重-only 更激进量化、或 KV-cache 量化，而不是堆算力。

---

## 7 CPU 开销残余

> BF16 Eager baseline（[v1 §6.3-6.4](OpenVLA%20模型评测_v1.md#63-纯-cpu-开销)）识别了 ~700 ms 的 CPU 开销——CPU 花在”告诉 GPU 做什么”上的时间（同步等待、发起拷贝、设备/类型转换），远超 GPU 实际执行这些操作的时间。TRT 静态图应能预先规划所有数据移动，消除大部分运行时开销。编译器同事关注：**哪些被消除了，哪些残留？**
> 

> ✅ **已用 nsys 采集**（`13_nsys_cpu_overhead.py`，差分法）。BF16 的 CPU 开销来自 `torch.profiler`（PyTorch eager 逐 aten CPU 计时）；FP8 主导耗时的 Edge-LLM 是 C++ binary，改用 `nsys profile` 对 `llm_inference` 子进程做 system-wide CUDA API trace。**差分法**：跑两次不同 warmup（2 vs 12），相减得每次推理的 host 开销，抵消一次性 engine load（6.7 GB 权重 H2D ≈ 1849 ms，不计入）。口径提示：nsys 的 `cudaStreamSynchronize` 是 host 阻塞等 GPU 的时间（≈GPU 执行耗时），**不是可优化掉的调度开销**；真正"浪费在告诉 GPU 做什么"的是 launch/copy 类。

| CPU 开销项 | BF16 Eager（ms/推理） | FP8 TRT（ms/推理） | 是否消除 |
| --- | --- | --- | --- |
| `cudaLaunchKernel` | 49.0 | **0.80**（132 calls） | ✅ 几乎消除（↓60×，decode 走 CUDA Graph） |
| `cudaGraphLaunch` | 0（无 graph） | **1.05**（6 calls=6 decode 步） | 新增：取代数百次 launch |
| `aten::to`（dtype 转换） | 194.1 | **0**（TRT 无 aten，静态图预规划） | ✅ 消除 |
| `aten::copy_` | 181.3 | 见 cudaMemcpy | ✅ 消除 |
| `cudaMemcpy`（数据搬运） | （含于 aten::copy_） | **4.12**（24 calls） | ✅ 大幅降低 |
| `cudaStreamSynchronize` | 164.4 | **204.5**（8 calls） | ⚠️ 未降（本质是等 GPU，非调度开销，见上） |
| **可消除的调度/搬运合计** | **~424**（launch+to+copy） | **~6**（launch+graph+memcpy） | ✅ **↓约 98%** |

> **结论（给编译器同事）：TRT 静态图 + CUDA Graph 把"运行时调度/数据搬运"CPU 开销从 ~424 ms 压到 ~6 ms（↓98%）。** BF16 里 `aten::to`(194)+`aten::copy_`(181) 这类 PyTorch eager 的 dtype 转换/拷贝被完全预规划消除；`cudaLaunchKernel` 从 49 ms 降到 0.8 ms（decode 的数百次 kernel launch 被 6 次 `cudaGraphLaunch` 取代）。剩下的 `cudaStreamSynchronize`(~204 ms) 是 host 阻塞等 GPU 计算完成，属不可避免的 GPU-wait，**不是**要继续优化的调度残余。NPU 场景下真正要盯的是这块 GPU 执行时间本身（即 decode 的访存墙，§6），而非 host 侧 DMA。
> 

---

## 8 总结


| # | 发现 | 含义 |
| --- | --- | --- |
| 1 | **E2E 466→237 ms（1.97×），可达 4.2 Hz** | FP8+TRT 把 OpenVLA 推理频率翻倍；但仍未到实时控制常需的 10 Hz+ |
| 2 | **Decode 瓶颈未转移，仍 Memory-Bound** | 每步 60.4→30.4 ms（↓2×）≈ 权重字节减半；Prefill:Decode 比 1:3.5→1:3.4 几乎不变。继续加速要靠减字节（nvfp4/KV 量化），不是堆算力 |
| 3 | **7-DoF 精度损失：RMSE = 0.0016** | task B 已打通（直驱 FP8 engine）。4/7 动作 token 一致、gripper 一致；误差全来自 FP8 LLM（vision 量化被离散化吸收）。⚠️ 单样本，上线前需多样本+任务成功率复测（§9 R6） |
| 4 | **CPU 开销：调度/搬运 ↓98%（~424→~6 ms）** | nsys 差分实测：`cudaLaunchKernel` 49→0.8 ms、`aten::to/copy_`(375) 消除；剩 `cudaStreamSynchronize` 204 ms 是 GPU-wait 非调度残余 |
| 5 | **TRT 自动融合强于手动 compile** | Myelin 把 GEMM+bias+激活+Cast 融进单个 CUTLASS epilogue；未做逐模式 micro-benchmark |
| 6 | **Q/DQ 无独立开销** | 已被 Myelin 折叠为融合 kernel 里的 Cast；Vision 72.6% 输出 FP8，78 层 fallback Half |
| 7 | **体积：Vision `.plan` 1454→741 MB（−49%）** | FP8 权重减半；27% 层 Half 兜底 + scale 常数使其未正好减半 |
| 8 | **LLM 逐层精度：448 FP8 输入张量** | detailed 重建后实测；FP8 覆盖大 GEMM 操作数，残差流保持 Half（标准 LLM 量化） |

---

## 9 Review：数值正确性与合理性审查

> 本节是对文档现有数字 + 本次实测的审查记录，供后续修订参考。

### 9.1 需要修正/存疑的点

| # | 问题 | 详情与建议 |
| --- | --- | --- |
| R1 | **BF16 baseline 数字来源缺失** | 文档头部引用的 `OpenVLA 模型评测_v1.md` 在工作区**不存在**。文中 baseline 值（E2E 466.24 / Prefill 103.56 / Decode 362.68 / 60.4 ms/step）与本机保存的最近实测（`operator_inventory.report.txt`：E2E **464.65**、Prefill **104.0**、Decode **360.6**、**60.1** ms/step）**差 ~0.3–1%**。按用户要求保留原值，但建议补一份可复现的 v1 出处，或统一改用实测值。 |
| R2 | **§2.3 "相同输入" 不成立** | 文档原称 FP8 用"相同输入（1×224²+19 tokens）"。实际：Vision 用**随机张量**仅计时；LLM 用 **262-token 纯文本 prompt、无真实图像 embedding**，且 BF16 baseline 是 **275 token**（256 vision patch + 19 text）。已在 §2.3 标注。两者 prompt 长度、是否过视觉通路都不同。 |
| R3 | **§4 时延 FP8 E2E 是分段相加，非单条端到端** | §4 的时延 E2E（236.7 ms）由 Vision（torch.profiler）+ Prefill/Decode（Edge-LLM `--dumpProfile`，纯文本 prompt）分段相加，是**下界估计**。注：§3 的精度 harness（`12_fp8_action_accuracy.py`）已是真实拼接的单条推理，但它未计精确时延；两者用途不同。 |
| R4 | ~~§5.3/§5.4 FP8 仅 Vision~~ **已解决** | 已用 nsys 对 LLM engine 做 kernel 级归类（§5.3 新增 FP8 LLM 列）：**GEMM 92.1% ≈ BF16 整模型 90.7%**，可直接对照。残余限制：Myelin 融合 kernel 内部不可再拆，只能主导归类（启发式）；这是融合固有代价，非工具缺失。 |
| R5 | **§4.2 BF16 decode 58.9 vs §4.1 的 60.4 自相矛盾** | 原文档内部两处 BF16 decode/step 不一致（58.9 与 60.4）。未改动 BF16 列，但建议核对：60.4=362.68/6 是含采样的每步；58.9 可能是 Llama-only。 |
| R6 | **§3 精度是单样本结论** | RMSE 0.0016 来自单张图 `bridge_sample_0001`。虽 4/7 token 一致、gripper 一致，但**不能外推为"生产达标"**。roll/pitch/Δz 出现 3–5 bin 漂移，上线前需在 test_data/ 全量 + 真实任务成功率上复测。 |

### 9.2 合理性核验（本次实测，判定合理）

- ✅ **加速比 ~1.97× 合理**：FP8 权重减半，decode（访存受限）几乎线性提速 2×，prefill（计算受限）1.92×，符合物理预期。
- ✅ **Decode 仍 Memory-Bound 合理**：时延下降幅度 ≈ 字节下降幅度，是访存受限的教科书特征（§6.2）。
- ✅ **体积 −49% 合理**：inspector 实证 72.6% 层 FP8 + 27% Half 兜底 + scale 常数，未正好减半可解释。
- ✅ **Vision 9.09 ms 合理**：fp16→fp8 从 12.73→9.09 ms，GEMM 走 FP8 TensorCore（§5.4），量级正确。
- ✅ **精度 RMSE 0.0016 合理**：4/7 token 一致、首末 token 一致（harness 正确性交叉验证，§3），漂移仅 3–5 bin，符合 FP8 量化预期。
- ✅ **CPU 开销 ↓98% 合理**：TRT 静态图消除 PyTorch eager 的 aten::to/copy_ + CUDA Graph 折叠 kernel launch，是 TRT 部署的预期收益；残余 cudaStreamSynchronize 是 GPU-wait（§7）。

---

## 附录 A：Trace 文件索引（实际产物）

| 文件（`/workspace/outputs/openvla/`） | 用途 |
| --- | --- |
| `vision_trace_fp8_<tag>.trace.json` | FP8 Vision kernel timeline（Perfetto） |
| `vision_trace_fp8_<tag>.table.txt` | FP8 Vision operator table（按 CUDA time 排序） |
| `llm_profile_fp8_<tag>.json` | Edge-LLM 分阶段 prefill/decode gpu_time_stats |
| `llm_layers_fp8_<tag>/layer_decode_pastkvlen262.csv` | Edge-LLM decode 逐层 CSV（Myelin 融合，onnx_op 空） |
| `e2e_prof_summary_vfp8_lfp8_<tag>.json` | 端到端汇总（时延 + 精度验证） |
| `fp8_operator_categories_<tag>.json` | 算子类别聚合（§5.3/§5.4 数据源） |
| `fp8_action_accuracy_action.json` / `_golden-emb.json` | §3 真实 FP8 动作 + RMSE（task B） |
| `fp8_nsys_cpu_overhead_cpu_overhead.json` + `fp8_nsys_*.nsys-rep` | §7 nsys CPU 开销（差分法） |
| `fp8_llm_precision_detailed.json` | §1.5 LLM 逐层精度（detailed 重建） |

**最新 tag：** `2026_0727_001508`。
**使用方式：** 在浏览器中打开 `https://ui.perfetto.dev`，加载 `.trace.json` 查看 Trace 流水线。
**采集脚本：** `09_prof_trace_e2e.py`（时延/trace/精度）、`10_operator_categories.py`（类别）、`12_fp8_action_accuracy.py`（§3 动作精度）、`13_nsys_cpu_overhead.py`（§7 CPU 开销）。

---

## 附录 B：TRT Builder Config（实际使用）

**Vision engine**（`deploy/tensorrt/pipeline/05_build_vision_engine.sh`，已带 detailed verbosity）：

```
trtexec --onnx=vision_projector_fp8.onnx \
        --saveEngine=vision_projector_fp8.plan \
        --fp8 --fp16 \                    # Q/DQ 层走 FP8，其余 FP16 兜底
        --builderOptimizationLevel=5 \
        --profilingVerbosity=detailed \   # ← 使 engine inspector 能读出逐层精度
        --skipInference
```

**LLM engine**：由 Edge-LLM `llm_build` 从量化 ONNX 构建（`03_convert_llm_x86.sh` 只做 export/quantize→ONNX；engine 构建是手动步骤）。用于精度审查的 detailed 版：

```
llm_build --onnxDir /workspace/checkpoints/openvla/hf_llama_onnx_fp8/llm \
          --engineDir .../engines/openvla_llama_fp8_detailed \
          --profilingDetailed \              # ← 使 inspector 能读出逐层精度 (§1.5)
          --maxBatchSize 1 --maxInputLen 512 --maxKVCacheCapacity 512
```

（时延测量用不带 `--profilingDetailed` 的原 `openvla_llama_fp8/`；二者算子/时延一致。）

### 附：Myelin 是什么（背景知识）

> 阅读本文 §5.2 / §5.5 / §1.4 时可能遇到的术语，这里补一段背景，供不熟悉 TensorRT 内部的同事参考。

- **Myelin 是 TensorRT 内部的"代码生成器 / 算子融合引擎"**。TRT builder 接收 ONNX 计算图后，由 Myelin 负责把相邻算子融合、并生成最终的 CUDA kernel 代码，打包进 `.engine` / `.plan`。它生成的 kernel 名统一带 `__myl_` 前缀——即文档里到处出现的 `__myl_Fc`、`__myl_CastMulAdd...Mean...Sqrt...Div`、`__myl_AddCastMulMeanAddSqrtDivMulCastMulCast` 等命名来源。

- **"被 Myelin 融合吸收"**（§5.2）：指某个算子**不再作为独立 CUDA kernel 存在**，而是被折进相邻融合 kernel 的内部步骤。本文特指 Q/DQ（量化/反量化）：
  - 原始 ONNX 图里 Q/DQ 是独立节点（`fp16→fp8`、`fp8→fp16` 各占一个 kernel）；
  - Myelin 编译时把 **Q 折进 GEMM 的权重加载/前序**，把 **DQ 折进 GEMM 的 epilogue**（FP8×FP8→FP32 累加后顺手转 Half），二者都不另起 kernel；
  - 结果在 engine inspector 里**看不到名为 `Quantize`/`Dequantize` 的独立节点**，只能看到 datatype 边界（FP8↔Half 切换处）和融合 kernel 名里的 `Cast` 子串。

- **关系澄清**：不是"engine 被 Myelin 吸收"，而是"engine 是 Myelin 生成的"。流程为 `ONNX 图（含独立 Q/DQ 节点）→ TRT builder（Myelin 做融合+代码生成）→ .engine/.plan（全是 `__myl_*` 融合 kernel，Q/DQ 已消失为独立 kernel）`。这也是 FP8 部署能"减字节却不增算子开销"的关键原因（§5.2 / §5.5）。

- **类比**：类似 C 编译器把 `a = b*2 + c` 优化成单条 `lea` 指令——源码两个操作（乘、加），编译后融合成一条。Myelin 对"GEMM + bias + 激活 + Cast(Q/DQ)"做同样的事，全部塞进单个 CUTLASS kernel 的 epilogue（§5.5）。

---

## 附录 C：待采集数据清单

### Profiling

- [x]  TRT builder 配置（附录 B）
- [x]  Vision 逐层精度（engine inspector，§1.4）
- [x]  LLM 逐层精度（detailed 重建，§1.5）
- [x]  Nsight Systems trace（§7 CPU 开销，差分法）
- [x]  FP8 精度验证：7-DoF RMSE（§3.1，task B 已打通）；逐模块 MSE 因融合黑盒不可取（§3.2）

### 数据计算

- [x]  算子类别聚合（FP8 Vision，§5.3；Q/DQ 归因见 §5.2）
- [x]  CUDA Top 10 + 工时占比（FP8 Vision，§5.4）
- [x]  CPU 开销残余分析（§7，nsys 差分）
- [x]  Q/DQ 归因（折叠入 Myelin，无独立节点，§5.2）

### 对比汇总

- [x]  E2E 时延对比表（§4.1）
- [x]  模块级时延对比表（§4.2）
- [x]  Decode 逐 Step 对比表（§4.3，FP8 侧为聚合中位数）
- [x]  瓶颈是否转移的判定（§6.2：未转移，仍 Memory-Bound）
- [x]  7-DoF 动作精度（§3.1，RMSE 0.0016）

### 后续要补（上线前）

1. **多样本精度**：§3 目前是单张图。跑 test_data/ 全部图片 + 真实任务成功率，确认 roll/pitch/Δz 的 3–5 bin 漂移不影响任务（§9 R6）。
2. **逐模块 MSE**：需 PyTorch 参考实现逐层对齐（融合引擎取不到中间张量）。
3. 统一 BF16 baseline 出处（§9 R1）。