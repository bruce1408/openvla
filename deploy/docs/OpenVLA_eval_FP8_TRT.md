# OpenVLA 模型评测_FP8_TRT_骨架

# OpenVLA 模型评测 — FP8 量化 + TensorRT 生产部署分析

> **📌 一句话结论（待采集后填入）：** OpenVLA FP8 + TensorRT 在 Thor 上的 E2E 推理时延为 `[待采集]` ms，相比 BF16 Eager baseline（466 ms）加速 `[待计算]`×。Decode 瓶颈是否从 Memory-Bound 转移：`[待判断]`。量化引入的精度损失：7-DoF 动作 RMSE = `[待采集]`。
> 
> 
> **文档定位：** [OpenVLA 模型评测_v1.md](OpenVLA%20模型评测_v1.md)（BF16 Eager baseline）的 FP8 生产部署姊妹篇。目标读者同为硬件和编译器同事——核心问题从”模型长什么样、瓶颈在哪”转变为”**量化后算子行为怎么变、瓶颈是否转移、硬件和编译器需要怎么适配**”。
> 

> **版本：** v1-skeleton | **依赖：** [BF16 baseline v1](OpenVLA%20模型评测_v1.md) | **状态：** 骨架，待 FP8 profiling 后填入数据 | **完整版产出：** `OpenVLA 模型评测_FP8_TRT.md`
> 

---

## 1 量化方案

> `[待采集]` 以下为推荐方案，实际选择需与算法团队确认。
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

| 模块 | 量化策略 | 原因 |
| --- | --- | --- |
| Vision Backbone | FP8 PTQ | 占 E2E 仅 3.4%，但统一格式减少 dtype 转换 |
| MLP Projector | FP8 PTQ | 权重仅 10 MB，量化开销忽略不计 |
| Llama Decoder 32 层 | **FP8 PTQ** | **收益核心**：待定 |
| → lm_head | FP8 PTQ（per-channel） | 单块最大（待定），精度敏感 |
| Embedding | FP8 PTQ 或保留 BF16 | 仅 Prefill 用一次，保留 BF16 可避免精度损失 |
| KV-cache | FP8（可选） | 待定 MB → 待定 MB，非瓶颈，视 TRT-Edge-LLM 支持而定 |

### 1.3 精度验证计划

| 指标 | 测量方法 |
| --- | --- |
| 7-DoF 动作 RMSE | 相同输入（图片+指令），对比 BF16 vs FP8 输出 |
| 各模块输出 MSE | 对 Vision / Projector / 每层 Decoder / lm_head 注册 hook |
| 任务成功率 | `[N]` 个典型操作任务 |
| 注意力分布 KL 散度 | 检查 attention weights 是否因量化偏移 |

---

## 2 测试环境

### 2.1 硬件环境

同 [baseline §1.1](OpenVLA%20模型评测_v1.md#11-硬件环境配置)：NVIDIA Jetson AGX Thor，SM110，20 SMs，LPDDR5X 122.86 GB。

### 2.2 软件环境

| 软件 | 版本 | 备注 |
| --- | --- | --- |
| TensorRT | `[待确认]` | 需 ≥ 10.x 以支持 SM110 FP8 TensorCore |
| TensorRT-Edge-LLM | `[待确认]` | Llama Decoder 的 FP8 plugin |
| PyTorch（导出用） | 2.12.0 alpha | 导出 ONNX / torch-TRT |
| CUDA | 13.2 | 不变 |
| cuBLASLt | `[待确认]` | 需支持 FP8 matmul |

### 2.3 Profiling 配置

| 项目 | BF16 baseline | FP8 本实验 |
| --- | --- | --- |
| 工具 | `torch.profiler` | TRT `trtexec` + `nsys` |
| 输入 | 1 × 224² + 19 tokens → 6 步 | **相同输入** |
| 轮数 | 10 轮取均值 | **10 轮** |
| 精度 | BF16 | **FP8 E4M3** |
| Trace | 3 个 chrome trace | **3 个（prefill / decode / full）** |
| 额外采集 | — | TRT layer-wise timing（`trtexec --dumpProfile`） |

---

## 3 精度分析

> `[待采集]` 确定 FP8 量化引入的误差是否在可接受范围。
> 

### 3.1 7-DoF 动作输出对比

| 动作维度 | BF16 输出 | FP8 输出 | 绝对误差 | 相对误差 |
| --- | --- | --- | --- | --- |
| Δx | `[待采集]` | `[待采集]` | `[待采集]` | `[待采集]` |
| Δy | `[待采集]` | `[待采集]` | `[待采集]` | `[待采集]` |
| Δz | `[待采集]` | `[待采集]` | `[待采集]` | `[待采集]` |
| roll | `[待采集]` | `[待采集]` | `[待采集]` | `[待采集]` |
| pitch | `[待采集]` | `[待采集]` | `[待采集]` | `[待采集]` |
| yaw | `[待采集]` | `[待采集]` | `[待采集]` | `[待采集]` |
| gripper | `[待采集]` | `[待采集]` | `[待采集]` | `[待采集]` |
| **RMSE** | — | — | `[待采集]` | — |

### 3.2 逐模块量化误差

| 模块 | 输出 MSE（vs BF16） | 是否需要混合精度 |
| --- | --- | --- |
| Vision Backbone（fusion 后） | `[待采集]` | `[待判断]` |
| MLP Projector | `[待采集]` | `[待判断]` |
| Llama Layer 0（early） | `[待采集]` | `[待判断]` |
| Llama Layer 15（middle） | `[待采集]` | `[待判断]` |
| Llama Layer 31（late） | `[待采集]` | `[待判断]` |
| lm_head | `[待采集]` | `[待判断]` |
| Attention weights（avg） | `[待采集]` | `[待判断]` |

### 3.3 精度判定

```
[待采集]：
- FP8 精度是否满足生产要求？ [是/否/需混合精度]
- 哪些层需要保留 BF16？（候选：lm_head / attention softmax / 首尾层）
- 混合精度的额外成本：Q/DQ 节点数量 + 延迟开销
```

---

## 4 推理性能对比

> `[待采集]` 所有数据与 BF16 baseline（[v1 §4](OpenVLA%20模型评测_v1.md#4-推理性能分析)）逐项对比。
> 

### 4.1 E2E 时延

| 阶段 | BF16 Eager（ms） | FP8 TRT（ms） | 加速比 |
| --- | --- | --- | --- |
| **Prefill** | 103.56 | `[待采集]` | `[待计算]` |
| **Decode（6 steps）** | 362.68 | `[待采集]` | `[待计算]` |
| **E2E** | 466.24 | `[待采集]` | `[待计算]` |
| **可达频率** | 2.14 Hz | `[待计算]` | — |

> 时延稳定性（10 轮波动范围）：`[待采集]`
> 

### 4.2 模块级时延

| 模块 | BF16 Eager（ms） | FP8 TRT（ms） | 变化 |
| --- | --- | --- | --- |
| Vision Backbone | 15.44 | `[待采集]` | — |
| Projector | 0.72 | `[待采集]` | — |
| Llama — Prefill | 86.30 | `[待采集]` | — |
| Llama — Decode（per step） | 58.9 | `[待采集]` | — |
| lm_head | 1.10 | `[待采集]` | — |

### 4.3 阶段拆解

#### Prefill 阶段

| 子阶段 | BF16（ms） | FP8 TRT（ms） | 变化 |
| --- | --- | --- | --- |
| ① 视觉编码 | 15.44 | `[待采集]` | — |
| ② MLP Projector | 0.72 | `[待采集]` | — |
| ③ Llama ×32 层 | 86.30 | `[待采集]` | — |
| ④ lm_head | 1.10 | `[待采集]` | — |
| **Prefill 合计** | **103.56** | `[待采集]` | — |

#### Decode 阶段（6 步自回归）

| Step | Q-Len | BF16（ms） | FP8 TRT（ms） |
| --- | --- | --- | --- |
| 1 | 1 | 63.2 | `[待采集]` |
| 2 | 1 | 60.0 | `[待采集]` |
| 3 | 1 | 59.9 | `[待采集]` |
| 4 | 1 | 59.9 | `[待采集]` |
| 5 | 1 | 59.8 | `[待采集]` |
| 6 | 1 | 59.9 | `[待采集]` |
| **均值** | — | **60.4** | `[待采集]` |

---

## 5 算子行为变化

> `[待采集]` FP8 下算子行为发生了什么根本变化——这是硬件和编译器同事最关心的章节。
> 

### 5.1 算子总览对比

| 指标 | BF16 Eager | FP8 TRT | 变化 |
| --- | --- | --- | --- |
| Profiler 记录 | 177 条 | `[待采集]` | 待确认 |
| 总调用次数 | 74,261 | `[待采集]` | 待确认 |
| CUDA 耗时（各算子之和） | 1,745 ms | `[待采集]` | 待确认 |
| CPU 耗时（各算子之和） | 1,211 ms | `[待采集]` | 待确认 |
| **E2E 实际耗时** | **466 ms** | `[待采集]` | 待确认 |

> 各算子之和 > 实际耗时的原因：GPU 多个 SM 同时跑不同算子。本节后续分析只看各算子占 GPU 总工时的**比例**。
> 

### 5.2 新增算子：Q/DQ（量化 / 反量化）

FP8 引入了 BF16 baseline 中不存在的算子类别。核心问题：Q/DQ 是否能被编译器融合消除，还是需要独立的硬件转换单元？

| Q/DQ 类型 | 位置 |
| --- | --- |
| Q: BF16→FP8（权重） | 权重加载后 |
| Q: BF16→FP8（激活） | 每层输入 |
| DQ: FP8→BF16（激活） | 每层输出 |
| DQ: FP8→BF16（最终输出） | lm_head 后 |

> `[待采集]` 实测 Q/DQ 节点数量、开销、以及 TRT 实际融合了哪些。
> 

### 5.3 算子类别聚合对比

| 类别 | BF16 CUDA 工时占比 | FP8 TRT CUDA 工时占比 | 占比变化 |
| --- | --- | --- | --- |
| GEMM / TensorCore | 90.7% | `[待采集]` | `[待计算]` |
| Memory / Layout | 2.7% | `[待采集]` | `[待计算]` |
| Attention | 1.9% | `[待采集]` | `[待计算]` |
| Elementwise | 1.8% | `[待采集]` | `[待计算]` |
| Norm / Reduction | 0.7% | `[待采集]` | `[待计算]` |
| **Q/DQ（新增）** | **0%** | `[待采集]` | **新增类别** |
| 其他 | 2.2% | `[待采集]` | `[待计算]` |

### 5.4 CUDA 耗时 Top 10 对比

| # | BF16 Eager Top 10 | BF16 工时占比 | FP8 TRT Top 10 | FP8 工时占比 |
| --- | --- | --- | --- | --- |
| 1 | `aten::linear` | 22.9% | `[待采集]` | `[待采集]` |
| 2 | `aten::matmul` | 22.4% | `[待采集]` | `[待采集]` |
| 3 | `aten::mm` | 22.3% | `[待采集]` | `[待采集]` |
| 4–8 | nvjet GEMMs | 21.3% | `[待采集]` | `[待采集]` |
| 9 | `aten::cat` | 0.8% | `[待采集]` | `[待采集]` |
| 10 | `aten::mul` | 0.7% | `[待采集]` | `[待采集]` |

> ⚠️ 表中工时占比为各算子 GPU 执行时间的累加比例，不是实际推理耗时中的比例。有意义的数字是**占比排序和变化趋势**。
> 

### 5.5 TRT Myelin 自动融合效果

BF16 baseline（[v1 §6.5](OpenVLA%20模型评测_v1.md#65-算子融合分析)）中使用 PyTorch 内置编译器（`torch.compile`）模拟了 3 种融合的收益。FP8 + TRT 下，Myelin builder 会自动融合。对比两者的差距：

| 融合模式 | BF16 Eager（ms） | BF16 compile（ms） | FP8 TRT 自动融合（ms） |
| --- | --- | --- | --- |
| RMSNorm + Linear | 0.167 | 0.110 | `[待采集]` |
| Linear + SiLU + Mul | 0.971 | 0.935 | `[待采集]` |
| Linear + GELU | 0.042 | 0.071 ⚠️ | `[待采集]` |

---

## 6 瓶颈分析

> `[待采集]` 核心问题：FP8 权重减半后，Decode 还是 Memory-Bound 吗？
> 

### 6.1 双模式瓶颈对比

BF16 下 Prefill 和 Decode 的瓶颈完全不同（[v1 §6.1](OpenVLA%20模型评测_v1.md#61-双模式瓶颈对比)）。FP8 权重减半后，对比变化：

| 指标 | BF16 Prefill | BF16 Decode | **FP8 Prefill** | **FP8 Decode** |
| --- | --- | --- | --- | --- |
| 时延 (ms) | 103.6 | 362.7 | `[待采集]` | `[待采集]` |
| 权重加载量 | 12.9 GB | 12.9 GB | `[待采集]` | `[待采集]` |
| M 维度 | 256–275 | 1 | `[待采集]` | `[待采集]` |
| GFLOPs | ~3,784 | ~80 | `[待采集]` | `[待采集]` |
| Arithmetic Intensity | ~240 FLOP/byte | ~1.0 FLOP/byte | `[待采集]` | `[待采集]` |
| 瓶颈类型 | 权重加载受限 | 纯 Memory-Bound | `[待判断]` | `[待判断]` |

> M 维度 = 矩阵乘法一次处理多少个 token。详见 [v1 §6.1](OpenVLA%20模型评测_v1.md#61-双模式瓶颈对比)。
> 

### 6.2 关键判断：瓶颈是否转移

`[待采集]` 用以下实测数据判断：

- Decode 每步时延是否从 60.4 ms 显著下降？`[待采集]`
- 如果 Decode 每步降至 < 30 ms，且 Prefill/Decode 时延比从 1:3.5 变为接近 1:1，则瓶颈已从 Memory-Bound 转移。
- 如果 Decode 每步仍在 50 ms 以上，则 Memory-Bound 依然主导。

---

## 7 CPU 开销残余

> `[待采集]` BF16 Eager baseline（[v1 §6.3-6.4](OpenVLA%20模型评测_v1.md#63-纯-cpu-开销)）识别了 ~700 ms 的 CPU 开销——CPU 花在”告诉 GPU 做什么”上的时间（同步等待、发起拷贝、设备/类型转换），远超 GPU 实际执行这些操作的时间。TRT 静态图应能预先规划所有数据移动，消除大部分运行时开销。编译器同事关注：**哪些被消除了，哪些残留？**
> 

| CPU 开销项 | BF16 Eager（ms） | FP8 TRT（ms） | 是否消除 |
| --- | --- | --- | --- |
| `cudaStreamSynchronize` | 164.4 | `[待采集]` | `[待判断]` |
| `cudaLaunchKernel` | 49.0 | `[待采集]` | `[待判断]` |
| `aten::to` | 194.1 | `[待采集]` | `[待判断]` |
| `aten::copy_` | 181.3 | `[待采集]` | `[待判断]` |
| `aten::transpose` / `aten::t` | 18.3 | `[待采集]` | `[待判断]` |
| 其他 | ~124 | `[待采集]` | `[待判断]` |
| **合计** | **~731** | `[待采集]` | — |

> `[待采集]` TRT 图优化后残余的 CPU 开销主要来自哪些算子？NPU 场景下是否需要 DMA 继续优化？
> 

---

## 8 总结

> `[待采集]` 以下为骨架，待 FP8 实测后填入。
> 

| # | 发现 | 含义 |
| --- | --- | --- |
| 1 | GEMM 占比变化：`[待采集]` | `[待判断]` |
| 2 | Decode 瓶颈是否转移：`[待判断]` | `[待判断]` |
| 3 | 7-DoF 精度损失：RMSE = `[待采集]` | `[待判断]` |
| 4 | CPU 开销残余：`[待采集]` ms | `[待判断]` |
| 5 | TRT 自动融合 vs 手动 compile 差距：`[待采集]` | `[待判断]` |
| 6 | Q/DQ 节点开销：`[待采集]`（如未融合） | `[待判断]` |

---

## 附录 A：Trace 文件索引

| 文件 | 用途 |
| --- | --- |
| `fp8_trt_prefill_trace.json` | FP8 TRT Prefill kernel timeline |
| `fp8_trt_decode_trace.json` | FP8 TRT Decode kernel timeline |
| `fp8_trt_full_trace.json` | FP8 TRT 完整 timeline |
| `fp8_layer_timing.csv` | `trtexec --dumpProfile` 逐层时延 |

**使用方式：** 在浏览器中打开 `https://ui.perfetto.dev`，加载 `.json` 文件查看 Trace 流水线。

---

## 附录 B：TRT Builder Config 参考

```
[待采集] 实际参数待调优后确认：

trtexec --onnx=openvla_fp8.onnx \
        --precision=fp8 \
        --calibProfile=calib_profile.json \
        --useCudaGraph \
        --layerNormFusion=1 \
        --useMyelin \
        --dumpProfile \
        --dumpLayerInfo \
        --iterations=10 \
        --workspace=[待定]
```

---

## 附录 C：待采集数据清单

### Profiling

- [ ]  TRT builder 最终配置
- [ ]  `trtexec --dumpProfile` 全量 layer timing（prefill / decode / full）
- [ ]  Nsight Systems trace（3 个）
- [ ]  FP8 精度验证：7-DoF RMSE + 逐模块 MSE

### 数据计算

- [ ]  算子类别聚合（FP8 版，含 Q/DQ 新类别）
- [ ]  CUDA Top 10 + 工时占比（FP8 版）
- [ ]  CPU 开销残余分析
- [ ]  Q/DQ 节点数量 + 开销

### 对比汇总

- [ ]  E2E 时延对比表
- [ ]  模块级时延对比表
- [ ]  Decode 逐 Step 对比表
- [ ]  瓶颈是否转移的判定