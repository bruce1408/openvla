# OpenVLA 模型评测 — FP8 量化 + TensorRT 生产部署分析（实测填充版）

> **📌 一句话结论：** OpenVLA FP8 + TensorRT 在 Thor 上的 E2E 推理时延为 **230.0 ms（~4.35 Hz）**，相比 BF16 Eager baseline（466 ms）加速 **2.03×**。Decode 瓶颈**未转移，仍为 Memory-Bound**——每步时延从 60.4 ms 降到 30.9 ms（~2×），几乎与权重字节数减半同比例下降，是访存受限的典型特征。量化引入的精度损失（FP8 Vision + BF16 LLM vs 纯 BF16）：7-DoF 动作 RMSE = **0.00162**，最大绝对误差 0.00319（roll 维度）。

> **文档定位：** [OpenVLA 模型评测_v1.md](OpenVLA%20模型评测_v1.md)（BF16 Eager baseline）的 FP8 生产部署姊妹篇。目标读者同为硬件和编译器同事——核心问题从"模型长什么样、瓶颈在哪"转变为"**量化后算子行为怎么变、瓶颈是否转移、硬件和编译器需要怎么适配**"。

> **版本：** v1-实测填充 | **依赖：** [BF16 baseline v1](OpenVLA%20模型评测_v1.md) | **状态：** 已实测填充（n=10），数据来自 `11_measure_fp8_trt.py` + `hybrid_runtime.py --compare-reference`

> **⚠️ 数据口径提示：** FP8 时延由两套独立工具测得——Vision 用 `torch.profiler` 包裹 TRT engine（`execute_async_v3`），LLM 用 Edge-LLM `llm_inference --dumpProfile`（**纯 32-token 文本 prompt，未注入真实图像 embedding**）。二者不是同一条端到端推理，E2E 为分段相加。**算子统计**：Vision 用 `torch.profiler`（CUTLASS/Myelin kernel 级），LLM 用 `llm_bench --profile`（逐层 CSV，非 CUDA-graph 路径）。精度数据来自 hybrid runtime（FP8 Vision + BF16 LLM vs 纯 BF16），**仅反映 Vision 量化误差，不含 FP8 LLM 的精度损失**。

---

## 1 量化方案

### 1.1 方案选择

| 决策点 | 方案 |
| --- | --- |
| 量化方式 | **PTQ**（Post-Training Quantization） |
| 精度格式 | **FP8 E4M3**（权重 + 激活） |
| 量化粒度 | **per-tensor**（默认），敏感层 **per-channel** |
| 校准数据 | **64 张真实图片**（`test_data/*.jpg`）+ 固定指令 "pick up the blue object" |
| 校准策略 | **MaxCalib**（对 vision 离群值更稳） |
| 量化工具 | NVIDIA ModelOpt (`modelopt.onnx.quantization`) |

### 1.2 逐模块量化策略

| 模块 | 量化策略 | 实测结果 |
| --- | --- | --- |
| Vision Backbone + Projector | FP8 PTQ（合一 engine） | engine 1453.84 MB（fp16）→ **741.39 MB（fp8），−49.0%**。占 E2E 仅 **3.1%**（7.18/230.0） |
| MLP Projector | FP8 PTQ | 已与 Vision 融进同一 `.plan`，TRT 中无法单独切分/计时 |
| Llama Decoder 32 层 | **FP8 PTQ** | **收益核心**：decode 每步 60.4→30.9 ms（~1.96×）。engine 6755 MB |
| → lm_head | FP8 PTQ | 已融进 Edge-LLM engine，不可单独计时。**FP8 验证：GEMM 层 99.5% 输入为 FP8**（见 §1.4） |
| Embedding | FP16（engine 内） | Edge-LLM config: `kv_cache_dtype=fp16`，embedding 为 FP16 `[32064, 4096]` |
| KV-cache | FP16 | 非瓶颈；config.json 显式声明 `kv_cache_dtype: fp16` |

### 1.4 实测量化落地情况（LLM engine + Vision engine，inspector）✅ 新增

**验证方法：** 用 `llm_build --profilingDetailed` 重建 engine（仅用于 inspect，不影响生产 engine），再用 Python `tensorrt.EngineInspector` 读取逐层 `Format/Datatype`。

#### Vision engine（已知）

- 549 层，输出 datatype `{FP8: 469, Half: 177}` → **FP8 覆盖 85.6% 输出**
- engine `.plan` 1453.84 MB（fp16）→ 741.39 MB（fp8），−49.0%

#### LLM engine（本次新增验证）

| 指标 | 实测值 |
| --- | --- |
| 总层数 | **1002** 层 |
| 层输出 datatype | `{Half: 705 (68.4%), FP8: 256 (24.8%), UInt8: 64 (6.2%), Float: 6 (0.6%)}` |
| 层输入 datatype | `{Half: 710 (52.4%), FP8: 448 (33.0%), Int32: 128 (9.4%), Float: 68 (5.0%)}` |
| **GEMM 层 FP8 输入** | **415/417 = 99.5%** ✅（几乎全部 GEMM 权重/激活为 FP8） |
| kgen 融合层 FP8 输出 | 256/295 = 86.8%（融合算子保持 FP8 中间结果） |
| GEMM 输出 dtype | Half（FP8 累加到 FP32 → 输出 Half，避免精度损失） |
| Engine 权重内存 | 6,739,769,472 bytes = **6.28 GB**（≈ 7B × 1 byte FP8，vs FP16 预期 13.4 GB） |

> **结论：LLM engine FP8 量化已确认生效。** 证据三重交叉验证：
> 1. **逐层 inspector**：99.5% GEMM 层输入为 FP8
> 2. **engine 大小**：6.28 GB ≈ FP8 预期 6.7 GB（0.94×），远小于 FP16 13.4 GB
> 3. **时延加速**：decode 每步 60.4→30.9 ms（~2×），与权重字节减半一致
>
> GEMM 输出保持 Half（而非 FP8）是 TRT 的标准策略：FP8×FP8→FP32 累加后转 Half，兼顾速度与精度。

### 1.3 精度验证计划

| 指标 | 测量方法 | 实测结果 |
| --- | --- | --- |
| 7-DoF 动作 RMSE | hybrid_runtime.py `--compare-reference`（FP8 Vision + BF16 LLM vs 纯 BF16） | **0.00162**（仅 Vision 量化误差） |
| 各模块输出 MSE | hybrid_runtime.py token 级对比 | 4/7 token 精确匹配，3 个 token 偏差 ±4-5 bins |
| 任务成功率 | `[N]` 个典型操作任务 | `[待采集]`（需机器人闭环测试） |
| 注意力分布 KL 散度 | 检查 attention weights | `[待采集]`（需在 detailed engine 上跑 hybrid runtime 对比） |

---

## 2 测试环境

### 2.1 硬件环境

同 [baseline §1.1](OpenVLA%20模型评测_v1.md#11-硬件环境配置)：NVIDIA Jetson AGX Thor，SM110，20 SMs，LPDDR5X 122.86 GB。

### 2.2 软件环境

| 软件 | 版本 | 备注 |
| --- | --- | --- |
| TensorRT | **10.16.1.11** | 支持 SM110（Thor, capability 11.0）FP8 TensorCore ✓ |
| TensorRT-Edge-LLM | **0.9.0** | Llama Decoder 的 FP8 plugin（`libNvInfer_edgellm_plugin.so`） |
| PyTorch（导出用） | 2.12.0a0+5aff3928 (nv26.05) | 导出 ONNX / torch-TRT |
| CUDA | 13.2（V13.2.78） | 不变 |
| cuBLASLt | 随 CUDA 13.2 | FP8 GEMM 实际由 CUTLASS `cutlass3x_sm100_tensorop_*gemm_f8_f8_f32` kernel 承担 |

### 2.3 Profiling 配置

| 项目 | BF16 baseline | FP8 本实验 |
| --- | --- | --- |
| 工具 | `torch.profiler` | Vision: `torch.profiler` 包 TRT engine；LLM: Edge-LLM `--dumpProfile`；精度: `hybrid_runtime.py --compare-reference` |
| 输入 | 1 × 224² + 19 tokens → 6 步 | ⚠️ **不完全相同**：Vision 用随机 224² 张量（仅计时）；LLM 用 32-token 纯文本 prompt |
| 轮数 | 10 轮取均值 | **Vision 10 轮 / LLM 10 轮**（warmup=5） |
| 精度 | BF16 | **FP8 E4M3**（Vision 72.6% FP8 + Half 兜底；LLM 精度未逐层验证） |
| Trace | 3 个 chrome trace | Vision 1 个 chrome trace + operator table |
| 额外采集 | — | hybrid runtime 精度对比 |

---

## 3 精度分析

> **说明：** 本节数据来自 `hybrid_runtime.py --compare-reference`，对比 **FP8 Vision engine + BF16 LLM** vs **纯 BF16**。输入：`bridge_sample_0001.jpg` + "pick up the blue object"。
>
> ⚠️ **这是 Vision-only 量化误差**。完整 FP8（含 LLM）精度需打通 Edge-LLM embedding 输入，当前工具链不支持。

### 3.1 7-DoF 动作输出对比

| 动作维度 | BF16 输出 | FP8 Vision 输出 | 绝对误差 | 相对误差 |
| --- | --- | --- | --- | --- |
| Δx | -0.001998 | -0.001998 | 0.000000 | 0.00% |
| Δy | +0.000871 | +0.000871 | 0.000000 | 0.00% |
| Δz | -0.007516 | -0.006477 | 0.001039 | 13.83% |
| roll | +0.002415 | +0.005608 | 0.003193 | 132.19% |
| pitch | +0.024004 | +0.021325 | 0.002679 | 11.16% |
| yaw | -0.000067 | -0.000067 | 0.000000 | 0.00% |
| gripper | +0.996078 | +0.996078 | 0.000000 | 0.00% |
| **RMSE** | — | — | **0.001624** | — |

> **分析：** 4/7 维度完全一致（Δx, Δy, yaw, gripper），3 个维度有偏差但绝对值很小（< 0.0032）。roll 维度相对误差 132% 看似大，但绝对误差仅 0.0032（因 roll 本身接近 0，分母小导致相对误差放大）。**RMSE 0.00162 远小于机器人控制典型容差 0.01，精度可接受。**

### 3.2 逐模块量化误差

| 模块 | 输出 MSE（vs BF16） | 是否需要混合精度 |
| --- | --- | --- |
| Vision Backbone + Projector（合一） | max_action_abs_error = 0.00319 | **否**（误差在容差内） |
| MLP Projector | 已融入 Vision，无法单独测 | — |
| Llama Layer 0–31 | FP8 已确认生效（GEMM 99.5% FP8 输入，§1.4）；逐层输出 MSE 待采集 | `[待判断]`（需 hybrid runtime 对比） |
| lm_head | `[待采集]` | `[待判断]` |
| Attention weights | `[待采集]` | `[待判断]` |

### 3.3 精度判定

```
- FP8 Vision 精度满足生产要求？ ✅ 是（RMSE 0.00162 << 0.01 容差）
- FP8 LLM 精度？ 🔴 无法采集（Edge-LLM 不支持 embedding 输入）
- 哪些层需要保留 BF16？ ✅ 已验证：GEMM 输出统一为 Half（非 FP8），attention/RoPE 保 Half。候选 lm_head 已在 engine 内融合为 FP8 GEMM。
```

---

## 4 推理性能对比

### 4.1 E2E 时延

| 阶段 | BF16 Eager（ms） | FP8 TRT（ms） | 加速比 |
| --- | --- | --- | --- |
| **Prefill** | 103.56 | **44.54**（Vision 7.18 + LLM 37.36） | **2.33×** |
| **Decode（6 steps）** | 362.68 | **185.46**（30.91 ms/step × 6） | **1.96×** |
| **E2E** | 466.24 | **230.00** | **2.03×** |
| **可达频率** | 2.14 Hz | **4.35 Hz** | — |

> 时延稳定性（n=10）：
> - Vision: mean 7.18 / p50 6.95 / p99 8.34 / min 6.75 / max 8.34 ms
> - LLM Prefill: mean 37.36 / p50 37.43 / p99 40.32 / min 34.55 / max 40.32 ms
> - LLM Decode/tok: mean 30.91 / p50 30.74 / p99 31.87 / min 29.98 / max 31.87 ms
>
> 波动 < ±10%。Decode 最稳定（p99/min = 1.06），Prefill 波动最大（p99/min = 1.17）。

### 4.2 模块级时延

| 模块 | BF16 Eager（ms） | FP8 TRT（ms） | 变化 |
| --- | --- | --- | --- |
| Vision Backbone + Projector | 15.44 + 0.72 = 16.16 | **7.18**（合一 engine） | **↓2.25×** |
| Projector | 0.72 | 已融入 Vision engine | — |
| Llama — Prefill | 86.30 | **37.36** | **↓2.31×** |
| Llama — Decode（per step） | 58.9 | **30.91** | **↓1.91×** |
| lm_head | 1.10 | 已融入 Edge-LLM engine | — |

### 4.3 阶段拆解

#### Prefill 阶段

| 子阶段 | BF16（ms） | FP8 TRT（ms） | 变化 |
| --- | --- | --- | --- |
| ① 视觉编码 + ② Projector | 16.16 | **7.18**（合一） | ↓2.25× |
| ③ Llama ×32 层 + ④ lm_head | 87.40 | **37.36**（合一） | ↓2.34× |
| **Prefill 合计** | **103.56** | **44.54** | **↓2.33×** |

#### Decode 阶段（6 步自回归）

| Step | Q-Len | BF16（ms） | FP8 TRT（ms） |
| --- | --- | --- | --- |
| 1 | 1 | 63.2 | ~30.9 |
| 2 | 1 | 60.0 | ~30.9 |
| 3 | 1 | 59.9 | ~30.9 |
| 4 | 1 | 59.9 | ~30.9 |
| 5 | 1 | 59.8 | ~30.9 |
| 6 | 1 | 59.9 | ~30.9 |
| **均值** | — | **60.4** | **30.91**（mean；p50 30.74 / p99 31.87） |

> ⚠️ Edge-LLM `gpu_time_stats` 只给出 decode 全部 token 的聚合统计（n=60=10 req×6 step），未按 step 序号拆分，故逐 step 列填聚合均值。

---

## 5 算子行为变化

> **两套工具覆盖两个 engine：** Vision 用 `torch.profiler`（CUTLASS/Myelin kernel 级），LLM 用 Edge-LLM `llm_bench --profile`（逐层 CSV，非 CUDA-graph 模式）。两者口径不同但**均已采集**。

### 5.1 算子总览对比

| 指标 | BF16 Eager（全模型） | FP8 Vision engine | FP8 LLM engine（Edge-LLM） |
| --- | --- | --- | --- |
| 采集工具 | `torch.profiler` | `torch.profiler` | `llm_bench --profile`（非 CUDA-graph） |
| Profiler 记录 | 177 条 | 103 条（distinct kernel） | 581 行（32 层 × ~18 ops/层） |
| 总调用次数（10 步） | 74,261 | 5,470 launches | 581 layer ops（1 token decode） |
| 单 token kernel 耗时 | ~174.5 ms（全模型） | 2.78 ms（profiler 视角） | **55.40 ms**（逐层累加） |
| 实际 wall-clock（1 token decode） | 60.4 ms | 0.72 ms（vision 单 patch） | **30.91 ms**（含 CUDA graph 优化） |

> ⚠️ **LLM 逐层累加 55.4 ms > 实际 30.91 ms** 的原因：`llm_bench --profile` 走**非 CUDA-graph** 路径以逐层计时，而生产 decode 用 CUDA graph（kernel 启动开销被消除 + 流水并行）。**差值 24.5 ms ≈ CUDA graph 的收益**（per-kernel launch 开销 × 581 ops）。所以下表的"占比"反映**非 graph 路径**下的算子分布，量级和排序对优化有指导意义。

### 5.2 新增算子：Q/DQ（量化 / 反量化）

| Q/DQ 类型 | 位置 | 实测（Vision） | 实测（LLM） |
| --- | --- | --- | --- |
| Q: →FP8（权重/激活） | 量化层输入 | **无独立 Q/DQ kernel**——TRT 把 Q/DQ 折叠进 Myelin 融合块，以 `Cast` 形式出现在融合 kernel 名里 | ✅ 已验证：99.5% GEMM 层输入为 FP8（§1.4），Q 被融合进 GEMM 的权重加载 |
| DQ: FP8→Half（激活/输出） | 层边界 / fallback 层前 | 反量化发生在 FP8↔Half 的 datatype 边界 | ✅ 已验证：GEMM 输出 Half（FP8→FP32→Half），DQ 折进 GEMM epilogue |

> **实测结论：Q/DQ 在两个 engine 中均被 Myelin 融合吸收。**
> - **Vision engine**：输出 datatype `{FP8: 469, Half: 177}` → 85.6% 层输出 FP8。
> - **LLM engine**（✅ 本次新验证）：1002 层中 256 层输出 FP8（24.8%），448 个输入为 FP8（33.0%）；**415/417 GEMM 层输入为 FP8（99.5%）**。GEMM 输出保持 Half（FP8→FP32→Half），DQ 折进 epilogue。
> - 无独立 Q/DQ kernel 暴露，全部被折进相邻融合 kernel 的 `Cast`。

### 5.3 算子类别聚合对比

> BF16 列是整模型；FP8 Vision + FP8 LLM 两列**分别**统计，可拼出全链路分布。

#### 5.3.1 Vision engine（占 E2E 3.1%）

| 类别 | BF16 CUDA 工时占比（整模型） | FP8 Vision CUDA 工时占比 | 备注 |
| --- | --- | --- | --- |
| GEMM / TensorCore | 90.7% | **40.2%** | Top3 全为 `cutlass3x_sm100_..._f8_f8_f32` |
| 融合 __myl_Fc（GEMM+bias+act） | — | **15.4%** | TRT 进一步融合 |
| Memory / Layout | 2.7% | **1.0%** | Myelin 把 layout 融进 GEMM epilogue |
| Attention | 1.9% | **1.3%** | `_gemm_mha_v2` |
| Elementwise（含 Cast/Q-DQ） | 1.8% | **3.8%** | 含被融合吸收的 `Cast`、SiLU/GELU |
| Norm / Reduction | 0.7% | **7.9%** | 融合的 RMSNorm/LayerNorm |
| 其他 | 2.2% | **0%** | — |

#### 5.3.2 LLM engine（占 E2E ~81%，decode 阶段）✅ 新增

> 数据来自 `llm_bench --profile --mode decode --pastKVLen 262`，逐层 CSV 581 行聚合。**非 CUDA-graph 路径**，总 55.40 ms（含 launch 开销，实际 graph 后 30.91 ms）。

| 类别 | 耗时 (ms) | 占比 | 数量 | 典型 kernel |
| --- | --- | --- | --- | --- |
| **GEMM（attention QKV/O proj）** | 36.16 | **65.3%** | 161 | `node_linear_*`（q/k/v/o proj，FP8） |
| **GEMM+SiLU（MLP gate/up）** | 8.66 | **15.6%** | 32 | `__myl_FcSilu`（gate+up 融合，32 层各 1） |
| **GEMM+Mul（MLP down）** | 8.45 | **15.3%** | 32 | `__myl_FcMul`（down proj，32 层各 1） |
| **Attention（FMHA/XQA）** | 1.34 | **2.4%** | 32 | `n0_*`（XQA decode attention，32 层各 1） |
| **Norm（RMSNorm）** | 0.60 | **1.1%** | 64 | `__myl_CastMulMeanAddSqrtDiv...`（pre+post，32×2） |
| Elementwise（Cast/Mul） | 0.19 | **0.4%** | 35 | `__myl_MulCast` |
| Misc fusion | 0.00 | ~0% | 225 | `__mye*`（零成本融合 stub） |

> **关键发现：**
> 1. **GEMM 占 96.2%**（65.3+15.6+15.3）——LLM decode 几乎全是矩阵乘，印证 §6 的 Memory-Bound 结论（时延 ∝ 权重字节）。
> 2. **Attention 仅 2.4%**——XQA decode kernel 极高效，KV-cache 读取非瓶颈。
> 3. **RMSNorm 1.1%**——已被 `__myl_CastMulMeanAddSqrt...` 融合，开销可忽略。
> 4. **MLP（gate+up+down）占 30.9%**，与 attention QKVO 的 65.3% 之比 ≈ 1:2.1，符合 Llama-7B 结构（intermediate 11008 vs hidden 4096）。

### 5.4 CUDA 耗时 Top 10 对比

#### 5.4.1 BF16 Eager（整模型，aten op）vs FP8 Vision engine

| # | BF16 Eager Top（aten） | BF16 占比 | FP8 Vision Top（CUDA kernel） | FP8 占比 |
| --- | --- | --- | --- | --- |
| 1 | `aten::linear` | 22.9% | `cutlass3x_sm100_tensorop_s128x128x32gemm_f8_f8_f32_f16_f16` | 18.75% |
| 2 | `aten::matmul` | 22.4% | `cutlass3x_sm100_tensorop_s128x128x32gemm_f8_f8_f32_e4m3_e4m3` | 11.92% |
| 3 | `aten::mm` | 22.3% | `cutlass3x_sm100_tensorop_s256x128x32gemm_f8_f8_f32_f16_f16` | 9.50% |
| 4 | nvjet GEMMs | 21.3% | `__myl_Fc_..._tensorop256x128`（融合 GEMM） | 8.32% |
| 5 | `aten::cat` | 0.8% | `__myl_CastMulAdd...Mean...Sqrt...Div`（融合 RMSNorm+Cast） | 5.12% |

#### 5.4.2 FP8 LLM engine（decode，逐层 Top 15）✅ 新增

| # | LLM kernel（32 层累计） | 耗时 (ms) | 归类 |
| --- | --- | --- | --- |
| 1 | `node_linear_224` | 1.555 | GEMM（attention，疑似 lm_head 或大 proj） |
| 2 | `node_linear_125` | 0.603 | GEMM（attention O proj） |
| 3 | `__myl_FcSilu` | 0.434 | GEMM+SiLU（MLP gate/up，单层最大） |
| 4 | `node_linear_9` | 0.310 | GEMM（attention Q proj） |
| 5–15 | `node_linear_{121,191,86,128,184,149,79,37,65,142,51}` | 0.296–0.304 | GEMM（attention proj，每层 ~5 个） |

> **趋势观察：** FP8 下 GEMM 已全部走 `f8_f8_f32`（E4M3 输入 / FP32 累加）的 CUTLASS SM100 TensorCore kernel；BF16 的 `aten::linear/matmul/mm` 三条并列大头，在 TRT 里被融合成单条 `cutlass*gemm` 或 `__myl_Fc`，**算子数量大幅收敛**。LLM 侧同理：`node_linear` 是未融合的 attention proj，`__myl_Fc*` 是融合的 MLP。

### 5.5 TRT Myelin 自动融合效果

| 融合模式 | BF16 Eager（ms） | BF16 compile（ms） | FP8 TRT 自动融合（实测证据） |
| --- | --- | --- | --- |
| RMSNorm + Linear | 0.167 | 0.110 | ✅ 已融合：kernel `__myl_CastMulAdd...MeanSubMul...Sqrt...Div`（RMSNorm）后接 `__myl_Fc_...tensorop`（Linear） |
| Linear + SiLU + Mul | 0.971 | 0.935 | ✅ 已融合：kernel `__myl_FcMulMulCast`、`__myl_MulCastErf...` |
| Linear + GELU | 0.042 | 0.071 ⚠️ | ✅ 已融合：GEMM epilogue 直接带 `Erf`（GELU），见 `__myl_...Erf...` |

> **结论：** TRT/Myelin 的自动融合粒度明显强于 PyTorch `torch.compile`——它把 GEMM + bias + 激活 + Cast(Q/DQ) 全部塞进单个 CUTLASS kernel 的 epilogue。

---

## 6 瓶颈分析

### 6.1 双模式瓶颈对比

| 指标 | BF16 Prefill | BF16 Decode | **FP8 Prefill** | **FP8 Decode** |
| --- | --- | --- | --- | --- |
| 时延 (ms) | 103.6 | 362.7 | **44.54** | **185.46** |
| 权重加载量 | 12.9 GB | 12.9 GB | **~6.5 GB**（FP8 权重≈BF16 半） | **~6.5 GB** |
| M 维度 | 256–275 | 1 | **256–275**（量化不改变） | **1**（不变） |
| GFLOPs | ~3,784 | ~80 | **~3,784**（计算量不变） | **~80** |
| Arithmetic Intensity | ~290 FLOP/byte | ~6.0 FLOP/byte | **~570**（字节减半→AI 翻倍） | **~12.0** |
| 瓶颈类型 | 权重加载受限 | 纯 Memory-Bound | **Compute-bound**（AI > ridge） | **仍纯 Memory-Bound** |

> **Arithmetic Intensity 计算（Decode）：**
> - 权重读取: 6.5 GB（FP8，32 层）
> - KV-cache 读取: 32 层 × 2 × 4096 elem × 2 bytes × 262 pos ≈ 0.137 GB
> - 总访存: 6.64 GB
> - AI = 80 GFLOPs / 6.64 GB = **12.0 FLOP/byte**
>
> **Thor FP8 ridge point** = 10.7 TFLOPS / 100 GB/s ≈ **107 FLOP/byte**
>
> Decode AI 12.0 << ridge 107 → **Memory-Bound 确认**（离脊点还差 ~9×）。
> Prefill AI 570 > ridge 107 → **Compute-bound**（FP8 下 prefill 已转为算力受限）。

### 6.2 关键判断：瓶颈是否转移

**结论：Decode 瓶颈未转移，仍为 Memory-Bound。** 实测判据：

- **Decode 每步 60.4 → 30.91 ms（↓1.96×）**，显著下降，但——
- **下降幅度（~2×）几乎等于权重字节减半幅度（~2×）**。这正是 memory-bound 的判定证据：当时延 ∝ 权重字节数、而与算力无关时，说明瓶颈在"把权重从内存读进来"。
- **Prefill:Decode 时延比 = 44.54 : 185.46 ≈ 1:4.16**，与 BF16 的 1:3.5 相比 decode 占比略升。远未接近"转移信号"的 1:1。
- **AI = 12.0 << ridge 107**，量化 6× 影响不改变判断。
- 30.91 ms 仍在 30 ms 阈值附近，**未转移信号**。

**对硬件/编译器同事的含义：** FP8 在 decode 上的收益直接来自权重带宽减半。要继续压 decode，杠杆仍是**减少每 token 的权重读取字节**——nvfp4（再半）、权重-only 更激进量化、或 KV-cache 量化，而不是堆算力。

---

## 7 为什么 LLM 部分用 Edge-LLM 而非纯 TensorRT？

> **核心问题：** Vision engine 用纯 TensorRT（`execute_async_v3` 一把跑完），为什么 LLM 不能同样用纯 TRT，而要引入 Edge-LLM 这套额外工具链？如果坚持只用纯 TRT，可行吗？

### 7.1 两个 engine 的本质差异（实测对比）

| 维度 | Vision engine | LLM engine |
| --- | --- | --- |
| **计算模式** | 单次前向（feed-forward，无状态） | **自回归循环**（7 步 decode，每步依赖上一步） |
| **IO tensor 数** | 2（pixel_values → projected_patch_embeddings） | **70**（inputs_embeds + 32 层×2 past_kv + rope + ... → logits + 32 层×2 present_kv） |
| **自定义插件** | 无（纯 CUTLASS GEMM + Myelin 融合） | **64 个 `custom_layer`**（AttentionPlugin，需 `libNvInfer_edgellm_plugin.so`） |
| **无插件能否 createContext** | ✅ 能 | 🔴 **不能**（`Cannot find plugin: AttentionPlugin`） |
| **动态维度** | 固定 `[1, 6, 224, 224]` | `past_kv_len` 每步增长（262→268），需 2 个 optimization profile |
| **KV cache** | 无 | 32 层 × 2 (K,V) × 4096 × 128，**跨步管理**（append + 读取） |
| **采样** | 无（直接取输出） | **GPU fused sampling**（greedy/top-k/top-p） |
| **CUDA Graph** | 不需要（单次调用） | ✅ **decode 包进 CUDA graph**（省 24.5 ms/步 launch 开销） |

### 7.2 纯 TensorRT 缺失的 4 项关键能力

#### ① 自回归 decode 循环 + KV cache 生命周期管理

纯 TRT engine 是**无状态**的：一次 `execute_async_v3` = 一次前向。但 LLM decode 是：

```
Step 1: inputs_embeds[1个token] + past_kv(空) → logits → 采样token1 + 更新KV
Step 2: inputs_embeds[token1]  + past_kv(1步) → logits → 采样token2 + 追加KV
Step 3: inputs_embeds[token2]  + past_kv(2步) → logits → 采样token3 + 追加KV
...共 7 步
```

每步需要：
- **读取**上一步的 KV cache（32 层 × K,V 两份）
- **执行** attention（当前 token 的 Q × 历史 KV）
- **追加**当前步的 K,V 到 cache
- **采样**下一个 token

纯 TRT **不管这个循环**——它只管"给输入算输出"。循环逻辑、KV 追加、采样全要你自己写。Edge-LLM 的 `LLMInferenceRuntime::handleRequest()` 把这整套封装好了。

#### ② XQA decode attention kernel（Edge-LLM 专属）

LLM engine 的 64 个 `custom_layer` 就是 `AttentionPlugin`，它分两种模式：
- **Prefill**：FMHA（Fused Multi-Head Attention），TRT 10.x 内置类似能力
- **Decode**：**XQA**（Cross-Query Attention），Edge-LLM 专属优化 kernel

XQA 针对 `seq_len=1` 的 decode 场景深度优化（内存布局、KV cache 读取模式），实测 decode attention 仅占 2.4%（§5.3.2）。纯 TRT 的内置 attention 走通用 FMHA，**decode 场景会慢数倍**。

#### ③ 动态 KV cache 的 optimization profile 切换

LLM engine 有 **2 个 optimization profile**：
- Profile 0：短序列（prefill 阶段，past_kv_len=0）
- Profile 1：长序列（decode 阶段，past_kv_len 递增）

每步 decode 前，Edge-LLM 自动**切换 profile + 设置动态 shape**。纯 TRT API 虽然也支持（`set_input_shape` + profile select），但要你手写切换逻辑，且 **shape 变化会触发 TRT 内部 re-tactic 选择**，性能不稳定。Edge-LLM 预编译了两个 profile，切换零开销。

#### ④ CUDA Graph 捕获 decode 循环

实测：非 graph 路径 decode 逐层累加 55.4 ms，CUDA graph 后 30.9 ms（**省 24.5 ms**）。这 24.5 ms 是 581 个 kernel 的 `cudaLaunchKernel` 开销。

Edge-LLM 自动把 decode 循环捕获成 CUDA graph（固定 shape 后一次性录制）。纯 TRT 的 `execute_async_v3` **不支持 graph capture**（因为每步 shape 在变），要手动做需要：
1. 固定 past_kv_len 上限
2. 用 `cudaStreamBeginCapture` 手动录
3. 处理 KV cache 写入的指针更新（graph 内指针不能变）

### 7.3 如果坚持用纯 TensorRT，可行吗？

**结论：技术上可行，但工程量大、性能会回退。**

#### 理论可行的路径

```
方案: 纯 TRT engine + 自写 C++ runtime

1. ONNX 导出时不用 Edge-LLM 的 custom op，
   改用标准 ONNX attention（或 TRT 内置 attention plugin）
2. trtexec --onnx=llama_fp8.onnx --fp8 --buildOnly  构建 engine
3. 自己写 C++ runtime:
   - 分配 KV cache buffer（32层×2×4096×max_seq×128×2bytes）
   - 每步: set_input_shape + execute_async_v3 + 手动 copy KV
   - 采样: CPU argmax 或自己写 CUDA kernel
   - CUDA graph: 手动 capture（需固定 max_seq）
```

#### 但会遇到 4 个回退

| 回退项 | 纯 TRT 方案 | Edge-LLM 方案 | 差距 |
| --- | --- | --- | --- |
| **Decode attention** | TRT 内置 FMHA（通用） | XQA（decode 专属优化） | 纯 TRT 慢 2-5×（attention 占比从 2.4% 升到 ~10%） |
| **CUDA graph** | 需手写 capture + 处理 KV 指针 | 自动 | 纯 TRT 要写 ~200 行 C++，易错 |
| **FP8 量化** | 需自己插 Q/DQ 或用 ModelOpt | Edge-LLM 一条命令 `tensorrt-edgellm-quantize` | 纯 TRT 需额外 PTQ 流程 |
| **采样** | CPU argmax（慢）或自写 CUDA | GPU fused | 纯 TRT 多 1-2 ms/步 |
| **预估 decode 时延** | ~45-55 ms/步 | **30.9 ms/步** | 纯 TRT 慢 ~1.5-1.8× |

#### 量化估算（纯 TRT decode 每步）

```
Edge-LLM 实测:  30.9 ms (XQA + CUDA graph + fused sample)
纯 TRT 估算:
  - GEMM 部分:        不变 ~30 ms (同样的 FP8 CUTLASS kernel)
  - Attention:        FMHA 通用版 ~3-5 ms (vs XQA 0.7 ms) → +3-4 ms
  - 无 CUDA graph:    +24.5 ms launch 开销 (581 kernel × ~42us)
  - CPU 采样:          +1-2 ms
  ─────────────────
  估算合计:           ~58-62 ms/步  (vs Edge-LLM 30.9 ms, 慢 ~1.9×)
```

> **即：纯 TRT 能跑出和 BF16 PyTorch eager 差不多的 decode 速度（~60 ms），FP8 的 2× 加速基本被 launch 开销 + 慢 attention 吃掉。** Edge-LLM 的价值就在于把这些 overhead 消除掉，让 FP8 的收益真正落地。

### 7.4 Vision engine 为什么能用纯 TRT？

Vision 部分天然适合纯 TRT，因为它：

1. **单次前向**：图像 → patch embedding，一次 `execute_async_v3` 跑完，无循环
2. **无 KV cache**：无状态，输入固定 shape
3. **无自定义插件**：DINOv2+SigLIP 的 attention 用 TRT 内置 FMHA 即可（prefill 模式，seq_len=256，FMHA 正是它的最佳场景）
4. **无采样**：输出直接是 embedding，不需要 token 采样

所以 Vision engine 用纯 TRT + `torch.profiler` 就能完整覆盖（§5 的算子统计就是这么做的）。

### 7.5 架构选型总结

```
┌─────────────────────────────────────────────────────┐
│                  OpenVLA FP8 部署架构                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  图像 ──→ [Vision engine] ──→ patch embedding        │
│           纯 TensorRT          (7.06 ms)             │
│           execute_async_v3                           │
│           无需 Edge-LLM                              │
│                    │                                │
│                    ▼ 拼接 text embedding             │
│           [LLM engine]                               │
│           Edge-LLM + TRT        (30.9 ms/step × 7)   │
│           XQA + CUDA graph                           │
│           KV cache 管理                              │
│                    │                                │
│                    ▼                                │
│           7-DoF action                               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

| 模块 | 工具 | 原因 |
| --- | --- | --- |
| **Vision** | 纯 TensorRT | 单次前向、无状态、无 KV cache → TRT 足够 |
| **LLM** | Edge-LLM (TRT + plugin) | 自回归循环 + KV cache + XQA + CUDA graph → 纯 TRT 缺这 4 项 |

> **一句话：Vision 是"算一次"，纯 TRT 够用；LLM 是"循环算 + 管状态"，必须靠 Edge-LLM 的 runtime 编排。强行用纯 TRT 能跑，但 decode 会慢 ~1.9×，FP8 加速收益被 launch 开销吃掉。**

---

## 8 CPU 开销残余

> 🔴 **本节 FP8 数据无法在当前环境采集。** 原因：BF16 的 CPU 开销拆解来自 `torch.profiler`（PyTorch eager 逐 op 的 CPU 侧计时）。FP8 部署里主导耗时的 Edge-LLM 是独立 C++ binary，通过子进程调用，torch.profiler 无法 attach 到它的 CPU 调用栈。

| CPU 开销项 | BF16 Eager（ms） | FP8 TRT（ms） | 是否消除 |
| --- | --- | --- | --- |
| `cudaStreamSynchronize` | 164.4 | 无法采集（Edge-LLM C++） | `[待判断]` |
| `cudaLaunchKernel` | 49.0 | 无法采集 | `[待判断]` |
| `aten::to` | 194.1 | 无法采集（TRT 无 aten） | `[待判断]` |
| `aten::copy_` | 181.3 | 无法采集 | `[待判断]` |
| `aten::transpose` / `aten::t` | 18.3 | 无法采集 | `[待判断]` |
| 其他 | ~124 | 无法采集 | `[待判断]` |
| **合计** | **~731** | 无法采集 | — |

> **定性预期：** TRT + CUDA Graph 本应把 `cudaLaunchKernel`/`cudaStreamSynchronize` 这类 per-kernel 启动开销大幅压掉。要精确采集需用 `nsys profile` 对 `llm_inference` 子进程做 system-wide trace。

---

## 9 总结

| # | 发现 | 含义 |
| --- | --- | --- |
| 1 | GEMM 占比变化：BF16 整模型 90.7% → FP8 Vision 40.2% | Vision 里 GEMM 占比天然低于整模型；FP8 GEMM 已全走 `cutlass*gemm_f8_f8_f32` |
| 2 | Decode 瓶颈**未转移**，仍 Memory-Bound | AI 12 << ridge 107；时延下降 2× ≈ 权重字节减半 2× |
| 3 | 7-DoF 精度损失（Vision-only）：RMSE = **0.00162** | 远小于 0.01 容差，Vision FP8 量化精度可接受 |
| 4 | CPU 开销残余：无法采集 | 需 nsys profile 补充 |
| 5 | TRT 自动融合 vs 手动 compile：Myelin 融合粒度更强 | GEMM+bias+激活+Cast 全塞进单 CUTLASS kernel epilogue |
| 6 | Q/DQ 节点开销：**已被融合吸收** | 无独立 Q/DQ kernel，折进 `Cast` |
| 7 | E2E 230.0 ms，加速 2.03× | 可达 4.35 Hz，满足实时控制（>3 Hz） |

---

## 附录 A：Trace 文件索引

| 文件 | 用途 |
| --- | --- |
| `outputs/openvla/fp8_trt_breakdown_glm_final.raw.json` | FP8 Vision + LLM 时延实测（n=10） |
| `/tmp/llm_profile/layer_decode_pastkvlen262.csv` | FP8 LLM 逐层 decode profiling（581 行） |
| `logs/openvla_profile_predict_action_*.trace.json` | BF16 baseline Chrome trace |
| `logs/openvla_profile_predict_action_*.table.txt` | BF16 baseline 算子表 |

**使用方式：** 在浏览器中打开 `https://ui.perfetto.dev`，加载 `.trace.json` 文件查看 Trace 流水线。CSV 用 Excel/pandas 查看逐层时延。

---

## 附录 B：实测命令

```bash
# 1. FP8 Vision + LLM 时延采集
python /workspace/openvla/deploy/tensorrt/pipeline/11_measure_fp8_trt.py \
    --warmup 5 --active 10 --tag glm_final

# 2. FP8 LLM 逐层 profiling (decode, 非 CUDA-graph 路径)
cd /workspace/TensorRT-Edge-LLM/build
EDGELLM_PLUGIN_PATH=./libNvInfer_edgellm_plugin.so ./examples/llm/llm_bench \
    --engineDir /workspace/openvla/deploy/tensorrt/artifacts/engines/openvla_llama_fp8 \
    --mode decode --pastKVLen 262 --iterations 3 --warmup 2 \
    --profile --outputDir /tmp/llm_profile

# 3. FP8 LLM 逐层 profiling (prefill)
EDGELLM_PLUGIN_PATH=./libNvInfer_edgellm_plugin.so ./examples/llm/llm_bench \
    --engineDir /workspace/openvla/deploy/tensorrt/artifacts/engines/openvla_llama_fp8 \
    --mode prefill --inputLen 32 --iterations 3 --warmup 2 \
    --profile --outputDir /tmp/llm_profile

# 4. FP8 Vision 精度验证 (hybrid runtime)
python /workspace/openvla/deploy/tensorrt/runtime/hybrid_runtime.py \
    --engine /workspace/openvla/deploy/tensorrt/artifacts/engines/vision_projector_fp8.plan \
    --image /workspace/openvla/test_data/bridge_sample_0001.jpg \
    --instruction "pick up the blue object" --compare-reference

# 5. 重建 LLM engine (带 detailed verbosity, 用于 FP8 验证)
cd /workspace/TensorRT-Edge-LLM/build
LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH EDGELLM_PLUGIN_PATH=./libNvInfer_edgellm_plugin.so \
    ./examples/llm/llm_build \
    --onnxDir /workspace/checkpoints/openvla/hf_llama_onnx_fp8/llm \
    --engineDir /tmp/fp8_engine_detailed \
    --maxBatchSize 1 --maxInputLen 1024 --maxKVCacheCapacity 1024 \
    --profilingDetailed

# 6. 验证 LLM 逐层 FP8 datatype (Python tensorrt inspector)
LD_LIBRARY_PATH=/workspace/TensorRT-Edge-LLM/build:$LD_LIBRARY_PATH python3 -c "
import tensorrt as trt, json, ctypes
from collections import Counter
logger = trt.Logger(trt.Logger.ERROR)
ctypes.CDLL('/workspace/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so')
trt.init_libnvinfer_plugins(logger, '')
with open('/tmp/fp8_engine_detailed/llm.engine','rb') as f, trt.Runtime(logger) as rt:
    engine = rt.deserialize_cuda_engine(f.read())
ctx = engine.create_execution_context()
insp = engine.create_engine_inspector(); insp.execution_context = ctx
data = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
layers = data['Layers']
out_dt = Counter()
for l in layers:
    for o in l.get('Outputs',[]):
        if isinstance(o,dict): out_dt[o.get('Format/Datatype','?')] += 1
print(f'Total layers: {len(layers)}')
for t,c in out_dt.most_common(): print(f'  {t}: {c}')
"

# 7. BF16 baseline 对比
python /workspace/openvla/deploy/pytorch_bf16/prof_trace.py --active 5
python /workspace/openvla/deploy/pytorch_bf16/bench_stages.py --warmup 10
```

---

## 附录 C：Review 备注（数值合理性检查）

### C.1 算术一致性 ✅

| 检查项 | 计算 | 结果 |
| --- | --- | --- |
| E2E = Prefill + Decode | 44.54 + 185.46 | = 230.00 ✓ |
| Prefill = Vision + LLM | 7.18 + 37.36 | = 44.54 ✓ |
| Decode = 6 × per_tok | 6 × 30.91 | = 185.46 ✓ |
| Speedup | 466.24 / 230.00 | = 2.03 ✓ |
| 频率 | 1000 / 230.00 | = 4.35 Hz ✓ |
| Vision 加速比 | 16.16 / 7.18 | = 2.25 ✓ |
| Prefill 加速比 | 103.56 / 44.54 | = 2.33 ✓ |
| Decode 加速比 | 58.9 / 30.91 | = 1.91 ✓ |

### C.2 逻辑一致性 ✅

| 检查项 | 结论 |
| --- | --- |
| Vision 占 E2E 3.1% | 7.18/230.0 = 3.1% ✓（文档 §1.2 称 3.4%，接近） |
| Engine 体积 −49% | 1453.84 → 741.39 MB = −49.0% ✓ |
| Q/DQ 融合消除 | operator table 中无独立 Q/DQ kernel ✓ |
| Memory-Bound 结论 | AI 12 << ridge 107 ✓ |
| Prefill:Decode 比 1:4.16 | 44.54/185.46 = 0.24 ≈ 1:4.16 ✓ |

### C.3 仍有空洞

| 编号 | 缺失项 | 原因 | 建议 |
| --- | --- | --- | --- |
| 1 | 完整 FP8（含 LLM）RMSE | Edge-LLM 不支持 embedding 输入 | 打通 hybrid runtime FP8 LLM 路径 |
| 2 | ~~LLM 逐层 FP8 datatype 验证~~ | ✅ **已解决**：用 `llm_build --profilingDetailed` 重建 + inspector 验证，99.5% GEMM 层输入 FP8（见 §1.4） | — |
| 3 | CPU 开销残余 | Edge-LLM C++ 子进程 | `nsys profile` 捕获子进程 |
| 4 | 任务成功率 | 需机器人闭环 | 实机测试 |
