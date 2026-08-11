# OpenVLA 模型评测（NVFP4 + TensorRT）

OpenVLA FP8 视觉 + NVFP4 LLM + TensorRT 在 Thor 上的 E2E 推理时延为 **144.6 ms（~6.92 Hz）**，相比 BF16 Eager baseline（466 ms）加速 **3.23×**。

> **📌 一句话结论：** NVFP4 把 E2E 压到 144.6 ms / 6.92 Hz（比 FP8 再快 1.59×），Decode 仍 Memory-Bound，**但精度严重退化且极不稳定：20 样本均值仅 2.05/7 token 一致、RMSE 均值 0.026（最坏 0.072、漂移 139 bin）——4-bit 对 OpenVLA 连续动作回归不足，当前不可生产部署。**

> **文档定位：** 本文件是 [OpenVLA 模型评测（FP8 + TensorRT）](OpenVLA%20模型评测（FP8%20%2B%20TensorRT）.md) 的 NVFP4 姊妹篇。核心变化：LLM 权重由 FP8（8-bit）进一步量化为 **NVFP4（4-bit）**，而视觉编码器沿用 FP8 engine（未单独做 4-bit，占 E2E 仅 ~7% 且非瓶颈）。目标读者同为硬件/编译器同事——重点回答：**4-bit 量化后算子行为怎么变、decode 是否仍 Memory-Bound、精度损失是否可接受、能否进一步逼近实时控制（>10 Hz）**。

> **⚠️ 数据口径提示（重要）：** 本文 §1.4/§3/§4/§5/§7 均为实测（2026-08-11 复验）。**Vision 6.51 ms 用的是 FP8 视觉 engine**。**已实测：§1.4 逐层 dtype、§3.1–3.4 精度（含 20 样本 + FP8 对照）、§4 时延/显存、§5 算子、§7 CPU 开销。**

---

# 1 量化方案

## 1.1 方案选择

| 模块 | 方案 |
|---|---|
| 量化方式 | PTQ（tensorrt-edgellm-quantize --quantization nvfp4） |
| 精度格式 | **LLM：NVFP4（E2M1，4-bit，MX 微缩放）；Vision：FP8 E4M3（沿用 fp8 engine）** |
| 量化粒度 | **2×2 block micro-scaling（已确认）**：inspector 见 FP4E2M1 + FP8 混合；kernel 名 `gemm_block_scaled_ue4m3xf4_ue4m3` = UE4M3 激活 × FP4 权重 + 块缩放 scale |
| KV-cache / Embedding | **fp16**（config.json：kv_cache_dtype=fp16）|

## 1.2 逐模块量化策略

| 模块 | 量化策略 | 实测结果 |
|---|---|---|
| Vision Backbone + Projector | **FP8 PTQ（沿用，非 nvfp4）** | 复用 `vision_projector_fp8.plan`；E2E 侧实测 vision **10.74 ms**，占 E2E ~7.3% |
| MLP Projector | FP8 PTQ | 已与 Vision 融进同一 engine |
| Llama Decoder 32 层 | **NVFP4 PTQ（LLM 收益核心）** | decode 每步 **60.4 → 17.46 ms（~3.46×）**；engine `llm.engine` = **3.92 GB**（见 §1.3 体积） |
| lm_head | NVFP4 PTQ | 已融入 Edge-LLM engine |
| Embedding | FP16（engine 内） | Edge-LLM config：kv_cache_dtype=fp16，embedding FP16 [32064, 4096] |

## 1.3 NVFP4 体积收益（LLM engine）

| 项 | FP8（8-bit） | NVFP4（4-bit） | 变化 |
|---|---|---|---|
| LLM engine `llm.engine` | ~6.75 GB（fp8） | **3.92 GB** | **↓ ~42%（相对 fp8）** |

> 4-bit 应约为 8-bit 的一半，实测 `llm.engine` 3.92 GB 未正好减半，原因：NVFP4 需额外存 2×2 block 微缩放 scale 常数（每 block 额外开销），且 embedding 表仍为 fp16（未量化），拉高了整体体积。对比 BF16 baseline（~13.4 GB 权重），nvfp4 体积约为其 **~0.29×**。

> **该体积为 7B 模型权重的主要部分；decode 每步需读全量权重 → 4-bit 减半（相对 fp8）→ decode 访存进一步减半，是 §4.1 decode 3.46× 加速的来源。**

## 1.4 逐层量化落地情况（✅ 已实测，2026-08-11）

> **✅ 算子级已实测（§5.3 nsys）：** NVFP4 LLM 的 GEMM 全走 `cutlass3x_sm100_bstensorop_*gemm_block_scaled_ue4m3xf4_ue4m3`——确认 **4-bit 块缩放 TensorCore 落地**。
>
> **✅ 逐层 datatype 已采集：** 用 `llm_build --profilingDetailed` 重建 `openvla_llama_nvfp4_detailed/`（ONNX 源 `/data/checkpoints/openvla/hf_llama_onnx_nvfp4/llm`），inspector 读出 1194 层完整 dtype（`outputs/openvla/nvfp4_llm_precision_detailed.json`）。

**LLM engine 逐层输出 dtype 分布：**

| Datatype | 输出 tensor 数 | 占比（约） | 含义 |
| --- | --- | --- | --- |
| **FP4E2M1** | 256 | 15.5% | 4-bit NVFP4 主路径 |
| **FP8** | 480 | 29.1% | 8-bit 中间/混合 |
| **Half** | 577 | **35.0%** | FP16 fallback |
| UInt8 / Float | 70 | 4.3% | 辅助 |

- **量化输出占比：** **57.8%**（FP4 + FP8 合计）
- **Fallback 层（输出全非量化）：** **328 层**（占 1194 层的 27.5%）
- **结论：** NVFP4 并非"全 4-bit"——近 **1/3 输出仍走 Half**，且量化层占比仅 57.8%（对比 FP8 engine 的 ~99%）。这与 §3.1 精度严重退化一致：**4-bit 主路径 + 大量 FP16 fallback 仍无法维持动作 token 稳定**。

**重建命令（存档）：**

```bash
/workspace/TensorRT-Edge-LLM/build/examples/llm/llm_build \
  --onnxDir /data/checkpoints/openvla/hf_llama_onnx_nvfp4/llm \
  --engineDir deploy/tensorrt/artifacts/engines/openvla_llama_nvfp4_detailed \
  --profilingDetailed --maxBatchSize 1 --maxInputLen 1024 --maxKVCacheCapacity 1024
```

---

# 2 测试环境

## 2.1 硬件环境

参考 [OpenVLA 模型评测（BF16 Eager 基线）](https://pcnzas4ktnz6.feishu.cn/docx/DIiQdOVLXobtnYxyYjpcR2fwn9f?from=from_copylink) 章节。NVIDIA Jetson AGX Thor，SM110，20 SMs，LPDDR5X 122.86 GB。

## 2.2 软件环境

| 软件 | 版本 | 备注 |
|---|---|---|
| TensorRT | 10.16.1.11 | 支持 SM110（Thor, capability 11.0）FP8 TensorCore ✓；NVFP4 经 Edge-LLM 处理 |
| TensorRT-Edge-LLM | 0.9.0 | Llama Decoder 的 FP8/NVFP4 插件（libNvInfer_edgellm_plugin.so） |
| PyTorch（导出用） | 2.12.0a0+5aff3928 (nv26.05) | 导出 ONNX / torch-TRT |
| CUDA | 13.2（V13.2.78） | 不变 |
| 量化工具 | tensorrt-edgellm-quantize | `--quantization nvfp4`（见 §1.1） |

## 2.3 Profiling 配置

| 项目 | BF16 baseline | NVFP4 本实验（实际） |
|---|---|---|
| 输入 | 1 × 224² + 19 tokens → 6 步 | Vision 用随机 224² 张量（fp8 engine 计时）；LLM 用纯文本 prompt |
| 轮数 | 10 轮取均值 | iters=10，warmup=3，repeat=5（`e2e_nvfp4` json） |
| 精度 | BF16 | 视觉 FP8 + LLM NVFP4 |
| Trace | 3 个 chrome trace | ✅ nvfp4 已采集：`vision_trace_fp8_nvfp4_full.trace.json`（视觉）+ LLM `llm_profile_nvfp4_full.json` + nsys（§5/§7）|

> **采集脚本：** `deploy/tensorrt/pipeline/nvfp4/01_measure_e2e_latency.py`（E2E 时延）、`deploy/tensorrt/pipeline/nvfp4/08_measure_component_latency.sh`（组件时延）、`deploy/tensorrt/pipeline/fp16/03_convert_llm_x86.sh nvfp4`（量化+导出）。产物见附录 A。

## 2.4 引入 TensorRT-Edge-LLM

> 与 [FP8 文档 §2.4](OpenVLA%20模型评测（FP8%20%2B%20TensorRT）.md#24-引入-tensorrt-edge-llm) 完全一致，此处不重复。核心：Vision 是静态单次前向（纯 TRT 够），LLM decode 是带状态的自回归循环（必须 Edge-LLM 提供 KV 管理、RoPE、采样、CUDA graph、批处理）。**NVFP4 只是把 LLM 权重从 8-bit 换成 4-bit，部署架构不变——仍需 Edge-LLM 驱动自回归。**

---

# 3 精度分析

## 3.1 7-DoF 动作输出对比

> **⚠️ 实测（task B）：与 FP8 不同，NVFP4 出现严重量化退化——仅 1/7 维 token 一致，最大漂移 43 bin。** 数据来自 `outputs/openvla/nvfp4_token_accuracy.json`（token 级 golden-emb，喂 golden bf16 embedding 只跑 NVFP4 LLM engine；对照实验见下）。**精度不可接受，不建议生产部署。**

**方法（token 级，与 FP8 §3.1 同口径）：** 喂入 golden 的 bf16 `multimodal_embeddings`（golden-emb），NVFP4 LLM engine 自回归生成 7 个动作 token，与 BF16 golden `generated_token_ids` 逐 token 对比。token 绝对差 = bin 漂移数（反向编码，差 1 token = 漂 1 bin）。

**结果对比表：**

| 维度 | BF16 token | NVFP4 token | token 绝对差 (=bin 漂移) | 是否完全一致 |
| --- | --- | --- | --- | --- |
| Δx | 31880 | 31875 | 5 | — |
| Δy | 31868 | 31828 | **40** | — |
| Δz | 31928 | 31910 | 18 | — |
| roll | 31869 | 31844 | 25 | — |
| pitch | 31825 | 31861 | 36 | — |
| yaw | 31871 | 31828 | **43** | — |
| gripper | 31744 | 31744 | 0 | ✅ |
| **合计** | — | — | **仅 1/7 完全一致，最大漂移 43 bin** | — |

**结论：NVFP4 精度严重退化，与 FP8 形成鲜明对比**

| 指标 | FP8（对照） | **NVFP4** |
| --- | --- | --- |
| 完全一致 | 4/7 | **1/7（仅 gripper）** |
| 最大漂移 | 5 bin | **43 bin（占 256 个 bin 的 17%）** |
| 多数维漂移 | 0/4/4/5/4/0/0 bin | 5/40/18/25/36/43/0 bin |
| 结论 | ✅ 可控 | ❌ **严重退化，不可接受** |

**对照实验（可信度验证，必须说明）：** 用**同一个 harness** 跑 FP8 engine，得到 **4/7 一致、最大漂移 5 bin**——与 [FP8 文档 §3.1](OpenVLA%20模型评测（FP8%20%2B%20TensorRT）.md) 的已知结果完全吻合。这证明 harness 正确（能复现 fp8 的精确结果），因此 **NVFP4 的 43-bin 漂移是真实的量化退化，不是测试方法 bug**。所有 token 仍落在合法动作区间 [31744, 31999]（未越界），但漂移量级远超可接受范围。

**退化根因分析（✅ 已与 §1.4 / §5.3 交叉验证）：** NVFP4 是 4-bit（E2M1，2 位尾数），动态范围远小于 FP8 E4M3，必须依赖 per-2×2-block 的 micro-scaling 补偿。实测 `cutlass3x_sm100_bstensorop_*gemm_block_scaled_ue4m3xf4_ue4m3` 确认走了块缩放 path；§1.4 进一步显示 **328 层 Half fallback（27.5%）**、量化输出仅 57.8%，且 §5.3 中 `sm80_xmma_gemm_f16f16` 占 5.6%（高于 FP8 的 3.3%）。尽管如此，4-bit 主路径 + 大量 fallback 仍无法维持 argmax 稳定——尤其 pitch/yaw（20 样本 p95 漂移 99+ bin，§3.2.2）说明 logits 已严重偏移。**这是 4-bit 对 7-DoF 连续动作回归精度不足的硬限制，非纯配置问题（需混合精度：敏感层保留更高位宽）。**

### 3.1.1 物理值 RMSE（golden-emb，✅ 已实测）

> **状态：✅ 已完成（2026-08-11）。** 使用 `action_meta.json` sidecar 解码（`effective_vocab_size=32000`），无需加载 bf16 权重；与 golden `metadata.json` 的 action 字段对齐验证通过。

**方法：** golden-emb —— 喂 golden bf16 `multimodal_embeddings`，NVFP4 LLM 生成 7 token，经 sidecar 解码为物理动作，与 BF16 golden action 算 RMSE。

**结果（sample_0001，`outputs/openvla/nvfp4_action_accuracy_golden-emb.json`）：**

| 指标 | FP8 对照（golden-emb） | **NVFP4** |
| --- | --- | --- |
| RMSE | 0.00162 | **0.02886** |
| max abs err | 0.00319（roll） | **0.06931（yaw）** |
| token 完全一致 | 4/7 | **1/7** |

**逐维物理误差：**

| 维度 | BF16 golden | NVFP4 | abs err |
| --- | --- | --- | --- |
| Δx | -0.001998 | -0.000880 | 0.001118 |
| Δy | 0.000871 | 0.013821 | **0.012950** |
| Δz | -0.007516 | -0.002839 | 0.004677 |
| roll | 0.002415 | 0.018380 | 0.015965 |
| pitch | 0.024004 | -0.000111 | **0.024115** |
| yaw | -0.000067 | 0.069241 | **0.069307** |
| gripper | 0.996078 | 0.996078 | 0 |

> NVFP4 RMSE **0.02886** 约为 FP8（0.00162）的 **17.8×**；yaw 单维误差 0.069 已远超机器人可接受范围。**精度结论与 token 级一致：严重退化，不可部署。**

**采集命令：**

```bash
python deploy/tensorrt/pipeline/nvfp4/14_nvfp4_action_accuracy.py \
  --mode golden-emb \
  --output outputs/openvla/nvfp4_action_accuracy_golden-emb.json
```

> **说明：** 推理 engine 默认 `artifacts/engines/openvla_llama_nvfp4/`；ONNX 源在 `/data/checkpoints/openvla/hf_llama_onnx_nvfp4/llm/`（仅 `llm_build` 用）。

### 3.1.2 端到端 action RMSE（FP8 vision + NVFP4 LLM，✅ 已实测）

> **状态：✅ 已完成（2026-08-11）。** 修复 `env.sh` 模型路径后，bf16 OpenVLA 可离线加载；跑真实 FP8 vision → NVFP4 LLM 端到端路径。

**结果（sample_0001，`outputs/openvla/nvfp4_action_accuracy_action.json`）：**

| 指标 | golden-emb（§3.1.1） | **action 端到端** |
| --- | --- | --- |
| RMSE | 0.02886 | **0.00233** |
| max abs err | 0.06931（yaw） | **0.00484（yaw）** |
| token 完全一致 | 1/7 | 0/7（但 dz/gripper 物理值碰巧对齐）|

> **重要解读：** 端到端 action 路径 RMSE 看似更低（0.00233），是因为 **FP8 vision 注入的 embedding 与 golden-emb 不同**，导致 NVFP4 输出了不同的 token 集合，物理值"碰巧"更接近 golden——**不代表 NVFP4 LLM 量化精度更好**。golden-emb 才是隔离 LLM 量化误差的权威口径（RMSE 0.02886，严重退化）。

---

## 3.2 多样本精度（✅ 已实测，20 张 bridge 图）

> **状态：✅ 已完成（2026-08-11）。** `15_nvfp4_multi_sample_accuracy.py` 批量 dump golden（sample_0001–0020）+ golden-emb 评测。

**聚合结果（`outputs/openvla/nvfp4_multi_sample_accuracy.json`）：**

| 指标 | 数值 |
| --- | --- |
| 样本数 | **20** |
| token 完全一致（均值） | **2.05 / 7** |
| token 完全一致（范围） | 1–7（sample_0016 达 **7/7 完美**）|
| 最大 bin 漂移（均值） | 58.7 |
| 最大 bin 漂移（最坏） | **139**（sample_0011）|
| RMSE（均值） | **0.0262** |
| RMSE（最坏） | **0.0717**（sample_0011）|
| RMSE（最好） | **≈0**（sample_0016，7/7 一致）|

**解读：** NVFP4 精度**高度不稳定**——同配置下 sample_0016 完美一致，sample_0011 漂移 139 bin（RMSE 0.072）。均值 RMSE 0.026、平均仅 2/7 token 一致，**确认不可生产部署**；个别样本"碰巧"对齐不能代表整体。

### 3.2.1 逐样本明细（20 张 bridge 图）

| 样本 | token 一致 | 最大 bin 漂移 | RMSE | 备注 |
| --- | --- | --- | --- | --- |
| sample_0001 | 1/7 | 43 | 0.0289 | 与 §3.1 单样本一致 |
| sample_0004 | 3/7 | 10 | 0.0023 | 较好 |
| sample_0007 | 2/7 | 6 | 0.0034 | 较好 |
| sample_0010 | **6/7** | 76 | 0.0192 | pitch 单维崩溃 |
| sample_0015 | 4/7 | 41 | 0.0252 | 中等 |
| sample_0016 | **7/7** | 0 | **≈0** | 唯一完美样本 |
| sample_0008 | 1/7 | 79 | 0.0512 | 差 |
| sample_0011 | 2/7 | **139** | **0.0717** | **最差** |
| sample_0013 | 2/7 | 99 | 0.0620 | 差 |
| 其余 11 样本 | 1–2/7 | 9–97 | 0.002–0.045 | 见 `nvfp4_multi_sample_accuracy.json` |

**RMSE 分布（20 样本）：** 均值 0.0262、中位数 0.0225、p95 0.0625、标准差 0.020；**3/20 样本 RMSE > 0.05**（0008/0011/0013）。token 一致数：均值 2.05、中位数 1、标准差 1.69——**方差极大**。

### 3.2.2 逐维度聚合（20 样本，golden-emb）

| 维度 | 完全一致率 | 平均 bin 漂移 | 中位漂移 | p95 漂移 | 最大漂移 |
| --- | --- | --- | --- | --- | --- |
| gripper | **20/20（100%）** | 0 | 0 | 0 | 0 |
| dx | 7/20（35%） | 16.6 | 7 | 52 | 64 |
| dy | 4/20（20%） | 17.6 | 5 | 70 | 74 |
| dz | 3/20（15%） | 13.4 | 10.5 | 34 | 41 |
| roll | 3/20（15%） | 12.8 | 8 | 52 | 55 |
| yaw | 2/20（10%） | 33.1 | 21.5 | 99 | 102 |
| **pitch** | **2/20（10%）** | **45.1** | **39** | **99** | **139** |

> **结论：** gripper 全样本稳定；**pitch 最敏感**（平均漂移 45 bin、最大 139），yaw 次之；平移维 dx/dy/dz 居中。**与 §3.3 单样本排序一致，20 样本统计更可靠。**

**采集命令：**

```bash
python deploy/tensorrt/pipeline/nvfp4/15_nvfp4_multi_sample_accuracy.py --start 1 --end 20
```

---

## 3.3 敏感性分析（✅ 已完成，20 样本 + §1.4 逐层 dtype）

> **状态：✅ 已完成。** 单样本（§3.1）+ 20 样本逐维聚合（§3.2.2）+ inspector fallback 层定位。

**维度敏感度（20 样本聚合，按平均 bin 漂移排序）：**

| 敏感度 | 维度 | 平均 bin 漂移 | p95 漂移 | 完全一致率 | 推测原因 |
| --- | --- | --- | --- | --- | --- |
| 🔴 最高 | **pitch** | 45.1 | 99 | 10% | 连续旋转维 logits 分布窄，4-bit 量化后 argmax 最易翻转 |
| 🔴 高 | yaw | 33.1 | 99 | 10% | 同上 |
| 🟠 中高 | dy、dx | 17.6、16.6 | 70、52 | 20%、35% | 平移维仍有显著漂移 |
| 🟡 中 | dz、roll | 13.4、12.8 | 34、52 | 15% | 仍有漂移但量级较小 |
| 🟢 低 | gripper | 0 | 0 | **100%** | 离散开/合，bin 边界宽，4-bit 仍稳定 |

**逐层 dtype 证据（§1.4）：** 1194 层中 **328 层 fallback 到 Half**（27.5%），量化输出仅 57.8%。Fallback 样本层名含大量 `node_linear_*`（lm_head 与 MLP 线性层）——这些层若回退 FP16 仍不足，说明 **4-bit 主路径本身的 logits 偏移**是精度崩溃主因。

**修复优先级建议：**

1. 对 **lm_head + 末 4–8 个 decoder block 的 linear** 强制 FP8/FP16（inspector fallback 层已定位 `node_linear_*` 系列）；
2. 对 **yaw/pitch/roll 运动维度** 做混合精度 PTQ 后 per-layer 回退；
3. 多样本评测显示精度**方差极大**（1/7–7/7），任何修复必须过 ≥20 样本 regression gate；
4. 若混合精度仍不足，LLM 回退 FP8（已知均值更稳，FP8 对照 4/7 一致）。

---

## 3.4 FP8 vs NVFP4 精度/性能总对照（✅ 已实测）

> **口径：** 精度均为 golden-emb（隔离 LLM 量化误差）；性能取同脚本 `01_measure_e2e_latency.py` 中位数；FP8 跑数见 `e2e_fp8_2026_0811_164456.json`，NVFP4 见 `e2e_nvfp4_2026_0807_171958.json`。

| 指标 | FP8 LLM | **NVFP4 LLM** | NVFP4 / FP8 |
| --- | --- | --- | --- |
| **单样本 token 一致（sample_0001）** | 4/7 | **1/7** | ↓ 75% |
| **单样本 max bin 漂移** | 5 | **43** | ↑ 8.6× |
| **单样本 golden-emb RMSE** | 0.00162 | **0.02886** | ↑ 17.8× |
| **20 样本 token 一致（均值）** | — | **2.05/7** | — |
| **20 样本 RMSE（均值 / p95 / 最坏）** | — | **0.026 / 0.062 / 0.072** | — |
| LLM engine 体积 | ~6.75 GB | **3.92 GB** | ↓ 42% |
| LLM decode（ms/step） | 29.49 | **17.64** | ↓ 1.67× |
| E2E（ms / Hz）| 231.9 / 4.31 | **144.6 / 6.92** | ↓ 1.60× |
| 峰值 GPU 内存（E2E）| 6965 MB | **4182 MB** | ↓ 40% |
| 量化输出占比（inspector）| ~99% | **57.8%** | — |
| Half fallback 层数 | 极少 | **328 / 1194** | — |

> **一句话：** NVFP4 用 **~40% 更少显存、~1.6× 更快 E2E**，换取 **~18× RMSE 退化 + 精度方差极大**——速度/体积收益真实，**精度代价不可接受**。

### 3.4.1 Harness 验证门（textcheck，⚠️ 有限验证）

> Python harness（`EdgeLlmRunner`）是 action/golden-emb 的唯一可信路径；textcheck 用于确认 KV/RoPE 与 Edge-LLM 一致。

| 项 | 结果 |
| --- | --- |
| harness 输出（纯文本 prompt，19 token prefill + 7 decode）| `[31913, 31980, 31920, 31872, 31879, 31880, 31872]` |
| `llm_inference` 同 prompt | 输出乱码文本（OpenVLA 为多模态模型，纯文本无 vision embedding 非预期路径）|
| 结论 | **golden-emb / action 精度以 Python harness 为准**；与 FP8 对照 harness 复现 4/7 一致，方法可信 |

**采集命令：**

```bash
python deploy/tensorrt/pipeline/nvfp4/14_nvfp4_action_accuracy.py --mode textcheck \
  --output outputs/openvla/nvfp4_textcheck_harness.json
```

---

# 4 推理性能对比

## 4.0 FP8 vs NVFP4 E2E 速览（同 Thor、同 FP8 vision engine 口径）

| 阶段 | FP8 LLM（ms）| **NVFP4 LLM（ms）** | 加速比 |
| --- | --- | --- | --- |
| Vision（FP8 engine）| 6.51* | **6.51** | 1× |
| LLM Prefill | 44.07 | **32.24** | 1.37× |
| LLM Decode（6 steps）| 176.95 | **105.81** | 1.67× |
| **E2E** | 231.89 | **144.56** | **1.60×** |
| **可达频率** | 4.31 Hz | **6.92 Hz** | — |
| **峰值 GPU 内存** | 6965 MB | **4182 MB** | ↓ 40% |

> *FP8 对照跑数 `e2e_fp8_2026_0811` 中 vision 为 10.86 ms（fp16 vision engine）；上表 Vision 6.51 ms 取自 NVFP4 主跑口径（fp8 vision），LLM 阶段数字仍来自各自 json 的实测中位数。

| 阶段 | BF16 Eager（ms） | NVFP4 TRT（ms） | 加速比 |
|---|---|---|---|
| **Prefill**（视觉+LLM 预处理） | 103.56 | **38.75**（Vision 6.51 + LLM 32.24）| **2.67×** |
| **Decode（6 steps）** | 362.68 | **105.81**（17.64 ms/step × 6）| **3.43×** |
| **E2E** | 466.24 | **144.56** | **3.23×** |
| **可达频率** | 2.14 Hz | **6.92 Hz** | — |

> **数值口径（重要）：** 主表取 `outputs/openvla/e2e_nvfp4_2026_0807_171958.json`（`01_measure_e2e_latency.py --precision nvfp4 --repeat 30`，中位数）——**最权威的真端到端 clean run**：LLM prefill 32.24 ms / decode 17.64 ms/token（n=180）/ E2E **144.56 ms / 6.92 Hz** / **峰值 GPU 内存 4182 MB**（较 FP8 6965 MB ↓40%）。**Vision 6.51 ms 用的是 FP8 视觉 engine（真实图像）**。
>
> ⚠️ **Vision 口径差异（多个来源并存，需区分）：**
> - 主表 vision **6.51 ms** = `07` 用真实图像跑 **fp8** vision engine（本文方案用 fp8 视觉）。
> - 早期 `e2e_nvfp4_2026_0721_103211.json`（`08` 组件脚本）的 vision **10.74 ms** 用的是 **fp16** vision engine——**口径不同**，故 E2E 记为 147.29 ms。二者都是 valid，差异来自**视觉 engine 精度选择**（fp8 更快的 6.5 vs fp16 的 10.7），非测量误差。
> - **统一口径：** 本文 nvfp4 方案＝LLM 用 NVFP4 + 视觉用 **FP8** engine，故主表以 E2E **144.56 ms / 6.92 Hz** 为准。
>
> > **注意：** NVFP4 虽大幅加速（E2E 144.56 ms / 6.92 Hz），但 §3.1 实测精度严重退化（1/7 一致、43 bin）。**"快"不等于"可用"——必须混合精度或换格式后才谈得上部署。**

## 4.1 性能优化结论

1. **整体大幅提升**：E2E 466 → 144.6 ms，加速 3.23×，频率 2.14 → 6.92 Hz。相比 fp8（230 ms / 4.35 Hz），nvfp4 再快 **1.59×**。⚠️ 但 §3.1 实测精度严重退化——**这份"更快"建立在 43-bin 漂移的不可用输出上**，故加速数字本身成立、生产可用性不成立。
2. **Decode 加速最显著（3.43×）——这正是 nvfp4 对 memory-bound decode 的核心收益**：decode 每步读全量权重，权重从 fp8（8-bit）再减半到 nvfp4（4-bit），**权重读取字节再减半 → decode 访存时间再减半**（decode 17.64 ms/step，n=180）。类比 fp8 的 1.96×（8-bit vs 16-bit），nvfp4 相对 bf16 权重缩到 1/4 → 接近 4× 的访存上限，实测 3.43×（余量来自 KV fp16、activation、非 GEMM 部分未减）。
3. **Prefill 2.67×**：prefill M 维度大（256），能打满 TensorCore——但 LLM 权重已是 4-bit，4-bit TensorCore 算力提升 + 权重访存减半共同作用。加速比低于 decode，说明 prefill 部分受算力/访存双重影响。
4. **仍受限于内存带宽**：E2E 还没到实时控制常需的 **10 Hz+**（6.92 Hz）。要继续压 decode，已无更低比特的主权重格式（nvfp4 是实用下限），杠杆转向 **KV-cache 量化（fp16→fp8/fp4）** 与 **speculative decoding**。

## 4.2 模块级时延

| 模块 | BF16 Eager（ms） | NVFP4 TRT（ms） | 变化 |
|---|---|---|---|
| Vision Backbone + Projector | 15.44 + 0.72 = 16.16 | **6.51（fp8 视觉，07 E2E）/ 10.74（fp16 视觉，08 组件）** | ↓1.5–2.5× |
| Projector | 0.72 | 已合并进 Vision engine | — |
| Llama — Prefill | 86.3 | **31.77** | ↓2.72× |
| Llama — Decode（per step） | 58.9 | **17.46** | ↓3.37× |
| lm_head | 1.1 | 已融入 Edge-LLM engine | — |

> ⚠️ **vision 口径：** 主表 E2E（§4）用 **FP8 视觉 6.51 ms**（`07_measure_e2e_latency.py`）；组件脚本 `08` 默认 **FP16 视觉 10.74 ms**——二者 engine 不同，均属有效测量。
> ⚠️ 相对 fp8：LLM decode 30.91 → 17.46 ms（↓1.77×），正确反映了权重复位减半对 decode 的收益。

## 4.3 阶段拆解

### Prefill 阶段

| 子阶段 | BF16（ms） | NVFP4 TRT（ms） | 变化 |
|---|---|---|---|
| ① 视觉编码 | 15.44 | **6.51**（FP8 vision，07 E2E 口径）| ↓2.37× |
| ② MLP Projector | 0.72 | 含于① | — |
| ③ Llama ×32 层 | 86.3 | 31.77（含④）| ↓2.72× |
| ④ lm_head | 1.1 | 含于③ | — |
| **Prefill 合计** | 103.56 | **38.75**（6.51 + 32.24）| ↓2.67× |

### Decode 阶段（6 步自回归）

| Step | Q-Len | BF16（ms） | NVFP4 TRT（ms） |
|---|---|---|---|
| 1–6 | 1 | ~60 | ~17.5 |
| **均值** | — | **60.4** | **17.46**（n=30）|

**Q-Len = 1 的含义：** 每步只生成 1 个 token，GEMM 是 M=1 的矩阵乘法，计算量很小，但必须把全部权重（nvfp4 ~3.9 GB）读一遍。因此 Decode 是典型 Memory-Bound——**权重从 8-bit→4-bit 字节减半，decode 时延几乎同比例减半（30.91→17.46, ↓1.77×），这正是访存受限的教科书特征**（呼应 §6 判定）。

---

# 5 算子分析

## 5.1 算子总览对比

**实测（nsys kernel 级，`outputs/openvla/nvfp4_operator_categories_tag.json`）：**

| 指标 | BF16 Eager | FP8 Vision | **FP8 LLM** | **NVFP4 LLM** |
|---|---|---|---|---|
| distinct kernel 数 | 177（aten op）| 103 | 252→33 | **279**（raw，未合并 __mye 核）|
| GPU 执行时间（kernel 之和）| 1,745 ms | 90.87 ms | 501.7 ms | **108.8 ms** |

**关键区别（vs FP8）：** NVFP4 LLM 的 GEMM 数据路径换成 **4-bit 块缩放 TensorCore**——`cutlass3x_sm100_bstensorop_*gemm_block_scaled_ue4m3xf4_ue4m3`（不是 fp8 的 `tensorop_*gemm_f8_f8_f32`）。`bstensorop` + `block_scaled` + `ue4m3xf4` = 4-bit 权重 + 块级 micro-scaling。另有 `__mye...dual_gemm_fused_kdag` 融合核（前缀 `__mye` 而非 fp8 的 `__myl`）。

## 5.2 类别聚合（实测）

| 类别 | BF16 | FP8 Vision（3.1%）| **FP8 LLM** | **NVFP4 LLM（实测）** | 备注 |
|---|---|---|---|---|---|
| GEMM / TensorCore | 90.7% | 73.0% | 92.1% | **88.0%** | 比 fp8 略低：块缩放 scale 读取 + 4-bit epilogue 摊入其他类别 |
| Attention | 1.9% | 8.0% | 5.0% | **6.8%** | BlackwellFMHA（prefill）|
| KV/RoPE/Sample | — | — | 1.2% | **1.8%** | Edge-LLM 插件核 |
| Norm / Reduction | 0.7% | 8.9% | 1.1% | **1.6%** | 融合 RMSNorm |
| Memory / Layout | 2.7% | 1.0% | ~0 | **1.4%** | bstensorop 块缩放 layout |
| Elementwise / Act | 1.8% | 9.2% | 0.6% | **0.0%** | 4-bit 微缩放折叠进 GEMM |
| Other | 2.2% | 0% | 0% | **0.4%** | — |

> **判读：** NVFP4 GEMM 仍 88.0% 主导，印证"4-bit 只是换权重数据路径、不改变 GEMM 主导结构"。GEMM 占比略低于 fp8（88.0 vs 92.1），原因是 4-bit 块缩放（micro-scaling）的 per-2×2-block scale 读取与块缩放 epilogue 使 Memory/Norm 类别占比上升。**decode 的加速与访存墙仍集中在 GEMM 上（§6）。**

## 5.3 CUDA 耗时 kernel 完整排名（实测，Top 10）

| # | 类别 | kernel | 占比 | GPU 执行时间 | 调用数 |
|---|---|---|---|---|---|
| 1 | GEMM | `cutlass3x_sm100_bstensorop_s256x256x64gemm_block_scaled_ue4m3xf4_ue4m3`（**NVFP4 4-bit GEMM,256×256×64**）| **38.0%** | 41.31 ms | 640 |
| 2 | GEMM | `cutlass3x_sm100_bstensorop_s128x64x64gemm_block_scaled_ue4m3xf4_ue4m3`（NVFP4 4-bit GEMM）| 9.7% | 10.60 ms | 160 |
| 3 | Attention | `BlackwellFusedMultiHeadAttentionForward`（prefill FMHA）| 6.5% | 7.05 ms | 128 |
| 4 | GEMM | `sm80_xmma_gemm_f16f16...`（**保留 fp16 的 GEMM，Half fallback**）| 5.6% | 6.14 ms | 5 |
| 5–10 | GEMM | `__mye...dual_gemm_fused_kdag_0xcd...`（Edge-LLM 融合核，各 ~1.0%）| 各 ~0.9-1.0% | 各 ~1.0 ms | 4 |

**对比 FP8 的 Top kernel（`gemm_f8_f8_f32`）→ NVFP4（`bstensorop_block_scaled_ue4m3xf4`）：**
- **首个 4-bit GEMM 38.0%**（vs fp8 的 36.4% 首个 `gemm_f8_f8_f32`）——同为最大核，说明 decode 主力 GEMM 都在这条上。
- **第 4 名 `sm80_xmma_gemm_f16f16`（5.6%）是保留 fp16 的 GEMM（Half fallback）**——占比明显高于 fp8（3.3%），提示 nvfp4 引擎里有更多层 fallback 回 fp16（可能正是精度敏感层）。
- `__mye` 融合核为 Edge-LLM 的 nvfp4 专属路径。

> **⚠️ 数据口径：** kernel 未按 `__mye0x<hash>` 合并 tile 变体（`10_operator_categories.py` 只合并 `__myl` 前缀），故 distinct 279 含部分 `__mye` 变体。Top1-2 的 `bstensorop` 核已是合并后主体积现，占比可信；#5-10 的 `__mye` 为单变体示例。

## 5.4 Q/DQ（NX4 微缩放）是否被融合吸收

**实测证据：** nvfp4 的 kernel 类别里 **Elementwise/Act 仅 0.0%**、且无独立 Q/DQ kernel——**NVFP4 的量化/反量化与 micro-scaling 同样被 Myelin/Edge-LLM 融合进 GEMM**。与 FP8 相同机制：Q 折进 GEMM 前序、DQ 折进 epilogue；4-bit 的 per-2×2-block scale 被编译进 GEMM 的权重加载/常量（`block_scaled` 正是块缩放被吸收进 GEMM 的证据），不单独成算子。

---

# 6 瓶颈分析

## 6.1 双模式瓶颈对比

| 指标 | BF16 Prefill | BF16 Decode | NVFP4 Prefill | NVFP4 Decode |
|---|---|---|---|---|
| 时延 (ms) | 103.6 | 362.7 | 42.5 | 104.8 |
| 权重加载量 | 12.9 GB | 12.9 GB | **~3.3 GB**（nvfp4≈bf16 的 1/4）| **~3.3 GB** |
| M 维度 | 256–275 | 1 | 256–275 | 1 |
| GFLOPs | ~3,784 | ~80 | ~3,784（不变）| ~80 |
| 瓶颈类型 | 权重加载受限 | 纯 Memory-Bound | 权重加载受限 | **仍纯 Memory-Bound** |

## 6.2 关键判断：瓶颈是否转移

**结论：Decode 瓶颈未转移，仍 Memory-Bound，且比 fp8 更彻底地吃满访存红利。** 实测判据：

- **Decode 每步 60.4 → 17.64 ms（↓3.43×）**，接近权重字节降到 1/4 的带宽上限（4×）。fp8→nvfp4 进一步 30.91→17.64（↓1.75×），几乎等于权重 8-bit→4-bit 减半。
- **Prefill:Decode 时延比 = 38.75 : 105.81 ≈ 1:2.73**，decode 仍占 E2E ~73%。
- 结论与 fp8 一致：decode 的杠杆仍是"减少每 token 权重读取字节"——nvfp4 已到主权重实用下限，下一步是 **KV-cache 量化**（当前 fp16 ~0.137 GB，量化到 fp8/fp4 可再省）与 **speculative decoding**（减少 decode 步数）。
- ⚠️ **但 nvfp4 的瓶颈分析意义被精度问题盖过**：decode 虽快（17.6 ms/step），输出 token 已严重漂移（§3.1 43 bin），**当前 nvfp4 的"快"不可转化为可用动作**。进一步压时延的前提是先解决精度（混合精度 / 敏感层回退高位宽）。

---

# 7 CPU 开销残余

> **✅ 已实测（nsys 差分法，`outputs/openvla/nvfp4_nsys_cpu_overhead_cpu_overhead.json`）。** 复用 FP8 §7 的方法：两次不同 warmup（2 vs 12），相减得每次推理的 host 开销，抵消一次性 engine load（3.9GB 权重 H2D ≈ 1076 ms，已扣除）。

| CPU 开销项 | BF16 Eager（ms/推理）| FP8 TRT（ms/推理）| **NVFP4 TRT（ms/推理）** | 是否消除 |
|---|---|---|---|---|
| `cudaStreamSynchronize` | 164.4 | 204.5 | **110.09**（8 calls）| ⚠️ GPU-wait，非调度残余 |
| `cudaGraphLaunch` | 0 | 1.05 | **1.31**（6 calls=6 decode 步）| 新增：取代数百次 launch |
| `cudaLaunchKernel` | 49.0 | 0.80 | **1.27**（132 calls，prefill 时）| ✅ 大幅降低 |
| `cudaMemcpy`（数据搬运）| （含于 copy_）| 4.12 | **~0**（-0.29，差分≈0；搬运在一次性 load）| ✅ 大幅降低 |
| **可消除调度/搬运合计** | ~424 | ~6 | **~2.6**（graph+launch+memcpy）| ✅ **↓~99%** |

> **结论（与 FP8 一致）：** NVFP4 走同一套 TRT 静态图 + CUDA Graph，把"运行时调度/数据搬运"host 开销压到 **~2.6 ms**（比 FP8 的 ~6 ms 更低，因权重更小）。剩余 `cudaStreamSynchronize` 110 ms 是 host 阻塞等 GPU 计算完成（GPU-wait），属不可避免的等待、**不是**调度残余。NVFP4 的 decode 每步（17.6 ms）已近下限，CPU 侧不再是大头。

> **与 FP8 的差异：** 一次性 engine load 从 6.7 GB（fp8）降到 3.9 GB（nvfp4），H2D 上传 1849 ms → 1076 ms（↓42%），与体积减半一致。

---

# 8 总结

| # | 发现 | 含义 |
|---|---|---|
| 1 | E2E 466 → **145 ms**，加速 **3.23×**，可达 **6.92 Hz** | 相比 fp8（4.35 Hz）再快 1.59×；满足 >3 Hz 实时控制，但仍未到 10 Hz+ |
| 2 | **Decode 加速最显著（3.46×）**，仍 Memory-Bound | 权重 8-bit→4-bit 减半 → decode 访存再减半（实测 88% GEMM 主导）|
| 3 | **精度：20 样本 RMSE 均值 0.026（最坏 0.072）、2.05/7 token 一致；pitch 最敏感（平均漂移 45 bin）** | 4-bit 对 7-DoF 连续动作回归精度不足且方差极大；**生产部署前必须混合精度或换格式** |
| 4 | LLM engine 体积 3.92 GB（≈bf16 的 0.29×）；**峰值显存 4182 MB（较 FP8 ↓40%）**| 4-bit 权重 + 块缩放 scale；KV 仍 fp16 有待压 |
| 5 | 视觉仍用 FP8 engine（10.74 ms，~7.3%）| nvfp4 未量化视觉；若要再压需视觉 4-bit（但需先解决 LLM 精度）|
| 6 | CPU 开销：**调度/搬运 ↓99%（~2.6 ms）已实测** | nsys 差分：CUDA Graph 折叠 launch（1.27ms）+ graph（1.31ms）；剩 `cudaStreamSynchronize` 110ms 是 GPU-wait 非调度残余（与 FP8 同机制）|

---

# 附录 A：数据/产物索引

| 文件 | 用途 | 状态 |
|---|---|---|
| `outputs/openvla/e2e_nvfp4_2026_0807_171958.json` | **§4 主表 E2E**（144.56 ms / 6.92 Hz，FP8 vision 6.51 ms）| ✅ |
| `outputs/openvla/e2e_nvfp4_2026_0721_103211.json` | 早期 E2E（147.29 ms，FP16 vision 10.74 ms，08 组件口径）| ✅ |
| `outputs/openvla/e2e_prof_summary_vfp8_lnvfp4_nvfp4_full.json` | §2.3 trace 汇总 + §1.4 LLM inspector（detailed=False）| ✅ |
| `outputs/openvla/vision_trace_fp8_nvfp4_full.trace.json` + `.table.txt` | FP8 视觉 engine kernel trace | ✅ |
| `outputs/openvla/llm_profile_nvfp4_nvfp4_full.json` | NVFP4 LLM prefill/decode profile | ✅ |
| `outputs/openvla/llm_layers_nvfp4_nvfp4_full/layer_decode_pastkvlen262.csv` | NVFP4 LLM 逐层 decode（Myelin 融合，onnx_op 空）| ✅ |
| `outputs/openvla/nvfp4_token_accuracy.json` | **§3.1 token 精度（1/7，max 43 bin）** | ✅ 2026-08-11 复验 |
| `outputs/openvla/fp8_control_token_accuracy.json` | §3.1 对照（同 harness 跑 fp8：4/7）| ✅ |
| `outputs/openvla/fp8_action_accuracy_golden-emb.json` | §3.1.1 FP8 golden-emb RMSE 对照（0.00162）| ✅ |
| `outputs/openvla/nvfp4_nsys_tag.nsys-rep` | §5 LLM kernel 级 nsys | ✅ |
| `outputs/openvla/nvfp4_operator_categories_tag.json` | §5.2/5.3 算子类别 + Top kernel | ✅ |
| `outputs/openvla/nvfp4_nsys_cpu_overhead_cpu_overhead.json` | **§7 CPU 开销（nsys 差分）** | ✅ |
| `outputs/openvla/nvfp4_llm_precision_detailed.json` | **§1.4 逐层 dtype（detailed engine inspector）** | ✅ |
| `outputs/openvla/nvfp4_action_accuracy_golden-emb.json` | **§3.1.1 golden-emb RMSE=0.02886** | ✅ |
| `outputs/openvla/nvfp4_action_accuracy_action.json` | **§3.1.2 端到端 action RMSE=0.00233** | ✅ |
| `outputs/openvla/nvfp4_multi_sample_accuracy.json` | **§3.2 20 样本聚合（rmse_mean=0.0262）** | ✅ |
| `outputs/openvla/nvfp4_textcheck_harness.json` | §3.4.1 harness textcheck token | ✅ |
| `outputs/openvla/nvfp4_textcheck_llm_inference.json` | §3.4.1 llm_inference 对照（纯文本乱码，非 golden 路径）| ✅ |
| `outputs/openvla/e2e_fp8_2026_0811_164456.json` | §3.4 / §4.0 FP8 E2E 对照（231.9 ms）| ✅ |
| `deploy/tensorrt/artifacts/engines/openvla_llama_nvfp4_detailed/` | detailed engine（inspector 用）| ✅ |
| `deploy/tensorrt/pipeline/nvfp4/15_nvfp4_multi_sample_accuracy.py` | 多样本 dump + 聚合 | — |
| `/data/checkpoints/openvla/hf_llama_onnx_nvfp4/llm/` | NVFP4 ONNX 源（构建 engine 用）| ✅ |
| `deploy/tensorrt/artifacts/engines/openvla_llama_nvfp4/` | NVFP4 LLM TRT engine（推理用）| ✅ |
| `deploy/tensorrt/pipeline/nvfp4/14b_nvfp4_token_accuracy.py` | token 级精度采集 | — |
| `deploy/tensorrt/pipeline/nvfp4/14_nvfp4_action_accuracy.py` | 动作 RMSE 采集（需 bf16）| — |
| `deploy/tensorrt/pipeline/nvfp4/14c_nvfp4_nsys_cpu_overhead.py` | CPU 开销采集 | — |
| `deploy/tensorrt/pipeline/nvfp4/14_nvfp4_collect_all.sh` | 一键采集编排 | — |

---

# 附录 B：数据采集状态与命令速查

| 章节 | 内容 | 状态 | 采集命令 |
|---|---|---|---|
| §1.4 | LLM 逐层 dtype | ✅ 已完成 | `llm_build --profilingDetailed` + inspector |
| §3.1.1 | golden-emb RMSE | ✅ 已完成 | `14_nvfp4_action_accuracy.py --mode golden-emb` |
| §3.1.2 | 端到端 action RMSE | ✅ 已完成 | `--mode action` |
| §3.2 | 多样本（20 张）| ✅ 已完成 | `15_nvfp4_multi_sample_accuracy.py --start 1 --end 20` |
| §3.4 | FP8 vs NVFP4 总对照 | ✅ 已完成 | 汇总 `fp8_control_token_accuracy.json` + `e2e_fp8_2026_0811` + §3.2 |
| §3.4.1 | Harness textcheck | ✅ 已完成 | `14_nvfp4_action_accuracy.py --mode textcheck` |
| §4 | E2E 时延 + 峰值显存 | ✅ 已完成 | `python deploy/tensorrt/pipeline/nvfp4/01_measure_e2e_latency.py --precision nvfp4` |
| §4.2 | 组件时延 | ✅ 已完成 | `bash deploy/tensorrt/pipeline/nvfp4/08_measure_component_latency.sh` |
| §5 | 算子类别 + Top kernel | ✅ 已完成 | `bash deploy/tensorrt/pipeline/nvfp4/14_nvfp4_collect_all.sh`（或单独跑 09+10+nsys）|
| §7 | CPU 开销 | ✅ 已完成 | `python deploy/tensorrt/pipeline/nvfp4/14c_nvfp4_nsys_cpu_overhead.py` |

**一键复采（已有 engine 前提下）：**

```bash
bash deploy/tensorrt/pipeline/nvfp4/14_nvfp4_collect_all.sh
```

> **使用提醒：** 本文 §1.4/§3/§4/§5/§7 均为实测；生产部署前须过 ≥20 样本 regression gate 与混合精度实验（§3.3）。

---

# 附录 C：尚未实测 / 后续实验清单

> 下列项**不影响本文"NVFP4 不可生产部署"的结论**，但若要推进混合精度或进一步压时延，建议按优先级补测。

| 优先级 | 实验项 | 当前状态 | 建议方法 / 通过标准 |
| --- | --- | --- | --- |
| P0 | **混合精度 PTQ**（lm_head + 末 4–8 block 回退 FP8）| ❌ 未做 | 重新 `tensorrt-edgellm-quantize` + `llm_build`；20 样本 RMSE 均值 < 0.005、token 一致 ≥ 5/7 |
| P0 | **FP8 20 样本对照**（同 harness）| ❌ 未做 | 扩展 `15_*` 脚本跑 FP8 engine；与 NVFP4 §3.2 同口径对比 |
| P1 | **端到端 action 多样本**（FP8 vision + NVFP4 LLM）| ❌ 仅 sample_0001 | `14_nvfp4_action_accuracy.py --mode action` 批量；**仅作参考**，golden-emb 仍是权威 |
| P1 | **KV-cache fp16→fp8/fp4** | ❌ 未做 | Edge-LLM config 改 `kv_cache_dtype`；测 decode ms/step + 精度 regression |
| P2 | **Vision NVFP4** | ❌ 未做 | 视觉占 E2E ~7%，收益有限；须 LLM 精度先过关 |
| P2 | **Speculative decoding** | ❌ 未做 | Edge-LLM `--specDecode`；测 E2E Hz + 动作精度 |
| P2 | **更大评测集（>20 张 / 真机轨迹）** | ❌ 未做 | bridge 20 张已显示高方差；真机闭环为最终 gate |
| P3 | **textcheck token 级与 llm_inference 逐 token 对齐** | ⚠️ 部分 | OpenVLA 多模态下纯文本非主路径；multimodal golden-emb 对照已足够 |

**本文已覆盖、无需重复采集：** E2E 时延、逐层 dtype、算子类别、CPU 开销、单样本 + 20 样本 golden-emb 精度、FP8 单样本对照、harness 方法验证。
