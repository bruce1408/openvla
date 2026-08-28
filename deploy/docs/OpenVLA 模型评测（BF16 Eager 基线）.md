# OpenVLA 模型评测（BF16 Eager 基线）

OpenVLA 在 NVIDIA Jetson AGX Thor 平台上，基于** PyTorch BF16 Eager **推理对 OpenVLA 进行端侧部署前评测分析，E2E 推理时延为 **466ms\(2\.14Hz\)。**评测覆盖输入/输出张量形状、prefill/decode 时延拆解、aten/CUDA 算子耗时统计以及算子兼容性分析。当前实验未使用模型量化技术，其结果作为后续 TensorRT/TensorRT\-LLM 或 NPU 部署优化的 baseline。

---

# 评测背景与测试环境

## **1\.1 硬件环境配置**

|**硬件类**|**配置项**|**具体参数值 / 指标**|**备注 / 状态**|
|---|---|---|---|
|**处理器与架构**|GPU 型号|**NVIDIA Thor**|Jetson agx thor|
||计算能力 \(Compute Capability\)|11|最新一代具身智能/车载算力架构|
||SM 数量|20 个|2560 CUDA Cores, 80 Tensor Cores|
||显存/内存时钟|4266 MHz|\-|
||内存位宽|256 bit|\-|
||理论峰值内存带宽|约 0\.273 TB/s|\-|
||系统内存 \(RAM\)|统一内存 LPDDR5X, **122\.86 GB**||
|**频率与功耗**|GPU GPC 时钟|**1575 MHz**（已锁到 max）|峰值 GEMM 读自 gpu-gpc-0|
||当前功耗模式|**MAXN**|稠密 BF16 规格约 259 TFLOPS（2070/8）；~256 即此档|

## 1\.2 软件环境配置

|**软件类**|**软件 / 依赖库名称**|**版本号**|**备注**|
|---|---|---|---|
|**操作系统**|OS|**Ubuntu 24\.04\.4 LTS**|Noble Numbat|
||内核版本 \(Kernel\)|Linux 6\.8\.12\-1021\-tegra|架构: `aarch64`|
|**英伟达生态**<br>|JetPack / L4T 版本|R39 \(release\) revision 2\.0|\-|
||NVIDIA 驱动版本|595\.78|\-|
||CUDA 运行时|13\.2|\-|
||cuDNN 版本|9\.2\.2\.0 \(Raw: 92200\)|\-|
|**推理与加速**|TensorRT|10\.16\.1\.11|ARM64 CUDA 13\.2 专用版|
||PyTorch|2\.12\.0 alpha|自带 BF16 原生硬件支持|
||accelerate|0\.30\.1|\-|
||执行模式|Eager \(no compile\) / SDPA → FlashAttn|\-|
|**模型与工具**|Transformers|4\.40\.1|\-|
||Tokenizers|0\.19\.1|\-|
||timm|0\.9\.10|视觉主干网络依赖|
||CUDA Tile IR|tileiras 1\.3\.0（sm\_110 支持）|见附录 D|

## 1\.3 Profiling 配置

|**项目**|**配置**|
|---|---|
|工具|torch\.profiler（CPU scheduling \+ CUDA kernel 双端采集）|
|输入|1 × 224² RGB 图像 \+ 1 × 自然语言指令（19 tokens）→ 自回归 6 步 → 7\-DoF 动作|
|轮数|**10 轮，取平均时延**（下文所有时延数据均为 10 轮均值，除非特别标注）|
|精度|BF16（全模型，无量化）|
|Trace|3 个 chrome trace：`trace_{prefill,decode,full_predict_action}.json`|
|模块 Hook|对 vision backbone / projector / lm\_head / LlamaLM 单独记录前向耗时|
|记录总数|全链路 **177** 条，Prefill 单独 122 条，Decode 单独 72 条|

# OpenVLA 模型

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=Y2FjMDJmMzA2YzAzMjg5NGJjNDNkMjYzZjMwOWYyNDBfZmRkNmI0M2JjZTFmODQyY2JlOWZhNjc3NjVjMzU0MmFfSUQ6NzY2MDQ4MzI4MDIxMDQ0NzMwMl8xNzg3Nzk1NTYzOjE3ODc4ODE5NjNfVjM)

## 模型拓扑结构

OpenVLA 基于 **Prismatic VLM** 架构，核心组件如下：

|**组件**|**骨干**|**层数**|**输出形状**|**权重体积**|**说明**|
|---|---|---|---|---|---|
|**Vision Backbone**|DINOv2 ViT\-L \+ SigLIP ViT\-S|—|`[1, 256, fused_dim]`|\~600 MB|双路提取视觉特征，经跨注意力机制融合|
|**MLP Projector**<br>|2层 Linear \+ GELU<br>|2|`[1, 256, 5120]`|\~10 MB|将视觉 token 映射到 LLM 嵌入空间|
|**Llama Decoder**<br>|32层 Transformer|32|`[1, 275, 5120]`|**\~12\.0 GB**|`hidden_dim=5120`, `num_heads=32`, `head_dim=128`|
|其中 lm\_head|Linear 5120→vocab|1|`[1, 275, vocab]`|\~250 MB|仅用序列最后一位置的输出投影到动作|
|**Action Head**|—|—|`[1, 7]`|—|从 lm\_head 输出提取 7\-DoF 动作|

## 输入与输出

OpenVLA 的输入是**图片 \+ 文本两者一起**：

|**输入**|**规格**|**说明**|
|---|---|---|
|**图片**|1 × RGB, 224×224|机器人当前视角的单帧观测——"现在看到了什么"|
|**文本**|1 条自然语言指令 \(≤19 tokens\)|任务描述，如 \*"pick up the gray ball"\*、*"open the gripper"*|

模型"看到了什么（图）"\+"要做什么（指令）"，输出下一步的 **7 维动作**：

|**输出**|**维度**|**说明**|
|---|---|---|
|xyz 位移|3|末端执行器的平移增量|
|姿态|3|旋转（欧拉角 / 6D 表示）|
|夹爪开合|1|连续值，控制 gripper 开闭程度|

> 一张图（一路图像输入 ）\+ 一句指令（任务目标）→ 一个 7\-DoF 动作（下一步怎么动）
> 
> 

在实际控制循环中，指令只需给一次，图片每步更新。模型每次输出一个动作 → 机器人执行 → 场景改变 → 拍新图 → 预测下一个动作，反复直到任务完成。

# 推理性能分析

## 模块级延迟占比统计

|**模块**|**前向次数**|**平均时延 \(ms\)**|**%E2E**|**结论**|
|---|---|---|---|---|
|Vision Backbone \(DINOv2\+SigLIP\+fusion\)|1 \(prefill\)|15\.44|**3\.4%**<br>|忽略不计<br>|
|Projector MLP|1 \(prefill\)|0\.72|**0\.2%**|忽略不计|
|lm\_head|7 \(1 pref \+ 6 dec\)|1\.10|**1\.7%**|很小|
|**Llama LM — Prefill**|**1** \(seq\_len=275\)|**86\.30**|**18\.5%**|GEMM 瓶颈（seq 并行）|
|**Llama LM — Decode \(per step\)**|**6** \(q\_len=1\)|**58\.9**|**76\.3%**|**Memory\-Bound 瓶颈**|
|**Llama LM 合计**|**7**|—|**94\.9%**|**耗时主要模块**|

> 注：以上均为 10 轮推理的均值。除 Step 1 首次写入 KV\-cache 略慢（63\.2 ms）外，Decode 各步之间波动极小（±0\.1 ms）。
> 
> 

## 端到端时延

|**阶段**|**前向次数**|**时延 \(ms\)**|**占比**|**说明**|
|---|---|---|---|---|
|**Prefill**|1|**103\.56**<br>|22\.2%<br>|视觉 backbone \+ 全序列首次前向 \(seq\_len=275\)|
|**Decode**|6|**362\.68**|77\.8%|单 token 自回归 × 6 步，KV\-cache 逐步增长|
|**E2E**|—|**466\.24**|100%|predict\_action 完整调用|
|**可达频率**|—|—|**2\.14 Hz**|—|

**结论：**10 轮推理中，E2E 时延分布在 464\.7 \~ 477\.4 ms 之间，波动范围仅 ±6 ms（\~2\.7%），表现比较稳定。

## 阶段拆解

### prefill 阶段（1 次前向，103\.56 ms）

Prefill 是单次完整前向，seq\_len=275，没有自回归循环。按执行顺序拆分为 4 个子阶段：

|**子阶段**|**时延 \(ms\)**|**占 Prefill**|**说明**|
|---|---|---|---|
|① 视觉编码（DINOv2 \+ SigLIP \+ 多模态融合）|15\.44|14\.9%|图片 \[1,3,224,224\] → 视觉 token \[1,256,fused\_dim\]|
|② MLP Projector|0\.72|0\.7%|视觉 token 投影到 LLM 空间 \[1,256,5120\]|
|③ Llama Decoder ×32 层|86\.30|83\.3%|全序列 275 tokens 并行前向，GEMM 为 M=256\-275 的大矩阵乘法|
|④ lm\_head（提取动作）|1\.10|1\.1%|序列末位 → 7\-DoF 动作输出|
|**Prefill 合计**|**103\.56**|**100%**|—|

### Decode 阶段（6 步自回归，362\.68 ms）

|**Step**|**Q\-Len**|**KV\-Len**|**平均 \(ms\)**|**备注**|
|---|---|---|---|---|
|1|1|276|**63\.2**|KV\-cache 首次写入，较后续步骤多 \~3\.2 ms（HBM 分配 \+ 初始化开销）|
|2|1|277|60\.0|—|
|3|1|278|59\.9|—|
|4|1|279|59\.9|非常稳定（±0\.1 ms，Memory\-Bound 特征）|
|5|1|280|59\.8|—|
|6|1|281|59\.9|—|
|**均值**|—|—|**60\.4**||

Step 2–6 几乎完全一致（±0\.1 ms），与 Prefill 不同，Decode 每步只生成 1 个 token（M=1）。

# 算子分析

## 模块统计

|**模块**|**数量**|**占比**|**推理开销**|
|---|---|---|---|
|**Linear**|437|56\.3%|**高** — 计算核心|
|LayerNorm|105|13\.5%|低|
|LlamaRMSNorm|65|8\.4%|低（可融合）|
|GELU|54|7\.0%|低（可融合）|
|LayerScale|48|6\.2%|极低|
|LlamaRotaryEmbedding|32|4\.1%|低|
|SiLU|32|4\.1%|低（可融合）|
|Conv2d|2|0\.3%|极低（lower→GEMM）|
|Embedding|1|0\.1%|极低|
|**有效模块合计**|**776**|**100%**|—|

## 全量算子总览

|**指标**|**值**|
|---|---|
|Profiler 记录|177 条 \(85 aten:: \+ 92 CUDA kernel / profiler\)|
|总调用次数|74,261|
|CPU 总耗时|1,211 ms|
|CUDA 总耗时|1,745 ms|

## 类别聚合

|**类别**|**全量 \(177条\) CUDA 占比**|**ATen 层面 \(85条\) CUDA 占比**|
|---|---|---|
|**GEMM / TensorCore**|**90\.7%**|**89\.4%**|
|Memory / Layout|2\.7%|2\.3%|
|Attention|1\.9%|1\.9%|
|Elementwise|1\.8%|2\.0%|
|Norm / Reduction|0\.7%|0\.6%|
|其他|2\.2%|3\.8%|

## CUDA 耗时 Top 10

|**\#**|**算子**|**调用**|**CUDA \(ms\)**|**占比**|**类别**|
|---|---|---|---|---|---|
|1|`aten::linear`|1,782|**399\.0**|22\.9%|GEMM|
|2|`aten::matmul`|1,799|**390\.1**|22\.4%|GEMM|
|3|`aten::mm`|1,575|**389\.5**|22\.3%|GEMM|
|4|`nvjet_128x8_splitK`|384|**136\.3**|7\.8%|GEMM|
|5|`nvjet_64x8_splitK`|768|**104\.3**|6\.0%|GEMM|
|6|`nvjet_512x8_splitK`|192|**68\.8**|3\.9%|GEMM|
|7|`nvjet_256x144`|160|**34\.0**|1\.9%|GEMM|
|8|`nvjet_384x144`|64|**26\.7**|1\.5%|GEMM|
|9|`aten::cat`|1,075|14\.1|0\.8%|Memory|
|10|`aten::mul`|2,092|13\.0|0\.7%|Elem|



> Top 8 全部是 GEMM，合计 \~1,549 ms（88\.9%）。Top 3 ATen GEMM 算子（linear \+ matmul \+ mm）合计 1,179 ms（67\.5%）。nvjet（cuBLASLt 为 SM110 自动调度的 TensorCore 内核，15 个 tile 变体）合计 371 ms（21\.3%）。
> 
> 

## nvjet TensorCore 内核

Thor SM110 上 cuBLASLt 选择了 15 种不同 tile 配置。splitK 模式（用于大 K 维 GEMM）占主导：

|**Tile**|**调用**|**CUDA \(ms\)**|**占比 \(nvjet 内\)**|
|---|---|---|---|
|128×8 splitK|384|136\.3|36\.7%|
|64×8 splitK|768|104\.3|28\.1%|
|512×8 splitK|192|68\.8|18\.5%|
|256×144 2cta|160|34\.0|9\.1%|
|384×144|64|26\.7|7\.2%|
|其余 10 种|543|1\.3|0\.4%|

# 双模式瓶颈对比：Prefill vs Decode

Prefill 和 Decode 都要经过同一个 32 层 Llama Decoder。核心区别在于矩阵乘法的 **M 维度**（一次处理多少个 token）：

- **Prefill**：275 个 token 一起送入，矩阵乘法 `[275, K] × [K, N]`，M=275，一次完成。

- **Decode**：逐 token 生成，矩阵乘法 `[1, K] × [K, N]`，M=1，需要 6 步自回归。

## 核心对比

|**指标**|**Prefill**|**Decode (×6)**|
|---|---|---|
|时延 (ms)|103.6 (22.2%)|362.7 (77.8%)|
|GFLOPs|3,784|80|
|达成的 TFLOPS|36.5|0.22|
|算力利用率（vs 实测稠密 BF16 峰值 **224 TFLOPS**）|**16.3%**|**0.10%**|
|seq\_len \(M\)|256–275|1|
|Arithmetic Intensity|\~240 FLOP/byte|**\~1\.0 FLOP/byte**|
|瓶颈类型|权重加载受限|纯 Memory\-Bound|

Decode 计算量仅 Prefill 的 2\.1%，但总时延是 3\.5 倍。**每一次 Decode Step ≈ 把 12\.9 GB LLM 权重从显存读一遍，然后做 M=1 的微小 GEMM**。

## 算力利用率（峰值取 224 TFLOPS）

本评测的峰值分母取 **本机实测稠密 BF16 Tensor Core：224 TFLOPS**（**MAXN**，GPU 1575 MHz，`cuBLASLt` heuristic，`2048³` 最好一档）。测试方法、公式与完整结果见 **附录 C**。

相对手册：**224 / 259 ≈ 87%**（259 = MAXN 稀疏 FP4 2070 / 8 折成稠密 BF16；常说的 **~256** 就是这一档）。模型利用率仍除以 **224**（测到的峰值），不除以 256（折算规格）。

**达成算力**只看这次推理的 FLOPs 和时延：

```
达成 TFLOPS = GFLOPs / 时延_ms
利用率     = 达成 TFLOPS / 224
```

|阶段|GFLOPs|时延|达成 TFLOPS|利用率 vs 224|
|---|---|---|---|---|
|Prefill|3,784|103.56 ms|3,784 / 103.56 ≈ **36.5**|36.5 / 224 ≈ **16.3%**|
|Decode ×6|80|362.68 ms|80 / 362.68 ≈ **0.22**|0.22 / 224 ≈ **0.10%**|

`torch.mm` 默认 kernel 在本机大约只能到 **113 TFLOPS**。若用它当峰值，Prefill 会变成 32.4%、Decode 0.2%——那是「相对 PyTorch 默认 GEMM」而不是「相对 GPU 能测到的 BF16 峰值」。本报告统一用 **224**。

**为什么 Prefill 只有 16%、Decode 接近 0：**

- Prefill 的 M 维是 256–275，Arithmetic Intensity ~240 FLOP/byte，能喂一部分 Tensor Core，但形状远小于打满 224 的大方阵，再加权重从统一内存读入，所以到不了 100%。
- Decode 每步 M=1，AI ~1 FLOP/byte，时间几乎全花在把约 12.9 GB 权重读一遍上；堆 BF16 TFLOPS 几乎减不了 Decode 时延。

```bash
python openvla/deploy/pytorch_bf16/bench_peak_bf16.py
```

## 算子调度对比

|**维度**|**Prefill**|**Decode**|
|---|---|---|
|cudaLaunchKernel 次数|1,653|1,292|
|aten::linear 次数 / CUDA|432 次 / 70\.9 ms|225 次 / 54\.6 ms|
|最大 CPU 纯开销|cudaDeviceSynchronize: 49\.3 ms|cudaDeviceSynchronize: 24\.0 ms|
|CPU/CUDA 主导比<br>|12 CPU\-dominant / 8 CUDA\-dominant \(Top 20\)|14 / 6|

## 纯 CPU 开销

以下算子的 CUDA 端耗时为 0，完全在 CPU 端执行。合计 **\~295 ms**，占总 CPU 时间的 24\.4%。

|**算子**|**CPU \(ms\)**|**调用次数**|**原因**|
|---|---|---|---|
|`cudaStreamSynchronize`|**164\.4**|19|显式同步等待 GPU 完成|
|`cudaLaunchKernel`|**49\.0**|9,539|kernel launch 调度开销|
|`cuLaunchKernelEx`|10\.6|1,782|CUDA Graph kernel 启动|
|`aten::transpose`|10\.6|4,055|纯 stride 计算|
|`aten::t`|7\.7|1,782|纯 tensor 转置|
|`aten::empty_strided`|6\.6|1,875|CPU 端内存分配|
|`cudaLaunchKernelExC`|6\.0|1,377|CUDA 内核启动|
|`aten::empty`|5\.4|1,507|CPU 端内存分配|
|其他 \(\~20 项\)|\~35|—|—|
|**合计**|**\~295**|—|占总 CPU 24\.4%|

## CPU 开销显著大于 GPU 的算子

以下 4 个算子的 CPU 端耗时远超其 CUDA 端实际工作时间，CPU 花在同步等待、发起拷贝的时间，远超 GPU 实际执行操作的时间。

|**算子**|**CPU \(ms\)**|**CUDA \(ms\)**|**CPU/CUDA 比**|**问题本质**|
|---|---|---|---|---|
|`cudaStreamSynchronize`|164\.4|0\.0|**∞**|显式同步——CPU 空等 GPU，GPU 端 0 工作|
|`aten::to`|194\.1|6\.2|**31\.6×**|dtype/device 转换——CPU 调度开销是 GPU 工作的 31 倍|
|`aten::_to_copy`|191\.3|6\.2|**31\.1×**|`to` 的实际拷贝——同上|
|`aten::copy_`|181\.3|7\.0|**25\.8×**|显式内存拷贝——CPU 发起开销远超 GPU DMA|
|**合计**|**731\.1**|**19\.4**|**37\.7×**|**浪费 731 ms CPU，GPU 仅工作 19 ms**|

## 算子融合

用 torch\.compile 来模拟融合效果：分别在两种模式下跑同一个 Decode step，对比两者的时延差，作为”融合能省多少时间”的近似参考。

|**融合模式**|**Eager \(ms\)**|**Compiled \(ms\)**|**收益**|**说明**|
|---|---|---|---|---|
|**RMSNorm \+ Linear**|0\.167|0\.110|**\+33\.8%**|省掉 norm 中间写回 HBM|
|Linear \+ SiLU \+ Mul \(MLP\)|0\.971|0\.935|**\+3\.8%**|两个 elementwise 融入 GEMM|
|Linear \+ GELU \(projector\)|0\.042|0\.071|**\-69\.3%** ⚠️|融合后退化，需调优|

以 **RMSNorm \+ Linear** 为例，对比融合前后的执行差异：

|**维度**|**Eager（未融合）**|**Compiled（融合后）**|
|---|---|---|
|**执行步骤**<br>|① RMSNorm 计算结果 → ② 写回 HBM → ③ Linear 从 HBM 读取 → ④ GEMM|① RMSNorm \+ Linear 在单 kernel 内完成|
|**HBM 往返**|2 次（写 norm 输出 \+ 读 Linear 输入）|**0 次**（中间结果留在寄存器/共享内存）|
|**延迟 \(decode M=1\)**|0\.167 ms|0\.110 ms|
|**加速比**|—|**\+33\.8%**|
|**精度影响**|—|无（数值等价）|

> 每条 Decoder Layer 有 2 个 RMSNorm（attn 前 \+ FFN 前），32 层 × 6 steps = **384 次/推理**——累计可省约 **22 ms**。
> 
> 

# 附录

## 附录 A：算子清单 Top 50

完整 177 条见 operator\_inventory\.json， **A\_aten\_operators\_with\_shapes** 字段。此处仅列 CUDA 耗时最高的 50 条。

|**\#**|**类别**|**算子**|**calls**|**CUDA ms**|**%**|
|---|---|---|---|---|---|
|1|GEMM|`aten::linear`|1,782|399\.0|22\.9|
|2|GEMM|`aten::matmul`|1,799|390\.1|22\.4|
|3|GEMM|`aten::mm`|1,575|389\.5|22\.3|
|4|GEMM|`nvjet_128x8_splitK`|384|136\.3|7\.8|
|5|GEMM|`nvjet_64x8_splitK`|768|104\.3|6\.0|
|6|GEMM|`nvjet_512x8_splitK`|192|68\.8|3\.9|
|7|GEMM|`nvjet_256x144_2cta`|160|34\.0|1\.9|
|8|GEMM|`nvjet_384x144`|64|26\.7|1\.5|
|9|Mem|`aten::cat`|1,075|14\.1|0\.8|
|10|Elem|`aten::mul`|2,092|13\.0|0\.7|
|11|GEMM|`cublas splitKreduce`|1,350|12\.5|0\.7|
|12|Mem|`CatArrayBatchedCopy`|512|9\.7|0\.6|
|13|GEMM|`aten::addmm`|207|9\.5|0\.5|
|14|Attn|`aten::sdpa`|275|8\.2|0\.5|
|15|Attn|`aten::_sdpa_flash`|275|8\.2|0\.5|
|16|Attn|`aten::_flash_fwd`|275|8\.2|0\.5|
|17|Elem|`elemwise_kernel Mul`|1,009|7\.9|0\.5|
|18|Mem|`aten::copy_`|1,669|7\.0|0\.4|
|19|GEMM|`nvjet_256x8_splitK`|6|5\.9|0\.3|
|20|Elem|`aten::add`|1,469|6\.2|0\.4|
|21|Mem|`aten::to`|2,738|6\.2|0\.4|
|22|Mem|`aten::_to_copy`|1,600|6\.2|0\.4|
|23|Attn|`flash_fwd 128×128`|192|4\.9|0\.3|
|24|Mem|`CatArrayBatched_vec`|320|3\.7|0\.2|
|25|Mem|`direct_copy unrolled`|686|4\.2|0\.2|
|26|Norm|`aten::layer_norm`|102|2\.0|0\.1|
|27|Norm|`aten::native_layer_norm`|102|2\.0|0\.1|
|28|Elem|`aten::neg`|448|1\.8|0\.1|
|29|Elem|`aten::pow`|455|1\.4|0\.1|
|30|Norm|`aten::mean`|455|2\.1|0\.1|
|31|Attn|`flash_fwd causal`|32|2\.0|0\.1|
|32|Mem|`bf16_copy vec`|903|2\.0|0\.1|
|33|GEMM|`nvjet_256x128`|27|1\.8|0\.1|
|34|GEMM|`nvjet_224x128`|27|1\.6|0\.1|
|35|RT|`Activity Buffer`|3|1\.5|0\.1|
|36|Elem|`aten::silu`|224|1\.5|0\.1|
|37|GEMM|`nvjet_128x136`|48|1\.2|0\.1|
|38|Norm|`vec_layer_norm`|102|1\.1|0\.1|
|39|GEMM|`nvjet_176x128`|27|1\.1|0\.1|
|40|GEMM|`nvjet_192x144`|1|1\.1|0\.1|
|41|GEMM|`nvjet_256x136`|24|1\.0|0\.1|
|42|Elem|`aten::rsqrt`|455|1\.0|0\.1|
|43|GEMM|`cublas splitK bias`|27|0\.9|0\.1|
|44|GEMM|`nvjet_192x144_4x1f`|24|0\.9|0\.1|
|45|Mem|`direct_copy 128w`|54|0\.8|0\.0|
|46|Mem|`aten::contiguous`|54|0\.8|0\.0|
|47|Mem|`aten::clone`|54|0\.8|0\.0|
|48|Attn|`flash_fwd 64×128`|24|0\.7|0\.0|
|49|Elem|`aten::gelu`|53|0\.7|0\.0|
|50|GEMM|`nvjet_448x128`|3|0\.7|0\.0|

## 附录 B：Trace 文件索引

|**文件**|**用途**|
|---|---|
|`trace_prefill.json`|Prefill 阶段 kernel timeline|
|`trace_decode.json`|Decode 阶段 kernel timeline|
|`trace_full_predict_action.json`|完整 predict\_action timeline|

\[trace\_decode\.json\]

\[trace\_full\_predict\_action\.json\]

\[trace\_prefill\.json\]

\[operator\_inventory\.json\]

**工具**：`https://ui.perfetto.dev` 打开加载上面json文件可以看到对应的算子执行的Trace流水线

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NjFkYjRkYTU5NzFjMTAwNGI3NTdlNmE0NTM1ZmFjNTZfYTE1ZjI3NmFlM2Q0ZGQ5ZGNhNGQ4YjM2NDczNTMzYTJfSUQ6NzY2MDQ4NDM0OTI5NjkyMTc5OF8xNzg3Nzk1NTYzOjE3ODc4ODE5NjNfVjM)

## 附录 C：BF16 峰值算力测试

脚本：`openvla/deploy/pytorch_bf16/bench_peak_bf16.cu`（`python bench_peak_bf16.py`）。环境：MAXN，GPU 1575 MHz。输入输出 BF16，累加 FP32。

### 代码里怎么统计

对每一个形状 `(M,N,K)`，cuBLASLt heuristic 最多给出 32 个算法。每个算法：warmup 10 次后，用 CUDA Event 对 40 次 `cublasLtMatmul` 取平均时间 `t`，再算

```
TFLOPS(M,N,K,algo) = (2 × M × N × K) / t / 1e12
```

（MAC=2：点积里乘、加各计 1 FLOP。）该形状保留 **算法中的最大值**；全部形状再取一次最大值，打印为 `best dense BF16`。伪代码：

```
for (M,N,K) in shapes:          # 2048³、4096³、8192³ 等 15 组
    algos = GetHeuristic(...)   # ≤32 个 kernel
    best_shape = max_algo  2MNK / mean_time(algo)
global_peak = max_shape  best_shape     → 本机为 224 TFLOPS
```

因此 224 **不是**某一次计时的平均数，也 **不是** 15 个形状的平均，而是「形状 × 算法」二维搜索里的 **max**。同一次运行里，kernel 时长 ≥0.20 ms 的子集再取 max，得到约 196（4096³），用于判断短 kernel 是否被 L2 抬高。

### 为什么正文用 224

峰值的操作定义就是上述 `global_peak`：在算力密集的 GEMM 上，允许选最快 kernel，看硬件 **能打到多高**。本机最大值出现在 `2048×2048×2048`、heuristic 列表第 0 号算法、0.077 ms → **224 TFLOPS**。

不采用同形状 `torch.mm` 的约 146：`torch.mm` 只走库的 **默认一个** 算法，统计口径是「Eager 常用路径」，不是上面的 `max_algo`。不采用 ~113：那是 `torch.mm` 的 **8192³**，形状与 2048³ 不同，不能和 224 直接比。不采用 8192 的 cuBLASLt（约 139）：那是同一套 `max_algo` 下更大、更吃带宽的点，吞吐更低，不是 `global_peak`。

224 对照手册稠密 BF16（稀疏 FP4 2070/8 ≈ 259，常写作 ~256）约为 87%。短 kernel 可能受益于 L2，故 224 是瞬时最好；更稳可报 196。本报告分母取 224。利用率 = 模型达成 TFLOPS / 224，达成值仍用解析 FLOPs / 推理时延，与峰值同一套 `2MNK`。

## 附录 D：CUDA Tile IR 工具链



> CUDA Tile IR 未在本次 Eager 模式 profiling 中使用，但 SM110 支持使其成为后续 NPU 编译器后端（Tile IR → NPU 指令映射）的潜在路径；
> 
> 

|**项目**|**配置**|
|---|---|
|后端汇编器|`tileiras`（NVIDIA \(R\) Cuda Tile IR optimizing assembler）|
|可执行路径|`/usr/local/cuda-13.2/bin/tileiras`|
|版本|release 13\.2, V13\.2\.78|
|构建日期|2026\-03\-20|
|文件大小|\~90\.5 MB|
|支持架构|sm\_80/86/87/88/89/100/103/110/120/121（含本机 Thor 的 **sm\_110**）|
|运行时库|`libnvidia-tileiras.so.595.58.03`|
|Python 前端|cuda\.tile 版本 1\.3\.0（/usr/local/lib/python3\.12/dist\-packages/cuda/tile）|



