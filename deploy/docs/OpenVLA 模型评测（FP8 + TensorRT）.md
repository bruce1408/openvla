# OpenVLA 模型评测（FP8 \+ TensorRT）

OpenVLA FP8 \+ TensorRT 在 Thor 上的 E2E 推理时延为** 230\.0 ms（\~4\.35 Hz）**，相比 BF16 Eager baseline**（466 ms）加速 2\.03×**。

---

# 1 量化方案

## 1\.1 方案选择

|**模块**|**方案**|
|---|---|
|量化方式|PTQ|
|精度格式|FP8 E4M3（权重 \+ 激活）|
|量化粒度|per\-tensor（默认），敏感层 per\-channel|

## 1\.2 逐模块量化策略

|**模块**|**量化策略**|**实测结果**|
|---|---|---|
|Vision Backbone \+ Projector|FP8 PTQ（Vision \+ Projector 合并为单个 engine）|engine 1453\.84 MB（fp16）→ 741\.39 MB（fp8）。<br>Latency 占 E2E 仅 3\.1%（7\.18/230\.0）|
|MLP Projector|FP8 PTQ|已与 Vision 融进同一 \.engine|
|Llama Decoder 32 层|**FP8 PTQ**|收益核心：decode 每步 60\.4→30\.9 ms（\~1\.96×）。engine \~6\.6 GB|
|lm\_head|FP8 PTQ|已融进 Edge\-LLM engine|
|Embedding|FP16（engine 内）|Edge\-LLM config: kv\_cache\_dtype=fp16，embedding 为 FP16 \[32064, 4096\]|

## 1\.3 Vision engine 实测量化

**验证方法：** 构建 engine 后，逐层读取 Format/Datatype 以确认 FP8 是否生效。

- 549 层，输出 datatype `{FP8: 469, Half: 177}` → **FP8 覆盖 72\.6% 输出**（469 / 646 个输出张量）

- engine `.engine` 1\.42GB（fp16）→ 0\.72GB（fp8），−49\.0%

## 1\.4 LLM engine实测量化

**验证方法：**LLM engine（openvla\_llama\_fp8/）由 Edge\-LLM  默认构建，需要单独重建 openvla\_llama\_fp8\_detailed/ 才能读出逐层精度。

- 读取 938 层数据类型（带 `Format/Datatype` 字段的层）。把所有张量按类型计数：

    - 输出：`{FP8: 256, Half: 641, UInt8: 64, Float: 6}`

    - 输入：`{Half: 646, FP8: 448, Int32: 128, Float: 68, Int64: 2}`

- 其中 `Int32 / Int64 / UInt8 / Float` 是位置索引、KV 记账等辅助张量，不参与矩阵乘；真正做计算的只有 FP8 和 Half。

---

# 2 测试环境

## 2\.1 硬件环境

参考 [OpenVLA 模型评测（BF16 Eager 基线）](https://pcnzas4ktnz6.feishu.cn/docx/DIiQdOVLXobtnYxyYjpcR2fwn9f?from=from_copylink) **1\.2 软件环境配置** 章节；

## 2\.2 软件环境

|**软件**|**版本**|**备注**|
|---|---|---|
|TensorRT|**10\.16\.1\.11**|支持 SM110（Thor, capability 11\.0）FP8 TensorCore|
|TensorRT\-Edge\-LLM|**0\.9\.0**|Llama Decoder 的 FP8 plugin（libNvInfer\_edgellm\_plugin\.so）|
|PyTorch（导出用）|2\.12\.0a0\+5aff3928 \(nv26\.05\)|导出 ONNX / torch\-TRT|
|CUDA|13\.2（V13\.2\.78）|不变|
|cuBLASLt|随 CUDA 13\.2|FP8 GEMM 实际由 CUTLASS cutlass3x\_sm100\_tensorop\_\*gemm\_f8\_f8\_f32 kernel 承担|

## 2\.3 Profiling 配置

|**项目**|**BF16 baseline**|**FP8 本实验**|
|---|---|---|
|输入|1 × 224² \+ 19 tokens → 6 步|Vision 用随机 224² 张量；LLM 用 32\-token 纯文本 prompt|
|轮数|10 轮取均值|Vision 10 轮 / LLM 10 轮（warmup=5）|
|精度|BF16|FP8 E4M3|
|Trace|3 个 chrome trace|Vision 1 个 chrome trace \+ operator table|

## 2\.4 引入 TensorRT\-Edge\-LLM

**Vision 是”静态单次前向”，LLM decode 是”带状态的自回归循环”**

|**维度**|**Vision（DINOv2\+SigLIP\+Projector）**|**Llama Decoder**|
|---|---|---|
|输入形状|固定 \(1,6,224,224\)|每步序列长度增长（prefill 262 → decode 逐 token \+1）|
|执行方式|**单次 forward，跑完即止**|自回归 N 步循环，每步把上一步的 token 喂回去|
|KV\-cache|无|逐步增长、需原地写入/复用（否则每步重算全历史）|
|专用算子|无（都是标准 conv/gemm/norm）|RoPE、带 KV 的 paged/XQA 注意力、top\-k 采样、seq\_len 记账|
|生成循环/采样|无|有（argmax/top\-k 选 token \+ 追加，直到 EOS 或长度上限）|

TensorRT 不具备以下 3 点：

1. **KV\-cache 管理：**分配、逐步增长、原地写入、复用；

2. **LLM 专用融合核：**RoPE、写 KV、paged/XQA 注意力、采样等（这些不是 TRT 的原生 layer）；

3. **自回归运行时：**编排”生成一步 → 更新 KV → 再生成”的循环，并把每步小 kernel 用 CUDA graph 重放。

**Edge\-LLM 可以补齐 TensorRT 的短板，它可以认为是 TensorRT \+ LLM 插件库（****`libNvInfer_edgellm_plugin.so`****，提供上述专用核），用 C\+\+ 生成运行时（KV 管理、decode 循环、CUDA graph、批处理）。所以，这里的场景需要使用 Edge\-LLM。**

---

# 3 精度分析

## 3\.1 7\-DoF 动作输出对比

对比 BF16 推理结果和 FP8 推理结果；

|**维度**|**BF16 token**|**FP8 token**|**token 绝对差**|
|---|---|---|---|
|Δx|31880|31880|0|
|Δy|31868|31868|0|
|Δz|31928|31924|4|
|roll|31869|31864|5|
|pitch|31825|31829|4|
|yaw|31871|31871|0|
|gripper|31744|31744|0|
|**合计**|—|—|**4/7 完全一致，最大漂移 5 bin**|

**总结：**

- 7 个动作里有 4 个完全没变：Δx、Δy、yaw 和 gripper 的 FP8 token 与 BF16 一模一样。在这个样本中，FP8 没有改变水平移动、偏航旋转和夹爪开合决策。

- 另外 3 个动作有小幅变化：Δz 相差 4 个档位，roll 相差 5 个档位，pitch 相差 4 个档位。最大偏差是 5 个档位；

- 输出格式正常：bf16 vs fp8 生成的所有 token 都位于动作区间 \[31744, 31999\]。FP8 没有生成越界 token，也没有出现无法解码的异常输出。

---

# 4 推理性能对比

|**阶段**|**BF16 Eager（ms）**|**FP8 TRT（ms）**|**加速比**|
|---|---|---|---|
|**Prefill**|103\.56|44\.54（Vision 7\.18 \+ LLM 37\.36）|**2\.33×**|
|**Decode（6 steps）**|362\.68|185\.46（**30\.91 ms/step × 6**）|**1\.96×**|
|**E2E**|466\.24|**230**|**2\.03×**|
|**可达频率**|2\.14 Hz|**4\.35 Hz**|—|

## 4\.1 E2E 时延

模型完整端到端推理总耗时由 **Prefill 预处理阶段** \+ **Decode 动作生成阶段** 两部分组成；其中 Prefill 内部又拆分为视觉编码器、LLM 文本预处理两个子模块。

FP8 量化 \+ TensorRT 方案下各子模块平均时延（10 次采样均值）：

- Vision Backbone 视觉编码: 平均 **7\.18 ms；**

- LLM Prefill 预处理：平均 **37\.36 ms；**

- LLM Decode 单 Token 生成：平均 **30\.91 ms ****/ 步****；**

**性能优化结论：**

1. 整体实时性大幅改善

    1. FP8 \+ TensorRT 端到端总时延**从 466 ms 降至 230 ms**，整体推理加速比 2\.03×，频率从 2\.14 Hz 提升至 4\.35 Hz，基本满足实时控制需求（\>3 Hz）。

2. Prefill 阶段优化幅度最高，加速 2\.33×

    1. Prefill 单次批量处理 262 个 Prompt Token，矩阵行维度充足，能够充分打满 GPU TensorCore 算力单元，释放 FP8 量化优化收益：

    - FP8 专用 TensorCore 提供近 2 倍浮点算力；

- LLM Decode 阶段加速比 1\.96x：

    - Decode 阶段每次仅生成 1 个 Token，无法充分利用 TensorCore 并行算力，FP8 算力增益无法发挥；仅受益于权重体积减半带来的内存带宽节省；

## 4\.2 模块级时延

|**模块**|**BF16 Eager（ms）**|**FP8 TRT（ms）**|**变化**|
|---|---|---|---|
|Vision Backbone \+ Projector|15\.44 \+ 0\.72 = 16\.16|**7\.18（合并 engine）**<br>|**↓2\.25×**<br>|
|Projector|0\.72|已合并进 Vision engine|—|
|Llama — Prefill|86\.3|**37\.36**|**↓2\.31×**|
|Llama — Decode（per step）|58\.9|**30\.91**|**↓1\.91×**<br>|
|lm\_head|1\.1|已融入 Edge\-LLM engine|—|

## 4\.3 阶段拆解

### Prefill 阶段

|**子阶段**|**BF16（ms）**|**FP8 TRT（ms）**|**变化**|
|---|---|---|---|
|① 视觉编码|15\.44|**7\.18（含②）**|↓2\.15×|
|② MLP Projector|0\.72|含于①|—|
|③ Llama ×32 层|86\.3|37\.36（含④）|↓2\.31×|
|④ lm\_head|1\.1|含于③|—|
|**Prefill 合计**|**103\.56**|**44\.54**|**↓2\.33×**|

### Decode 阶段

6 步自回归

|**Step**|**Q\-Len**|**BF16（ms）**|**FP8 TRT（ms）**|
|---|---|---|---|
|1|1|63\.2|\~30\.9|
|2|1|60|\~30\.9|
|3|1|59\.9|\~30\.9|
|4|1|59\.9|\~30\.9|
|5|1|59\.8|\~30\.9|
|6|1|59\.9|\~30\.9|
|**均值**|—|**60\.4**|30\.91|

**Q\-Len = 1 的含义：** 自回归解码每步只生成 1 个新 token，那么 query 长度恒为 1（拿当前 token 去查 KV\-cache）。这决定 Decode 的 GEMM 是 M=1 的矩阵乘法——每次计算量很小，但必须把全部权重（\~6\.6 GB）读一遍，因此 Decode 是典型的 Memory\-Bound 场景。

---

# 5 算子分析

## 5\.1 算子总览对比

|**指标**|**BF16 Eager**|**FP8 Vision BackBone**|**FP8 LLM**|
|---|---|---|---|
|distinct kernel 数|177（aten op）|103 raw → 78 合并|252 raw → 33 合并|
|GPU 执行时间（kernel 之和）|1,745 ms|90\.87 ms（10 步=9\.09/步）|501\.7 ms（多 pass 之和）|
|**E2E 实际耗时**|**466 ms**|—|**230\.0 ms（全链路）**|

raw 表示融合前的原始 kernel 数\(对应 ONNX 图的独立算子\)，合并表示**Myelin（附件 B 详解）** 把相邻算子拼成单个 `__myl_*` kernel 后的数量FP8 LLM 从 252 个收敛到 33 个。

## 5\.2 算子类别聚合对比

|**类别**|**BF16**|**FP8 Vision（3\.1%）**|**FP8 LLM（\~97%）**|**备注**|
|---|---|---|---|---|
|GEMM / TensorCore|90\.70%<br>|73\.00%<br>|92\.10%|LLM GEMM 占比与 BF16 整模型 90\.7% 高度吻合；<br>已全走 cutlass3x\_sm100\_…\_f8\_f8\_f32 FP8 TensorCore|
|Attention<br>|1\.90%|8\.00%|5\.00%|LLM 侧为 prefill 的 BlackwellFusedMultiHeadAttention（decode 的 XQA 更小）|
|KV/RoPE/Sample|—|—<br>|1\.20%|LLM runtime 插件核（RoPE\+写 KV、seqlen 记账、topK 采样、embedding lookup）——LLM 特有，Vision 无|
|Norm / Reduction|0\.70%|8\.90%|1\.10%<br>|融合的 RMSNorm（\_myl…Mean…Sqrt…Div）；Vision 占比高是因其分母只有 vision|
|Elementwise / Act|1\.80%|9\.20%|0\.60%|含被融合吸收的 Cast（即 Q/DQ）、SiLU/GELU|
|Memory / Layout|2\.70%|1\.00%|\~0|Myelin 把 layout 融进 GEMM epilogue|
|Q/DQ（新增）|0%|折叠入 Cast|折叠入 Cast|量化算子，已经融合到算子里了|
|其他|2\.20%|0%|0%|LLM 侧全部命中类别|

总结：模型量化 \+ TensorRT 部署并没有改变模型的算子结构，GEMM 仍然主导 GPU 执行时间（decode 的真正瓶颈是权重访存），Norm/激活合计仅占 \~3%。**算子内部的数据路径在FP8量化的时候改变了**：BF16 走 `nvjet_sm110_tst_*gemm_f16_f16_f16`（FP16×FP16→FP16），FP8 走 `cutlass3x_sm100_tensorop_*gemm_f8_f8_f32`（E4M3×E4M3→FP32 累加）。Attention 5%、norm/激活/KV 插件合计 \~3%。**decode 的加速与瓶颈都集中在这 92% 的 GEMM 上**。

## 5\.3 CUDA 耗时 kernel 完整排名

|**\#**|**类别**|**kernel（合并后）**|**占比**|**GPU 执行****总****时间**|**调用数**|
|---|---|---|---|---|---|
|1|GEMM|cutlass3x\_sm100\_tensorop\_s256x256x32gemm\_f8\_f8\_f32\_f16\_f16（FP8 GEMM，256×256 tile）|36\.40%|182\.59 ms|832|
|2|GEMM|\_\_myl\_Fc\_0x33c3fcd6…tensorop256x256（融合 GEMM 模板 A，多 tile 合并）|20\.20%|101\.50 ms|1248|
|3|GEMM|\_\_myl\_Fc\_0xd8d4a5ac…tensorop256x256（融合 GEMM 模板 B，多 tile 合并）|17\.40%|87\.47 ms|416|
|4|GEMM|\_\_myl\_Fc\_0x64bea4ca…tensorop256x256（融合 GEMM 模板 C，多 tile 合并）|6\.60%|33\.00 ms|416|
|5|Attention|BlackwellFusedMultiHeadAttentionForward（prefill FMHA）|5\.00%|24\.99 ms|416|
|6|GEMM|cutlass3x\_sm100\_tensorop\_s128x128x32gemm\_f8\_f8\_f32\_f16\_f16（FP8 GEMM，128×128 tile）|5\.00%|24\.94 ms|160|
|7|GEMM|sm80\_xmma\_gemm\_f16f16（保留 fp16 的 GEMM，Half fallback）|3\.30%|16\.50 ms|14|
|8|Norm|\_\_myl\_AddCastMulMeanAddSqrtDivMulCastMulCast（融合 RMSNorm）|1\.00%|4\.94 ms|806|
|9|KV/RoPE|trt\_edgellm::applyRopeWriteKVSplitQKVKernel（RoPE\+写 KV 插件核）|0\.50%|2\.43 ms|416|
|10|Act|\_\_myl\_SiluMulMulCast（融合 SiLU 激活）|0\.40%|1\.78 ms|416|

**总结：**

1. **GEMM 占绝对主导**：top10 里 1到4、6、7 都是 GEMM，合计 ≈ 89%（整个 LLM 的 GEMM 类为 92\.1%）。

2. **Attention 5\.0%**（5，**几乎全是 prefill 阶段的全序列注意力（262×262，用 FMHA 融合核）贡献的；decode 阶段每步只是 1 个 query 对 KV\-cache 的注意力（用 XQA 核），小到只有 0\.1%。）**

3. **RMSNorm 1\.0%**（8——已被 \_\_myl\_AddCastMulMeanAddSqrt\.\.\. 融合），开销可忽略。

4. **KV/RoPE/采样 ≈ 1\.2%**（9 起的 trt\_edgellm:: 插件核）——LLM 特有的 runtime 核，Vision 侧没有。

# 6 总结

**实测判据：** Decode 每步 从60\.4 → 30\.91 ms（↓1\.96×），下降幅度几乎等于权重字节减半幅度（2×）， FP8 在 decode 上的收益直接来自权重带宽减半。要继续压 decode，杠杆仍是**减少每 token 的权重读取字节，例如使用更低比特数据格式**nvfp4（再半）进行量化。

|**\#**|**总结**|**含义**|
|---|---|---|
|1|GEMM 占比：BF16 整模型 90\.7% / FP8 Vision 73\.0% / FP8 LLM 92\.1%|Vision 里 GEMM 占比天然低于整模型（分母只有 vision）；FP8 GEMM 已全走 cutlass\*gemm\_f8\_f8\_f32|
|2|Decode 仍 Memory\-Bound，瓶颈未转移|FP8 权重减半 → 访存时间减半 → 加速 \~2×，与算力无关。优化方向：nvfp4 / KV\-cache 量化|
|3|7\-DoF 精度损失（Vision\-only）：最大漂移 5 bin|4/7 维完全一致，其余3个维度漂移 4–5 个bin格子，全部落在合法区间内，精度可接受|
|4|TRT 自动融合 vs 手动 compile：Myelin 融合粒度更强|GEMM\+bias\+激活\+Cast 全塞进单 CUTLASS kernel epilogue|
|5|Q/DQ 节点开销：已被融合吸收|无独立 Q/DQ kernel，被缝进相邻的 GEMM 融合 kernel，Quant operator 融合到 GEMM 开头,DQuant 融合进 GEMM 结尾， epilogue\(累加完顺手转 Half\)。只在融合 GEMM 的名字里留下 Cast。 |
|6|E2E 230\.0 ms，加速 2\.03×|可达 4\.35 Hz，满足实时控制（\>3 Hz）|

---

# 附录 A：Trace 文件索引

|**文件**|**用途**|
|---|---|
|fp8\_trt\_breakdown\_final\.raw\.json|FP8 Vision  LLM 时延实测汇总|
|layer\_decode\_pastkvlen262\.csv|FP8 LLM 逐层 decode profiling|
|vision\_trace\_fp8\_final\.table\.txt|FP8 Vision operator table（按 CUDA time 排序）|
|vision\_trace\_fp8\_final\.trace\.json|FP8 Vision trace（CUDA kernel 级 timeline，Perfetto 打开）|

\[fp8\_trt\_breakdown\_final\.raw\.json\]

\[layer\_decode\_pastkvlen262\.csv\]

\[vision\_trace\_fp8\_final\.table\.txt\]

\[vision\_trace\_fp8\_final\.trace\.json\]

**使用方式：** 

- 在浏览器中打开 `https://ui.perfetto.dev`，加载 `.trace.json` 文件查看 Trace 流水线。

- CSV 用 Excel/pandas 查看逐层时延。

# 附B：Myelin 背景知识

**Myelin** 是 TensorRT 内置的图优化编译器，负责在生成 engine 时对计算图做算子融合。它会将相邻的小算子（如 GEMM \+ bias \+ 激活函数 \+ 类型转换）合并成单个 CUDA kernel，中间结果留在寄存器/共享内存中，避免反复读写显存。文档中出现的 `__myl_*` / `__mye_*` 前缀 kernel 即为 Myelin 自动生成的融合 kernel。

