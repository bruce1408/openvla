# OpenVLA 模型评测（FP8 \+ TensorRT）

OpenVLA FP8 \+ TensorRT 在 Thor 上的 E2E 推理时延为 230\.0 ms（\~4\.35 Hz），相比 BF16 Eager baseline（466 ms）加速 2\.03×。

---

# 1 量化方案

## 1\.1 方案选择

|决策点|方案|
|---|---|
|量化方式|**PTQ**（Post\-Training Quantization）|
|精度格式|**FP8 E4M3**（权重 \+ 激活）|
|量化粒度|**per\-tensor**（默认），敏感层 **per\-channel**|

## 1\.2 逐模块量化策略

|模块|量化策略|实测结果|
|---|---|---|
|Vision Backbone \+ Projector|FP8 PTQ（Vision \+ Projector 合并为单个 engine）<br>|engine 1453\.84 MB（fp16）→ **741\.39 MB（fp8）**。<br>占 E2E 仅 **3\.1%**（7\.18/230\.0）|
|MLP Projector|FP8 PTQ<br>|已与 Vision 融进同一 \.engine|
|Llama Decoder 32 层|**FP8 PTQ**|**收益核心**：decode 每步 60\.4→30\.9 ms（\~1\.96×）。<br>engine size 6755 MB|
|lm\_head|FP8 PTQ|已融进 Edge\-LLM engine|
|Embedding|FP16（engine 内）|Edge\-LLM config: <br>kv\_cache\_dtype=fp16，<br>embedding 为 FP16 \[32064, 4096\]|

## 1\.3 实测量化落地情况（Vision engine，engine inspector）

**验证方法：** Vision engine 用 `trtexec --profilingVerbosity=detailed` 构建，再用 Python `tensorrt.EngineInspector` 读取逐层 `Format/Datatype`。（LLM engine 的逐层精度见 §1\.4。）

- 549 层，输出 datatype `{FP8: 469, Half: 177}` → **FP8 覆盖 72\.6% 输出**（469 / 646 个输出张量）

- engine `.plan` 1453\.84 MB（fp16）→ 741\.39 MB（fp8），−49\.0%

## 1\.4 实测量化落地情况（LLM engine，detailed 重建后）

**为什么要单独重建：** 生产 LLM engine（`openvla_llama_fp8/`）由 Edge\-LLM `llm_build` 默认构建，**未开 detailed verbosity**——对它跑 inspector 只返回 970 个层名字符串、读不出 `Format/Datatype`。故用 `llm_build --profilingDetailed` 单独重建 `openvla_llama_fp8_detailed/`（仅供 inspect，不影响生产 engine 与时延），才能读出逐层精度。

- **读出 938 层的逐层数据类型**（"结构化层" = 带 datatype 字段的层；生产版只有层名、读不出）。输出 `{FP8: 256, Half: 641, UInt8: 64, Float: 6}`，输入 `{Half: 646, FP8: 448, Int32: 128, Float: 68, Int64: 2}`——其中 `Int/UInt/Float` 是索引、KV 记账等辅助张量，真正做矩阵乘的只有 FP8 和 Half。

- **判断 FP8 覆盖多少计算，要看"输入 FP8 张量数"，而不是"输出 FP8 占比"。** 因为 FP8 GEMM 的数据流是 **FP8 进 → FP32 累加 → Half 出**：
    - 输出只有 **26\.5% 是 FP8**（256/967）——偏低且误导：结果都写成了 Half，但乘法确实在 FP8 上跑。
    - 输入有 **448 个 FP8 张量**，才是真实覆盖：**32 层 × 每层 7 个大 GEMM（q/k/v/o \+ gate/up/down）× 2 个操作数 = 448**，即每层的大矩阵乘都在 FP8 上跑。
    - 其余保持 Half 的（残差流 / norm / softmax / KV）按标准 LLM 量化留高精度；对比 Vision（层间直接保持 FP8、72\.6% 输出即 FP8），LLM 的 FP8 更克制。

- **注意：** 此 detailed engine 仅用于精度审查；时延 / 算子测量仍用不带 detailed 的生产 `openvla_llama_fp8/`（二者算子与时延一致，只差 verbosity 元数据）。

---

# 2 测试环境

## 2\.1 硬件环境

参考[OpenVLA 模型评测（BF16 Eager 基线）](https://pcnzas4ktnz6.feishu.cn/docx/DIiQdOVLXobtnYxyYjpcR2fwn9f?from=from_copylink)**1\.2 软件环境配置** 章节；

## 2\.2 软件环境

|**软件**|**版本**|**备注**|
|---|---|---|
|TensorRT|**10\.16\.1\.11**<br>|支持 SM110（Thor, capability 11\.0）FP8 TensorCore ✓|
|TensorRT\-Edge\-LLM|**0\.9\.0**|Llama Decoder 的 FP8 plugin（libNvInfer\_edgellm\_plugin\.so）|
|PyTorch（导出用）|2\.12\.0a0\+5aff3928 \(nv26\.05\)|导出 ONNX / torch\-TRT|
|CUDA|13\.2（V13\.2\.78）|不变|
|cuBLASLt|随 CUDA 13\.2<br>|FP8 GEMM 实际由 CUTLASS cutlass3x\_sm100\_tensorop\_\*gemm\_f8\_f8\_f32 kernel 承担|

## 2\.3 Profiling 配置

|**项目**|**BF16 baseline**|**FP8 本实验**|
|---|---|---|
|输入|1 × 224² \+ 19 tokens → 6 步<br>|Vision 用随机 224² 张量；LLM 用 32\-token 纯文本 prompt|
|轮数|10 轮取均值|Vision 10 轮 / LLM 10 轮（warmup=5）|
|精度|BF16|FP8 E4M3|
|Trace|3 个 chrome trace|Vision 1 个 chrome trace \+ operator table|

---

## 2\.4 引入 TensorRT\-Edge\-LLM

**Vision 是"静态单次前向"，LLM decode 是"带状态的自回归循环"**

|**维度**|**Vision（DINOv2\+SigLIP\+Projector）**|**Llama Decoder**|
|---|---|---|
|输入形状|固定 \(1,6,224,224\)|每步序列长度增长（prefill 262 → decode 逐 token \+1）|
|执行方式|**单次 forward，跑完即止**|自回归 N 步循环，每步把上一步的 token 喂回去|
|KV\-cache|无|必须：逐步增长、需原地写入/复用（否则每步重算全历史）|
|专用算子|无（都是标准 conv/gemm/norm）|RoPE、带 KV 的 paged/XQA 注意力、top\-k 采样、seq\_len 记账|
|生成循环/采样|无|有（argmax/top\-k 选 token \+ 追加，直到 EOS 或长度上限）|

TensorRT 不具备以下 3 点：

1. **KV\-cache 管理**——分配、逐步增长、原地写入、复用；

2. **LLM 专用融合核**——RoPE、写 KV、paged/XQA 注意力、采样等（这些不是 TRT 的原生 layer）；

3. **自回归运行时**——编排"生成一步 → 更新 KV → 再生成"的循环，并把每步小 kernel 用 CUDA graph 重放。

**TensorRT\-Edge\-LLM 可以补齐上面 TensorRT 的短板，**它 = TensorRT（底层 GEMM/kernel 引擎）\+ **LLM 插件库**（`libNvInfer_edgellm_plugin.so`，提供上述专用核）\+ **C\+\+ 生成运行时**（KV 管理、decode 循环、CUDA graph、批处理）。所以，**Vision 无这些需求 → 纯 TRT 足矣；Llama decoder 每一样都要 → 用 Edge\-LLM。**

# 3 精度分析

## 3\.1 7\-DoF 动作输出对比

对比bf16 推理结果和 fp8 推理结果；

|维度|BF16 token|FP8 token|token 绝对差|
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

- token 序列：FP8 `[31880, 31868, 31924, 31864, 31829, 31871, 31744]` vs BF16 golden `[31880, 31868, 31928, 31869, 31825, 31871, 31744]`。

- 4/7 维 token 完全相同（Δx/Δy/yaw/gripper，零误差）；其余 3 维（Δz/roll/pitch）各漂 4–5 个 bin。

- 全部 token 落在合法动作区间 `[31744, 31999]`，没有越界或乱码。

---

# 4 推理性能对比

## 4\.1 E2E 时延

|阶段|BF16 Eager（ms）|FP8 TRT（ms）|加速比|
|---|---|---|---|
|**Prefill**|103\.56|**44\.54**（Vision 7\.18 \+ LLM 37\.36）|**2\.33×**|
|**Decode（6 steps）**|362\.68|**185\.46**（30\.91 ms/step × 6）|**1\.96×**|
|**E2E**|466\.24|**230\.00**|**2\.03×**|
|**可达频率**|2\.14 Hz|**4\.35 Hz**|—|

> - FP8 Prefill = Vision 7\.18 ms \+ LLM 37\.36 ms，详细拆分见 §4\.3。
> 
> 时延稳定性（n=10，均值）：
> 
> - Vision: **7\.18 ms**（max\-min 波动 1\.59 ms）
> 
> - LLM Prefill: **37\.36 ms**（max\-min 波动 5\.77 ms）
> 
> - LLM Decode/tok: **30\.91 ms**（max\-min 波动 1\.89 ms，三阶段中最稳定）
> 
> 



## 4\.2 模块级时延

|**模块**|**BF16 Eager（ms）**|**FP8 TRT（ms）**|**变化**|
|---|---|---|---|
|Vision Backbone \+ Projector|15\.44 \+ 0\.72 = 16\.16|**7\.18（合并 engine）**|**↓2\.25×**|
|Projector|0\.72|已合并进 Vision engine|—|
|Llama — Prefill|86\.30|**37\.36**|**↓2\.31×**|
|Llama — Decode（per step）|58\.9|**30\.91**|**↓1\.91×**|
|lm\_head|1\.10|已融入 Edge\-LLM engine|—|

## 4\.3 阶段拆解

### Prefill 阶段

|**子阶段**|**BF16（ms）**|**FP8 TRT（ms）**|**变化**|
|---|---|---|---|
|① 视觉编码|15\.44|**7\.18（含②）**|↓2\.15×|
|② MLP Projector|0\.72|含于①|—|
|③ Llama ×32 层|86\.30|**37\.36**（含④）|↓2\.31×|
|④ lm\_head|1\.10|含于③|—|
|**Prefill 合计**|**103\.56**|**44\.54**|**↓2\.33×**|

### Decode 阶段

6 步自回归

|**Step**|**Q\-Len**|**BF16（ms）**|**FP8 TRT（ms）**|
|---|---|---|---|
|1|1|63\.2|\~30\.9|
|2|1|60\.0|\~30\.9|
|3|1|59\.9|\~30\.9|
|4|1|59\.9|\~30\.9|
|5|1|59\.8|\~30\.9|
|6|1|59\.9|\~30\.9|
|**均值**|—|**60\.4**|**30\.91**（mean；p50 30\.74 / p99 31\.87）|

---

# 5 算子分析

## 5\.1 算子总览对比

|**指标**|**BF16 Eager**|**FP8 Vision**|**FP8 LLM**|
|---|---|---|---|
|distinct kernel 数|177（aten op）|103 raw → 78 合并|252 raw → 33 合并|
|GPU 工时（kernel 之和）|1,745 ms|90\.87 ms（10 步=9\.09/步）|501\.7 ms（多 pass 之和）|
|**E2E 实际耗时**|**466 ms**|—|**230\.0 ms（全链路）**|

## 5\.2 算子类别聚合对比

|**类别**|**BF16**|**FP8 Vision（3\.1%）**|**FP8 LLM（\~97%）**|**备注**|
|---|---|---|---|---|
|GEMM / TensorCore|90\.70%|73\.00%<br>|92\.10%<br>|LLM GEMM 占比与 BF16 整模型 90\.7% 高度吻合；已全走 cutlass3x\_sm100\_\.\.\.\_f8\_f8\_f32 FP8 TensorCore|
|Attention|1\.90%|8\.00%|5\.00%|LLM 侧为 prefill 的 BlackwellFusedMultiHeadAttention（decode 的 XQA 更小）|
|KV/RoPE/Sample|—|—|1\.20%|LLM runtime 插件核（RoPE\+写 KV、seqlen 记账、topK 采样、embedding lookup）——LLM 特有，Vision 无|
|Norm / Reduction|0\.70%|8\.90%|1\.10%|融合的 RMSNorm（\_\_myl\_\.\.\.Mean\.\.\.Sqrt\.\.\.Div）；Vision 占比高是因其分母只有 vision|
|Elementwise / Act|1\.80%|9\.20%|0\.60%|含被融合吸收的 Cast（即 Q/DQ）、SiLU/GELU|
|Memory / Layout|2\.70%|1\.00%|\~0|Myelin 把 layout 融进 GEMM epilogue|
|Q/DQ（新增）|0%|折叠入 Cast|折叠入 Cast|量化算子，已经融合到算子里了|
|其他|2\.20%|0%|0%|LLM 侧全部命中类别|

总结： 模型量化 \+ TensorRT 部署并没有改变模型的算子结构，GEMM 主导 GPU 执行时间（decode 的真正瓶颈是权重访存，见 §6），Norm/激活合计仅占 \~3%。**算子内部的数据路径在执行量化的时候改变了**：BF16 走 `nvjet_sm110_tst_*gemm_f16_f16_f16`（FP16×FP16→FP16），FP8 走 `cutlass3x_sm100_tensorop_*gemm_f8_f8_f32`（E4M3×E4M3→FP32 累加）。Attention 5%、norm/激活/KV 插件合计 \~3%——**decode 的加速与瓶颈都集中在这 92% 的 GEMM 上**。

## 5\.3 CUDA 耗时 kernel 完整排名

|**\#**|**类别**|**kernel（合并后）**|**占比**|**GPU 执行时间**|**调用数**|
|---|---|---|---|---|---|
|1<br>|GEMM|cutlass3x\_sm100\_tensorop\_s256x256x32gemm\_f8\_f8\_f32\_f16\_f16（FP8 GEMM，256×256 tile）|36\.40%|182\.59 ms|832|
|2|GEMM|\_\_myl\_Fc\_0x33c3fcd6\.\.\.tensorop256x256（融合 GEMM 模板 A，多 tile 合并）|20\.20%|101\.50 ms|1248|
|3|GEMM|\_\_myl\_Fc\_0xd8d4a5ac\.\.\.tensorop256x256（融合 GEMM 模板 B，多 tile 合并）|17\.40%|87\.47 ms|416|
|4|GEMM|\_\_myl\_Fc\_0x64bea4ca\.\.\.tensorop256x256（融合 GEMM 模板 C，多 tile 合并）|6\.60%|33\.00 ms|416|
|5|Attention|BlackwellFusedMultiHeadAttentionForward（prefill FMHA）|5\.00%|24\.99 ms|416|
|6|GEMM|cutlass3x\_sm100\_tensorop\_s128x128x32gemm\_f8\_f8\_f32\_f16\_f16（FP8 GEMM，128×128 tile）|5\.00%|24\.94 ms|160|
|7|GEMM|sm80\_xmma\_gemm\_f16f16（保留 fp16 的 GEMM，Half fallback）|3\.30%|16\.50 ms|14|
|8|Norm|\_\_myl\_AddCastMulMeanAddSqrtDivMulCastMulCast（融合 RMSNorm）|1\.00%|4\.94 ms|806|
|9|KV/RoPE|trt\_edgellm::applyRopeWriteKVSplitQKVKernel（RoPE\+写 KV 插件核）|0\.50%|2\.43 ms|416|
|10|Act|\_\_myl\_SiluMulMulCast（融合 SiLU 激活）|0\.40%|1\.78 ms|416|

> **总结**
> 
> 1. **GEMM 绝对主导**：top10 里 #1–#4、#6、#7 都是 GEMM，合计 ≈ 89%（整个 LLM 的 GEMM 类为 92\.1%，见 §5\.2）——印证 §6 的 Memory\-Bound 结论（时延 ∝ 权重字节）。
> 
> 2. **Attention 5\.0%**（#5）——这 5% 几乎全来自 **prefill**：它要对 262 个 prompt token 做一次全序列（262×262）注意力，计算量大，走融合核 `BlackwellFusedMultiHeadAttention`（FMHA）。**decode** 每步只是 1 个新 query 对整段 KV\-cache 的注意力（1×262），走单\-query 注意力核 XQA（表中 #19 `kernel_mha`），单步极小、合计仅 0\.1%。
> 
> 3. **RMSNorm 1\.0%**（#8）——已被 `__myl_AddCastMulMeanAddSqrt...` 融合，开销可忽略。
> 
> 4. **KV/RoPE/采样 ≈ 1\.2%**（#9 起的 `trt_edgellm::` 插件核，即 §5\.2 的 KV/RoPE/Sample 类）——LLM 特有的 runtime 核，Vision 侧没有。
> 
> 

# 6 瓶颈分析

## 6\.1 双模式瓶颈对比

|**指标**|**BF16 Prefill**|**BF16 Decode**|**FP8 Prefill**|**FP8 Decode**|
|---|---|---|---|---|
|时延 \(ms\)|103\.6|362\.7|**44\.54**|**185\.46**|
|权重加载量|12\.9 GB|12\.9 GB|**\~6\.5 GB**（FP8 权重≈BF16 半）|**\~6\.5 GB**|
|M 维度|256–275|1|**256–275**（量化不改变）|**1**（不变）|
|GFLOPs|\~3,784|\~80|**\~3,784**（计算量不变）|**\~80**|
|Arithmetic Intensity|\~290 FLOP/byte|\~6\.0 FLOP/byte|**\~570**（字节减半→AI 翻倍）|**\~12\.0**|
|瓶颈类型|权重加载受限|纯 Memory\-Bound|**Compute\-bound**（AI \> ridge）|**仍纯 Memory\-Bound**|

> **Arithmetic Intensity 计算（Decode）：**
> 
> - 权重读取: 6\.5 GB（FP8，32 层）
> 
> - KV\-cache 读取: 32 层 × 2 × 4096 elem × 2 bytes × 262 pos ≈ 0\.137 GB
> 
> - 总访存: 6\.64 GB
> 
> - AI = 80 GFLOPs / 6\.64 GB = **12\.0 FLOP/byte**
> 
> - **Thor FP8 ridge point** = 10\.7 TFLOPS / 100 GB/s ≈ **107 FLOP/byte**
> 
> - 
> 
> - Decode AI 12\.0 \<\< ridge 107 → **Memory\-Bound 确认**（离脊点还差 \~9×）。
> 
> - Prefill AI 570 \> ridge 107 → **Compute\-bound**（FP8 下 prefill 已转为算力受限）。
> 
> 

## 6\.2 关键判断：瓶颈是否转移

**结论：Decode 瓶颈未转移，仍为 Memory\-Bound。** 实测判据：

- **Decode 每步 60\.4 → 30\.91 ms（↓1\.96×）**，显著下降，但——

- **下降幅度（****2×）几乎等于权重字节减半幅度（****2×）**。这正是 memory\-bound 的判定证据：当时延 ∝ 权重字节数、而与算力无关时，说明瓶颈在"把权重从内存读进来"。

- **Prefill:Decode 时延比 = 44\.54 : 185\.46 ≈ 1:4\.16**，与 BF16 的 1:3\.5 相比 decode 占比略升。远未接近"转移信号"的 1:1。

- **AI = 12\.0 \<\< ridge 107**，量化 6× 影响不改变判断。

- 30\.91 ms 仍在 30 ms 阈值附近，**未转移信号**。

**对硬件/编译器同事的含义：** FP8 在 decode 上的收益直接来自权重带宽减半。要继续压 decode，杠杆仍是**减少每 token 的权重读取字节**——nvfp4（再半）、权重\-only 更激进量化、或 KV\-cache 量化，而不是堆算力。

## 6\.3 总结

|\#|发现|含义|
|---|---|---|
|1|GEMM 占比：BF16 整模型 90\.7% / FP8 Vision 73\.0% / FP8 LLM 92\.1%|Vision 里 GEMM 占比天然低于整模型（分母只有 vision）；FP8 GEMM 已全走 `cutlass*gemm_f8_f8_f32`|
|2|Decode 瓶颈**未转移**，仍 Memory\-Bound|AI 12 \<\< ridge 107；时延下降 2× ≈ 权重字节减半 2×|
|3|7\-DoF 精度损失（Vision\-only）：RMSE = **0\.00162**|远小于 0\.01 容差，Vision FP8 量化精度可接受|
|4|CPU 开销残余：无法采集|需 nsys profile 补充|
|5|TRT 自动融合 vs 手动 compile：Myelin 融合粒度更强|GEMM\+bias\+激活\+Cast 全塞进单 CUTLASS kernel epilogue|
|6|Q/DQ 节点开销：**已被融合吸收**|无独立 Q/DQ kernel，折进 `Cast`|
|7|E2E 230\.0 ms，加速 2\.03×|可达 4\.35 Hz，满足实时控制（\>3 Hz）|

---

# 附录 A：Trace 文件索引

|文件|用途|
|---|---|
|`outputs/openvla/vision_trace_fp8_glm_final_clean.trace.json`|**FP8 Vision Chrome/Perfetto trace（推荐）**：单次迭代、仅 CUDA，重叠事件少、Perfetto import 警告最少|
|`outputs/openvla/vision_trace_fp8_glm_final.trace.json`|同上的 10 步完整版（kernel 分布更有代表性，但 Perfetto 会报较多 `slice_spill` 警告，非致命）|
|`outputs/openvla/vision_trace_fp8_glm_final.table.txt`|FP8 Vision operator table（按 CUDA time 排序，§5 数据源）|
|`outputs/openvla/fp8_trt_breakdown_glm_final.raw.json`|FP8 Vision \+ LLM 时延实测汇总（n=10，§4 数据源；含 operator\_table 文本）|
|`outputs/openvla/llm_layers_fp8_<tag>/layer_decode_pastkvlen262.csv`|FP8 LLM 逐层 decode profiling（由 `09_prof_trace_e2e.py` 生成；原 `/tmp` 路径为临时产物、已清除，需重跑）|

**使用方式：** 在浏览器打开 `https://ui.perfetto.dev` 加载 `vision_trace_fp8_glm_final_clean.trace.json` 查看 Vision 的 kernel 级 timeline；operator table / CSV 用文本或 Excel/pandas 查看逐层时延。
**关于 `slice_spill` import 警告：** TensorRT 推理是异步多 CUDA stream，kernel/memcpy 在时间上会真实重叠，torch.profiler 导出的 chrome trace 保留了这种重叠，Perfetto 会把它们标为 `slice_spill_overlapping_complete_event`——**属信息性提示、非致命，timeline 仍完整可看**（点掉报错面板即可）。想完全无警告可改用旧版查看器 `edge://tracing` / `about:tracing`。
**注：** Vision 的 chrome trace 由 `09_prof_trace_e2e.py`（torch.profiler 包 TRT engine）生成；LLM 是 Edge\-LLM C\+\+ runtime，torch.profiler attach 不上，**没有 chrome trace**——LLM 的算子分布用 nsys（§5）而非 Perfetto。

# 附B： Myelin 背景知识

- **Myelin 是 TensorRT 内部的"代码生成器 / 算子融合引擎"**。TRT builder 接收 ONNX 计算图后，由 Myelin 负责把相邻算子融合、并生成最终的 CUDA kernel 代码，打包进 `.engine` / `.plan`。它生成的 kernel 名统一带 `__myl_` 前缀——即文档里到处出现的 `__myl_Fc`、`__myl_CastMulAdd...Mean...Sqrt...Div`、`__myl_AddCastMulMeanAddSqrtDivMulCastMulCast` 等命名来源。

- **"被 Myelin 融合吸收"**（§5\.2）：指某个算子**不再作为独立 CUDA kernel 存在**，而是被折进相邻融合 kernel 的内部步骤。本文特指 Q/DQ（量化/反量化）：

    - 原始 ONNX 图里 Q/DQ 是独立节点（`fp16→fp8`、`fp8→fp16` 各占一个 kernel）；

    - Myelin 编译时把 **Q 折进 GEMM 的权重加载/前序**，把 **DQ 折进 GEMM 的 epilogue**（FP8×FP8→FP32 累加后顺手转 Half），二者都不另起 kernel；

    - 结果在 engine inspector 里**看不到名为 **`Quantize`**/**`Dequantize`** 的独立节点**，只能看到 datatype 边界（FP8↔Half 切换处）和融合 kernel 名里的 `Cast` 子串。

- **关系澄清**：不是"engine 被 Myelin 吸收"，而是"engine 是 Myelin 生成的"。流程为 `ONNX 图（含独立 Q/DQ 节点）→ TRT builder（Myelin 做融合+代码生成）→ .engine/.plan（全是 __myl_* 融合 kernel，Q/DQ 已消失为独立 kernel）`。这也是 FP8 部署能"减字节却不增算子开销"的关键原因（§5\.2）。

- **类比**：类似 C 编译器把 `a = b*2 + c` 优化成单条 `lea` 指令——源码两个操作（乘、加），编译后融合成一条。Myelin 对"GEMM \+ bias \+ 激活 \+ Cast\(Q/DQ\)"做同样的事，全部塞进单个 CUTLASS kernel 的 epilogue（§5\.2）。

