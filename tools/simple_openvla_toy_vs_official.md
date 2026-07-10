# Toy OpenVLA 与官方 OpenVLA 对比文档

本文档对比 `simple_openvla_toy.py`（教学用简化实现）与本仓库中 OpenVLA 官方实现（`prismatic/` 与 HF 移植版 `modeling_prismatic.py`），说明简化版替换了哪些部分，以及它们之间的对应关系。

---

## 1. 整体算法结构（相同）

两者高层流程完全一致，这是 toy 版抓对的核心：

```
RGB 图像 + 语言指令 → 视觉/文本特征 → 融合 → 动作 token → 反离散化 → 连续 7-DoF 动作
```

区别只在于每个模块的“实现载体”，而非算法主线。

---

## 2. 逐组件对比

### 2.1 视觉编码器

| 维度 | Toy 代码 | 官方 OpenVLA |
|---|---|---|
| 实现 | `TinyResNet16Backbone`，8 个残差块（`simple_openvla_toy.py:180`） | DINOv2 + SigLIP 双 ViT 融合（`prismatic/models/backbones/vision/dinosiglip_vit.py:21`） |
| 输出 | 全局池化成**单个向量** `[B, 128]`（`:216`） | **patch token 序列** `[B, 256, 2176]`，两个 backbone 沿特征维 cat（`dinosiglip_vit.py:142`） |
| 层选取 | 最后池化层 | 取**倒数第二层**的 patch tokens（`dinosiglip_vit.py:63`） |

**关键差异**：官方是双 backbone 拼接 + 保留 patch 序列（256 个 token 进入 LLM）；toy 版简化成单 CNN + 全局池化成一个向量。这是最大的结构性简化。

### 2.2 语言编码器

| 维度 | Toy 代码 | 官方 OpenVLA |
|---|---|---|
| 实现 | `TinyTextEncoder` = Embedding + 单层 GRU（`simple_openvla_toy.py:220`） | Llama-2-7B（`llama2-7b-pure`，`prismatic/models/backbones/llm/llama2.py:24`） |
| 分词 | 空白正则分词，词表 128（`SimpleTokenizer`，`:54`） | Llama-2 SentencePiece BPE，词表 32000 |
| 句子表示 | GRU 最后隐藏状态 `[B, D]` | 自回归 decoder，不产生单一“句向量” |

**本质概念差异**：官方 Llama-2 既是编码器又是解码器（动作由它自回归生成）；toy 的 GRU 只做编码，动作由单独分类头产出。

### 2.3 多模态融合（最核心的概念替换）

| 维度 | Toy 代码 | 官方 OpenVLA |
|---|---|---|
| 方式 | `cat([图像向量, 文本向量])` + 两层 MLP（`simple_openvla_toy.py:246`） | projector 把图像 patch 投影进 LLM token 空间，再作为 token 拼进序列 |
| 融合位置 | 特征向量层面直接拼接 | **token 序列层面**：`cat([BOS, 图像tokens, 文本tokens])`（`prismatic/models/vlms/prismatic.py:389`） |
| projector | 无（直接 MLP 融合） | `FusedMLPProjector`（`prismatic/util/nn_utils.py:37`），2176 维 → 4096 维 |

**关键差异**：官方把每个图像 patch 变成与文本 token 同维（4096）的 token，插在 BOS 之后、文本之前，整条序列丢进 Llama 做自注意力。toy 的 `fusion` MLP 对应官方 `projector` + LLM 自注意力融合这一整块，被极大简化。

### 2.4 动作离散化 ActionTokenizer

| 维度 | Toy 代码 | 官方 OpenVLA |
|---|---|---|
| bins | 256 均匀分箱（`simple_openvla_toy.py:98`） | 256 均匀分箱（`prismatic/vla/action_tokenizer.py:31`） |
| 区间 | `[-1, 1]` | `[-1, 1]` |
| bin 中心 | `bin_centers`（`:113`） | `bin_centers`（255 个，`action_tokenizer.py:32`） |
| **bin ↔ 词表** | **独立分类头**，bin 索引直接是类别（`action_head`，`:253`） | **映射到 Llama 词表最后 256 个 token**：`token_id = vocab_size - bin`（`action_tokenizer.py:36`） |

**关键差异**：toy 用 `Linear` 分类头直接预测 bin 索引；官方把动作 bin 复用为 LLM 词表尾部 256 个 token，让 Llama 像生成文字一样“生成动作”。数学上等价（都是 256 类分类），但载体不同。分箱实现细节：toy 用 `scaled * bins` 取整（`:129`），官方用 `np.digitize`（`action_tokenizer.py:41`），边界 off-by-one 处理略有不同，但思路一致。

### 2.5 反归一化 unnormalize

| 维度 | Toy 代码 | 官方 OpenVLA |
|---|---|---|
| 公式 | `0.5*(a+1)*(q99-q01)+q01`（`simple_openvla_toy.py:149`） | **完全相同**（`prismatic/models/vlas/openvla.py:94`） |
| q01/q99 来源 | demo 里 =-1/1，恒等变换 | 来自各数据集 `norm_stats`（config.json 内 20+ 数据集统计） |
| mask | 无 | per-dim mask，夹爪维（index 6）不做反归一化（`openvla.py:97`） |

**差异**：公式一致；官方多一个 mask，夹爪维不参与仿射反归一化。

### 2.6 推理解码

| 维度 | Toy 代码 | 官方 OpenVLA |
|---|---|---|
| 方式 | 分类头一次性输出，`argmax` 所有维（`simple_openvla_toy.py:296`） | 自回归 `generate`，逐 token 生成，`max_new_tokens=action_dim`（`openvla.py:79`） |

**差异**：toy 一次前向并行出 7 维；官方 Llama 自回归逐个吐出 7 个 action token（贪心解码）。

### 2.7 Prompt 模板（几乎一致）

- Toy：`In: What action should the robot take to {instruction}?\nOut:`（`simple_openvla_toy.py:361`）
- 官方：`In: {msg}\nOut: `（`PurePromptBuilder`，`base_prompter.py:36`），msg = `What action should the robot take to {instruction}?`

**差异**：官方 `Out:` 后有空格，且推理时补一个 SentencePiece 空 token（`29871`，`openvla.py:58`）对齐训练分词；toy 用简单分词器无需此步。

### 2.8 训练损失

| 维度 | Toy 代码 | 官方 OpenVLA |
|---|---|---|
| 损失 | 对 7 个动作维做交叉熵（`simple_openvla_toy.py:268`） | next-token 交叉熵，只在动作 token 上算损失 |
| mask | 无 | prompt/图像 token 全部 `IGNORE_INDEX=-100`，只对最后 `action_dim+1` 个 token 算 loss（`datasets.py:63`） |

**差异**：本质都是对动作 bin 做交叉熵。官方在整条语言序列里 mask 掉非动作位置；toy 因用独立分类头天然只有动作输出，无需 mask。

---

## 3. 对应关系速查表

| Toy 组件 | 对应官方组件 | 简化程度 |
|---|---|---|
| `TinyResNet16Backbone` | DINOv2 + SigLIP 双 ViT | 大（单 CNN 替双 ViT，向量替 patch 序列） |
| `TinyTextEncoder`(GRU) | Llama-2-7B | 大（编码器替代自回归 LLM） |
| `SimpleTokenizer` | Llama-2 SentencePiece BPE | 大 |
| `fusion` MLP | `FusedMLPProjector` + LLM 自注意力 | 大（向量拼接替 token 序列拼接） |
| `action_head`(Linear) | 复用 Llama 词表尾部 256 token | 中（等价分类，载体不同） |
| `ActionTokenizer` | `prismatic/vla/action_tokenizer.py` | 小（分箱逻辑基本一致） |
| `unnormalize` | `openvla.py` 反归一化 | 小（缺 mask） |
| `openvla_prompt` | `PurePromptBuilder` | 极小（几乎一致） |
| `predict_action`(argmax) | 自回归 `generate` | 中（并行分类 vs 自回归） |
| `loss`(交叉熵) | masked next-token CE | 小（等价，少了 mask 机制） |

---

## 4. 官方架构数据流参考

```
PIL image ──┬─ dino transform ──► DINOv2 ViT-L (倒数第二层 patches) ─┐
            └─ siglip transform ─► SigLIP SO400m (倒数第二层 patches)─┴─cat(dim=2)─► [B,256,2176]
                                                                                        │
                                                              FusedMLPProjector (→4096) │
                                                                                        ▼
instruction ─► "In: What action...?\nOut: " ─► Llama2 tokenizer ─► embed ─┐        image tokens
                                                                          ▼             │
                                          cat([BOS, image_tokens, text_tokens]) ◄───────┘
                                                                          ▼
                                                    Llama-2-7B decoder (greedy generate)
                                                                          ▼
                                          7 个 action tokens (词表最后 256 个 ID)
                                                                          ▼
                              decode → 255 bin centers → 归一化 [-1,1] → q01/q99 反归一化 → 7-DoF 动作
```

训练：交叉熵（在 Llama 内部）只对这 7 个 action token（+ 可选 EOS）计算，其余（BOS、图像 patch、prompt）用 `IGNORE_INDEX=-100` 屏蔽。

---

## 5. 官方 OpenVLA 输入输出与逐层张量尺寸

以下尺寸取自本仓库 OpenVLA-7B 的 `config.json` 与 `modeling_prismatic.py`（HF 移植版）。

### 5.1 输入输出形式

- **单张图片输入**（非连续多帧）。OpenVLA-7B 是单帧模型：每次推理输入一张 RGB 图 + 一条语言指令，输出一个 7 维动作。模型本身没有时序/多帧机制，时序靠外部控制循环反复调用实现。
- **图片尺寸**：`224 × 224`（`config.json` 的 `image_sizes: [224, 224]`，resize 策略 `resize-naive`）。
- **输入**：`RGB 图像 (224×224)` + `指令文本`。
- **输出**：`7-DoF 连续动作`（xyz 位移 3 + 姿态 3 + 夹爪 1）。

### 5.2 关键固定维度（来自 config）

| 参数 | 值 | 来源 |
|---|---|---|
| 图像分辨率 | 224×224 | `image_sizes` |
| patch 大小 | 14 | timm 模型名 `patch14` |
| patch 数 | 256（16×16） | 224 / 14 = 16 |
| DINOv2 ViT-L 特征维 | 1024 | `vit_large_patch14_reg4_dinov2` |
| SigLIP SO400m 特征维 | 1152 | `vit_so400m_patch14_siglip_224` |
| 融合视觉维 | 2176 (=1024+1152) | `modeling_prismatic.py:102` |
| LLM 隐藏维 | 4096 | Llama-2-7B |
| LLM 层数 | 32 | Llama-2-7B |
| 词表大小 | 32064（32000 pad 到 64 的倍数） | `text_config.vocab_size` |
| 动作 bins | 256 | `n_action_bins` |
| 动作维度 | 7 | `norm_stats` |

### 5.3 逐层张量尺寸流转

以 `batch=1`、文本 token 数记为 `T`（prompt 约 20~30 个 token）为例。

**(1) 视觉 backbone（双 ViT 融合）**

```
输入 pixel_values:  [1, 6, 224, 224]      # 6 通道 = dino 的 3 通道 + siglip 的 3 通道 堆叠
   ├─ split(dim=1) → dino 图 [1,3,224,224] + siglip 图 [1,3,224,224]   (modeling_prismatic.py:120)
   ├─ DINOv2 ViT-L   → [1, 256, 1024]     # 取倒数第二层 patch tokens
   ├─ SigLIP SO400m  → [1, 256, 1152]
   └─ cat(dim=2) →    [1, 256, 2176]      (modeling_prismatic.py:123)
输出 patch_features: [1, 256, 2176]
```

**(2) Projector（FusedMLPProjector）**

```
输入:  [1, 256, 2176]
   ├─ fc1: 2176 → 8704 (=4×2176), GELU     (modeling_prismatic.py:140)
   ├─ fc2: 8704 → 4096, GELU               (modeling_prismatic.py:141)
   └─ fc3: 4096 → 4096                     (modeling_prismatic.py:142)
输出 projected_patch_embeddings: [1, 256, 4096]   # 每个图像 patch 变成一个 LLM token
```

**Projector 到底做了什么、为什么需要它**

核心作用一句话：**把视觉特征“翻译”成 LLM 能理解的 token 向量**，让图像 patch 可以像文字 token 一样被塞进 Llama 的输入序列。

问题背景：视觉 backbone 输出的每个 patch 是 2176 维（DINOv2 1024 + SigLIP 1152 拼接），而 Llama-2 的 token embedding 空间是 4096 维。两者维度不同、语义空间也不同（一个是“图像特征空间”，一个是“语言 token 嵌入空间”），不能直接拼接。projector 就是这两个空间之间的**可学习桥梁**。

它做两件事：

1. **维度对齐**：把 2176 维映射到 4096 维，使图像 patch 向量和文本 token 向量维度一致，能拼进同一条序列。
2. **语义对齐**：通过训练，让投影后的图像向量落在 Llama 的 token 嵌入空间里“说得通”的位置，使 LLM 能把它当作有意义的 token 来做注意力。

具体结构（`prismatic/util/nn_utils.py:37`，即 `FusedMLPProjector`，对应 config 里的 `no-align+fused-gelu-mlp`）是一个**逐 patch 独立作用的 3 层 MLP**：

```
Linear(2176 → 8704)  →  GELU  →  Linear(8704 → 4096)  →  GELU  →  Linear(4096 → 4096)
```

要点说明：

- **逐 patch 作用**：MLP 只在最后的特征维上运算，对 256 个 patch 是共享同一套权重、彼此独立处理的。所以它只改变每个 patch 的“维度和语义表示”，**不改变 patch 数量（仍是 256）**，也不在 patch 之间做信息交互（patch 间的交互留给后面 Llama 的自注意力去做）。
- **为什么先升到 8704（=4×2176）再降到 4096**：这个“先扩张后压缩”的瓶颈式设计（类似 Transformer FFN）给了投影更强的非线性表达能力，比单层线性映射更能学到跨模态的复杂对应关系。fused backbone 用 3 层（多一次 4× 扩张），单 backbone 版本只用 2 层（`Linear→GELU→Linear`）。
- **GELU 非线性**：夹在线性层之间，让映射不是简单的线性变换，从而能拟合“图像特征 → 语言语义”这种非线性关系。
- **`no-align` 的含义**：OpenVLA 没有单独的“视觉-语言对齐预训练阶段”，projector 是和整个 VLM 一起端到端训练出来的。

一句话对应关系：**projector ≈ 你 toy 代码里 `fusion` MLP 的“图像分支投影”部分**。区别在于 toy 版是把图像压成一个向量后和文本向量拼接再过 MLP；官方是把每个图像 patch 独立投影成一个 4096 维 token，保留 256 个 token 送进 LLM，真正的图文融合交给 Llama 的自注意力完成。

**(3) 序列拼接（构造 LLM 输入）**

```
文本 input_ids: [1, T]
   └─ embed → text_embeddings: [1, T, 4096]

拼接（BOS 之后插入图像 tokens）:                    (modeling_prismatic.py:383)
   cat([ text_emb[:, :1, :],           # BOS:        [1, 1,   4096]
         projected_patch_embeddings,   # 图像 tokens: [1, 256, 4096]
         text_emb[:, 1:, :] ], dim=1)  # 剩余文本:    [1, T-1, 4096]

输出 multimodal_embeddings: [1, 1+256+(T-1), 4096] = [1, 256+T, 4096]
```

**(4) Llama-2-7B decoder**

```
输入 inputs_embeds:  [1, 256+T, 4096]
   └─ 32 层 Transformer decoder（自注意力），每层输入/输出均为 [1, 256+T, 4096]
   ├─ 最后 hidden state:  [1, 256+T, 4096]
   └─ lm_head (4096 → 32064)
输出 logits:  [1, 256+T, 32064]
```

**(5) 自回归动作解码**

```
generate(max_new_tokens=7)  → 逐个吐出 7 个 action token   (modeling_prismatic.py:518)
取最后 7 个 token id:  [7]
   ├─ token_id → bin: discretized = vocab_size - token_id   (modeling_prismatic.py:522)
   ├─ bin → 归一化动作: bin_centers[bin] → [7]，范围 [-1,1]
   └─ 反归一化: 0.5*(a+1)*(q99-q01)+q01（夹爪维按 mask 跳过）  (modeling_prismatic.py:526)
输出:  [7]   连续 7-DoF 动作
```

### 5.4 端到端一图流

```
RGB 224×224 ─┬─dino──►[1,256,1024]─┐
             └─siglip►[1,256,1152]─┴─cat─►[1,256,2176]
                                            │ FusedMLPProjector
                                            ▼
指令文本 ──►[1,T]─►embed─►[1,T,4096]        [1,256,4096]  (图像 tokens)
                                │                │
                       cat([BOS, 图像 tokens, 文本 tokens])
                                ▼
                        [1, 256+T, 4096]
                                │ Llama-2-7B × 32 层
                                ▼
                        [1, 256+T, 32064]  (logits)
                                │ generate 7 步
                                ▼
                        7 个 action token → 7 维连续动作
```

**核心记忆点**：图像永远被编码成 **256 个 token（每个 4096 维）**，和文本 token 一起拼成长度为 `256 + T` 的序列喂给 Llama；Llama 再自回归吐出 **7 个动作 token**，解码成 7-DoF 动作。

---

## 6. 一句话总结

Toy 代码在**算法骨架（图文→动作 token→连续动作）、动作离散化/反归一化、prompt 模板**上与官方高度一致；主要替换集中在三个“重型组件”：

1. **双 ViT 视觉塔 → 小 CNN**（且 patch 序列被压成单向量）
2. **Llama-2 自回归 LLM → GRU 编码器 + 独立分类头**
3. **token 序列级融合 → 向量拼接 MLP**

这三处替换把一个 7B 自回归 VLM 压成可在 CPU 上运行的教学模型，同时保留了理解 OpenVLA 所需的全部关键概念。
