# 量化 Scale：Float 乘法 vs Multiply-Shift

> **📌 一句话结论：** 开源量化工具算出 `scale` 后，经常直接用浮点乘法做缩放；硬件整数通路则常用 `multiply + shift` 近似同一个 `scale`。两边相减得到的差叫 **系数量化误差**（也称 **scale 定点近似误差**）。该误差通常远小于 INT8/INT4 的信号量化误差，所以多数工具在模拟阶段不拆；只有 bit-exact、窄位宽乘法器、或纯整数后端才需要按硬件把 `scale` 收成 `mul+shift`。

> **文档定位：** 概念说明，不绑定某一块芯片或某一次评测。回答三个问题：拆和不拆差在哪、这个差叫什么、为什么工具常常不拆。

---

# 1 问题

量化里几乎总会碰到缩放：

```text
y = x * scale
```

软件模拟（PyTorch fake-quant、很多 PTQ 校准脚本）算出 `scale` 之后，**直接用 float32/float16 乘上去**。

不少整数推理硬件没有便宜的浮点除法，会把同一个 `scale` 拆成：

```text
y ≈ (x * multiplier) >> shift
```

也就是：

```text
scale ≈ multiplier × 2^(-shift)
```

于是出现一个自然的问题：

**模拟阶段如果没用硬件真实的 multiply-shift，会不会多引入一层误差？为什么很多工具不把 `scale` 拆开？**

答案是：**会有误差，但在常见 INT8/INT4 流程里通常可以忽略；工具不拆，是因为它们模拟的是量化噪声，不是某一块芯片的定点指令。**

---

# 2 两种实现分别在算什么

## 2.1 不拆：浮点 scale

```text
y_float = x * scale_float
```

- `scale_float` 一般是 float32（有时是 FP16 / FP8 块缩放）。
- 实现简单，GPU 上跑得快。
- Fake-quant / QAT / 多数 PTQ 校准都用这种。

Fake-quant 的典型形态：

```text
x_q   = round(clamp(x / scale + zp))
x_hat = (x_q - zp) * scale          # 这里的 * scale 仍是浮点
```

它的目标是让网络「感觉到」低比特台阶，从而选好 `scale`、微调权重。  
它**不保证**和某块 NPU 的每一条整数指令 bit-exact。

## 2.2 拆：multiply + shift

```text
y_hw = (x * multiplier) >> shift
```

- `multiplier` 是定点数，常见 Q31（约 31 bit 小数）。
- `shift` 是右移位数。
- TFLite integer、gemmlowp、CMSIS-NN、部分 DSP/NPU 走这条路。
- 本质是把实数 `scale` 收成 **二元有理数**（dyadic rational）：`m / 2^n`。

TFLite / gemmlowp 的典型步骤：

```text
real_multiplier = scale_in * scale_weight / scale_out
quantized_multiplier, shift = QuantizeMultiplier(real_multiplier)
output = saturating_rounding_doubling_high_mul(acc, quantized_multiplier) >> shift
```

## 2.3 现代硬件并不都靠 mul+shift

| 通路 | Scale 怎么用 |
|---|---|
| 老整数通路（TFLite integer、CMSIS-NN、部分 DSP） | `mul + shift`（Q31 / Q15 等） |
| GPU / TensorRT / 不少数据中心推理 | int 累加后用 **FP32/FP16 scale** requantize，或先 dequant 再算 |
| NVFP4 等块缩放格式 | scale 本身就是浮点（如 FP8/E4M3 微缩放），硬件按浮点 scale 乘 |

因此：**「硬件一定把 scale 拆成 mul+shift」并不成立。**  
若量化工具默认拆成 Q31，反而可能和 GPU、TensorRT、NVFP4 **更不对齐**。

---

# 3 拆和不拆的差值叫什么

对同一个输入 `x`、同一个数学 `scale`，比较两种实现：

```text
误差 = y_float − y_hw
     = x * scale_float  −  ((x * multiplier) >> shift)
     = x * (scale_float − multiplier × 2^(-shift))
```

这个差叫：

**系数量化误差**（coefficient quantization error）

文档里若要写得更不容易误会，用：

**scale 定点近似误差**（fixed-point approximation error of the scale）

也有人叫 **量化乘数误差**（quantized multiplier error），对应 TFLite 把 `real_multiplier` 收成 Q31 的那一步。

### 3.1 为什么不叫「量化误差」

「量化误差」在工程里默认指：**把权重/激活打成低比特台阶**。  
这里量化的不是信号，而是线性变换 `y = x * scale` 里的 **系数 `scale`**。

| 名字 | 在量化什么 | 例子 |
|---|---|---|
| **信号量化误差** | 权重、激活 | INT8 / NVFP4 把连续值打成台阶 |
| **系数量化误差**（本文） | scale / multiplier | float scale vs `mul+shift` |
| **运算舍入误差** | 中间算术 | 右移时 round / truncate / 饱和 |

更广的伞名 **requantization error**（重量化误差）可以罩住后两行，但还包含舍入、饱和，范围更大。  
从「模拟器和芯片对不齐」的角度看，它属于 **simulation–hardware numerics mismatch** 的一种。

类比：

- 把激活收成 INT8：用一把很粗的尺子量长度 → **信号量化误差**
- 把「每格代表多少米」从精确小数改成「乘一个整数再右移」→ **系数量化误差**

---

# 4 这个误差有多大

以常见 Q31 multiplier 为例，float32 的 `scale` 被收成约 31 bit 定点小数，相对误差大约：

```text
|scale_hw − scale_fp| / |scale|  ≈  2^(-31)  ≈  5e-10
```

INT8 把数值压到 256 个台阶，相对误差量级大约是 `2^(-8)`（约 0.4%）。

| 误差来源 | 大致量级 |
|---|---|
| INT8 量化（权重/激活） | ~ `2^(-8)` |
| INT4 | ~ `2^(-4)` |
| Q31 multiply-shift 近似 scale | ~ `2^(-31)` |
| 舍入方式不同（RN vs truncate） | 常到 1 ULP，有时比 scale 近似更大 |
| 累加器饱和 / 溢出 | 可能直接错一个输出 |

**结论：** 系数量化误差通常远小于一个量化台阶。网络看得到的是台阶，看不到 `scale` 表示上那一点微调。

数量级例子：若 `scale ≈ 1/255 ≈ 0.0039215686`，Q31 还原后与原 scale 大约在第 9～10 位小数之后才分叉；而 INT8 一个台阶的高度就是 `scale` 本身。

---

# 5 为什么很多开源工具不拆

不是不知道硬件可以这么做，而是模拟目标和硬件并不总是同一件事。

### （1）Fake-quant 模拟的是量化噪声，不是芯片指令

校准 / QAT 要回答的是：量化成低比特之后，任务精度还够不够。  
不保证：和某块 NPU 逐元素一致。

### （2）硬件实现不统一，工具很难「拆一次适配所有芯片」

同样叫 requantize，不同芯片可能是 Q31、Q15、先左移再乘、浮点 scale、per-channel / per-block 不同位宽。  
更常见的职责划分：

```text
量化工具：用 float scale 找到「该量化成什么样」
导出/编译器：再按目标硬件把 scale 收成 mul+shift 或 FP scale
```

### （3）GPU 上仿真整数 mul+shift 成本高、收益小

每层都仿真 saturating doubling multiply、rounding-right-shift，图又慢又难维护；对准确率 / RMSE 几乎看不见收益。

### （4）先拆反而可能更错

目标若是 TensorRT、GPU、NVFP4，硬件本身就可能用浮点 scale。工具绑死 Q31，是在模拟一个**并不存在**的整数通路。

---

# 6 什么时候应该拆

下面这些场景，才值得把 `scale` 收成硬件真实形式：

1. **要 bit-exact**：芯片输出必须和模拟器逐元素一致（量产签核、对拍）。
2. **目标硬件乘法器很窄**：例如只有 16-bit multiplier，近似误差不再是 `2^(-31)`，可能到 `2^(-15)`，这时才可能进指标。
3. **纯整数 MCU/DSP 部署**：TFLite Micro、CMSIS-NN 等，导出时本来就会 `QuantizeMultiplier()`。
4. **其他误差源都已对齐**：舍入、饱和、zero-point、累加位宽都一致后，再抠 scale 表示才有意义。

做 PTQ/QAT 选 scale、看任务精度，用 float scale 通常就够。

---

# 7 真要对齐硬件，优先对齐这些

若目标是「模拟量化」贴近上板，优先对齐下面几条，收益远大于拆 scale：

1. **舍入**：round-to-nearest-even、round-half-up、还是截断。
2. **饱和**：int8 是 `[-128,127]` 还是 `[-127,127]`。
3. **累加位宽**：int32 是否会在累加中途饱和。
4. **zero-point** 是参与整数卷积，还是只在 requantize 时补。
5. **scale 粒度**：per-tensor / per-channel / per-block。
6. **先确认芯片**：到底用 float scale 还是 mul+shift，再决定要不要拆。

很多「模拟和上板对不齐」，根因是上面几条，不是 float scale。

---

# 8 小结

| 问题 | 答案 |
|---|---|
| 拆和不拆有没有差？ | 有。 |
| 这个差叫什么？ | **系数量化误差** / **scale 定点近似误差**。 |
| 要不要为此改量化工具？ | 通常不要。Q31 近似比 INT8/INT4 台阶误差小几个数量级。 |
| 为什么工具不拆？ | 在模拟量化噪声；硬件不统一；现代 GPU/TRT/NVFP4 可能本来就用浮点 scale。 |
| 何时拆？ | bit-exact、窄乘法器、纯整数后端。应由编译器/部署栈按芯片收，而不是所有量化工具一上来就拆。 |

---

# 9 建议的自检

1. 把某个 `scale` 收成 Q31，算 `|m * 2^(-s) − scale|`，再和 `scale / 256`（一个 INT8 台阶）比，看差多少倍。
2. 同一套权重：float requantize vs TFLite 风格 `mul+shift`，看输出最大绝对差是否远小于 1 个量化台阶。
3. 读 TFLite `QuantizeMultiplier` / gemmlowp `SaturatingRoundingDoublingHighMul`，对照硬件在近似什么。

参考：Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*（把实数乘数量化成定点）。DSP 教材中的 **coefficient quantization** 与本文是同一类数学问题。
