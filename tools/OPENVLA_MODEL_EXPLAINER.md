# OpenVLA 模型说明：用途、输入输出与实现流程

本文档用于说明 OpenVLA 是什么、它接收什么输入、输出什么动作，以及在本项目 `/workspace/openvla` 中是如何被加载和调用的。它面向端侧部署和评测场景，重点解释工程实现链路，而不是完整复述论文训练细节。

## 1. OpenVLA 是什么

OpenVLA 的全称可以理解为 Open Vision-Language-Action model，即开源的视觉-语言-动作模型。

普通视觉语言模型通常做的是：

```text
图像 + 文本问题 -> 文本回答
```

OpenVLA 面向机器人控制，做的是：

```text
当前相机图像 + 语言任务指令 -> 机器人下一步动作
```

也就是说，它不是只回答“机器人应该做什么”，而是直接输出可交给机器人控制器执行的低层动作向量。

在官方示例中，典型使用方式是：

```python
image = get_from_camera(...)
prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
robot.act(action, ...)
```

因此，OpenVLA 可以放在机器人控制闭环中反复调用：

```text
相机取图 -> 构造语言指令 -> OpenVLA 推理 -> 输出 action -> 机器人执行 -> 下一帧
```

如果控制闭环目标是 10Hz，则一次完整模型侧推理最好小于 100 ms；如果目标是 20Hz，则最好小于 50 ms。

## 2. 模型架构概览

本仓库中的 `openvla-7b` 是一个基于 Prismatic VLM 的 VLA 模型。它由三个主要部分组成：

```text
视觉编码器 -> 投影层 -> 语言模型主干 -> 动作 token -> 连续动作
```

更具体地说：

1. 视觉编码器负责把 RGB 图像编码成视觉特征。
2. 投影层把视觉特征映射到语言模型可以消费的 embedding 空间。
3. LLM 主干基于图像特征和文本 prompt 自回归生成 token。
4. OpenVLA 不把生成 token 当作普通文字，而是把最后若干 token 解释为动作 token。
5. `ActionTokenizer` 将动作 token 解码为归一化连续动作。
6. `predict_action()` 根据训练数据集统计量把归一化动作反归一化为真实动作输出。

官方 README 中说明，`openvla-7b` 基于 Prismatic `prism-dinosiglip-224px` VLM，视觉侧融合 DINOv2 和 SigLIP，语言模型侧基于 Llama-2。

## 3. 模型输入是什么

OpenVLA 的输入由两部分组成：视觉输入和语言输入。

### 3.1 视觉输入

视觉输入通常是一张当前时刻的 RGB 图像。

在本项目中，测试脚本里可以看到：

```python
image = Image.new("RGB", (224, 224), color=(128, 128, 128))
```

正式机器人部署时，这张图应该来自机器人相机，例如腕部相机、第三视角相机或经过选择的一路 RGB 图像。

需要注意：

- OpenVLA 官方 `openvla-7b` 使用 224px 视觉输入配置。
- 输入图片会经过 `AutoProcessor` 处理，包括 resize、crop、normalize 等步骤。
- 在当前 runtime 的 benchmark 中，默认只测单张图片输入，不测多相机融合。

对应代码位置：

- `tools/test_openvla_local.py`
- `tools/benchmark_openvla_thor.py`
- `prismatic/extern/hf/processing_prismatic.py`

### 3.2 语言输入

语言输入是一条自然语言任务指令，例如：

```text
move the robot arm forward
```

OpenVLA 推理时会把它包装成固定 prompt 格式：

```text
In: What action should the robot take to move the robot arm forward?
Out:
```

在本项目代码中对应：

```python
prompt = f"In: What action should the robot take to {instruction}?\nOut:"
```

或者 benchmark 中的：

```python
def prompt_for(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"
```

`AutoProcessor` 会同时处理文本和图片：

```python
inputs = processor(prompt, image).to(DEVICE, dtype=model_dtype())
```

处理完成后，`inputs` 里通常包含文本 token、attention mask 和图像 tensor。

## 4. 模型输出是什么

OpenVLA 输出的是机器人动作向量，而不是自然语言文本。

官方 README 明确写到：

```python
# Predict Action (7-DoF; un-normalize for BridgeData V2)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
```

也就是说，常见输出是 7 维连续动作。

可以理解为：

```text
[x, y, z, roll, pitch, yaw, gripper]
```

其中：

- `x, y, z`：末端执行器位置增量或目标变化量。
- `roll, pitch, yaw`：末端执行器姿态增量或旋转变化量。
- `gripper`：夹爪开合状态或夹爪控制量。

你之前测试输出类似：

```text
action: [ 3.37000054e-03  8.70916770e-04  5.73473754e-03  1.13558153e-02
 -6.13951550e-03 -6.67113405e-05  9.96078431e-01]
```

这就是一个 7 维动作向量。

最后一维接近 `1.0`，通常表示夹爪状态接近某个极端值。但它到底代表“开”还是“合”，需要结合具体机器人数据集和控制器约定判断，不能只看数值本身。

## 5. OpenVLA 如何把动作变成 token

OpenVLA 的一个关键设计是：它不是直接用一个回归头输出连续浮点动作，而是借用语言模型的 token 生成机制输出动作 token。

训练时，连续动作会先归一化到 `[-1, 1]`，然后离散化为若干 bin。代码在：

```text
prismatic/vla/action_tokenizer.py
```

核心逻辑：

```python
class ActionTokenizer:
    def __init__(self, tokenizer, bins=256, min_action=-1, max_action=1):
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
```

含义是：

```text
连续动作值 -> clip 到 [-1, 1] -> 分到 256 个离散区间 -> 映射成 LLM 词表末尾的 token
```

推理时，流程反过来：

```text
模型生成 action token -> decode 成 [-1, 1] 内的归一化连续动作 -> 反归一化成真实动作
```

对应代码：

```python
normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(
    predicted_action_token_ids.cpu().numpy()
)
```

## 6. predict_action 的实现流程

核心实现位于：

```text
prismatic/models/vlas/openvla.py
```

其中 `predict_action()` 做了以下事情。

### 6.1 构造 prompt

```python
prompt_builder = self.get_prompt_builder()
prompt_builder.add_turn(
    role="human",
    message=f"What action should the robot take to {instruction.lower()}?"
)
prompt_text = prompt_builder.get_prompt()
```

这一步把用户指令转换成模型训练时熟悉的对话格式。

### 6.2 文本 tokenization

```python
input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
```

语言指令被转成 token ID。

对于 Llama tokenizer，代码还会补一个特殊 token，以匹配训练时格式：

```python
if not torch.all(input_ids[:, -1] == 29871):
    input_ids = torch.cat(...)
```

### 6.3 图像预处理

```python
pixel_values = image_transform(image)
```

图像被转换成模型需要的 tensor 格式。

### 6.4 调用 generate 生成动作 token

```python
generated_ids = super(PrismaticVLM, self).generate(
    input_ids=input_ids,
    pixel_values=pixel_values,
    max_new_tokens=self.get_action_dim(unnorm_key),
    **kwargs
)
```

这里非常关键：

```python
max_new_tokens=self.get_action_dim(unnorm_key)
```

如果动作空间是 7 维，模型就生成 7 个动作 token。

### 6.5 提取最后几个 token 作为动作

```python
predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key):]
```

生成序列的最后若干 token 被当作动作 token。

### 6.6 token 解码成归一化动作

```python
normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(...)
```

此时动作仍在归一化空间，通常位于 `[-1, 1]` 附近。

### 6.7 动作反归一化

```python
action_norm_stats = self.get_action_stats(unnorm_key)
action_high = np.array(action_norm_stats["q99"])
action_low = np.array(action_norm_stats["q01"])
actions = 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low
```

这里使用训练数据集的动作统计量，把归一化动作还原成真实机器人控制量。

这就是为什么推理时必须传：

```python
unnorm_key="bridge_orig"
```

如果 `unnorm_key` 不匹配，动作尺度可能错误，机器人执行效果会明显变差，甚至危险。

## 7. 本项目 runtime 如何加载 OpenVLA

本项目的端侧 runtime 位于：

```text
./
```

关键文件包括：

```text
env.sh
runtime_env.py
tools/test_openvla_local.py
tools/benchmark_openvla_thor.py
```

### 7.1 env.sh

`env.sh` 定义运行环境，例如：

```bash
export OPENVLA_MODEL_ID="${OPENVLA_MODEL_ID:-openvla/openvla-7b}"
export OPENVLA_DEVICE="${OPENVLA_DEVICE:-cuda:0}"
export OPENVLA_ATTN_IMPLEMENTATION="${OPENVLA_ATTN_IMPLEMENTATION:-sdpa}"
export OPENVLA_UNNORM_KEY="${OPENVLA_UNNORM_KEY:-bridge_orig}"
export HF_HOME="/workspace/openvla/hf_cache"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export OPENVLA_REVISION="47a0ec7fc4ec123775a391911046cf33cf9ed83f"
```

这些配置的作用：

- `OPENVLA_MODEL_ID`：模型 ID 或本地模型路径。
- `OPENVLA_DEVICE`：推理设备。
- `OPENVLA_ATTN_IMPLEMENTATION`：attention 后端，例如 `sdpa`。
- `OPENVLA_UNNORM_KEY`：动作反归一化统计量。
- `HF_HOME`：Hugging Face 本地缓存位置。
- `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`：强制离线加载。
- `OPENVLA_REVISION`：固定模型仓库 commit，提高可复现性。

### 7.2 runtime_env.py

`runtime_env.py` 会自动 source `env.sh`，并把环境变量注入 Python 进程。

这样脚本里只需要：

```python
from runtime_env import MODEL_PATH, MODEL_REVISION
```

后续加载模型时使用：

```python
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    revision=MODEL_REVISION,
    trust_remote_code=True,
    local_files_only=True,
)
```

`local_files_only=True` 表示只从本地缓存加载，不联网下载。

### 7.3 test_openvla_local.py

这是最小功能验证脚本。它会：

1. 打印 PyTorch、CUDA、GPU 信息。
2. 加载 OpenVLA processor 和 model。
3. 构造一张 224x224 灰色图。
4. 构造一条测试指令。
5. 调用 `model.predict_action()`。
6. 打印 action。

适合用于确认：模型能不能加载、CUDA 能不能用、action 是否能正常输出。

### 7.4 benchmark_openvla_thor.py

这是性能评测脚本。它不只看能不能跑，还会统计：

- 预处理耗时。
- CPU 到 GPU 的数据搬运耗时。
- `predict_action()` 推理耗时。
- 端到端模型侧耗时。
- 显存峰值。
- p50/p90/p95/p99 尾延迟。

适合写端侧部署评测报告。

## 8. 输入输出的端到端数据流

可以把本项目中的推理链路理解成：

```text
PIL Image + instruction string
        |
        v
prompt_for / prompt 模板
        |
        v
AutoProcessor
  - 文本 tokenization
  - 图像 resize/normalize
        |
        v
inputs.to(cuda:0, bfloat16)
        |
        v
OpenVLA / PrismaticVLM
  - vision backbone
  - projector
  - LLM generate
        |
        v
action token ids
        |
        v
ActionTokenizer.decode_token_ids_to_actions
        |
        v
normalized action in [-1, 1]
        |
        v
unnormalize with bridge_orig stats
        |
        v
7-DoF continuous action
```

## 9. 为什么需要 trust_remote_code

加载 OpenVLA 时通常会看到：

```python
trust_remote_code=True
```

原因是 Hugging Face 标准 `AutoModelForVision2Seq` 不内置 OpenVLA/Prismatic 的自定义模型类。OpenVLA 仓库提供了自定义的：

```text
configuration_prismatic.py
modeling_prismatic.py
processing_prismatic.py
```

`trust_remote_code=True` 允许 Transformers 加载这些自定义代码。

在生产或长期部署中，建议固定 revision：

```bash
export OPENVLA_REVISION="47a0ec7fc4ec123775a391911046cf33cf9ed83f"
```

这样可以避免远程代码更新后行为变化，提升可复现性和安全性。

## 10. 如何理解 action 输出

OpenVLA 输出的 action 是连续向量，但它不是绝对通用的机器人命令格式。它的具体含义取决于训练数据集和机器人控制接口。

以常见 7 维输出为例：

```text
[x, y, z, roll, pitch, yaw, gripper]
```

常见解释是：

- 前 3 维控制末端位置变化。
- 中间 3 维控制末端姿态变化。
- 最后一维控制夹爪。

但实际部署时必须确认：

1. 坐标系是机器人 base frame、eef frame，还是相机 frame。
2. 动作是 delta action 还是 absolute target。
3. 姿态是 Euler、axis-angle，还是其他表示。
4. gripper 的数值方向是 `1=open` 还是 `1=close`。
5. 控制器是否还会做缩放、限幅、平滑或安全检查。

因此，OpenVLA 输出不能不经确认就直接接入真实机器人。

## 11. 与评测脚本的关系

`benchmark_openvla_thor.py` 测的是模型侧延迟，不包含完整机器人系统延迟。

它包含：

```text
processor_time_ms + h2d_time_ms + predict_action_total_time_ms = model_e2e_time_ms 附近
```

它不包含：

- 相机曝光和图像采集耗时。
- ROS/网络传输耗时。
- 机器人控制器执行耗时。
- 低层伺服周期耗时。
- 安全策略、滤波器、轨迹插值耗时。

所以，如果要判断真实 10Hz/20Hz 控制闭环是否成立，需要在 `model_e2e_time_ms` 之外，再加上相机、通信和控制器耗时。

## 12. 常见问题

### 12.1 为什么输出不是文字

因为 OpenVLA 把动作也表示成 token。模型确实通过 LLM 的 generate 机制生成 token，但这些 token 会被解释为动作，而不是自然语言回答。

### 12.2 为什么要反归一化

训练时不同机器人、不同数据集的动作尺度不同。为了统一训练，动作会被归一化。推理时必须根据对应数据集统计量还原真实尺度。

### 12.3 为什么 `unnorm_key` 很重要

`unnorm_key` 决定使用哪套动作统计量。如果模型训练时混合了多个数据集，错误的 `unnorm_key` 会导致动作幅度和夹爪含义不匹配。

### 12.4 为什么要固定 revision

OpenVLA 使用 `trust_remote_code=True` 加载自定义代码。固定 revision 可以保证加载的模型代码、配置和 processor 版本一致，避免远程仓库更新导致部署行为变化。

### 12.5 OpenVLA 是否支持多图像输入

当前本文档描述的是本仓库中 `openvla-7b` 的常规单图像输入链路。README 中提到的后续 OFT 等工作可能支持更多输入形式和更高频控制，但那不是当前 runtime 脚本默认评测的路径。

## 13. 相关代码索引

| 主题 | 文件 |
| --- | --- |
| 官方模型说明和最小推理示例 | `README.md` |
| OpenVLA `predict_action()` 实现 | `prismatic/models/vlas/openvla.py` |
| 动作 token 离散化/解码 | `prismatic/vla/action_tokenizer.py` |
| Runtime 环境加载 | `runtime_env.py` |
| 最小本地推理测试 | `tools/test_openvla_local.py` |
| Thor 端侧性能评测 | `tools/benchmark_openvla_thor.py` |
| 工具版 benchmark | `tools/benchmark_openvla_thor.py` |

## 14. 一句话总结

OpenVLA 是一个把“看见的图像”和“听到的任务指令”直接转换成机器人下一步动作的视觉-语言-动作模型。它内部仍然使用 VLM/LLM 的 token 生成机制，但生成的 token 会被解释为动作 token，并通过动作解码与反归一化得到 7 维连续机器人控制向量。
