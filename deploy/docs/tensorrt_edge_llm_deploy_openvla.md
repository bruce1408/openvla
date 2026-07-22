# Review OpenVLA加速方案

> [OpenVLA部署与加速方案](chatgpt-conversation://6a55db71-f094-83e8-b9d5-8432ae566b4a)这个加速方案，请你帮我review ，然后给出具体的代码和操作步骤。谢谢

<details><summary>8 previous messages</summary>

> 我会先用 GitHub 工作流检查 `cdd_dev` 分支的真实代码结构，再核对方案里依赖版本与接口是否仍成立。然后我会把可直接运行的首阶段代码、验证门槛和 Thor 上的命令整理成仓库内交付物。
>
> 仓库已拉取。我现在先核对分支中的模型接口、现有 benchmark 与环境封装；这一步会决定哪些示例代码可以原样使用，哪些必须按该分支改写。
>
> <details><summary>Searched the web</summary>
>
> <details><summary>Searched the web</summary>
>
> - Searched the web for `site:github.com/NVIDIA TensorRT Edge-LLM Jetson Thor official ...`
> - Searched the web for `'Llama 2'`
> - Searched the web for `'tensorrt-edgellm-export'`
>
> </details>
>
> </details>
>
> 初步 review 已确认主路线正确，但有三处必须修正：当前 Edge‑LLM 的公开矩阵列的是 Llama 3.x，并未明确列 Llama 2；“256 类 Action Head”属于受约束解码优化，不能先假设与全词表完全等价；另外 Golden 数据必须同时保存 attention mask 和“空 token 是否补入”，否则后续 prefill 很容易对不齐。下面的实现会把这三点做成显式验证。
>
> <details><summary>Created 18 files, edited 6 files</summary>
>
> - Wrote `./work/openvla-cdd_dev/.gitignore` (+4 -1)
> - Wrote `./work/openvla-cdd_dev/deploy/__init__.py` (+1 -0)
> - Wrote `./work/openvla-cdd_dev/deploy/tensorrt/__init__.py` (+1 -0)
> - Wrote `./work/openvla-cdd_dev/deploy/tensorrt/pipeline/05_build_vision_engine.sh` (+27 -0)
> - Wrote `./work/openvla-cdd_dev/deploy/tensorrt/common.py` (+166 -0)
> - Wrote `./work/openvla-cdd_dev/deploy/tensorrt/pipeline/00_dump_golden.py` (+125 -0)
> - Wrote `./work/openvla-cdd_dev/deploy/tensorrt/pipeline/01_export_vision_projector_onnx.py` (+85 -0)
> - Wrote `./work/openvla-cdd_dev/deploy/tensorrt/pipeline/02_extract_llama_checkpoint.py` (+45 -0)
> - Wrote `./work/openvla-cdd_dev/deploy/tensorrt/pipeline/04_export_action_params.py` (+55 -0)
> - Wrote `./work/openvla-cdd_dev/deploy/tensorrt/runtime/__init__.py` (+1 -0)
> - Wrote `./work/openvla-cdd_dev/deploy/tensorrt/runtime/hybrid_runtime.py` (+128 -0)
> - Wrote `./work/openvla-cdd_dev/deploy/tensorrt/runtime/trt_runner.py` (+89 -0)
> - Wrote `./work/openvla-cdd_dev/deploy/tensorrt/tests/test_common.py` (+37 -0)
> - Wrote `./work/openvla-cdd_dev/deploy/tensorrt/validate/validate_vision_engine.py` (+58 -0)
> - Wrote `./work/openvla-cdd_dev/deploy/tensorrt/pipeline/01_export_vision_projector_onnx.py` (+89 -14)
> - Wrote `./work/openvla-cdd_dev/tools/bench_e2e.py` (+9 -5)
> - Wrote `./work/openvla-cdd_dev/deploy/tensorrt/README.md` (+271 -0)
> - Wrote `./work/openvla-cdd_dev/deploy/tensorrt/pipeline/00_dump_golden.py` (+17 -0)
> - Wrote `./work/openvla-cdd_dev/deploy/tensorrt/pipeline/04_export_action_params.py` (+27 -17)
> - Wrote `./work/openvla-cdd_dev/deploy/tensorrt/README.md` (+11 -1)
> - Wrote `./work/openvla-cdd_dev/deploy/tensorrt/runtime/cpp/CMakeLists.txt` (+12 -0)
> - Wrote `./work/openvla-cdd_dev/deploy/tensorrt/runtime/cpp/action_decoder.cpp` (+54 -0)
> - Wrote `./work/openvla-cdd_dev/deploy/tensorrt/runtime/cpp/action_decoder.h` (+30 -0)
> - Wrote `./work/openvla-cdd_dev/deploy/tensorrt/runtime/cpp/action_decoder_test.cpp` (+23 -0)
>
> </details>
>
> 首阶段代码已经落地：包括 Golden、合并/拆分 ONNX 导出、TensorRT 10 零拷贝式 Torch 缓冲执行器、手动 KV-cache Hybrid runtime，以及可独立测试的 C++17 动作解码器。现在我在做静态/原生测试，并会把无法在这台非 Jetson 环境验证的部分明确标出来。
>
> <details><summary>Created a file, edited 2 files</summary>
>
> - Wrote `./work/openvla-cdd_dev/deploy/tensorrt/runtime/cpp/action_decoder_test.cpp` (+1 -0)
> - Wrote `./work/openvla-cdd_dev/.gitignore` (+1 -0)
> - Wrote `./work/openvla-cdd_dev/.gitignore` (+1 -1)
> - Wrote `./work/openvla-cdd_dev/.gitignore` (+2 -0)
> - Wrote `./outputs/OpenVLA_Thor_TensorRT_Review.md` (+117 -0)
>
> </details>
>
> 补丁已经在一个干净的 `cdd_dev` 克隆上通过 `git apply --check` 和实际应用验证；Python 与 shell 静态检查也通过。唯一未完成的是依赖 Thor/CUDA/模型权重的实机执行，以及本机 AppleClang/SDK 不匹配导致无法完成的 C++ 编译测试（已在报告中区分为环境问题，Thor 上仍需跑 CTest）。
</details>

已完成 review，并基于 `cdd_dev@6d76469` 写出了可直接应用的代码补丁和操作手册。

主要修正了原方案中的三个关键假设：

- TensorRT Edge-LLM 当前公开矩阵明确列出 Llama 3.x，未明确列出 OpenVLA 使用的 Llama 2。因此必须先做兼容性导出探测，失败时按官方扩展机制注册模型，不能假定一键可用。[支持矩阵](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/supported-models.html)、[定制指南](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/customization/customization-guide.html)
- 256 类 Action Head 属于“约束到动作词表”的语义优化，必须先证明每一步 reference argmax 都在动作 token 区间内。
- OpenVLA 的 256 个动作 token 实际对应 255 个 bin centers，C++ 不能按 256 个中心硬编码。

交付物：

- [完整 Review 报告](<./outputs/OpenVLA_Thor_TensorRT_Review.md>)
- [可直接应用的代码补丁](<./outputs/openvla_thor_tensorrt.patch>)

补丁包含：

- Golden tensor dump
- 合并/拆分式 Vision + Projector ONNX 导出
- `trtexec` engine 构建脚本
- TensorRT 10 + Torch CUDA buffer runner
- TensorRT Vision + PyTorch Llama Hybrid runtime
- 手动 prefill、KV cache 和逐 token decode
- Vision embedding 数值验证
- OpenVLA Llama checkpoint 提取
- Action metadata 导出
- C++17 FP32 Action Decoder
- benchmark token 计时修正
- 完整 Thor 操作手册

应用补丁：

```bash
git checkout cdd_dev
git apply /Users/brucetrie/Documents/Codex/2026-07-17/new-chat/outputs/openvla_thor_tensorrt.patch
```

在 Thor 上建议先执行：

```bash
source env.sh

python deploy/tensorrt/pipeline/00_dump_golden.py --dtype bf16

python deploy/tensorrt/pipeline/01_export_vision_projector_onnx.py \
  --mode combined \
  --opset 17

bash deploy/tensorrt/pipeline/05_build_vision_engine.sh

python deploy/tensorrt/validate/validate_vision_engine.py \
  --engine deploy/tensorrt/artifacts/engines/vision_projector_fp16.plan

python deploy/tensorrt/runtime/hybrid_runtime.py \
  --engine deploy/tensorrt/artifacts/engines/vision_projector_fp16.plan \
  --compare-reference
```

补丁已在干净的 `cdd_dev` 副本上通过实际应用验证；Python 编译、shell 语法和 diff 检查均通过。CUDA、ONNX、TensorRT 及模型推理仍需在 Thor 实机完成。Edge-LLM 的官方部署流程确实是 checkpoint → ONNX → `llm_build` → C++ runtime，并正式支持 Jetson Thor/JetPack 7.x。[官方概览](https://nvidia.github.io/TensorRT-Edge-LLM/latest/overview.html)、[Quick Start](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/quick-start-guide.html)

> (openvla-venv) ubuntu@thor-t5000:/workspace/openvla$ python deploy/tensorrt/validate/validate_vision_engine.py \
>   --engine deploy/tensorrt/artifacts/engines/vision_projector_fp16.plan
> [07/17/2026-02:47:21] [TRT] [W] Using default stream in enqueueV3() may lead to performance issues due to additional calls to cudaStreamSynchronize()by TensorRT to ensure correct synchronization. Please use non-default stream instead.
> {
>   "shape": [
>     1,
>     256,
>     4096
>   ],
>   "max_abs_error": 0.2109375,
>   "mean_abs_error": 0.005661080591380596,
>   "cosine_similarity": 0.999889420041932,
>   "min_cosine": 0.999,
>   "passed": true
> }
> (openvla-venv) ubuntu@thor-t5000:/workspace/openvla$ python deploy/tensorrt/runtime/hybrid_runtime.py \
>   --engine deploy/tensorrt/artifacts/engines/vision_projector_fp16.plan \
>   --compare-reference
> /opt/openvla-venv/lib/python3.12/site-packages/transformers/utils/hub.py:124: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
>   warnings.warn(
> /opt/openvla-venv/lib/python3.12/site-packages/huggingface_hub/file_download.py:949: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
>   warnings.warn(
> /opt/openvla-venv/lib/python3.12/site-packages/huggingface_hub/file_download.py:949: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
>   warnings.warn(
> Downloading shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 13888.42it/s]
> Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  5.43it/s]
> /opt/openvla-venv/lib/python3.12/site-packages/huggingface_hub/file_download.py:949: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
>   warnings.warn(
> [07/17/2026-02:48:02] [TRT] [W] Using default stream in enqueueV3() may lead to performance issues due to additional calls to cudaStreamSynchronize()by TensorRT to ensure correct synchronization. Please use non-default stream instead.
> {
>   "token_ids": [
>     31880,
>     31868,
>     31928,
>     31869,
>     31825,
>     31871,
>     31744
>   ],
>   "action": [
>     -0.001998181687667966,
>     0.0008709149551577866,
>     -0.007516350597143173,
>     0.002415493130683899,
>     0.02400439791381359,
>     -6.671343726338819e-05,
>     0.9960784316062927
>   ],
>   "empty_token_appended": true,
>   "tokens_in_action_range": true,
>   "reference_token_ids": [
>     31880,
>     31868,
>     31928,
>     31869,
>     31825,
>     31871,
>     31744
>   ],
>   "token_exact_match": true,
>   "max_action_abs_error": 0.0
> }
>
> 这里按照你的建议，已经跑完了 上面的两个命令。你看是否可以开始下一步了 ？

可以进入下一阶段，但建议先把当前结果扩大到多图片回归，再开始 Edge‑LLM 集成。

你现在已经通过了关键的单样本 Gate：

- Vision 输出 shape 正确：`[1, 256, 4096]`
- cosine `0.999889 > 0.999`
- 7 个 action token 全部在合法区间
- Hybrid 手动 prefill/KV-cache/decode 与 HF reference 完全一致
- 连续动作误差为 `0`

其中 token exact match 比 cosine 更有说服力。`max_abs_error=0.2109` 虽然看起来偏大，但均值只有 `0.00566`，并且没有改变最终 token；暂时可以接受，仍需用更多图片观察。

需要先处理两个事项：

1. `empty_token_appended=true`

请检查 Golden metadata：

```bash
jq '{
  empty_token_appended,
  stock_predict_action_max_abs_error,
  tokens_in_action_range,
  generated_token_ids
}' deploy/tensorrt/artifacts/golden/sample_0001/metadata.json
```

重点确认：

```text
stock_predict_action_max_abs_error = 0
```

因为 Hybrid reference 使用了对齐后的 attention mask，而官方 `predict_action()` 的空 token 处理略有差异。这个字段为 0 才能证明控制输出也与官方路径一致。

2. TensorRT default stream 警告

它不影响本次正确性，但会引入额外同步，影响后续性能数字。开始正式 benchmark 前，应将 TensorRT runner 改为专用非默认 CUDA stream。

下一步推荐顺序：

```text
多图片 token 回归
    ↓
导出 action metadata
    ↓
测试 C++ Action Decoder
    ↓
提取 OpenVLA 内部 Llama
    ↓
执行 Edge-LLM 兼容性导出探测
```

先跑 100 张图片回归：

```bash
mkdir -p logs

for image in test_data/*.jpg; do
  python deploy/tensorrt/runtime/hybrid_runtime.py \
    --engine deploy/tensorrt/artifacts/engines/vision_projector_fp16.plan \
    --image "$image" \
    --compare-reference
done | tee logs/hybrid_100_images.log
```

统计结果：

```bash
grep -c '"token_exact_match": true' logs/hybrid_100_images.log
grep -c '"token_exact_match": false' logs/hybrid_100_images.log
grep -c '"tokens_in_action_range": false' logs/hybrid_100_images.log
```

目标：

```text
token_exact_match true  = 100
token_exact_match false = 0
tokens_in_action_range false = 0
```

通过后就可以开始 Edge‑LLM compatibility probe：

```bash
python deploy/tensorrt/pipeline/04_export_action_params.py \
  --unnorm-key bridge_orig

cmake -S deploy/tensorrt/runtime/cpp \
  -B deploy/tensorrt/runtime/cpp/build

cmake --build deploy/tensorrt/runtime/cpp/build -j

ctest \
  --test-dir deploy/tensorrt/runtime/cpp/build \
  --output-on-failure

python deploy/tensorrt/pipeline/02_extract_llama_checkpoint.py \
  --device cpu

tensorrt-edgellm-export \
  deploy/tensorrt/artifacts/hf_llama \
  deploy/tensorrt/artifacts/edgellm_onnx
```

结论：单样本 Vision TensorRT 和 Hybrid runtime 已经成功，可以继续；但先完成 `stock_predict_action_max_abs_error=0` 和多图片 `100% token exact match` 两个检查，再正式替换 LLM 后端。

> 第三阶段：x86 GPU 主机运行 compatibility probe
> 如何在x86 安装 呢 ？

<details><summary>2 previous messages</summary>

> 我会按当前 TensorRT Edge‑LLM 官方安装指南，把 x86 主机分成“只做 checkpoint/ONNX 导出”和“同时在 x86 构建运行”的两种配置，避免误装 Jetson 侧组件。先核对 v0.9 对 CUDA、TensorRT、Python 和 ModelOpt 的要求。
>
> <details><summary>Searched the web</summary>
>
> <details><summary>Searched the web</summary>
>
> - Searched the web for `https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/installation.html`
>
> </details>
>
> </details>
</details>

可以。x86 主机只负责：

```text
OpenVLA 内部 Llama checkpoint
        ↓
TensorRT Edge-LLM compatibility probe
        ↓
导出 LLM ONNX
        ↓
把 ONNX 复制到 Thor
```

不要在 x86 上构建给 Thor 使用的 TensorRT engine；TensorRT plan 应在目标 Thor 上构建。

当前官方要求是 x86-64 Linux、Ubuntu 22.04/24.04、Ampere 或更新 GPU、CUDA 12.x/13.x、Python 3.10+。7B/8B 模型建议准备 20–48GB VRAM 和 20–48GB RAM；FP8 导出需要显著更多内存。[官方安装要求](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/installation.html)

## 方案 A：Docker 安装，推荐

### 1. 检查 x86 主机

```bash
nvidia-smi
docker --version
```

验证 Docker 能访问 GPU：

```bash
docker run --rm --gpus all \
  nvcr.io/nvidia/pytorch:25.12-py3 \
  nvidia-smi
```

如果 `--gpus all` 失败，需要先安装 NVIDIA Container Toolkit。

### 2. 固定 Edge-LLM 版本

截至 2026-07-17，建议先固定 `v0.9.0`，不要直接跟随 `main`，避免 x86 exporter 和 Thor runtime 版本不一致。[v0.9.0 Release](https://github.com/NVIDIA/TensorRT-Edge-LLM/releases/tag/v0.9.0)

```bash
mkdir -p ~/openvla-edgellm
cd ~/openvla-edgellm

git clone \
  --branch v0.9.0 \
  https://github.com/NVIDIA/TensorRT-Edge-LLM.git

cd TensorRT-Edge-LLM
git submodule update --init --recursive
git rev-parse HEAD
```

保存最后输出的 commit，Thor 也应使用相同版本。

### 3. 启动容器

```bash
mkdir -p ~/openvla-edgellm/models
mkdir -p ~/openvla-edgellm/artifacts

cd ~/openvla-edgellm/TensorRT-Edge-LLM

docker run \
  --gpus all \
  --name openvla-edgellm-x86 \
  -it \
  -v "$PWD":/workspace/TensorRT-Edge-LLM \
  -v "$HOME/openvla-edgellm/models":/workspace/models \
  -v "$HOME/openvla-edgellm/artifacts":/workspace/artifacts \
  -w /workspace/TensorRT-Edge-LLM \
  nvcr.io/nvidia/pytorch:25.12-py3 \
  bash
```

以后重新进入：

```bash
docker start -ai openvla-edgellm-x86
```

### 4. 容器内安装 exporter

```bash
cd /workspace/TensorRT-Edge-LLM

python3 --version
nvcc --version
nvidia-smi

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install .
```

Compatibility probe 只需要 `pip install .`，暂时不要安装量化组件。

验证：

```bash
export EDGE_LLM_PATH=/workspace/TensorRT-Edge-LLM
export PYTHONPATH="${EDGE_LLM_PATH}:${PYTHONPATH:-}"

tensorrt-edgellm-export --help
```

如果之后需要 FP8/NVFP4 量化，再安装：

```bash
python3 -m pip install ".[tools]"

tensorrt-edgellm-quantize --help
```

官方将基础 checkpoint exporter 和量化工具拆成了两个安装层级：`pip install .` 用于基础导出，`pip install ".[tools]"` 才加入 ModelOpt、校准数据等额外组件。[官方安装说明](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/installation.html)

## 5. 在 Thor 上提取 OpenVLA Llama

先在 Thor 执行：

```bash
cd /workspace/openvla
source /opt/openvla-venv/bin/activate
source env.sh

python deploy/tensorrt/pipeline/02_extract_llama_checkpoint.py \
  --device cpu
```

检查结果：

```bash
du -sh deploy/tensorrt/artifacts/hf_llama

ls -lh deploy/tensorrt/artifacts/hf_llama
```

应包含类似：

```text
config.json
generation_config.json
model-00001-of-*.safetensors
model.safetensors.index.json
tokenizer.json / tokenizer.model
tokenizer_config.json
special_tokens_map.json
openvla_extraction.json
```

## 6. 将 checkpoint 复制到 x86

在 x86 主机执行：

```bash
rsync -avP \
  ubuntu@thor-t5000:/workspace/openvla/deploy/tensorrt/artifacts/hf_llama/ \
  ~/openvla-edgellm/models/openvla-llama/
```

检查：

```bash
du -sh ~/openvla-edgellm/models/openvla-llama
```

## 7. 先检查 checkpoint 配置

进入容器：

```bash
docker start -ai openvla-edgellm-x86
```

执行：

```bash
python3 - <<'PY'
from transformers import AutoConfig

path = "/workspace/models/openvla-llama"
config = AutoConfig.from_pretrained(path, local_files_only=True)

print("model_type:", config.model_type)
print("architectures:", config.architectures)
print("hidden_size:", config.hidden_size)
print("num_hidden_layers:", config.num_hidden_layers)
print("num_attention_heads:", config.num_attention_heads)
print("num_key_value_heads:", config.num_key_value_heads)
print("vocab_size:", config.vocab_size)
print("max_position_embeddings:", config.max_position_embeddings)
PY
```

预期重点：

```text
model_type: llama
hidden_size: 4096
num_hidden_layers: 32
vocab_size: 32064
```

这里的 `32064` 是 padded LM vocabulary；OpenVLA 动作反解码使用的有效词表仍是 `32000`。

## 8. 执行 compatibility probe

```bash
cd /workspace/TensorRT-Edge-LLM

export EDGE_LLM_PATH=/workspace/TensorRT-Edge-LLM
export PYTHONPATH="${EDGE_LLM_PATH}:${PYTHONPATH:-}"

tensorrt-edgellm-export \
  /workspace/models/openvla-llama \
  /workspace/artifacts/openvla-llama-onnx
```

建议保存日志：

```bash
tensorrt-edgellm-export \
  /workspace/models/openvla-llama \
  /workspace/artifacts/openvla-llama-onnx \
  2>&1 | tee /workspace/artifacts/openvla-export.log
```

成功后检查：

```bash
find /workspace/artifacts/openvla-llama-onnx \
  -maxdepth 3 \
  -type f \
  -printf '%p %k KB\n' \
  | sort

du -sh /workspace/artifacts/openvla-llama-onnx
```

正常情况下应出现：

```text
openvla-llama-onnx/
└── llm/
    ├── config.json
    ├── *.onnx
    └── 外部权重文件
```

官方当前推荐的手动路径也是在 x86 上执行 `tensorrt-edgellm-export`，再将 ONNX 传到边缘设备并使用 `llm_build` 构建 engine。[官方 Quick Start](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/quick-start-guide.html)

## 9. 成功后复制回 Thor

在 x86 主机执行：

```bash
rsync -avP \
  ~/openvla-edgellm/artifacts/openvla-llama-onnx/ \
  ubuntu@thor-t5000:/workspace/openvla/deploy/tensorrt/artifacts/edgellm_onnx/
```

然后才在 Thor 上执行：

```bash
/path/to/TensorRT-Edge-LLM/build/examples/llm/llm_build \
  --onnxDir /workspace/openvla/deploy/tensorrt/artifacts/edgellm_onnx/llm \
  --engineDir /workspace/openvla/deploy/tensorrt/artifacts/engines/openvla_llama \
  --maxBatchSize 1 \
  --maxInputLen 512 \
  --maxKVCacheCapacity 512
```

如果 compatibility probe 报 `unsupported model`、权重 key 不匹配、RoPE 配置不支持或 ONNX lowering 失败，请保留完整的 `openvla-export.log`。这时不要继续 build engine，需要根据具体错误判断是简单的 config 兼容问题，还是要为 Llama 2 增加 Edge‑LLM model registration。

> 两套命令本质上没有冲突：`tensorrt-edgellm-export` 的参数完全一样，区别在于“在哪台机器执行”。
>
> 以我后来给你的详细步骤为准。官方当前要求：
>
> - Thor：提取 OpenVLA Llama、构建 C++ Action Decoder、构建最终 TensorRT Engine。
> - x86 Linux + NVIDIA GPU：运行 `tensorrt-edgellm-export`，生成 ONNX。
>
> 之前那组命令是逻辑流程的简写，没有明确区分执行主机。
>
> ## 正确执行顺序
>
> ### 第一阶段：Thor 上执行
>
> 进入 OpenVLA 环境：
>
> ```bash
> cd /workspace/openvla
>
> source /opt/openvla-venv/bin/activate
> source env.sh
> ```
>
> #### 1. 导出动作元数据
>
> ```bash
> python deploy/tensorrt/pipeline/04_export_action_params.py \
>   --unnorm-key bridge_orig
>
> cat deploy/tensorrt/artifacts/action_meta/action_meta.json
> ```
>
> #### 2. 构建并测试 C++ Action Decoder
>
> ```bash
> cmake -S deploy/tensorrt/runtime/cpp \
>   -B deploy/tensorrt/runtime/cpp/build
>
> cmake --build deploy/tensorrt/runtime/cpp/build -j
>
> ctest \
>   --test-dir deploy/tensorrt/runtime/cpp/build \
>   --output-on-failure
> ```
>
> 预期结果：
>
> ```text
> 100% tests passed
> ```
>
> 这一步只验证：
>
> ```text
> Action Token
> → bin center
> → q01/q99 反归一化
> → 连续动作
> ```
>
> 它与 Edge-LLM ONNX 导出相互独立。
>
> #### 3. 提取 OpenVLA 内部的 Llama
>
> ```bash
> python deploy/tensorrt/pipeline/02_extract_llama_checkpoint.py \
>   --device cpu
> ```
>
> 检查：
>
> ```bash
> du -sh deploy/tensorrt/artifacts/hf_llama
>
> ls -lh deploy/tensorrt/artifacts/hf_llama
>
> cat deploy/tensorrt/artifacts/hf_llama/openvla_extraction.json
> ```
>
> 完成这一步后，Thor 上应该有：
>
> ```text
> /workspace/openvla/deploy/tensorrt/artifacts/hf_llama
> ```
>
> ### 第二阶段：复制到 x86 GPU 主机
>
> 在 x86 主机上执行：
>
> ```bash
> mkdir -p /workspace/openvla-export
> ```
>
> 然后从 x86 主机拉取 Thor 上的模型：
>
> ```bash
> scp -r \
>   ubuntu@thor-t5000:/workspace/openvla/deploy/tensorrt/artifacts/hf_llama \
>   /workspace/openvla-export/
> ```
>
> 如果 `thor-t5000` 不能解析，就换成 Thor 的 IP 地址。
>
> ### 第三阶段：x86 GPU 主机运行 compatibility probe
>
> 安装并激活 TensorRT Edge-LLM 后执行：
>
> ```bash
> source /workspace/edgellm-venv/bin/activate
>
> export EDGE_LLM_PATH=/workspace/TensorRT-Edge-LLM
> export PYTHONPATH="${EDGE_LLM_PATH}:${PYTHONPATH:-}"
>
> tensorrt-edgellm-export --help
> ```
>
> 然后执行真正的兼容性探测：
>
> ```bash
> tensorrt-edgellm-export \
>   /workspace/openvla-export/hf_llama \
>   /workspace/openvla-export/edgellm_onnx \
>   2>&1 | tee /workspace/openvla-export/edgellm_export.log
> ```
>
> 这条命令等价于之前的：
>
> ```bash
> tensorrt-edgellm-export \
>   deploy/tensorrt/artifacts/hf_llama \
>   deploy/tensorrt/artifacts/edgellm_onnx
> ```
>
> 只是现在使用绝对路径，并明确要求在 x86 GPU 主机运行。
>
> NVIDIA 当前明确将 exporter 放在 x86 GPU 主机，而 Thor 负责 C++ Runtime 和 Engine 构建。[官方安装说明](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/installation.html)
>
> ## 现在你应该执行到哪里
>
> 你现在可以先在 Thor 上完成：
>
> ```text
> 04_export_action_params.py
> → C++ Action Decoder 测试
> → 02_extract_llama_checkpoint.py
> → 检查 hf_llama 目录
> ```
>
> 然后暂停。下一步要看你是否有一台满足以下条件的机器：
>
> ```text
> x86-64 Linux
> NVIDIA GPU
> 20–48 GB 左右显存或可用内存空间
> CUDA 12.x/13.x
> Python 3.10+
> ```
>
> 不要直接在 Thor 上执行 `tensorrt-edgellm-export`。如果你只有 Thor、没有 x86 NVIDIA GPU 主机，需要调整导出路线。
>
> 帮我看看上面的步骤，是否合理。

<details><summary>2 previous messages</summary>

> 我会按官方当前的“x86 exporter / Thor runtime”职责边界逐条核对，并重点检查路径权限、版本一致性、日志失败码和 7B 内存条件；这些细节最容易让流程看似成功但实际留下错误产物。
>
> <details><summary>Searched the web</summary>
>
> <details><summary>Searched the web</summary>
>
> - Searched the web for `https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/installation.html`
>
> </details>
>
> </details>
</details>

整体流程合理，职责划分也正确。对于 OpenVLA 的手动兼容性探测，应采用：

```text
Thor：提取 checkpoint、最终 engine build/runtime
x86 GPU：checkpoint exporter / quantization / ONNX export
```

这与 NVIDIA 当前的 manual export 路线一致。[官方安装说明](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/installation.html)、[Manual Export 流程](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/quick-start-guide.html)

不过建议修正以下细节。

## 需要修正的地方

### 1. 不要把“禁止在 Thor 导出”说成普遍规则

官方 High-Level API 可能在 Thor 上编排导出、构建和运行；但对于我们现在采用的：

```text
手动 checkpoint exporter
+
自定义 OpenVLA Llama 2 compatibility probe
```

应明确要求在 x86 GPU 主机执行。

更准确的表述是：

> 本项目当前选择 manual checkpoint-export workflow，因此 `tensorrt-edgellm-export` 在 x86 GPU 主机运行。

### 2. x86 的 VRAM 和 RAM 不是二选一

原文：

```text
20–48 GB 左右显存或可用内存空间
```

应改为：

```text
7B FP16/BF16 导出建议：
GPU VRAM：20–48 GB
CPU RAM：20–48 GB
磁盘：至少预留 50 GB
```

两者都需要。FP8 导出要求更高，官方提示可能需要约 5–6 倍模型大小的显存和最高约 18–20 倍的主机内存。[内存要求](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/installation.html)

### 3. `/workspace` 在 x86 上可能没有写权限

普通用户执行：

```bash
mkdir -p /workspace/openvla-export
```

可能得到 `Permission denied`。

建议改为：

```bash
mkdir -p "$HOME/openvla-export"
```

后续统一使用：

```text
$HOME/openvla-export/hf_llama
$HOME/openvla-export/edgellm_onnx
```

如果 x86 环境本来就是容器，才使用 `/workspace`。

### 4. x86 和 Thor 必须使用同一 Edge-LLM 版本

ONNX exporter、builder 和 runtime 需要版本对齐。建议两端都固定 `v0.9.0`：

```bash
git clone \
  --branch v0.9.0 \
  https://github.com/NVIDIA/TensorRT-Edge-LLM.git

cd TensorRT-Edge-LLM
git submodule update --init --recursive
git rev-parse HEAD
```

记录 commit，并在 Thor 使用相同 commit。[v0.9.0 Release](https://github.com/NVIDIA/TensorRT-Edge-LLM/releases/tag/v0.9.0)

### 5. `tee` 会掩盖 exporter 失败状态

原命令：

```bash
tensorrt-edgellm-export ... 2>&1 | tee edgellm_export.log
```

如果没有 `pipefail`，exporter 失败但 `tee` 成功时，shell 可能返回 0。

应使用：

```bash
set -o pipefail

tensorrt-edgellm-export \
  "$HOME/openvla-export/hf_llama" \
  "$HOME/openvla-export/edgellm_onnx_probe1" \
  2>&1 | tee "$HOME/openvla-export/edgellm_export.log"
```

每次探测使用新的输出目录，避免旧 ONNX 文件造成假成功。

### 6. 大文件传输建议使用 `rsync`

`scp -r` 可以工作，但中断后通常要重新复制。建议：

```bash
rsync -avP \
  ubuntu@thor-t5000:/workspace/openvla/deploy/tensorrt/artifacts/hf_llama/ \
  "$HOME/openvla-export/hf_llama/"
```

注意源路径末尾的 `/`，这样不会形成重复的：

```text
hf_llama/hf_llama/
```

## 修订后的完整流程

### 阶段一：Thor

```bash
cd /workspace/openvla
source /opt/openvla-venv/bin/activate
source env.sh
```

导出动作元数据：

```bash
python deploy/tensorrt/pipeline/04_export_action_params.py \
  --unnorm-key bridge_orig
```

测试 C++ Action Decoder：

```bash
cmake \
  -S deploy/tensorrt/runtime/cpp \
  -B deploy/tensorrt/runtime/cpp/build \
  -DCMAKE_BUILD_TYPE=Release

cmake --build \
  deploy/tensorrt/runtime/cpp/build \
  -j"$(nproc)"

ctest \
  --test-dir deploy/tensorrt/runtime/cpp/build \
  --output-on-failure
```

提取微调后的 Llama：

```bash
python deploy/tensorrt/pipeline/02_extract_llama_checkpoint.py \
  --device cpu
```

检查：

```bash
du -sh deploy/tensorrt/artifacts/hf_llama

cat \
  deploy/tensorrt/artifacts/hf_llama/openvla_extraction.json
```

`--device cpu` 会加载完整 7B 权重，但 Thor 128GB 统一内存足够。需要确认磁盘有至少约 20GB 可用空间。

### 阶段二：复制到 x86

```bash
mkdir -p "$HOME/openvla-export"

rsync -avP \
  ubuntu@thor-t5000:/workspace/openvla/deploy/tensorrt/artifacts/hf_llama/ \
  "$HOME/openvla-export/hf_llama/"
```

检查：

```bash
du -sh "$HOME/openvla-export/hf_llama"

ls -lh "$HOME/openvla-export/hf_llama"
```

### 阶段三：x86 安装并执行 Probe

进入固定版本：

```bash
cd "$HOME/TensorRT-Edge-LLM"
git describe --tags --always
git rev-parse HEAD
```

激活独立环境：

```bash
source "$HOME/edgellm-venv/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install .
```

基础 FP16/BF16 compatibility probe 不需要先安装：

```bash
python -m pip install ".[tools]"
```

`[tools]` 是量化、ModelOpt、词表缩减等后续功能所需；基础 exporter 使用 `pip install .` 即可。

设置环境：

```bash
export EDGE_LLM_PATH="$HOME/TensorRT-Edge-LLM"
export PYTHONPATH="${EDGE_LLM_PATH}:${PYTHONPATH:-}"

tensorrt-edgellm-export --help
```

检查 checkpoint：

```bash
python - <<'PY'
from transformers import AutoConfig

path = "/home/你的用户名/openvla-export/hf_llama"
config = AutoConfig.from_pretrained(path, local_files_only=True)

print("model_type:", config.model_type)
print("architectures:", config.architectures)
print("hidden_size:", config.hidden_size)
print("layers:", config.num_hidden_layers)
print("heads:", config.num_attention_heads)
print("kv_heads:", config.num_key_value_heads)
print("vocab_size:", config.vocab_size)
PY
```

这里建议将路径换成实际绝对路径，不要在 Python heredoc 中依赖 `$HOME` 自动展开。

执行 Probe：

```bash
set -o pipefail

tensorrt-edgellm-export \
  "$HOME/openvla-export/hf_llama" \
  "$HOME/openvla-export/edgellm_onnx_probe1" \
  2>&1 | tee "$HOME/openvla-export/edgellm_export.log"
```

成功后检查：

```bash
test -f \
  "$HOME/openvla-export/edgellm_onnx_probe1/llm/config.json"

find "$HOME/openvla-export/edgellm_onnx_probe1" \
  -maxdepth 3 \
  -type f \
  -printf '%p %k KB\n' \
  | sort

du -sh "$HOME/openvla-export/edgellm_onnx_probe1"
```

## 最终判断

你的步骤约九成正确，可以执行。需要重点修改：

- 明确是 manual exporter 路线；
- VRAM 和 RAM 都要满足；
- x86 使用 `$HOME`，避免 `/workspace` 权限问题；
- x86 和 Thor 固定同一个 Edge-LLM tag/commit；
- 使用 `set -o pipefail`；
- 使用新输出目录和 `rsync`；
- export 成功后检查 `llm/config.json`，不能只看命令是否结束。

最关键的阶段性结果是 `edgellm_export.log`。如果 Probe 失败，下一步应根据日志判断属于 checkpoint key、Llama 2 config、RoPE、词表 padding，还是 ONNX lowering 兼容问题。

> 我还是不理解，这里为什么 thor 的模型，要转到 x86 服务器 用tensorrt-edge-llm 部署

关键误解在这里：

> 不是把模型转到 x86 上部署。x86 只是“编译主机”，真正部署和推理仍然在 Thor。

这类似交叉编译：

```text
源代码                  中间代码               目标机器程序
HF checkpoint  → x86 → ONNX       → Thor → TensorRT Engine
```

完整链路是：

```text
OpenVLA Llama 权重
      │
      │ 复制到 x86（仅用于转换）
      ▼
TensorRT Edge-LLM Exporter
      │
      ▼
LLM ONNX
      │
      │ 复制回 Thor
      ▼
Thor 上执行 llm_build
      │
      ▼
Thor 专用 TensorRT Engine
      │
      ▼
Thor C++ Runtime 推理
```

## 为什么需要 x86？

TensorRT Edge-LLM 被 NVIDIA 划分为两个组件：

| 阶段 | 执行位置 | 作用 |
|---|---|---|
| Python Exporter/Quantization | x86 Linux + NVIDIA GPU | 读取 HF 权重、重构模型、导出 ONNX |
| C++ Builder/Runtime | Jetson Thor | 编译 Thor TensorRT Engine、执行推理 |

官方 manual workflow 明确要求 checkpoint exporter 在 x86 GPU 主机运行，而 Thor 负责最终 engine build 和 runtime。[官方安装说明](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/installation.html)

原因主要是 exporter 依赖完整的 Python、PyTorch、Transformers、ONNX、ModelOpt 等工具链，NVIDIA 当前把这套“模型编译前端”主要支持在 x86 环境；Thor 侧则专注于 TensorRT C++ builder/runtime。

## “Thor 的模型”其实不是 Thor 专用模型

`hf_llama` 里面是：

```text
config.json
tokenizer
*.safetensors
```

这些 Hugging Face 权重与 CPU 架构和 GPU 架构无关。它不是“Thor 模型”，只是恰好存放在 Thor 上。

真正和 Thor 硬件绑定的是最终的：

```text
*.engine
*.plan
```

三种产物的区别：

| 产物 | 是否绑定 Thor |
|---|---|
| HF `.safetensors` | 否 |
| ONNX | 基本不绑定硬件，但绑定 Edge-LLM 导出契约 |
| TensorRT `.engine/.plan` | 是，必须在目标 Thor 软件/硬件环境构建 |

所以：

- checkpoint 可以在 Thor、x86 或存储服务器上；
- ONNX 在 x86 生成；
- TensorRT Engine 必须在 Thor 构建。

## 为什么最初在 Thor 提取 Llama？

只是因为你的完整 OpenVLA 模型已经缓存在 Thor 上，而且 Thor 的 OpenVLA 环境已经能正常加载它。

这一步：

```bash
python 02_extract_llama_checkpoint.py
```

做的是：

```text
完整 OpenVLA checkpoint
    ├── DINOv2
    ├── SigLIP
    ├── Projector
    └── Llama 2  ← 单独保存这一部分
```

它不一定必须在 Thor 执行。

如果 x86 主机也能访问 OpenVLA checkpoint，并安装独立的 OpenVLA 环境，可以直接在 x86 提取 Llama：

```text
x86 下载/复制完整 OpenVLA
       ↓
x86 提取 hf_llama
       ↓
x86 Edge-LLM exporter
```

这样可以省掉：

```text
Thor hf_llama → x86
```

但需要在 x86 同时维护两套隔离环境：

```text
openvla-venv
    → 负责读取 OpenVLA、提取 Llama

edgellm-venv
    → 负责 Llama checkpoint → ONNX
```

不能把两套依赖混在一个 venv，因为 OpenVLA 固定了较老的 Transformers/TIMM 版本。

## 最终数据流

你当前最稳妥的方案是：

```text
Thor
├── OpenVLA Vision → TensorRT Engine（已经成功）
└── 提取 hf_llama
          │
          ▼
x86 GPU
└── hf_llama → Edge-LLM ONNX
          │
          ▼
Thor
└── ONNX → TensorRT LLM Engine
          │
          ▼
最终 Runtime
├── Thor Vision Engine
├── Thor LLM Engine
└── C++ Action Decoder
```

因此 x86 不是部署服务器，只是一次性的模型转换/编译工作站。机器人运行时完全不依赖 x86，也不需要网络连接到 x86。
