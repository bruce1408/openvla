# OpenVLA BF16 / FP8 的 LIBERO 评测

`run_libero_deploy_eval.py` 使用同一个 LIBERO rollout 循环评测 BF16 PyTorch 和 FP8 TensorRT，
`compare_libero_results.py` 对相同 task / initial-state episode 做配对比较。

## 评测有效性的前提

每个 LIBERO suite 都有独立的微调 checkpoint：

| Suite | Checkpoint | 动作统计 key |
| --- | --- | --- |
| `libero_spatial` | `openvla/openvla-7b-finetuned-libero-spatial` | `libero_spatial[_no_noops]` |
| `libero_object` | `openvla/openvla-7b-finetuned-libero-object` | `libero_object[_no_noops]` |
| `libero_goal` | `openvla/openvla-7b-finetuned-libero-goal` | `libero_goal[_no_noops]` |
| `libero_10` | `openvla/openvla-7b-finetuned-libero-10` | `libero_10[_no_noops]` |

BF16 checkpoint、FP8 vision engine、FP8 LLM engine、`embedding.safetensors` 和
`action_meta.json` 必须来自同一个 suite checkpoint。评测器会检查动作统计 key、词表尺寸和
`source_model`，不允许把 `bridge_orig` 或其他 suite 的部署产物混进来。

当前仓库内已有 FP8 产物来自 `openvla/openvla-7b` 基座模型，动作统计是 `bridge_orig`，
因此不能直接用于有效的 LIBERO 成功率评测。

## 环境

先安装 LIBERO 和官方仿真依赖：

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt
```

正式复现 OpenVLA 论文口径时使用 `--preprocessing official`。它包含官方 JPEG round-trip、
Lanczos3 resize 和 90% center crop，需要 TensorFlow。仓库当前 Python 3.12 环境没有安装
LIBERO 和 TensorFlow；OpenVLA 官方复现环境为 Python 3.10.13。TensorRT Python binding、
LIBERO 与 TensorFlow 需要出现在运行评测脚本的同一个 Python 环境中。

`--preprocessing portable` 只依赖 Pillow，适合部署环境的链路冒烟；它不应用于和论文成功率直接对比。

## 先做 BF16 冒烟

以下命令只跑 task 0 的第一个初始状态：

```bash
python experiments/robot/libero/run_libero_deploy_eval.py \
  --backend bf16 \
  --checkpoint /path/to/openvla-7b-finetuned-libero-spatial \
  --task-suite-name libero_spatial \
  --task-ids 0 \
  --num-trials-per-task 1 \
  --preprocessing official \
  --local-files-only \
  --run-name spatial-bf16-smoke
```

正式单 seed 评测为 10 tasks x 50 episodes：

```bash
python experiments/robot/libero/run_libero_deploy_eval.py \
  --backend bf16 \
  --checkpoint /path/to/openvla-7b-finetuned-libero-spatial \
  --task-suite-name libero_spatial \
  --num-trials-per-task 50 \
  --preprocessing official \
  --local-files-only \
  --run-name spatial-bf16-seed7
```

加入 `--resume` 可续跑同一个 `--run-name`。脚本每完成一个 episode 就追加 JSONL，并原子更新 summary。

## 为 LIBERO 重建 FP8 产物

以本地 LIBERO-Spatial checkpoint 为例：

```bash
export OPENVLA_MODEL_ID=/path/to/openvla-7b-finetuned-libero-spatial

python deploy/tensorrt/pipeline/01_export_vision_projector_onnx.py
python deploy/tensorrt/pipeline/01b_quantize_vision_fp8.py
python deploy/tensorrt/pipeline/02_extract_llama_checkpoint.py
python deploy/tensorrt/pipeline/04_export_action_params.py \
  --unnorm-key libero_spatial_no_noops

bash deploy/tensorrt/pipeline/05_build_vision_engine.sh \
  deploy/tensorrt/artifacts/onnx/vision_projector_fp8.onnx \
  deploy/tensorrt/artifacts/engines/vision_projector_fp8.plan \
  fp8
```

LLM 仍按 `03_convert_llm_x86.sh` 和 TensorRT Edge-LLM 的 `llm_build` 流程生成
`openvla_llama_fp8/llm.engine`。重建时建议给每个 suite 使用独立 artifacts 目录或立即归档，避免后一个
suite 覆盖前一个。`04_export_action_params.py` 现在会把 `source_model` 和 `source_revision` 写入元数据。

重建后先用 `12_fp8_action_accuracy.py` 或代表图像验证 token/action，再进入仿真 rollout。

## FP8 评测

```bash
python experiments/robot/libero/run_libero_deploy_eval.py \
  --backend fp8 \
  --checkpoint /path/to/openvla-7b-finetuned-libero-spatial \
  --task-suite-name libero_spatial \
  --num-trials-per-task 50 \
  --preprocessing official \
  --local-files-only \
  --vision-engine /path/to/spatial/vision_projector_fp8.plan \
  --llm-engine-dir /path/to/spatial/openvla_llama_fp8 \
  --action-metadata /path/to/spatial/action_meta.json \
  --edge-llm-plugin /workspace/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so \
  --run-name spatial-fp8-seed7
```

旧版 `action_meta.json` 没有 provenance 时，可以在人工核验整套产物来源后加
`--allow-missing-fp8-provenance`。该选项不会绕过 suite 动作统计 key 校验。

## 配对比较

```bash
python experiments/robot/libero/compare_libero_results.py \
  --bf16-summary experiments/logs/libero_deploy/spatial-bf16-seed7.summary.json \
  --fp8-summary experiments/logs/libero_deploy/spatial-fp8-seed7.summary.json \
  --output experiments/logs/libero_deploy/spatial-bf16-vs-fp8.json
```

输出包括总体和逐 task 成功率、FP8 相对 BF16 的百分点变化、动作推理平均延迟与加速比，以及
`both_success / bf16_only / fp8_only / both_failure` 的配对 episode 数。比较器默认要求 suite、task、
episode 数、初始状态索引、seed、horizon、center crop 和预处理方式完全相同。

论文表格使用 3 seeds x 500 rollouts。单 seed 跑通后，应分别以三个 seed 运行 BF16/FP8，并报告
每个 suite 的均值和跨 seed 波动；不能用单张图的动作 RMSE替代 LIBERO 任务成功率。
