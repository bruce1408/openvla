# OpenVLA LIBERO 仿真评测（BF16 vs FP8）

本文档说明在 **5090 服务器**上对 OpenVLA **LIBERO-Spatial** 做 BF16 PyTorch 与 FP8 TensorRT 配对仿真评测的完整流程。

仿真评测与「下载 LIBERO 演示数据集（HDF5）」是两件不同的事：

| 类型 | 路径 | 用途 | 仿真评测是否需要 |
| --- | --- | --- | --- |
| **Benchmark 资源** | `bddl_files/`、`init_files/`、`assets/` | 任务定义、初始状态、3D 模型 | ✅ 需要（随 LIBERO 安装） |
| **Demonstration 数据集** | `LIBERO/libero/datasets/` | 人类遥操作 HDF5，**训练用** | ❌ **不需要** |

`[Warning]: datasets path ... does not exist` 可忽略；若只想消除警告：

```bash
mkdir -p /share_data/bruce/workspace/ai/LIBERO/libero/datasets
```

更详细的脚本参数说明见：`experiments/robot/libero/DEPLOY_EVAL.md`。

---

## 1. 环境与路径（5090 本机）

| 项 | 路径 |
| --- | --- |
| Python 环境 | `torch270_128`（Py 3.10，torch 2.7.0+cu128） |
| OpenVLA 仓库 | `/share_data/bruce/workspace/ai/openvla` |
| LIBERO（editable） | `/share_data/bruce/workspace/ai/LIBERO` |
| Checkpoint | `/share_data/huggingface/models/openvla-7b-finetuned-libero-spatial` |
| TensorRT | `/share_data/bruce/software/TensorRT-11.2.1.2` |
| Edge-LLM build | `/home/bruce/TensorRT-Edge-LLM/build` |
| FP8 vision engine | `deploy/tensorrt/artifacts/engines/vision_projector_fp8.plan` |
| FP8 LLM engine | `deploy/tensorrt/artifacts/engines/openvla_llama_fp8/llm.engine` |
| action 元数据 | `deploy/tensorrt/artifacts/action_meta/action_meta.json` |

### 1.1 LIBERO 安装（一次性）

```bash
conda activate torch270_128

pip install -e /share_data/bruce/workspace/ai/LIBERO
pip install -r /share_data/bruce/workspace/ai/openvla/experiments/robot/libero/libero_requirements.txt
pip install draccus==0.8.0

# robosuite 1.4.1 需 mujoco 2.3.7（不要用 3.x）
pip install mujoco==2.3.7
```

**不要**安装 `LIBERO/requirements.txt`（会降级 PyTorch）。

验证：

```bash
python -c "from libero.libero import benchmark; print(list(benchmark.get_benchmark_dict().keys()))"
```

### 1.2 公共环境变量

每次新开 terminal 建议执行：

```bash
export PY=/home/bruce/miniconda3/envs/torch270_128/bin/python
export REPO=/share_data/bruce/workspace/ai/openvla
export CKPT=/share_data/huggingface/models/openvla-7b-finetuned-libero-spatial
export BUILD=/home/bruce/TensorRT-Edge-LLM/build
export TRT=/share_data/bruce/software/TensorRT-11.2.1.2

export PATH="/home/bruce/miniconda3/envs/torch270_128/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$BUILD:$TRT/lib:/usr/local/cuda/lib64"
export PYTHONPATH="$REPO"
export OPENVLA_PREFIX="$REPO"
export OPENVLA_MODEL_ID="$CKPT"
export OPENVLA_UNNORM_KEY=libero_spatial
export EDGE_LLM_DIR=/home/bruce/TensorRT-Edge-LLM
export EDGELLM_PLUGIN_PATH="$BUILD/libNvInfer_edgellm_plugin.so"
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd "$REPO"
source env_gpu.sh
```

---

## 2. 评测类型说明

### 2.1 单样本动作 RMSE（固定 golden，非仿真）

脚本：`deploy/tensorrt/pipeline/fp8/05_fp8_action_accuracy.py`

- **golden-emb**：固定 BF16 embedding，只测 LLM FP8 量化误差
- **action**：FP8 vision + FP8 LLM 全链路

```bash
OUT=$REPO/deploy/tensorrt/artifacts/validate

$PY deploy/tensorrt/pipeline/fp8/05_fp8_action_accuracy.py \
  --mode golden-emb --device cuda:0 --unnorm-key libero_spatial \
  --output "$OUT/fp8_golden_emb.json"

$PY deploy/tensorrt/pipeline/fp8/05_fp8_action_accuracy.py \
  --mode action --device cuda:0 --unnorm-key libero_spatial \
  --output "$OUT/fp8_action.json"
```

### 2.2 LIBERO 仿真成功率（BF16 vs FP8 配对）

脚本：`experiments/robot/libero/run_libero_deploy_eval.py`

- 在 MuJoCo 里 rollout，统计任务成功率
- BF16 与 FP8 必须使用相同 `task_ids`、`seed`、`num_trials_per_task`、初始状态
- 默认 `--preprocessing portable`（Pillow，部署向）；对齐论文需 TensorFlow + `--preprocessing official`

---

## 3. 冒烟测试（单 task，快速验证链路）

```bash
export CUDA_VISIBLE_DEVICES=1

# BF16
$PY experiments/robot/libero/run_libero_deploy_eval.py \
  --backend bf16 --checkpoint "$CKPT" \
  --task-suite-name libero_spatial --task-ids 0 --num-trials-per-task 2 \
  --preprocessing portable --local-files-only --device cuda:0 \
  --attn-implementation sdpa --run-name spatial-bf16-smoke

# FP8
$PY experiments/robot/libero/run_libero_deploy_eval.py \
  --backend fp8 --checkpoint "$CKPT" \
  --task-suite-name libero_spatial --task-ids 0 --num-trials-per-task 2 \
  --preprocessing portable --local-files-only --device cuda:0 \
  --vision-engine "$REPO/deploy/tensorrt/artifacts/engines/vision_projector_fp8.plan" \
  --llm-engine-dir "$REPO/deploy/tensorrt/artifacts/engines/openvla_llama_fp8" \
  --action-metadata "$REPO/deploy/tensorrt/artifacts/action_meta/action_meta.json" \
  --edge-llm-plugin "$BUILD/libNvInfer_edgellm_plugin.so" \
  --run-name spatial-fp8-smoke

# 配对对比
$PY experiments/robot/libero/compare_libero_results.py \
  --bf16-summary experiments/logs/libero_deploy/spatial-bf16-smoke.summary.json \
  --fp8-summary experiments/logs/libero_deploy/spatial-fp8-smoke.summary.json \
  --output experiments/logs/libero_deploy/spatial-bf16-vs-fp8-smoke.json
```

日志目录：`experiments/logs/libero_deploy/`。

---

## 4. 单卡扩大评测

**推荐在 tmux 中运行**（断开 SSH 不中断）：

```bash
tmux new -s libero-eval
cd /share_data/bruce/workspace/ai/openvla
source env_gpu.sh
export CUDA_VISIBLE_DEVICES=1

bash experiments/robot/libero/run_expanded_eval.sh \
  2>&1 | tee experiments/logs/libero_deploy/spatial-expanded-seed7.log
```

默认配置：

- **Suite**：`libero_spatial`，10 tasks × 10 trials = 100 episodes/backend
- **Run tag**：`spatial-expanded-seed7`
- 顺序：BF16 全部跑完 → FP8 全部跑完 → 自动 compare

可选环境变量：

```bash
TRIALS=20 RUN_TAG=my-eval bash experiments/robot/libero/run_expanded_eval.sh
```

耗时粗估（portable 预处理）：

| 规模 | BF16 | FP8 | 合计（顺序） |
| --- | --- | --- | --- |
| 100 episodes | ~1.3 h | ~25 min | ~2 h |
| 500 episodes（论文单 seed） | ~6.5 h | ~2 h | ~8–9 h |

---

## 5. 八卡并行评测（推荐）

脚本：`experiments/robot/libero/run_expanded_eval_multigpu.sh`

**GPU 布局**（BF16 与 FP8 同时跑）：

| GPU | Backend | Task 范围 |
| --- | --- | --- |
| 0 | BF16 shard 0 | 0–2 |
| 1 | BF16 shard 1 | 3–5 |
| 2 | BF16 shard 2 | 6–7 |
| 3 | BF16 shard 3 | 8–9 |
| 4 | FP8 shard 0 | 0–2 |
| 5 | FP8 shard 1 | 3–5 |
| 6 | FP8 shard 2 | 6–7 |
| 7 | FP8 shard 3 | 8–9 |

### 5.1 启动

```bash
tmux new -s libero-8gpu
cd /share_data/bruce/workspace/ai/openvla
source env_gpu.sh

bash experiments/robot/libero/run_expanded_eval_multigpu.sh \
  2>&1 | tee experiments/logs/libero_deploy/spatial-expanded-seed7-8gpu.log
```

默认 `RUN_TAG=spatial-expanded-seed7-8gpu`，`TRIALS=10`。

### 5.2 监控

```bash
# 总日志
tail -f experiments/logs/libero_deploy/spatial-expanded-seed7-8gpu.log

# 单个 worker（例：BF16 GPU0）
tail -f experiments/logs/libero_deploy/spatial-expanded-seed7-8gpu-bf16-s0.worker.log

# 各 shard 已完成 episode 数（目标：bf16/fp8 各 100 行）
wc -l experiments/logs/libero_deploy/spatial-expanded-seed7-8gpu-*-s*.jsonl
```

正常推进时 worker 日志会出现：

```text
task=00 episode=003 success=False steps=220 total=3/4 (75.0%)
```

### 5.3 断点续跑

每个 shard 使用独立 `--run-name` 且带 `--resume`；中断后**重新执行同一脚本**即可跳过已完成 episode。

**注意**：不要多个进程写同一个 `--run-name`（会竞争同一 `.jsonl` 文件）。

### 5.4 手动 merge + compare

若 worker 已全部跑完但主脚本未执行 merge，可手动：

```bash
LOG=experiments/logs/libero_deploy
TAG=spatial-expanded-seed7-8gpu

$PY experiments/robot/libero/merge_libero_deploy_shards.py \
  --shards $LOG/${TAG}-bf16-s{0,1,2,3}.jsonl \
  --shard-summaries $LOG/${TAG}-bf16-s{0,1,2,3}.summary.json \
  --output-jsonl $LOG/${TAG}-bf16.jsonl \
  --output-summary $LOG/${TAG}-bf16.summary.json \
  --all-task-ids 0 1 2 3 4 5 6 7 8 9

$PY experiments/robot/libero/merge_libero_deploy_shards.py \
  --shards $LOG/${TAG}-fp8-s{0,1,2,3}.jsonl \
  --shard-summaries $LOG/${TAG}-fp8-s{0,1,2,3}.summary.json \
  --output-jsonl $LOG/${TAG}-fp8.jsonl \
  --output-summary $LOG/${TAG}-fp8.summary.json \
  --all-task-ids 0 1 2 3 4 5 6 7 8 9

$PY experiments/robot/libero/compare_libero_results.py \
  --bf16-summary $LOG/${TAG}-bf16.summary.json \
  --fp8-summary $LOG/${TAG}-fp8.summary.json \
  --output $LOG/${TAG}-bf16-vs-fp8.json
```

---

## 6. 结果文件说明

| 文件 | 含义 |
| --- | --- |
| `{run-name}.jsonl` | 每个 episode 一行：task_id、success、steps、latency 等 |
| `{run-name}.summary.json` | 汇总成功率、延迟分位数、逐 task 统计 |
| `{run-name}-bf16-vs-fp8.json` | BF16/FP8 配对对比：成功率差、延迟加速比、pair_outcomes |

`compare_libero_results.py` 输出字段示例：

- `bf16_success_rate` / `fp8_success_rate`
- `success_rate_delta_percentage_points`
- `latency_speedup`
- `pair_outcomes`：`both_success` / `bf16_only` / `fp8_only` / `both_failure`

---

## 7. 相关脚本索引

| 脚本 | 作用 |
| --- | --- |
| `experiments/robot/libero/run_libero_deploy_eval.py` | 单进程 BF16/FP8 仿真 rollout |
| `experiments/robot/libero/run_expanded_eval.sh` | 单卡：BF16 → FP8 → compare |
| `experiments/robot/libero/run_expanded_eval_multigpu.sh` | 八卡：4×BF16 + 4×FP8 并行 |
| `experiments/robot/libero/merge_libero_deploy_shards.py` | 合并多卡 shard 的 jsonl/summary |
| `experiments/robot/libero/compare_libero_results.py` | BF16 vs FP8 配对对比 |
| `deploy/tensorrt/pipeline/fp8/05_fp8_action_accuracy.py` | 单样本 golden RMSE |

---

## 8. 常见问题

### Q1：`datasets path does not exist`

仿真评测**不需要** HDF5 演示数据。创建空目录或忽略即可。

### Q2：下载 `download_libero_datasets.py` 失败

本机若无法访问 HuggingFace（`Network is unreachable`），**不影响仿真评测**。演示数据仅训练需要；可在联网机器下载后 scp 到 `LIBERO/libero/datasets/`。

### Q3：离线加载 `processing_prismatic.py` 失败

确保使用 in-repo 类加载（`deploy/tensorrt/common.py` 的 `load_openvla_processor`），且设置：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

评测脚本 `--local-files-only` 且**不要**对 processor 传 `trust_remote_code=True`（会触发 Hub `auto_map`）。

### Q4：PyTorch 2.7 加载 init states 失败

`run_libero_deploy_eval.py` 已在 `main()` 中对 `torch.load` 设置 `weights_only=False`（LIBERO init state pickle 需要）。

### Q5：论文 84.7% 与本机数字的关系

论文 **84.7%** 是 **BF16 全精度** LIBERO-Spatial、3 seeds × 500 rollouts、**official** 预处理的结果。本流程默认 portable 预处理，适合部署链路验证；应用 `--preprocessing official` + TensorFlow 才能更接近论文口径。

### Q6：FP8 产物必须与 LIBERO checkpoint 一致

`action_meta.json` 的 `unnorm_key` 与 `source_model` 必须与 `--checkpoint` 匹配（`libero_spatial` 微调权重）。不可用 `bridge_orig` 基座模型的 FP8 产物评 LIBERO。

---

## 9. 推荐工作流（从零到报告）

1. **确认 FP8 引擎**已用 LIBERO-Spatial checkpoint 构建（vision + LLM + `action_meta.json`）。
2. **单样本 RMSE**：`05_fp8_action_accuracy.py`（golden-emb + action）。
3. **冒烟**：task 0 × 2 trials，确认 BF16/FP8 均无 error。
4. **正式评测**：八卡 `run_expanded_eval_multigpu.sh`（或单卡 `run_expanded_eval.sh`）。
5. **读结果**：`experiments/logs/libero_deploy/{RUN_TAG}-bf16-vs-fp8.json`。

---

## 10. 与 Thor 文档口径的差异

| 项 | Thor 文档 | 5090 本机 |
| --- | --- | --- |
| Baseline | 可能为 FP16 TRT | **BF16 PyTorch** |
| Vision | FP8 TRT | FP8 TRT |
| LLM | Edge-LLM FP8 | Edge-LLM FP8（本机重编 engine） |
| 指标 | RMSE + 仿真成功率 | 同上 |

单样本 FP8 RMSE 在本机仍明显高于 Thor 同口径（~0.0016），仿真成功率对比更能反映部署可用性。
