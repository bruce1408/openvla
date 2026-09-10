#!/usr/bin/env bash
# =============================================================================
# 8-GPU LIBERO-Spatial BF16 vs FP8 配对评测（一键：启动 worker → merge → compare）
#
# 用法（只需两步）：
#   cd /share_data/bruce/workspace/ai/openvla
#   source env_gpu.sh
#   bash experiments/robot/libero/run_expanded_eval_multigpu.sh
#
# 可选：命令行仍可通过环境变量临时覆盖下方默认值，例如：
#   TRIALS=10 RUN_TAG=smoke-test bash experiments/robot/libero/run_expanded_eval_multigpu.sh
#
# GPU 模式（GPU_MODE）：
#   full8  — 默认。BF16 先用 GPU 0-7 跑满 8 shard，完成后 FP8 再用 0-7 跑满 8 shard
#   dual4  — 旧布局。BF16 用 0-3、FP8 用 4-7，两套 backend 同时并行（各 4 shard）
# =============================================================================
set -euo pipefail

# ---------- 人工配置（改这里即可；环境变量可临时覆盖） ----------
: "${TRIALS:=50}"                          # 每 task rollout 数；论文单 seed = 50
: "${RUN_TAG:=spatial-full-seed7-8gpu}"    # 日志/run-name 前缀
: "${SEED:=7}"                              # 评测随机种子
: "${ENV_SEED:=0}"                          # LIBERO 环境 seed
: "${PREPROCESSING:=portable}"              # portable | official（official 需 TensorFlow）
: "${TASK_SUITE:=libero_spatial}"           # LIBERO suite
: "${GPU_MODE:=full8}"                      # full8 | dual4
: "${NUM_SHARDS:=8}"                        # 并行 shard 数（full8=8 GPU，dual4=4 GPU/backend）
: "${FORCE_RESTART:=0}"                     # 1 = 删除当前 RUN_TAG 的 shard 产物后重跑

# 路径（一般不用改）
: "${PY:=/home/bruce/miniconda3/envs/torch270_128/bin/python}"
: "${REPO:=/share_data/bruce/workspace/ai/openvla}"
: "${CKPT:=/share_data/huggingface/models/openvla-7b-finetuned-libero-spatial}"
: "${BUILD:=/home/bruce/TensorRT-Edge-LLM/build}"
: "${TRT:=/share_data/bruce/software/TensorRT-11.2.1.2}"
: "${LOGDIR:=$REPO/experiments/logs/libero_deploy}"
: "${MAIN_LOG:=$LOGDIR/${RUN_TAG}.log}"

ALL_TASK_IDS=(0 1 2 3 4 5 6 7 8 9)
GPUS=()
TASK_SPLITS=()
BF16_GPUS=()
FP8_GPUS=()

# 按 task 数均分到 NUM_SHARDS（10 tasks / 8 shards → 2+2+1+1+1+1+1+1）
build_task_splits() {
  TASK_SPLITS=()
  local num_tasks=${#ALL_TASK_IDS[@]}
  if (( NUM_SHARDS < 1 || NUM_SHARDS > num_tasks )); then
    echo "error: NUM_SHARDS=$NUM_SHARDS must be in [1, $num_tasks]" >&2
    exit 1
  fi

  local base=$((num_tasks / NUM_SHARDS))
  local remainder=$((num_tasks % NUM_SHARDS))
  local idx=0
  for ((s = 0; s < NUM_SHARDS; s++)); do
    local count=$base
    if (( s < remainder )); then
      count=$((count + 1))
    fi
    local start_id=${ALL_TASK_IDS[$idx]}
    local end_id=${ALL_TASK_IDS[$((idx + count - 1))]}
    if [[ "$start_id" == "$end_id" ]]; then
      TASK_SPLITS+=("$start_id")
    else
      TASK_SPLITS+=("$start_id-$end_id")
    fi
    idx=$((idx + count))
  done
}

configure_gpu_layout() {
  case "$GPU_MODE" in
    full8)
      NUM_SHARDS="${NUM_SHARDS:-8}"
      GPUS=(0 1 2 3 4 5 6 7)
      if (( NUM_SHARDS != 8 )); then
        echo "warning: GPU_MODE=full8 expects NUM_SHARDS=8 (got $NUM_SHARDS); using first 8 GPUs" >&2
      fi
      NUM_SHARDS=8
      ;;
    dual4)
      NUM_SHARDS="${NUM_SHARDS:-4}"
      if (( NUM_SHARDS != 4 )); then
        echo "warning: GPU_MODE=dual4 expects NUM_SHARDS=4 (got $NUM_SHARDS); forcing 4" >&2
      fi
      NUM_SHARDS=4
      BF16_GPUS=(0 1 2 3)
      FP8_GPUS=(4 5 6 7)
      ;;
    *)
      echo "error: GPU_MODE must be full8 or dual4 (got $GPU_MODE)" >&2
      exit 1
      ;;
  esac
  build_task_splits
}

# ---------- 帮助 ----------
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage:
  source env_gpu.sh   # 在仓库根目录执行一次（本脚本不会 source）
  bash experiments/robot/libero/run_expanded_eval_multigpu.sh

Edit defaults at the top of this script, or override via env:
  TRIALS=50 RUN_TAG=my-run PREPROCESSING=official bash ...

Config (current defaults):
  TRIALS=$TRIALS
  RUN_TAG=$RUN_TAG
  SEED=$SEED
  PREPROCESSING=$PREPROCESSING
  GPU_MODE=$GPU_MODE
  NUM_SHARDS=$NUM_SHARDS
  FORCE_RESTART=$FORCE_RESTART

Outputs:
  \$LOGDIR/\${RUN_TAG}-bf16.summary.json
  \$LOGDIR/\${RUN_TAG}-fp8.summary.json
  \$LOGDIR/\${RUN_TAG}-bf16-vs-fp8.json
EOF
  exit 0
fi

# ---------- 环境 ----------
export PYTHONUNBUFFERED=1
export PATH="/home/bruce/miniconda3/envs/torch270_128/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$BUILD:$TRT/lib:/usr/local/cuda/lib64"
export PYTHONPATH="$REPO"
export OPENVLA_PREFIX="$REPO"
export OPENVLA_MODEL_ID="${OPENVLA_MODEL_ID:-$CKPT}"
export OPENVLA_UNNORM_KEY="${OPENVLA_UNNORM_KEY:-libero_spatial}"
export EDGE_LLM_DIR="${EDGE_LLM_DIR:-/home/bruce/TensorRT-Edge-LLM}"
export EDGELLM_PLUGIN_PATH="${EDGELLM_PLUGIN_PATH:-$BUILD/libNvInfer_edgellm_plugin.so}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "$REPO"
configure_gpu_layout

if [[ ! -f "$REPO/env_gpu.sh" ]]; then
  echo "warning: $REPO/env_gpu.sh not found" >&2
fi
if [[ -z "${OPENVLA_MODEL_ID:-}" ]]; then
  echo "error: 请先 source env_gpu.sh，或设置 OPENVLA_MODEL_ID" >&2
  exit 1
fi

mkdir -p "$LOGDIR"

log() { echo "[$(date -Is)] $*" | tee -a "$MAIN_LOG"; }

maybe_clean_shards() {
  if [[ "$FORCE_RESTART" != "1" ]]; then
    return 0
  fi
  log "FORCE_RESTART=1: removing existing shard outputs for RUN_TAG=$RUN_TAG"
  rm -f "$LOGDIR/${RUN_TAG}"-bf16-s*.jsonl "$LOGDIR/${RUN_TAG}"-bf16-s*.summary.json
  rm -f "$LOGDIR/${RUN_TAG}"-fp8-s*.jsonl "$LOGDIR/${RUN_TAG}"-fp8-s*.summary.json
  rm -f "$LOGDIR/${RUN_TAG}"-bf16.jsonl "$LOGDIR/${RUN_TAG}"-bf16.summary.json
  rm -f "$LOGDIR/${RUN_TAG}"-fp8.jsonl "$LOGDIR/${RUN_TAG}-fp8.summary.json"
  rm -f "$LOGDIR/${RUN_TAG}"-bf16-vs-fp8.json
  rm -f "$LOGDIR/${RUN_TAG}"-*-s*.worker.log
}

COMMON=(
  --checkpoint "$CKPT"
  --task-suite-name "$TASK_SUITE"
  --num-trials-per-task "$TRIALS"
  --preprocessing "$PREPROCESSING"
  --local-files-only
  --device cuda:0
  --seed "$SEED"
  --env-seed "$ENV_SEED"
)

launch_shard() {
  local backend="$1"
  local gpu="$2"
  local shard_idx="$3"
  local task_ids="$4"
  local run_name="${RUN_TAG}-${backend}-s${shard_idx}"
  local log_file="$LOGDIR/${run_name}.worker.log"

  echo "[$(date -Is)] launch ${backend} shard${shard_idx} tasks=${task_ids} GPU=${gpu} -> ${run_name}" >>"$MAIN_LOG"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    if [[ "$backend" == "bf16" ]]; then
      "$PY" experiments/robot/libero/run_libero_deploy_eval.py \
        --backend bf16 \
        "${COMMON[@]}" \
        --task-ids "$task_ids" \
        --attn-implementation sdpa \
        --run-name "$run_name" \
        --resume
    else
      "$PY" experiments/robot/libero/run_libero_deploy_eval.py \
        --backend fp8 \
        "${COMMON[@]}" \
        --task-ids "$task_ids" \
        --vision-engine "$REPO/deploy/tensorrt/artifacts/engines/vision_projector_fp8.plan" \
        --llm-engine-dir "$REPO/deploy/tensorrt/artifacts/engines/openvla_llama_fp8" \
        --action-metadata "$REPO/deploy/tensorrt/artifacts/action_meta/action_meta.json" \
        --edge-llm-plugin "$BUILD/libNvInfer_edgellm_plugin.so" \
        --run-name "$run_name" \
        --resume
    fi
  ) >"$log_file" 2>&1 &
  LAUNCH_PID=$!
}

wait_workers() {
  local label="$1"
  shift
  local pids=("$@")
  log "waiting for ${#pids[@]} ${label} workers: ${pids[*]}"
  local fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      log "${label} worker failed: pid=${pid}"
      fail=1
    fi
  done
  if (( fail )); then
    log "one or more ${label} workers failed; check $LOGDIR/${RUN_TAG}-*.worker.log"
    exit 1
  fi
}

run_backend_full8() {
  local backend="$1"
  local pids=()
  log "======== ${backend^^} phase (GPUs ${GPUS[*]}) ========"
  for ((i = 0; i < NUM_SHARDS; i++)); do
    launch_shard "$backend" "${GPUS[$i]}" "$i" "${TASK_SPLITS[$i]}"
    pids+=("$LAUNCH_PID")
  done
  wait_workers "$backend" "${pids[@]}"
}

run_backends_dual4() {
  local pids=()
  log "======== BF16+FP8 parallel (BF16 GPUs ${BF16_GPUS[*]} | FP8 GPUs ${FP8_GPUS[*]}) ========"
  for ((i = 0; i < NUM_SHARDS; i++)); do
    launch_shard bf16 "${BF16_GPUS[$i]}" "$i" "${TASK_SPLITS[$i]}"
    pids+=("$LAUNCH_PID")
    launch_shard fp8 "${FP8_GPUS[$i]}" "$i" "${TASK_SPLITS[$i]}"
    pids+=("$LAUNCH_PID")
  done
  wait_workers "bf16+fp8" "${pids[@]}"
}

maybe_clean_shards

log "======== LIBERO 8-GPU eval ========"
log "RUN_TAG=$RUN_TAG TRIALS=$TRIALS SEED=$SEED PREPROCESSING=$PREPROCESSING"
log "GPU_MODE=$GPU_MODE shards=$NUM_SHARDS task_splits=${TASK_SPLITS[*]}"
log "main log: $MAIN_LOG"

case "$GPU_MODE" in
  full8)
    run_backend_full8 bf16
    run_backend_full8 fp8
    ;;
  dual4)
    run_backends_dual4
    ;;
esac

BF16_SHARDS=()
BF16_SUMMARIES=()
FP8_SHARDS=()
FP8_SUMMARIES=()
for ((i = 0; i < NUM_SHARDS; i++)); do
  BF16_SHARDS+=("$LOGDIR/${RUN_TAG}-bf16-s${i}.jsonl")
  BF16_SUMMARIES+=("$LOGDIR/${RUN_TAG}-bf16-s${i}.summary.json")
  FP8_SHARDS+=("$LOGDIR/${RUN_TAG}-fp8-s${i}.jsonl")
  FP8_SUMMARIES+=("$LOGDIR/${RUN_TAG}-fp8-s${i}.summary.json")
done

log "merging BF16 shards"
"$PY" experiments/robot/libero/merge_libero_deploy_shards.py \
  --shards "${BF16_SHARDS[@]}" \
  --shard-summaries "${BF16_SUMMARIES[@]}" \
  --output-jsonl "$LOGDIR/${RUN_TAG}-bf16.jsonl" \
  --output-summary "$LOGDIR/${RUN_TAG}-bf16.summary.json" \
  --all-task-ids "${ALL_TASK_IDS[@]}" | tee -a "$MAIN_LOG"

log "merging FP8 shards"
"$PY" experiments/robot/libero/merge_libero_deploy_shards.py \
  --shards "${FP8_SHARDS[@]}" \
  --shard-summaries "${FP8_SUMMARIES[@]}" \
  --output-jsonl "$LOGDIR/${RUN_TAG}-fp8.jsonl" \
  --output-summary "$LOGDIR/${RUN_TAG}-fp8.summary.json" \
  --all-task-ids "${ALL_TASK_IDS[@]}" | tee -a "$MAIN_LOG"

log "compare BF16 vs FP8"
"$PY" experiments/robot/libero/compare_libero_results.py \
  --bf16-summary "$LOGDIR/${RUN_TAG}-bf16.summary.json" \
  --fp8-summary "$LOGDIR/${RUN_TAG}-fp8.summary.json" \
  --output "$LOGDIR/${RUN_TAG}-bf16-vs-fp8.json" | tee -a "$MAIN_LOG"

log "done: $LOGDIR/${RUN_TAG}-bf16-vs-fp8.json"
