#!/usr/bin/env bash
# =============================================================================
# 官方预处理 BF16 8 卡并行（run_official_shard.py，TensorFlow 预处理）
# 支持四个 LIBERO suite，按 task 分片（与 spatial 历史结果命名一致）。
#
# 用法:
#   SUITE=object bash experiments/robot/libero/official_repro/run_official_bf16_multigpu_suite.sh
#   SUITE=goal SEED=42 TRIALS=50 bash ...
# =============================================================================
set -euo pipefail

: "${SUITE:=spatial}"
: "${TRIALS:=50}"
: "${SEED:=7}"
: "${NUM_SHARDS:=8}"
: "${ATTENTION_IMPL:=sdpa}"
: "${PY:=/home/bruce/miniconda3/envs/torch270_128/bin/python}"
: "${REPO:=/share_data/bruce/workspace/ai/openvla}"
: "${LOGDIR:=$REPO/experiments/logs/libero_official}"
: "${FORCE_RESTART:=0}"
: "${RESUME:=1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/libero_suites.sh"
resolve_libero_suite "$SUITE"

: "${RUN_TAG:=$(official_bf16_run_tag "$SEED")}"

MAIN_LOG="$LOGDIR/${RUN_TAG}.log"
ALL_TASK_IDS=(0 1 2 3 4 5 6 7 8 9)
GPUS=(0 1 2 3 4 5 6 7)
TASK_SPLITS=()

build_task_splits() {
  TASK_SPLITS=()
  local num_tasks=${#ALL_TASK_IDS[@]}
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

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "$REPO"
build_task_splits
mkdir -p "$LOGDIR"

log() { echo "[$(date -Is)] $*" | tee -a "$MAIN_LOG"; }

maybe_clean_shards() {
  if [[ "$FORCE_RESTART" != "1" ]]; then
    return 0
  fi
  log "FORCE_RESTART=1: removing shard outputs for RUN_TAG=$RUN_TAG"
  rm -f "$LOGDIR/${RUN_TAG}"-s*.jsonl "$LOGDIR/${RUN_TAG}"-s*.summary.json
  rm -f "$LOGDIR/${RUN_TAG}.summary.json"
  rm -f "$LOGDIR/${RUN_TAG}"-s*.worker.log
}

launch_shard() {
  local gpu="$1"
  local shard_idx="$2"
  local task_ids="$3"
  local run_name="${RUN_TAG}-s${shard_idx}"
  local worker_log="$LOGDIR/${run_name}.worker.log"

  local resume_flag=""
  if [[ "$RESUME" == "1" ]]; then
    resume_flag="--resume True"
  fi

  echo "[$(date -Is)] launch shard${shard_idx} tasks=${task_ids} GPU=${gpu} -> ${run_name}" >>"$MAIN_LOG"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PY" "$SCRIPT_DIR/run_official_shard.py" \
      --pretrained_checkpoint "$CKPT" \
      --task_suite_name "$TASK_SUITE" \
      --task_ids "$task_ids" \
      --num_trials_per_task "$TRIALS" \
      --seed "$SEED" \
      --center_crop True \
      --attn_implementation "$ATTENTION_IMPL" \
      --run_name "$run_name" \
      --log_dir "$LOGDIR" \
      $resume_flag
  ) >"$worker_log" 2>&1 &
  LAUNCH_PID=$!
}

maybe_clean_shards

log "======== Official BF16 8-GPU eval (suite=$TASK_SUITE) ========"
log "RUN_TAG=$RUN_TAG TRIALS=$TRIALS SEED=$SEED CKPT=$CKPT"
log "task_splits=${TASK_SPLITS[*]}"

pids=()
for ((i = 0; i < NUM_SHARDS; i++)); do
  launch_shard "${GPUS[$i]}" "$i" "${TASK_SPLITS[$i]}"
  pids+=("$LAUNCH_PID")
done

log "waiting for workers: ${pids[*]}"
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    log "worker failed: pid=$pid"
    fail=1
  fi
done
if (( fail )); then
  log "one or more workers failed; check $LOGDIR/${RUN_TAG}-s*.worker.log"
  exit 1
fi

SUMMARIES=()
for ((i = 0; i < NUM_SHARDS; i++)); do
  SUMMARIES+=("$LOGDIR/${RUN_TAG}-s${i}.summary.json")
done

log "merging shard summaries"
"$PY" "$SCRIPT_DIR/merge_official_summaries.py" \
  --summaries "${SUMMARIES[@]}" \
  --output "$LOGDIR/${RUN_TAG}.summary.json" \
  --run-name "$RUN_TAG" | tee -a "$MAIN_LOG"

echo ""
echo "===== BF16 合并结果 (suite=$TASK_SUITE) ====="
"$PY" - <<PY
import json
from pathlib import Path
p = Path("$LOGDIR") / "${RUN_TAG}.summary.json"
d = json.loads(p.read_text())
print(f"总成功率: {d['successes']}/{d['episodes']} = {100*d['success_rate']:.2f}%")
print(f"summary: {p}")
for t in d.get("tasks", []):
    print(f"  task {t['task_id']}: {100*t['success_rate']:.1f}% ({t['successes']}/{t['episodes']})")
PY

log "done"
