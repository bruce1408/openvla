#!/usr/bin/env bash
# =============================================================================
# 官方预处理 8 卡并行评测 —— 按 episode 均衡分片（支持四个 LIBERO suite）
#
# BACKEND:
#   bf16  — PyTorch BF16（run_libero_deploy_eval.py, preprocessing=official）
#   fp8   — TensorRT FP8（需先 build_fp8_artifacts_for_suite.sh）
#
# 用法:
#   cd /share_data/bruce/workspace/ai/openvla && source env_gpu.sh
#
#   # Spatial BF16（500 ep）
#   SUITE=spatial BACKEND=bf16 bash experiments/robot/libero/official_repro/run_official_eval_multigpu_balanced.sh
#
#   # Object FP8（需先构建 FP8 产物）
#   SUITE=object BACKEND=fp8 bash experiments/robot/libero/official_repro/run_official_eval_multigpu_balanced.sh
#
# 可选环境变量:
#   SEED TRIALS NUM_SHARDS FORCE_RESTART RESUME RUN_TAG
#   FP8_VISION_ENGINE_OVERRIDE / FP8_LLM_ENGINE_DIR_OVERRIDE / FP8_ACTION_META_OVERRIDE
# =============================================================================
set -euo pipefail

: "${SUITE:=spatial}"
: "${BACKEND:=fp8}"
: "${TRIALS:=50}"
: "${SEED:=7}"
: "${NUM_SHARDS:=8}"
: "${PY:=/home/bruce/miniconda3/envs/torch270_128/bin/python}"
: "${REPO:=/share_data/bruce/workspace/ai/openvla}"
: "${BUILD:=/home/bruce/TensorRT-Edge-LLM/build}"
: "${TRT:=/share_data/bruce/software/TensorRT-11.2.1.2}"
: "${LOGDIR:=$REPO/experiments/logs/libero_official}"
: "${FORCE_RESTART:=0}"
: "${RESUME:=1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/libero_suites.sh"
resolve_libero_suite "$SUITE"

if [[ -z "${RUN_TAG:-}" ]]; then
  if [[ "$BACKEND" == "fp8" ]]; then
    RUN_TAG="$(official_fp8_balanced_run_tag "$SEED")"
  else
    RUN_TAG="official-${SUITE_SHORT}-seed${SEED}-8gpu-bf16-balanced"
  fi
fi

MAIN_LOG="$LOGDIR/${RUN_TAG}.log"
PLAN_JSON="$LOGDIR/${RUN_TAG}.shard_plan.json"
ALL_TASK_IDS=(0 1 2 3 4 5 6 7 8 9)
GPUS=(0 1 2 3 4 5 6 7)
EPISODE_SPECS=()

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO"
export OPENVLA_PREFIX="$REPO"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export LD_LIBRARY_PATH="$BUILD:$TRT/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export EDGELLM_PLUGIN_PATH="${EDGELLM_PLUGIN_PATH:-$BUILD/libNvInfer_edgellm_plugin.so}"

cd "$REPO"
mkdir -p "$LOGDIR"

log() { echo "[$(date -Is)] $*" | tee -a "$MAIN_LOG"; }

if [[ "$BACKEND" == "fp8" ]] && ! fp8_artifacts_ready; then
  echo "FP8 artifacts missing for suite=$SUITE_SHORT" >&2
  echo "  vision:  $FP8_VISION_ENGINE" >&2
  echo "  llm:     $FP8_LLM_ENGINE_DIR/llm.engine" >&2
  echo "  meta:    $FP8_ACTION_META" >&2
  echo "Run: SUITE=$SUITE_SHORT bash $SCRIPT_DIR/build_fp8_artifacts_for_suite.sh" >&2
  exit 1
fi

build_episode_specs() {
  "$PY" "$SCRIPT_DIR/episode_balanced_sharding.py" \
    --num-tasks "$NUM_TASKS" \
    --trials "$TRIALS" \
    --shards "$NUM_SHARDS" \
    --format json >"$PLAN_JSON"

  EPISODE_SPECS=()
  while IFS= read -r spec; do
    EPISODE_SPECS+=("$spec")
  done < <("$PY" - <<PY
import json
for item in json.load(open("$PLAN_JSON")):
    print(item["episode_spec"])
PY
)
}

maybe_clean_shards() {
  if [[ "$FORCE_RESTART" != "1" ]]; then
    return 0
  fi
  log "FORCE_RESTART=1: removing shard outputs for RUN_TAG=$RUN_TAG"
  rm -f "$LOGDIR/${RUN_TAG}"-s*.jsonl "$LOGDIR/${RUN_TAG}"-s*.summary.json
  rm -f "$LOGDIR/${RUN_TAG}.summary.json" "$LOGDIR/${RUN_TAG}.jsonl"
  rm -f "$LOGDIR/${RUN_TAG}"-s*.worker.log
  rm -f "$PLAN_JSON"
}

launch_shard() {
  local gpu="$1"
  local shard_idx="$2"
  local episode_spec="$3"
  local run_name="${RUN_TAG}-s${shard_idx}"
  local worker_log="$LOGDIR/${run_name}.worker.log"

  local resume_flag=""
  if [[ "$RESUME" == "1" ]]; then
    resume_flag="--resume"
  fi

  local backend_args=()
  if [[ "$BACKEND" == "bf16" ]]; then
    backend_args=(--backend bf16 --attn-implementation sdpa)
  else
    backend_args=(
      --backend fp8
      --vision-engine "$FP8_VISION_ENGINE"
      --llm-engine-dir "$FP8_LLM_ENGINE_DIR"
      --action-metadata "$FP8_ACTION_META"
      --edge-llm-plugin "$BUILD/libNvInfer_edgellm_plugin.so"
    )
  fi

  echo "[$(date -Is)] launch ${BACKEND} balanced shard${shard_idx} GPU=${gpu} spec=${episode_spec} -> ${run_name}" >>"$MAIN_LOG"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PY" "$REPO/experiments/robot/libero/run_libero_deploy_eval.py" \
      "${backend_args[@]}" \
      --checkpoint "$CKPT" \
      --task-suite-name "$TASK_SUITE" \
      --episode-spec "$episode_spec" \
      --num-trials-per-task "$TRIALS" \
      --preprocessing official \
      --local-files-only \
      --device cuda:0 \
      --seed "$SEED" \
      --env-seed 0 \
      --output-dir "$LOGDIR" \
      --run-name "$run_name" \
      $resume_flag
  ) >"$worker_log" 2>&1 &
  LAUNCH_PID=$!
}

build_episode_specs
maybe_clean_shards

log "======== Official ${BACKEND^^} 8-GPU eval (balanced, suite=$TASK_SUITE) ========"
log "RUN_TAG=$RUN_TAG TRIALS=$TRIALS SEED=$SEED CKPT=$CKPT"
log "shard plan: $PLAN_JSON"
"$PY" "$SCRIPT_DIR/episode_balanced_sharding.py" \
  --num-tasks "$NUM_TASKS" --trials "$TRIALS" --shards "$NUM_SHARDS" --format table | tee -a "$MAIN_LOG"

pids=()
for ((i = 0; i < NUM_SHARDS; i++)); do
  launch_shard "${GPUS[$i]}" "$i" "${EPISODE_SPECS[$i]}"
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

SHARDS=()
SUMMARIES=()
for ((i = 0; i < NUM_SHARDS; i++)); do
  SHARDS+=("$LOGDIR/${RUN_TAG}-s${i}.jsonl")
  SUMMARIES+=("$LOGDIR/${RUN_TAG}-s${i}.summary.json")
done

log "merging ${BACKEND} shards"
"$PY" "$REPO/experiments/robot/libero/merge_libero_deploy_shards.py" \
  --shards "${SHARDS[@]}" \
  --shard-summaries "${SUMMARIES[@]}" \
  --output-jsonl "$LOGDIR/${RUN_TAG}.jsonl" \
  --output-summary "$LOGDIR/${RUN_TAG}.summary.json" \
  --all-task-ids "${ALL_TASK_IDS[@]}" | tee -a "$MAIN_LOG"

echo ""
echo "===== ${BACKEND^^} 合并结果 (suite=$TASK_SUITE, balanced) ====="
"$PY" - <<PY
import json
from pathlib import Path
p = Path("$LOGDIR") / "${RUN_TAG}.summary.json"
d = json.loads(p.read_text())
print(f"总成功率: {d['successes']}/{d['episodes']} = {100*d['success_rate']:.2f}%")
if d.get("action_latency_ms", {}).get("mean") is not None:
    print(f"mean latency: {d['action_latency_ms']['mean']:.1f} ms")
print(f"summary: {p}")
for t in d.get("tasks", []):
    print(f"  task {t['task_id']}: {100*t['success_rate']:.1f}% ({t['successes']}/{t['episodes']})")
PY

log "done"
