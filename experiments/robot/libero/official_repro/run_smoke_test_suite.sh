#!/usr/bin/env bash
# =============================================================================
# 单个 LIBERO suite 冒烟测试（1 task × 2 trials，官方预处理）
#
# 用法:
#   SUITE=object bash experiments/robot/libero/official_repro/run_smoke_test_suite.sh
#   SUITE=spatial BACKEND=fp8 bash ...   # 测 FP8 链路
# =============================================================================
set -euo pipefail

: "${SUITE:=spatial}"
: "${BACKEND:=bf16}"
: "${SEED:=7}"
: "${PY:=/home/bruce/miniconda3/envs/torch270_128/bin/python}"
: "${REPO:=/share_data/bruce/workspace/ai/openvla}"
: "${BUILD:=/home/bruce/TensorRT-Edge-LLM/build}"
: "${LOGDIR:=$REPO/experiments/logs/libero_official}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/libero_suites.sh"
resolve_libero_suite "$SUITE"

RUN_TAG="official-${SUITE_SHORT}-smoke-${BACKEND}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO"
export OPENVLA_PREFIX="$REPO"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TF_CPP_MIN_LOG_LEVEL=2
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="$BUILD:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export EDGELLM_PLUGIN_PATH="${EDGELLM_PLUGIN_PATH:-$BUILD/libNvInfer_edgellm_plugin.so}"

cd "$REPO"
mkdir -p "$LOGDIR"

echo ">>> Smoke: suite=$TASK_SUITE backend=$BACKEND task=0 trials=2"

if [[ "$BACKEND" == "bf16" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  "$PY" "$SCRIPT_DIR/run_official_shard.py" \
    --pretrained_checkpoint "$CKPT" \
    --task_suite_name "$TASK_SUITE" \
    --task_ids 0 \
    --num_trials_per_task 2 \
    --seed "$SEED" \
    --center_crop True \
    --attn_implementation sdpa \
    --run_name "$RUN_TAG" \
    --log_dir "$LOGDIR"
else
  if ! fp8_artifacts_ready; then
    echo "FP8 artifacts missing; run: SUITE=$SUITE bash $SCRIPT_DIR/build_fp8_artifacts_for_suite.sh" >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  "$PY" "$REPO/experiments/robot/libero/run_libero_deploy_eval.py" \
    --backend fp8 \
    --checkpoint "$CKPT" \
    --task-suite-name "$TASK_SUITE" \
    --task-ids 0 \
    --num-trials-per-task 2 \
    --preprocessing official \
    --local-files-only \
    --device cuda:0 \
    --seed "$SEED" \
    --vision-engine "$FP8_VISION_ENGINE" \
    --llm-engine-dir "$FP8_LLM_ENGINE_DIR" \
    --action-metadata "$FP8_ACTION_META" \
    --edge-llm-plugin "$BUILD/libNvInfer_edgellm_plugin.so" \
    --output-dir "$LOGDIR" \
    --run-name "$RUN_TAG"
fi

echo ""
echo "Summary: $LOGDIR/${RUN_TAG}.summary.json"
