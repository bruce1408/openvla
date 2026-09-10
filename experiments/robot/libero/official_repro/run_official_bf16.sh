#!/usr/bin/env bash
# =============================================================================
# 官方 BF16 单卡评测（run_official_shard.py，TensorFlow 预处理）
#
# 用法:
#   SUITE=spatial bash experiments/robot/libero/official_repro/run_official_bf16.sh
#   SUITE=object TRIALS=50 SEED=7 bash ...
# =============================================================================
set -euo pipefail

: "${SUITE:=spatial}"
: "${TRIALS:=50}"
: "${SEED:=7}"
: "${ATTENTION_IMPL:=sdpa}"
: "${PY:=/home/bruce/miniconda3/envs/torch270_128/bin/python}"
: "${REPO:=/share_data/bruce/workspace/ai/openvla}"
: "${LOGDIR:=$REPO/experiments/logs/libero_official}"
: "${RESUME:=1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/libero_suites.sh"
resolve_libero_suite "$SUITE"

: "${RUN_TAG:=official-${SUITE_SHORT}-seed${SEED}}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "$REPO"
mkdir -p "$LOGDIR"

RESUME_FLAG=""
if [[ "$RESUME" == "1" ]]; then
  RESUME_FLAG="--resume True"
fi

echo "[$(date -Is)] Official BF16 eval: suite=$TASK_SUITE RUN_TAG=$RUN_TAG TRIALS=$TRIALS SEED=$SEED"
echo "  checkpoint: $CKPT"
echo "  log dir:    $LOGDIR"

"$PY" "$SCRIPT_DIR/run_official_shard.py" \
  --pretrained_checkpoint "$CKPT" \
  --task_suite_name "$TASK_SUITE" \
  --num_trials_per_task "$TRIALS" \
  --seed "$SEED" \
  --center_crop True \
  --attn_implementation "$ATTENTION_IMPL" \
  --run_name "$RUN_TAG" \
  --log_dir "$LOGDIR" \
  $RESUME_FLAG

echo ""
echo "===== 结果 ====="
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
