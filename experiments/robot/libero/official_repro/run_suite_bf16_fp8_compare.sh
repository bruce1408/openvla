#!/usr/bin/env bash
# =============================================================================
# 单个 LIBERO suite：官方 BF16 vs FP8 完整流程
#
#   1) （可选）跑 FP8 8 卡均衡评测
#   2) 与已有 official BF16 结果配对对比
#
# BF16 默认 tag: official-{suite}-seed{N}-8gpu  （run_official_bf16_multigpu_suite.sh）
# FP8  默认 tag: official-{suite}-seed{N}-8gpu-fp8-balanced
#
# 用法:
#   SUITE=spatial bash experiments/robot/libero/official_repro/run_suite_bf16_fp8_compare.sh
#   SUITE=object SKIP_FP8=1 bash ...   # 仅对比（FP8 已跑完）
#   SUITE=spatial BF16_TAG=... FP8_TAG=... bash ...  # 自定义 tag
# =============================================================================
set -euo pipefail

: "${SUITE:=spatial}"
: "${SEED:=7}"
: "${SKIP_FP8:=0}"
: "${PY:=/home/bruce/miniconda3/envs/torch270_128/bin/python}"
: "${REPO:=/share_data/bruce/workspace/ai/openvla}"
: "${LOGDIR:=$REPO/experiments/logs/libero_official}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/libero_suites.sh"
resolve_libero_suite "$SUITE"

: "${BF16_TAG:=$(official_bf16_run_tag "$SEED")}"
: "${FP8_TAG:=$(official_fp8_balanced_run_tag "$SEED")}"
COMPARE_OUT="$(official_compare_output "$SEED" "$LOGDIR")"

cd "$REPO"

if [[ "$SKIP_FP8" != "1" ]]; then
  echo ">>> Step 1/2: Run official FP8 eval (suite=$TASK_SUITE)"
  SUITE="$SUITE" SEED="$SEED" RUN_TAG="$FP8_TAG" BACKEND=fp8 \
    bash "$SCRIPT_DIR/run_official_eval_multigpu_balanced.sh"
else
  echo ">>> Step 1/2: SKIP_FP8=1, using existing FP8 results ($FP8_TAG)"
fi

echo ""
echo ">>> Step 2/2: Compare official BF16 vs FP8 (suite=$TASK_SUITE)"
BF16_SHARDS=("$LOGDIR/${BF16_TAG}"-s*.jsonl)
FP8_SHARDS=("$LOGDIR/${FP8_TAG}"-s*.jsonl)

if [[ ! -f "$LOGDIR/${BF16_TAG}.summary.json" ]]; then
  echo "BF16 summary missing: $LOGDIR/${BF16_TAG}.summary.json" >&2
  echo "Run: SUITE=$SUITE bash $SCRIPT_DIR/run_official_bf16_multigpu_suite.sh" >&2
  exit 1
fi

"$PY" "$SCRIPT_DIR/compare_official_bf16_fp8.py" \
  --bf16-shards "${BF16_SHARDS[@]}" \
  --fp8-shards "${FP8_SHARDS[@]}" \
  --bf16-summary "$LOGDIR/${BF16_TAG}.summary.json" \
  --fp8-summary "$LOGDIR/${FP8_TAG}.summary.json" \
  --output "$COMPARE_OUT"

echo ""
echo "Compare JSON: $COMPARE_OUT"
