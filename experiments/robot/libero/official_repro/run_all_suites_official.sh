#!/usr/bin/env bash
# =============================================================================
# 四个 LIBERO suite 批量官方评测编排脚本
#
# MODE:
#   bf16      — 仅 BF16（run_official_shard.py，8 卡 task 分片）
#   fp8       — 仅 FP8（需各 suite FP8 产物；8 卡 episode 均衡）
#   build_fp8 — 为各 suite 构建 FP8 TensorRT 产物
#   compare   — BF16 vs FP8 配对对比（需两边结果已存在或 SKIP_FP8=1）
#   all       — build_fp8 + bf16 + fp8 + compare（默认跳过已有 build）
#
# 用法:
#   cd /share_data/bruce/workspace/ai/openvla && source env_gpu.sh
#
#   # 四个 suite 全部跑 BF16
#   MODE=bf16 bash experiments/robot/libero/official_repro/run_all_suites_official.sh
#
#   # 只跑 object + goal 的 FP8 评测（spatial 已有产物）
#   MODE=fp8 SUITES="object goal" bash ...
#
#   # 构建 object 的 FP8 产物
#   MODE=build_fp8 SUITES=object bash ...
#
# 环境变量:
#   SUITES="spatial object goal 10"   默认全部
#   SEED=7
#   TRIALS=50
#   SKIP_EXISTING=1                   build_fp8 时跳过已就绪产物（默认 1）
#   STOP_ON_ERROR=1                   遇错停止（默认 1）
# =============================================================================
set -euo pipefail

: "${MODE:=bf16}"
: "${SUITES:=spatial object goal 10}"
: "${SEED:=7}"
: "${TRIALS:=50}"
: "${SKIP_EXISTING:=1}"
: "${STOP_ON_ERROR:=1}"
: "${REPO:=/share_data/bruce/workspace/ai/openvla}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/libero_suites.sh"

cd "$REPO"

log() { echo "[$(date -Is)] $*"; }

on_fail() {
  log "FAILED: suite=$1 mode=$2"
  if [[ "$STOP_ON_ERROR" == "1" ]]; then
    exit 1
  fi
}

run_build_fp8() {
  local suite="$1"
  resolve_libero_suite "$suite"
  if [[ "$SKIP_EXISTING" == "1" ]] && fp8_artifacts_ready; then
    log "skip build_fp8 ($suite): artifacts ready"
    return 0
  fi
  log ">>> build_fp8: $suite"
  SUITE="$suite" STAGE=all bash "$SCRIPT_DIR/build_fp8_artifacts_for_suite.sh" \
    || on_fail "$suite" build_fp8
}

run_bf16() {
  local suite="$1"
  log ">>> bf16 eval: $suite (seed=$SEED)"
  SUITE="$suite" SEED="$SEED" TRIALS="$TRIALS" \
    bash "$SCRIPT_DIR/run_official_bf16_multigpu_suite.sh" \
    || on_fail "$suite" bf16
}

run_fp8() {
  local suite="$1"
  resolve_libero_suite "$suite"
  if ! fp8_artifacts_ready; then
    log "FP8 artifacts missing for $suite; run MODE=build_fp8 first"
    on_fail "$suite" fp8
    return 1
  fi
  log ">>> fp8 eval: $suite (seed=$SEED)"
  SUITE="$suite" SEED="$SEED" TRIALS="$TRIALS" BACKEND=fp8 \
    bash "$SCRIPT_DIR/run_official_eval_multigpu_balanced.sh" \
    || on_fail "$suite" fp8
}

run_compare() {
  local suite="$1"
  log ">>> compare: $suite (seed=$SEED)"
  SUITE="$suite" SEED="$SEED" SKIP_FP8=1 \
    bash "$SCRIPT_DIR/run_suite_bf16_fp8_compare.sh" \
    || on_fail "$suite" compare
}

echo "=============================================="
echo " OpenVLA 官方 LIBERO 全 suite 评测"
echo " MODE=$MODE  SUITES=$SUITES  SEED=$SEED  TRIALS=$TRIALS"
echo "=============================================="

for suite in $SUITES; do
  case "$MODE" in
    build_fp8)
      run_build_fp8 "$suite"
      ;;
    bf16)
      run_bf16 "$suite"
      ;;
    fp8)
      run_fp8 "$suite"
      ;;
    compare)
      run_compare "$suite"
      ;;
    all)
      run_build_fp8 "$suite"
      run_bf16 "$suite"
      run_fp8 "$suite"
      run_compare "$suite"
      ;;
    *)
      echo "Unknown MODE=$MODE (use bf16|fp8|build_fp8|compare|all)" >&2
      exit 2
      ;;
  esac
done

echo ""
log "batch done (MODE=$MODE)"
if [[ "$MODE" == "bf16" || "$MODE" == "fp8" || "$MODE" == "compare" || "$MODE" == "all" ]]; then
  "$SCRIPT_DIR/summarize_all_suites.py" \
    --logdir "$REPO/experiments/logs/libero_official" \
    --seed "$SEED" \
    --suites $SUITES || true
fi
