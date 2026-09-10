#!/usr/bin/env bash
# =============================================================================
# 论文口径：指定 suite 的 3 seed × 500 rollouts BF16 评测
#
# 用法:
#   SUITE=spatial bash experiments/robot/libero/official_repro/run_paper_3seeds_suite.sh
#   SUITE=object SEEDS=7,42,123 USE_MULTIGPU=1 bash ...
# =============================================================================
set -euo pipefail

: "${SUITE:=spatial}"
: "${TRIALS:=50}"
: "${SEEDS:=7,42,123}"
: "${USE_MULTIGPU:=1}"
: "${REPO:=/share_data/bruce/workspace/ai/openvla}"
: "${LOGDIR:=$REPO/experiments/logs/libero_official}"
: "${PY:=/home/bruce/miniconda3/envs/torch270_128/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/libero_suites.sh"
resolve_libero_suite "$SUITE"

RESULTS=()
IFS=',' read -r -a SEED_LIST <<< "$SEEDS"

echo "=============================================="
echo " OpenVLA 论文复现: $TASK_SUITE BF16"
echo " seeds: ${SEED_LIST[*]}"
echo " trials/task: $TRIALS"
echo " multigpu: $USE_MULTIGPU"
echo "=============================================="

for seed in "${SEED_LIST[@]}"; do
  seed="$(echo "$seed" | xargs)"
  echo ""
  echo ">>> Running seed=$seed"

  if [[ "$USE_MULTIGPU" == "1" ]]; then
    SUITE="$SUITE" TRIALS="$TRIALS" SEED="$seed" \
      bash "$SCRIPT_DIR/run_official_bf16_multigpu_suite.sh"
    summary="$LOGDIR/$(official_bf16_run_tag "$seed").summary.json"
  else
    SUITE="$SUITE" TRIALS="$TRIALS" SEED="$seed" TASK_SUITE="$TASK_SUITE" CKPT="$CKPT" \
      RUN_TAG="official-${SUITE_SHORT}-seed${seed}" \
      bash "$SCRIPT_DIR/run_official_bf16.sh"
    summary="$LOGDIR/official-${SUITE_SHORT}-seed${seed}.summary.json"
  fi

  rate="$("$PY" -c "import json; d=json.load(open('$summary')); print(f\"{d['success_rate']:.6f}\")")"
  RESULTS+=("$seed:$rate:$summary")
  echo "seed $seed -> $(python3 -c "print(f'{float('$rate')*100:.2f}%')")"
done

echo ""
echo "=============================================="
echo " 3-seed 汇总 ($TASK_SUITE)"
echo "=============================================="

"$PY" - <<PY
import json
import statistics
from pathlib import Path

results = """$(printf '%s\n' "${RESULTS[@]}")""".strip().splitlines()
rates = []
print(f"{'seed':>6}  {'rate':>8}  summary")
print("-" * 60)
for line in results:
    seed, rate, path = line.split(":", 2)
    r = float(rate)
    rates.append(r)
    print(f"{seed:>6}  {100*r:7.2f}%  {Path(path).name}")

mean = statistics.mean(rates)
stdev = statistics.stdev(rates) if len(rates) > 1 else 0.0
print("-" * 60)
print(f"mean:  {100*mean:.2f}%")
print(f"stdev: {100*stdev:.2f}%")
PY
