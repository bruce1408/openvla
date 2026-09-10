#!/bin/bash
# 自动检测并显示当前正在运行的 LIBERO 评测任务进度
# 支持: FP8 balanced (run_official_eval_multigpu_balanced.sh) / BF16 suite (run_official_bf16_multigpu_suite.sh)
# 手动覆盖: TAG=xxx TGT_LIST=... TASK_LIST=... WORKER_PAT=... bash process_info.sh
LOGDIR=/share_data/bruce/workspace/ai/openvla/experiments/logs/libero_official
export LOGDIR

FP8_TAG=official-10-seed7-8gpu-fp8-balanced
BF16_TAG=official-10-seed7-8gpu

fp8_running=$(pgrep -f "run_official_eval_multigpu_balanced" | wc -l)
bf16_running=$(pgrep -f "run_official_bf16_multigpu_suite" | wc -l)

if [[ "$fp8_running" -gt 0 ]]; then
  export TAG="$FP8_TAG"
  export MODE="FP8 balanced"
  export TGT_LIST="63,63,63,63,62,62,62,62"
  export TASK_LIST="0,1|1,2|2,3|3,4,5|5,6|6,7|7,8|8,9"
  export WORKER_PAT="run_libero_deploy_eval"
  export EXPECT_WORKERS=8
elif [[ "$bf16_running" -gt 0 ]]; then
  export TAG="$BF16_TAG"
  export MODE="BF16 suite"
  export TGT_LIST="100,100,50,50,50,50,50,50"
  export TASK_LIST="0-1|2-3|4|5|6|7|8|9"
  export WORKER_PAT="run_official_shard"
  export EXPECT_WORKERS=8
else
  # 没有活动任务: 默认显示最近跑过的 FP8 balanced, 标注为静态状态
  export TAG="${TAG:-$FP8_TAG}"
  export MODE="no-active-task (static snapshot)"
  export TGT_LIST="${TGT_LIST:-63,63,63,63,62,62,62,62}"
  export TASK_LIST="${TASK_LIST:-0,1|1,2|2,3|3,4,5|5,6|6,7|7,8|8,9}"
  export WORKER_PAT="${WORKER_PAT:-run_libero_deploy_eval}"
  export EXPECT_WORKERS=0
fi

python3 - <<'PY'
import os, re
from datetime import datetime
from pathlib import Path

logdir = Path(os.environ["LOGDIR"])
tag = os.environ["TAG"]
mode = os.environ["MODE"]
targets = [int(x) for x in os.environ["TGT_LIST"].split(",")]
tasks = os.environ["TASK_LIST"].split("|")

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})")
TQDM_RE = re.compile(r"(\d+)%\|.*?\| *(\d+)/(\d+) \[[\d:]+<([\d:]+), *([\d.]+)s/it\]")


def clean(s):
    s = ANSI_RE.sub("", s)
    s = s.replace("\r", " ")
    return s.strip()


def launch_ts(s):
    main_log = logdir / f"{tag}.log"
    if not main_log.exists():
        return None
    pat = re.compile(rf"launch .*shard{s}\b")
    t = None
    for ln in main_log.read_text(errors="ignore").splitlines():
        if pat.search(ln):
            m = TS_RE.search(ln)
            if m:
                t = datetime.strptime(m.group(1) + " " + m.group(2), "%Y-%m-%d %H:%M:%S")
    return t


now = datetime.now()
print(f"===== {mode}: {tag} =====")
print(f"{'shard':<6}{'tasks':<9}{'done/tgt':<11}{'speed':<11}{'shard ETA':<11}last line")
tot_done = tot_tgt = 0
etas = []
for s in range(8):
    tgt = targets[s]
    jl = logdir / f"{tag}-s{s}.jsonl"
    done = sum(1 for _ in jl.open()) if jl.exists() else 0
    log = logdir / f"{tag}-s{s}.worker.log"
    speed = None
    last = ""
    if log.exists():
        text = log.read_text(errors="ignore")
        lines = text.splitlines()
        for ln in reversed(lines):
            m = TQDM_RE.search(ln)
            if m:
                speed = float(m.group(5))
                last = clean(ln)[-70:]
                break
        if speed is None:
            eps = [ln for ln in lines if re.search(r"episode=\d+", ln)]
            if eps:
                last = clean(eps[-1])[-90:]
                start = launch_ts(s) or datetime.fromtimestamp(log.stat().st_mtime)
                el = max((now - start).total_seconds(), 1)
                if len(eps) > 0:
                    speed = el / len(eps)
    tot_done += done
    tot_tgt += tgt
    if done >= tgt:
        st = "DONE"
        speed = None
        last = "DONE"
    elif speed:
        eta_s = (tgt - done) * speed
        etas.append(eta_s)
        h, rem = divmod(int(eta_s), 3600)
        st = f"{h}h{rem // 60:02d}m"
    else:
        st = "??"
    print(f"s{s:<5}{tasks[s]:<9}{done}/{tgt:<9}{('%.0fs/ep' % speed) if speed else '-':<11}{st:<11}{last}")
print(f"\ntotal: {tot_done}/{tot_tgt} ({100 * tot_done / tot_tgt:.1f}%)")
if etas:
    h, rem = divmod(int(max(etas)), 3600)
    print(f"overall ETA (slowest shard): ~{h}h{rem // 60:02d}m")
PY

echo ""
echo "--- GPU usage ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""
echo "--- workers alive (expect ${EXPECT_WORKERS}) ---"
pgrep -f "$WORKER_PAT" | wc -l
echo "--- fp8 parent / bf16 parent ---"
pgrep -f "run_official_eval_multigpu_balanced" | wc -l
pgrep -f "run_official_bf16_multigpu_suite" | wc -l
