#!/bin/bash
set -euo pipefail

# 以脚本自身所在目录为基准,避免依赖调用时的工作目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$SCRIPT_DIR"

cd "$SCRIPT_DIR"

# 让日志输出到脚本旁边的 openvla/logs,而非其他 OPENVLA_PREFIX
export OPENVLA_PREFIX="$RUNTIME_DIR"

# 轻量 profile:trace 体积小,方便用 Perfetto 网页打开(定位瓶颈 2 步足够)
python deploy/pytorch_bf16/prof_trace.py --instruction "pick up the object" --active 2 --no-memory --no-stack


# 详细 profile(体积大,几百 MB,Perfetto 网页可能打不开;适合 TensorBoard/离线分析)
# python deploy/pytorch_bf16/prof_trace.py --instruction "pick up the object" --active 8 --tensorboard


# 再用 benchmark 拿客观数字:
# python deploy/pytorch_bf16/bench_e2e.py --warmup 10 --iters 100 --measure-generate
