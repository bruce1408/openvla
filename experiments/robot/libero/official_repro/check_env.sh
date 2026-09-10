#!/usr/bin/env bash
# 检查官方 LIBERO 复现所需依赖
set -euo pipefail

: "${PY:=/home/bruce/miniconda3/envs/torch270_128/bin/python}"

echo "Python: $PY"
"$PY" - <<'PY'
import sys

errors = []
warnings = []

def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as exc:
        errors.append(f"{name}: {exc}")

def warn(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as exc:
        warnings.append(f"{name}: {exc}")

check("torch", lambda: __import__("torch"))
check("transformers", lambda: __import__("transformers"))
check("tensorflow (official preprocessing)", lambda: __import__("tensorflow"))
check("libero", lambda: __import__("libero"))
check("draccus", lambda: __import__("draccus"))
warn("flash_attn (paper default; sdpa fallback if missing)", lambda: __import__("flash_attn"))

import torch
print(f"     torch version: {torch.__version__}")
print(f"     CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"     GPU count: {torch.cuda.device_count()}")

if warnings:
    print("\n[WARN]")
    for w in warnings:
        print(f"  - {w}")
    print("  可设置 ATTENTION_IMPL=sdpa 作为 flash-attn 替代")

if errors:
    print("\n[FAIL] 缺少必要依赖:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)

print("\n环境检查通过，可以运行官方复现脚本。")
PY
