#!/usr/bin/env python3
"""Compile and run the cuBLASLt BF16 peak GEMM (bench_peak_bf16.cu).

bench_peak_tensorcore.py CAN time BF16, but only via torch.mm (one cuBLAS
kernel). That is typically 110-160 TFLOPS on Thor, not the ~228/259
datasheet ceiling.

This wrapper builds a small C++ program that:
  - uses CUDA_R_16BF Tensor Cores
  - asks cuBLASLt for many algorithms (heuristic)
  - times each and keeps the best shape/algo

    python bench_peak_bf16.py
    python bench_peak_bf16.py --iters 50
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
SRC = DIR / "bench_peak_bf16.cu"
BIN = DIR / "bench_peak_bf16"


def compile_if_needed() -> None:
    if BIN.exists() and BIN.stat().st_mtime >= SRC.stat().st_mtime:
        return
    nvcc = os.environ.get("NVCC", "nvcc")
    cmd = [
        nvcc,
        "-O3",
        "-std=c++17",
        "-arch=sm_110",
        str(SRC),
        "-lcublasLt",
        "-o",
        str(BIN),
    ]
    print("compile:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    p = argparse.ArgumentParser(description="cuBLASLt BF16 Tensor Core peak")
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--force-rebuild", action="store_true")
    args, rest = p.parse_known_args()
    if args.force_rebuild and BIN.exists():
        BIN.unlink()
    compile_if_needed()
    cmd = [str(BIN), "--iters", str(args.iters), "--warmup", str(args.warmup), *rest]
    print("run:", " ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
