#!/usr/bin/env python3
# =============================================================================
# 13_nsys_cpu_overhead.py - 用 nsys 采集 Edge-LLM 的 host 侧 CPU 开销 (文档 §7)
# =============================================================================
#
# 目标 (文档 §7):
#   BF16 baseline 的 CPU 开销 (cudaStreamSynchronize / cudaLaunchKernel / cudaMemcpy)
#   来自 torch.profiler。FP8 主导耗时的 Edge-LLM 是 C++ binary,torch.profiler 无法
#   attach。这里用 Nsight Systems (nsys) 对 llm_inference 子进程做 system-wide trace,
#   从 CUDA API 时间线读出 host 侧开销,填 §7 的 FP8 列。
#
# 做法:
#   1. nsys profile -t cuda,nvtx  <llm_inference ...>  → .nsys-rep
#   2. nsys stats --report cuda_api_sum --format csv   → 各 CUDA API 的总耗时/调用数
#   3. 汇总关注项 (cudaStreamSynchronize/cudaMemcpy*/cudaLaunchKernel/...) → JSON
#
# 用法:
#   python 13_nsys_cpu_overhead.py
#   python 13_nsys_cpu_overhead.py --warmup 3
# =============================================================================

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
from pathlib import Path

EDGE_LLM_DIR = Path(os.environ.get("EDGE_LLM_DIR", "/workspace/TensorRT-Edge-LLM"))
REPO = Path(__file__).resolve().parents[3]
ARTIFACTS = REPO / "deploy/tensorrt/artifacts"
LLM_ENGINE_DIR = ARTIFACTS / "engines/openvla_llama_fp8"
EDGELLM_BIN = EDGE_LLM_DIR / "build/examples/llm/llm_inference"
PLUGIN = EDGE_LLM_DIR / "build/libNvInfer_edgellm_plugin.so"
SMOKE_INPUT = ARTIFACTS / "smoke_input.json"
LOGS_DIR = Path(os.environ.get("OPENVLA_LOGS_DIR", "/workspace/outputs/openvla"))

NSYS = "/usr/local/cuda/bin/nsys"

# §7 关注的 host 侧开销项 (子串匹配 CUDA API 名)
FOCUS = ["cudaStreamSynchronize", "cudaMemcpy", "cudaLaunchKernel", "cudaMalloc",
         "cudaFree", "cudaEventSynchronize", "cudaStreamWaitEvent", "cudaGraphLaunch"]


def run_nsys(tag: str, warmup: int) -> Path:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib/aarch64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")
    env["EDGELLM_PLUGIN_PATH"] = str(PLUGIN)
    rep = LOGS_DIR / f"fp8_nsys_{tag}"
    cmd = [
        NSYS, "profile", "-t", "cuda,nvtx", "-o", str(rep), "--force-overwrite", "true",
        str(EDGELLM_BIN),
        "--engineDir", str(LLM_ENGINE_DIR),
        "--inputFile", str(SMOKE_INPUT),
        "--outputFile", "/tmp/nsys_llm_out.json",
        "--warmup", str(warmup),
    ]
    print(f"  nsys profile → {rep}.nsys-rep")
    subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return rep.with_suffix(".nsys-rep")


def parse_cuda_api(rep_path: Path) -> list[dict]:
    """nsys stats cuda_api_sum → 解析 CSV。列: Time(%),Total Time(ns),Num Calls,Avg,...,Name。"""
    out = subprocess.run(
        [NSYS, "stats", "--report", "cuda_api_sum", "--format", "csv",
         "--force-export=true", str(rep_path)],
        check=True, capture_output=True, text=True,
    ).stdout
    # nsys 输出里 CSV 前有若干说明行,定位表头行
    lines = out.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith('"Time') or ln.startswith("Time")), 0)
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    rows = []
    for r in reader:
        name = (r.get("Name") or "").strip()
        if not name:
            continue
        # 兼容不同列名
        total_ns = r.get("Total Time (ns)") or r.get("Total Time(ns)") or "0"
        calls = r.get("Num Calls") or r.get("Instances") or "0"
        try:
            rows.append({"name": name, "total_ms": round(float(total_ns) / 1e6, 3),
                         "calls": int(float(calls))})
        except ValueError:
            continue
    return rows


def focus_map(rows: list[dict]) -> dict:
    m = {}
    for f in FOCUS:
        matched = [r for r in rows if f.lower() in r["name"].lower()]
        m[f] = {"total_ms": round(sum(x["total_ms"] for x in matched), 3),
                "calls": sum(x["calls"] for x in matched)}
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="nsys CPU 开销采集 (文档 §7)")
    ap.add_argument("--low", type=int, default=2, help="低 warmup 次数 (基线,含一次性 load)")
    ap.add_argument("--high", type=int, default=12, help="高 warmup 次数;delta 隔离每次推理开销")
    ap.add_argument("--tag", default="cpu_overhead")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("FP8 Edge-LLM CPU 开销采集 (nsys, 差分法隔离一次性 load)")
    print("=" * 60)

    # 差分法: 一次运行 = (warmup + 1) 次完整 prefill+decode。两次不同 warmup 的差
    # = (high-low) 次推理的 host 开销;固定的 engine load (6.7GB 权重上传) 被抵消。
    passes_low, passes_high = args.low + 1, args.high + 1
    n_diff = passes_high - passes_low

    print(f"\n[1/2] 低基线 warmup={args.low} ({passes_low} passes)...")
    rep_lo = run_nsys(f"{args.tag}_lo", args.low)
    rows_lo = parse_cuda_api(rep_lo)
    lo = focus_map(rows_lo)

    print(f"[2/2] 高 warmup={args.high} ({passes_high} passes)...")
    rep_hi = run_nsys(f"{args.tag}_hi", args.high)
    hi = focus_map(parse_cuda_api(rep_hi))

    # 每次推理 (prefill+6 decode) 的 host 侧开销 = delta / n_diff
    per_inf = {}
    for f in FOCUS:
        d_ms = hi[f]["total_ms"] - lo[f]["total_ms"]
        d_calls = hi[f]["calls"] - lo[f]["calls"]
        per_inf[f] = {"per_inference_ms": round(d_ms / n_diff, 3),
                      "per_inference_calls": round(d_calls / n_diff, 1),
                      "total_ms_low_run": lo[f]["total_ms"]}

    # 一次性 load 估计 = 低基线里被差分判定为固定的部分 (cudaMemcpy 权重上传等)
    load_once = {f: round(lo[f]["total_ms"] - per_inf[f]["per_inference_ms"] * passes_low, 3)
                 for f in FOCUS}

    result = {
        "note": ("差分法: 两次 nsys 运行 (warmup=low/high),per-inference = delta/(high-low),"
                 "抵消一次性 engine load (6.7GB 权重 H2D)。per_inference = 单次 prefill+6decode 的 host 开销。"
                 "Edge-LLM decode 用 CUDA Graph,per-kernel launch/sync 已被 graph 吸收。"),
        "engine": str(LLM_ENGINE_DIR),
        "method": {"low_warmup": args.low, "high_warmup": args.high, "diff_passes": n_diff},
        "nsys_reports": [str(rep_lo), str(rep_hi)],
        "per_inference_overhead": per_inf,
        "one_time_load_estimate_ms": load_once,
        "top15_all_api_low_run": sorted(rows_lo, key=lambda r: r["total_ms"], reverse=True)[:15],
    }

    out_path = Path(args.output) if args.output else LOGS_DIR / f"fp8_nsys_cpu_overhead_{args.tag}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n--- §7 每次推理 host 侧开销 (差分, /{n_diff} passes) ---")
    for f in sorted(FOCUS, key=lambda k: per_inf[k]["per_inference_ms"], reverse=True):
        p = per_inf[f]
        print(f"  {f:26s} {p['per_inference_ms']:8.3f} ms/inf  ({p['per_inference_calls']:.1f} calls/inf)")
    print(f"\n  一次性 load 估计 (cudaMemcpy 权重上传): {load_once.get('cudaMemcpy')} ms")
    print(f"输出: {out_path}")


if __name__ == "__main__":
    main()
