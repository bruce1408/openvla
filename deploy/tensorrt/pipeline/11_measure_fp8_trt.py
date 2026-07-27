#!/usr/bin/env python3
"""
measure_fp8_trt.py - OpenVLA FP8 + TensorRT 端到端 + 模块级时延采集

- 风格对齐 pytorch_bf16/bench_stages.py / prof_trace.py:输出 JSONL + summary.json
- 测量 Vision engine (FP8 .plan) + LLM engine (Edge-LLM FP8) 的 E2E 时延
- 输出:
    - fp8_trt_breakdown_<tag>.jsonl     每张图片一行
    - fp8_trt_breakdown_<tag>.summary.json  包含 p50/p90/mean
    - fp8_trt_breakdown_<tag>.raw.json     原始测量值(供 fp8 文档填表)

用法:
    python measure_fp8_trt.py
    python measure_fp8_trt.py --active 10 --warmup 3
    python measure_fp8_trt.py --limit 5 --image-dir /path/to/images
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
EDGE_LLM_DIR = Path(os.environ.get("EDGE_LLM_DIR", "/workspace/TensorRT-Edge-LLM"))
ARTIFACTS = REPO / "deploy/tensorrt/artifacts"
VISION_ENGINE = ARTIFACTS / "engines/vision_projector_fp8.plan"
LLM_ENGINE_DIR = ARTIFACTS / "engines/openvla_llama_fp8"
EDGELLM_BIN = EDGE_LLM_DIR / "build/examples/llm/llm_inference"
LLM_BENCH = EDGE_LLM_DIR / "build/examples/llm/llm_bench"
EDGELLM_PLUGIN_PATH = EDGE_LLM_DIR / "build/libNvInfer_edgellm_plugin.so"
SMOKE_INPUT = ARTIFACTS / "smoke_input.json"


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def stat_block(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.mean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


# ---------------------------------------------------------------------------
# Vision engine timing (调用 trtexec --dumpProfile 或直接 execute_async_v3)
# ---------------------------------------------------------------------------

def measure_vision_engine(iterations: int, warmup: int) -> dict[str, Any]:
    """用 torch.profiler 包 vision engine execute_async_v3,采集每次 kernel 耗时。

    等价于 09_prof_trace_e2e.py 的 vision 部分。
    """
    import tensorrt as trt
    import torch
    from torch.profiler import ProfilerActivity, profile

    logger = trt.Logger(trt.Logger.WARNING)
    with open(VISION_ENGINE, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    input_name = output_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_name = name

    device = torch.device("cuda:0")
    in_shape = tuple(context.get_tensor_shape(input_name))
    out_shape = tuple(context.get_tensor_shape(output_name))
    inp = torch.rand(in_shape, dtype=torch.float16, device=device).contiguous()
    out = torch.empty(out_shape, dtype=torch.float16, device=device).contiguous()
    context.set_tensor_address(input_name, inp.data_ptr())
    context.set_tensor_address(output_name, out.data_ptr())
    stream = torch.cuda.Stream()

    # Warmup
    for _ in range(warmup):
        context.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()

    # 1. 纯 wall-clock 测量 (无 profiler 开销)
    timings: list[float] = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        context.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    # 2. 用 torch.profiler 拿纯 CUDA kernel 工时 (单次,避免污染)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            context.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize()

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)

    return {
        "engine": str(VISION_ENGINE),
        "engine_size_mb": round(VISION_ENGINE.stat().st_size / 1024 / 1024, 2),
        "iterations": iterations,
        "warmup": warmup,
        "wall_clock_ms": stat_block(timings),
        "input_shape": list(in_shape),
        "output_shape": list(out_shape),
        "operator_table": table,
    }


# ---------------------------------------------------------------------------
# LLM engine timing (调 Edge-LLM llm_inference --dumpProfile)
# ---------------------------------------------------------------------------

def measure_llm_engine(iterations: int, warmup: int) -> dict[str, Any]:
    """调用 Edge-LLM llm_inference --dumpProfile,采集 prefill + decode 时延。"""
    if not EDGELLM_BIN.exists():
        return {"error": f"llm_inference not found at {EDGELLM_BIN}"}
    if not SMOKE_INPUT.exists():
        return {"error": f"smoke input not found at {SMOKE_INPUT}"}

    env = os.environ.copy()
    env["EDGELLM_PLUGIN_PATH"] = str(EDGELLM_PLUGIN_PATH)

    cmd = [
        str(EDGELLM_BIN),
        "--engineDir", str(LLM_ENGINE_DIR),
        "--inputFile", str(SMOKE_INPUT),
        "--dumpProfile",
        "--warmup", str(warmup),
        "--outputFile", "/tmp/fp8_llm_output.json",
    ]

    print(f"[LLM] running: {' '.join(cmd)}")
    prefill_runs: list[float] = []
    decode_runs: list[float] = []
    last_out = ""
    for i in range(iterations):
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
        out = proc.stdout + "\n" + proc.stderr
        last_out = out
        for line in out.splitlines():
            if "Average Time per Run" in line and "Prefill" in out[max(0, out.find(line) - 200):out.find(line)]:
                # 在 LLM Prefill 段
                try:
                    val = float(line.split(":")[-1].strip().replace("ms", ""))
                    prefill_runs.append(val)
                except ValueError:
                    pass
            if "LLM Generation - Total Runs" in line:
                # 格式: "LLM Generation - Total Runs: 6, Total GPU Time: 183.21 ms, Average: 30.54 ms"
                try:
                    parts = line.split(",")
                    avg_part = [p for p in parts if "Average" in p][0]
                    val = float(avg_part.split(":")[-1].strip().replace("ms", ""))
                    decode_runs.append(val)
                except (ValueError, IndexError):
                    pass

    return {
        "engine_dir": str(LLM_ENGINE_DIR),
        "config": json.loads((LLM_ENGINE_DIR / "config.json").read_text()),
        "iterations": iterations,
        "prefill_ms_per_run": prefill_runs,
        "decode_per_tok_ms_per_run": decode_runs,
        "prefill_ms": stat_block(prefill_runs),
        "decode_per_tok_ms": stat_block(decode_runs),
        "raw_output_tail": last_out[-2500:],
    }


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--active", type=int, default=10)
    parser.add_argument("--output-dir", default="/workspace/outputs/openvla")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    tag = args.tag if args.tag else str(int(time.time()))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / f"fp8_trt_breakdown_{tag}.raw.json"
    summary_path = output_dir / f"fp8_trt_breakdown_{tag}.summary.json"

    print("=" * 60)
    print("OpenVLA FP8 + TensorRT 端到端时延采集")
    print(f"  Vision engine: {VISION_ENGINE}")
    print(f"  LLM engine:    {LLM_ENGINE_DIR}")
    print(f"  Warmup={args.warmup}, Active={args.active}")
    print("=" * 60)

    # 1. Vision engine
    print("\n[1/2] Vision engine timing...")
    vision = measure_vision_engine(args.active, args.warmup)
    vision_wall_ms = vision["wall_clock_ms"]["mean"]
    print(f"  Vision wall-clock mean: {vision_wall_ms:.2f} ms")
    print(f"  Vision p50/p99: {vision['wall_clock_ms']['p50']:.2f} / {vision['wall_clock_ms']['p99']:.2f} ms")

    # 2. LLM engine
    print("\n[2/2] LLM engine timing (Edge-LLM llm_inference --dumpProfile)...")
    llm = measure_llm_engine(args.active, args.warmup)
    if "error" in llm:
        print(f"  ⚠️ {llm['error']}")
        llm_prefill_ms = None
        llm_decode_per_tok_ms = None
    else:
        llm_prefill_ms = llm["prefill_ms"]
        llm_decode_per_tok_ms = llm["decode_per_tok_ms"]
        print(f"  LLM Prefill mean: {llm_prefill_ms['mean']:.2f} ms (n={len(llm['prefill_ms_per_run'])})")
        print(f"  LLM Decode/tok mean: {llm_decode_per_tok_ms['mean']:.2f} ms (n={len(llm['decode_per_tok_ms_per_run'])})")

    # 3. 汇总
    bf16_eager = {
        "vision_backbone_ms": 15.44,
        "projector_ms": 0.72,
        "llm_prefill_ms": 86.30,
        "llm_decode_per_tok_ms": 58.9,  # 60.4 是 paper 报告, 58.9 是算术均值
        "lm_head_ms": 1.10,
        "e2e_ms": 466.24,
    }

    # 4. 计算 E2E 估算
    if "error" not in llm and "wall_clock_ms" in vision:
        e2e_ms = vision["wall_clock_ms"]["mean"] + llm["prefill_ms"]["mean"] + 6 * llm["decode_per_tok_ms"]["mean"]
        fp8_summary = {
            "vision_ms": round(vision["wall_clock_ms"]["mean"], 2),
            "llm_prefill_ms": round(llm["prefill_ms"]["mean"], 2),
            "llm_decode_per_tok_ms": round(llm["decode_per_tok_ms"]["mean"], 2),
            "decode_6_steps_ms": round(6 * llm["decode_per_tok_ms"]["mean"], 2),
            "e2e_ms_estimated": round(e2e_ms, 2),
            "speedup_vs_bf16": round(bf16_eager["e2e_ms"] / e2e_ms, 2),
        }
    else:
        fp8_summary = {"error": "missing measurements"}

    summary = {
        "tag": tag,
        "env": {
            "gpu": "NVIDIA Thor (SM110, 20 SMs, LPDDR5X 122.86 GB)",
            "trt_version": "10.16.1.11",
            "edgellm_version": "0.9.0",
            "cuda_version": "13.2",
            "torch_version": "2.12.0a0+5aff3928",
        },
        "vision_engine": vision,
        "llm_engine": llm,
        "bf16_eager_baseline": bf16_eager,
        "fp8_summary": fp8_summary,
        "notes": (
            "FP8 vision 用 torch.profiler + execute_async_v3 计时(含 H2D+kernel+D2H)。\n"
            "FP8 LLM 用 Edge-LLM llm_inference --dumpProfile 计时(纯文本 prompt,非真实图像 embedding)。\n"
            "E2E = vision + LLM prefill + 6×(LLM decode/tok) - 注意口径:不一定等于实际端到端推理。"
        ),
    }

    raw_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ Raw data: {raw_path}")
    print(f"  Vision engine: {vision_wall_ms:.2f} ms (mean)")
    if "error" in llm:
        print(f"  ⚠️ LLM 测量失败, 请人工运行 {EDGELLM_BIN} --dumpProfile")


if __name__ == "__main__":
    main()
