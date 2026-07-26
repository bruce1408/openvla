#!/usr/bin/env python3
# =============================================================================
# prof_trace_e2e.py - TensorRT 端到端推理 trace 分析 (vision + LLM)
# =============================================================================
#
# 目标:
#   对 TensorRT 部署链路做算子级 trace,生成 Chrome trace / operator table /
#   flamegraph,类似 pytorch_bf16/prof_trace.py 对 PyTorch 模型做的事。
#
# 两个阶段分别 trace:
#   1. Vision engine (DINOv2+SigLIP+projector)
#      用 torch.profiler 包裹 TensorRT execute_async_v3,捕获 CUDA kernel 级 trace。
#      TensorRT 的 kernel 对 torch.profiler 可见(都走 CUDA runtime)。
#
#   2. LLM engine (Edge-LLM prefill+decode)
#      Edge-LLM 是 C++ binary,torch.profiler 无法直接 attach。分两层拿数据:
#      (a) llm_inference --dumpProfile: 分阶段 (prefill/decode) 的 gpu_time_stats;
#      (b) llm_bench --profile: decode 逐层 CSV (category: mha/gemm/kgen_other),
#          用于定位 LLM 内部瓶颈算子。
#
# 输出产物:
#   - vision_trace.json        Chrome trace (Perfetto 可视化)
#   - vision_table.txt         vision operator 耗时表 (按 CUDA time 排序)
#   - llm_profile.json         Edge-LLM 分阶段 profile 原始数据
#   - llm_layers_*/layer_*.csv LLM decode 逐层耗时 CSV
#   - e2e_prof_summary.json    汇总报告
#
# 说明: TensorRT engine 的 CUDA kernel 没有 Python 调用栈,故不导出 flamegraph
#       (对 TRT 无意义,只会产生空文件)。用 Chrome trace + operator table 分析。
#
# 用法:
#   python prof_trace_e2e.py --precision fp8
#   python prof_trace_e2e.py --precision nvfp4 --warmup 3 --active 10
#   python prof_trace_e2e.py --precision fp8 --skip-vision   # 只 trace LLM
#   python prof_trace_e2e.py --precision fp8 --skip-llm      # 只 trace vision
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

EDGE_LLM_DIR = Path(os.environ.get("EDGE_LLM_DIR", "/workspace/TensorRT-Edge-LLM"))
OPENVLA_DIR = Path(os.environ.get("OPENVLA_DIR", "/workspace/openvla"))
ARTIFACTS = OPENVLA_DIR / "deploy/tensorrt/artifacts"
LOGS_DIR = Path(os.environ.get("OPENVLA_LOGS_DIR", "/workspace/outputs/openvla"))


def infer_precision(path: Path | str) -> str:
    """从 engine/文件名推断数据精度 (fp8 / fp16 / nvfp4 / bf16),用于标注输出。"""
    name = str(path).lower()
    for prec in ("nvfp4", "fp16", "fp8", "bf16", "int8", "fp32"):
        if prec in name:
            return prec
    return "unknown"


# ---------------------------------------------------------------------------
# Vision engine profiling (torch.profiler 包裹 TensorRT execution)
# ---------------------------------------------------------------------------

def load_vision_engine(engine_path: Path):
    """加载 TensorRT vision engine,返回 (context, input_name, output_name, stream)。"""
    import tensorrt as trt
    import torch

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
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
    return context, input_name, output_name, in_shape, out_shape, inp, out, stream


def profile_vision(engine_path: Path, warmup: int, active: int, output_dir: Path, tag: str) -> dict:
    """用 torch.profiler trace vision engine 的 CUDA kernel。"""
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function, schedule

    vision_precision = infer_precision(engine_path)
    context, in_name, out_name, in_shape, out_shape, inp, out, stream = load_vision_engine(engine_path)

    # 预热 (profiler 之前的独立 warmup,确保 kernel cache 就绪)
    for _ in range(warmup):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    # 文件名嵌入 vision 精度,便于区分 fp16 / fp8 的 trace 产物
    base = output_dir / f"vision_trace_{vision_precision}_{tag}"
    chrome_path = base.with_suffix(".trace.json")
    table_path = base.with_suffix(".table.txt")

    sched = schedule(wait=1, warmup=2, active=active, repeat=1)
    total_steps = 1 + 2 + active

    print(f"  vision profiling: {active} active steps (wait=1, warmup=2)")
    print(f"  input_shape={in_shape}, output_shape={out_shape}")

    # 不用 with_stack: TensorRT engine 的 CUDA kernel 没有 Python 调用栈,
    # export_stacks 只会产生空文件,对 TRT 场景无意义 (不同于 PyTorch 有 Python 层栈)。
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for step in range(total_steps):
            with record_function("vision_inference"):
                context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            prof.step()

    # Chrome trace
    prof.export_chrome_trace(str(chrome_path))
    print(f"  chrome trace: {chrome_path}")

    # Operator table
    table = prof.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total", row_limit=30
    )
    table_path.write_text(table, encoding="utf-8")
    print(f"  operator table: {table_path}")

    # 提取汇总统计
    events = prof.key_averages()

    # 用 SELF CUDA time 统计真实 kernel 总耗时。
    # 注意: record_function("vision_inference") 是一个包裹层,它的 total CUDA time
    # 等于所有子 kernel 之和;若用 total 求和会与子节点重复计数(double count)。
    # self time 只算节点自身、不含子节点,包裹层的 self time≈0,因此对所有节点的
    # self time 求和 = 不重复的真实 kernel 总时间。
    def get_self_cuda(e) -> float:
        for attr in ("self_cuda_time_total", "self_device_time_total"):
            val = getattr(e, attr, 0)
            if val and val > 0:
                return float(val)
        return 0.0

    def get_total_cuda(e) -> float:
        for attr in ("cuda_time_total", "device_time_total"):
            val = getattr(e, attr, 0)
            if val and val > 0:
                return float(val)
        return 0.0

    # 真实 kernel = 有 self CUDA time 的节点,且排除我们自己加的 record_function 注解
    # ("vision_inference")——它是包裹层,新版 torch 会给它归因部分 CUDA time,
    # 计入会导致总和偏高。
    WRAPPER_NAMES = {"vision_inference"}
    kernel_ops = [e for e in events if get_self_cuda(e) > 0 and e.key not in WRAPPER_NAMES]
    kernel_ops.sort(key=get_self_cuda, reverse=True)

    total_self_us = sum(get_self_cuda(e) for e in kernel_ops)
    total_self_ms_per_step = (total_self_us / 1000.0) / active
    # top5 用 self time 排序 (真实最耗时的 kernel)
    top5 = [(e.key, round(get_self_cuda(e) / 1000.0 / active, 4), e.count // active if e.count >= active else e.count) for e in kernel_ops[:5]]

    return {
        "precision": vision_precision,
        "engine": str(engine_path),
        "total_cuda_time_ms": round(total_self_ms_per_step, 2),
        "top5_operators": top5,
        "operator_count": len(kernel_ops),
        "input_shape": list(in_shape),
        "output_shape": list(out_shape),
        "active_steps": active,
        "artifacts": {
            "chrome_trace": str(chrome_path),
            "operator_table": str(table_path),
        },
    }


# ---------------------------------------------------------------------------
# LLM engine profiling (Edge-LLM --dumpProfile)
# ---------------------------------------------------------------------------

def build_long_prompt(target_tokens: int = 262) -> str:
    """构造 tokenize 后接近 target_tokens 的 prompt (与 07_measure_e2e_latency.py 一致)。"""
    base = "In: What action should the robot take to "
    reps = max(1, (target_tokens - 12) // 23)
    filler = "pick up the blue object and place it on the table " * reps
    return base + filler + "?\nOut:"


def profile_llm(engine_dir: Path, action_dim: int, warmup: int, active: int, output_dir: Path, tag: str) -> dict:
    """用 Edge-LLM --dumpProfile 获取 prefill/decode 的分阶段 GPU 时延。"""
    llm_precision = infer_precision(engine_dir)
    prompt = build_long_prompt()
    # 文件名嵌入 LLM 精度,便于区分 fp8 / nvfp4 的 profile 产物
    probe_in = output_dir / f"llm_input_{llm_precision}_{tag}.json"
    probe_prof = output_dir / f"llm_profile_{llm_precision}_{tag}.json"
    probe_out = output_dir / f"llm_output_{llm_precision}_{tag}.json"

    # 放 active 个相同 request,拿统计中位数
    one_req = {"messages": [{"role": "user", "content": prompt}]}
    payload = {
        "apply_chat_template": False,
        "max_generate_length": action_dim,
        "temperature": 0.0,
        "requests": [dict(one_req) for _ in range(active)],
    }
    probe_in.write_text(json.dumps(payload))

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib/aarch64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")
    env["EDGELLM_PLUGIN_PATH"] = str(EDGE_LLM_DIR / "build/libNvInfer_edgellm_plugin.so")

    cmd = [
        str(EDGE_LLM_DIR / "build/examples/llm/llm_inference"),
        "--engineDir", str(engine_dir),
        "--inputFile", str(probe_in),
        "--outputFile", str(probe_out),
        "--warmup", str(warmup),
        "--dumpProfile",
        "--profileOutputFile", str(probe_prof),
        "--maxGenerateLength", str(action_dim),
    ]

    print(f"  LLM profiling: {active} requests, warmup={warmup}")
    subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    prof = json.loads(probe_prof.read_text())
    print(f"  edge-llm profile: {probe_prof}")

    # 提取 prefill 和 decode 统计
    def get_stage(stage_id: str) -> dict:
        for stage in prof.get("stages", []):
            if stage.get("stage_id") == stage_id:
                return stage.get("gpu_time_stats", {})
        return {}

    prefill = get_stage("llm_prefill")
    decode = get_stage("llm_generation")

    # decode 总时间 = 每 token 中位数 × (action_dim - 1)
    decode_per_token_median = decode.get("median_ms", 0.0)
    decode_steps = max(action_dim - 1, 0)
    decode_total = decode_per_token_median * decode_steps

    result = {
        "precision": llm_precision,
        "engine": str(engine_dir),
        "prefill_median_ms": prefill.get("median_ms", 0.0),
        "prefill_mean_ms": prefill.get("mean_ms", 0.0),
        "prefill_count": prefill.get("count", 0),
        "prefill_tokens": prof.get("prefill", {}).get("computed_tokens", 0),
        "decode_per_token_median_ms": decode_per_token_median,
        "decode_per_token_mean_ms": decode.get("mean_ms", 0.0),
        "decode_steps_per_action": decode_steps,
        "decode_total_median_ms": round(decode_total, 2),
        "peak_memory_mb": prof.get("peak_unified_memory_mb"),
        "artifacts": {
            "llm_profile": str(probe_prof),
        },
    }

    # --- LLM 层级 profiling (llm_bench --profile 生成逐层 CSV) ---
    layer = profile_llm_layers(engine_dir, action_dim, warmup, output_dir, tag, llm_precision)
    if layer:
        result["layer_breakdown"] = layer

    return result


def profile_llm_layers(engine_dir: Path, action_dim: int, warmup: int, output_dir: Path, tag: str, precision: str = "unknown") -> dict:
    """用 llm_bench --profile 做 LLM 的逐层 profiling (decode 阶段),解析 CSV。

    Edge-LLM 不把内部 kernel 暴露给外部 profiler,但 llm_bench --profile 会输出
    逐层耗时 CSV (含 category: mha/gemm/kgen_other),用于定位 LLM 的瓶颈算子。
    注意: 层级 profiling 走非 CUDA-graph 路径,单层时间之和会略高于 CUDA-graph 的
    真实时延,仅用于"相对占比/瓶颈定位",不用作绝对时延。
    """
    import csv

    bench_bin = EDGE_LLM_DIR / "build/examples/llm/llm_bench"
    if not bench_bin.exists():
        print("  (跳过层级 profiling: 找不到 llm_bench)")
        return {}

    csv_dir = output_dir / f"llm_layers_{precision}_{tag}"
    csv_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib/aarch64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")
    env["EDGELLM_PLUGIN_PATH"] = str(EDGE_LLM_DIR / "build/libNvInfer_edgellm_plugin.so")

    # decode 阶段逐层 profiling (decode 是瓶颈,占端到端 ~70%)
    cmd = [
        str(bench_bin),
        "--engineDir", str(engine_dir),
        "--mode", "decode",
        "--pastKVLen", "262",
        "--osl", str(max(action_dim - 1, 1)),
        "--warmup", str(warmup),
        "--iterations", "10",
        "--profile",
        "--extractLayerInfo", "shapes,onnx_ops",
        "--outputDir", str(csv_dir),
    ]
    print(f"  LLM 层级 profiling (decode, --profile)...")
    subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 找生成的 CSV
    csv_files = list(csv_dir.glob("layer_*.csv"))
    if not csv_files:
        print("  (未生成层级 CSV)")
        return {}
    csv_path = csv_files[0]

    # 解析 CSV: 按 category 汇总 + top 层
    category_ms: dict[str, float] = {}
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                t = float(row.get("time_ms_mean", 0) or 0)
            except ValueError:
                t = 0.0
            cat = row.get("category", "unknown") or "unknown"
            category_ms[cat] = category_ms.get(cat, 0.0) + t
            rows.append((row.get("layer_name", "")[:60], row.get("onnx_op", ""), cat, t))

    rows.sort(key=lambda r: r[3], reverse=True)
    total_ms = sum(category_ms.values())
    top10 = [{"layer": r[0], "onnx_op": r[1], "category": r[2], "time_ms": round(r[3], 4)} for r in rows[:10]]
    category_summary = {
        k: {"time_ms": round(v, 4), "pct": round(v / total_ms * 100, 1) if total_ms else 0.0}
        for k, v in sorted(category_ms.items(), key=lambda kv: kv[1], reverse=True)
    }

    print(f"  层级 CSV: {csv_path}")
    print(f"  单步 decode 层级耗时合计(非graph): {total_ms:.3f} ms | 层数: {len(rows)}")
    return {
        "csv": str(csv_path),
        "total_layer_ms_per_decode_step": round(total_ms, 3),
        "num_layers": len(rows),
        "category_breakdown": category_summary,
        "top10_layers": top10,
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TensorRT e2e trace analysis: vision (torch.profiler) + LLM (Edge-LLM --dumpProfile)."
    )
    parser.add_argument("--precision", choices=("nvfp4", "fp8"), default="fp8")
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3, help="profiler warmup steps (vision) / llm_inference warmup")
    parser.add_argument("--active", type=int, default=10, help="recorded steps (vision) / llm requests")
    parser.add_argument("--vision-engine", default=str(ARTIFACTS / "engines/vision_projector_fp16.plan"))
    parser.add_argument("--skip-vision", action="store_true", help="跳过 vision profiling")
    parser.add_argument("--skip-llm", action="store_true", help="跳过 LLM profiling")
    parser.add_argument("--output-dir", default=None, help="trace 产物输出目录")
    parser.add_argument("--tag", default=None, help="文件名标签 (默认时间戳)")
    args = parser.parse_args()

    llm_engine_dir = ARTIFACTS / f"engines/openvla_llama_{args.precision}"
    vision_engine = Path(args.vision_engine)

    # 前置检查
    if not args.skip_vision and not vision_engine.exists():
        sys.exit(f"错误: 找不到 vision engine: {vision_engine}")
    if not args.skip_llm and not (llm_engine_dir / "llm.engine").exists():
        sys.exit(f"错误: 找不到 LLM engine: {llm_engine_dir}/llm.engine")

    output_dir = Path(args.output_dir) if args.output_dir else LOGS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag if args.tag else time.strftime("%Y_%m%d_%H%M%S")

    # 记录两个精度: vision engine 与 LLM engine 各自的量化精度
    vision_precision = infer_precision(vision_engine) if not args.skip_vision else None
    llm_precision = infer_precision(llm_engine_dir) if not args.skip_llm else None

    print("=" * 64)
    print(f"TensorRT E2E Trace Analysis")
    print(f"  vision engine : {vision_engine if not args.skip_vision else 'SKIPPED'}  [{vision_precision or '-'}]")
    print(f"  LLM engine    : {llm_engine_dir if not args.skip_llm else 'SKIPPED'}  [{llm_precision or '-'}]")
    print(f"  action_dim={args.action_dim} | warmup={args.warmup} | active={args.active}")
    print(f"  output_dir    : {output_dir}")
    print(f"  tag           : {tag}")
    print("=" * 64)

    result = {
        "llm_precision": llm_precision,
        "vision_precision": vision_precision,
        "precision": args.precision,  # 兼容旧字段: --precision 指定的 LLM 精度
        "action_dim": args.action_dim,
        "config": {"warmup": args.warmup, "active": args.active},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # --- 1. Vision trace ---
    if not args.skip_vision:
        print(f"\n>>> [1/2] Vision engine trace (torch.profiler → Chrome trace)...")
        vision_result = profile_vision(vision_engine, args.warmup, args.active, output_dir, tag)
        result["vision"] = vision_result

        print(f"\n  --- Vision Summary ---")
        print(f"  total CUDA time (per step): {vision_result['total_cuda_time_ms']:.2f} ms")
        print(f"  operator count: {vision_result['operator_count']}")
        print(f"  top-5 operators (name, cuda_ms, count):")
        for name, t, cnt in vision_result["top5_operators"]:
            print(f"    {name[:60]:60s}  {t:.4f} ms  ×{cnt}")

    # --- 2. LLM trace ---
    if not args.skip_llm:
        print(f"\n>>> [2/2] LLM engine trace (Edge-LLM --dumpProfile)...")
        llm_result = profile_llm(llm_engine_dir, args.action_dim, args.warmup, args.active, output_dir, tag)
        result["llm"] = llm_result

        print(f"\n  --- LLM Summary ---")
        print(f"  prefill median : {llm_result['prefill_median_ms']:.2f} ms  (tokens={llm_result['prefill_tokens']}, count={llm_result['prefill_count']})")
        print(f"  decode/token   : {llm_result['decode_per_token_median_ms']:.2f} ms  × {llm_result['decode_steps_per_action']} = {llm_result['decode_total_median_ms']:.2f} ms")
        print(f"  peak memory    : {llm_result['peak_memory_mb']} MB")
        lb = llm_result.get("layer_breakdown")
        if lb:
            print(f"\n  --- LLM 层级分类 (decode, 相对占比) ---")
            for cat, v in lb["category_breakdown"].items():
                print(f"    {cat:14s}  {v['time_ms']:8.4f} ms  ({v['pct']:4.1f}%)")
            print(f"  top-3 耗时层:")
            for it in lb["top10_layers"][:3]:
                print(f"    {it['layer'][:50]:50s}  {it['time_ms']:.4f} ms  [{it['category']}]")

    # --- 汇总 ---
    print("\n" + "=" * 64)
    print("E2E Trace Summary")
    print("-" * 64)

    v_ms = result.get("vision", {}).get("total_cuda_time_ms", 0.0)
    p_ms = result.get("llm", {}).get("prefill_median_ms", 0.0)
    d_ms = result.get("llm", {}).get("decode_total_median_ms", 0.0)
    e2e = v_ms + p_ms + d_ms
    hz = 1000.0 / e2e if e2e > 0 else 0.0

    if not args.skip_vision:
        print(f"  Vision  [{vision_precision:>5}]: {v_ms:8.2f} ms  ({v_ms/e2e*100:4.1f}%)  [torch.profiler CUDA total]")
    if not args.skip_llm:
        print(f"  Prefill [{llm_precision:>5}]: {p_ms:8.2f} ms  ({p_ms/e2e*100:4.1f}%)  [Edge-LLM gpu_time median]")
        print(f"  Decode  [{llm_precision:>5}]: {d_ms:8.2f} ms  ({d_ms/e2e*100:4.1f}%)  [Edge-LLM gpu_time median × steps]")
    print("-" * 64)
    print(f"  E2E     : {e2e:8.2f} ms  (~{hz:.1f} Hz)")
    print("=" * 64)

    # 保存汇总 (文件名含 vision/LLM 精度,例: e2e_prof_summary_vfp16_lfp8_<tag>.json)
    prec_tag_parts = []
    if vision_precision:
        prec_tag_parts.append(f"v{vision_precision}")
    if llm_precision:
        prec_tag_parts.append(f"l{llm_precision}")
    prec_tag = "_".join(prec_tag_parts) if prec_tag_parts else "na"
    summary_path = output_dir / f"e2e_prof_summary_{prec_tag}_{tag}.json"
    result["e2e_median_ms"] = round(e2e, 2)
    result["hz"] = round(hz, 1)
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n汇总报告: {summary_path}")

    print("\n查看方式:")
    if not args.skip_vision:
        print(f"  Vision Chrome trace : 在 https://ui.perfetto.dev 打开 {result['vision']['artifacts']['chrome_trace']}")
        print(f"  Vision operator table: {result['vision']['artifacts']['operator_table']}")
    if not args.skip_llm:
        print(f"  LLM stage profile  : {result['llm']['artifacts']['llm_profile']}")
        lb = result['llm'].get("layer_breakdown")
        if lb:
            print(f"  LLM layer CSV      : {lb['csv']}")


if __name__ == "__main__":
    main()
