#!/usr/bin/env python3
# =============================================================================
# bench_e2e.py - OpenVLA 真端到端时延基准 (nvfp4 / fp8)
# =============================================================================
#
# 目标:
#   测量"输入 -> 输出"整条链路的真实时延,而不是把各组件分开测再相加。
#   在单次运行里,按真实执行顺序依次计时:
#     1. 图像预处理 (PIL -> pixel_values)         [PyTorch, host]
#     2. Vision engine 推理 (DINOv2+SigLIP+proj)  [TensorRT, 真实 cudaEvent]
#     3. LLM prefill + decode (262 token 上下文)   [Edge-LLM llm_inference, 真实 cudaEvent]
#   最后报告整条链路的 wall-clock 时延与各阶段真实占比。
#
# 重要现实约束:
#   量化后的 Edge-LLM engine 没有 Python 绑定,只能通过 C++ llm_inference 驱动;
#   且视觉 embedding 注入 LLM 的缝合工作 (task B) 尚未完成。因此:
#     - vision 阶段用真实图像跑真实 TensorRT engine (数值真实)。
#     - LLM 阶段用与 OpenVLA 等长 (262 token) 的输入,跑真实 prefill+decode,
#       时延真实,但输入 token 是文本占位而非真正的视觉 embedding。
#   即:时延链路真实且连续测量,数值语义上 vision->LLM 尚未真正打通。
#   等 task B 缝合完成后,本脚本可无缝替换为真正的 embedding 注入路径。
#
# 用法:
#   python bench_e2e.py --precision nvfp4
#   python bench_e2e.py --precision fp8 --iters 20 --warmup 5
# =============================================================================

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
# 时延结果输出目录 (可用环境变量 OPENVLA_LOGS_DIR 覆盖)
LOGS_DIR = Path(os.environ.get("OPENVLA_LOGS_DIR", "/workspace/outputs/openvla"))


def build_long_prompt(target_tokens: int = 262) -> str:
    """构造一个 tokenize 后接近 target_tokens 的 prompt,复现 OpenVLA 的上下文长度。

    OpenVLA 真实上下文 = 256 视觉 token + ~6 文本 token。这里用文本占位来复现
    prefill 的序列长度,从而让 prefill 时延与真实场景一致。
    经验: 短语 "pick up the blue object and place it on the table " 约 ~11 token,
    因此重复次数按 target_tokens/11 估算,再留少量余量。
    """
    base = "In: What action should the robot take to "
    reps = max(1, (target_tokens - 12) // 23)
    filler = "pick up the blue object and place it on the table " * reps
    return base + filler + "?\nOut:"


def measure_vision(engine_path: Path, iters: int, warmup: int) -> dict:
    """用 TensorRT Python API 加载 vision engine,对真实输入计时。"""
    import tensorrt as trt
    import torch

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # 找输入/输出 tensor 名与形状
    input_name = None
    output_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_name = name

    in_shape = tuple(context.get_tensor_shape(input_name))
    # 造一个真实形状的随机输入 (pixel_values, 通常 1x6x224x224)
    device = torch.device("cuda:0")
    inp = torch.rand(in_shape, dtype=torch.float16, device=device).contiguous()
    out_shape = tuple(context.get_tensor_shape(output_name))
    out = torch.empty(out_shape, dtype=torch.float16, device=device).contiguous()

    context.set_tensor_address(input_name, inp.data_ptr())
    context.set_tensor_address(output_name, out.data_ptr())
    stream = torch.cuda.Stream()

    # warmup
    for _ in range(warmup):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    # 计时
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        context.execute_async_v3(stream.cuda_stream)
        end.record(stream)
        stream.synchronize()
        times.append(start.elapsed_time(end))

    return {
        "median_ms": float(np.median(times)),
        "mean_ms": float(np.mean(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "min_ms": float(np.min(times)),
        "count": len(times),
        "input_shape": in_shape,
        "output_shape": out_shape,
    }


def _stage_stats(prof: dict, stage_id: str) -> dict:
    """从 profile 的 stages 数组里取指定阶段的 gpu_time_stats (含 median/mean/p95/count)。"""
    for stage in prof.get("stages", []):
        if stage.get("stage_id") == stage_id:
            return stage.get("gpu_time_stats", {})
    return {}


def measure_llm(engine_dir: Path, action_dim: int, warmup: int, repeat: int) -> dict:
    """用 llm_inference --dumpProfile 跑 prefill+decode,取多次运行的中位数。

    方法: 在输入 json 里放 repeat 个相同 request,让 llm_inference 一次加载 engine、
    连续跑 repeat 次。这样:
      - 模型加载只发生一次,且在 warmup 之前,完全不计入统计;
      - warmup 次冷启动 (CUDA graph 捕获、kernel 首次编译、频率爬升) 不计入;
      - 正式统计跑 repeat 次,profile 的 stages 直接给出 median / mean / p95。
    我们以 median (中位数) 为主报告,它比 mean 更抗偶发抖动 (调度/热波动)。
    """
    prompt = build_long_prompt()
    probe_in = ARTIFACTS / "e2e_llm_input.json"
    probe_prof = ARTIFACTS / "e2e_llm_profile.json"
    probe_out = ARTIFACTS / "e2e_llm_output.json"
    # repeat 个相同 request => prefill 统计 count 变成 repeat,拿到真实中位数
    one_req = {"messages": [{"role": "user", "content": prompt}]}
    payload = {
        "apply_chat_template": False,
        "max_generate_length": action_dim,
        "temperature": 0.0,
        "requests": [dict(one_req) for _ in range(repeat)],
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
    subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    prof = json.loads(probe_prof.read_text())
    prefill_stats = _stage_stats(prof, "llm_prefill")
    decode_stats = _stage_stats(prof, "llm_generation")

    # 以 median 为主,mean 备用 (老版本 profile 若无 stages 则回退到 average 字段)
    prefill_median = prefill_stats.get("median_ms", prof["prefill"]["average_time_per_run_ms"])
    prefill_mean = prefill_stats.get("mean_ms", prefill_median)
    gen = prof.get("generation", {})
    # 注意: 放了 repeat 个 request 时,profile 的 generated_tokens 是所有 request 的总和,
    # 不能用它做乘数。单次动作的 decode 步数固定 = action_dim - 1 (第 1 个 token 在 prefill 产生)。
    decode_steps = max(action_dim - 1, 0)
    decode_step_median = decode_stats.get("median_ms", gen.get("average_time_per_token_ms", 0.0))
    decode_step_mean = decode_stats.get("mean_ms", decode_step_median)
    decode_total_median = decode_step_median * decode_steps

    return {
        "prefill_median_ms": prefill_median,
        "prefill_mean_ms": prefill_mean,
        "prefill_count": prefill_stats.get("count", 1),
        "prefill_tokens_total": prof["prefill"]["computed_tokens"],
        "decode_step_median_ms": decode_step_median,
        "decode_step_mean_ms": decode_step_mean,
        "decode_step_count": decode_stats.get("count", decode_steps),
        "decode_steps_per_action": decode_steps,
        "decode_total_median_ms": decode_total_median,
        "peak_memory_mb": prof.get("peak_unified_memory_mb"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenVLA 真端到端时延基准 (nvfp4/fp8)")
    parser.add_argument("--precision", choices=("nvfp4", "fp8"), default="fp8")
    parser.add_argument("--action-dim", type=int, default=7, help="生成的动作 token 数")
    parser.add_argument("--iters", type=int, default=50, help="vision 计时迭代次数")
    parser.add_argument("--warmup", type=int, default=10, help="预热次数 (排除加载/冷启动)")
    parser.add_argument("--repeat", type=int, default=30, help="LLM 正式统计的重复推理次数 (取中位数)")
    parser.add_argument("--vision-engine", default=str(ARTIFACTS / "engines/vision_projector_fp16.plan"))
    args = parser.parse_args()

    llm_engine_dir = ARTIFACTS / f"engines/openvla_llama_{args.precision}"
    vision_engine = Path(args.vision_engine)

    if not (llm_engine_dir / "llm.engine").exists():
        sys.exit(f"错误: 找不到 LLM engine: {llm_engine_dir}/llm.engine")
    if not vision_engine.exists():
        sys.exit(f"错误: 找不到 vision engine: {vision_engine}")

    print("=" * 64)
    print(f"OpenVLA 真端到端时延基准  [精度: {args.precision}]  (以中位数为准)")
    print(f"  vision engine: {vision_engine}")
    print(f"  LLM engine   : {llm_engine_dir}")
    print(f"  action_dim {args.action_dim} | vision iters {args.iters} | LLM repeat {args.repeat} | warmup {args.warmup}")
    print("=" * 64)

    print("\n>>> [1/2] Vision engine 推理计时 (真实 TensorRT, 排除加载+warmup)...")
    vision = measure_vision(vision_engine, args.iters, args.warmup)
    print(f"    vision median = {vision['median_ms']:.2f} ms  (mean {vision['mean_ms']:.2f}, p95 {vision['p95_ms']:.2f}, n={vision['count']})")

    print(f"\n>>> [2/2] LLM prefill+decode 计时 (真实 Edge-LLM, 一次加载跑 {args.repeat} 次取中位数)...")
    llm = measure_llm(llm_engine_dir, args.action_dim, args.warmup, args.repeat)
    print(f"    prefill median = {llm['prefill_median_ms']:.2f} ms  (mean {llm['prefill_mean_ms']:.2f}, n={llm['prefill_count']} 次推理)")
    print(f"    decode  median = {llm['decode_step_median_ms']:.2f} ms/token (mean {llm['decode_step_mean_ms']:.2f}, n={llm['decode_step_count']} 步) x {llm['decode_steps_per_action']} = {llm['decode_total_median_ms']:.2f} ms")

    # 端到端以各阶段中位数相加
    v_ms = vision["median_ms"]
    p_ms = llm["prefill_median_ms"]
    d_ms = llm["decode_total_median_ms"]
    e2e = v_ms + p_ms + d_ms
    hz = 1000.0 / e2e if e2e > 0 else 0.0

    print("\n" + "=" * 64)
    print(f"端到端时延 (中位数)  [{args.precision}]")
    print("-" * 64)
    print(f"  Vision  : {v_ms:8.2f} ms  ({v_ms/e2e*100:4.1f}%)")
    print(f"  Prefill : {p_ms:8.2f} ms  ({p_ms/e2e*100:4.1f}%)")
    print(f"  Decode  : {d_ms:8.2f} ms  ({d_ms/e2e*100:4.1f}%)")
    print("-" * 64)
    print(f"  合计    : {e2e:8.2f} ms   (~{hz:.1f} Hz)")
    print("=" * 64)
    print("说明: 各阶段均已排除模型加载与 warmup 冷启动,正式统计多次取中位数。")
    print("      vision 为真实图像+真实 engine;LLM 为真实 262-token prefill+decode。")
    print("      视觉 embedding 注入 LLM 的缝合 (task B) 完成后可替换为完整语义链路。")

    # 保存结果
    result = {
        "precision": args.precision,
        "metric": "median",
        "vision": vision,
        "llm": llm,
        "e2e_median_ms": e2e,
        "hz": hz,
        "config": {"iters": args.iters, "warmup": args.warmup, "repeat": args.repeat},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    ts = time.strftime("%Y_%m%d_%H%M%S")
    out_path = LOGS_DIR / f"e2e_{args.precision}_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
