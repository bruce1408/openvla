"""
延迟评测脚本 (latency breakdown)

对 test_data/ 下的真实图片逐张评测 OpenVLA 的推理延迟，并把端到端延迟拆解成各阶段：
  1. preprocess      图像预处理 (processor: resize/normalize/tokenize)
  2. h2d              host -> device 数据搬运
  3. vision_backbone DINOv2 + SigLIP 融合视觉特征提取
  4. projector        视觉特征 -> LLM embedding 的 MLP 投影
  5. llm_prefill      LLM 处理 (prompt + 图像 patch) 的首次 forward (含首个动作 token)
  6. llm_decode       后续自回归逐 token 解码 (每个动作维度一个 token)
  7. postprocess      反离散化 + 反归一化 (CPU/numpy)

OpenVLA 的动作预测是自回归的：predict_action 内部调用 generate()，
逐个生成 7 个 token (7 维动作)，每个 token 需要一次 LLM forward。

结果写入 JSONL (每张图片一行) + 一个 summary.json (统计 p50/p90/p95/p99)。
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

RUNTIME_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RUNTIME_DIR))
from runtime_env import MODEL_PATH, MODEL_REVISION

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


MODEL_ID = str(MODEL_PATH)
DEVICE = os.getenv("OPENVLA_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
ATTN_IMPLEMENTATION = os.getenv("OPENVLA_ATTN_IMPLEMENTATION", "sdpa")
UNNORM_KEY = os.getenv("OPENVLA_UNNORM_KEY", "bridge_orig")

DEFAULT_IMAGE_DIR = RUNTIME_DIR / "test_data"

# 用于在各阶段打点的全局计时表 (由 hook 填充)
_STAGE_TIMES: dict[str, float] = {}


def model_dtype() -> torch.dtype:
    return torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32


def sync() -> None:
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()


def prompt_for(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[index]


def stat_block(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


def register_stage_hooks(model: Any) -> list[Any]:
    """给 vision_backbone / projector / language_model 挂 hook，测量其耗时。

    llm 会被多次调用 (prefill + 每个 decode token)，这里累加总时间并单独记录首次 (prefill)。
    """
    handles: list[Any] = []
    state = {"vb_start": 0.0, "proj_start": 0.0, "llm_calls": 0, "llm_start": 0.0}

    def vb_pre(_m: Any, _i: Any) -> None:
        sync()
        state["llm_calls"] = 0  # 每次 predict 开始时重置 LLM 调用计数
        state["vb_start"] = time.perf_counter()

    def vb_post(_m: Any, _i: Any, _o: Any) -> None:
        sync()
        _STAGE_TIMES["vision_backbone_ms"] = (time.perf_counter() - state["vb_start"]) * 1000.0

    def proj_pre(_m: Any, _i: Any) -> None:
        sync()
        state["proj_start"] = time.perf_counter()

    def proj_post(_m: Any, _i: Any, _o: Any) -> None:
        sync()
        _STAGE_TIMES["projector_ms"] = (time.perf_counter() - state["proj_start"]) * 1000.0

    def llm_pre(_m: Any, _i: Any) -> None:
        sync()
        state["llm_start"] = time.perf_counter()

    def llm_post(_m: Any, _i: Any, _o: Any) -> None:
        sync()
        elapsed = (time.perf_counter() - state["llm_start"]) * 1000.0
        state["llm_calls"] += 1
        if state["llm_calls"] == 1:
            _STAGE_TIMES["llm_prefill_ms"] = elapsed
            _STAGE_TIMES["llm_decode_ms"] = 0.0
            _STAGE_TIMES["llm_decode_tokens"] = 0
        else:
            _STAGE_TIMES["llm_decode_ms"] += elapsed
            _STAGE_TIMES["llm_decode_tokens"] += 1

    handles.append(model.vision_backbone.register_forward_pre_hook(vb_pre))
    handles.append(model.vision_backbone.register_forward_hook(vb_post))
    handles.append(model.projector.register_forward_pre_hook(proj_pre))
    handles.append(model.projector.register_forward_hook(proj_post))
    handles.append(model.language_model.register_forward_pre_hook(llm_pre))
    handles.append(model.language_model.register_forward_hook(llm_post))
    return handles


def timed_predict(model: Any, processor: Any, image_path: Path, instruction: str) -> dict[str, Any]:
    _STAGE_TIMES.clear()
    prompt = prompt_for(instruction)

    # 1. 预处理
    image = Image.open(image_path).convert("RGB")
    sync()
    t0 = time.perf_counter()
    inputs = processor(prompt, image)
    t1 = time.perf_counter()

    # 2. h2d
    inputs = inputs.to(DEVICE, dtype=model_dtype())
    sync()
    t2 = time.perf_counter()

    # 3-6. 推理 (hook 在内部各阶段打点)
    with torch.inference_mode():
        action = model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
    sync()
    t3 = time.perf_counter()

    preprocess_ms = (t1 - t0) * 1000.0
    h2d_ms = (t2 - t1) * 1000.0
    inference_ms = (t3 - t2) * 1000.0  # 含 vision+projector+llm+postprocess (GPU侧)
    e2e_ms = (t3 - t0) * 1000.0

    vision_ms = _STAGE_TIMES.get("vision_backbone_ms", 0.0)
    projector_ms = _STAGE_TIMES.get("projector_ms", 0.0)
    prefill_ms = _STAGE_TIMES.get("llm_prefill_ms", 0.0)
    decode_ms = _STAGE_TIMES.get("llm_decode_ms", 0.0)
    decode_tokens = _STAGE_TIMES.get("llm_decode_tokens", 0)
    llm_total_ms = prefill_ms + decode_ms
    # postprocess = 推理总时间 - 各GPU阶段 (含embedding/拼接/采样等未单独打点部分)
    other_ms = inference_ms - vision_ms - projector_ms - llm_total_ms

    return {
        "image": image_path.name,
        "instruction": instruction,
        "action": [float(x) for x in list(action)],
        "e2e_ms": e2e_ms,
        "preprocess_ms": preprocess_ms,
        "h2d_ms": h2d_ms,
        "inference_ms": inference_ms,
        "vision_backbone_ms": vision_ms,
        "projector_ms": projector_ms,
        "llm_prefill_ms": prefill_ms,
        "llm_decode_ms": decode_ms,
        "llm_decode_tokens": decode_tokens,
        "llm_decode_per_token_ms": (decode_ms / decode_tokens) if decode_tokens else 0.0,
        "llm_total_ms": llm_total_ms,
        "other_ms": other_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--glob", default="*.jpg")
    parser.add_argument("--instruction", default="pick up the object")
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    images = sorted(image_dir.glob(args.glob))
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images matched {image_dir}/{args.glob}")

    output_dir = Path(os.getenv("OPENVLA_LOGS_DIR", "/workspace/outputs/openvla"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else output_dir / f"latency_breakdown_{time.strftime('%Y_%m%d_%H%M%S')}.jsonl"
    summary_path = output_path.with_suffix(".summary.json")

    print("model:", MODEL_ID)
    print("device:", DEVICE)
    print("attention:", ATTN_IMPLEMENTATION)
    print("unnorm_key:", UNNORM_KEY)
    print("images:", len(images))
    print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        torch.cuda.reset_peak_memory_stats()

    processor = AutoProcessor.from_pretrained(MODEL_PATH, revision=MODEL_REVISION, trust_remote_code=True, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        revision=MODEL_REVISION,
        attn_implementation=ATTN_IMPLEMENTATION,
        torch_dtype=model_dtype(),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    ).to(DEVICE)
    model.eval()

    register_stage_hooks(model)

    # 预热 (排除 CUDA 首次 kernel 编译)
    for i in range(args.warmup):
        timed_predict(model, processor, images[0], args.instruction)
        print(f"warmup {i + 1}/{args.warmup}")

    rows: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as log_file:
        for index, image_path in enumerate(images):
            row = timed_predict(model, processor, image_path, args.instruction)
            rows.append(row)
            log_file.write(json.dumps(row, ensure_ascii=True) + "\n")
            log_file.flush()
            print(
                f"[{index + 1}/{len(images)}] {image_path.name}: "
                f"e2e={row['e2e_ms']:.1f} | pre={row['preprocess_ms']:.1f} "
                f"vis={row['vision_backbone_ms']:.1f} proj={row['projector_ms']:.2f} "
                f"prefill={row['llm_prefill_ms']:.1f} decode={row['llm_decode_ms']:.1f}"
                f"({row['llm_decode_tokens']}tok)"
            )

    # 汇总
    stage_keys = [
        "e2e_ms", "preprocess_ms", "h2d_ms", "inference_ms",
        "vision_backbone_ms", "projector_ms", "llm_prefill_ms",
        "llm_decode_ms", "llm_decode_per_token_ms", "llm_total_ms", "other_ms",
    ]
    summary: dict[str, Any] = {
        "count": len(rows),
        "model_id": MODEL_ID,
        "device": DEVICE,
        "attention": ATTN_IMPLEMENTATION,
        "unnorm_key": UNNORM_KEY,
        "decode_tokens_per_action": rows[0]["llm_decode_tokens"] if rows else 0,
        "stages": {key: stat_block([r[key] for r in rows]) for key in stage_keys},
    }
    # 平均各阶段占比 (基于 mean)
    e2e_mean = summary["stages"]["e2e_ms"]["mean"]
    summary["stage_share_pct_of_e2e"] = {
        key: round(summary["stages"][key]["mean"] / e2e_mean * 100.0, 2)
        for key in ["preprocess_ms", "h2d_ms", "vision_backbone_ms", "projector_ms",
                    "llm_prefill_ms", "llm_decode_ms", "other_ms"]
    }
    if torch.cuda.is_available():
        summary["max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        summary["max_memory_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024 / 1024

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print("\ndetail:", output_path)
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
