import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from runtime_env import MODEL_PATH


import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


MODEL_ID = str(MODEL_PATH)
DEVICE = os.getenv("OPENVLA_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
ATTN_IMPLEMENTATION = os.getenv("OPENVLA_ATTN_IMPLEMENTATION", "sdpa")
UNNORM_KEY = os.getenv("OPENVLA_UNNORM_KEY", "bridge_orig")
POWER_MODE = os.getenv("THOR_POWER_MODE", "unknown")


class TokenTimingStreamer:
    def __init__(self) -> None:
        self.first_token_time: float | None = None
        self.last_token_time: float | None = None
        self.token_count = 0
        self._seen_prompt = False

    def put(self, value: Any) -> None:
        now = time.perf_counter()
        if not self._seen_prompt:
            self._seen_prompt = True
            return
        token_count = int(value.numel()) if hasattr(value, "numel") else 1
        if self.first_token_time is None:
            self.first_token_time = now
        self.last_token_time = now
        self.token_count += token_count

    def end(self) -> None:
        return


def dtype() -> torch.dtype:
    if DEVICE.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


def sync() -> None:
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[index]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "count": len(rows),
        "model_id": MODEL_ID,
        "device": DEVICE,
        "attention": ATTN_IMPLEMENTATION,
        "power_mode": POWER_MODE,
    }
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and key.endswith("_ms")
        }
    )
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if key in row]
        metrics[key] = {
            "mean": statistics.mean(values),
            "p50": percentile(values, 50),
            "p90": percentile(values, 90),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
            "min": min(values),
            "max": max(values),
        }
    if DEVICE.startswith("cuda"):
        metrics["max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        metrics["max_memory_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024 / 1024
    return metrics


def load_image(path: str | None) -> Image.Image:
    if path:
        return Image.open(path).convert("RGB")
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


def prompt_for(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def timed_predict_action(model: Any, processor: Any, image: Image.Image, instruction: str) -> dict[str, Any]:
    total_start = time.perf_counter()

    processor_start = time.perf_counter()
    inputs = processor(prompt_for(instruction), image)
    processor_end = time.perf_counter()

    h2d_start = time.perf_counter()
    inputs = inputs.to(DEVICE, dtype=dtype())
    sync()
    h2d_end = time.perf_counter()

    infer_start = time.perf_counter()
    with torch.inference_mode():
        action = model.predict_action(
            **inputs,
            unnorm_key=UNNORM_KEY,
            do_sample=False,
        )
    sync()
    infer_end = time.perf_counter()

    total_end = time.perf_counter()
    return {
        "processor_time_ms": (processor_end - processor_start) * 1000.0,
        "h2d_time_ms": (h2d_end - h2d_start) * 1000.0,
        "predict_action_total_time_ms": (infer_end - infer_start) * 1000.0,
        "model_e2e_time_ms": (total_end - total_start) * 1000.0,
        "action_preview": str(action)[:160],
    }


def timed_generate_tokens(
    model: Any,
    processor: Any,
    image: Image.Image,
    instruction: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    inputs = processor(prompt_for(instruction), image).to(DEVICE, dtype=dtype())
    streamer = TokenTimingStreamer()
    sync()
    start = time.perf_counter()
    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            streamer=streamer,
        )
    sync()
    end = time.perf_counter()

    total_ms = (end - start) * 1000.0
    if streamer.first_token_time is None:
        return {
            "generate_total_time_ms": total_ms,
            "generate_token_count": streamer.token_count,
        }

    ttft_ms = (streamer.first_token_time - start) * 1000.0
    decode_total_ms = total_ms - ttft_ms
    token_count = max(streamer.token_count, 1)
    return {
        "generate_total_time_ms": total_ms,
        "generate_ttft_ms": ttft_ms,
        "generate_prefill_total_time_ms": ttft_ms,
        "generate_decode_total_time_ms": decode_total_ms,
        "generate_token_count": streamer.token_count,
        "generate_time_per_decode_token_ms": decode_total_ms / token_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None)
    parser.add_argument("--instruction", default="move the robot arm forward")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--measure-generate", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_dir = Path(os.getenv("OPENVLA_PREFIX", ".")) / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else output_dir / f"openvla_thor_benchmark_{int(time.time())}.jsonl"
    summary_path = output_path.with_suffix(".summary.json")

    print("model:", MODEL_ID)
    print("device:", DEVICE)
    print("attention:", ATTN_IMPLEMENTATION)
    print("power_mode:", POWER_MODE)
    print("output:", output_path)
    print("torch:", torch.__version__)
    print("cuda runtime:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
        torch.cuda.reset_peak_memory_stats()

    image = load_image(args.image)
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        attn_implementation=ATTN_IMPLEMENTATION,
        torch_dtype=dtype(),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    ).to(DEVICE)
    model.eval()

    for index in range(args.warmup):
        timed_predict_action(model, processor, image, args.instruction)
        print(f"warmup {index + 1}/{args.warmup}")

    rows: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as log_file:
        for index in range(args.iters):
            row = {
                "ts": now_iso(),
                "iter": index,
                "model_id": MODEL_ID,
                "device": DEVICE,
                "attention": ATTN_IMPLEMENTATION,
                "power_mode": POWER_MODE,
                "instruction": args.instruction,
            }
            row.update(timed_predict_action(model, processor, image, args.instruction))
            if args.measure_generate:
                try:
                    row.update(
                        timed_generate_tokens(
                            model,
                            processor,
                            image,
                            args.instruction,
                            args.max_new_tokens,
                        )
                    )
                except Exception as exc:
                    row["generate_timing_error"] = repr(exc)
            rows.append(row)
            log_file.write(json.dumps(row, ensure_ascii=True) + "\n")
            log_file.flush()
            print(
                f"iter {index + 1}/{args.iters}: "
                f"predict_action={row['predict_action_total_time_ms']:.2f} ms, "
                f"e2e={row['model_e2e_time_ms']:.2f} ms"
            )

    summary = summarize(rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
