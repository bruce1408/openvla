import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

RUNTIME_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RUNTIME_DIR))
from runtime_env import MODEL_PATH, MODEL_REVISION

import torch
from PIL import Image
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from transformers import AutoModelForVision2Seq, AutoProcessor


MODEL_ID = str(MODEL_PATH)
DEVICE = os.getenv("OPENVLA_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
ATTN_IMPLEMENTATION = os.getenv("OPENVLA_ATTN_IMPLEMENTATION", "sdpa")
UNNORM_KEY = os.getenv("OPENVLA_UNNORM_KEY", "bridge_orig")


def dtype() -> torch.dtype:
    if DEVICE.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


def sync() -> None:
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()


def load_image(path: str | None) -> Image.Image:
    if path:
        return Image.open(path).convert("RGB")
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


def prompt_for(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def build_activities() -> list[ProfilerActivity]:
    activities = [ProfilerActivity.CPU]
    if DEVICE.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)
    return activities


def run_step(
    model: Any,
    inputs: Any,
    measure_generate: bool,
    max_new_tokens: int,
) -> None:
    if measure_generate:
        with record_function("generate"):
            with torch.inference_mode():
                model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
    else:
        with record_function("predict_action"):
            with torch.inference_mode():
                model.predict_action(
                    **inputs,
                    unnorm_key=UNNORM_KEY,
                    do_sample=False,
                )
    sync()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile OpenVLA inference with torch.profiler "
        "(Chrome trace + operator table + flamegraph stacks)."
    )
    
    parser.add_argument("--image", default="/workspace/openvla/test_data/bridge_sample_0001.jpg", help="Path to an input image (default: gray 224x224).")
    parser.add_argument("--instruction", default="move the robot arm forward")
    parser.add_argument("--wait", type=int, default=1, help="Profiler schedule: idle steps before warmup.")
    parser.add_argument("--warmup", type=int, default=3, help="Profiler schedule: warmup steps (not recorded).")
    parser.add_argument("--active", type=int, default=5, help="Profiler schedule: recorded steps.")
    parser.add_argument("--repeat", type=int, default=1, help="Profiler schedule: number of cycles.")
    parser.add_argument("--measure-generate", action="store_true", help="Profile generate() instead of predict_action().")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--row-limit", type=int, default=30, help="Rows in the printed operator table.")
    parser.add_argument("--output-dir", default=None, help="Directory for trace artifacts (default: <prefix>/logs).")
    parser.add_argument("--tag", default=None, help="Filename tag (default: unix timestamp).")
    parser.add_argument("--no-stack", action="store_true", help="Disable stack recording (skips flamegraph export).")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory profiling.")
    parser.add_argument("--tensorboard", action="store_true", help="Also emit a TensorBoard trace directory.")
    args = parser.parse_args()

    with_stack = not args.no_stack
    profile_memory = not args.no_memory

    # Default to the logs dir next to this script's runtime root, independent of
    # any global OPENVLA_PREFIX. Override with --output-dir when needed.
    default_dir = RUNTIME_DIR / "logs"
    output_dir = Path(args.output_dir) if args.output_dir else default_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag if args.tag else str(int(time.time()))
    mode = "generate" if args.measure_generate else "predict_action"
    base = output_dir / f"openvla_profile_{mode}_{tag}"
    chrome_trace_path = base.with_suffix(".trace.json")
    stacks_path = base.with_suffix(".stacks.txt")
    table_path = base.with_suffix(".table.txt")
    tb_dir = base.parent / f"{base.name}_tb"

    print("model:", MODEL_ID)
    print("device:", DEVICE)
    print("attention:", ATTN_IMPLEMENTATION)
    print("unnorm_key:", UNNORM_KEY)
    print("mode:", mode)
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
        revision=MODEL_REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        revision=MODEL_REVISION,
        attn_implementation=ATTN_IMPLEMENTATION,
        torch_dtype=dtype(),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    ).to(DEVICE)

    model.eval()

    inputs = processor(prompt_for(args.instruction), image).to(DEVICE, dtype=dtype())

    total_steps = args.wait + args.warmup + args.active
    sched = schedule(wait=args.wait, warmup=args.warmup, active=args.active, repeat=args.repeat)

    sort_key = "cuda_time_total" if DEVICE.startswith("cuda") else "cpu_time_total"

    print(f"schedule: wait={args.wait} warmup={args.warmup} active={args.active} "
          f"repeat={args.repeat} -> {total_steps * args.repeat} steps")

    # Note: do NOT pass a tensorboard_trace_handler here. It calls
    # export_chrome_trace internally, and calling export_chrome_trace again
    # afterwards raises "Trace is already saved.". We export once below and
    # copy the trace into the TensorBoard dir when requested.
    with profile(
        activities=build_activities(),
        schedule=sched,
        record_shapes=True,
        profile_memory=profile_memory,
        with_stack=with_stack,
    ) as prof:
        for step in range(total_steps * args.repeat):
            run_step(model, inputs, args.measure_generate, args.max_new_tokens)
            prof.step()
            print(f"step {step + 1}/{total_steps * args.repeat}")

    # Chrome / Perfetto timeline trace (exported exactly once).
    prof.export_chrome_trace(str(chrome_trace_path))
    print("chrome trace:", chrome_trace_path)

    # TensorBoard: reuse the same trace, named so the plugin can discover it.
    if args.tensorboard:
        import shutil

        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_trace = tb_dir / f"{DEVICE.replace(':', '_')}.{int(time.time())}.pt.trace.json"
        shutil.copyfile(chrome_trace_path, tb_trace)

    # Flamegraph stacks (view with FlameGraph: flamegraph.pl <stacks> > out.svg).
    if with_stack:
        stack_metric = "self_cuda_time_total" if DEVICE.startswith("cuda") else "self_cpu_time_total"
        try:
            prof.export_stacks(str(stacks_path), stack_metric)
            print("flamegraph stacks:", stacks_path, f"(metric={stack_metric})")
        except Exception as exc:
            print("export_stacks failed:", repr(exc))

    # Operator table.
    table = prof.key_averages(group_by_input_shape=True).table(
        sort_by=sort_key,
        row_limit=args.row_limit,
    )

    table_path.write_text(table, encoding="utf-8")
    print("operator table:", table_path)
    print(table)

    if args.tensorboard:
        print("tensorboard dir:", tb_dir)
        print("view with: tensorboard --logdir", tb_dir)

    if torch.cuda.is_available():
        print("max_memory_allocated_mb:", torch.cuda.max_memory_allocated() / 1024 / 1024)
        print("max_memory_reserved_mb:", torch.cuda.max_memory_reserved() / 1024 / 1024)

    print("done.")
    print("view chrome trace at chrome://tracing or https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
