import argparse
import json
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
from transformers import AutoModelForVision2Seq, AutoProcessor


MODEL_ID = str(MODEL_PATH)
DEVICE = os.getenv("OPENVLA_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
ATTN_IMPLEMENTATION = os.getenv("OPENVLA_ATTN_IMPLEMENTATION", "sdpa")
UNNORM_KEY = os.getenv("OPENVLA_UNNORM_KEY", "bridge_orig")

DEFAULT_IMAGE_DIR = RUNTIME_DIR / "test_data"


def model_dtype() -> torch.dtype:
    if DEVICE.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


def sync() -> None:
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()


def prompt_for(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def predict_one(
    model: Any,
    processor: Any,
    image_path: Path,
    instruction: str,
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(prompt_for(instruction), image).to(DEVICE, dtype=model_dtype())

    sync()
    start = time.perf_counter()
    with torch.inference_mode():
        action = model.predict_action(
            **inputs,
            unnorm_key=UNNORM_KEY,
            do_sample=False,
        )
    sync()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # action is typically a numpy array of 7 values (dx, dy, dz, droll, dpitch, dyaw, gripper)
    action_list = [float(x) for x in list(action)]
    return {
        "image": image_path.name,
        "instruction": instruction,
        "action": action_list,
        "predict_action_time_ms": elapsed_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--glob", default="*.jpg")
    parser.add_argument("--instruction", default="pick up the object")
    parser.add_argument("--limit", type=int, default=0, help="0 = all images")
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
    output_path = Path(args.output) if args.output else output_dir / f"openvla_infer_{time.strftime('%Y_%m%d_%H%M%S')}.jsonl"

    print("model:", MODEL_ID)
    print("device:", DEVICE)
    print("attention:", ATTN_IMPLEMENTATION)
    print("unnorm_key:", UNNORM_KEY)
    print("images:", len(images))
    print("instruction:", args.instruction)
    print("output:", output_path)
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

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
        torch_dtype=model_dtype(),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    ).to(DEVICE)
    model.eval()

    with output_path.open("w", encoding="utf-8") as log_file:
        for index, image_path in enumerate(images):
            row = predict_one(model, processor, image_path, args.instruction)
            log_file.write(json.dumps(row, ensure_ascii=True) + "\n")
            log_file.flush()
            action_preview = ", ".join(f"{v:+.4f}" for v in row["action"])
            print(
                f"[{index + 1}/{len(images)}] {image_path.name} "
                f"({row['predict_action_time_ms']:.1f} ms): [{action_preview}]"
            )

    print("done. results saved to:", output_path)


if __name__ == "__main__":
    main()
