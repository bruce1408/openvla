#!/usr/bin/env python3
"""Validate representative or known-mismatch images in one model process.

The TensorRT engine, OpenVLA processor, and PyTorch model are loaded once.  By
default this script checks seven representative samples selected from the 29
images that differed in the BF16-reference comparison.  Pass
``--all-mismatches`` to check all 29, or ``--images`` for a custom list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from deploy.tensorrt.common import (  # noqa: E402
    DEFAULT_INSTRUCTION,
    DEFAULT_UNNORM_KEY,
    action_token_bounds,
    build_multimodal_inputs,
    decode_action_tokens,
    load_openvla,
    move_batch_to_device,
    prepare_action_prompt,
    prompt_for,
    torch_dtype,
    write_json,
)
from deploy.tensorrt.runtime.hybrid_runtime import greedy_action_decode  # noqa: E402
from deploy.tensorrt.runtime.trt_runner import TensorRTRunner  # noqa: E402


KNOWN_MISMATCH_NAMES = [
    "bridge_sample_0002.jpg",
    "bridge_sample_0013.jpg",
    "bridge_sample_0019.jpg",
    "bridge_sample_0020.jpg",
    "bridge_sample_0023.jpg",
    "bridge_sample_0031.jpg",
    "bridge_sample_0034.jpg",
    "bridge_sample_0037.jpg",
    "bridge_sample_0040.jpg",
    "bridge_sample_0046.jpg",
    "bridge_sample_0047.jpg",
    "bridge_sample_0055.jpg",
    "bridge_sample_0056.jpg",
    "bridge_sample_0061.jpg",
    "bridge_sample_0062.jpg",
    "bridge_sample_0063.jpg",
    "bridge_sample_0069.jpg",
    "bridge_sample_0071.jpg",
    "bridge_sample_0073.jpg",
    "bridge_sample_0077.jpg",
    "bridge_sample_0080.jpg",
    "bridge_sample_0081.jpg",
    "bridge_sample_0083.jpg",
    "bridge_sample_0085.jpg",
    "bridge_sample_0089.jpg",
    "bridge_sample_0090.jpg",
    "bridge_sample_0091.jpg",
    "bridge_sample_0094.jpg",
    "bridge_sample_0097.jpg",
]

REPRESENTATIVE_NAMES = [
    "bridge_sample_0002.jpg",
    "bridge_sample_0023.jpg",
    "bridge_sample_0047.jpg",
    "bridge_sample_0063.jpg",
    "bridge_sample_0085.jpg",
    "bridge_sample_0094.jpg",
    "bridge_sample_0097.jpg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-process TensorRT-vs-PyTorch action-token validation."
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=REPO_ROOT / "deploy/tensorrt/artifacts/engines/vision_projector_fp16.plan",
    )
    parser.add_argument("--image-dir", type=Path, default=REPO_ROOT / "test_data")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--all-mismatches",
        action="store_true",
        help="Validate all 29 images that mismatched against the BF16 reference.",
    )
    selection.add_argument(
        "--images",
        type=Path,
        nargs="+",
        help="Custom image paths. Relative paths are resolved from the repository root.",
    )
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--llm-dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--semantic-tolerance", type=float, default=1.0e-6)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "tensorrt_logs/representative_fp16_validation.jsonl",
    )
    return parser.parse_args()


def resolve_images(args: argparse.Namespace) -> list[Path]:
    if args.images:
        images = [path if path.is_absolute() else REPO_ROOT / path for path in args.images]
    else:
        names = KNOWN_MISMATCH_NAMES if args.all_mismatches else REPRESENTATIVE_NAMES
        images = [args.image_dir / name for name in names]

    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing input images: {missing}")
    return images


def first_mismatch(actual: np.ndarray, reference: np.ndarray) -> int | None:
    indices = np.flatnonzero(actual != reference)
    return int(indices[0]) if indices.size else None


def validate_one(
    image_path: Path,
    processor: Any,
    model: Any,
    runner: TensorRTRunner,
    args: argparse.Namespace,
    action_dim: int,
    low_token: int,
    high_token: int,
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(prompt_for(args.instruction), image, return_tensors="pt")
    inputs = move_batch_to_device(inputs, args.device, torch_dtype(args.llm_dtype))
    input_ids, attention_mask, appended = prepare_action_prompt(
        inputs["input_ids"], inputs.get("attention_mask")
    )

    with torch.inference_mode():
        projected = runner({"pixel_values": inputs["pixel_values"]})[
            "projected_patch_embeddings"
        ]
        multimodal_embeddings, multimodal_attention_mask = build_multimodal_inputs(
            model, input_ids, attention_mask, projected
        )
        generated = greedy_action_decode(
            model.language_model,
            multimodal_embeddings,
            multimodal_attention_mask,
            action_dim,
        )
        reference_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=inputs["pixel_values"],
            max_new_tokens=action_dim,
            do_sample=False,
        )[0, -action_dim:]

    token_ids = generated[0].detach().cpu().numpy()
    reference_np = reference_ids.detach().cpu().numpy()
    action = decode_action_tokens(model, token_ids, args.unnorm_key)
    reference_action = decode_action_tokens(model, reference_np, args.unnorm_key)
    action_abs_error = np.abs(action - reference_action)
    mismatch_positions = np.flatnonzero(token_ids != reference_np).astype(int).tolist()

    return {
        "image": str(image_path.relative_to(REPO_ROOT)),
        "llm_dtype": args.llm_dtype,
        "token_ids": token_ids.tolist(),
        "reference_token_ids": reference_np.tolist(),
        "token_exact_match": bool(np.array_equal(token_ids, reference_np)),
        "mismatch_positions": mismatch_positions,
        "first_mismatch_position": first_mismatch(token_ids, reference_np),
        "tokens_in_action_range": bool(
            np.all((token_ids >= low_token) & (token_ids <= high_token))
        ),
        "action": action.tolist(),
        "reference_action": reference_action.tolist(),
        "action_abs_error": action_abs_error.tolist(),
        "max_action_abs_error": float(action_abs_error.max()),
        "empty_token_appended": appended,
    }


def summarize(rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    errors = np.asarray([row["max_action_abs_error"] for row in rows], dtype=np.float64)
    exact = [row for row in rows if row["token_exact_match"]]
    token_mismatch = [row for row in rows if not row["token_exact_match"]]
    semantic_mismatch = [row for row in rows if row["max_action_abs_error"] > tolerance]
    invalid_range = [row for row in rows if not row["tokens_in_action_range"]]

    return {
        "count": len(rows),
        "token_exact_count": len(exact),
        "token_mismatch_count": len(token_mismatch),
        "semantic_action_mismatch_count": len(semantic_mismatch),
        "invalid_action_token_range_count": len(invalid_range),
        "semantic_tolerance": tolerance,
        "max_action_abs_error": float(errors.max()) if errors.size else 0.0,
        "mean_action_abs_error": float(errors.mean()) if errors.size else 0.0,
        "p95_action_abs_error": float(np.percentile(errors, 95)) if errors.size else 0.0,
        "token_mismatch_images": [row["image"] for row in token_mismatch],
        "semantic_mismatch_images": [row["image"] for row in semantic_mismatch],
        "passed": len(exact) == len(rows) and not semantic_mismatch and not invalid_range,
    }


def main() -> None:
    args = parse_args()
    images = resolve_images(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading OpenVLA once with dtype={args.llm_dtype} ...")
    processor, model = load_openvla(args.device, args.llm_dtype)
    runner = TensorRTRunner(args.engine, args.device)
    action_dim = int(model.get_action_dim(args.unnorm_key))
    low_token, high_token = action_token_bounds(model)

    rows: list[dict[str, Any]] = []
    with args.output.open("w", encoding="utf-8") as output_file:
        for index, image_path in enumerate(images, start=1):
            row = validate_one(
                image_path,
                processor,
                model,
                runner,
                args,
                action_dim,
                low_token,
                high_token,
            )
            rows.append(row)
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_file.flush()
            status = "PASS" if row["token_exact_match"] else "FAIL"
            print(
                f"[{index:02d}/{len(images):02d}] {status} {image_path.name} "
                f"max_action_error={row['max_action_abs_error']:.9g} "
                f"mismatch_positions={row['mismatch_positions']}"
            )

    summary = summarize(rows, args.semantic_tolerance)
    summary.update(
        {
            "engine": str(args.engine),
            "llm_dtype": args.llm_dtype,
            "instruction": args.instruction,
            "unnorm_key": args.unnorm_key,
            "detail_log": str(args.output),
        }
    )
    summary_path = args.output.with_suffix(".summary.json")
    write_json(summary_path, summary)
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"detail:  {args.output}")
    print(f"summary: {summary_path}")

    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
