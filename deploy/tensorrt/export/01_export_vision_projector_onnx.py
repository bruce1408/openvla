#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from deploy.tensorrt.common import (  # noqa: E402
    load_openvla,
    move_batch_to_device,
    prompt_for,
    resolve_device,
    resolve_dtype,
    tensor_to_numpy,
)


class VisionProjectorWrapper(nn.Module):
    def __init__(self, vision_backbone: nn.Module, projector: nn.Module) -> None:
        super().__init__()
        self.vision_backbone = vision_backbone
        self.projector = projector

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patch_features = self.vision_backbone(pixel_values)
        return self.projector(patch_features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export OpenVLA DINOv2+SigLIP+projector to a fixed-shape ONNX model."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "deploy/tensorrt/artifacts/onnx/vision_projector_fp16.onnx",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype",
        choices=("fp16", "fp32"),
        default="fp16",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--attention",
        default=os.getenv("OPENVLA_ATTN_IMPLEMENTATION", "sdpa"),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run onnx.checker after export. This consumes additional host memory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if not device.startswith("cuda") and args.dtype == "fp16":
        raise SystemExit("FP16 ONNX export should be run on CUDA. Use --dtype fp32 for CPU.")
    dtype = resolve_dtype(args.dtype, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading OpenVLA checkpoint...")
    processor, model = load_openvla(
        device=device,
        dtype=dtype,
        attention_implementation=args.attention,
    )

    # Use the real processor to derive the model's actual fused image tensor
    # shape instead of hard-coding [1, 6, 224, 224].
    dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    batch = processor(prompt_for("pick up the object"), dummy_image)
    batch = move_batch_to_device(batch, device=device, dtype=dtype)
    pixel_values = batch["pixel_values"].contiguous()

    wrapper = VisionProjectorWrapper(
        model.vision_backbone,
        model.projector,
    ).eval().to(device=device, dtype=dtype)

    # Drop the 7B language model before tracing to reduce memory pressure.
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    with torch.inference_mode():
        reference = wrapper(pixel_values)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    print("Input:", tuple(pixel_values.shape), pixel_values.dtype)
    print("Output:", tuple(reference.shape), reference.dtype)
    print("Exporting:", args.output)

    started = time.perf_counter()
    try:
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (pixel_values,),
                str(args.output),
                input_names=["pixel_values"],
                output_names=["projected_patch_embeddings"],
                opset_version=args.opset,
                do_constant_folding=True,
                dynamic_axes=None,
                verbose=False,
            )
    except Exception as exc:
        raise RuntimeError(
            "Combined vision+projector ONNX export failed. The most likely "
            "compatibility point is timm get_intermediate_layers used by the "
            "DINOv2/SigLIP monkey-patched forward methods. Preserve this error "
            "log; the next fallback is to export DINOv2, SigLIP and projector "
            "as three separate graphs."
        ) from exc

    elapsed = time.perf_counter() - started
    np.save(
        args.output.with_suffix(".input.npy"),
        tensor_to_numpy(pixel_values),
    )
    np.save(
        args.output.with_suffix(".reference.npy"),
        tensor_to_numpy(reference),
    )

    if args.check:
        import onnx

        onnx_model = onnx.load(str(args.output))
        onnx.checker.check_model(onnx_model)
        print("onnx.checker: OK")

    metadata = {
        "onnx": args.output.name,
        "input_name": "pixel_values",
        "input_shape": list(pixel_values.shape),
        "input_dtype": str(pixel_values.dtype),
        "output_name": "projected_patch_embeddings",
        "output_shape": list(reference.shape),
        "output_dtype": str(reference.dtype),
        "opset": args.opset,
        "fixed_shape": True,
        "export_seconds": elapsed,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    size_gib = args.output.stat().st_size / (1024**3)
    print(f"Export complete in {elapsed:.1f}s; ONNX main file: {size_gib:.2f} GiB")
    print("Reference input:", args.output.with_suffix(".input.npy"))
    print("Reference output:", args.output.with_suffix(".reference.npy"))


if __name__ == "__main__":
    main()
