#!/usr/bin/env python3
"""Export the fused DINOv2 + SigLIP + projector graph as fixed-shape ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from deploy.tensorrt.common import load_openvla, write_json  # noqa: E402


class VisionProjectorWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.vision_backbone = model.vision_backbone
        self.projector = model.projector

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.projector(self.vision_backbone(pixel_values))


class FeaturizerWrapper(nn.Module):
    def __init__(self, featurizer: nn.Module) -> None:
        super().__init__()
        self.featurizer = featurizer

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.featurizer(image)


class ProjectorWrapper(nn.Module):
    def __init__(self, projector: nn.Module) -> None:
        super().__init__()
        self.projector = projector

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        return self.projector(patch_features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--mode",
        choices=("combined", "split"),
        default="combined",
        help="Use split mode to isolate DINO, SigLIP, or projector export failures.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "deploy/tensorrt/artifacts/onnx/vision_projector_fp16.onnx",
    )
    return parser.parse_args()


def export_module(
    module: nn.Module,
    example: torch.Tensor,
    output: Path,
    input_name: str,
    output_name: str,
    opset: int,
) -> list[int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        expected = module(example)
        torch.onnx.export(
            module,
            (example,),
            str(output),
            input_names=[input_name],
            output_names=[output_name],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=None,
        )
    return list(expected.shape)


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda"):
        raise SystemExit("FP16 vision export must run on CUDA for this model.")

    _, model = load_openvla(args.device, "fp16")
    channels = 6 if model.config.use_fused_vision_backbone else 3
    image_sizes = list(model.config.image_sizes)
    if len(set(image_sizes)) != 1:
        raise SystemExit(f"One fixed tensor cannot represent different branch sizes: {image_sizes}")
    image_size = int(image_sizes[0])
    dummy = torch.zeros((1, channels, image_size, image_size), dtype=torch.float16, device=args.device)

    if args.mode == "combined":
        output_shape = export_module(
            VisionProjectorWrapper(model).eval(),
            dummy,
            args.output,
            "pixel_values",
            "projected_patch_embeddings",
            args.opset,
        )
    else:
        if not model.config.use_fused_vision_backbone:
            raise SystemExit("Split mode currently targets OpenVLA's fused DINOv2 + SigLIP backbone.")
        split_dir = args.output.parent / "split"
        image_a, image_b = torch.split(dummy, [3, 3], dim=1)
        dino = FeaturizerWrapper(model.vision_backbone.featurizer).eval()
        siglip = FeaturizerWrapper(model.vision_backbone.fused_featurizer).eval()
        dino_shape = export_module(dino, image_a, split_dir / "dino_fp16.onnx", "image", "patches", args.opset)
        siglip_shape = export_module(
            siglip, image_b, split_dir / "siglip_fp16.onnx", "image", "patches", args.opset
        )

        with torch.inference_mode():
            patch_features = torch.cat([dino(image_a), siglip(image_b)], dim=2)
        
        output_shape = export_module(
            ProjectorWrapper(model.projector).eval(),
            patch_features,
            split_dir / "projector_fp16.onnx",
            "patch_features",
            "projected_patch_embeddings",
            args.opset,
        )
        
        write_json(
            split_dir / "manifest.json",
            {
                "dino_output_shape": dino_shape,
                "siglip_output_shape": siglip_shape,
                "projector_input_shape": list(patch_features.shape),
                "projector_output_shape": output_shape,
            },
        )
        print(f"Exported split graphs under {split_dir}")
        return

    manifest = {
        "precision": "fp16",
        "opset": args.opset,
        "input_name": "pixel_values",
        "input_shape": list(dummy.shape),
        "output_name": "projected_patch_embeddings",
        "output_shape": output_shape,
        "vision_backbone_id": model.config.vision_backbone_id,
        "fixed_shape": True,
    }
    
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    print(f"Exported {args.output}")
    print(manifest)


if __name__ == "__main__":
    main()
