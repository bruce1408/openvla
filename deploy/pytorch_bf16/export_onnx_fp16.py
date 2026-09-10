#!/usr/bin/env python3
"""Export OpenVLA vision backbone + projector to fixed-shape FP16 ONNX.

**完整 OpenVLA ONNX 导出请用 shell 脚本（vision + LLM 两步）：**

  bash deploy/pytorch_bf16/export_onnx_fp16.sh          # 默认 all
  bash deploy/pytorch_bf16/export_onnx_fp16.sh vision   # 仅本文件的 vision 部分
  bash deploy/pytorch_bf16/export_onnx_fp16.sh llm      # 仅 LLM（Edge-LLM）

OpenVLA 部署标准拆分为两个 ONNX 子图（无法合并为单个 ONNX 做端到端动作 decode）：
  1. **Vision + Projector**（本 Python 脚本）— `pixel_values` → `projected_patch_embeddings`
  2. **LLM** — 从 OpenVLA 提取 Llama 权重后，用 `tensorrt-edgellm-export` 导出
     到 `artifacts/hf_llama_onnx_fp16/llm/model.onnx`

单独运行本文件只会得到 vision ONNX；这是预期行为，不是导出失败。

Example:
  cd /path/to/openvla
  source env_gpu.sh
  python deploy/pytorch_bf16/export_onnx_fp16.py --device cuda:0

  # 拆分导出（定位 DINO / SigLIP / projector 哪一段失败）
  python deploy/pytorch_bf16/export_onnx_fp16.py --mode split --verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from deploy.tensorrt.common import load_openvla, write_json  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "deploy/pytorch_bf16/artifacts/onnx/vision_projector_fp16.onnx"


class VisionProjectorWrapper(nn.Module):
    """DINOv2 + SigLIP featurizers (fused) + projector MLP."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="CUDA device for export (FP16 weights on GPU)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--mode",
        choices=("combined", "split"),
        default="combined",
        help="combined: single vision+projector graph; split: dino / siglip / projector separately",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output .onnx path (combined mode)")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run onnx.checker and a short ORT session smoke test after export",
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


def verify_onnx(onnx_path: Path, input_name: str, dummy: torch.Tensor) -> None:
    import numpy as np

    try:
        import onnx
    except ImportError as exc:
        raise SystemExit("Install onnx to use --verify: pip install onnx") from exc

    model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.checker.check_model(model)
    print(f"onnx.checker: OK ({onnx_path})")

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipped ORT smoke test")
        return

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feed = {input_name: dummy.detach().cpu().numpy()}
    outputs = session.run(None, feed)
    print(f"onnxruntime smoke: input {feed[input_name].shape} -> output {outputs[0].shape}")


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda"):
        raise SystemExit("FP16 ONNX export must run on CUDA for this OpenVLA checkpoint.")

    _, model = load_openvla(args.device, "fp16")

    channels = 6 if model.config.use_fused_vision_backbone else 3
    image_sizes = list(model.config.image_sizes)
    if len(set(image_sizes)) != 1:
        raise SystemExit(f"Fixed-shape export requires identical branch sizes, got {image_sizes}")
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
        manifest = {
            "component": "vision_projector",
            "precision": "fp16",
            "opset": args.opset,
            "input_name": "pixel_values",
            "input_shape": list(dummy.shape),
            "output_name": "projected_patch_embeddings",
            "output_shape": output_shape,
            "vision_backbone_id": model.config.vision_backbone_id,
            "fixed_shape": True,
            "model_path": str(getattr(model.config, "_name_or_path", "")),
        }
        manifest_path = args.output.with_suffix(".manifest.json")
        write_json(manifest_path, manifest)
        print(f"Exported {args.output}")
        print(f"Manifest  {manifest_path}")
        if args.verify:
            verify_onnx(args.output, "pixel_values", dummy)
        return

    if not model.config.use_fused_vision_backbone:
        raise SystemExit("Split mode targets OpenVLA's fused DINOv2 + SigLIP backbone.")

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

    manifest = {
        "component": "vision_projector_split",
        "precision": "fp16",
        "opset": args.opset,
        "dino_output_shape": dino_shape,
        "siglip_output_shape": siglip_shape,
        "projector_input_shape": list(patch_features.shape),
        "projector_output_shape": output_shape,
        "split_dir": str(split_dir),
    }
    write_json(split_dir / "manifest.json", manifest)
    print(f"Exported split graphs under {split_dir}")
    if args.verify:
        verify_onnx(split_dir / "dino_fp16.onnx", "image", image_a)
        verify_onnx(split_dir / "siglip_fp16.onnx", "image", image_b)
        verify_onnx(split_dir / "projector_fp16.onnx", "patch_features", patch_features)


if __name__ == "__main__":
    main()
