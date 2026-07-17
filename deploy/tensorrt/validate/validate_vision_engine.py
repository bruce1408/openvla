#!/usr/bin/env python3
"""Compare TensorRT projected vision embeddings against a golden dump."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from deploy.tensorrt.runtime.trt_runner import TensorRTRunner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=REPO_ROOT / "deploy/tensorrt/artifacts/golden/sample_0001",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-cosine", type=float, default=0.999)
    args = parser.parse_args()

    pixel_values = torch.from_numpy(np.load(args.golden_dir / "pixel_values.npy")).to(args.device)
    golden = np.load(args.golden_dir / "projected_patch_embeddings.npy").astype(np.float32)
    runner = TensorRTRunner(args.engine, args.device)
    output = runner({"pixel_values": pixel_values})["projected_patch_embeddings"]
    torch.cuda.synchronize()
    actual = output.float().cpu().numpy()

    diff = np.abs(actual - golden)
    golden_flat = golden.reshape(-1).astype(np.float64)
    actual_flat = actual.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(golden_flat) * np.linalg.norm(actual_flat)
    cosine = float(np.dot(golden_flat, actual_flat) / denom)
    metrics = {
        "shape": list(actual.shape),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "cosine_similarity": cosine,
        "min_cosine": args.min_cosine,
        "passed": cosine >= args.min_cosine,
    }
    print(json.dumps(metrics, indent=2))
    if not metrics["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
