from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from runtime_env import MODEL_PATH, MODEL_REVISION


EMPTY_ACTION_TOKEN_ID = 29871


def prompt_for(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def resolve_device(value: str | None = None) -> str:
    if value:
        return value
    return os.getenv(
        "OPENVLA_DEVICE",
        "cuda:0" if torch.cuda.is_available() else "cpu",
    )


def resolve_dtype(name: str, device: str) -> torch.dtype:
    normalized = name.lower()
    if not device.startswith("cuda") and normalized != "fp32":
        return torch.float32
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {name}. Choose bf16, fp16, or fp32.")
    return mapping[normalized]


def load_openvla(
    *,
    device: str,
    dtype: torch.dtype,
    attention_implementation: str,
) -> tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        revision=MODEL_REVISION,
        attn_implementation=attention_implementation,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    model.eval()
    return processor, model


def find_default_image(repo_root: Path) -> Path:
    candidates: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        candidates.extend(sorted((repo_root / "test_data").rglob(pattern)))
    if not candidates:
        raise FileNotFoundError(
            "No image was supplied and no image was found under test_data/."
        )
    return candidates[0]


def load_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def move_batch_to_device(batch: Any, *, device: str, dtype: torch.dtype) -> Any:
    # transformers.BatchFeature.to() preserves integer tensors while converting
    # floating-point tensors to dtype.
    return batch.to(device, dtype=dtype)


def ensure_empty_action_token(input_ids: torch.Tensor) -> torch.Tensor:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected input_ids [1, seq], got {tuple(input_ids.shape)}")
    if torch.all(input_ids[:, -1] == EMPTY_ACTION_TOKEN_ID):
        return input_ids
    empty = torch.full(
        (input_ids.shape[0], 1),
        EMPTY_ACTION_TOKEN_ID,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    return torch.cat([input_ids, empty], dim=1)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.numpy()


def decode_action_tokens(
    model: Any,
    token_ids: torch.Tensor | np.ndarray,
    *,
    unnorm_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    ids = (
        token_ids.detach().cpu().numpy()
        if isinstance(token_ids, torch.Tensor)
        else np.asarray(token_ids)
    )
    ids = ids.astype(np.int64, copy=False).reshape(-1)

    discretized = model.vocab_size - ids
    discretized = np.clip(
        discretized - 1,
        a_min=0,
        a_max=model.bin_centers.shape[0] - 1,
    )
    normalized = np.asarray(model.bin_centers[discretized], dtype=np.float32)

    stats = model.get_action_stats(unnorm_key)
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(
        stats.get("mask", np.ones_like(q01, dtype=bool)),
        dtype=bool,
    )
    actions = np.where(
        mask,
        0.5 * (normalized + 1.0) * (q99 - q01) + q01,
        normalized,
    ).astype(np.float32)
    return normalized, actions
