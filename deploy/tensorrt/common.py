"""Shared helpers for the staged OpenVLA TensorRT deployment workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from runtime_env import MODEL_PATH, MODEL_REVISION


EMPTY_TOKEN_ID = 29871
DEFAULT_INSTRUCTION = "pick up the blue object"
DEFAULT_UNNORM_KEY = "bridge_orig"


def prompt_for(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}; choose from {sorted(mapping)}") from exc


def load_openvla(device: str, dtype_name: str = "bf16") -> tuple[Any, Any]:
    """Load the pinned OpenVLA processor and model used by this branch."""

    dtype = torch_dtype(dtype_name)
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        revision=MODEL_REVISION,
        attn_implementation="sdpa",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    model.eval()
    return processor, model


def move_batch_to_device(batch: Any, device: str, dtype: torch.dtype) -> Any:
    """Move a BatchFeature without converting integer token tensors."""

    for key, value in list(batch.items()):
        if not isinstance(value, torch.Tensor):
            continue
        if value.is_floating_point():
            batch[key] = value.to(device=device, dtype=dtype)
        else:
            batch[key] = value.to(device=device)
    return batch


def prepare_action_prompt(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
    """Apply OpenVLA's empty-token rule and keep the attention mask aligned.

    The stock ``predict_action`` only appends token 29871. The standard OpenVLA
    prompt already ends in that token, so no append normally happens. For custom
    tokenizers/prompts, extending the mask here prevents a malformed prefill.
    The returned boolean is persisted in golden metadata for auditability.
    """

    if torch.all(input_ids[:, -1] == EMPTY_TOKEN_ID):
        return input_ids, attention_mask, False

    empty = torch.full(
        (input_ids.shape[0], 1),
        EMPTY_TOKEN_ID,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    input_ids = torch.cat([input_ids, empty], dim=1)
    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(attention_mask[:, :1])],
            dim=1,
        )
    return input_ids, attention_mask, True


def build_multimodal_inputs(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    projected_patch_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reproduce ``modeling_prismatic.py``: BOS + vision + remaining text."""

    text_embeddings = model.get_input_embeddings()(input_ids)
    projected_patch_embeddings = projected_patch_embeddings.to(text_embeddings.dtype)
    multimodal_embeddings = torch.cat(
        [
            text_embeddings[:, :1, :],
            projected_patch_embeddings,
            text_embeddings[:, 1:, :],
        ],
        dim=1,
    )

    multimodal_attention_mask = None
    if attention_mask is not None:
        visual_mask = torch.ones(
            projected_patch_embeddings.shape[:2],
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        multimodal_attention_mask = torch.cat(
            [attention_mask[:, :1], visual_mask, attention_mask[:, 1:]],
            dim=1,
        )
    return multimodal_embeddings, multimodal_attention_mask


def action_token_bounds(model: Any) -> tuple[int, int]:
    """Return the inclusive token range used by the 256 action symbols."""

    effective_vocab_size = int(model.vocab_size)
    n_action_tokens = int(model.config.n_action_bins)
    return effective_vocab_size - n_action_tokens, effective_vocab_size - 1


def decode_action_tokens(model: Any, token_ids: np.ndarray, unnorm_key: str) -> np.ndarray:
    """Decode action token IDs exactly like ``OpenVLAForActionPrediction``."""

    token_ids = np.asarray(token_ids, dtype=np.int64)
    discretized = int(model.vocab_size) - token_ids
    discretized = np.clip(discretized - 1, 0, model.bin_centers.shape[0] - 1)
    normalized = model.bin_centers[discretized]

    stats = model.get_action_stats(unnorm_key)
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    actions = np.where(mask, 0.5 * (normalized + 1.0) * (high - low) + low, normalized)
    return np.asarray(actions, dtype=np.float32)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
