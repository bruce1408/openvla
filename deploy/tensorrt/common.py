"""Shared helpers for the staged OpenVLA TensorRT deployment workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# 必须在 import transformers 之前导入 runtime_env:
# runtime_env 会 source env.sh 并设置 HF_HOME / HF_HUB_CACHE / TRANSFORMERS_CACHE 等。
# huggingface_hub 在首次 import 时会把缓存路径冻结成模块常量,若晚于 transformers 导入,
# 这些环境变量将不生效,离线加载会去错误的空 hub 目录从而报 LocalEntryNotFoundError。
from runtime_env import MODEL_PATH, MODEL_REVISION  # noqa: E402  (must precede transformers)

import os

import numpy as np
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


def _default_unnorm_key() -> str:
    """Pick action-stats key: explicit env, else infer from the local model directory name."""

    env_key = os.environ.get("OPENVLA_UNNORM_KEY")
    name = Path(MODEL_PATH).name.lower()
    inferred = None
    for needle, key in (
        ("libero-spatial", "libero_spatial"),
        ("libero-object", "libero_object"),
        ("libero-goal", "libero_goal"),
        ("libero-10", "libero_10"),
    ):
        if needle in name:
            inferred = key
            break
    # env_gpu.sh used to default to bridge_orig even on LIBERO checkpoints.
    if env_key and not (env_key == "bridge_orig" and inferred is not None):
        return env_key
    return inferred or env_key or "bridge_orig"


EMPTY_TOKEN_ID = 29871
DEFAULT_INSTRUCTION = "pick up the blue object"
DEFAULT_UNNORM_KEY = _default_unnorm_key()
ACTION_META_PATH = Path(__file__).resolve().parent / "artifacts/action_meta/action_meta.json"


def load_action_meta(path: Path | str | None = None) -> dict[str, Any]:
    """Load exported action detokenization sidecar (04_export_action_params.py)."""

    meta_path = Path(path) if path is not None else ACTION_META_PATH
    return json.loads(meta_path.read_text(encoding="utf-8"))


def action_token_bounds_from_meta(meta: dict[str, Any] | None = None) -> tuple[int, int]:
    """Inclusive action-token ID range from the sidecar."""

    meta = meta or load_action_meta()
    return int(meta["action_token_id_min"]), int(meta["action_token_id_max"])


def decode_action_tokens_from_meta(
    token_ids: np.ndarray,
    meta: dict[str, Any] | None = None,
    unnorm_key: str = DEFAULT_UNNORM_KEY,
) -> np.ndarray:
    """Decode action tokens via action_meta.json (no bf16 model required)."""

    meta = meta or load_action_meta()
    if meta.get("unnorm_key") != unnorm_key:
        raise ValueError(
            f"action_meta unnorm_key={meta.get('unnorm_key')!r} != requested {unnorm_key!r}"
        )

    token_ids = np.asarray(token_ids, dtype=np.int64)
    vocab_size = int(meta["effective_vocab_size"])
    bin_centers = np.asarray(meta["bin_centers"], dtype=np.float64)
    low = np.asarray(meta["q01"], dtype=np.float64)
    high = np.asarray(meta["q99"], dtype=np.float64)
    mask = np.asarray(meta.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)

    discretized = vocab_size - token_ids
    discretized = np.clip(discretized - 1, 0, bin_centers.shape[0] - 1)
    normalized = bin_centers[discretized]
    actions = np.where(mask, 0.5 * (normalized + 1.0) * (high - low) + low, normalized)
    return np.asarray(actions, dtype=np.float64)


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


def _register_local_openvla() -> None:
    """Use in-repo Prismatic classes so offline loads do not fetch openvla/openvla-7b *.py.

    LIBERO fine-tune configs keep auto_map pointed at the Hub repo
    (``openvla/openvla-7b--processing_prismatic.*``). The weight directory only
    has safetensors, so Auto* + trust_remote_code would try the network.
    """

    AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True)


def load_openvla_processor(
    checkpoint: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> PrismaticProcessor:
    """Load PrismaticProcessor without Hub auto_map fetches."""

    _register_local_openvla()
    return PrismaticProcessor.from_pretrained(
        checkpoint or MODEL_PATH,
        revision=revision if revision is not None else MODEL_REVISION,
        local_files_only=local_files_only,
    )


def load_openvla_model(
    checkpoint: str | None = None,
    revision: str | None = None,
    device: str = "cuda:0",
    attn_implementation: str = "sdpa",
    dtype_name: str = "bf16",
    local_files_only: bool = False,
) -> OpenVLAForActionPrediction:
    """Load OpenVLAForActionPrediction without Hub auto_map fetches."""

    _register_local_openvla()
    dtype = torch_dtype(dtype_name)
    model = OpenVLAForActionPrediction.from_pretrained(
        checkpoint or MODEL_PATH,
        revision=revision if revision is not None else MODEL_REVISION,
        attn_implementation=attn_implementation,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()
    return model


def load_openvla(device: str, dtype_name: str = "bf16") -> tuple[Any, Any]:
    """Load the pinned OpenVLA processor and model used by this branch."""

    processor = load_openvla_processor(local_files_only=True)
    model = load_openvla_model(device=device, dtype_name=dtype_name, local_files_only=True)
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
