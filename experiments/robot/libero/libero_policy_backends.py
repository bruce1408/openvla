"""BF16 and FP8 policy backends for a shared LIBERO rollout loop."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


EMPTY_TOKEN_ID = 29871


def prompt_for(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def candidate_unnorm_keys(task_suite_name: str) -> tuple[str, str]:
    return task_suite_name, f"{task_suite_name}_no_noops"


def resolve_unnorm_key(available: Any, task_suite_name: str) -> str:
    available_keys = set(available)
    for key in candidate_unnorm_keys(task_suite_name):
        if key in available_keys:
            return key
    raise ValueError(
        f"Checkpoint has no action statistics for {task_suite_name!r}. "
        f"Expected one of {candidate_unnorm_keys(task_suite_name)}, "
        f"available={sorted(available_keys)}. Use the checkpoint fine-tuned for this LIBERO suite."
    )


def decode_action_tokens_from_metadata(token_ids: Any, metadata: dict[str, Any]) -> np.ndarray:
    """Decode action token IDs without loading the BF16 model."""

    token_ids = np.asarray(token_ids, dtype=np.int64)
    action_dim = int(metadata["action_dim"])
    if token_ids.shape != (action_dim,):
        raise ValueError(f"Expected {action_dim} action tokens, got shape {token_ids.shape}")

    effective_vocab_size = int(metadata["effective_vocab_size"])
    bin_centers = np.asarray(metadata["bin_centers"], dtype=np.float32)
    discretized = np.clip(effective_vocab_size - token_ids - 1, 0, len(bin_centers) - 1)
    normalized = bin_centers[discretized]

    low = np.asarray(metadata["q01"], dtype=np.float32)
    high = np.asarray(metadata["q99"], dtype=np.float32)
    mask = np.asarray(metadata.get("mask", [True] * action_dim), dtype=bool)
    if low.shape != (action_dim,) or high.shape != (action_dim,) or mask.shape != (action_dim,):
        raise ValueError("Action metadata q01/q99/mask dimensions do not match action_dim")
    action = np.where(mask, 0.5 * (normalized + 1.0) * (high - low) + low, normalized)
    return np.asarray(action, dtype=np.float32)


@dataclass(frozen=True)
class PolicyPrediction:
    action: np.ndarray
    latency_ms: float
    token_ids: tuple[int, ...] | None = None


class Bf16Policy:
    name = "bf16"

    def __init__(
        self,
        checkpoint: str,
        task_suite_name: str,
        device: str,
        revision: str | None = None,
        attn_implementation: str = "sdpa",
        local_files_only: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.torch = torch
        self.device = torch.device(device)
        self.checkpoint = checkpoint
        load_args = {
            "revision": revision,
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        }
        self.processor = AutoProcessor.from_pretrained(checkpoint, **load_args)
        self.model = AutoModelForVision2Seq.from_pretrained(
            checkpoint,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            **load_args,
        ).to(self.device)
        self.model.eval()

        statistics_path = Path(checkpoint) / "dataset_statistics.json"
        if statistics_path.is_file():
            self.model.norm_stats = json.loads(statistics_path.read_text(encoding="utf-8"))
        self.unnorm_key = resolve_unnorm_key(self.model.norm_stats, task_suite_name)
        self.action_dim = int(self.model.get_action_dim(self.unnorm_key))

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "checkpoint": self.checkpoint,
            "unnorm_key": self.unnorm_key,
            "action_dim": self.action_dim,
        }

    def predict(self, image: Image.Image, instruction: str) -> PolicyPrediction:
        torch = self.torch
        batch = self.processor(prompt_for(instruction), image, return_tensors="pt")
        for key, value in list(batch.items()):
            if not isinstance(value, torch.Tensor):
                continue
            dtype = torch.bfloat16 if value.is_floating_point() else value.dtype
            batch[key] = value.to(device=self.device, dtype=dtype)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            action = self.model.predict_action(
                **batch,
                unnorm_key=self.unnorm_key,
                do_sample=False,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return PolicyPrediction(np.asarray(action, dtype=np.float32), latency_ms)


class Fp8Policy:
    name = "fp8"

    def __init__(
        self,
        checkpoint: str,
        task_suite_name: str,
        device: str,
        vision_engine: Path,
        llm_engine_dir: Path,
        action_metadata: Path,
        edge_llm_plugin: Path,
        revision: str | None = None,
        local_files_only: bool = False,
        require_provenance: bool = True,
    ) -> None:
        import torch
        from transformers import AutoProcessor

        from deploy.tensorrt.runtime.edge_llm_runner import EdgeLlmRunner
        from deploy.tensorrt.runtime.trt_runner import TensorRTRunner

        self.torch = torch
        self.device = torch.device(device)
        self.checkpoint = checkpoint
        self.vision_engine_path = Path(vision_engine)
        self.llm_engine_dir = Path(llm_engine_dir)
        self.action_metadata_path = Path(action_metadata)
        self.action_metadata = json.loads(self.action_metadata_path.read_text(encoding="utf-8"))

        expected_keys = candidate_unnorm_keys(task_suite_name)
        self.unnorm_key = str(self.action_metadata.get("unnorm_key", ""))
        if self.unnorm_key not in expected_keys:
            raise ValueError(
                f"FP8 action metadata uses {self.unnorm_key!r}, but suite {task_suite_name!r} "
                f"requires one of {expected_keys}. Rebuild every FP8 artifact from the matching "
                "LIBERO fine-tuned checkpoint."
            )
        source_model = self.action_metadata.get("source_model")
        if require_provenance and source_model is None:
            raise ValueError(
                "FP8 action metadata has no source_model provenance. Re-run "
                "04_export_action_params.py after rebuilding from the LIBERO checkpoint, or pass "
                "--allow-missing-fp8-provenance only for legacy artifacts you verified manually."
            )
        if source_model is not None and str(source_model) != checkpoint:
            raise ValueError(
                f"FP8 artifacts identify source_model={source_model!r}, but --checkpoint={checkpoint!r}"
            )

        load_args = {
            "revision": revision,
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        }
        self.processor = AutoProcessor.from_pretrained(checkpoint, **load_args)
        self.vision_runner = TensorRTRunner(self.vision_engine_path, device)
        self.llm_runner = EdgeLlmRunner(self.llm_engine_dir, edge_llm_plugin, device)
        self.embedding_table = self.llm_runner.load_embedding_table()

        self.action_dim = int(self.action_metadata["action_dim"])
        if self.embedding_table.shape[0] != self.llm_runner.vocab_size:
            raise ValueError("Embedding table and Edge-LLM vocabulary sizes differ")
        if int(self.action_metadata["padded_lm_head_size"]) != self.llm_runner.vocab_size:
            raise ValueError("Action metadata and Edge-LLM padded vocabulary sizes differ")

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "checkpoint": self.checkpoint,
            "unnorm_key": self.unnorm_key,
            "action_dim": self.action_dim,
            "vision_engine": str(self.vision_engine_path),
            "llm_engine_dir": str(self.llm_engine_dir),
            "action_metadata": str(self.action_metadata_path),
        }

    def _prepare_input_ids(self, input_ids: Any) -> Any:
        torch = self.torch
        input_ids = input_ids.to(self.device)
        if not torch.all(input_ids[:, -1] == int(self.action_metadata.get("empty_token_id", EMPTY_TOKEN_ID))):
            empty_token = torch.full(
                (input_ids.shape[0], 1),
                int(self.action_metadata.get("empty_token_id", EMPTY_TOKEN_ID)),
                dtype=input_ids.dtype,
                device=self.device,
            )
            input_ids = torch.cat([input_ids, empty_token], dim=1)
        return input_ids

    def predict(self, image: Image.Image, instruction: str) -> PolicyPrediction:
        torch = self.torch
        batch = self.processor(prompt_for(instruction), image, return_tensors="pt")
        input_ids = self._prepare_input_ids(batch["input_ids"])
        pixel_values = batch["pixel_values"].to(self.device)

        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            vision_outputs = self.vision_runner({"pixel_values": pixel_values})
            projected = vision_outputs.get("projected_patch_embeddings")
            if projected is None:
                projected = next(iter(vision_outputs.values()))
            if projected.ndim == 2:
                projected = projected.unsqueeze(0)

            text_embeddings = self.embedding_table[input_ids]
            projected = projected.to(dtype=text_embeddings.dtype)
            multimodal = torch.cat(
                [text_embeddings[:, :1], projected, text_embeddings[:, 1:]],
                dim=1,
            )
            token_ids = self.llm_runner.generate(multimodal[0], self.embedding_table, self.action_dim)
        torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000.0

        low = int(self.action_metadata["action_token_id_min"])
        high = int(self.action_metadata["action_token_id_max"])
        if any(token < low or token > high for token in token_ids):
            raise RuntimeError(f"FP8 generated non-action token(s): {token_ids}; expected [{low}, {high}]")
        action = decode_action_tokens_from_metadata(token_ids, self.action_metadata)
        return PolicyPrediction(action, latency_ms, tuple(token_ids))


def create_policy(args: Any) -> Bf16Policy | Fp8Policy:
    common = {
        "checkpoint": args.checkpoint,
        "task_suite_name": args.task_suite_name,
        "device": args.device,
        "revision": args.revision,
        "local_files_only": args.local_files_only,
    }
    if args.backend == "bf16":
        return Bf16Policy(attn_implementation=args.attn_implementation, **common)
    return Fp8Policy(
        vision_engine=args.vision_engine,
        llm_engine_dir=args.llm_engine_dir,
        action_metadata=args.action_metadata,
        edge_llm_plugin=args.edge_llm_plugin,
        require_provenance=not args.allow_missing_fp8_provenance,
        **common,
    )
