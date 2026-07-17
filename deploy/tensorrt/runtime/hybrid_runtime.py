#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from deploy.tensorrt.common import (  # noqa: E402
    decode_action_tokens,
    ensure_empty_action_token,
    find_default_image,
    load_image,
    load_openvla,
    move_batch_to_device,
    prompt_for,
    resolve_device,
    resolve_dtype,
)


class TensorRTVisionRunner:
    """TensorRT 10-style runner using PyTorch CUDA tensors as engine buffers."""

    def __init__(self, engine_path: Path, device: str) -> None:
        if not device.startswith("cuda"):
            raise ValueError("TensorRT runner requires a CUDA device.")
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "Python package 'tensorrt' is not importable in this environment."
            ) from exc

        self.trt = trt
        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        serialized = engine_path.read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(serialized)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        inputs: list[str] = []
        outputs: list[str] = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                inputs.append(name)
            else:
                outputs.append(name)
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                f"Expected one input and one output, got inputs={inputs}, outputs={outputs}"
            )
        self.input_name = inputs[0]
        self.output_name = outputs[0]

    def _torch_dtype(self, trt_dtype: Any) -> torch.dtype:
        trt = self.trt
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int8: torch.int8,
            trt.bool: torch.bool,
        }
        if hasattr(trt, "bfloat16"):
            mapping[trt.bfloat16] = torch.bfloat16
        if trt_dtype not in mapping:
            raise TypeError(f"Unsupported TensorRT tensor dtype: {trt_dtype}")
        return mapping[trt_dtype]

    def __call__(self, pixel_values: torch.Tensor) -> torch.Tensor:
        expected_dtype = self._torch_dtype(
            self.engine.get_tensor_dtype(self.input_name)
        )
        input_tensor = pixel_values.to(
            device=self.device,
            dtype=expected_dtype,
        ).contiguous()

        if not self.context.set_input_shape(
            self.input_name,
            tuple(input_tensor.shape),
        ):
            raise RuntimeError(
                f"TensorRT rejected input shape {tuple(input_tensor.shape)}"
            )

        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        if any(dim < 0 for dim in output_shape):
            raise RuntimeError(f"Unresolved TensorRT output shape: {output_shape}")
        output_dtype = self._torch_dtype(
            self.engine.get_tensor_dtype(self.output_name)
        )
        output = torch.empty(
            output_shape,
            device=self.device,
            dtype=output_dtype,
        )

        self.context.set_tensor_address(
            self.input_name,
            input_tensor.data_ptr(),
        )
        self.context.set_tensor_address(
            self.output_name,
            output.data_ptr(),
        )
        stream = torch.cuda.current_stream(self.device)
        ok = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3() returned false.")
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid OpenVLA runtime: manual prefill/KV-cache/decode with either "
            "PyTorch or TensorRT vision+projector."
        )
    )
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--instruction", default="pick up the object")
    parser.add_argument(
        "--unnorm-key",
        default=os.getenv("OPENVLA_UNNORM_KEY", "bridge_orig"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="LLM/PyTorch model precision.",
    )
    parser.add_argument(
        "--attention",
        default=os.getenv("OPENVLA_ATTN_IMPLEMENTATION", "sdpa"),
    )
    parser.add_argument(
        "--vision-backend",
        choices=("pytorch", "tensorrt"),
        default="pytorch",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=REPO_ROOT
        / "deploy/tensorrt/artifacts/engines/vision_projector_fp16.plan",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Do not run the original model.predict_action() comparison.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "deploy/tensorrt/artifacts/hybrid_runtime_result.json",
    )
    return parser.parse_args()


def reference_predict_with_tokens(
    model: Any,
    inputs: Any,
    *,
    unnorm_key: str,
) -> tuple[np.ndarray, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    original_generate = model.generate

    def wrapped_generate(*args: Any, **kwargs: Any) -> Any:
        output = original_generate(*args, **kwargs)
        sequences = output.sequences if hasattr(output, "sequences") else output
        captured["sequences"] = sequences.detach().clone()
        return output

    model.generate = wrapped_generate  # type: ignore[method-assign]
    try:
        with torch.inference_mode():
            action = model.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=False,
            )
    finally:
        model.generate = original_generate  # type: ignore[method-assign]

    action_dim = model.get_action_dim(unnorm_key)
    tokens = captured["sequences"][0, -action_dim:]
    return np.asarray(action, dtype=np.float32), tokens


def manual_predict(
    model: Any,
    inputs: Any,
    *,
    unnorm_key: str,
    vision_backend: str,
    trt_runner: TensorRTVisionRunner | None,
) -> dict[str, Any]:
    input_ids = ensure_empty_action_token(inputs["input_ids"])
    pixel_values = inputs["pixel_values"]
    action_dim = model.get_action_dim(unnorm_key)

    if vision_backend == "pytorch":
        patch_features = model.vision_backbone(pixel_values)
        projected = model.projector(patch_features)
    else:
        if trt_runner is None:
            raise RuntimeError("TensorRT backend selected without a runner.")
        projected = trt_runner(pixel_values)

    text_embeddings = model.get_input_embeddings()(input_ids)
    projected = projected.to(
        device=text_embeddings.device,
        dtype=text_embeddings.dtype,
    )
    multimodal_embeddings = torch.cat(
        [
            text_embeddings[:, :1, :],
            projected,
            text_embeddings[:, 1:, :],
        ],
        dim=1,
    )

    # Batch size is one and there is no padding. A full all-ones mask is
    # semantically equivalent to the source model's multimodal attention mask.
    attention_mask = torch.ones(
        multimodal_embeddings.shape[:2],
        dtype=torch.long,
        device=multimodal_embeddings.device,
    )

    prefill = model.language_model(
        input_ids=None,
        attention_mask=attention_mask,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=multimodal_embeddings,
        labels=None,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    next_token = torch.argmax(
        prefill.logits[:, -1, :],
        dim=-1,
    )
    generated = [next_token]
    past_key_values = prefill.past_key_values

    for _ in range(action_dim - 1):
        output = model.language_model(
            input_ids=next_token[:, None],
            attention_mask=None,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=None,
            labels=None,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        past_key_values = output.past_key_values
        next_token = torch.argmax(
            output.logits[:, -1, :],
            dim=-1,
        )
        generated.append(next_token)

    token_ids = torch.stack(generated, dim=1)[0]
    normalized, action = decode_action_tokens(
        model,
        token_ids,
        unnorm_key=unnorm_key,
    )
    return {
        "token_ids": token_ids,
        "normalized_action": normalized,
        "action": action,
        "projected_shape": list(projected.shape),
        "multimodal_shape": list(multimodal_embeddings.shape),
    }


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    def pct(p: float) -> float:
        index = min(
            len(ordered) - 1,
            round((p / 100.0) * (len(ordered) - 1)),
        )
        return ordered[index]
    return {
        "mean": statistics.mean(values),
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    if args.iters < 1:
        raise SystemExit("--iters must be >= 1")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    image_path = args.image or find_default_image(REPO_ROOT)

    processor, model = load_openvla(
        device=device,
        dtype=dtype,
        attention_implementation=args.attention,
    )
    image = load_image(image_path)
    inputs = processor(prompt_for(args.instruction), image)
    inputs = move_batch_to_device(inputs, device=device, dtype=dtype)

    runner = None
    if args.vision_backend == "tensorrt":
        if not args.engine.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {args.engine}")
        runner = TensorRTVisionRunner(args.engine, device)

    reference_action = None
    reference_tokens = None
    if not args.skip_reference:
        reference_action, reference_tokens = reference_predict_with_tokens(
            model,
            inputs,
            unnorm_key=args.unnorm_key,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    result = None
    for _ in range(args.warmup):
        with torch.inference_mode():
            result = manual_predict(
                model,
                inputs,
                unnorm_key=args.unnorm_key,
                vision_backend=args.vision_backend,
                trt_runner=runner,
            )
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    latencies: list[float] = []
    for _ in range(args.iters):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            result = manual_predict(
                model,
                inputs,
                unnorm_key=args.unnorm_key,
                vision_backend=args.vision_backend,
                trt_runner=runner,
            )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000.0)

    assert result is not None
    token_ids = result["token_ids"].detach().cpu()
    action = np.asarray(result["action"], dtype=np.float32)

    comparison: dict[str, Any] | None = None
    if reference_tokens is not None and reference_action is not None:
        reference_tokens_cpu = reference_tokens.detach().cpu()
        comparison = {
            "tokens_exact_match": bool(
                torch.equal(token_ids, reference_tokens_cpu)
            ),
            "manual_tokens": token_ids.tolist(),
            "reference_tokens": reference_tokens_cpu.tolist(),
            "action_max_abs_error": float(
                np.max(np.abs(action - reference_action))
            ),
            "manual_action": action.tolist(),
            "reference_action": reference_action.tolist(),
        }

    payload = {
        "image": str(image_path.resolve()),
        "instruction": args.instruction,
        "unnorm_key": args.unnorm_key,
        "vision_backend": args.vision_backend,
        "engine": str(args.engine) if args.vision_backend == "tensorrt" else None,
        "device": device,
        "dtype": str(dtype),
        "attention": args.attention,
        "warmup": args.warmup,
        "iters": args.iters,
        "latency_ms": summarize(latencies),
        "projected_shape": result["projected_shape"],
        "multimodal_shape": result["multimodal_shape"],
        "token_ids": token_ids.tolist(),
        "normalized_action": np.asarray(
            result["normalized_action"],
            dtype=np.float32,
        ).tolist(),
        "action": action.tolist(),
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("\nSaved:", args.output)
    if comparison is not None and not comparison["tokens_exact_match"]:
        raise SystemExit(
            "FAIL: manual runtime action tokens differ from model.predict_action()."
        )


if __name__ == "__main__":
    main()
