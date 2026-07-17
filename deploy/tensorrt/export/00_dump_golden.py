#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
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
    find_default_image,
    load_image,
    load_openvla,
    move_batch_to_device,
    prompt_for,
    resolve_device,
    resolve_dtype,
    tensor_to_numpy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump a reproducible PyTorch golden trace from OpenVLA predict_action()."
    )
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--instruction", default="pick up the object")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "deploy/tensorrt/artifacts/golden/sample_0001",
    )
    parser.add_argument(
        "--unnorm-key",
        default=os.getenv("OPENVLA_UNNORM_KEY", "bridge_orig"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
    )
    parser.add_argument(
        "--attention",
        default=os.getenv("OPENVLA_ATTN_IMPLEMENTATION", "sdpa"),
    )
    return parser.parse_args()


def clone_tensor(value: Any) -> torch.Tensor | None:
    return value.detach().clone() if isinstance(value, torch.Tensor) else None


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    image_path = args.image or find_default_image(REPO_ROOT)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(0)

    print("Loading model...")
    processor, model = load_openvla(
        device=device,
        dtype=dtype,
        attention_implementation=args.attention,
    )
    image = load_image(image_path)
    prompt = prompt_for(args.instruction)
    inputs = processor(prompt, image)
    inputs = move_batch_to_device(inputs, device=device, dtype=dtype)

    captures: dict[str, Any] = {}
    step_logits: list[torch.Tensor] = []
    lm_call_count = 0

    def vision_hook(_module: Any, _inputs: Any, output: Any) -> None:
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        captures["patch_features"] = clone_tensor(tensor)

    def projector_hook(_module: Any, _inputs: Any, output: Any) -> None:
        captures["projected_patch_embeddings"] = clone_tensor(output)

    def lm_pre_hook(_module: Any, _args: Any, kwargs: dict[str, Any]) -> None:
        nonlocal lm_call_count
        if lm_call_count == 0:
            captures["multimodal_embeddings"] = clone_tensor(
                kwargs.get("inputs_embeds")
            )
            captures["multimodal_attention_mask"] = clone_tensor(
                kwargs.get("attention_mask")
            )
            captures["prefill_input_ids"] = clone_tensor(kwargs.get("input_ids"))
        lm_call_count += 1

    def lm_hook(_module: Any, _inputs: Any, output: Any) -> None:
        logits = output.logits if hasattr(output, "logits") else output[0]
        step_logits.append(logits[:, -1, :].detach().float().cpu())

    handles = [
        model.vision_backbone.register_forward_hook(vision_hook),
        model.projector.register_forward_hook(projector_hook),
        model.language_model.register_forward_pre_hook(
            lm_pre_hook,
            with_kwargs=True,
        ),
        model.language_model.register_forward_hook(lm_hook),
    ]

    original_generate = model.generate

    def captured_generate(*generate_args: Any, **generate_kwargs: Any) -> Any:
        input_ids = (
            generate_args[0]
            if generate_args
            else generate_kwargs.get("input_ids")
        )
        captures["generate_input_ids"] = clone_tensor(input_ids)
        captures["generate_attention_mask"] = clone_tensor(
            generate_kwargs.get("attention_mask")
        )
        output = original_generate(*generate_args, **generate_kwargs)
        sequences = output.sequences if hasattr(output, "sequences") else output
        captures["generated_ids"] = clone_tensor(sequences)
        return output

    model.generate = captured_generate  # type: ignore[method-assign]

    try:
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            action = model.predict_action(
                **inputs,
                unnorm_key=args.unnorm_key,
                do_sample=False,
            )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    finally:
        model.generate = original_generate  # type: ignore[method-assign]
        for handle in handles:
            handle.remove()

    generated_ids = captures.get("generated_ids")
    if generated_ids is None:
        raise RuntimeError("Failed to intercept generated token IDs.")

    action_dim = model.get_action_dim(args.unnorm_key)
    action_token_ids = generated_ids[0, -action_dim:]
    normalized_action, decoded_action = decode_action_tokens(
        model,
        action_token_ids,
        unnorm_key=args.unnorm_key,
    )
    action = np.asarray(action, dtype=np.float32)

    if not np.allclose(action, decoded_action, rtol=0.0, atol=1e-6):
        raise RuntimeError(
            "Independent action decoder does not match model.predict_action(). "
            f"max_abs={np.max(np.abs(action - decoded_action))}"
        )

    arrays: dict[str, Any] = {
        "processor_input_ids": inputs.get("input_ids"),
        "processor_attention_mask": inputs.get("attention_mask"),
        "pixel_values": inputs.get("pixel_values"),
        "generate_input_ids": captures.get("generate_input_ids"),
        "generate_attention_mask": captures.get("generate_attention_mask"),
        "patch_features": captures.get("patch_features"),
        "projected_patch_embeddings": captures.get(
            "projected_patch_embeddings"
        ),
        "multimodal_embeddings": captures.get("multimodal_embeddings"),
        "multimodal_attention_mask": captures.get(
            "multimodal_attention_mask"
        ),
        "generated_ids": generated_ids,
        "action_token_ids": action_token_ids,
        "normalized_action": normalized_action,
        "action": action,
    }
    if step_logits:
        arrays["step_last_logits"] = torch.cat(step_logits, dim=0)

    saved: dict[str, dict[str, Any]] = {}
    for name, value in arrays.items():
        if value is None:
            continue
        array = tensor_to_numpy(value) if isinstance(value, torch.Tensor) else np.asarray(value)
        path = output_dir / f"{name}.npy"
        np.save(path, array)
        saved[name] = {
            "file": path.name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }

    stats = model.get_action_stats(args.unnorm_key)
    metadata = {
        "model_path": str(getattr(model.config, "_name_or_path", "")),
        "image": str(image_path.resolve()),
        "instruction": args.instruction,
        "prompt": prompt,
        "unnorm_key": args.unnorm_key,
        "device": device,
        "model_dtype": str(dtype),
        "attention_implementation": args.attention,
        "predict_action_ms": elapsed_ms,
        "action_dim": action_dim,
        "empty_action_token_id": 29871,
        "effective_vocab_size": int(model.vocab_size),
        "configured_vocab_size": int(model.config.text_config.vocab_size),
        "pad_to_multiple_of": int(model.config.pad_to_multiple_of),
        "n_action_bins": int(model.config.n_action_bins),
        "action_q01": list(map(float, stats["q01"])),
        "action_q99": list(map(float, stats["q99"])),
        "action_mask": list(
            map(
                bool,
                stats.get(
                    "mask",
                    np.ones_like(stats["q01"], dtype=bool),
                ),
            )
        ),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": sys.version,
        "platform": platform.platform(),
        "saved_arrays": saved,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nGolden trace saved:", output_dir)
    print("Action tokens:", action_token_ids.detach().cpu().tolist())
    print("Action:", action.tolist())
    print(f"predict_action: {elapsed_ms:.3f} ms")
    for name, info in saved.items():
        print(f"  {name}: {info['shape']} {info['dtype']}")


if __name__ == "__main__":
    main()
