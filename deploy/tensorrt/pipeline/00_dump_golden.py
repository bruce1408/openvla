#!/usr/bin/env python3
"""Dump the tensors that form the PyTorch correctness oracle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from deploy.tensorrt.common import (  # noqa: E402
    DEFAULT_INSTRUCTION,
    DEFAULT_UNNORM_KEY,
    action_token_bounds,
    build_multimodal_inputs,
    decode_action_tokens,
    load_openvla,
    move_batch_to_device,
    prepare_action_prompt,
    prompt_for,
    tensor_to_numpy,
    torch_dtype,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=REPO_ROOT / "test_data/bridge_sample_0001.jpg")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--skip-predict-action-check",
        action="store_true",
        help="Skip the extra stock predict_action pass used to verify the golden decode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "deploy/tensorrt/artifacts/golden/sample_0001",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor, model = load_openvla(args.device, args.dtype)

    image = Image.open(args.image).convert("RGB")
    inputs = processor(prompt_for(args.instruction), image, return_tensors="pt")
    inputs = move_batch_to_device(inputs, args.device, torch_dtype(args.dtype))
    input_ids, attention_mask, appended_empty_token = prepare_action_prompt(
        inputs["input_ids"], inputs.get("attention_mask")
    )
    action_dim = int(model.get_action_dim(args.unnorm_key))

    with torch.inference_mode():
        patch_features = model.vision_backbone(inputs["pixel_values"])
        projected = model.projector(patch_features)
        multimodal_embeddings, multimodal_attention_mask = build_multimodal_inputs(
            model, input_ids, attention_mask, projected
        )
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=inputs["pixel_values"],
            max_new_tokens=action_dim,
            do_sample=False,
        )
        stock_action = None
        if not args.skip_predict_action_check:
            stock_action = model.predict_action(
                **inputs,
                unnorm_key=args.unnorm_key,
                do_sample=False,
            )

    action_token_ids = generated_ids[0, -action_dim:].detach().cpu().numpy()
    action = decode_action_tokens(model, action_token_ids, args.unnorm_key)
    stock_action_max_abs_error = None
    if stock_action is not None:
        stock_action = np.asarray(stock_action, dtype=np.float32)
        stock_action_max_abs_error = float(np.max(np.abs(action - stock_action)))
    low_token, high_token = action_token_bounds(model)

    np.save(args.output_dir / "input_ids.npy", input_ids.detach().cpu().numpy())
    if attention_mask is not None:
        np.save(args.output_dir / "attention_mask.npy", attention_mask.detach().cpu().numpy())
    np.save(args.output_dir / "pixel_values.npy", tensor_to_numpy(inputs["pixel_values"]))
    np.save(args.output_dir / "patch_features.npy", tensor_to_numpy(patch_features))
    np.save(args.output_dir / "projected_patch_embeddings.npy", tensor_to_numpy(projected))
    np.save(args.output_dir / "multimodal_embeddings.npy", tensor_to_numpy(multimodal_embeddings))
    if multimodal_attention_mask is not None:
        np.save(
            args.output_dir / "multimodal_attention_mask.npy",
            multimodal_attention_mask.detach().cpu().numpy(),
        )
    np.save(args.output_dir / "generated_token_ids.npy", action_token_ids)
    np.save(args.output_dir / "action.npy", action)

    tokens_in_action_range = bool(np.all((action_token_ids >= low_token) & (action_token_ids <= high_token)))
    metadata = {
        "image": str(args.image.resolve()),
        "instruction": args.instruction,
        "unnorm_key": args.unnorm_key,
        "dtype": args.dtype,
        "empty_token_appended": appended_empty_token,
        "action_dim": action_dim,
        "effective_vocab_size": int(model.vocab_size),
        "padded_lm_head_size": int(model.config.text_config.vocab_size),
        "n_action_tokens": int(model.config.n_action_bins),
        "n_bin_centers": int(model.bin_centers.shape[0]),
        "action_token_id_min": low_token,
        "action_token_id_max": high_token,
        "tokens_in_action_range": tokens_in_action_range,
        "shapes": {
            "input_ids": list(input_ids.shape),
            "pixel_values": list(inputs["pixel_values"].shape),
            "patch_features": list(patch_features.shape),
            "projected_patch_embeddings": list(projected.shape),
            "multimodal_embeddings": list(multimodal_embeddings.shape),
        },
        "generated_token_ids": action_token_ids.tolist(),
        "action": action.tolist(),
        "stock_predict_action_max_abs_error": stock_action_max_abs_error,
    }
    write_json(args.output_dir / "metadata.json", metadata)
    print(f"Golden dump written to {args.output_dir}")
    print(metadata)


if __name__ == "__main__":
    main()
