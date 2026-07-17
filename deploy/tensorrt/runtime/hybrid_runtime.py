#!/usr/bin/env python3
"""TensorRT vision + manual PyTorch Llama prefill/KV-cache/decode runtime."""

from __future__ import annotations

import argparse
import json
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
    torch_dtype,
)
from deploy.tensorrt.runtime.trt_runner import TensorRTRunner  # noqa: E402


def greedy_action_decode(
    language_model: torch.nn.Module,
    multimodal_embeddings: torch.Tensor,
    multimodal_attention_mask: torch.Tensor | None,
    action_dim: int,
) -> torch.Tensor:
    """Generate one token in prefill and ``action_dim - 1`` from the KV cache."""

    outputs = language_model(
        inputs_embeds=multimodal_embeddings,
        attention_mask=multimodal_attention_mask,
        use_cache=True,
        return_dict=True,
    )
    past_key_values = outputs.past_key_values
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    generated = [next_token]

    for _ in range(action_dim - 1):
        outputs = language_model(
            input_ids=next_token[:, None],
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated.append(next_token)
    return torch.stack(generated, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=REPO_ROOT / "test_data/bridge_sample_0001.jpg")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--llm-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--compare-reference", action="store_true")
    args = parser.parse_args()

    processor, model = load_openvla(args.device, args.llm_dtype)
    runner = TensorRTRunner(args.engine, args.device)
    image = Image.open(args.image).convert("RGB")
    inputs = processor(prompt_for(args.instruction), image, return_tensors="pt")
    inputs = move_batch_to_device(inputs, args.device, torch_dtype(args.llm_dtype))
    input_ids, attention_mask, appended = prepare_action_prompt(inputs["input_ids"], inputs.get("attention_mask"))
    action_dim = int(model.get_action_dim(args.unnorm_key))

    with torch.inference_mode():
        trt_outputs = runner({"pixel_values": inputs["pixel_values"]})
        projected = trt_outputs["projected_patch_embeddings"]
        multimodal_embeddings, multimodal_attention_mask = build_multimodal_inputs(
            model, input_ids, attention_mask, projected
        )
        generated = greedy_action_decode(
            model.language_model,
            multimodal_embeddings,
            multimodal_attention_mask,
            action_dim,
        )

    token_ids = generated[0].detach().cpu().numpy()
    action = decode_action_tokens(model, token_ids, args.unnorm_key)
    low_token, high_token = action_token_bounds(model)
    result: dict[str, object] = {
        "token_ids": token_ids.tolist(),
        "action": action.tolist(),
        "empty_token_appended": appended,
        "tokens_in_action_range": bool(np.all((token_ids >= low_token) & (token_ids <= high_token))),
    }

    if args.compare_reference:
        with torch.inference_mode():
            reference_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=inputs["pixel_values"],
                max_new_tokens=action_dim,
                do_sample=False,
            )[0, -action_dim:]
        reference_np = reference_ids.detach().cpu().numpy()
        reference_action = decode_action_tokens(model, reference_np, args.unnorm_key)
        result.update(
            {
                "reference_token_ids": reference_np.tolist(),
                "token_exact_match": bool(np.array_equal(token_ids, reference_np)),
                "max_action_abs_error": float(np.max(np.abs(action - reference_action))),
            }
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
