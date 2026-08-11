#!/usr/bin/env python3
# =============================================================================
# 15_nvfp4_multi_sample_accuracy.py - 多样本 golden dump + NVFP4 token/RMSE 聚合
# =============================================================================
#
# 用法:
#   python 15_nvfp4_multi_sample_accuracy.py --start 1 --end 20
#   python 15_nvfp4_multi_sample_accuracy.py --skip-dump   # 仅评测已有 golden
# =============================================================================

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

EDGE_LLM_DIR = Path(os.environ.get("EDGE_LLM_DIR", "/workspace/TensorRT-Edge-LLM"))
ARTIFACTS = REPO_ROOT / "deploy/tensorrt/artifacts"
DEFAULT_LLM_ENGINE_DIR = ARTIFACTS / "engines/openvla_llama_nvfp4"
GOLDEN_ROOT = ARTIFACTS / "golden"
TEST_DATA = REPO_ROOT / "test_data"
LOGS_DIR = Path(os.environ.get("OPENVLA_LOGS_DIR", "/workspace/outputs/openvla"))
PLUGIN = EDGE_LLM_DIR / "build/libNvInfer_edgellm_plugin.so"
N_NEW = 7
DIMS = ["dx", "dy", "dz", "roll", "pitch", "yaw", "gripper"]


def sample_dir(index: int) -> Path:
    return GOLDEN_ROOT / f"sample_{index:04d}"


def image_path(index: int) -> Path:
    return TEST_DATA / f"bridge_sample_{index:04d}.jpg"


def dump_goldens(start: int, end: int, device: str, dtype: str, instruction: str, unnorm_key: str) -> list[int]:
    from PIL import Image
    import torch

    from deploy.tensorrt.common import (
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

    dumped: list[int] = []
    processor, model = load_openvla(device, dtype)
    for idx in range(start, end + 1):
        out_dir = sample_dir(idx)
        img = image_path(idx)
        if not img.is_file():
            print(f"[skip dump] 缺少图片: {img}")
            continue
        if (out_dir / "metadata.json").is_file():
            print(f"[skip dump] 已存在: {out_dir}")
            dumped.append(idx)
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        image = Image.open(img).convert("RGB")
        inputs = processor(prompt_for(instruction), image, return_tensors="pt")
        inputs = move_batch_to_device(inputs, device, torch_dtype(dtype))
        input_ids, attention_mask, appended_empty_token = prepare_action_prompt(
            inputs["input_ids"], inputs.get("attention_mask")
        )
        action_dim = int(model.get_action_dim(unnorm_key))

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

        action_token_ids = generated_ids[0, -action_dim:].detach().cpu().numpy()
        action = decode_action_tokens(model, action_token_ids, unnorm_key)
        low_token, high_token = action_token_bounds(model)

        np.save(out_dir / "input_ids.npy", input_ids.detach().cpu().numpy())
        if attention_mask is not None:
            np.save(out_dir / "attention_mask.npy", attention_mask.detach().cpu().numpy())
        np.save(out_dir / "pixel_values.npy", tensor_to_numpy(inputs["pixel_values"]))
        np.save(out_dir / "multimodal_embeddings.npy", tensor_to_numpy(multimodal_embeddings))
        if multimodal_attention_mask is not None:
            np.save(out_dir / "multimodal_attention_mask.npy", multimodal_attention_mask.detach().cpu().numpy())
        np.save(out_dir / "generated_token_ids.npy", action_token_ids)
        np.save(out_dir / "action.npy", action)

        metadata = {
            "image": str(img.resolve()),
            "instruction": instruction,
            "unnorm_key": unnorm_key,
            "dtype": dtype,
            "empty_token_appended": appended_empty_token,
            "action_dim": action_dim,
            "generated_token_ids": action_token_ids.tolist(),
            "action": np.asarray(action, dtype=float).tolist(),
            "tokens_in_action_range": bool(
                np.all((action_token_ids >= low_token) & (action_token_ids <= high_token))
            ),
        }
        write_json(out_dir / "metadata.json", metadata)
        print(f"[dump] sample_{idx:04d} -> {out_dir}")
        dumped.append(idx)
    return dumped


def evaluate_samples(indices: list[int], engine_dir: Path, device: str, unnorm_key: str) -> dict:
    import torch
    from safetensors.numpy import load_file

    from deploy.tensorrt.common import decode_action_tokens_from_meta, load_action_meta
    from deploy.tensorrt.runtime.edge_llm_runner import EdgeLlmRunner

    ctypes.CDLL(str(PLUGIN))
    runner = EdgeLlmRunner(engine_dir, PLUGIN, device)
    emb_tbl = torch.from_numpy(load_file(str(engine_dir / "embedding.safetensors"))["embedding"]).to(device)
    action_meta = load_action_meta()

    per_sample = []
    for idx in indices:
        gdir = sample_dir(idx)
        meta_path = gdir / "metadata.json"
        mm_path = gdir / "multimodal_embeddings.npy"
        if not meta_path.is_file() or not mm_path.is_file():
            print(f"[skip eval] 缺少 golden: {gdir}")
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        golden_tokens = np.array(meta["generated_token_ids"], dtype=np.int64)
        golden_action = np.array(meta["action"], dtype=np.float64)
        mm = np.load(mm_path)
        stitched = torch.from_numpy(mm[0] if mm.ndim == 3 else mm).to(device)

        toks = np.array(runner.generate(stitched, emb_tbl, N_NEW), dtype=np.int64)
        nvfp4_action = decode_action_tokens_from_meta(toks, action_meta, unnorm_key)
        abs_err = np.abs(nvfp4_action - golden_action)
        token_diff = np.abs(toks - golden_tokens)

        per_sample.append({
            "sample": f"sample_{idx:04d}",
            "golden_tokens": golden_tokens.tolist(),
            "nvfp4_tokens": toks.tolist(),
            "token_abs_diff": token_diff.tolist(),
            "num_exact": int(np.sum(token_diff == 0)),
            "max_diff_bin": int(token_diff.max()),
            "rmse": float(np.sqrt(np.mean((nvfp4_action - golden_action) ** 2))),
            "max_abs_err": float(abs_err.max()),
        })
        print(
            f"[eval] sample_{idx:04d} exact={per_sample[-1]['num_exact']}/{N_NEW} "
            f"max_bin={per_sample[-1]['max_diff_bin']} rmse={per_sample[-1]['rmse']:.5f}"
        )

    if not per_sample:
        return {"error": "no samples evaluated"}

    exact_counts = [s["num_exact"] for s in per_sample]
    max_bins = [s["max_diff_bin"] for s in per_sample]
    rmses = [s["rmse"] for s in per_sample]
    return {
        "method": "golden-emb + action_meta sidecar decode",
        "engine": str(engine_dir / "llm.engine"),
        "n_samples": len(per_sample),
        "token_exact_mean": float(np.mean(exact_counts)),
        "token_exact_min": int(min(exact_counts)),
        "token_exact_max": int(max(exact_counts)),
        "max_diff_bin_mean": float(np.mean(max_bins)),
        "max_diff_bin_max": int(max(max_bins)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_max": float(max(rmses)),
        "rmse_min": float(min(rmses)),
        "per_sample": per_sample,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="多样本 golden dump + NVFP4 精度聚合")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=20)
    ap.add_argument("--skip-dump", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--instruction", default="pick up the blue object")
    ap.add_argument("--unnorm-key", default="bridge_orig")
    ap.add_argument("--llm-engine", default=str(DEFAULT_LLM_ENGINE_DIR))
    ap.add_argument("--output", default=str(LOGS_DIR / "nvfp4_multi_sample_accuracy.json"))
    args = ap.parse_args()

    indices = list(range(args.start, args.end + 1))
    if not args.skip_dump:
        indices = dump_goldens(
            args.start, args.end, args.device, args.dtype, args.instruction, args.unnorm_key
        )

    # 评测所有已有 golden（含之前 dump 的 sample_0001）
    eval_indices = [
        idx for idx in range(args.start, args.end + 1)
        if (sample_dir(idx) / "metadata.json").is_file()
    ]
    result = evaluate_samples(eval_indices, Path(args.llm_engine), args.device, args.unnorm_key)
    result["eval_range"] = {"start": args.start, "end": args.end}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n聚合结果: {out}")
    if "rmse_mean" in result:
        print(
            f"  samples={result['n_samples']}  token_exact_mean={result['token_exact_mean']:.2f}/7  "
            f"max_bin_max={result['max_diff_bin_max']}  rmse_mean={result['rmse_mean']:.5f}"
        )


if __name__ == "__main__":
    main()
