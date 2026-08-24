#!/usr/bin/env python3
# =============================================================================
# 16_nvfp4_multi_sample_action.py - 多样本端到端 action（FP8 vision + NVFP4 LLM）
# =============================================================================
#
# 用法:
#   python 16_nvfp4_multi_sample_action.py --start 1 --end 20
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import runtime_env  # noqa: E402,F401

from deploy.tensorrt.runtime.edge_llm_runner import EdgeLlmRunner  # noqa: E402

EDGE_LLM_DIR = Path(os.environ.get("EDGE_LLM_DIR", "/workspace/TensorRT-Edge-LLM"))
ARTIFACTS = REPO_ROOT / "deploy/tensorrt/artifacts"
VISION_ENGINE = ARTIFACTS / "engines/vision_projector_fp8.plan"
DEFAULT_LLM_ENGINE_DIR = ARTIFACTS / "engines/openvla_llama_nvfp4"
GOLDEN_ROOT = ARTIFACTS / "golden"
PLUGIN = EDGE_LLM_DIR / "build/libNvInfer_edgellm_plugin.so"
LOGS_DIR = Path(os.environ.get("OPENVLA_LOGS_DIR", "/workspace/outputs/openvla"))
N_NEW = 7
DIMS = ["dx", "dy", "dz", "roll", "pitch", "yaw", "gripper"]


def make_vision_runner(device: str):
    from deploy.tensorrt.runtime.trt_runner import TensorRTRunner
    import torch

    runner = TensorRTRunner(VISION_ENGINE, device)

    def _run(pixel_values_np: np.ndarray):
        pv = torch.from_numpy(pixel_values_np).to(device)
        out = runner({"pixel_values": pv})
        key = "projected_patch_embeddings" if "projected_patch_embeddings" in out else list(out)[0]
        return out[key]

    return _run


def main() -> None:
    ap = argparse.ArgumentParser(description="多样本端到端 action RMSE（FP8 vision + NVFP4 LLM）")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--llm-engine", default=str(DEFAULT_LLM_ENGINE_DIR))
    ap.add_argument("--unnorm-key", default="bridge_orig")
    ap.add_argument("--output", default=str(LOGS_DIR / "nvfp4_multi_sample_action.json"))
    args = ap.parse_args()

    import torch

    from deploy.tensorrt.common import (
        action_token_bounds,
        build_multimodal_inputs,
        decode_action_tokens,
        load_openvla,
    )

    engine = EdgeLlmRunner(Path(args.llm_engine), PLUGIN, args.device)
    emb_tbl = engine.load_embedding_table()
    processor, model = load_openvla(args.device, "bf16")
    run_vision = make_vision_runner(args.device)

    per_sample = []
    for idx in range(args.start, args.end + 1):
        gdir = GOLDEN_ROOT / f"sample_{idx:04d}"
        meta_path = gdir / "metadata.json"
        pv_path = gdir / "pixel_values.npy"
        ids_path = gdir / "input_ids.npy"
        if not meta_path.is_file() or not pv_path.is_file() or not ids_path.is_file():
            print(f"[skip] 缺少 golden: {gdir}")
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        golden_tokens = np.array(meta["generated_token_ids"], dtype=np.int64)
        golden_action = np.array(meta["action"], dtype=np.float64)
        input_ids = torch.from_numpy(np.load(ids_path)).to(args.device)
        attn = torch.ones_like(input_ids)
        pv = np.load(pv_path).astype(np.float32)
        projected = run_vision(pv).to(torch.float32)
        if projected.dim() == 2:
            projected = projected.unsqueeze(0)
        with torch.inference_mode():
            mm_emb, _ = build_multimodal_inputs(model, input_ids, attn, projected)
        toks = np.array(engine.generate(mm_emb[0], emb_tbl, N_NEW), dtype=np.int64)
        action = np.asarray(decode_action_tokens(model, toks, args.unnorm_key), dtype=np.float64)
        token_diff = np.abs(toks - golden_tokens)
        abs_err = np.abs(action - golden_action)
        rec = {
            "sample": f"sample_{idx:04d}",
            "golden_tokens": golden_tokens.tolist(),
            "nvfp4_tokens": toks.tolist(),
            "token_abs_diff": token_diff.tolist(),
            "num_exact": int(np.sum(token_diff == 0)),
            "max_diff_bin": int(token_diff.max()),
            "rmse": float(np.sqrt(np.mean((action - golden_action) ** 2))),
            "max_abs_err": float(abs_err.max()),
        }
        per_sample.append(rec)
        print(
            f"[action] sample_{idx:04d} exact={rec['num_exact']}/{N_NEW} "
            f"max_bin={rec['max_diff_bin']} rmse={rec['rmse']:.5f}"
        )

    exact_counts = [s["num_exact"] for s in per_sample]
    max_bins = [s["max_diff_bin"] for s in per_sample]
    rmses = [s["rmse"] for s in per_sample]
    result = {
        "method": "FP8 vision engine + NVFP4 LLM + bf16 decode",
        "engine": str(Path(args.llm_engine) / "llm.engine"),
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
        "eval_range": {"start": args.start, "end": args.end},
        "note": "端到端口径；golden-emb 仍是隔离 LLM 量化误差的权威口径",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n聚合结果: {out}")
    print(
        f"  samples={result['n_samples']}  token_exact_mean={result['token_exact_mean']:.2f}/7  "
        f"max_bin_max={result['max_diff_bin_max']}  rmse_mean={result['rmse_mean']:.5f}"
    )


if __name__ == "__main__":
    main()
