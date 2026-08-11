#!/usr/bin/env python3
# =============================================================================
# 14b_nvfp4_token_accuracy.py - NVFP4 7-DoF 动作 token 精度 (token 级, 无需 bf16 model)
# =============================================================================
#
# 目标 (文档 §3.1):
#   对比 NVFP4 LLM engine 与 BF16 golden 的 7 个动作 token。
#   关键: 直接喂 golden 预拼的 bf16 multimodal_embeddings (golden-emb 思路),
#   只跑 NVFP4 engine 生成 token,与 golden metadata 的 generated_token_ids 对比。
#   纯 token 级 → 不需要加载 bf16 model (bin_centers/action_stats),避开离线 HF 依赖。
#
# 为什么 token 级就够:
#   OpenVLA 输出物理动作值 = decode(token, bin_centers, q01/q99),是 token 的单调映射。
#   token 绝对差 = bin 漂移数,直接反映量化扰动幅度 (见模板文档 §3.1 反推)。
#   若需要物理值 RMSE,需 bf16 model 的 bin_centers/action_stats (离线不可用,标注待补)。
#
# 用法:
#   python 14b_nvfp4_token_accuracy.py                       # golden-emb 路径
#   python 14b_nvfp4_token_accuracy.py --output out.json
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
LLM_ENGINE_DIR = ARTIFACTS / "engines/openvla_llama_nvfp4"
GOLDEN_DIR = ARTIFACTS / "golden/sample_0001"
LOGS_DIR = Path(os.environ.get("OPENVLA_LOGS_DIR", "/workspace/outputs/openvla"))

N_NEW = 7


def main() -> None:
    ap = argparse.ArgumentParser(description="NVFP4 7-DoF 动作 token 精度 (token 级)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--llm-engine", default=str(LLM_ENGINE_DIR),
                    help="LLM engine 目录 (默认 nvfp4; 可传 fp8 做 harness 交叉验证)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import torch
    from deploy.tensorrt.runtime.edge_llm_runner import EdgeLlmRunner
    from safetensors.numpy import load_file

    ctypes.CDLL(str(EDGE_LLM_DIR / "build/libNvInfer_edgellm_plugin.so"))
    plugin_path = EDGE_LLM_DIR / "build/libNvInfer_edgellm_plugin.so"
    engine_dir = Path(args.llm_engine)
    runner = EdgeLlmRunner(engine_dir, plugin_path, args.device)
    emb_tbl = load_file(str(engine_dir / "embedding.safetensors"))["embedding"]
    emb_tbl = torch.from_numpy(emb_tbl).to(args.device)

    # golden 数据
    meta = json.loads((GOLDEN_DIR / "metadata.json").read_text())
    golden_tokens = np.array(meta["generated_token_ids"], dtype=np.int64)
    mm = np.load(GOLDEN_DIR / "multimodal_embeddings.npy")  # (1,276,4096) bf16 golden
    stitched = torch.from_numpy(mm[0] if mm.ndim == 3 else mm).to(args.device)

    toks = runner.generate(stitched, emb_tbl, N_NEW)
    toks_np = np.array(toks, dtype=np.int64)

    diff = np.abs(toks_np - golden_tokens)  # token 绝对差 = bin 漂移数
    exact = toks_np == golden_tokens
    n_exact = int(exact.sum())

    low, high = int(meta["action_token_id_min"]), int(meta["action_token_id_max"])
    in_range = bool(np.all((toks_np >= low) & (toks_np <= high)))

    dims = ["dx", "dy", "dz", "roll", "pitch", "yaw", "gripper"]
    per_dim = [
        {"dim": d, "bf16_token": int(g), "nvfp4_token": int(n),
         "abs_diff": int(a), "exact": bool(e)}
        for d, g, n, a, e in zip(dims, golden_tokens, toks_np, diff, exact)
    ]

    result = {
        "precision": "nvfp4",
        "method": "golden-emb (喂 golden bf16 multimodal_embeddings, 只跑 LLM engine)",
        "engine": str(engine_dir),
        "n_new": N_NEW,
        "golden_tokens": golden_tokens.tolist(),
        "nvfp4_tokens": toks_np.tolist(),
        "token_abs_diff": diff.tolist(),
        "num_exact": n_exact,
        "num_total": len(toks_np),
        "frac_exact": round(n_exact / len(toks_np), 3),
        "max_diff_bin": int(diff.max()),
        "tokens_in_action_range": in_range,
        "action_range": [low, high],
        "per_dim": per_dim,
        "note": (
            "token 绝对差 = bin 漂移数 (OpenVLA 动作区间反向编码, 差 1 token = 漂 1 bin)。"
            "物理值 RMSE 需 bf16 model 的 bin_centers/action_stats, 本环境离线未加载, 待补。"
        ),
    }

    out = Path(args.output) if args.output else LOGS_DIR / "nvfp4_token_accuracy.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print("NVFP4 7-DoF 动作 token 精度 (token 级, golden-emb)")
    print(f"  golden : {golden_tokens.tolist()}")
    print(f"  nvfp4  : {toks_np.tolist()}")
    print(f"  完全一致: {n_exact}/{len(toks_np)}  |  最大 bin 漂移: {int(diff.max())}")
    print(f"  全部落在动作区间 [{low},{high}] : {in_range}")
    print(f"\n  per-dim:")
    for p in per_dim:
        flag = "OK" if p["exact"] else f"Δ{p['abs_diff']}"
        print(f"    {p['dim']:7s} bf16={p['bf16_token']:6d} nvfp4={p['nvfp4_token']:6d}  {flag}")
    print(f"\n输出: {out}")


if __name__ == "__main__":
    main()
