#!/usr/bin/env python3
# =============================================================================
# 12_fp8_action_accuracy.py - FP8 端到端真实动作 + 精度 (RMSE vs BF16 golden)
# =============================================================================
#
# 目标 (文档 §3):
#   打通 "task B" —— 把真实 vision embedding 注入 FP8 Edge-LLM engine,产出真实的
#   7-DoF 动作,再对比 BF16 golden 算 RMSE。此前 llm_inference 只能吃占位文本→乱码,
#   本脚本绕过 Edge-LLM C++ runtime,直接用 TensorRT Python API 驱动 llm.engine。
#
# 为什么能绕过 C++ runtime:
#   FP8 llm.engine 暴露的是标准显式 KV 契约 (inspector 确认):
#     IN : inputs_embeds(1,S,4096) f16, past_key_values_0..31(1,2,32,kv,128) f16,
#          rope_rotary_cos_sin(1,1024,128) f32, context_lengths(1) i32,
#          kvcache_start_index(1) i32, last_token_ids(1,1) i64
#     OUT: logits(1,1,32064) f32, present_key_values_0..31(1,2,32,kv,128) f16
#   KV 为原地 (past/present 同一 buffer,容量 1024),kvcache_start_index=写入位置。
#   约定取自 Edge-LLM 源码:
#     - RoPE: cpp/runtime/llmRuntimeUtils.cpp:43 initializeNormalRopeCosSinCacheHost
#       invFreq = pos / theta^(2d/128), cos∈[0:64], sin∈[64:128], theta=10000
#     - index: cpp/runtime/preprocess/stepPreparer.cpp
#       prefill: last_token_ids=S-1, context_lengths=S, start=0
#       decode : last_token_ids=0,   context_lengths=kv+1, start=kv
#
# 验证门 (关键,先跑再信):
#   --mode textcheck 用纯文本 prompt 跑本 harness,和 Edge-LLM llm_inference 在同一
#   engine 上的输出 token 逐一比对。都为 FP8,若 harness 的 KV/RoPE 正确则应完全一致。
#   只有 token-exact 通过后,才相信 --mode action 的 vision 注入结果。
#
# 用法:
#   python 12_fp8_action_accuracy.py --mode textcheck   # 先验证 harness 正确
#   python 12_fp8_action_accuracy.py --mode action      # 真实 FP8 动作 + RMSE
#   python 12_fp8_action_accuracy.py --mode golden-emb   # 用 golden 的 bf16 embedding
#                                                        # 喂 FP8 LLM (隔离 LLM 量化误差)
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

from deploy.tensorrt.runtime.edge_llm_runner import EdgeLlmRunner  # noqa: E402

EDGE_LLM_DIR = Path(os.environ.get("EDGE_LLM_DIR", "/workspace/TensorRT-Edge-LLM"))
ARTIFACTS = REPO_ROOT / "deploy/tensorrt/artifacts"
VISION_ENGINE = ARTIFACTS / "engines/vision_projector_fp8.plan"
LLM_ENGINE_DIR = ARTIFACTS / "engines/openvla_llama_fp8"
GOLDEN_DIR = ARTIFACTS / "golden/sample_0001"
PLUGIN = EDGE_LLM_DIR / "build/libNvInfer_edgellm_plugin.so"
LOGS_DIR = Path(os.environ.get("OPENVLA_LOGS_DIR", "/workspace/outputs/openvla"))

ROPE_THETA = 10000.0
ROPE_MAXLEN = 1024
HEAD_DIM = 128
KV_CAPACITY = 1024
N_LAYERS = 32
N_KV_HEADS = 32
HIDDEN = 4096
VOCAB = 32064


# ---------------------------------------------------------------------------
# RoPE cos/sin cache  (镜像 llmRuntimeUtils.cpp:initializeNormalRopeCosSinCacheHost)
# ---------------------------------------------------------------------------

def build_rope_cos_sin(max_len: int = ROPE_MAXLEN, rotary_dim: int = HEAD_DIM,
                       theta: float = ROPE_THETA) -> np.ndarray:
    half = rotary_dim // 2
    pos = np.arange(max_len, dtype=np.float64)[:, None]        # (L,1)
    d = np.arange(half, dtype=np.float64)[None, :]             # (1,half)
    inv_freq = pos / np.power(theta, 2.0 * d / rotary_dim)     # (L,half)
    cache = np.empty((max_len, rotary_dim), dtype=np.float32)
    cache[:, :half] = np.cos(inv_freq)
    cache[:, half:] = np.sin(inv_freq)
    return cache[None]                                         # (1,L,rotary_dim)


# ---------------------------------------------------------------------------
# FP8 LLM engine driver (显式 KV,原地 buffer)
# ---------------------------------------------------------------------------

class Fp8LlmEngine:
    def __init__(self, engine_path: Path, device: str = "cuda:0"):
        import tensorrt as trt
        import torch

        self.trt = trt
        self.torch = torch
        self.device = torch.device(device)
        ctypes.CDLL(str(PLUGIN))
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(self.device)

        # RoPE cache (常驻)
        rope = build_rope_cos_sin()
        self.rope = torch.from_numpy(rope).to(self.device)

        # 每层原地 KV buffer: past 和 present 绑同一块
        self.kv = [
            torch.zeros((1, 2, N_KV_HEADS, KV_CAPACITY, HEAD_DIM), dtype=torch.float16, device=self.device)
            for _ in range(N_LAYERS)
        ]
        self.logits = torch.empty((1, 1, VOCAB), dtype=torch.float32, device=self.device)

    def _run(self, embeds: "torch.Tensor", start: int, valid_len: int, last_idx: int) -> "torch.Tensor":
        torch = self.torch
        trt = self.trt
        seq = embeds.shape[0]
        ctx = self.context
        embeds = embeds.to(self.device, dtype=torch.float16).contiguous().view(1, seq, HIDDEN)

        ctx_lengths = torch.tensor([valid_len], dtype=torch.int32, device=self.device)
        start_idx = torch.tensor([start], dtype=torch.int32, device=self.device)
        last_tok = torch.tensor([[last_idx]], dtype=torch.int64, device=self.device)

        ctx.set_input_shape("inputs_embeds", (1, seq, HIDDEN))
        ctx.set_tensor_address("inputs_embeds", embeds.data_ptr())
        ctx.set_input_shape("rope_rotary_cos_sin", tuple(self.rope.shape))
        ctx.set_tensor_address("rope_rotary_cos_sin", self.rope.data_ptr())
        ctx.set_input_shape("context_lengths", (1,))
        ctx.set_tensor_address("context_lengths", ctx_lengths.data_ptr())
        ctx.set_input_shape("kvcache_start_index", (1,))
        ctx.set_tensor_address("kvcache_start_index", start_idx.data_ptr())
        ctx.set_input_shape("last_token_ids", (1, 1))
        ctx.set_tensor_address("last_token_ids", last_tok.data_ptr())
        for i in range(N_LAYERS):
            ptr = self.kv[i].data_ptr()
            ctx.set_input_shape(f"past_key_values_{i}", (1, 2, N_KV_HEADS, KV_CAPACITY, HEAD_DIM))
            ctx.set_tensor_address(f"past_key_values_{i}", ptr)
            ctx.set_tensor_address(f"present_key_values_{i}", ptr)  # 原地
        ctx.set_tensor_address("logits", self.logits.data_ptr())

        ok = ctx.execute_async_v3(self.stream.cuda_stream)
        if not ok:
            raise RuntimeError("execute_async_v3 returned False")
        self.stream.synchronize()
        return self.logits.view(VOCAB).clone()

    def generate(self, prefill_embeds: "torch.Tensor", embedding_table: "torch.Tensor",
                 n_new: int) -> list[int]:
        """prefill_embeds: (S,4096). 返回生成的 n_new 个 token id (greedy)。"""
        torch = self.torch
        for kv in self.kv:
            kv.zero_()
        S = prefill_embeds.shape[0]
        logits = self._run(prefill_embeds, start=0, valid_len=S, last_idx=S - 1)
        tok = int(torch.argmax(logits).item())
        out = [tok]
        cur = S
        for _ in range(n_new - 1):
            emb = embedding_table[tok].view(1, HIDDEN)
            logits = self._run(emb, start=cur, valid_len=cur + 1, last_idx=0)
            tok = int(torch.argmax(logits).item())
            out.append(tok)
            cur += 1
        return out


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def load_embedding_table(device):
    import torch
    from safetensors.numpy import load_file
    tbl = load_file(str(LLM_ENGINE_DIR / "embedding.safetensors"))["embedding"]  # (32064,4096) f16
    return torch.from_numpy(tbl).to(device)


def run_vision(pixel_values_np: np.ndarray, device: str):
    """跑 FP8 vision engine → projected patch embeddings (256,4096)。"""
    sys.path.insert(0, str(REPO_ROOT))
    from deploy.tensorrt.runtime.trt_runner import TensorRTRunner
    import torch
    runner = TensorRTRunner(VISION_ENGINE, device)
    pv = torch.from_numpy(pixel_values_np).to(device)
    out = runner({"pixel_values": pv})
    key = "projected_patch_embeddings" if "projected_patch_embeddings" in out else list(out)[0]
    return out[key]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="FP8 端到端真实动作 + RMSE (文档 §3)")
    ap.add_argument("--mode", choices=("action", "textcheck", "golden-emb"), default="action")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--unnorm-key", default="bridge_orig")
    ap.add_argument("--n-new", type=int, default=7)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import torch

    engine = EdgeLlmRunner(LLM_ENGINE_DIR, PLUGIN, args.device)
    emb_tbl = engine.load_embedding_table()

    result: dict = {"mode": args.mode, "engine": str(LLM_ENGINE_DIR / "llm.engine")}

    if args.mode == "textcheck":
        # 纯文本验证门: 用 embedding.safetensors 把 smoke prompt token 转 embedding,
        # 跑 harness,和 llm_inference 输出 token 比对。
        smoke = json.loads((ARTIFACTS / "smoke_input.json").read_text())
        # smoke_input requests[0] 的 token 需先 tokenize;用 golden 的 input_ids 兜底
        from deploy.tensorrt.common import load_openvla, prompt_for
        processor, _ = load_openvla(args.device, "bf16")
        content = smoke["requests"][0]["messages"][0]["content"] if "requests" in smoke else "In: What action should the robot take to pick up the blue object?\nOut:"
        ids = processor.tokenizer(content, return_tensors="np")["input_ids"][0]
        embeds = emb_tbl[torch.from_numpy(ids).to(args.device)]
        toks = engine.generate(embeds, emb_tbl, args.n_new)
        result.update({"prompt_token_len": int(len(ids)), "harness_tokens": toks})
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("\n>>> 现在运行 llm_inference 对同一 prompt 取 token,人工比对 harness_tokens 是否一致。")
        return

    # action / golden-emb: 需要 golden + 拼接
    from deploy.tensorrt.common import load_openvla, build_multimodal_inputs, decode_action_tokens, action_token_bounds
    meta = json.loads((GOLDEN_DIR / "metadata.json").read_text())
    golden_action = np.array(meta["action"], dtype=np.float64)
    golden_tokens = list(meta["generated_token_ids"])
    processor, model = load_openvla(args.device, "bf16")

    input_ids = torch.from_numpy(np.load(GOLDEN_DIR / "input_ids.npy")).to(args.device)
    attn = torch.ones_like(input_ids)

    if args.mode == "golden-emb":
        # 直接用 golden 的 bf16 stitched embedding 喂 FP8 LLM → 隔离 LLM 量化误差
        mm = np.load(GOLDEN_DIR / "multimodal_embeddings.npy")  # (1,276,4096)
        stitched = torch.from_numpy(mm[0]).to(args.device)
        source = "golden bf16 multimodal_embeddings"
    else:
        # 真实 FP8 路径: FP8 vision engine → 拼接
        pv = np.load(GOLDEN_DIR / "pixel_values.npy").astype(np.float32)
        projected = run_vision(pv, args.device).to(torch.float32)
        if projected.dim() == 2:
            projected = projected.unsqueeze(0)
        with torch.inference_mode():
            mm_emb, _ = build_multimodal_inputs(model, input_ids, attn, projected)
        stitched = mm_emb[0]
        source = "FP8 vision engine + embedding.safetensors text"

    toks = engine.generate(stitched, emb_tbl, args.n_new)
    toks_np = np.array(toks)
    action = decode_action_tokens(model, toks_np, args.unnorm_key)
    action = np.asarray(action, dtype=np.float64)

    abs_err = np.abs(action - golden_action)
    rel_err = abs_err / (np.abs(golden_action) + 1e-8)
    rmse = float(np.sqrt(np.mean((action - golden_action) ** 2)))
    low, high = action_token_bounds(model)

    dims = ["dx", "dy", "dz", "roll", "pitch", "yaw", "gripper"]
    result.update({
        "embedding_source": source,
        "fp8_tokens": toks,
        "golden_tokens": golden_tokens,
        "token_exact_match": bool(np.array_equal(toks_np, np.array(golden_tokens))),
        "tokens_in_action_range": bool(np.all((toks_np >= low) & (toks_np <= high))),
        "fp8_action": action.tolist(),
        "golden_action": golden_action.tolist(),
        "per_dim": {d: {"fp8": float(action[i]), "bf16": float(golden_action[i]),
                        "abs_err": float(abs_err[i]), "rel_err": float(rel_err[i])}
                    for i, d in enumerate(dims)},
        "rmse": rmse,
        "max_abs_err": float(abs_err.max()),
    })

    out_path = Path(args.output) if args.output else LOGS_DIR / f"fp8_action_accuracy_{args.mode}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print(f"mode={args.mode}  source={source}")
    print(f"FP8 tokens   : {toks}")
    print(f"golden tokens: {golden_tokens}  (exact match: {result['token_exact_match']})")
    print(f"{'dim':8s} {'fp8':>12s} {'bf16':>12s} {'abs_err':>12s}")
    for i, d in enumerate(dims):
        print(f"{d:8s} {action[i]:12.6f} {golden_action[i]:12.6f} {abs_err[i]:12.6f}")
    print(f"RMSE = {rmse:.6f}   max_abs_err = {abs_err.max():.6f}")
    print(f"输出: {out_path}")


if __name__ == "__main__":
    main()
