"""
算子清单脚本 (operator inventory) —— 面向 NPU 硬件设计

目的：把 OpenVLA-7B 一次 predict_action 推理中用到的所有算子完整列出来，
供 NPU 算子开发参考。输出三个维度：

  A. aten 底层算子清单 (torch.profiler)
     - 每种 aten 算子的调用次数、CPU/CUDA 累计耗时、占比
     - 这是硬件真正需要实现的算子级别 (matmul / softmax / layer_norm / gelu ...)

  B. nn.Module 层清单 (模型结构遍历)
     - 每类 Module (Linear / LayerNorm / Conv2d / Attention / SiLU ...) 的数量
     - 这是从网络结构角度看的算子组成

  C. 关键张量形状 (forward hook)
     - 各类算子的典型 输入/输出 shape，用于评估 NPU 的 tiling / 数据流设计

结果分别写入 operator_inventory.json 和一份可读的文本报告。
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from runtime_env import MODEL_PATH

import torch
from PIL import Image
from torch.profiler import profile, ProfilerActivity
from transformers import AutoModelForVision2Seq, AutoProcessor


MODEL_ID = str(MODEL_PATH)
DEVICE = os.getenv("OPENVLA_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
ATTN_IMPLEMENTATION = os.getenv("OPENVLA_ATTN_IMPLEMENTATION", "sdpa")
UNNORM_KEY = os.getenv("OPENVLA_UNNORM_KEY", "bridge_orig")

DEFAULT_IMAGE_DIR = Path(__file__).resolve().parents[1] / "test_data"


def model_dtype() -> torch.dtype:
    return torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32


def prompt_for(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def shape_of(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return list(x.shape)
    if isinstance(x, (tuple, list)):
        return [shape_of(v) for v in x if isinstance(v, torch.Tensor)] or None
    return None


# ---------------- B. nn.Module 层清单 ----------------
def collect_module_inventory(model: torch.nn.Module) -> dict[str, Any]:
    leaf_counter: Counter = Counter()
    act_counter: Counter = Counter()
    detail: dict[str, int] = defaultdict(int)

    for _name, module in model.named_modules():
        children = list(module.children())
        if children:
            continue  # 只统计叶子模块
        cls = type(module).__name__
        leaf_counter[cls] += 1
        # 激活函数单独归类
        if cls in {"GELU", "SiLU", "ReLU", "QuickGELU", "GELUActivation", "SiLUActivation", "Tanh", "Sigmoid"}:
            act_counter[cls] += 1

    return {
        "leaf_module_counts": dict(leaf_counter.most_common()),
        "activation_counts": dict(act_counter.most_common()),
        "total_leaf_modules": sum(leaf_counter.values()),
    }


# ---------------- C. 关键张量形状 (forward hook) ----------------
def register_shape_hooks(model: torch.nn.Module, shape_records: dict[str, list]) -> list[Any]:
    handles: list[Any] = []
    # 我们关心这些算子类的 IO shape
    watch_types = {
        "Linear", "Conv2d", "LayerNorm", "RMSNorm", "LlamaRMSNorm",
        "GELU", "SiLU", "Embedding", "Attention", "LlamaAttention",
        "LlamaSdpaAttention", "LlamaMLP", "PatchEmbed",
    }
    seen: set = set()

    def make_hook(cls_name: str):
        def hook(module: Any, inputs: Any, output: Any) -> None:
            in_shape = shape_of(inputs)
            out_shape = shape_of(output)
            key = f"{cls_name}|{in_shape}|{out_shape}"
            if key in seen:
                return
            seen.add(key)
            extra: dict[str, Any] = {}
            if isinstance(module, torch.nn.Linear):
                extra = {"in_features": module.in_features, "out_features": module.out_features,
                         "bias": module.bias is not None}
            elif isinstance(module, torch.nn.Conv2d):
                extra = {"in_ch": module.in_channels, "out_ch": module.out_channels,
                         "kernel": list(module.kernel_size), "stride": list(module.stride)}
            shape_records[cls_name].append({
                "input_shape": in_shape, "output_shape": out_shape, **extra,
            })
        return hook

    for _name, module in model.named_modules():
        cls = type(module).__name__
        if cls in watch_types and not list(module.children()):
            handles.append(module.register_forward_hook(make_hook(cls)))
        # Attention 类通常有子模块，也挂上
        elif "Attention" in cls:
            handles.append(module.register_forward_hook(make_hook(cls)))
    return handles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None)
    parser.add_argument("--instruction", default="pick up the object")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    image_dir = DEFAULT_IMAGE_DIR
    if args.image:
        image_path = Path(args.image)
    else:
        candidates = sorted(image_dir.glob("*.jpg"))
        image_path = candidates[0] if candidates else None

    output_dir = Path(os.getenv("OPENVLA_PREFIX", str(Path(__file__).resolve().parents[1]))) / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.output) if args.output else output_dir / "operator_inventory.json"
    out_report = out_json.with_suffix(".report.txt")

    print("model:", MODEL_ID)
    print("device:", DEVICE)
    print("attention:", ATTN_IMPLEMENTATION)
    print("image:", image_path)

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        attn_implementation=ATTN_IMPLEMENTATION,
        torch_dtype=model_dtype(),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    ).to(DEVICE)
    model.eval()

    prompt = prompt_for(args.instruction)
    if image_path is not None:
        image = Image.open(image_path).convert("RGB")
    else:
        image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    inputs = processor(prompt, image).to(DEVICE, dtype=model_dtype())

    # ---- B. Module 清单 (静态) ----
    module_inv = collect_module_inventory(model)

    # ---- C. Shape hook (跑一次收集) ----
    shape_records: dict[str, list] = defaultdict(list)
    shape_handles = register_shape_hooks(model, shape_records)
    with torch.inference_mode():
        model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
    for h in shape_handles:
        h.remove()

    # ---- A. aten 算子 profiler ----
    activities = [ProfilerActivity.CPU]
    if DEVICE.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)
    # 预热
    with torch.inference_mode():
        model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()

    with profile(activities=activities, record_shapes=False, with_stack=False) as prof:
        with torch.inference_mode():
            model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()

    # 聚合 aten 算子
    aten_ops: list[dict[str, Any]] = []
    for evt in prof.key_averages():
        name = evt.key
        cuda_us = 0.0
        for attr in ("cuda_time_total", "device_time_total", "self_cuda_time_total", "self_device_time_total"):
            if hasattr(evt, attr):
                try:
                    cuda_us = float(getattr(evt, attr))
                    if cuda_us:
                        break
                except Exception:
                    pass
        aten_ops.append({
            "op": name,
            "count": int(evt.count),
            "cpu_time_total_us": float(evt.cpu_time_total),
            "cuda_time_total_us": cuda_us,
        })
    aten_ops.sort(key=lambda r: max(r["cuda_time_total_us"], r["cpu_time_total_us"]), reverse=True)

    # 只保留 aten:: / 关键算子, 归并算子族
    op_family: Counter = Counter()
    for r in aten_ops:
        base = r["op"].split(".")[0]
        op_family[base] += r["count"]

    inventory = {
        "model_id": MODEL_ID,
        "device": DEVICE,
        "attention": ATTN_IMPLEMENTATION,
        "unnorm_key": UNNORM_KEY,
        "action_dim_tokens": model.get_action_dim(UNNORM_KEY),
        "architecture": {
            "vision_backbone": "DINOv2 ViT-L/14 + SigLIP ViT-SO400M (fused)",
            "projector": "3-layer MLP + GELU",
            "llm_backbone": "Llama-2-7B",
            "note": "predict_action = autoregressive generate() of action tokens",
        },
        "B_module_inventory": module_inv,
        "C_tensor_shapes": {k: v for k, v in shape_records.items()},
        "A_aten_operators": aten_ops,
        "A_aten_operator_families": dict(op_family.most_common()),
    }
    out_json.write_text(json.dumps(inventory, indent=2, ensure_ascii=True), encoding="utf-8")

    # ---- 可读文本报告 ----
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("OpenVLA-7B 算子清单 (面向 NPU 硬件设计)")
    lines.append("=" * 70)
    lines.append(f"model      : {MODEL_ID}")
    lines.append(f"device     : {DEVICE}")
    lines.append(f"attention  : {ATTN_IMPLEMENTATION}")
    lines.append(f"action dim : {inventory['action_dim_tokens']} tokens (自回归生成)")
    lines.append("")
    lines.append("架构: DINOv2+SigLIP 融合视觉 -> 3层MLP projector -> Llama-2-7B")
    lines.append("")
    lines.append("-" * 70)
    lines.append("B. nn.Module 叶子层统计 (网络结构视角)")
    lines.append("-" * 70)
    for cls, cnt in module_inv["leaf_module_counts"].items():
        lines.append(f"  {cls:<28} x {cnt}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("A. aten 底层算子清单 (硬件需实现的算子, 按耗时排序)")
    lines.append("-" * 70)
    lines.append(f"  {'operator':<40} {'count':>8} {'cpu_us':>12} {'cuda_us':>12}")
    for r in aten_ops:
        lines.append(f"  {r['op']:<40} {r['count']:>8} {r['cpu_time_total_us']:>12.1f} {r['cuda_time_total_us']:>12.1f}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("A2. 算子族汇总 (按调用次数)")
    lines.append("-" * 70)
    for fam, cnt in op_family.most_common():
        lines.append(f"  {fam:<40} {cnt:>8}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("C. 关键算子典型张量形状 (NPU tiling/数据流参考)")
    lines.append("-" * 70)
    for cls, records in shape_records.items():
        lines.append(f"  [{cls}] 共 {len(records)} 种不同 shape:")
        for rec in records[:12]:
            extra = {k: v for k, v in rec.items() if k not in ("input_shape", "output_shape")}
            lines.append(f"     in={rec['input_shape']} -> out={rec['output_shape']}  {extra if extra else ''}")
        if len(records) > 12:
            lines.append(f"     ... (+{len(records) - 12} more)")
    report_text = "\n".join(lines)
    out_report.write_text(report_text, encoding="utf-8")

    print(report_text)
    print("\njson  :", out_json)
    print("report:", out_report)


if __name__ == "__main__":
    main()
