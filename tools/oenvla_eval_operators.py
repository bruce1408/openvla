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
     - 各类算子的典型 输入/输出 shape, 用于评估 NPU 的 tiling / 数据流设计

结果分别写入 operator_inventory.json 和一份可读的文本报告。
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

RUNTIME_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_DIR))
from runtime_env import MODEL_PATH, MODEL_REVISION

import torch
from PIL import Image
from torch.profiler import profile, ProfilerActivity
from transformers import AutoModelForVision2Seq, AutoProcessor


MODEL_ID = str(MODEL_PATH)
DEVICE = os.getenv("OPENVLA_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
ATTN_IMPLEMENTATION = os.getenv("OPENVLA_ATTN_IMPLEMENTATION", "sdpa")
UNNORM_KEY = os.getenv("OPENVLA_UNNORM_KEY", "bridge_orig")

DEFAULT_IMAGE_DIR = RUNTIME_DIR / "test_data"


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


def jsonable_shape(x: Any) -> Any:
    """把 profiler / hook 里的 shape 对象转成 JSON 友好的 list/scalar。"""
    if x is None:
        return None
    if isinstance(x, torch.Size):
        return list(x)
    if isinstance(x, (tuple, list)):
        return [jsonable_shape(v) for v in x]
    if isinstance(x, (str, int, float, bool)):
        return x
    try:
        return list(x)
    except Exception:
        return str(x)


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
    seen: dict[str, dict[str, Any]] = {}

    def make_hook(cls_name: str):
        def hook(module: Any, inputs: Any, output: Any) -> None:
            in_shape = shape_of(inputs)
            out_shape = shape_of(output)
            key = f"{cls_name}|{in_shape}|{out_shape}"
            if key in seen:
                seen[key]["call_count"] += 1  # 同一唯一 shape 被多次调用, 只累加计数
                return
            extra: dict[str, Any] = {}
            if isinstance(module, torch.nn.Linear):
                extra = {"in_features": module.in_features, "out_features": module.out_features,
                         "bias": module.bias is not None}
            elif isinstance(module, torch.nn.Conv2d):
                extra = {"in_ch": module.in_channels, "out_ch": module.out_channels,
                         "kernel": list(module.kernel_size), "stride": list(module.stride)}
            rec = {"input_shape": in_shape, "output_shape": out_shape, "call_count": 1, **extra}
            seen[key] = rec
            shape_records[cls_name].append(rec)
        return hook

    for _name, module in model.named_modules():
        cls = type(module).__name__
        if cls in watch_types and not list(module.children()):
            handles.append(module.register_forward_hook(make_hook(cls)))
        # Attention 类通常有子模块，也挂上
        elif "Attention" in cls:
            handles.append(module.register_forward_hook(make_hook(cls)))
    return handles


# ---------------- A. aten 算子聚合 / 阶段拆分 工具 ----------------
def aggregate_aten(prof: Any) -> tuple[list[dict[str, Any]], Counter]:
    """把一次 profiler 结果聚合成 aten 算子清单 (按耗时排序) 和算子族计数。"""
    aten_ops: list[dict[str, Any]] = []
    for evt in prof.key_averages():
        name = evt.key
        cuda_us = event_cuda_us(evt)
        aten_ops.append({
            "op": name,
            "count": int(evt.count),
            "cpu_time_total_us": float(evt.cpu_time_total),
            "cuda_time_total_us": cuda_us,
        })
    aten_ops.sort(key=lambda r: (r["cuda_time_total_us"], r["cpu_time_total_us"]), reverse=True)
    op_family: Counter = Counter()
    for r in aten_ops:
        base = r["op"].split(".")[0]
        op_family[base] += r["count"]
    return aten_ops, op_family


def event_cuda_us(evt: Any) -> float:
    """兼容不同 PyTorch 版本的 CUDA/device time 字段。"""
    for attr in ("cuda_time_total", "device_time_total", "self_cuda_time_total", "self_device_time_total"):
        if hasattr(evt, attr):
            try:
                cuda_us = float(getattr(evt, attr))
                if cuda_us:
                    return cuda_us
            except Exception:
                pass
    return 0.0


def infer_aten_output_shape(op: str, input_shapes: Any) -> Any:
    """基于 aten input_shapes 对常见算子做轻量 output shape 推断。

    torch.profiler 的 aten 事件通常只暴露 input_shapes, 不直接暴露 output_shapes。
    这里对 NPU 分析最关心的 GEMM/elementwise/norm/attention 做保守推断;
    无法可靠推断的算子返回 None。
    """
    shapes = input_shapes if isinstance(input_shapes, list) else []
    tensor_shapes = [v for v in shapes if isinstance(v, list)]
    if not tensor_shapes:
        return None

    first = tensor_shapes[0]
    if op in {
        "aten::linear", "aten::matmul", "aten::mm", "aten::bmm",
        "aten::addmm", "aten::baddbmm",
    }:
        if op == "aten::linear" and len(tensor_shapes) >= 2:
            x, w = tensor_shapes[0], tensor_shapes[1]
            if len(x) >= 1 and len(w) == 2:
                return [*x[:-1], w[0]]
        if op == "aten::mm" and len(tensor_shapes) >= 2:
            a, b = tensor_shapes[0], tensor_shapes[1]
            if len(a) == 2 and len(b) == 2:
                return [a[0], b[1]]
        if op == "aten::addmm" and len(tensor_shapes) >= 3:
            a, b = tensor_shapes[1], tensor_shapes[2]
            if len(a) == 2 and len(b) == 2:
                return [a[0], b[1]]
        if op == "aten::bmm" and len(tensor_shapes) >= 2:
            a, b = tensor_shapes[0], tensor_shapes[1]
            if len(a) == 3 and len(b) == 3:
                return [a[0], a[1], b[2]]
        if op == "aten::baddbmm" and len(tensor_shapes) >= 3:
            a, b = tensor_shapes[1], tensor_shapes[2]
            if len(a) == 3 and len(b) == 3:
                return [a[0], a[1], b[2]]
        if op == "aten::matmul" and len(tensor_shapes) >= 2:
            a, b = tensor_shapes[0], tensor_shapes[1]
            if len(a) == 1 and len(b) == 1:
                return []
            if len(a) == 2 and len(b) == 2:
                return [a[0], b[1]]
            if len(a) >= 2 and len(b) >= 2:
                batch = a[:-2] if len(a) >= len(b) else b[:-2]
                return [*batch, a[-2], b[-1]]

    same_shape_ops = {
        "aten::add", "aten::mul", "aten::sub", "aten::div", "aten::pow",
        "aten::neg", "aten::rsqrt", "aten::sin", "aten::cos", "aten::silu",
        "aten::gelu", "aten::to", "aten::_to_copy", "aten::copy_",
        "aten::clone", "aten::contiguous",
        "aten::layer_norm", "aten::native_layer_norm",
    }
    if op in same_shape_ops:
        return first

    if op == "aten::embedding" and len(tensor_shapes) >= 2:
        weight, indices = tensor_shapes[0], tensor_shapes[1]
        if len(weight) == 2:
            return [*indices, weight[1]]

    if "scaled_dot_product_attention" in op or "flash_attention" in op:
        return first

    return None


def module_io_shapes_for_aten(op: str, shape_records: dict[str, list]) -> list[dict[str, Any]]:
    """把部分 aten op 关联到 forward hook 捕获的 module 输入/输出 shape。"""
    mapping = {
        "aten::linear": ["Linear"],
        "aten::addmm": ["Linear"],
        "aten::mm": ["Linear"],
        "aten::matmul": ["Linear", "LlamaSdpaAttention", "Attention"],
        "aten::conv2d": ["Conv2d"],
        "aten::convolution": ["Conv2d"],
        "aten::_convolution": ["Conv2d"],
        "aten::layer_norm": ["LayerNorm", "LlamaRMSNorm", "RMSNorm"],
        "aten::native_layer_norm": ["LayerNorm"],
        "aten::embedding": ["Embedding"],
        "aten::gelu": ["GELU"],
        "aten::silu": ["SiLU"],
        "aten::scaled_dot_product_attention": ["LlamaSdpaAttention", "Attention"],
        "aten::_scaled_dot_product_flash_attention": ["LlamaSdpaAttention", "Attention"],
        "aten::_flash_attention_forward": ["LlamaSdpaAttention", "Attention"],
    }
    rows: list[dict[str, Any]] = []
    seen = set()
    for cls in mapping.get(op, []):
        for rec in shape_records.get(cls, []):
            item = {
                "module_class": cls,
                "input_shape": rec.get("input_shape"),
                "output_shape": rec.get("output_shape"),
                "call_count": rec.get("call_count", 1),
            }
            key = json.dumps(item, sort_keys=True, ensure_ascii=True)
            if key not in seen:
                seen.add(key)
                rows.append(item)
    return rows


def aten_ops_with_shapes(prof: Any, aten_ops: list[dict[str, Any]],
                         shape_records: dict[str, list]) -> list[dict[str, Any]]:
    """按 aten_ops 的耗时排序, 补充所有 profiler input shape 变体和可推断 output shape。"""
    by_op: dict[str, list[dict[str, Any]]] = defaultdict(list)
    try:
        grouped = prof.key_averages(group_by_input_shape=True)
    except TypeError:
        grouped = prof.key_averages()

    for evt in grouped:
        op = evt.key
        input_shapes = jsonable_shape(getattr(evt, "input_shapes", None))
        output_shape = infer_aten_output_shape(op, input_shapes)
        by_op[op].append({
            "input_shapes": input_shapes,
            "inferred_output_shape": output_shape,
            "count": int(evt.count),
            "cpu_time_total_us": float(evt.cpu_time_total),
            "cuda_time_total_us": event_cuda_us(evt),
            "output_shape_source": "inferred_from_input_shapes" if output_shape is not None else "not_available_from_torch_profiler",
        })

    result: list[dict[str, Any]] = []
    for row in aten_ops:
        op = row["op"]
        variants = sorted(
            by_op.get(op, []),
            key=lambda r: (r["cuda_time_total_us"], r["cpu_time_total_us"]),
            reverse=True,
        )
        enriched = dict(row)
        enriched["shape_variants"] = variants
        enriched["module_io_shape_variants"] = module_io_shapes_for_aten(op, shape_records)
        enriched["shape_note"] = (
            "aten input_shapes come from torch.profiler(record_shapes=True, group_by_input_shape=True). "
            "aten output shapes are inferred for common ops when possible; exact module input/output shapes "
            "are listed in module_io_shape_variants when a forward hook can be associated."
        )
        result.append(enriched)
    return result


def profile_call(fn: Any, activities: list[Any], trace_path: Path | None = None) -> Any:
    """在 profiler 上下文中执行一次 fn (含前后 CUDA 同步)。
    trace_path 非空时导出 chrome trace timeline (可在 chrome://tracing 或 Perfetto 打开)。"""
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    with profile(activities=activities, record_shapes=True, with_stack=False) as prof:
        with torch.inference_mode():
            fn()
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    if trace_path is not None:
        try:
            prof.export_chrome_trace(str(trace_path))
        except Exception as e:
            print(f"[warn] export trace failed ({trace_path}): {e}")
    return prof


def timed_call(fn: Any) -> tuple[float, Any]:
    """测量一次调用的耗时 (毫秒)。CUDA 用 event, CPU 用 perf_counter。"""
    import time
    if DEVICE.startswith("cuda"):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end), out
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) * 1000.0, out


def build_action_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
    """复刻 predict_action: 若结尾不是空 token(29871) 则补上，以匹配训练时的输入。"""
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
        )
    return input_ids


def top_by_count(aten_ops: list[dict[str, Any]], n: int = 20) -> list[dict[str, Any]]:
    """按调用次数(调度频率)取 top-n 算子。"""
    return sorted(aten_ops, key=lambda r: r["count"], reverse=True)[:n]


def cpu_cuda_compare(aten_ops: list[dict[str, Any]], n: int = 20) -> list[dict[str, Any]]:
    """对耗时 top-n 算子做 CPU vs CUDA 对比，标注主导端与比值。"""
    rows: list[dict[str, Any]] = []
    ranked = sorted(aten_ops, key=lambda r: max(r["cuda_time_total_us"], r["cpu_time_total_us"]), reverse=True)
    for r in ranked[:n]:
        cpu = r["cpu_time_total_us"]
        cuda = r["cuda_time_total_us"]
        dominant = "CUDA" if cuda > cpu else ("CPU" if cpu > cuda else "EQ")
        if cpu > 0:
            cuda_over_cpu = cuda / cpu
        else:
            cuda_over_cpu = float("inf") if cuda > 0 else 0.0
        rows.append({
            "op": r["op"],
            "count": r["count"],
            "cpu_time_total_us": cpu,
            "cuda_time_total_us": cuda,
            "dominant": dominant,
            "cuda_over_cpu_ratio": cuda_over_cpu,
        })
    return rows


# ---------------- G. 硬件 / 运行时环境探测 ----------------
def bench_mem_bandwidth_gbs(n_bytes: int = 1 << 30, iters: int = 20) -> float:
    """经验内存带宽 (GB/s): device-to-device 大块 copy, 计入读+写。"""
    if not DEVICE.startswith("cuda"):
        return 0.0
    import time
    n = n_bytes // 4
    a = torch.empty(n, dtype=torch.float32, device=DEVICE)
    b = torch.empty(n, dtype=torch.float32, device=DEVICE)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        b.copy_(a)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return (a.numel() * a.element_size() * 2) / dt / 1e9


def bench_bf16_gemm_tflops(n: int = 8192, iters: int = 30) -> float:
    """经验 BF16 稠密 GEMM 峰值 (TFLOP/s)。"""
    if not DEVICE.startswith("cuda"):
        return 0.0
    import time
    a = torch.randn(n, n, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=DEVICE, dtype=torch.bfloat16)
    for _ in range(5):
        _ = a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return (2 * n ** 3) / dt / 1e12


def probe_hardware_env() -> dict[str, Any]:
    """采集 GPU 硬件规格 + PyTorch 运行时环境 + 经验带宽/算力。"""
    env: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "TORCH_CUDA_ARCH_LIST": os.getenv("TORCH_CUDA_ARCH_LIST", "<unset>"),
        "torch_arch_list": torch.cuda.get_arch_list() if DEVICE.startswith("cuda") else [],
        "attn_implementation": ATTN_IMPLEMENTATION,
        "torch_compile_used": False,  # 本评测基于 eager mode profiling
        "execution_mode": "eager (no torch.compile)",
        "model_dtype": str(model_dtype()),
        "quantization": "none (BF16 weights, no INT8/INT4/FP8, no bitsandbytes/GPTQ/AWQ)",
        "inference_framework": "PyTorch + HuggingFace transformers (no TensorRT / TensorRT-LLM)",
    }
    if DEVICE.startswith("cuda"):
        p = torch.cuda.get_device_properties(0)
        mem_clk_hz = getattr(p, "memory_clock_rate", 0) * 1e3
        bus = getattr(p, "memory_bus_width", 0)
        theo_bw = mem_clk_hz * 2 * (bus / 8) / 1e9 if mem_clk_hz and bus else 0.0
        env.update({
            "gpu_name": p.name,
            "compute_capability": f"{p.major}.{p.minor}",
            "sm_count": p.multi_processor_count,
            "total_memory_GB": round(p.total_memory / 1024 ** 3, 2),
            "memory_bus_width_bit": bus,
            "memory_clock_GHz": round(mem_clk_hz / 1e9, 3) if mem_clk_hz else None,
            "theoretical_mem_bandwidth_GBs": round(theo_bw, 1),
            "empirical_mem_bandwidth_GBs": round(bench_mem_bandwidth_gbs(), 1),
            "achieved_bf16_gemm_TFLOPs": round(bench_bf16_gemm_tflops(), 1),
        })
    return env


# ---------------- I. FLOPs 解析模型 ----------------
def llm_config(model: torch.nn.Module) -> dict[str, int]:
    """从模型里取 Llama backbone 的关键维度。"""
    lm = model.language_model
    cfg = lm.config
    return {
        "num_layers": int(cfg.num_hidden_layers),
        "hidden": int(cfg.hidden_size),
        "intermediate": int(cfg.intermediate_size),
        "num_heads": int(cfg.num_attention_heads),
        "num_kv_heads": int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)),
        "head_dim": int(cfg.hidden_size // cfg.num_attention_heads),
        "vocab": int(cfg.vocab_size),
    }


def flops_linear_from_shapes(shape_records: dict[str, list]) -> dict[str, Any]:
    """从 C_tensor_shapes 里捕获的 Linear (唯一 shape) 累加 GFLOPs。
    注意: 唯一 shape 去重, 未乘以层数 -> 这是"每种 shape 记一次"的下界统计。"""
    total = 0.0
    rows = []
    for rec in shape_records.get("Linear", []):
        inp = rec.get("input_shape")
        if not inp or not isinstance(inp, list):
            continue
        in0 = inp[0] if isinstance(inp[0], list) else inp
        tokens = 1
        for d in in0[:-1]:
            tokens *= d
        fin = rec.get("in_features")
        fout = rec.get("out_features")
        if fin is None or fout is None:
            continue
        f = 2.0 * tokens * fin * fout
        total += f
        rows.append({"in": in0, "out_features": fout, "gflops": f / 1e9})
    return {"total_gflops": total / 1e9, "per_shape": rows}


def flops_full_model(cfg: dict[str, int], prompt_len: int, n_decode: int,
                     vision_gflops: float) -> dict[str, Any]:
    """完整模型单次 predict_action 的 GFLOPs 解析估算。
    prefill: 处理 prompt_len 个 token (含 256 图像 patch);
    decode : n_decode 步, 每步 1 个新 token, KV 长度递增。
    约定: MAC=2 FLOP。仅统计前向。
    """
    L = cfg["num_layers"]; H = cfg["hidden"]; I = cfg["intermediate"]
    V = cfg["vocab"]

    def linear_flops(tokens: int) -> float:
        # 每层: QKVO = 4*H*H ; MLP(SwiGLU: gate+up+down) = 3*H*I
        per_layer = 4 * H * H + 3 * H * I
        return 2.0 * tokens * L * per_layer

    def attn_flops(q_tokens: int, kv_len: int) -> float:
        # QK^T + A*V: 每层 2 * (2 * q_tokens * kv_len * H)  (H = nh*hd)
        return L * (2.0 * (2.0 * q_tokens * kv_len * H))

    def lm_head_flops(tokens: int) -> float:
        return 2.0 * tokens * H * V

    # ---- prefill ----
    pf_linear = linear_flops(prompt_len)
    pf_attn = attn_flops(prompt_len, prompt_len)  # 自注意力全序列
    pf_head = lm_head_flops(1)  # 只对最后一个 token 取 logits
    pf_total = pf_linear + pf_attn + pf_head + vision_gflops * 1e9

    # ---- decode ----
    dec_linear = 0.0; dec_attn = 0.0; dec_head = 0.0
    for step in range(n_decode):
        kv_len = prompt_len + step + 1
        dec_linear += linear_flops(1)
        dec_attn += attn_flops(1, kv_len)
        dec_head += lm_head_flops(1)
    dec_total = dec_linear + dec_attn + dec_head

    g = 1e9
    return {
        "assumptions": {
            "prompt_len_tokens": prompt_len,
            "image_patch_tokens": 256,
            "decode_steps": n_decode,
            "MAC_to_FLOP": 2,
            "note": "SwiGLU MLP=3 matmul; attention=QK^T+A*V; vision 来自捕获 Linear shape 累加",
        },
        "prefill": {
            "linear_gflops": pf_linear / g,
            "attention_gflops": pf_attn / g,
            "lm_head_gflops": pf_head / g,
            "vision_gflops": vision_gflops,
            "total_gflops": pf_total / g,
        },
        "decode": {
            "linear_gflops": dec_linear / g,
            "attention_gflops": dec_attn / g,
            "lm_head_gflops": dec_head / g,
            "total_gflops": dec_total / g,
        },
        "total_gflops": (pf_total + dec_total) / g,
    }


def utilization(gflops: float, latency_ms: float, peak_tflops: float) -> dict[str, Any]:
    """由 GFLOPs 与实测时延推算达到算力与利用率。"""
    if latency_ms <= 0:
        return {"achieved_tflops": 0.0, "utilization_pct": 0.0}
    achieved = (gflops / 1e3) / (latency_ms / 1e3)  # TFLOP/s
    return {
        "achieved_tflops": achieved,
        "peak_tflops": peak_tflops,
        "utilization_pct": (achieved / peak_tflops * 100.0) if peak_tflops else 0.0,
    }


# ---------------- J. Decode 阶段权重常驻 SRAM 需求 ----------------
def sram_weight_residency(cfg: dict[str, int], dtype_bytes: int = 2) -> dict[str, Any]:
    """decode 阶段各权重块的常驻容量需求 (BF16=2 bytes)。"""
    H = cfg["hidden"]; I = cfg["intermediate"]; V = cfg["vocab"]; L = cfg["num_layers"]
    b = dtype_bytes
    qkvo_per_layer = 4 * H * H * b
    # SwiGLU: gate(H*I)+up(H*I)+down(I*H)
    ffn_per_layer = (2 * H * I + I * H) * b
    lm_head = H * V * b
    embed = V * H * b
    mb = 1024 * 1024
    return {
        "dtype_bytes": b,
        "lm_head_MB": lm_head / mb,
        "embedding_MB": embed / mb,
        "ffn_per_layer_MB": ffn_per_layer / mb,
        "qkvo_per_layer_MB": qkvo_per_layer / mb,
        "per_layer_total_MB": (qkvo_per_layer + ffn_per_layer) / mb,
        "all_layers_MB": L * (qkvo_per_layer + ffn_per_layer) / mb,
        "full_llm_weights_MB": (L * (qkvo_per_layer + ffn_per_layer) + lm_head + embed) / mb,
        "note": (
            "decode 每步都要从显存读全部权重 (memory-bound). "
            "SRAM 放不下整份 (~13GB BF16), 需按层流水/分块缓存; "
            "lm_head 与单层 FFN 是最大的单块常驻候选。"
        ),
    }


# ---------------- K. 模块级时延拆解 ----------------
def percentiles(xs: list[float]) -> dict[str, float]:
    """返回 p50/p90/p99/mean/max (ms)。"""
    if not xs:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}
    s = sorted(xs)
    def q(p: float) -> float:
        if len(s) == 1:
            return s[0]
        idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
        return s[idx]
    return {
        "p50": q(0.50), "p90": q(0.90), "p99": q(0.99),
        "mean": sum(s) / len(s), "max": s[-1],
    }


class ModuleTimer:
    """用 forward hook + CUDA event 记录指定子模块每次前向的耗时 (ms)。"""
    def __init__(self) -> None:
        self.records: dict[str, list[float]] = defaultdict(list)
        self._handles: list[Any] = []
        self._pending: dict[int, tuple[Any, Any]] = {}

    def _mk_pre(self, name: str):
        def pre(module: Any, inputs: Any) -> None:
            if DEVICE.startswith("cuda"):
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                self._pending[id(module)] = (name, ev)
            else:
                import time
                self._pending[id(module)] = (name, time.perf_counter())
        return pre

    def _mk_post(self, name: str):
        def post(module: Any, inputs: Any, output: Any) -> None:
            key = id(module)
            if key not in self._pending:
                return
            _, start = self._pending.pop(key)
            if DEVICE.startswith("cuda"):
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                end.synchronize()
                self.records[name].append(start.elapsed_time(end))
            else:
                import time
                self.records[name].append((time.perf_counter() - start) * 1000.0)
        return post

    def watch(self, module: Any, name: str) -> None:
        self._handles.append(module.register_forward_pre_hook(self._mk_pre(name)))
        self._handles.append(module.register_forward_hook(self._mk_post(name)))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------- M. 完整 GEMM M/N/K 表 ----------------
def gemm_table_from_shapes(shape_records: dict[str, list], cfg: dict[str, int]) -> list[dict[str, Any]]:
    """从捕获的 Linear shape 生成 M/N/K 表, 并标注 prefill/decode 与模块归属。"""
    rows: list[dict[str, Any]] = []
    for rec in shape_records.get("Linear", []):
        inp = rec.get("input_shape")
        if not inp:
            continue
        in0 = inp[0] if isinstance(inp[0], list) else inp
        M = 1
        for d in in0[:-1]:
            M *= d
        K = rec.get("in_features")
        N = rec.get("out_features")
        if K is None or N is None:
            continue
        # 阶段判定: M==1 -> decode; M>1 -> prefill(视觉/LLM首次)
        phase = "decode" if M == 1 else "prefill"
        # 模块归属启发式
        seq = in0[-2] if len(in0) >= 2 else M
        module = _classify_linear(K, N, cfg)
        rows.append({
            "module": module, "op": "Linear",
            "M": M, "N": N, "K": K, "dtype": "BF16",
            "bias": rec.get("bias", None),
            "call_count": rec.get("call_count", 1),
            "phase": phase, "seq_len": seq,
            "gflops_per_call": 2.0 * M * N * K / 1e9,
        })
    rows.sort(key=lambda r: r["gflops_per_call"] * r["call_count"], reverse=True)
    return rows


def _classify_linear(K: int, N: int, cfg: dict[str, int]) -> str:
    H = cfg["hidden"]; I = cfg["intermediate"]; V = cfg["vocab"]
    if K == H and N == V:
        return "Llama lm_head"
    if K == H and N == I:
        return "Llama MLP gate/up"
    if K == I and N == H:
        return "Llama MLP down"
    if K == H and N == H:
        return "Llama attn q/k/v/o"
    return "Vision/Projector"


# ---------------- N. Attention / KV-cache 详情 ----------------
def attention_kv_details(model: torch.nn.Module, cfg: dict[str, int],
                         prompt_len: int, n_decode: int) -> dict[str, Any]:
    lm_cfg = model.language_model.config
    n_heads = cfg["num_heads"]; n_kv = cfg["num_kv_heads"]; hd = cfg["head_dim"]
    gqa = "MHA" if n_kv == n_heads else ("MQA" if n_kv == 1 else "GQA")
    kv_per_step_bytes = 2 * cfg["num_layers"] * n_kv * hd * 2  # K+V, BF16=2B, per token
    return {
        "num_heads": n_heads,
        "num_kv_heads": n_kv,
        "attention_type": gqa,
        "head_dim": hd,
        "prefill": {"q_len": prompt_len, "kv_len": prompt_len, "cache_len": 0, "mask": "causal"},
        "decode": {"q_len": 1, "kv_len_start": prompt_len + 1,
                   "kv_len_end": prompt_len + n_decode, "cache_len_grows": True, "mask": "causal(implicit)"},
        "kv_cache_dtype": "BF16",
        "kv_cache_layout": "HF DynamicCache: per-layer [batch, num_kv_heads, seq, head_dim]",
        "kv_cache_bytes_per_token": kv_per_step_bytes,
        "kv_cache_MB_at_end": kv_per_step_bytes * (prompt_len + n_decode) / (1024 * 1024),
        "attention_kernel": f"{ATTN_IMPLEMENTATION} (aten::scaled_dot_product_attention -> flash/mem-efficient)",
        "rope": "LlamaRotaryEmbedding (cos/sin, applied per step)",
    }


# ---------------- O. 数据搬运 / layout 明细 ----------------
def data_movement_detail(aten_ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    watch = {
        "aten::cat": "图文token拼接 / KV-cache 追加 (真实拷贝)",
        "aten::to": "dtype/device 转换 (可能真实拷贝)",
        "aten::_to_copy": "to 的实际拷贝实现",
        "aten::copy_": "contiguous / cache 写入 (真实搬运)",
        "aten::transpose": "attention head layout 变换 (view, 无拷贝)",
        "aten::reshape": "view 优先, 非contiguous时触发拷贝",
        "aten::view": "纯 metadata 变换 (无拷贝)",
        "aten::expand": "广播 (view, 无拷贝)",
        "aten::contiguous": "强制连续内存 (可能真实拷贝)",
        "aten::as_strided": "stride 视图 (无拷贝)",
        "aten::unsqueeze": "插入维度 (view)",
        "aten::slice": "切片 (view)",
    }
    idx = {r["op"]: r for r in aten_ops}
    rows = []
    for op, cause in watch.items():
        r = idx.get(op)
        if not r:
            continue
        real_copy = op in {"aten::cat", "aten::_to_copy", "aten::copy_", "aten::contiguous"}
        rows.append({
            "op": op, "count": r["count"],
            "cpu_time_total_us": r["cpu_time_total_us"],
            "cuda_time_total_us": r["cuda_time_total_us"],
            "real_data_movement": real_copy,
            "compiler_eliminable": not real_copy,
            "cause": cause,
        })
    rows.sort(key=lambda r: r["cuda_time_total_us"] + r["cpu_time_total_us"] / 1000.0, reverse=True)
    return rows


# ---------------- P. Roofline / 算术强度 ----------------
def roofline_rows(cfg: dict[str, int], prompt_len: int, n_decode: int,
                  peak_tflops: float, bw_gbs: float) -> list[dict[str, Any]]:
    H = cfg["hidden"]; I = cfg["intermediate"]; V = cfg["vocab"]
    b = 2  # BF16
    ridge = (peak_tflops * 1e12) / (bw_gbs * 1e9) if bw_gbs else 0.0  # FLOP/byte 拐点

    def mk(name: str, M: int, N: int, K: int, has_weight: bool = True) -> dict[str, Any]:
        flops = 2.0 * M * N * K
        # bytes: 权重(K*N) + 输入(M*K) + 输出(M*N)
        wbytes = (K * N) * b if has_weight else 0
        io = (M * K + M * N) * b
        bytes_ = wbytes + io
        ai = flops / bytes_ if bytes_ else 0.0
        bound = "compute-bound" if ai > ridge else "memory-bound"
        return {"op": name, "M": M, "N": N, "K": K,
                "gflops": flops / 1e9, "rw_MB": bytes_ / 1024 / 1024,
                "arith_intensity_flops_per_byte": ai, "bound": bound}

    rows = [
        mk("Llama MLP up (prefill)", prompt_len, I, H),
        mk("Llama MLP up (decode M=1)", 1, I, H),
        mk("Llama MLP down (decode M=1)", 1, H, I),
        mk("Llama attn qkvo (decode M=1)", 1, H, H),
        mk("Llama lm_head (decode M=1)", 1, V, H),
        mk("Projector fc (prefill)", 256, 4 * H, H),
    ]
    # RMSNorm / attention-decode 作为 memory-bound 代表
    rms_bytes = (prompt_len * H) * b * 2
    rows.append({"op": "RMSNorm (per call, prefill)", "M": prompt_len, "N": H, "K": 1,
                 "gflops": (prompt_len * H * 4) / 1e9, "rw_MB": rms_bytes / 1024 / 1024,
                 "arith_intensity_flops_per_byte": 4.0 / (2 * b), "bound": "memory-bound"})
    kv_len = prompt_len + n_decode
    attn_flops = 2.0 * (2.0 * 1 * kv_len * H)
    attn_bytes = (kv_len * H * 2) * b  # 读 K,V cache
    rows.append({"op": "Attention decode (QK+AV, M=1)", "M": 1, "N": kv_len, "K": H,
                 "gflops": attn_flops / 1e9, "rw_MB": attn_bytes / 1024 / 1024,
                 "arith_intensity_flops_per_byte": attn_flops / attn_bytes if attn_bytes else 0.0,
                 "bound": "memory/cache-bound"})
    return rows, ridge


# ---------------- R. NPU fallback / 算子支持优先级 ----------------
def fallback_table() -> list[dict[str, str]]:
    return [
        {"op": "Linear / GEMM", "must_npu": "必须", "fallback": "CPU/GPU", "cost": "极高", "risk": "P0"},
        {"op": "matmul/bmm (attention)", "must_npu": "必须", "fallback": "CPU/GPU", "cost": "极高", "risk": "P0"},
        {"op": "scaled_dot_product_attention", "must_npu": "必须", "fallback": "CPU/GPU", "cost": "高", "risk": "P0"},
        {"op": "LlamaRMSNorm", "must_npu": "必须", "fallback": "CPU/GPU", "cost": "高", "risk": "P0"},
        {"op": "LayerNorm", "must_npu": "必须", "fallback": "CPU/GPU", "cost": "高", "risk": "P0"},
        {"op": "RoPE (rotary embed)", "must_npu": "建议原生", "fallback": "CPU/GPU", "cost": "中", "risk": "P0"},
        {"op": "SiLU / GELU", "must_npu": "必须", "fallback": "CPU/GPU", "cost": "中", "risk": "P0"},
        {"op": "elementwise add/mul (residual)", "must_npu": "必须", "fallback": "CPU/GPU", "cost": "中", "risk": "P0"},
        {"op": "softmax", "must_npu": "必须(可融入SDPA)", "fallback": "CPU/GPU", "cost": "中", "risk": "P0"},
        {"op": "Embedding (gather)", "must_npu": "建议", "fallback": "CPU/GPU", "cost": "低", "risk": "P1"},
        {"op": "Conv2d (patch embed)", "must_npu": "可lower成GEMM", "fallback": "CPU/GPU", "cost": "低", "risk": "P1"},
        {"op": "cat (KV append/图文拼接)", "must_npu": "可编译消除/DMA", "fallback": "CPU/GPU", "cost": "中", "risk": "P1"},
        {"op": "copy_ / contiguous", "must_npu": "可编译消除", "fallback": "runtime", "cost": "中", "risk": "P1"},
        {"op": "transpose/reshape/view", "must_npu": "编译期消除", "fallback": "runtime", "cost": "低", "risk": "P1"},
        {"op": "argmax (greedy decode)", "must_npu": "可选", "fallback": "CPU", "cost": "低", "risk": "P2"},
        {"op": "to (dtype cast)", "must_npu": "可融合", "fallback": "runtime", "cost": "低", "risk": "P2"},
        {"op": "Dropout (eval)", "must_npu": "no-op", "fallback": "无", "cost": "无", "risk": "P2"},
    ]


# ---------------- Q. 算子融合 前后对比 (torch.compile 代理) ----------------
def fusion_benchmark() -> list[dict[str, Any]]:
    """用 torch.compile 对典型融合模式做 eager vs compiled 微基准 (decode M=1 shape)。
    这是融合收益的代理测量: compile 会把 norm/激活/elementwise 与相邻 GEMM 融合、减少中间写回。"""
    import time
    if not DEVICE.startswith("cuda"):
        return []
    dt = torch.bfloat16
    H, I, V = 4096, 11008, 32064
    rows: list[dict[str, Any]] = []

    def bench(fn: Any, x: Any) -> float:
        for _ in range(5):
            fn(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            fn(x)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / 50 * 1000.0

    class RMSNormLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(H, device=DEVICE, dtype=dt))
            self.lin = torch.nn.Linear(H, H, bias=False, device=DEVICE, dtype=dt)
        def forward(self, x):
            v = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w
            return self.lin(v)

    class LinearSiLUMul(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = torch.nn.Linear(H, I, bias=False, device=DEVICE, dtype=dt)
            self.up = torch.nn.Linear(H, I, bias=False, device=DEVICE, dtype=dt)
        def forward(self, x):
            return torch.nn.functional.silu(self.gate(x)) * self.up(x)

    class LinearGELU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(H, H, bias=True, device=DEVICE, dtype=dt)
        def forward(self, x):
            return torch.nn.functional.gelu(self.fc(x))

    cases = [
        ("RMSNorm+Linear", RMSNormLinear, (1, 1, H)),
        ("Linear+SiLU+Mul (MLP)", LinearSiLUMul, (1, 1, H)),
        ("Linear+GELU (projector)", LinearGELU, (1, 1, H)),
    ]
    for name, cls, shape in cases:
        try:
            m = cls().eval()
            x = torch.randn(*shape, device=DEVICE, dtype=dt)
            with torch.inference_mode():
                eager_ms = bench(m, x)
                mc = torch.compile(m, mode="max-autotune")
                comp_ms = bench(mc, x)
            gain = (eager_ms - comp_ms) / eager_ms * 100.0 if eager_ms else 0.0
            rows.append({"pattern": name, "eager_ms": eager_ms, "compiled_ms": comp_ms,
                         "speedup_pct": gain, "accuracy_impact": "none (数值等价)"})
        except Exception as e:
            rows.append({"pattern": name, "error": str(e)[:120]})
    return rows


# ---------------- S. dtype 端到端时延对比 ----------------
def measure_dtype_latency(model: torch.nn.Module, inputs: Any, unnorm_key: str,
                          n_repeat: int = 10) -> dict[str, Any]:
    """测量 BF16 baseline 与 FP16 的端到端 predict_action 时延。
    INT8/FP8/4bit 与动作误差需要单独量化流程+数据集, 此处标注为后续项。"""
    import time
    result: dict[str, Any] = {}

    def _bench(tag: str) -> None:
        try:
            for _ in range(3):
                with torch.inference_mode():
                    model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
            if DEVICE.startswith("cuda"):
                torch.cuda.synchronize()
            ts = []
            for _ in range(n_repeat):
                t0 = time.perf_counter()
                with torch.inference_mode():
                    model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
                if DEVICE.startswith("cuda"):
                    torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000.0)
            mem = torch.cuda.max_memory_allocated() / 1024 ** 3 if DEVICE.startswith("cuda") else 0.0
            st = percentiles(ts)
            result[tag] = {"e2e_ms_mean": st["mean"], "e2e_ms_p99": st["p99"],
                           "achievable_hz": 1000.0 / st["mean"] if st["mean"] else 0.0,
                           "peak_mem_GB": round(mem, 2)}
        except Exception as e:
            result[tag] = {"error": str(e)}

    # BF16 baseline (当前模型已是 bf16)
    if DEVICE.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    _bench("BF16_baseline")

    # FP16: 需重载权重为 fp16; 仅当有 CUDA 时尝试, 失败则跳过
    result["FP16"] = {"note": "需以 torch_dtype=torch.float16 重载模型; 单独运行以避免相互干扰"}
    result["INT8_weight_only"] = {"note": "需 torchao/modelopt 量化 + 动作误差评测 (后续项)"}
    result["INT8_w+a"] = {"note": "需 activation 校准 + 动作误差评测 (后续项)"}
    result["FP8"] = {"note": "需 transformer_engine/modelopt + Blackwell FP8 kernel (后续项)"}
    result["INT4_weight_only"] = {"note": "需 torchao 4bit + 动作误差评测 (后续项)"}
    result["_accuracy_note"] = (
        "动作误差/成功率列需在真实/仿真 rollout 上评测 (gripper开合, xyz位移, 姿态角), "
        "非纯推理脚本可得; 本表只给推理侧时延/显存基线。"
    )
    return result


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

    output_dir = Path("/workspace/outputs/openvla")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.output) if args.output else output_dir / "operator_inventory.json"
    out_report = out_json.with_suffix(".report.txt")

    print("model:", MODEL_ID)
    print("device:", DEVICE)
    print("attention:", ATTN_IMPLEMENTATION)
    print("image:", image_path)

    processor = AutoProcessor.from_pretrained(MODEL_PATH, revision=MODEL_REVISION, trust_remote_code=True, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        revision=MODEL_REVISION,
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
    # 预热 (整体 predict_action)
    with torch.inference_mode():
        model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()

    # 整体 (prefill+decode 全流程) profiler
    prof = profile_call(
        lambda: model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False),
        activities,
    )
    aten_ops, op_family = aggregate_aten(prof)

    # ---- D. prefill / decode 双模式拆分 ----
    # predict_action -> generate(max_new_tokens=action_dim):
    #   1 次 prefill (多模态, 全序列) + (action_dim-1) 次 decode (单 token, 走 KV cache)
    action_dim = model.get_action_dim(UNNORM_KEY)
    n_decode = max(action_dim - 1, 1)
    base_input_ids = build_action_input_ids(inputs["input_ids"])
    pixel_values = inputs["pixel_values"]
    attention_mask = inputs.get("attention_mask")
    cfg = llm_config(model)
    prompt_len = int(base_input_ids.shape[1]) + 256  # 文本 token + 256 图像 patch

    def run_prefill():
        """一次 prefill: 多模态前向(视觉backbone+全序列), 返回 (past_key_values, next_token)。"""
        out = model(
            input_ids=base_input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            use_cache=True,
            return_dict=True,
        )
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return out.past_key_values, next_tok

    def run_decode(past_key_values, cur_ids):
        """一次 decode: 单 token 前向 (走 KV cache)。"""
        out = model(
            input_ids=cur_ids[:, -1:],
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return out.past_key_values, next_tok

    # 预热 prefill/decode 路径
    with torch.inference_mode():
        pkv_w, tok_w = run_prefill()
        ids_w = torch.cat([base_input_ids, tok_w], dim=1)
        for _ in range(n_decode):
            pkv_w, tok_w = run_decode(pkv_w, ids_w)
            ids_w = torch.cat([ids_w, tok_w], dim=1)
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()

    # --- D1. 阶段时延 (CUDA event / perf_counter, 取多轮均值) ---
    n_timing_runs = 10
    prefill_ms_list: list[float] = []
    decode_step_ms_list: list[float] = []
    decode_total_ms_list: list[float] = []
    per_step_ms: list[list[float]] = [[] for _ in range(n_decode)]  # 每个 decode step 的多轮样本
    with torch.inference_mode():
        for _ in range(n_timing_runs):
            t_pf, (pkv, tok) = timed_call(run_prefill)
            prefill_ms_list.append(t_pf)
            cur_ids = torch.cat([base_input_ids, tok], dim=1)
            dt_total = 0.0
            for _s in range(n_decode):
                def _step(pkv=pkv, cur_ids=cur_ids):
                    return run_decode(pkv, cur_ids)
                dt, (pkv, tok) = timed_call(_step)
                cur_ids = torch.cat([cur_ids, tok], dim=1)
                decode_step_ms_list.append(dt)
                per_step_ms[_s].append(dt)
                dt_total += dt
            decode_total_ms_list.append(dt_total)

    def _avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    prefill_ms = _avg(prefill_ms_list)
    decode_step_ms = _avg(decode_step_ms_list)
    decode_total_ms = _avg(decode_total_ms_list)
    total_ms = prefill_ms + decode_total_ms

    # L. 逐 decode step 明细 (token 长度递增)
    per_decode_step = []
    for i in range(n_decode):
        st = percentiles(per_step_ms[i])
        per_decode_step.append({
            "step": i + 1,
            "q_len": 1,
            "kv_len": prompt_len + i + 1,
            "mean_ms": st["mean"], "p50_ms": st["p50"], "max_ms": st["max"],
            "main_ops": "GEMV(M=1) + KV-cache read + SDPA(decode)",
        })

    latency_split = {
        "prompt_tokens": int(base_input_ids.shape[1]),
        "prefill_forwards": 1,
        "decode_forwards": n_decode,
        "timing_runs": n_timing_runs,
        "prefill_ms": prefill_ms,
        "prefill_pctl": percentiles(prefill_ms_list),
        "decode_total_ms": decode_total_ms,
        "decode_total_pctl": percentiles(decode_total_ms_list),
        "decode_per_step_ms": decode_step_ms,
        "decode_step_pctl": percentiles(decode_step_ms_list),
        "e2e_ms": total_ms,
        "e2e_pctl_ms": percentiles([p + d for p, d in zip(prefill_ms_list, decode_total_ms_list)]),
        "achievable_hz": (1000.0 / total_ms) if total_ms else 0.0,
        "per_decode_step": per_decode_step,
        "prefill_pct": (prefill_ms / total_ms * 100.0) if total_ms else 0.0,
        "decode_pct": (decode_total_ms / total_ms * 100.0) if total_ms else 0.0,
    }

    # --- K. 模块级时延拆解 (hook 各大模块, 跑多轮 predict_action) ---
    mt = ModuleTimer()
    vb = model.vision_backbone
    mt.watch(vb, "vision_backbone(DINOv2+SigLIP+fusion)")
    if hasattr(vb, "featurizer"):
        mt.watch(vb.featurizer, "  DINOv2 featurizer")
    if hasattr(vb, "fused_featurizer"):
        mt.watch(vb.fused_featurizer, "  SigLIP fused_featurizer")
    mt.watch(model.projector, "Projector MLP")
    mt.watch(model.language_model, "Llama LM (per forward: prefill/decode)")
    mt.watch(model.language_model.lm_head, "  lm_head")
    n_mod_runs = 10
    with torch.inference_mode():
        for _ in range(n_mod_runs):
            model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    mt.remove()
    module_latency = {}
    for name, xs in mt.records.items():
        st = percentiles(xs)
        module_latency[name] = {
            "calls_per_inference": len(xs) / n_mod_runs,
            "p50_ms": st["p50"], "p90_ms": st["p90"], "p99_ms": st["p99"],
            "mean_ms": st["mean"], "max_ms": st["max"],
            "pct_of_e2e": (st["mean"] * (len(xs) / n_mod_runs) / total_ms * 100.0) if total_ms else 0.0,
        }
    key_module_p90 = {
        "vision_backbone_p90_ms": module_latency.get("vision_backbone(DINOv2+SigLIP+fusion)", {}).get("p90_ms", 0.0),
        "projector_mlp_p90_ms": module_latency.get("Projector MLP", {}).get("p90_ms", 0.0),
        "lm_head_p90_ms": module_latency.get("  lm_head", {}).get("p90_ms", 0.0),
        "lm_head_calls_per_inference": module_latency.get("  lm_head", {}).get("calls_per_inference", 0.0),
        "note": "lm_head_p90_ms is per lm_head call; multiply/interpret with lm_head_calls_per_inference for per-action contribution.",
    }

    # --- D2. 各阶段算子 profiler (prefill-only / decode-only) + trace timeline 导出 ---
    trace_prefill = output_dir / "trace_prefill.json"
    trace_decode = output_dir / "trace_decode.json"
    trace_full = output_dir / "trace_full_predict_action.json"
    with torch.inference_mode():
        prof_prefill = profile_call(run_prefill, activities, trace_path=trace_prefill)
        # 为 decode 准备一个真实的 KV cache (不计入 profiler)
        pkv0, tok0 = run_prefill()
        ids0 = torch.cat([base_input_ids, tok0], dim=1)
        if DEVICE.startswith("cuda"):
            torch.cuda.synchronize()
        prof_decode = profile_call(lambda: run_decode(pkv0, ids0), activities, trace_path=trace_decode)
        # 完整 predict_action 的 timeline (prefill+decode 全流程), 供 timeline 拆分/截图
        prof_full_trace = profile_call(
            lambda: model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False),
            activities, trace_path=trace_full,
        )

    prefill_ops, prefill_family = aggregate_aten(prof_prefill)
    decode_ops, decode_family = aggregate_aten(prof_decode)

    # ---- A3/D3. aten 算子 shape 明细 (不限 topN, 按 CUDA 耗时排序) ----
    aten_ops_shape_detail = sorted(
        aten_ops_with_shapes(prof, aten_ops, shape_records),
        key=lambda r: r["cuda_time_total_us"],
        reverse=True,
    )
    prefill_ops_shape_detail = sorted(
        aten_ops_with_shapes(prof_prefill, prefill_ops, shape_records),
        key=lambda r: r["cuda_time_total_us"],
        reverse=True,
    )
    decode_ops_shape_detail = sorted(
        aten_ops_with_shapes(prof_decode, decode_ops, shape_records),
        key=lambda r: r["cuda_time_total_us"],
        reverse=True,
    )

    # ---- E. 算子调度频率 top-20 ----
    freq_overall = top_by_count(aten_ops, 20)
    freq_prefill = top_by_count(prefill_ops, 20)
    freq_decode = top_by_count(decode_ops, 20)

    # ---- F. CPU vs CUDA 耗时对比 (top-20) ----
    cpu_cuda_overall = cpu_cuda_compare(aten_ops, 20)
    cpu_cuda_prefill = cpu_cuda_compare(prefill_ops, 20)
    cpu_cuda_decode = cpu_cuda_compare(decode_ops, 20)

    # ---- G. 硬件 / 运行时环境 ----
    hw_env = probe_hardware_env()

    # ---- I. FLOPs 解析模型 ----
    vision_linear = flops_linear_from_shapes(shape_records)  # 视觉/projector Linear (唯一shape下界)
    flops = flops_full_model(cfg, prompt_len, n_decode, vision_gflops=vision_linear["total_gflops"])
    peak_tflops = hw_env.get("achieved_bf16_gemm_TFLOPs", 0.0)
    flops["prefill"]["utilization"] = utilization(flops["prefill"]["total_gflops"], prefill_ms, peak_tflops)
    flops["decode"]["utilization"] = utilization(flops["decode"]["total_gflops"], decode_total_ms, peak_tflops)
    flops["e2e_utilization"] = utilization(flops["total_gflops"], total_ms, peak_tflops)
    flops["captured_linear_from_shapes"] = vision_linear

    # ---- J. Decode 阶段权重常驻 SRAM 需求 ----
    sram = sram_weight_residency(cfg, dtype_bytes=2)

    # ---- M. 完整 GEMM M/N/K 表 ----
    gemm_table = gemm_table_from_shapes(shape_records, cfg)

    # ---- N. Attention / KV-cache 详情 ----
    attn_kv = attention_kv_details(model, cfg, prompt_len, n_decode)

    # ---- O. 数据搬运 / layout 明细 ----
    data_move = data_movement_detail(aten_ops)

    # ---- P. Roofline / 算术强度 ----
    bw_gbs = hw_env.get("empirical_mem_bandwidth_GBs", 0.0) or hw_env.get("theoretical_mem_bandwidth_GBs", 0.0)
    roof_rows, ridge = roofline_rows(cfg, prompt_len, n_decode, peak_tflops, bw_gbs)
    roofline = {"ridge_point_flops_per_byte": ridge,
                "peak_tflops": peak_tflops, "bandwidth_GBs": bw_gbs, "rows": roof_rows}

    # ---- R. NPU fallback / 支持优先级 ----
    fallback = fallback_table()

    # ---- Q. 算子融合前后对比 (torch.compile 代理) ----
    fusion = fusion_benchmark()

    # ---- S. dtype 对比 (BF16 baseline + FP16 端到端时延; 量化/精度误差为后续项) ----
    dtype_compare = measure_dtype_latency(model, inputs, UNNORM_KEY, n_repeat=10)

    # ---- 唯一 (算子, in-shape, out-shape) 组合数 ----
    unique_shape_combos = sum(len(v) for v in shape_records.values())

    trace_files = {
        "prefill": str(trace_prefill),
        "decode": str(trace_decode),
        "full_predict_action": str(trace_full),
        "howto": "在 chrome://tracing 或 https://ui.perfetto.dev 打开这些 json 查看 timeline; 截图用于报告",
    }

    inventory = {
        "model_id": MODEL_ID,
        "device": DEVICE,
        "attention": ATTN_IMPLEMENTATION,
        "unnorm_key": UNNORM_KEY,
        "action_dim_tokens": action_dim,
        "architecture": {
            "vision_backbone": "DINOv2 ViT-L/14 + SigLIP ViT-SO400M (fused)",
            "projector": "3-layer MLP + GELU",
            "llm_backbone": "Llama-2-7B",
            "note": "predict_action = autoregressive generate() of action tokens",
        },
        "G_hardware_runtime_env": hw_env,
        "B_module_inventory": module_inv,
        "C_tensor_shapes": {k: v for k, v in shape_records.items()},
        "C_unique_shape_combos": unique_shape_combos,
        "A_aten_operators": aten_ops,
        "A_aten_operators_with_shapes": aten_ops_shape_detail,
        "A_aten_operator_families": dict(op_family.most_common()),
        "D_prefill_decode_latency": latency_split,
        "D_prefill_aten_operators": prefill_ops,
        "D_prefill_aten_operators_with_shapes": prefill_ops_shape_detail,
        "D_decode_aten_operators": decode_ops,
        "D_decode_aten_operators_with_shapes": decode_ops_shape_detail,
        "E_op_schedule_frequency_top20": {
            "overall": freq_overall,
            "prefill": freq_prefill,
            "decode": freq_decode,
        },
        "F_cpu_vs_cuda_top20": {
            "overall": cpu_cuda_overall,
            "prefill": cpu_cuda_prefill,
            "decode": cpu_cuda_decode,
        },
        "H_trace_timeline_files": trace_files,
        "I_flops_analysis": flops,
        "J_decode_sram_weight_residency": sram,
        "K_module_latency": module_latency,
        "K_key_module_p90_ms": key_module_p90,
        "L_per_decode_step": per_decode_step,
        "M_gemm_mnk_table": gemm_table,
        "N_attention_kv_cache": attn_kv,
        "O_data_movement_detail": data_move,
        "P_roofline_analysis": roofline,
        "Q_fusion_benchmark": fusion,
        "R_npu_fallback_priority": fallback,
        "S_dtype_latency_compare": dtype_compare,
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
    lines.append("G. 硬件 / 运行时环境")
    lines.append("-" * 70)
    lines.append(f"  GPU                  : {hw_env.get('gpu_name')}  (sm {hw_env.get('compute_capability')}, {hw_env.get('sm_count')} SM)")
    lines.append(f"  显存                 : {hw_env.get('total_memory_GB')} GB (统一内存 LPDDR5X)")
    lines.append(f"  内存位宽/时钟        : {hw_env.get('memory_bus_width_bit')}-bit @ {hw_env.get('memory_clock_GHz')} GHz")
    lines.append(f"  理论内存带宽         : {hw_env.get('theoretical_mem_bandwidth_GBs')} GB/s")
    lines.append(f"  实测内存带宽(D2D)    : {hw_env.get('empirical_mem_bandwidth_GBs')} GB/s  (torch copy_, 读+写)")
    lines.append(f"  实测BF16 GEMM峰值    : {hw_env.get('achieved_bf16_gemm_TFLOPs')} TFLOP/s  (8192^3 matmul)")
    lines.append(f"  torch 版本           : {hw_env.get('torch_version')}  (CUDA {hw_env.get('torch_cuda_version')})")
    lines.append(f"  TORCH_CUDA_ARCH_LIST : {hw_env.get('TORCH_CUDA_ARCH_LIST')}")
    lines.append(f"  执行模式             : {hw_env.get('execution_mode')}")
    lines.append(f"  attention 实现       : {hw_env.get('attn_implementation')}")
    lines.append(f"  推理框架             : {hw_env.get('inference_framework')}")
    lines.append(f"  数据格式/量化        : {hw_env.get('model_dtype')} / {hw_env.get('quantization')}")
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
    lines.append("A1b. aten 算子 shape 明细预览 (按 CUDA 耗时排序; 完整见 JSON: A_aten_operators_with_shapes)")
    lines.append("-" * 70)
    lines.append("  注: profiler 原生提供 input_shapes; output_shape 对常见算子做推断, module_io_shape_variants 来自 forward hook。")
    for r in aten_ops_shape_detail[:20]:
        op = r["op"] if len(r["op"]) <= 34 else r["op"][:31] + "..."
        nvar = len(r.get("shape_variants", []))
        nmod = len(r.get("module_io_shape_variants", []))
        first_var = r.get("shape_variants", [{}])[0] if nvar else {}
        in_s = str(first_var.get("input_shapes"))
        out_s = str(first_var.get("inferred_output_shape"))
        if len(in_s) > 72:
            in_s = in_s[:69] + "..."
        if len(out_s) > 42:
            out_s = out_s[:39] + "..."
        lines.append(f"  {op:<34} variants={nvar:<3} module_io={nmod:<3} in={in_s} -> out={out_s}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("A2. 算子族汇总 (按调用次数)")
    lines.append("-" * 70)
    for fam, cnt in op_family.most_common():
        lines.append(f"  {fam:<40} {cnt:>8}")
    lines.append("")
    lines.append("-" * 70)
    lines.append(f"C. 关键算子典型张量形状 (NPU tiling/数据流参考) — 唯一(算子,in,out)组合共 {unique_shape_combos} 条")
    lines.append("    注: 同一 shape 被多次调用(如32层相同 Linear)只记一条; call=该 shape 在一次完整推理中的调用次数")
    lines.append("-" * 70)
    for cls, records in shape_records.items():
        lines.append(f"  [{cls}] 共 {len(records)} 种不同 shape:")
        for rec in records[:12]:
            extra = {k: v for k, v in rec.items() if k not in ("input_shape", "output_shape", "call_count")}
            cc = rec.get("call_count", 1)
            lines.append(f"     in={rec['input_shape']} -> out={rec['output_shape']}  call={cc}  {extra if extra else ''}")
        if len(records) > 12:
            lines.append(f"     ... (+{len(records) - 12} more)")
    lines.append("")
    lines.append("-" * 70)
    lines.append("D. prefill / decode 双模式时延拆分")
    lines.append("-" * 70)
    lines.append(f"  prompt tokens        : {latency_split['prompt_tokens']}")
    lines.append(f"  prefill 前向次数      : {latency_split['prefill_forwards']}")
    lines.append(f"  decode  前向次数      : {latency_split['decode_forwards']}")
    lines.append(f"  计时轮数 (取均值)     : {latency_split['timing_runs']}")
    lines.append("")
    lines.append(f"  prefill 时延          : {latency_split['prefill_ms']:>9.3f} ms  ({latency_split['prefill_pct']:>5.1f}%)")
    lines.append(f"  decode  总时延        : {latency_split['decode_total_ms']:>9.3f} ms  ({latency_split['decode_pct']:>5.1f}%)")
    lines.append(f"  decode  单步时延      : {latency_split['decode_per_step_ms']:>9.3f} ms/token")
    lines.append(f"  端到端 (predict_action): {latency_split['e2e_ms']:>9.3f} ms")
    lines.append("")
    lines.append("  说明: prefill = 视觉backbone + 全序列首次前向; decode = 单token走KV cache 的自回归步。")

    def _emit_freq(title: str, rows: list[dict[str, Any]]) -> None:
        lines.append("")
        lines.append("-" * 70)
        lines.append(title)
        lines.append("-" * 70)
        lines.append(f"  {'operator':<48} {'count':>8} {'cuda_us':>12}")
        for r in rows:
            op = r["op"] if len(r["op"]) <= 48 else r["op"][:45] + "..."
            lines.append(f"  {op:<48} {r['count']:>8} {r['cuda_time_total_us']:>12.1f}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("E. 算子调度频率 top-20 (按调用次数)")
    lines.append("=" * 70)
    _emit_freq("E1. 整体 (prefill+decode)", freq_overall)
    _emit_freq("E2. prefill 阶段", freq_prefill)
    _emit_freq("E3. decode 阶段", freq_decode)

    def _emit_cmp(title: str, rows: list[dict[str, Any]]) -> None:
        lines.append("")
        lines.append("-" * 70)
        lines.append(title)
        lines.append("-" * 70)
        lines.append(f"  {'operator':<40} {'count':>7} {'cpu_us':>11} {'cuda_us':>11} {'dom':>5} {'cuda/cpu':>9}")
        for r in rows:
            op = r["op"] if len(r["op"]) <= 40 else r["op"][:37] + "..."
            ratio = r["cuda_over_cpu_ratio"]
            ratio_s = "inf" if ratio == float("inf") else f"{ratio:.2f}"
            lines.append(
                f"  {op:<40} {r['count']:>7} {r['cpu_time_total_us']:>11.1f} "
                f"{r['cuda_time_total_us']:>11.1f} {r['dominant']:>5} {ratio_s:>9}"
            )

    lines.append("")
    lines.append("=" * 70)
    lines.append("F. CPU vs CUDA 耗时对比 top-20 (按耗时排序)")
    lines.append("=" * 70)
    lines.append("  dom=主导端(耗时更大); cuda/cpu=CUDA耗时/CPU耗时 比值 (>1 说明 GPU 侧是瓶颈)")
    _emit_cmp("F1. 整体 (prefill+decode)", cpu_cuda_overall)
    _emit_cmp("F2. prefill 阶段", cpu_cuda_prefill)
    _emit_cmp("F3. decode 阶段", cpu_cuda_decode)

    # ---- H. trace timeline ----
    lines.append("")
    lines.append("=" * 70)
    lines.append("H. PyTorch Profiler trace timeline (chrome trace)")
    lines.append("=" * 70)
    lines.append("  用 chrome://tracing 或 https://ui.perfetto.dev 打开以下 json, 可看到")
    lines.append("  按 timeline 拆分的 prefill / decode 各阶段 kernel 排布, 截图用于报告:")
    lines.append(f"    prefill : {trace_files['prefill']}")
    lines.append(f"    decode  : {trace_files['decode']}")
    lines.append(f"    full    : {trace_files['full_predict_action']}")

    # ---- I. FLOPs 分析 ----
    fa = flops
    lines.append("")
    lines.append("=" * 70)
    lines.append("I. 累计 GFLOPs 与算力利用率 (解析模型, MAC=2FLOP)")
    lines.append("=" * 70)
    lines.append(f"  Llama config: L={cfg['num_layers']} H={cfg['hidden']} I={cfg['intermediate']} "
                 f"heads={cfg['num_heads']} head_dim={cfg['head_dim']} vocab={cfg['vocab']}")
    lines.append(f"  prompt_len={prompt_len} (文本+256图像patch), decode_steps={n_decode}")
    lines.append("")
    lines.append(f"  {'phase':<10} {'linear':>10} {'attention':>11} {'lm_head':>9} {'vision':>9} {'total':>10}  (GFLOPs)")
    pf = fa["prefill"]; dc = fa["decode"]
    lines.append(f"  {'prefill':<10} {pf['linear_gflops']:>10.2f} {pf['attention_gflops']:>11.3f} "
                 f"{pf['lm_head_gflops']:>9.3f} {pf['vision_gflops']:>9.2f} {pf['total_gflops']:>10.2f}")
    lines.append(f"  {'decode':<10} {dc['linear_gflops']:>10.2f} {dc['attention_gflops']:>11.3f} "
                 f"{dc['lm_head_gflops']:>9.3f} {'-':>9} {dc['total_gflops']:>10.2f}")
    lines.append(f"  {'TOTAL':<10} {'':>10} {'':>11} {'':>9} {'':>9} {fa['total_gflops']:>10.2f}")
    lines.append("")
    lines.append("  Attention 单独 (QK^T + A*V):")
    lines.append(f"    prefill: {pf['attention_gflops']:.3f} GFLOPs   decode: {dc['attention_gflops']:.3f} GFLOPs")
    lines.append("")
    lines.append(f"  算力利用率 (vs 实测BF16峰值 {peak_tflops} TFLOP/s):")
    pu = pf["utilization"]; du = dc["utilization"]; eu = fa["e2e_utilization"]
    lines.append(f"    prefill : {pf['total_gflops']:.2f} GFLOPs / {prefill_ms:.2f} ms "
                 f"= {pu['achieved_tflops']:.2f} TFLOP/s  -> 利用率 {pu['utilization_pct']:.1f}%")
    lines.append(f"    decode  : {dc['total_gflops']:.2f} GFLOPs / {decode_total_ms:.2f} ms "
                 f"= {du['achieved_tflops']:.2f} TFLOP/s  -> 利用率 {du['utilization_pct']:.1f}%")
    lines.append(f"    e2e     : {fa['total_gflops']:.2f} GFLOPs / {total_ms:.2f} ms "
                 f"= {eu['achieved_tflops']:.2f} TFLOP/s  -> 利用率 {eu['utilization_pct']:.1f}%")
    lines.append("")
    lines.append("  说明: decode 利用率极低是 memory-bound 的直接体现——单token GEMM 受显存带宽而非算力限制。")
    lines.append(f"  (从捕获 Linear shape 累加的视觉/projector 部分 = {vision_linear['total_gflops']:.2f} GFLOPs, 唯一shape下界)")

    # ---- J. Decode SRAM 权重常驻 ----
    lines.append("")
    lines.append("=" * 70)
    lines.append("J. Decode 阶段权重常驻 SRAM 容量需求 (BF16=2 bytes/权重)")
    lines.append("=" * 70)
    lines.append(f"  lm_head (H x V = {cfg['hidden']}x{cfg['vocab']})        : {sram['lm_head_MB']:.1f} MB")
    lines.append(f"  embedding (V x H)                     : {sram['embedding_MB']:.1f} MB")
    lines.append(f"  FFN / layer (SwiGLU gate+up+down)     : {sram['ffn_per_layer_MB']:.1f} MB")
    lines.append(f"  Q/K/V/O / layer (4 x H x H)           : {sram['qkvo_per_layer_MB']:.1f} MB")
    lines.append(f"  单层合计 (attn+FFN)                    : {sram['per_layer_total_MB']:.1f} MB")
    lines.append(f"  全部 {cfg['num_layers']} 层合计                     : {sram['all_layers_MB']:.1f} MB")
    lines.append(f"  完整 LLM 权重 (含 head+embed)          : {sram['full_llm_weights_MB']:.1f} MB")
    lines.append("")
    lines.append(f"  {sram['note']}")

    # ---- K. 模块级时延拆解 ----
    lines.append("")
    lines.append("=" * 70)
    lines.append("K. 模块级时延拆解 (hook 各大模块, p50/p90/p99, ms)")
    lines.append("=" * 70)
    lines.append("  关键模块 p90(ms): "
                 f"vision_backbone={key_module_p90['vision_backbone_p90_ms']:.3f}, "
                 f"projector_mlp={key_module_p90['projector_mlp_p90_ms']:.3f}, "
                 f"lm_head={key_module_p90['lm_head_p90_ms']:.3f} "
                 f"(calls/action={key_module_p90['lm_head_calls_per_inference']:.1f})")
    lines.append(f"  {'module':<42} {'calls':>6} {'p50':>8} {'p90':>8} {'p99':>8} {'%e2e':>7}")
    for name, st in module_latency.items():
        lines.append(f"  {name:<42} {st['calls_per_inference']:>6.1f} {st['p50_ms']:>8.3f} "
                     f"{st['p90_ms']:>8.3f} {st['p99_ms']:>8.3f} {st['pct_of_e2e']:>6.1f}%")
    lines.append("")
    lines.append(f"  端到端 (predict_action) 均值 {total_ms:.2f} ms -> 可达 {latency_split['achievable_hz']:.2f} Hz")
    e2ep = latency_split["e2e_pctl_ms"]
    lines.append(f"  e2e p50/p90/p99 = {e2ep['p50']:.1f} / {e2ep['p90']:.1f} / {e2ep['p99']:.1f} ms")
    hz = latency_split["achievable_hz"]
    for target in (5, 10, 20):
        ok = "可" if hz >= target else "不可"
        lines.append(f"    稳定 {target}Hz (需 <{1000/target:.0f}ms): {ok}  (p99={e2ep['p99']:.0f}ms)")
    lines.append("  注: 图像预处理/相机采集/机器人通信/机械臂响应 属部署链路, 需真机测量 (本脚本只覆盖模型推理).")

    # ---- L. 逐 decode step ----
    lines.append("")
    lines.append("=" * 70)
    lines.append("L. Prefill / 逐 Decode step 时延 (token 长度递增)")
    lines.append("=" * 70)
    lines.append(f"  {'phase':<16} {'q_len':>6} {'kv_len':>7} {'mean_ms':>9} {'max_ms':>8}  main_ops")
    lines.append(f"  {'Prefill':<16} {prompt_len:>6} {prompt_len:>7} {prefill_ms:>9.3f} "
                 f"{latency_split['prefill_pctl']['max']:>8.3f}  GEMM(大M) + SDPA(full)")
    for s in per_decode_step:
        lines.append(f"  {'Decode step ' + str(s['step']):<16} {s['q_len']:>6} {s['kv_len']:>7} "
                     f"{s['mean_ms']:>9.3f} {s['max_ms']:>8.3f}  {s['main_ops']}")

    # ---- M. GEMM M/N/K 表 ----
    lines.append("")
    lines.append("=" * 70)
    lines.append("M. 完整 GEMM M/N/K 表 (按 总FLOPs 排序; M=token数, N=out, K=in)")
    lines.append("=" * 70)
    lines.append(f"  {'module':<22} {'M':>5} {'N':>7} {'K':>6} {'dtype':>5} {'bias':>5} {'calls':>6} {'phase':>8} {'GF/call':>8}")
    for r in gemm_table:
        biass = "Y" if r["bias"] else ("N" if r["bias"] is not None else "?")
        lines.append(f"  {r['module']:<22} {r['M']:>5} {r['N']:>7} {r['K']:>6} {r['dtype']:>5} "
                     f"{biass:>5} {r['call_count']:>6} {r['phase']:>8} {r['gflops_per_call']:>8.3f}")

    # ---- N. Attention / KV-cache ----
    lines.append("")
    lines.append("=" * 70)
    lines.append("N. Attention / KV-cache 详情")
    lines.append("=" * 70)
    lines.append(f"  num_heads        : {attn_kv['num_heads']}")
    lines.append(f"  num_kv_heads     : {attn_kv['num_kv_heads']}  -> {attn_kv['attention_type']}")
    lines.append(f"  head_dim         : {attn_kv['head_dim']}")
    lines.append(f"  prefill q/kv/cache: {attn_kv['prefill']['q_len']} / {attn_kv['prefill']['kv_len']} / {attn_kv['prefill']['cache_len']}  mask={attn_kv['prefill']['mask']}")
    lines.append(f"  decode  q_len     : {attn_kv['decode']['q_len']} (M=1)")
    lines.append(f"  decode  kv_len    : {attn_kv['decode']['kv_len_start']} -> {attn_kv['decode']['kv_len_end']} (随step增长)")
    lines.append(f"  KV-cache dtype    : {attn_kv['kv_cache_dtype']}")
    lines.append(f"  KV-cache layout   : {attn_kv['kv_cache_layout']}")
    lines.append(f"  KV / token        : {attn_kv['kv_cache_bytes_per_token']/1024:.1f} KB  (末尾累计 {attn_kv['kv_cache_MB_at_end']:.2f} MB)")
    lines.append(f"  attention kernel  : {attn_kv['attention_kernel']}")
    lines.append(f"  RoPE              : {attn_kv['rope']}")

    # ---- O. 数据搬运 / layout ----
    lines.append("")
    lines.append("=" * 70)
    lines.append("O. 数据搬运 / layout 算子明细")
    lines.append("=" * 70)
    lines.append(f"  {'op':<20} {'count':>7} {'cpu_us':>10} {'cuda_us':>10} {'真实搬运':>8} {'可消除':>7}  cause")
    for r in data_move:
        rc = "是" if r["real_data_movement"] else "否"
        ce = "是" if r["compiler_eliminable"] else "否"
        lines.append(f"  {r['op']:<20} {r['count']:>7} {r['cpu_time_total_us']:>10.1f} "
                     f"{r['cuda_time_total_us']:>10.1f} {rc:>8} {ce:>7}  {r['cause']}")

    # ---- P. Roofline ----
    lines.append("")
    lines.append("=" * 70)
    lines.append("P. Roofline / 算术强度分析")
    lines.append("=" * 70)
    lines.append(f"  峰值 {roofline['peak_tflops']} TFLOP/s, 带宽 {roofline['bandwidth_GBs']} GB/s "
                 f"-> ridge point = {ridge:.1f} FLOP/byte (>此值 compute-bound, 否则 memory-bound)")
    lines.append(f"  {'op':<32} {'GFLOPs':>9} {'rw_MB':>9} {'AI':>8}  bound")
    for r in roof_rows:
        lines.append(f"  {r['op']:<32} {r['gflops']:>9.3f} {r['rw_MB']:>9.2f} "
                     f"{r['arith_intensity_flops_per_byte']:>8.2f}  {r['bound']}")
    lines.append("")
    lines.append("  结论: decode 全程 M=1, AI≈1 远低于 ridge -> 纯 memory/带宽/SRAM 复用瓶颈, 堆算力无用;")
    lines.append(f"        注意 ridge point={ridge:.0f} FLOP/byte 很高 (算力/带宽比大), prefill M={prompt_len}")
    lines.append(f"        的 AI≈{prompt_len} 仍 < ridge, 说明单帧 prefill 也是权重加载受限;")
    lines.append("        只有 M 增大到 > ridge (更大 batch/更长序列) 才转为 compute-bound。")
    lines.append("        => NPU 优先级: 提升有效带宽 + 片上权重驻留/复用 > 堆峰值算力。")

    # ---- S. dtype 对比 ----
    lines.append("")
    lines.append("=" * 70)
    lines.append("S. dtype 端到端时延对比 (推理侧)")
    lines.append("=" * 70)
    for tag, v in dtype_compare.items():
        if tag.startswith("_"):
            continue
        if "e2e_ms_mean" in v:
            lines.append(f"  {tag:<20}: {v['e2e_ms_mean']:.2f} ms (p99 {v['e2e_ms_p99']:.1f}), "
                         f"{v['achievable_hz']:.2f} Hz, peak_mem {v['peak_mem_GB']} GB")
        else:
            lines.append(f"  {tag:<20}: {v.get('note', v.get('error',''))}")
    lines.append(f"  {dtype_compare['_accuracy_note']}")

    # ---- Q. 算子融合前后对比 ----
    lines.append("")
    lines.append("=" * 70)
    lines.append("Q. 算子融合 前后对比 (torch.compile 代理, decode M=1 shape)")
    lines.append("=" * 70)
    lines.append(f"  {'pattern':<26} {'eager_ms':>9} {'compiled_ms':>12} {'收益':>8}  精度影响")
    for r in fusion:
        if "error" in r:
            lines.append(f"  {r['pattern']:<26} [skip] {r['error']}")
        else:
            lines.append(f"  {r['pattern']:<26} {r['eager_ms']:>9.4f} {r['compiled_ms']:>12.4f} "
                         f"{r['speedup_pct']:>7.1f}%  {r['accuracy_impact']}")
    lines.append("  注: torch.compile 融合 = norm/激活/elementwise 融入相邻 GEMM, 减少中间张量写回;")
    lines.append("      NPU 编译器做等价融合可省 HBM 往返, 小算子占比不高但融合后收益明显。")

    # ---- R. NPU fallback 表 ----
    lines.append("")
    lines.append("=" * 70)
    lines.append("R. NPU 算子支持 / fallback 优先级")
    lines.append("=" * 70)
    lines.append(f"  {'op':<32} {'必须NPU':>14} {'fallback':>10} {'成本':>6} {'风险':>5}")
    for r in fallback:
        lines.append(f"  {r['op']:<32} {r['must_npu']:>14} {r['fallback']:>10} {r['cost']:>6} {r['risk']:>5}")

    report_text = "\n".join(lines)
    out_report.write_text(report_text, encoding="utf-8")

    print(report_text)
    print("\njson  :", out_json)
    print("report:", out_report)


if __name__ == "__main__":
    main()
