#!/usr/bin/env python3
# =============================================================================
# 10_operator_categories.py - FP8 TRT 算子类别聚合 (供 FP8 部署文档 §5 使用)
# =============================================================================
#
# 目标:
#   把 09_prof_trace_e2e.py 已经产出的 trace 产物,聚合成"算子类别 → CUDA 工时占比",
#   对齐 BF16 baseline 文档 §5.3 的口径 (GEMM / Memory / Attention / Elementwise /
#   Norm / Q-DQ / Other)。不重新跑引擎,只读取已有 artifacts。
#
# 关键限制 (必须如实告知读者):
#   1. Vision engine 的 CUDA kernel 对 torch.profiler 可见,能按 kernel 名归类。
#      但 TensorRT Myelin 会把多个算子融进一个 kernel 名 (例如
#      __myl_CastMulAdd...MeanSubMul...Sqrt...Div... = 融合的 RMSNorm+残差),
#      因此这里用"主导算子优先级"归类 (见 CATEGORY_RULES 的顺序):一个带 Fc/gemm/
#      tensorop 标记的融合 kernel 本质是 GEMM,就归 GEMM。这是启发式,不是精确切分。
#   2. LLM (Edge-LLM) engine 被 Myelin 整体融合,层级 CSV 的 onnx_op/category 全空
#      → 无法做 mha/gemm 细分 (全 kgen_other)。占端到端 ~94% 的 decode 因此拿不到
#      类别级拆分,只能给出融合层 Top-N。这是当前工具链的硬限制。
#   3. Q/DQ: FP8 的量化/反量化在 TRT 里被折叠进 Myelin 融合块 (以 Cast 形式),
#      没有独立的 Q/DQ kernel 计数。这里用 vision engine inspector 的
#      FP8↔Half datatype 边界作为"存在性"证据,而不是伪造一个节点数。
#
# 用法:
#   python 10_operator_categories.py --tag 2026_0727_001508
#   python 10_operator_categories.py --trace <vision_trace.json> --csv <layer.csv>
# =============================================================================

from __future__ import annotations

import argparse
import collections
import json
import os
import re
from pathlib import Path

# 同一个 Myelin 融合 GEMM (__myl_<op>_0x<hash>) 会派生数十个仅 tile/cga 配置不同的 kernel
# instance (后缀 _tensorop..._cga..._sm... 不同)。逐条列出是噪音,故按 __myl_<op>_0x<hash>
# 合并成一条 (cutlass/插件核等非 __myl 名不受影响,按全名保留,不同 tile size 仍算不同 kernel)。
_MYL_HASH = re.compile(r"(__myl_[A-Za-z0-9]+_0x[0-9a-f]+)")


def merge_key(name: str) -> str:
    m = _MYL_HASH.match(name)
    return m.group(1) if m else name


def merge_tile_variants(per_us: dict, per_cnt: dict) -> list[tuple[str, float, int, int]]:
    """把共享 __myl_<op>_0x<hash> 前缀的 tile 变体合并。返回 (显示名, us, count, 变体数)。"""
    m_us: dict[str, float] = collections.Counter()
    m_cnt: dict[str, int] = collections.Counter()
    m_var: dict[str, int] = collections.Counter()
    for name, us in per_us.items():
        k = merge_key(name)
        m_us[k] += us
        m_cnt[k] += per_cnt[name]
        m_var[k] += 1
    return [(k, m_us[k], m_cnt[k], m_var[k]) for k in m_us]

LOGS_DIR = Path(os.environ.get("OPENVLA_LOGS_DIR", "/workspace/outputs/openvla"))

# 类别归类规则: (类别名, [关键字...])。按此顺序匹配,命中即停 —— 顺序即优先级。
# 优先级理由: 融合 kernel 的成本由其"最重"的算子主导,GEMM > Attention > Norm > 激活 > 访存。
# 注意: KV/RoPE/Sample 放在 Norm 前 —— Edge-LLM 的 `initializeNormalRope...` 含 "norm"
#       子串,若不先匹配 rope 会被 Norm 误收;这些是 LLM runtime 插件核,非经典计算算子。
CATEGORY_RULES = [
    ("Attention",           ["mha", "fmha", "flash_attn", "attention"]),
    ("KV/RoPE/Sample",      ["rope", "writekv", "splitqkv", "seqlen", "kvend", "topk",
                             "sampling", "embeddinglookup", "incrementlength", "compactkv",
                             "compacttensor", "kvcache"]),
    ("GEMM/TensorCore",     ["gemm", "cutlass", "tensorop", "xmma", "matmul", "_fc_", "myl_fc", "fprop", "wgrad", "dgrad", "conv"]),
    ("Norm/Reduction",      ["mean", "sqrt", "rsqrt", "layernorm", "rmsnorm", "softmax", "reduce"]),
    ("Elementwise/Act",     ["erf", "silu", "gelu", "sigmoid", "tanh", "relu", "mul", "add", "sub", "div", "cast"]),
    ("Memory/Layout",       ["tran", "slic", "resh", "move", "conc", "memset", "memcpy", "copy", "permute", "concat"]),
    ("Q/DQ",                ["quantize", "dequant", "qdq"]),
]


def categorize(kernel_name: str) -> str:
    name = kernel_name.lower()
    for cat, keys in CATEGORY_RULES:
        if any(k in name for k in keys):
            return cat
    return "Other"


def categorize_vision_trace(trace_path: Path, topn: int = 10) -> dict:
    """从 chrome trace JSON 聚合 vision engine 的 CUDA kernel 类别占比。"""
    data = json.loads(trace_path.read_text())
    events = data.get("traceEvents", [])

    cat_us: dict[str, float] = collections.Counter()
    cat_cnt: dict[str, int] = collections.Counter()
    kernels: list[tuple[str, float, int]] = []  # (name, total_us, count) 先按 name 聚合
    per_kernel_us: dict[str, float] = collections.Counter()
    per_kernel_cnt: dict[str, int] = collections.Counter()

    for e in events:
        if e.get("cat") != "kernel":
            continue
        dur = float(e.get("dur", 0) or 0)  # 微秒
        if dur <= 0:
            continue
        name = e.get("name", "")
        cat = categorize(name)
        cat_us[cat] += dur
        cat_cnt[cat] += 1
        per_kernel_us[name] += dur
        per_kernel_cnt[name] += 1

    total_us = sum(cat_us.values())
    category_breakdown = {
        cat: {
            "cuda_us": round(us, 1),
            "pct": round(us / total_us * 100, 1) if total_us else 0.0,
            "kernel_launches": cat_cnt[cat],
        }
        for cat, us in sorted(cat_us.items(), key=lambda kv: kv[1], reverse=True)
    }

    merged = merge_tile_variants(per_kernel_us, per_kernel_cnt)
    merged.sort(key=lambda k: k[1], reverse=True)
    top_kernels = [
        {
            "kernel": k[0][:80],
            "category": categorize(k[0]),
            "cuda_us": round(k[1], 1),
            "pct": round(k[1] / total_us * 100, 1) if total_us else 0.0,
            "launches": k[2],
            "tile_variants": k[3],
        }
        for k in merged[:topn]
    ]

    return {
        "source_trace": str(trace_path),
        "total_cuda_us": round(total_us, 1),
        "distinct_kernels_raw": len(per_kernel_us),
        "distinct_kernels_merged": len(merged),
        "category_breakdown": category_breakdown,
        "top10_kernels": top_kernels,
    }


def summarize_llm_layers(csv_path: Path) -> dict:
    """LLM 层级 CSV 只能给出融合层 Top-N;类别细分因 Myelin 融合不可得,如实说明。"""
    import csv as _csv

    rows = []
    empty_onnx = 0
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        for r in reader:
            try:
                t = float(r.get("time_ms_mean", 0) or 0)
            except ValueError:
                t = 0.0
            if not (r.get("onnx_op") or "").strip():
                empty_onnx += 1
            rows.append((r.get("layer_name", "")[:70], t))
    rows.sort(key=lambda x: x[1], reverse=True)
    total = sum(t for _, t in rows)
    return {
        "source_csv": str(csv_path),
        "num_layers": len(rows),
        "categorizable": False,
        "reason": (
            "Edge-LLM engine 被 Myelin 整体融合,层级 CSV 的 onnx_op/category 字段全空 "
            f"({empty_onnx}/{len(rows)} 层无 onnx_op),无法做 mha/gemm/norm 细分。"
            "decode 占端到端 ~94%,这部分类别拆分是当前工具链的硬限制。"
        ),
        "total_layer_ms_per_decode_step_non_graph": round(total, 3),
        "top10_fused_layers": [{"layer": n, "time_ms": round(t, 4)} for n, t in rows[:10]],
    }


def categorize_llm_nsys(nsys_rep: Path, topn: int = 10) -> dict:
    """用 nsys 的 kernel 级 GPU 时间给 LLM engine 做真实算子类别聚合。

    与 vision 用 torch.profiler 抓 kernel 同理,但 LLM 是 Edge-LLM C++ runtime,
    torch.profiler 无法 attach,故改用 nsys。llm_bench 层级 CSV 的 category 字段全空
    (Myelin 融合丢了 onnx_op 来源),而 nsys 直接拿 CUDA kernel 名+GPU 时间,可归类。

    局限: 融合 kernel 内部仍不可再拆(一个 __myl_ 块混多算子,按主导算子归类)。
    覆盖: 该 .nsys-rep 是整个 llm_inference 进程(prefill+decode 全程,多 pass),
    是"整 LLM engine 的算子分布",非 decode-only。
    """
    import csv as _csv
    import io as _io
    import subprocess as _sp

    out = _sp.run(
        ["/usr/local/cuda/bin/nsys", "stats", "--report", "cuda_gpu_kern_sum",
         "--format", "csv", "--force-export=true", str(nsys_rep)],
        check=True, capture_output=True, text=True,
    ).stdout
    lines = out.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("Time (%)")), 0)
    rows = list(_csv.DictReader(_io.StringIO("\n".join(lines[start:]))))

    cat_ns: dict[str, float] = collections.Counter()
    cat_cnt: dict[str, int] = collections.Counter()
    per_ns: dict[str, float] = collections.Counter()
    per_cnt: dict[str, int] = collections.Counter()
    for r in rows:
        name = (r.get("Name") or "").strip()
        try:
            ns = float(r.get("Total Time (ns)", 0) or 0)
            inst = int(float(r.get("Instances", 0) or 0))
        except ValueError:
            continue
        if not name or ns <= 0:
            continue
        cat = categorize(name)
        cat_ns[cat] += ns
        cat_cnt[cat] += inst
        per_ns[name] += ns
        per_cnt[name] += inst

    total = sum(cat_ns.values())
    breakdown = {
        c: {"gpu_ms": round(ns / 1e6, 3),
            "pct": round(ns / total * 100, 1) if total else 0.0,
            "kernel_instances": cat_cnt[c]}
        for c, ns in sorted(cat_ns.items(), key=lambda kv: kv[1], reverse=True)
    }
    merged = merge_tile_variants(per_ns, per_cnt)  # 合并 __myl_<op>_0x<hash> 的 tile 变体
    merged.sort(key=lambda k: k[1], reverse=True)
    top10 = [{"kernel": k[0][:80], "category": categorize(k[0]),
              "gpu_ms": round(k[1] / 1e6, 3),
              "pct": round(k[1] / total * 100, 1) if total else 0.0,
              "instances": k[2],
              "tile_variants": k[3]}
             for k in merged[:topn]]
    return {
        "source_nsys_rep": str(nsys_rep),
        "coverage": "整个 llm_inference 进程 (prefill+decode 全程,多 pass);whole-LLM 分布,非 decode-only",
        "total_gpu_ms": round(total / 1e6, 3),
        "distinct_kernels_raw": len(per_ns),
        "distinct_kernels_merged": len(merged),
        "category_breakdown": breakdown,
        "top10_kernels": top10,
    }


def qdq_evidence_from_summary(summary: dict) -> dict:
    """Q/DQ 存在性证据: 从 precision_verification (inspector) 的 FP8↔Half 边界推断。"""
    pv = (summary.get("precision_verification") or {}).get("vision") or {}
    if not pv.get("detailed"):
        return {"available": False, "note": "无 detailed inspector 数据"}
    out_hist = pv.get("output_dtype_histogram", {})
    in_hist = pv.get("input_dtype_histogram", {})
    return {
        "available": True,
        "note": (
            "Q/DQ 在 TRT 里被折叠进 Myelin 融合块 (以 Cast 形式),无独立 kernel 计数。"
            "下列 FP8↔Half 的 datatype 边界即量化/反量化实际发生处的证据。"
        ),
        "vision_output_dtypes": out_hist,
        "vision_input_dtypes": in_hist,
        "fp8_output_pct": pv.get("quantized_output_pct"),
        "fallback_layer_count": pv.get("fallback_layer_count"),
    }


def find_latest(pattern: str) -> Path | None:
    cands = sorted(LOGS_DIR.glob(pattern))
    return cands[-1] if cands else None


def main() -> None:
    ap = argparse.ArgumentParser(description="FP8 TRT 算子类别聚合 (文档 §5)")
    ap.add_argument("--tag", default=None, help="e2e_prof_summary 的时间戳 tag,自动定位产物")
    ap.add_argument("--summary", default=None, help="e2e_prof_summary_*.json 路径 (覆盖 --tag)")
    ap.add_argument("--trace", default=None, help="vision chrome trace json (覆盖自动定位)")
    ap.add_argument("--csv", default=None, help="LLM 层级 csv (覆盖自动定位)")
    ap.add_argument("--llm-nsys", default=None,
                    help="LLM 的 .nsys-rep (kernel 级类别聚合);默认自动找 fp8_nsys_*_hi.nsys-rep")
    ap.add_argument("--output", default=None, help="输出 json 路径")
    ap.add_argument("--topn", type=int, default=10, help="每个 engine 导出的 kernel 排名条数 (默认 10)")
    args = ap.parse_args()

    # 定位 summary
    if args.summary:
        summary_path = Path(args.summary)
    elif args.tag:
        summary_path = find_latest(f"e2e_prof_summary_*{args.tag}*.json")
    else:
        summary_path = find_latest("e2e_prof_summary_vfp8_lfp8_*.json")
    summary = json.loads(summary_path.read_text()) if summary_path and summary_path.exists() else {}

    # 定位 trace / csv (优先命令行,其次 summary 里的 artifacts,最后 glob)
    trace_path = Path(args.trace) if args.trace else None
    if trace_path is None:
        t = (summary.get("vision", {}).get("artifacts", {}) or {}).get("chrome_trace")
        trace_path = Path(t) if t else (find_latest(f"vision_trace_fp8_*{args.tag or ''}*.trace.json"))

    csv_path = Path(args.csv) if args.csv else None
    if csv_path is None:
        c = (summary.get("llm", {}).get("layer_breakdown", {}) or {}).get("csv")
        csv_path = Path(c) if c else None

    # LLM nsys rep (kernel 级类别)
    llm_nsys = Path(args.llm_nsys) if args.llm_nsys else None
    if llm_nsys is None:
        cands = sorted(LOGS_DIR.glob("fp8_nsys_*_hi.nsys-rep")) or sorted(LOGS_DIR.glob("fp8_nsys_*.nsys-rep"))
        llm_nsys = cands[-1] if cands else None

    tag = args.tag or (summary.get("timestamp", "").replace(":", "").replace("-", "") or "manual")

    print("=" * 64)
    print("FP8 TRT 算子类别聚合 (文档 §5)")
    print(f"  summary : {summary_path}")
    print(f"  trace   : {trace_path}")
    print(f"  csv     : {csv_path}")
    print("=" * 64)

    result: dict = {"summary_source": str(summary_path), "tag": tag}

    # Vision 类别
    if trace_path and trace_path.exists():
        vis = categorize_vision_trace(trace_path, topn=args.topn)
        result["vision_operator_categories"] = vis
        print("\n--- Vision 算子类别 (self CUDA 工时占比) ---")
        for cat, v in vis["category_breakdown"].items():
            print(f"  {cat:18s} {v['pct']:5.1f}%  ({v['cuda_us']/1000:.2f} ms, {v['kernel_launches']} launches)")
        print("\n  Top-5 kernel:")
        for k in vis["top10_kernels"][:5]:
            print(f"    {k['pct']:5.1f}%  [{k['category']:16s}] {k['kernel'][:56]}")
    else:
        print("\n[!] 找不到 vision chrome trace,跳过 vision 类别")

    # LLM 算子类别 (nsys kernel 级) —— 真实类别表
    if llm_nsys and llm_nsys.exists():
        try:
            llm_cat = categorize_llm_nsys(llm_nsys, topn=args.topn)
            result["llm_operator_categories"] = llm_cat
            print(f"\n--- LLM 算子类别 (nsys kernel 级 GPU 工时) ---")
            print(f"  源: {llm_nsys.name}  |  {llm_cat['coverage']}")
            for cat, v in llm_cat["category_breakdown"].items():
                print(f"  {cat:18s} {v['pct']:5.1f}%  ({v['gpu_ms']:.2f} ms, {v['kernel_instances']} inst)")
            print("\n  Top-5 kernel:")
            for k in llm_cat["top10_kernels"][:5]:
                print(f"    {k['pct']:5.1f}%  [{k['category']:16s}] {k['kernel'][:52]}")
        except Exception as e:
            print(f"\n[!] LLM nsys 类别聚合失败: {e}")
    else:
        print("\n[!] 找不到 LLM .nsys-rep,跳过 kernel 级类别 (可先跑 13_nsys_cpu_overhead.py)")

    # LLM 融合层 (补充: 层级 CSV 的 Top 融合层,说明 Myelin 融合内部不可再拆)
    if csv_path and csv_path.exists():
        llm = summarize_llm_layers(csv_path)
        result["llm_layers"] = llm

    # Q/DQ 证据
    result["qdq_evidence"] = qdq_evidence_from_summary(summary)
    qe = result["qdq_evidence"]
    if qe.get("available"):
        print(f"\n--- Q/DQ 证据 (inspector 边界) ---")
        print(f"  FP8 输出占比: {qe['fp8_output_pct']}%  |  fallback 层: {qe['fallback_layer_count']}")
        print(f"  {qe['note']}")

    out_path = Path(args.output) if args.output else LOGS_DIR / f"fp8_operator_categories_{tag}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n输出: {out_path}")


if __name__ == "__main__":
    main()
