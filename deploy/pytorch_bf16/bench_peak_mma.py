"""Kernel-only Tensor Core peak TFLOPS (Thor / sm_110).

What this is for
----------------
`bench_peak.py` and `bench_peak_tensorcore.py` time a *Python API call*
(torch.mm / TE general_gemm). Profiler showed that one NVFP4
`general_gemm` is actually three GPU kernels:

    ~87%  MMA GEMM   nvjet_sm110_ootst_128x256_*_2cta_*
    ~12%  swizzle    layout transform, not peak math
    ~1%   NVFP4 scale

Those scripts therefore report ~470 TFLOPS. This script times **only
the MMA kernel**, which is the number you want when asking "what is
the silicon peak we can actually hit with dense Tensor Cores".

It still will not print 2070. Official 2070 is MAXN *sparse 2:4* FP4.
Dense math is:

    2070 sparse FP4 / 2 = 1035 dense FP4   (MAXN)
    1820 sparse FP4 / 2 =  910 dense FP4   (120W)
    /2 again            =  517 / 455 dense FP8
    /2 again            =  259 / 228 dense BF16   (~256)

This file measures the dense column. Getting 2070 needs a sparse 2:4
CUTLASS/TRT kernel, which is not exposed through TE's dense GEMM.

    python bench_peak_mma.py
    python bench_peak_mma.py --skip-bf16          # faster, FP8+NVFP4 only
    sudo nvpmodel -m 0 && sudo jetson_clocks     # MAXN + lock clocks
"""

from __future__ import annotations

import argparse
import os
import subprocess
import warnings
from typing import Callable

import torch
from torch.profiler import ProfilerActivity, profile

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def sync() -> None:
    torch.cuda.synchronize()


def tflops(m: int, n: int, k: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return (2.0 * m * n * k) / seconds / 1e12


def gpu_gpc_mhz() -> tuple[float, float]:
    base = "/sys/class/devfreq/gpu-gpc-0"
    try:
        cur = int(open(f"{base}/cur_freq").read()) / 1e6
        mx = int(open(f"{base}/max_freq").read()) / 1e6
        return cur, mx
    except OSError:
        return float("nan"), float("nan")


def nvpmodel_name() -> str:
    try:
        out = subprocess.check_output(["nvpmodel", "-q"], text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if "NV Power Mode" in ln and i + 1 < len(lines):
            return lines[i + 1]
        if ln[:1].isdigit() or "W" in ln or "MAXN" in ln.upper():
            return ln
    return lines[-1] if lines else "unknown"


# 256-aligned, K large enough to be compute-bound. Includes the shape
# that already showed the best NVFP4 kernel (2048,2048,8192).
SHAPES = (
    (2048, 2048, 8192),
    (2048, 2048, 12288),
    (2048, 4096, 4096),
    (2560, 2560, 8192),
    (3072, 3072, 8192),
    (4096, 4096, 4096),
    (4096, 4096, 8192),
    (4096, 256, 8192),
    (1024, 8192, 8192),
    (2048, 1024, 8192),
    (3072, 2048, 8192),
    (8192, 8192, 4096),
    (8192, 8192, 8192),
)

# Kernel names that are the actual Tensor Core MMA. Everything else
# (swizzle / scale / copy) is helper work and must not enter the peak.
_MMA_HINTS = ("nvjet", "cutlass", "tensorop", "ootst", "wgmma", "tcgen", "nvfp4_gemm")
_MMA_EXCLUDE = (
    "swizzle",
    "memcpy",
    "elementwise",
    "vectorized",
    "unrolled",
    "scale",
    "quant",
    "memset",
    "copy_kernel",
    "compute_nvfp4",
)


def _is_mma_kernel(name: str) -> bool:
    low = name.lower()
    if any(x in low for x in _MMA_EXCLUDE):
        return False
    return any(x in low for x in _MMA_HINTS) or ("gemm" in low and "copy" not in low)


def time_api_ms(fn: Callable[[], None], iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    sync()
    return start.elapsed_time(end) / iters


def time_mma_kernel_s(
    fn: Callable[[], None], profile_iters: int, warmup: int
) -> tuple[float, str]:
    """Return (seconds per call of MMA kernels only, longest kernel name)."""
    for _ in range(warmup):
        fn()
    sync()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(profile_iters):
            fn()
        sync()

    total_us = 0.0
    best_name = ""
    best_us = -1.0
    for evt in prof.key_averages():
        name = evt.key
        us = float(evt.self_device_time_total)
        if us <= 0 or not _is_mma_kernel(name):
            continue
        total_us += us
        per = us / max(evt.count, 1)
        if per > best_us:
            best_us = per
            best_name = name
    if total_us <= 0:
        return 0.0, ""
    return (total_us / profile_iters) * 1e-6, best_name


def _print_header() -> None:
    print(
        f"{'M':>5} {'N':>5} {'K':>5} {'API ms':>8} {'API TF':>8} "
        f"{'MMA ms':>8} {'MMA TF':>8}  kernel"
    )


def _print_row(
    m: int, n: int, k: int, api_ms: float, mma_s: float, kname: str
) -> None:
    api_tf = tflops(m, n, k, api_ms * 1e-3)
    mma_tf = tflops(m, n, k, mma_s)
    short = kname.replace("void ", "")
    if len(short) > 42:
        short = short[:39] + "..."
    print(
        f"{m:5d} {n:5d} {k:5d} {api_ms:8.3f} {api_tf:8.1f} "
        f"{mma_s*1e3:8.3f} {mma_tf:8.1f}  {short}"
    )


def _make_nvfp4_step(m: int, n: int, k: int, device: str):
    from transformer_engine.pytorch.cpp_extensions import general_gemm
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
    import transformer_engine_torch as tex

    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    qx = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1, rowwise=True, columnwise=True, with_rht=False
    )
    qw = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_rht=False,
        with_2d_quantization=True,
    )
    x4, w4 = qx(x), qw(w)
    out = torch.empty(m, n, device=device, dtype=torch.bfloat16)

    def step() -> None:
        general_gemm(w4, x4, layout="TN", out=out, out_dtype=torch.bfloat16)

    return step, (x, w, x4, w4, out)


def _make_fp8_step(m: int, n: int, k: int, device: str):
    a = torch.randn(m, k, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
    bt = b.t().contiguous()  # transpose NOT inside the timed region
    sa = torch.ones(1, device=device, dtype=torch.float32)
    sb = torch.ones(1, device=device, dtype=torch.float32)

    def step() -> None:
        torch._scaled_mm(
            a, bt, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16, use_fast_accum=True
        )

    return step, (a, b, bt)


def _make_bf16_step(m: int, n: int, k: int, device: str):
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b = torch.randn(k, n, device=device, dtype=torch.bfloat16)
    c = torch.empty(m, n, device=device, dtype=torch.bfloat16)

    def step() -> None:
        torch.mm(a, b, out=c)

    return step, (a, b, c)


def sweep(
    title: str,
    factory: Callable,
    device: str,
    iters: int,
    warmup: int,
    profile_iters: int,
    min_mma_ms: float,
) -> tuple[float, float, tuple[int, int, int] | None, str]:
    print(f"\n=== {title} ===")
    _print_header()
    best_api = 0.0
    best_mma = 0.0
    best_shape = None
    best_kname = ""
    for m, n, k in SHAPES:
        hold = None
        try:
            step, hold = factory(m, n, k, device)
            api_ms = time_api_ms(step, iters, warmup)
            mma_s, kname = time_mma_kernel_s(step, profile_iters, warmup=3)
        except Exception as exc:
            print(f"{m:5d} {n:5d} {k:5d}   skipped ({type(exc).__name__}: {exc})")
            del hold
            torch.cuda.empty_cache()
            continue
        _print_row(m, n, k, api_ms, mma_s, kname)
        api_tf = tflops(m, n, k, api_ms * 1e-3)
        mma_tf = tflops(m, n, k, mma_s)
        if api_ms >= min_mma_ms:
            best_api = max(best_api, api_tf)
        if mma_s * 1e3 >= min_mma_ms and mma_tf > best_mma:
            best_mma = mma_tf
            best_shape = (m, n, k)
            best_kname = kname
        del hold, step
        torch.cuda.empty_cache()
    print(
        f"--> API peak {best_api:.1f} TFLOPS | "
        f"MMA-kernel peak {best_mma:.1f} TFLOPS  shape={best_shape}"
    )
    return best_api, best_mma, best_shape, best_kname


def main() -> None:
    p = argparse.ArgumentParser(description="Kernel-only Tensor Core peak TFLOPS.")
    p.add_argument("--device", default=os.getenv("OPENVLA_DEVICE", "cuda:0"))
    p.add_argument("--iters", type=int, default=30, help="CUDA-event iters for API time")
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--profile-iters", type=int, default=8, help="profiler iters for MMA time")
    p.add_argument(
        "--min-ms",
        type=float,
        default=0.05,
        help="ignore shorter kernels when picking the peak",
    )
    p.add_argument("--skip-bf16", action="store_true")
    p.add_argument("--skip-fp8", action="store_true")
    p.add_argument("--skip-nvfp4", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    torch.backends.cuda.matmul.allow_tf32 = False

    cur, mx = gpu_gpc_mhz()
    print("gpu:", torch.cuda.get_device_name(0))
    print(
        "capability:",
        torch.cuda.get_device_capability(0),
        "| SMs:",
        torch.cuda.get_device_properties(0).multi_processor_count,
    )
    print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
    print(f"gpu gpc clock: {cur:.0f} / {mx:.0f} MHz   nvpmodel: {nvpmodel_name()}")
    print()
    print("datasheet T5000 MAXN  2070 sparse FP4  | dense FP4 1035 | FP8 517 | BF16 259")
    print("datasheet T5000 120W  1820 sparse FP4  | dense FP4  910 | FP8 455 | BF16 228")
    print("This script reports MMA-kernel TFLOPS (helpers excluded). Dense only, not 2:4 sparse.")
    print("Use the MMA TF column as 'achieved Tensor Core peak'.")

    results: dict[str, tuple[float, float]] = {}

    if not args.skip_bf16:
        api, mma, _, _ = sweep(
            "BF16 torch.mm  (cuBLAS Tensor Core)",
            _make_bf16_step,
            args.device,
            args.iters,
            args.warmup,
            args.profile_iters,
            args.min_ms,
        )
        results["BF16"] = (api, mma)

    if not args.skip_fp8:
        api, mma, _, _ = sweep(
            "FP8 torch._scaled_mm  (dense e4m3)",
            _make_fp8_step,
            args.device,
            args.iters,
            args.warmup,
            args.profile_iters,
            args.min_ms,
        )
        results["FP8"] = (api, mma)

    if not args.skip_nvfp4:
        try:
            import transformer_engine.pytorch  # noqa: F401
        except ImportError:
            print("\n=== NVFP4 ===\n  skipped (Transformer Engine not installed)")
        else:
            api, mma, shape, kname = sweep(
                "NVFP4 TE general_gemm  (dense block-scaled FP4)",
                _make_nvfp4_step,
                args.device,
                args.iters,
                args.warmup,
                args.profile_iters,
                args.min_ms,
            )
            results["NVFP4"] = (api, mma)
            if kname:
                print(f"    winning MMA kernel: {kname[:120]}")
            if shape:
                print(f"    winning shape: {shape}")

    print("\n========== summary (MMA kernel is the peak number) ==========")
    print(f"{'dtype':<8} {'API TF':>8} {'MMA TF':>8}   120W dense spec    MAXN dense spec")
    spec_120 = {"BF16": 228, "FP8": 455, "NVFP4": 910}
    spec_max = {"BF16": 259, "FP8": 517, "NVFP4": 1035}
    for name, (api, mma) in results.items():
        s12, smax = spec_120[name], spec_max[name]
        print(
            f"{name:<8} {api:8.1f} {mma:8.1f}   "
            f"{s12:4.0f} ({100*mma/s12:4.0f}%)         "
            f"{smax:4.0f} ({100*mma/smax:4.0f}%)"
        )

    print("\nHow to read this")
    print("  API TF  = whole library call (same idea as bench_peak_tensorcore.py).")
    print("  MMA TF  = only the Tensor Core GEMM kernel  <-- use this as achieved peak.")
    print("  2070 TFLOPS is sparse 2:4 FP4; this script is dense and cannot reach it.")
    print("  256 TFLOPS is dense BF16 paper spec; look at the BF16 MMA column, not NVFP4.")
    if cur < mx * 0.95:
        print(f"  Clock {cur:.0f} < max {mx:.0f} MHz.  sudo nvpmodel -m 0 && sudo jetson_clocks")


if __name__ == "__main__":
    main()
