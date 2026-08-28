"""Measure Tensor-Core peak TFLOPS on Thor, beyond torch.mm.

Why bench_peak.py cannot report ~256 TFLOPS
------------------------------------------
`bench_peak.py` times dense BF16 `torch.mm` (cuBLAS). On this box that is
~110-160 TFLOPS. The ~256 number is **not** a PyTorch BF16 measurement:

    2070  sparse FP4  (official MAXN headline)
  /    2  -> 1035 dense FP4
  /    2  ->  517 dense FP8
  /    2  -> ~259 dense BF16/FP16     == 2070 / 8

At the 120W mode the same chain is 1820/8 ≈ 228 TFLOPS dense BF16.
So 256 is a datasheet ceiling, and only for dense BF16 at MAXN.

This script instead runs the kernels that actually exercise Tensor Cores:

    BF16  : torch.mm                         (same as bench_peak.py)
    FP8   : torch._scaled_mm  e4m3            (dense FP8, ~2x BF16)
    NVFP4 : Transformer Engine general_gemm   (dense FP4 block-scaled)

FLOP count is always 2*M*N*K (one FMA = 2 FLOPs), regardless of dtype.
Lower-bit Tensor Cores complete those FMAs faster, so TFLOPS goes up.

Typical reading on Thor 120W (GPC already at max 1575 MHz):
    BF16  ~160 TFLOPS   vs datasheet ~228   (~70%)
    FP8   ~270 TFLOPS   vs datasheet ~455   (~60%)
    NVFP4 ~450 TFLOPS   vs datasheet ~910   (~50%)   << this is the "peak" path

You still will not see 2070: that needs 2:4 *sparse* FP4, not dense GEMM.

    sudo nvpmodel -m 0 && sudo jetson_clocks     # MAXN + lock clocks
    python bench_peak_tensorcore.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import warnings
from typing import Callable

import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)


def sync() -> None:
    torch.cuda.synchronize()


def time_ms(fn: Callable[[], None], iters: int, warmup: int) -> float:
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


def tflops(m: int, n: int, k: int, ms: float) -> float:
    return (2.0 * m * n * k) / (ms * 1e-3) / 1e12


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
    for line in out.splitlines():
        line = line.strip()
        if line and "NV Power Mode" not in line:
            return line
    return out.strip()[:80] or "unknown"


# Shapes chosen so Tensor Cores stay busy (multiples of 128/256) and K is
# large enough that the kernel is compute-bound, not launch-overhead bound.
SHAPES = (
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (2048, 2048, 8192),
    (2048, 4096, 4096),
    (4096, 4096, 4096),
    (4096, 4096, 8192),
    (4096, 8192, 8192),
    (6144, 6144, 6144),
    (8192, 4096, 8192),
    (8192, 8192, 4096),
    (8192, 8192, 8192),
    (1024, 8192, 8192),
    (2048, 2048, 12288),
)


def _print_row(m: int, n: int, k: int, ms: float, tf: float) -> None:
    print(f"{m:6d} {n:6d} {k:6d} {ms:10.3f} {tf:10.1f}")


def bench_bf16(device: str, iters: int, warmup: int, min_ms: float) -> float:
    print("\n=== BF16  torch.mm (dense, cuBLAS) ===")
    print(f"{'M':>6} {'N':>6} {'K':>6} {'ms/iter':>10} {'TFLOPS':>10}")
    best = 0.0
    for m, n, k in SHAPES:
        try:
            a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
            b = torch.randn(k, n, device=device, dtype=torch.bfloat16)
            c = torch.empty(m, n, device=device, dtype=torch.bfloat16)
        except RuntimeError as exc:
            print(f"{m:6d} {n:6d} {k:6d}   skipped ({type(exc).__name__})")
            continue

        def step() -> None:
            torch.mm(a, b, out=c)

        ms = time_ms(step, iters, warmup)
        tf = tflops(m, n, k, ms)
        _print_row(m, n, k, ms, tf)
        if ms >= min_ms:
            best = max(best, tf)
        del a, b, c
        torch.cuda.empty_cache()
    print(f"--> dense BF16 peak (ms >= {min_ms}): {best:.1f} TFLOPS")
    return best


def bench_fp8_scaled_mm(device: str, iters: int, warmup: int, min_ms: float) -> float:
    if not hasattr(torch, "_scaled_mm") or not hasattr(torch, "float8_e4m3fn"):
        print("\n=== FP8  torch._scaled_mm ===\n  skipped (no _scaled_mm / float8)")
        return 0.0
    print("\n=== FP8  torch._scaled_mm (dense e4m3, fast accum) ===")
    print(f"{'M':>6} {'N':>6} {'K':>6} {'ms/iter':>10} {'TFLOPS':>10}")
    best = 0.0
    scale_a = torch.ones(1, device=device, dtype=torch.float32)
    scale_b = torch.ones(1, device=device, dtype=torch.float32)
    for m, n, k in SHAPES:
        try:
            a = torch.randn(m, k, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
            b = torch.randn(n, k, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
        except RuntimeError as exc:
            print(f"{m:6d} {n:6d} {k:6d}   skipped ({type(exc).__name__})")
            continue

        def step() -> None:
            torch._scaled_mm(
                a,
                b.t(),
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
                use_fast_accum=True,
            )

        try:
            ms = time_ms(step, iters, warmup)
        except Exception as exc:
            print(f"{m:6d} {n:6d} {k:6d}   skipped ({type(exc).__name__}: {exc})")
            del a, b
            continue
        tf = tflops(m, n, k, ms)
        _print_row(m, n, k, ms, tf)
        if ms >= min_ms:
            best = max(best, tf)
        del a, b
        torch.cuda.empty_cache()
    print(f"--> dense FP8 peak (ms >= {min_ms}): {best:.1f} TFLOPS")
    return best


def _te_nvfp4_gemm(m: int, n: int, k: int, device: str, iters: int, warmup: int) -> float:
    """Pre-quantize once, then time only the NVFP4 Tensor Core GEMM."""
    from transformer_engine.pytorch.cpp_extensions import general_gemm
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
    import transformer_engine_torch as tex

    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    # Activations: 1D block scale. Weights: 2D 16x16 block scale (NVFP4 recipe).
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

    ms = time_ms(step, iters, warmup)
    del x, w, x4, w4, out
    torch.cuda.empty_cache()
    return ms


def _te_fp8_gemm(m: int, n: int, k: int, device: str, iters: int, warmup: int) -> float:
    from transformer_engine.pytorch.cpp_extensions import general_gemm
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
    import transformer_engine_torch as tex

    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    scale = torch.ones(1, device=device, dtype=torch.float32)
    amax = torch.zeros(1, device=device, dtype=torch.float32)
    x8 = Float8Quantizer(scale.clone(), amax.clone(), tex.DType.kFloat8E4M3)(x)
    w8 = Float8Quantizer(scale.clone(), amax.clone(), tex.DType.kFloat8E4M3)(w)
    out = torch.empty(m, n, device=device, dtype=torch.bfloat16)

    def step() -> None:
        general_gemm(w8, x8, layout="TN", out=out, out_dtype=torch.bfloat16)

    ms = time_ms(step, iters, warmup)
    del x, w, x8, w8, out
    torch.cuda.empty_cache()
    return ms


def bench_te(
    kind: str,
    runner: Callable[[int, int, int, str, int, int], float],
    device: str,
    iters: int,
    warmup: int,
    min_ms: float,
) -> float:
    print(f"\n=== {kind}  Transformer Engine general_gemm (pre-quantized) ===")
    print(f"{'M':>6} {'N':>6} {'K':>6} {'ms/iter':>10} {'TFLOPS':>10}")
    best = 0.0
    for m, n, k in SHAPES:
        try:
            ms = runner(m, n, k, device, iters, warmup)
        except Exception as exc:
            print(f"{m:6d} {n:6d} {k:6d}   skipped ({type(exc).__name__}: {exc})")
            continue
        tf = tflops(m, n, k, ms)
        _print_row(m, n, k, ms, tf)
        if ms >= min_ms:
            best = max(best, tf)
    print(f"--> {kind} peak (ms >= {min_ms}): {best:.1f} TFLOPS")
    return best


def main() -> None:
    p = argparse.ArgumentParser(
        description="Tensor Core peak TFLOPS: BF16 / FP8 / NVFP4 (not just torch.mm)."
    )
    p.add_argument("--device", default=os.getenv("OPENVLA_DEVICE", "cuda:0"))
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument(
        "--min-ms",
        type=float,
        default=0.08,
        help="ignore kernels shorter than this when picking the peak "
        "(short kernels inflate TFLOPS via launch/L2 effects)",
    )
    p.add_argument("--skip-bf16", action="store_true")
    p.add_argument("--skip-fp8", action="store_true")
    p.add_argument("--skip-te", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    torch.backends.cuda.matmul.allow_tf32 = False
    cur_mhz, max_mhz = gpu_gpc_mhz()
    mode = nvpmodel_name()

    print("gpu:", torch.cuda.get_device_name(0))
    print(
        "capability:",
        torch.cuda.get_device_capability(0),
        "| SMs:",
        torch.cuda.get_device_properties(0).multi_processor_count,
    )
    print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
    print(f"gpu gpc clock: {cur_mhz:.0f} / {max_mhz:.0f} MHz   nvpmodel: {mode}")
    print()
    print("datasheet T5000 MAXN  2070 sparse FP4")
    print("                  /8  =>  ~259 dense BF16   /4 => ~517 dense FP8   /2 => ~1035 dense FP4")
    print("datasheet T5000 120W  1820 sparse FP4")
    print("                  /8  =>  ~228 dense BF16   /4 => ~455 dense FP8   /2 =>  ~910 dense FP4")
    print("256 TFLOPS ≈ MAXN dense-BF16 ceiling. This script will not hit it with torch.mm.")
    print("Look at NVFP4 / FP8 rows for the highest Tensor Core throughput.")

    results: dict[str, float] = {}
    if not args.skip_bf16:
        results["BF16 torch.mm"] = bench_bf16(args.device, args.iters, args.warmup, args.min_ms)
    if not args.skip_fp8:
        results["FP8  _scaled_mm"] = bench_fp8_scaled_mm(
            args.device, args.iters, args.warmup, args.min_ms
        )
    if not args.skip_te:
        try:
            import transformer_engine.pytorch  # noqa: F401
        except ImportError:
            print("\n=== Transformer Engine ===\n  skipped (not installed)")
        else:
            results["FP8  TE gemm"] = bench_te(
                "FP8 TE", _te_fp8_gemm, args.device, args.iters, args.warmup, args.min_ms
            )
            results["NVFP4 TE gemm"] = bench_te(
                "NVFP4", _te_nvfp4_gemm, args.device, args.iters, args.warmup, args.min_ms
            )

    print("\n========== summary ==========")
    print(f"{'backend':<22} {'measured':>10}   vs 120W datasheet          vs MAXN datasheet")
    ceilings_120w = {
        "BF16 torch.mm": 228,
        "FP8  _scaled_mm": 455,
        "FP8  TE gemm": 455,
        "NVFP4 TE gemm": 910,
    }
    ceilings_maxn = {
        "BF16 torch.mm": 259,
        "FP8  _scaled_mm": 517,
        "FP8  TE gemm": 517,
        "NVFP4 TE gemm": 1035,
    }
    for name, tf in results.items():
        c12 = ceilings_120w.get(name, float("nan"))
        cmax = ceilings_maxn.get(name, float("nan"))
        e12 = (100.0 * tf / c12) if c12 else 0.0
        emax = (100.0 * tf / cmax) if cmax else 0.0
        print(
            f"{name:<22} {tf:8.1f} TF   {c12:6.0f} ({e12:4.0f}%)              "
            f"{cmax:6.0f} ({emax:4.0f}%)"
        )

    print("\nHow to read this")
    print("  • 256 TFLOPS is the MAXN *dense BF16* paper spec, not a torch.mm result.")
    print("  • BF16 row  = fair roofline for OpenVLA BF16 Eager (same as bench_peak.py).")
    print("  • FP8 / NVFP4 rows = real Tensor Core peak of *those* dtypes.")
    print("  • 2070 TFLOPS needs sparse 2:4 FP4 kernels; dense NVFP4 cannot reach it.")
    if cur_mhz < max_mhz * 0.95:
        print(
            f"  • GPC clock is {cur_mhz:.0f} MHz < max {max_mhz:.0f} MHz. "
            "Run: sudo nvpmodel -m 0 && sudo jetson_clocks"
        )


if __name__ == "__main__":
    main()
