"""Measure the REAL achievable bf16 peak compute (TFLOPS) and memory bandwidth
(GB/s) of this GPU, so the roofline in the profiling report can be calibrated
against measured hardware limits instead of datasheet guesses.

Compute peak  : large GEMMs, FLOP = 2*M*N*K per matmul.
Bandwidth peak: a large device-to-device copy, bytes = 2 * tensor_bytes
                (one read + one write).

Thor T5000 datasheet vs this script
-----------------------------------
Official headline is **2070 TFLOPS sparse FP4 at MAXN**, not dense BF16.

    2070  sparse FP4   (2:4 sparsity, 4-bit Tensor Core)
  /    2  -> 1035 dense FP4
  /    2  ->  517 dense FP8
  /    2  -> ~259 dense BF16/FP16     == 2070 / 8

So `2070 / 8 ≈ 259 TFLOPS` is the theoretical dense-BF16 Tensor Core ceiling
at MAXN, not what `torch.mm` is guaranteed to hit. This script measures
**dense BF16 through PyTorch/cuBLAS**, which is the right roofline for an
OpenVLA BF16 eager workload. On Thor that typically lands near ~100-140
TFLOPS, about half the datasheet ceiling (kernel/library efficiency, not
a broken FLOP formula).

At the named 120W mode the sparse-FP4 number is 1820, so dense BF16
ceiling is ~1820/8 ≈ 228 TFLOPS.

To approach the 2070 number you need all of: MAXN + sparse 2:4 + FP4
Tensor Core kernels (CUTLASS / TensorRT / Transformer Engine) — not
`torch.mm` in bf16.

Jetson note: for a true peak, first pin clocks to max:
    sudo nvpmodel -m 0      # MAXN power mode
    sudo jetson_clocks      # lock GPU/EMC clocks to max
Otherwise DVFS will report a number below the hardware ceiling.
"""

import argparse
import os
import time

import torch


def sync() -> None:
    torch.cuda.synchronize()


def time_ms(fn, iters: int, warmup: int) -> float:
    """Median-free mean wall time (ms) of `fn` over `iters`, after `warmup`."""
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


def gpu_gpc_mhz() -> tuple[float, float]:
    """Read Thor GPU GPC clock (MHz). Returns (current, max)."""
    base = "/sys/class/devfreq/gpu-gpc-0"
    try:
        cur = int(open(f"{base}/cur_freq").read()) / 1e6
        mx = int(open(f"{base}/max_freq").read()) / 1e6
        return cur, mx
    except OSError:
        return float("nan"), float("nan")


def bench_gemm(dtype: torch.dtype, device: str, iters: int, warmup: int) -> float:
    print(f"\n=== GEMM peak ({str(dtype).replace('torch.', '')}) ===")
    print(f"{'M':>6} {'N':>6} {'K':>6} {'ms/iter':>10} {'TFLOPS':>10}")
    best = 0.0
    # Square sizes plus a few rectangular ones: cuBLAS on Thor sm_110 is
    # ~20-30% faster on some non-square tiles than on 8192^3.
    shapes = (
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 4096, 4096),
        (8192, 8192, 8192),
        (16384, 8192, 4096),
        (12288, 12288, 12288),
        (16384, 16384, 16384),
    )
    for m, n, k in shapes:
        try:
            a = torch.randn(m, k, device=device, dtype=dtype)
            b = torch.randn(k, n, device=device, dtype=dtype)
            c = torch.empty(m, n, device=device, dtype=dtype)
        except RuntimeError as exc:  # OOM on the largest sizes
            print(f"{m:>6} {n:>6} {k:>6}   skipped ({type(exc).__name__})")
            continue

        def step() -> None:
            torch.mm(a, b, out=c)

        ms = time_ms(step, iters, warmup)
        flops = 2.0 * m * n * k
        tflops = flops / (ms * 1e-3) / 1e12
        best = max(best, tflops)
        print(f"{m:>6} {n:>6} {k:>6} {ms:>10.3f} {tflops:>10.1f}")
        del a, b, c
        torch.cuda.empty_cache()
    print(f"--> measured dense {str(dtype).replace('torch.', '')} peak: {best:.1f} TFLOPS")
    return best


def bench_fp8_probe(device: str, iters: int, warmup: int) -> float:
    """Optional FP8 Tensor Core sanity check via torch._scaled_mm.

    Dense FP8 datasheet (MAXN) is ~517 TFLOPS, about 2x dense BF16. If this
    lands near 2x the BF16 number, Tensor Cores are working and the BF16
    result is a real efficiency gap, not a broken timer.
    """
    if not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "_scaled_mm"):
        print("\n=== FP8 probe ===\n  skipped (no torch._scaled_mm)")
        return 0.0
    print("\n=== FP8 probe (torch._scaled_mm, dense e4m3) ===")
    print(f"{'M':>6} {'N':>6} {'K':>6} {'ms/iter':>10} {'TFLOPS':>10}")
    best = 0.0
    scale_a = torch.ones(1, device=device, dtype=torch.float32)
    scale_b = torch.ones(1, device=device, dtype=torch.float32)
    for m, n, k in ((4096, 4096, 4096), (8192, 8192, 8192)):
        try:
            a = torch.randn(m, k, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
            # _scaled_mm wants B as (K, N) after .t(), so store B as (N, K).
            b = torch.randn(n, k, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)

            def step() -> None:
                torch._scaled_mm(a, b.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)

            ms = time_ms(step, iters, warmup)
            tflops = (2.0 * m * n * k) / (ms * 1e-3) / 1e12
            best = max(best, tflops)
            print(f"{m:>6} {n:>6} {k:>6} {ms:>10.3f} {tflops:>10.1f}")
            del a, b
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"{m:>6} {n:>6} {k:>6}   skipped ({type(exc).__name__}: {exc})")
    print(f"--> measured dense FP8 peak: {best:.1f} TFLOPS")
    return best


def bench_gemm_sustained(dtype: torch.dtype, device: str, size: int, seconds: float, warmup: int) -> None:
    """Hold one large GEMM under sustained load and watch it settle.

    A real inference workload runs the GPU flat-out for many ms at a time, so the
    honest roofline denominator is the throttled steady-state TFLOPS, not the
    short-kernel boost peak. This runs the same GEMM for `seconds` and prints the
    TFLOPS of each ~fixed-work window, so you can see boost -> steady-state.
    """
    print(f"\n=== GEMM sustained ({str(dtype).replace('torch.', '')}, M=N=K={size}, ~{seconds:.0f}s) ===")
    try:
        a = torch.randn(size, size, device=device, dtype=dtype)
        b = torch.randn(size, size, device=device, dtype=dtype)
        c = torch.empty(size, size, device=device, dtype=dtype)
    except RuntimeError as exc:
        print(f"  skipped ({type(exc).__name__}) — try a smaller --sustained-size")
        return

    def step() -> None:
        torch.mm(a, b, out=c)

    for _ in range(warmup):
        step()
    sync()

    flops = 2.0 * size * size * size
    window_iters = 20
    print(f"{'t(s)':>6} {'TFLOPS':>10}")
    boost = 0.0
    last = 0.0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(window_iters):
            step()
        end.record()
        sync()
        tflops = flops / (start.elapsed_time(end) / window_iters * 1e-3) / 1e12
        boost = max(boost, tflops)
        last = tflops
        print(f"{time.perf_counter() - t0:>6.1f} {tflops:>10.1f}")

    drop = (1.0 - last / boost) * 100.0 if boost else 0.0
    print(f"--> burst (best window):      {boost:.1f} TFLOPS")
    print(f"--> sustained (last window):  {last:.1f} TFLOPS  (throttled {drop:.0f}% below burst)")
    print("    Use the SUSTAINED value as the roofline denominator for a real workload.")
    del a, b, c
    torch.cuda.empty_cache()


def bench_bandwidth(device: str, iters: int, warmup: int) -> None:
    print("\n=== Memory bandwidth (device-to-device copy) ===")
    print(f"{'bytes(MB)':>10} {'ms/iter':>10} {'GB/s':>10}")
    best = 0.0
    for mb in (256, 512, 1024, 2048):
        n = (mb * 1024 * 1024) // 2  # bf16 = 2 bytes/elem
        src = torch.randn(n, device=device, dtype=torch.bfloat16)
        dst = torch.empty_like(src)

        def step() -> None:
            dst.copy_(src)

        ms = time_ms(step, iters, warmup)
        moved = 2.0 * src.numel() * src.element_size()  # read + write
        gbps = moved / (ms * 1e-3) / 1e9
        best = max(best, gbps)
        print(f"{mb:>10} {ms:>10.3f} {gbps:>10.1f}")
        del src, dst
        torch.cuda.empty_cache()
    print(f"--> measured peak bandwidth: {best:.1f} GB/s")


def main() -> None:
    p = argparse.ArgumentParser(description="Measure real bf16 peak TFLOPS and bandwidth.")
    p.add_argument("--device", default=os.getenv("OPENVLA_DEVICE", "cuda:0"))
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--sustained-seconds", type=float, default=15.0, help="hold a large GEMM this long to read steady-state TFLOPS; 0 to skip")
    p.add_argument("--sustained-size", type=int, default=8192, help="M=N=K for the sustained-load GEMM")
    p.add_argument("--probe-fp8", default=True, help="also time dense FP8 _scaled_mm as a Tensor Core sanity check")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    torch.backends.cuda.matmul.allow_tf32 = False  # measure true bf16, not tf32
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    cur_mhz, max_mhz = gpu_gpc_mhz()
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0),
          "| SMs:", torch.cuda.get_device_properties(0).multi_processor_count)
    print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
    print("device:", args.device, "| iters:", args.iters, "| warmup:", args.warmup)
    print(f"gpu gpc clock: {cur_mhz:.0f} MHz idle / {max_mhz:.0f} MHz max")
    print("datasheet T5000 MAXN: 2070 sparse FP4  =>  dense BF16 ceiling ≈ 2070/8 ≈ 259 TFLOPS")
    print("datasheet T5000 120W: 1820 sparse FP4  =>  dense BF16 ceiling ≈ 1820/8 ≈ 228 TFLOPS")
    print("this script measures dense PyTorch GEMM, not sparse FP4")

    bf16_peak = bench_gemm(dtype, args.device, args.iters, args.warmup)
    if args.sustained_seconds > 0:
        bench_gemm_sustained(dtype, args.device, args.sustained_size, args.sustained_seconds, args.warmup)
    fp8_peak = bench_fp8_probe(args.device, args.iters, args.warmup) if args.probe_fp8 else 0.0
    bench_bandwidth(args.device, args.iters, args.warmup)

    print("\nHow to read these numbers")
    print(f"  measured dense {args.dtype}: {bf16_peak:.1f} TFLOPS  (use THIS for a BF16 PyTorch roofline)")
    if fp8_peak:
        print(f"  measured dense FP8 : {fp8_peak:.1f} TFLOPS  (expect ~2x BF16 if Tensor Cores are on)")
    print("  theoretical dense BF16 @ MAXN: ~259 TFLOPS  (2070 sparse FP4 / 8)")
    print("  theoretical dense BF16 @ 120W: ~228 TFLOPS  (1820 sparse FP4 / 8)")
    print("  official 2070 TFLOPS is sparse FP4, unreachable with torch.mm in bf16")
    print("\nRoofline ridge point = peak_TFLOPS*1e12 / peak_GBps*1e9  (FLOP/byte)")
    print("Compare your workload's arithmetic intensity against this ridge:")
    print("  AI < ridge  -> memory-bound;  AI > ridge -> compute-bound")


if __name__ == "__main__":
    main()
