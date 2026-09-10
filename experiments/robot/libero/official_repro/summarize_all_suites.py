#!/usr/bin/env python3
"""Summarize official LIBERO eval results across suites (BF16 / FP8 / compare)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_rate(summary: dict | None) -> str:
    if summary is None:
        return "—"
    return f"{100 * summary['success_rate']:.1f}% ({summary['successes']}/{summary['episodes']})"


def fmt_latency(summary: dict | None) -> str:
    if summary is None:
        return "—"
    lat = summary.get("action_latency_ms", {})
    mean = lat.get("mean")
    if mean is None:
        return "—"
    return f"{mean:.1f} ms"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=Path,
        default=Path("experiments/logs/libero_official"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["spatial", "object", "goal", "10"],
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict] = []
    print(f"{'suite':<10} {'BF16 official':<22} {'FP8 balanced':<22} {'compare':<12} {'BF16 lat':<10} {'FP8 lat'}")
    print("-" * 95)

    for suite in args.suites:
        bf16_tag = f"official-{suite}-seed{args.seed}-8gpu"
        fp8_tag = f"official-{suite}-seed{args.seed}-8gpu-fp8-balanced"
        compare_path = args.logdir / f"{bf16_tag}-bf16-vs-fp8-official.json"

        bf16 = load_summary(args.logdir / f"{bf16_tag}.summary.json")
        fp8 = load_summary(args.logdir / f"{fp8_tag}.summary.json")
        compare = load_summary(compare_path) if compare_path.is_file() else None

        compare_str = "—"
        if compare is not None:
            delta = compare.get("success_rate_delta_percentage_points")
            if delta is not None:
                compare_str = f"{delta:+.1f} pp"

        print(
            f"{suite:<10} {fmt_rate(bf16):<22} {fmt_rate(fp8):<22} {compare_str:<12} "
            f"{fmt_latency(bf16):<10} {fmt_latency(fp8)}"
        )

        rows.append(
            {
                "suite": suite,
                "seed": args.seed,
                "bf16_tag": bf16_tag,
                "fp8_tag": fp8_tag,
                "bf16_summary": str(args.logdir / f"{bf16_tag}.summary.json") if bf16 else None,
                "fp8_summary": str(args.logdir / f"{fp8_tag}.summary.json") if fp8 else None,
                "compare_json": str(compare_path) if compare else None,
                "bf16_success_rate": bf16["success_rate"] if bf16 else None,
                "fp8_success_rate": fp8["success_rate"] if fp8 else None,
                "success_rate_delta_pp": compare.get("success_rate_delta_percentage_points") if compare else None,
            }
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"seed": args.seed, "suites": rows}, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
