#!/usr/bin/env python3
"""Compare official-preprocessing BF16 vs FP8 LIBERO results from jsonl shards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl_files(paths: list[Path]) -> dict[tuple[int, int], dict[str, Any]]:
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for path in sorted(paths):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = int(row["task_id"]), int(row["episode_idx"])
                if key in rows:
                    raise ValueError(f"Duplicate episode {key} in {path}")
                rows[key] = row
    return rows


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def paired_interval(differences: list[float]) -> dict[str, float | None]:
    if not differences:
        return {"mean": None, "standard_error": None, "ci95_low": None, "ci95_high": None}
    values = np.asarray(differences, dtype=np.float64)
    mean = float(values.mean())
    if len(values) < 2:
        return {"mean": mean, "standard_error": None, "ci95_low": None, "ci95_high": None}
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
    return {
        "mean": mean,
        "standard_error": standard_error,
        "ci95_low": mean - 1.96 * standard_error,
        "ci95_high": mean + 1.96 * standard_error,
    }


def compare(
    bf16_rows: dict[tuple[int, int], dict[str, Any]],
    fp8_rows: dict[tuple[int, int], dict[str, Any]],
    bf16_summary: dict[str, Any] | None,
    fp8_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    paired_keys = sorted(set(bf16_rows) & set(fp8_rows))
    if not paired_keys:
        raise ValueError("No matching task/episode pairs between BF16 and FP8")

    pair_counts = {"both_success": 0, "bf16_only": 0, "fp8_only": 0, "both_failure": 0}
    differences = []
    task_data: dict[int, dict[str, Any]] = {}

    for key in paired_keys:
        bf16_success = bool(bf16_rows[key]["success"])
        fp8_success = bool(fp8_rows[key]["success"])
        differences.append(float(fp8_success) - float(bf16_success))
        if bf16_success and fp8_success:
            pair_counts["both_success"] += 1
        elif bf16_success:
            pair_counts["bf16_only"] += 1
        elif fp8_success:
            pair_counts["fp8_only"] += 1
        else:
            pair_counts["both_failure"] += 1

        task_id = key[0]
        task = task_data.setdefault(task_id, {"pairs": 0, "bf16_successes": 0, "fp8_successes": 0})
        task["pairs"] += 1
        task["bf16_successes"] += int(bf16_success)
        task["fp8_successes"] += int(fp8_success)

    per_task = []
    for task_id in sorted(task_data):
        task = task_data[task_id]
        pairs = task["pairs"]
        bf16_rate = task["bf16_successes"] / pairs
        fp8_rate = task["fp8_successes"] / pairs
        per_task.append(
            {
                "task_id": task_id,
                **task,
                "bf16_success_rate": bf16_rate,
                "fp8_success_rate": fp8_rate,
                "delta_percentage_points": (fp8_rate - bf16_rate) * 100.0,
            }
        )

    bf16_rate = sum(bool(bf16_rows[k]["success"]) for k in paired_keys) / len(paired_keys)
    fp8_rate = sum(bool(fp8_rows[k]["success"]) for k in paired_keys) / len(paired_keys)

    result: dict[str, Any] = {
        "preprocessing": "official",
        "paired_episodes": len(paired_keys),
        "bf16_success_rate": bf16_rate,
        "fp8_success_rate": fp8_rate,
        "success_rate_delta_percentage_points": (fp8_rate - bf16_rate) * 100.0,
        "paired_success_delta": paired_interval(differences),
        "pair_outcomes": pair_counts,
        "per_task": per_task,
    }

    if bf16_summary and "action_latency_ms" in bf16_summary:
        result["bf16_action_latency_mean_ms"] = bf16_summary["action_latency_ms"]["mean"]
    if fp8_summary and "action_latency_ms" in fp8_summary:
        result["fp8_action_latency_mean_ms"] = fp8_summary["action_latency_ms"]["mean"]
    if result.get("bf16_action_latency_mean_ms") and result.get("fp8_action_latency_mean_ms"):
        result["latency_speedup"] = (
            result["bf16_action_latency_mean_ms"] / result["fp8_action_latency_mean_ms"]
        )

    if bf16_summary:
        result["bf16_summary"] = bf16_summary.get("run_name") or bf16_summary.get("updated_at")
    if fp8_summary:
        result["fp8_summary"] = str(fp8_summary.get("policy", {}))

    return result


def print_report(result: dict[str, Any]) -> None:
    print("Official preprocessing: BF16 vs FP8")
    print(f"Paired episodes: {result['paired_episodes']}")
    print(f"BF16 success: {result['bf16_success_rate']:.2%}")
    print(f"FP8 success:  {result['fp8_success_rate']:.2%}")
    print(f"Delta:        {result['success_rate_delta_percentage_points']:+.2f} percentage points")
    delta = result["paired_success_delta"]
    if delta.get("ci95_low") is not None:
        print(
            f"95% CI:       [{delta['ci95_low']*100:+.2f}, {delta['ci95_high']*100:+.2f}] pp"
        )
    if result.get("latency_speedup") is not None:
        print(
            f"Latency:      {result['bf16_action_latency_mean_ms']:.1f} ms -> "
            f"{result['fp8_action_latency_mean_ms']:.1f} ms "
            f"({result['latency_speedup']:.2f}x)"
        )
    outcomes = result["pair_outcomes"]
    print(
        "Pairs:        "
        f"both={outcomes['both_success']}, bf16-only={outcomes['bf16_only']}, "
        f"fp8-only={outcomes['fp8_only']}, neither={outcomes['both_failure']}"
    )
    print("\nPer-task delta (FP8 - BF16, pp):")
    for task in result["per_task"]:
        print(
            f"  task {task['task_id']}: "
            f"BF16 {100*task['bf16_success_rate']:.0f}% | "
            f"FP8 {100*task['fp8_success_rate']:.0f}% | "
            f"Δ {task['delta_percentage_points']:+.0f}pp"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-shards", nargs="+", type=Path, required=True)
    parser.add_argument("--fp8-shards", nargs="+", type=Path, required=True)
    parser.add_argument("--bf16-summary", type=Path, default=None)
    parser.add_argument("--fp8-summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    bf16_rows = read_jsonl_files(args.bf16_shards)
    fp8_rows = read_jsonl_files(args.fp8_shards)
    bf16_summary = load_summary(args.bf16_summary) if args.bf16_summary else None
    fp8_summary = load_summary(args.fp8_summary) if args.fp8_summary else None

    result = compare(bf16_rows, fp8_rows, bf16_summary, fp8_summary)
    print_report(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON: {args.output}")


if __name__ == "__main__":
    main()
