#!/usr/bin/env python3
"""Compare paired BF16 and FP8 LIBERO evaluation outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


MATCHED_CONFIG_FIELDS = (
    "task_suite_name",
    "task_ids",
    "num_trials_per_task",
    "num_steps_wait",
    "max_steps",
    "seed",
    "env_seed",
    "center_crop",
    "preprocessing",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def result_path_for(summary_path: Path) -> Path:
    suffix = ".summary.json"
    if not summary_path.name.endswith(suffix):
        raise ValueError(f"Expected a {suffix} file: {summary_path}")
    return summary_path.with_name(summary_path.name[: -len(suffix)] + ".jsonl")


def read_rows(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = int(row["task_id"]), int(row["episode_idx"])
        if key in rows:
            raise ValueError(f"Duplicate episode {key} in {path}")
        rows[key] = row
    return rows


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


def compare(bf16_path: Path, fp8_path: Path, allow_config_mismatch: bool) -> dict[str, Any]:
    bf16_summary = read_json(bf16_path)
    fp8_summary = read_json(fp8_path)
    mismatches = {
        field: {
            "bf16": bf16_summary["run_config"].get(field),
            "fp8": fp8_summary["run_config"].get(field),
        }
        for field in MATCHED_CONFIG_FIELDS
        if bf16_summary["run_config"].get(field) != fp8_summary["run_config"].get(field)
    }
    if mismatches and not allow_config_mismatch:
        raise ValueError(f"Evaluation configurations differ: {mismatches}")

    bf16_rows = read_rows(result_path_for(bf16_path))
    fp8_rows = read_rows(result_path_for(fp8_path))
    paired_keys = sorted(set(bf16_rows) & set(fp8_rows))
    if not paired_keys:
        raise ValueError("The evaluations have no matching task/episode pairs")

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

    bf16_rate = sum(bool(bf16_rows[key]["success"]) for key in paired_keys) / len(paired_keys)
    fp8_rate = sum(bool(fp8_rows[key]["success"]) for key in paired_keys) / len(paired_keys)
    bf16_latency = bf16_summary["action_latency_ms"]["mean"]
    fp8_latency = fp8_summary["action_latency_ms"]["mean"]
    return {
        "bf16_summary": str(bf16_path),
        "fp8_summary": str(fp8_path),
        "config_mismatches": mismatches,
        "paired_episodes": len(paired_keys),
        "bf16_success_rate": bf16_rate,
        "fp8_success_rate": fp8_rate,
        "success_rate_delta_percentage_points": (fp8_rate - bf16_rate) * 100.0,
        "paired_success_delta": paired_interval(differences),
        "pair_outcomes": pair_counts,
        "bf16_action_latency_mean_ms": bf16_latency,
        "fp8_action_latency_mean_ms": fp8_latency,
        "latency_speedup": bf16_latency / fp8_latency if bf16_latency and fp8_latency else None,
        "per_task": per_task,
    }


def print_report(result: dict[str, Any]) -> None:
    print("BF16 vs FP8 LIBERO")
    print(f"Paired episodes: {result['paired_episodes']}")
    print(f"BF16 success: {result['bf16_success_rate']:.2%}")
    print(f"FP8 success:  {result['fp8_success_rate']:.2%}")
    print(f"Delta:        {result['success_rate_delta_percentage_points']:+.2f} percentage points")
    if result["latency_speedup"] is not None:
        print(
            f"Latency:      {result['bf16_action_latency_mean_ms']:.2f} ms -> "
            f"{result['fp8_action_latency_mean_ms']:.2f} ms "
            f"({result['latency_speedup']:.2f}x)"
        )
    outcomes = result["pair_outcomes"]
    print(
        "Pairs:        "
        f"both={outcomes['both_success']}, bf16-only={outcomes['bf16_only']}, "
        f"fp8-only={outcomes['fp8_only']}, neither={outcomes['both_failure']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-summary", type=Path, required=True)
    parser.add_argument("--fp8-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-config-mismatch", action="store_true")
    args = parser.parse_args()
    result = compare(args.bf16_summary, args.fp8_summary, args.allow_config_mismatch)
    print_report(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
