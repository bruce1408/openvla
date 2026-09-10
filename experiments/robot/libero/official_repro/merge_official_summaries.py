#!/usr/bin/env python3
"""Merge official LIBERO shard summaries into one combined summary."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_summary(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def merge_summaries(summaries: list[dict], output_path: Path, run_name: str) -> dict:
    if not summaries:
        raise ValueError("no summaries to merge")

    tasks = []
    total_episodes = 0
    total_successes = 0
    total_errors = 0

    for summary in summaries:
        for task in summary.get("tasks", []):
            tasks.append(task)
            total_episodes += int(task.get("episodes", 0))
            total_successes += int(task.get("successes", 0))
            total_errors += int(task.get("errors", 0))

    tasks.sort(key=lambda t: int(t["task_id"]))

    base_cfg = summaries[0].get("run_config", {})
    merged = {
        "schema_version": 1,
        "eval_path": "official",
        "updated_at": _utc_now(),
        "run_name": run_name,
        "run_config": {
            **base_cfg,
            "task_ids": "all",
        },
        "episodes": total_episodes,
        "successes": total_successes,
        "errors": total_errors,
        "success_rate": (total_successes / total_episodes) if total_episodes else 0.0,
        "tasks": tasks,
        "merge": {
            "shard_summaries": [str(s.get("_source_path", "")) for s in summaries],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(merged, f, indent=2)
        f.write("\n")

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge official LIBERO shard summaries")
    parser.add_argument("--summaries", nargs="+", required=True, help="Shard summary JSON paths")
    parser.add_argument("--output", required=True, help="Merged summary output path")
    parser.add_argument("--run-name", default="", help="Run name for merged summary")
    args = parser.parse_args()

    summaries = []
    for p in args.summaries:
        path = Path(p)
        data = load_summary(path)
        data["_source_path"] = str(path)
        summaries.append(data)

    run_name = args.run_name or Path(args.output).stem
    merged = merge_summaries(summaries, Path(args.output), run_name)

    print(f"Merged {len(summaries)} shards -> {args.output}")
    print(f"  episodes:  {merged['episodes']}")
    print(f"  successes: {merged['successes']}")
    print(f"  rate:      {100 * merged['success_rate']:.2f}%")


if __name__ == "__main__":
    main()
