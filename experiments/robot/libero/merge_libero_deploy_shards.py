#!/usr/bin/env python3
"""Merge multi-GPU LIBERO deploy eval shards into one jsonl + summary."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.run_libero_deploy_eval import (  # noqa: E402
    load_rows,
    summarize,
    write_summary,
)


def merge_rows(shard_paths: list[Path]) -> list[dict[str, Any]]:
    merged: dict[tuple[int, int], dict[str, Any]] = {}
    for shard_path in shard_paths:
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing shard jsonl: {shard_path}")
        for row in load_rows(shard_path):
            key = int(row["task_id"]), int(row["episode_idx"])
            if key in merged:
                raise ValueError(f"Duplicate episode {key} across shards")
            merged[key] = row
    return [merged[key] for key in sorted(merged)]


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", type=Path, required=True, help="Shard .jsonl files")
    parser.add_argument("--shard-summaries", nargs="+", type=Path, required=True, help="Shard .summary.json files")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument(
        "--all-task-ids",
        nargs="+",
        type=int,
        default=None,
        help="Full task id list for merged run_config (default: infer from rows)",
    )
    args = parser.parse_args()
    if len(args.shards) != len(args.shard_summaries):
        raise SystemExit("--shards and --shard-summaries must have the same length")

    rows = merge_rows(args.shards)
    if not rows:
        raise SystemExit("No rows found in shards")

    template = load_summary(args.shard_summaries[0])
    run_config = dict(template["run_config"])
    task_ids = args.all_task_ids or sorted({int(row["task_id"]) for row in rows})
    run_config["task_ids"] = task_ids

    summary = summarize(rows, run_config, template["policy"])
    summary["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["merge"] = {
        "shard_jsonl": [str(path) for path in args.shards],
        "shard_summaries": [str(path) for path in args.shard_summaries],
    }

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    write_summary(args.output_summary, summary)
    print(f"Merged {len(rows)} episodes -> {args.output_jsonl}")
    print(f"Success rate: {summary['successes']}/{summary['episodes']} ({summary['success_rate']:.1%})")


if __name__ == "__main__":
    main()
