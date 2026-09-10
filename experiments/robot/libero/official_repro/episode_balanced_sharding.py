#!/usr/bin/env python3
"""Build episode-balanced shard plans for multi-GPU LIBERO eval.

Instead of assigning whole tasks to GPUs (which yields uneven 100/50 splits for
10 tasks on 8 GPUs), this splits the flattened episode list as evenly as possible.

Example (10 tasks x 50 trials, 8 shards):
  old task-based: 100, 100, 50, 50, 50, 50, 50, 50
  balanced:        63,  63,  63,  63,  62,  62,  62,  62
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class Episode:
    task_id: int
    episode_idx: int


@dataclass
class ShardPlan:
    shard_idx: int
    episodes: list[Episode]

    @property
    def count(self) -> int:
        return len(self.episodes)

    def to_spec(self) -> str:
        """Serialize to deploy eval --episode-spec format, e.g. 0:0-24,1:0-12."""
        by_task: dict[int, list[int]] = defaultdict(list)
        for ep in self.episodes:
            by_task[ep.task_id].append(ep.episode_idx)

        parts: list[str] = []
        for task_id in sorted(by_task):
            indices = sorted(by_task[task_id])
            start = indices[0]
            prev = indices[0]
            for idx in indices[1:]:
                if idx == prev + 1:
                    prev = idx
                    continue
                parts.append(f"{task_id}:{start}-{prev}" if start != prev else f"{task_id}:{start}")
                start = prev = idx
            parts.append(f"{task_id}:{start}-{prev}" if start != prev else f"{task_id}:{start}")
        return ",".join(parts)


def flatten_episodes(num_tasks: int, trials_per_task: int) -> list[Episode]:
    return [
        Episode(task_id=task_id, episode_idx=episode_idx)
        for task_id in range(num_tasks)
        for episode_idx in range(trials_per_task)
    ]


def build_balanced_shards(
    num_tasks: int,
    trials_per_task: int,
    num_shards: int,
) -> list[ShardPlan]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")

    episodes = flatten_episodes(num_tasks, trials_per_task)
    total = len(episodes)
    if num_shards > total:
        raise ValueError(f"num_shards={num_shards} exceeds total episodes={total}")

    base = total // num_shards
    remainder = total % num_shards

    plans: list[ShardPlan] = []
    cursor = 0
    for shard_idx in range(num_shards):
        count = base + (1 if shard_idx < remainder else 0)
        shard_eps = episodes[cursor : cursor + count]
        cursor += count
        plans.append(ShardPlan(shard_idx=shard_idx, episodes=shard_eps))
    return plans


def summarize_plans(plans: Iterable[ShardPlan]) -> list[dict]:
    return [
        {
            "shard_idx": plan.shard_idx,
            "episodes": plan.count,
            "episode_spec": plan.to_spec(),
            "task_ids": sorted({ep.task_id for ep in plan.episodes}),
        }
        for plan in plans
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--format", choices=("spec", "json", "table"), default="spec")
    args = parser.parse_args()

    plans = build_balanced_shards(args.num_tasks, args.trials, args.shards)
    summary = summarize_plans(plans)

    if args.format == "json":
        print(json.dumps(summary, indent=2))
        return

    if args.format == "table":
        counts = [item["episodes"] for item in summary]
        print(f"total_episodes={args.num_tasks * args.trials} shards={args.shards}")
        print(f"per_shard={counts} min={min(counts)} max={max(counts)}")
        for item in summary:
            print(
                f"s{item['shard_idx']}: {item['episodes']} episodes | "
                f"tasks={item['task_ids']} | spec={item['episode_spec']}"
            )
        return

    for item in summary:
        print(item["episode_spec"])


if __name__ == "__main__":
    main()
