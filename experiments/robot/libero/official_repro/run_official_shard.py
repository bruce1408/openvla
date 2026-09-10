"""
Official OpenVLA LIBERO evaluation shard runner.

Uses the same TensorFlow preprocessing and model inference path as upstream
`run_libero_eval.py`, with added support for:
  - task_ids filtering (multi-GPU sharding)
  - structured JSONL + summary output
  - resume
  - configurable attention implementation (flash_attention_2 / sdpa)
"""

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

# Repo root on PYTHONPATH (experiments/robot/libero/official_repro -> openvla root)
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import get_vla_action  # noqa: E402
from experiments.robot.robot_utils import (  # noqa: E402
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.write("\n")
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def _parse_task_ids(raw: str) -> List[int]:
    ids: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            ids.extend(range(int(start_s), int(end_s) + 1))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def _load_completed_keys(jsonl_path: Path) -> set[tuple[int, int]]:
    done: set[tuple[int, int]] = set()
    if not jsonl_path.is_file():
        return done
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done.add((int(rec["task_id"]), int(rec["episode_idx"])))
    return done


def _load_jsonl_task_stats(jsonl_path: Path) -> tuple[dict[int, dict], int, int]:
    """Aggregate per-task stats from an existing JSONL (for resume / summary refresh)."""
    per_task: dict[int, dict] = {}
    total_episodes = 0
    total_successes = 0
    if not jsonl_path.is_file():
        return per_task, total_episodes, total_successes

    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            task_id = int(rec["task_id"])
            task = per_task.setdefault(
                task_id,
                {
                    "task_id": task_id,
                    "task_description": rec.get("task_description", ""),
                    "episodes": 0,
                    "successes": 0,
                    "errors": 0,
                },
            )
            task["episodes"] += 1
            total_episodes += 1
            if rec.get("success"):
                task["successes"] += 1
                total_successes += 1
            if rec.get("error"):
                task["errors"] += 1

    for task in per_task.values():
        eps = task["episodes"]
        task["success_rate"] = (task["successes"] / eps) if eps else 0.0

    return per_task, total_episodes, total_successes


def _expected_episode_keys(task_ids: list[int], num_trials_per_task: int) -> set[tuple[int, int]]:
    return {(task_id, episode_idx) for task_id in task_ids for episode_idx in range(num_trials_per_task)}


@dataclass
class OfficialShardConfig:
    # Model
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True
    attn_implementation: str = "flash_attention_2"

    # LIBERO
    task_suite_name: str = "libero_spatial"
    task_ids: str = ""  # e.g. "0-1" or "4" or "0,2,5"; empty = all tasks in suite
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    seed: int = 7

    # Output
    run_name: str = "official-libero-spatial-seed7"
    log_dir: str = "experiments/logs/libero_official"
    resume: bool = False
    local_files_only: bool = True


def _max_steps_for_suite(task_suite_name: str) -> int:
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }[task_suite_name]


def _load_model_with_attn(cfg: OfficialShardConfig):
    """Load VLA using in-repo Prismatic classes (works offline)."""
    import torch

    from deploy.tensorrt.common import load_openvla_model, load_openvla_processor

    attn = cfg.attn_implementation
    if attn == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            print("[warn] flash_attn not installed; falling back to sdpa")
            attn = "sdpa"

    print(f"[*] Loading VLA with attn_implementation={attn}")
    load_args = {
        "checkpoint": str(cfg.pretrained_checkpoint),
        "local_files_only": cfg.local_files_only,
    }
    processor = load_openvla_processor(**load_args)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_openvla_model(
        attn_implementation=attn,
        device=device,
        dtype_name="bf16",
        **load_args,
    )
    return model, processor


def _build_summary(cfg: OfficialShardConfig, task_stats: list[dict], total_episodes: int, total_successes: int) -> dict:
    return {
        "schema_version": 1,
        "eval_path": "official",
        "updated_at": _utc_now(),
        "run_config": {
            "model_family": cfg.model_family,
            "pretrained_checkpoint": str(cfg.pretrained_checkpoint),
            "task_suite_name": cfg.task_suite_name,
            "task_ids": cfg.task_ids or "all",
            "num_trials_per_task": cfg.num_trials_per_task,
            "num_steps_wait": cfg.num_steps_wait,
            "max_steps": _max_steps_for_suite(cfg.task_suite_name),
            "seed": cfg.seed,
            "center_crop": cfg.center_crop,
            "attn_implementation": cfg.attn_implementation,
            "preprocessing": "official_tensorflow",
        },
        "run_name": cfg.run_name,
        "episodes": total_episodes,
        "successes": total_successes,
        "errors": sum(t.get("errors", 0) for t in task_stats),
        "success_rate": (total_successes / total_episodes) if total_episodes else 0.0,
        "tasks": task_stats,
    }


@draccus.wrap()
def main(cfg: OfficialShardConfig) -> None:
    assert cfg.pretrained_checkpoint, "pretrained_checkpoint is required"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "center_crop must be True for image_aug checkpoints"

    set_seed_everywhere(cfg.seed)
    unnorm_key = cfg.task_suite_name

    # LIBERO loads init-state pickles via torch.load; PyTorch 2.6+ defaults weights_only=True.
    import torch

    _torch_load = torch.load

    def _torch_load_compat(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return _torch_load(*load_args, **load_kwargs)

    torch.load = _torch_load_compat  # type: ignore[method-assign]

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = log_dir / f"{cfg.run_name}.jsonl"
    summary_path = log_dir / f"{cfg.run_name}.summary.json"

    completed = _load_completed_keys(jsonl_path) if cfg.resume else set()

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()

    if cfg.task_ids.strip():
        selected_task_ids = _parse_task_ids(cfg.task_ids)
    else:
        selected_task_ids = list(range(task_suite.n_tasks))

    for tid in selected_task_ids:
        if tid < 0 or tid >= task_suite.n_tasks:
            raise ValueError(f"task_id {tid} out of range [0, {task_suite.n_tasks})")

    expected = _expected_episode_keys(selected_task_ids, cfg.num_trials_per_task)
    if cfg.resume and expected and expected <= completed:
        prior_by_task, total_episodes, total_successes = _load_jsonl_task_stats(jsonl_path)
        task_stats = []
        for task_id in selected_task_ids:
            task = prior_by_task.get(
                task_id,
                {
                    "task_id": task_id,
                    "task_description": "",
                    "episodes": 0,
                    "successes": 0,
                    "errors": 0,
                    "success_rate": 0.0,
                },
            )
            task_stats.append(task)
        final = _build_summary(cfg, task_stats, total_episodes, total_successes)
        _atomic_write_json(summary_path, final)
        print(
            f"All {len(expected)} episodes already complete for {cfg.run_name}; "
            f"refreshed summary ({total_successes}/{total_episodes} = {100 * final['success_rate']:.1f}%)."
        )
        print(f"Summary: {summary_path}")
        return

    # Official model load (supports sdpa fallback, offline-safe)
    model, processor = _load_model_with_attn(cfg)

    stats_path = Path(cfg.pretrained_checkpoint) / "dataset_statistics.json"
    if stats_path.is_file():
        with stats_path.open() as f:
            model.norm_stats = json.load(f)

    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.norm_stats, f"unnorm_key {unnorm_key} not in norm_stats"

    resize_size = get_image_resize_size(cfg)
    max_steps = _max_steps_for_suite(cfg.task_suite_name)

    prior_by_task: dict[int, dict] = {}
    if cfg.resume:
        prior_by_task, total_episodes, total_successes = _load_jsonl_task_stats(jsonl_path)
    else:
        total_episodes = 0
        total_successes = 0

    task_stats: list[dict] = []
    new_episodes = 0
    new_successes = 0

    print(f"Task suite: {cfg.task_suite_name}, tasks={selected_task_ids}, trials/task={cfg.num_trials_per_task}")

    jsonl_f = jsonl_path.open("a")
    try:
        for task_id in tqdm.tqdm(selected_task_ids, desc="tasks"):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

            task_episodes = 0
            task_successes = 0
            task_errors = 0

            for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc=f"task{task_id}", leave=False):
                if (task_id, episode_idx) in completed:
                    continue
                if episode_idx >= len(initial_states):
                    raise IndexError(
                        f"task {task_id} only has {len(initial_states)} init states, "
                        f"but episode_idx={episode_idx} (num_trials_per_task={cfg.num_trials_per_task})"
                    )

                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])

                t = 0
                success = False
                error_msg = None
                try:
                    while t < max_steps + cfg.num_steps_wait:
                        if t < cfg.num_steps_wait:
                            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                            t += 1
                            continue

                        img = get_libero_image(obs, resize_size)
                        observation = {
                            "full_image": img,
                            "state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                        }
                        action = get_vla_action(
                            model,
                            processor,
                            cfg.pretrained_checkpoint,
                            observation,
                            task_description,
                            unnorm_key,
                            center_crop=cfg.center_crop,
                        )
                        action = normalize_gripper_action(action, binarize=True)
                        if cfg.model_family == "openvla":
                            action = invert_gripper_action(action)

                        obs, _, done, _ = env.step(action.tolist())
                        if done:
                            success = True
                            break
                        t += 1
                except Exception as exc:
                    error_msg = str(exc)
                    task_errors += 1
                    print(f"[error] task={task_id} episode={episode_idx}: {exc}")

                task_episodes += 1
                new_episodes += 1
                if success:
                    task_successes += 1
                    new_successes += 1

                rec = {
                    "task_id": task_id,
                    "task_description": task_description,
                    "episode_idx": episode_idx,
                    "success": success,
                    "error": error_msg,
                    "steps": t,
                }
                jsonl_f.write(json.dumps(rec) + "\n")
                jsonl_f.flush()

                running_total_episodes = total_episodes + new_episodes
                running_total_successes = total_successes + new_successes
                summary = _build_summary(cfg, task_stats, running_total_episodes, running_total_successes)
                _atomic_write_json(summary_path, summary)

            prior = prior_by_task.get(task_id, {})
            merged_episodes = int(prior.get("episodes", 0)) + task_episodes
            merged_successes = int(prior.get("successes", 0)) + task_successes
            merged_errors = int(prior.get("errors", 0)) + task_errors
            task_stats.append(
                {
                    "task_id": task_id,
                    "task_description": task_description or prior.get("task_description", ""),
                    "episodes": merged_episodes,
                    "successes": merged_successes,
                    "errors": merged_errors,
                    "success_rate": (merged_successes / merged_episodes) if merged_episodes else 0.0,
                }
            )
            running_total_episodes = total_episodes + new_episodes
            running_total_successes = total_successes + new_successes
            summary = _build_summary(cfg, task_stats, running_total_episodes, running_total_successes)
            _atomic_write_json(summary_path, summary)

            if task_episodes:
                task_rate = 100 * task_successes / task_episodes
                new_part = f"{task_successes}/{task_episodes} ({task_rate:.1f}% new)"
            else:
                new_part = "0 new (all resumed)"
            print(
                f"task {task_id}: {merged_successes}/{merged_episodes} "
                f"({100 * merged_successes / merged_episodes:.1f}% total, {new_part}) | "
                f"total: {running_total_successes}/{running_total_episodes}"
            )
    finally:
        jsonl_f.close()

    final_total_episodes = total_episodes + new_episodes
    final_total_successes = total_successes + new_successes
    final = _build_summary(cfg, task_stats, final_total_episodes, final_total_successes)
    _atomic_write_json(summary_path, final)
    print(f"\nDone. success_rate={final['success_rate']:.4f} ({final_total_successes}/{final_total_episodes})")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
