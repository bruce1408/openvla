#!/usr/bin/env python3
"""Run identical LIBERO rollouts with an OpenVLA BF16 or FP8 backend."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.libero_policy_backends import create_policy  # noqa: E402


TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def parse_task_ids(value: str, num_tasks: int) -> list[int]:
    if value == "all":
        return list(range(num_tasks))
    selected: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            selected.update(range(int(start_text), int(end_text) + 1))
        else:
            selected.add(int(item))
    if not selected or min(selected) < 0 or max(selected) >= num_tasks:
        raise ValueError(f"--task-ids must select values in [0, {num_tasks - 1}]")
    return sorted(selected)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _official_preprocess(image: np.ndarray, center_crop: bool) -> np.ndarray:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "Official LIBERO image preprocessing requires tensorflow. Install the official "
            "LIBERO/OpenVLA dependencies, or use --preprocessing portable for a smoke test."
        ) from exc

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass
    tensor = tf.image.encode_jpeg(image)
    tensor = tf.io.decode_image(tensor, expand_animations=False, dtype=tf.uint8)
    tensor = tf.image.resize(tensor, (224, 224), method="lanczos3", antialias=True)
    tensor = tf.cast(tf.clip_by_value(tf.round(tensor), 0, 255), tf.uint8)
    if center_crop:
        float_image = tf.image.convert_image_dtype(tensor, tf.float32)[None]
        side = tf.sqrt(tf.constant(0.9, dtype=tf.float32))
        offset = (1.0 - side) / 2.0
        boxes = tf.reshape(tf.stack([offset, offset, offset + side, offset + side]), (1, 4))
        tensor = tf.image.crop_and_resize(float_image, boxes, [0], (224, 224))[0]
        tensor = tf.image.convert_image_dtype(tf.clip_by_value(tensor, 0, 1), tf.uint8, saturate=True)
    return tensor.numpy()


def _portable_preprocess(image: np.ndarray, center_crop: bool) -> np.ndarray:
    """Dependency-light approximation for smoke tests, not paper-number reproduction."""

    source = Image.fromarray(image).convert("RGB")
    encoded = io.BytesIO()
    source.save(encoded, format="JPEG", quality=95)
    encoded.seek(0)
    source = Image.open(encoded).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
    if center_crop:
        side = 224.0 * math.sqrt(0.9)
        offset = (224.0 - side) / 2.0
        source = source.crop((offset, offset, offset + side, offset + side)).resize(
            (224, 224), Image.Resampling.BILINEAR
        )
    return np.asarray(source, dtype=np.uint8)


def preprocess_observation(obs: dict[str, Any], method: str, center_crop: bool) -> Image.Image:
    image = np.asarray(obs["agentview_image"], dtype=np.uint8)[::-1, ::-1]
    if method == "official":
        image = _official_preprocess(image, center_crop)
    else:
        image = _portable_preprocess(image, center_crop)
    return Image.fromarray(image).convert("RGB")


def transform_action_for_libero(action: np.ndarray) -> np.ndarray:
    """Map OpenVLA gripper [0, 1] convention to LIBERO's command convention."""

    action = np.asarray(action, dtype=np.float32).copy()
    if action.shape != (7,):
        raise ValueError(f"LIBERO expects a 7D action, got {action.shape}")
    action[-1] = np.sign(2.0 * action[-1] - 1.0)
    action[-1] *= -1.0
    return action


def should_save_video(mode: str, success: bool) -> bool:
    return mode == "all" or mode == "success" and success or mode == "failure" and not success


def save_video(images: list[np.ndarray], path: Path) -> None:
    if not images:
        return
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=30) as writer:
        for image in images:
            writer.append_data(image)


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile_value))


def summarize(rows: list[dict[str, Any]], run_config: dict[str, Any], policy_metadata: dict[str, Any]) -> dict[str, Any]:
    successes = sum(bool(row["success"]) for row in rows)
    all_latencies = [latency for row in rows for latency in row.get("action_latencies_ms", [])]
    tasks: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_key = str(row["task_id"])
        task = tasks.setdefault(
            task_key,
            {
                "task_id": row["task_id"],
                "task_description": row["task_description"],
                "episodes": 0,
                "successes": 0,
                "errors": 0,
            },
        )
        task["episodes"] += 1
        task["successes"] += int(bool(row["success"]))
        task["errors"] += int(row.get("error") is not None)
    for task in tasks.values():
        task["success_rate"] = task["successes"] / task["episodes"] if task["episodes"] else 0.0

    return {
        "schema_version": 1,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_config": run_config,
        "policy": policy_metadata,
        "episodes": len(rows),
        "successes": successes,
        "errors": sum(row.get("error") is not None for row in rows),
        "success_rate": successes / len(rows) if rows else 0.0,
        "action_calls": len(all_latencies),
        "action_latency_ms": {
            "mean": float(np.mean(all_latencies)) if all_latencies else None,
            "p50": percentile(all_latencies, 50),
            "p95": percentile(all_latencies, 95),
            "p99": percentile(all_latencies, 99),
        },
        "tasks": [tasks[key] for key in sorted(tasks, key=int)],
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    artifacts = REPO_ROOT / "deploy/tensorrt/artifacts"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("bf16", "fp8"), required=True)
    parser.add_argument("--checkpoint", required=True, help="LIBERO suite fine-tuned checkpoint or local directory")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--task-suite-name", choices=tuple(TASK_MAX_STEPS), default="libero_spatial")
    parser.add_argument("--task-ids", default="all", help="all, a comma list, or ranges such as 0,2-4")
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None, help="Override the suite episode horizon")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preprocessing", choices=("official", "portable"), default="official")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--attn-implementation", default="sdpa", help="BF16 only")

    parser.add_argument("--vision-engine", type=Path, default=artifacts / "engines/vision_projector_fp8.plan")
    parser.add_argument("--llm-engine-dir", type=Path, default=artifacts / "engines/openvla_llama_fp8")
    parser.add_argument("--action-metadata", type=Path, default=artifacts / "action_meta/action_meta.json")
    parser.add_argument(
        "--edge-llm-plugin",
        type=Path,
        default=Path(os.environ.get("EDGE_LLM_DIR", "/workspace/TensorRT-Edge-LLM"))
        / "build/libNvInfer_edgellm_plugin.so",
    )
    parser.add_argument("--allow-missing-fp8-provenance", action="store_true")

    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "experiments/logs/libero_deploy")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-video", choices=("none", "success", "failure", "all"), default="none")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_trials_per_task < 1:
        raise SystemExit("--num-trials-per-task must be positive")
    os.environ.setdefault("MUJOCO_GL", "egl")
    set_seed(args.seed)

    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as exc:
        raise SystemExit(
            "LIBERO is not installed. Install the LIBERO repository and "
            "experiments/robot/libero/libero_requirements.txt before running this evaluator."
        ) from exc

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task_ids = parse_task_ids(args.task_ids, suite.n_tasks)
    horizon = args.max_steps or TASK_MAX_STEPS[args.task_suite_name]
    policy = create_policy(args)

    run_name = args.run_name or (
        f"{args.task_suite_name}-{args.backend}-seed{args.seed}-"
        f"{time.strftime('%Y%m%d-%H%M%S')}"
    )
    result_path = args.output_dir / f"{run_name}.jsonl"
    summary_path = args.output_dir / f"{run_name}.summary.json"
    if result_path.exists() and not args.resume:
        raise SystemExit(f"Output already exists: {result_path}; use --resume or a new --run-name")

    rows = load_rows(result_path) if args.resume else []
    completed = {(int(row["task_id"]), int(row["episode_idx"])) for row in rows}
    run_config = {
        "backend": args.backend,
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "task_suite_name": args.task_suite_name,
        "task_ids": task_ids,
        "num_trials_per_task": args.num_trials_per_task,
        "num_steps_wait": args.num_steps_wait,
        "max_steps": horizon,
        "seed": args.seed,
        "env_seed": args.env_seed,
        "center_crop": args.center_crop,
        "preprocessing": args.preprocessing,
    }
    if args.resume and summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("run_config") != run_config:
            raise SystemExit("Resume configuration differs from the existing summary; use a new --run-name")

    print(f"Backend: {args.backend} | suite: {args.task_suite_name} | tasks: {task_ids}")
    print(f"Results: {result_path}")
    for task_id in task_ids:
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        if args.num_trials_per_task > len(initial_states):
            raise ValueError(
                f"Task {task_id} has {len(initial_states)} initial states, "
                f"but {args.num_trials_per_task} trials were requested"
            )
        bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=256,
            camera_widths=256,
        )
        env.seed(args.env_seed)
        try:
            for episode_idx in range(args.num_trials_per_task):
                if (task_id, episode_idx) in completed:
                    continue
                action_latencies: list[float] = []
                replay_images: list[np.ndarray] = []
                success = False
                error = None
                steps = 0
                started = time.perf_counter()
                try:
                    env.reset()
                    obs = env.set_init_state(initial_states[episode_idx])
                    for timestep in range(horizon + args.num_steps_wait):
                        if timestep < args.num_steps_wait:
                            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                            continue
                        image = preprocess_observation(obs, args.preprocessing, args.center_crop)
                        replay_images.append(np.asarray(image))
                        prediction = policy.predict(image, task.language)
                        action_latencies.append(prediction.latency_ms)
                        action = transform_action_for_libero(prediction.action)
                        obs, _, done, _ = env.step(action.tolist())
                        steps += 1
                        if done:
                            success = True
                            break
                except Exception as exc:  # Keep a long benchmark resumable and auditable.
                    error = f"{type(exc).__name__}: {exc}"

                row = {
                    "task_id": task_id,
                    "task_description": task.language,
                    "episode_idx": episode_idx,
                    "success": success,
                    "error": error,
                    "steps": steps,
                    "wall_time_s": time.perf_counter() - started,
                    "action_latencies_ms": action_latencies,
                }
                append_row(result_path, row)
                rows.append(row)
                summary = summarize(rows, run_config, policy.metadata)
                write_summary(summary_path, summary)

                if should_save_video(args.save_video, success):
                    video_path = args.output_dir / "videos" / run_name / (
                        f"task{task_id:02d}-episode{episode_idx:03d}-success{int(success)}.mp4"
                    )
                    save_video(replay_images, video_path)
                print(
                    f"task={task_id:02d} episode={episode_idx:03d} success={success} "
                    f"steps={steps} total={summary['successes']}/{summary['episodes']} "
                    f"({summary['success_rate']:.1%})"
                )
                if error:
                    print(f"  error: {error}")
        finally:
            env.close()

    final_summary = summarize(rows, run_config, policy.metadata)
    write_summary(summary_path, final_summary)
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
