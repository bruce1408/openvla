from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from deploy.tensorrt.runtime.edge_llm_runner import build_rope_cos_sin
from experiments.robot.libero.compare_libero_results import compare
from experiments.robot.libero.libero_policy_backends import (
    decode_action_tokens_from_metadata,
    resolve_unnorm_key,
)
from experiments.robot.libero.run_libero_deploy_eval import (
    parse_task_ids,
    summarize,
    transform_action_for_libero,
)


class BackendHelpersTest(unittest.TestCase):
    def test_resolve_unnorm_key_accepts_no_noops_variant(self) -> None:
        self.assertEqual(
            resolve_unnorm_key({"bridge_orig", "libero_spatial_no_noops"}, "libero_spatial"),
            "libero_spatial_no_noops",
        )

    def test_decode_action_tokens_uses_metadata(self) -> None:
        metadata = {
            "action_dim": 2,
            "effective_vocab_size": 32000,
            "bin_centers": [-0.5, 0.0, 0.5],
            "q01": [0.0, -2.0],
            "q99": [2.0, 2.0],
            "mask": [True, False],
        }
        action = decode_action_tokens_from_metadata([31999, 31998], metadata)
        np.testing.assert_allclose(action, [0.5, 0.0])

    def test_rope_cache_layout(self) -> None:
        cache = build_rope_cos_sin(4, 8, 10000.0)
        self.assertEqual(cache.shape, (1, 4, 8))
        np.testing.assert_allclose(cache[0, 0, :4], 1.0)
        np.testing.assert_allclose(cache[0, 0, 4:], 0.0)


class EvaluationHelpersTest(unittest.TestCase):
    def test_parse_task_ids(self) -> None:
        self.assertEqual(parse_task_ids("0,2-4", 6), [0, 2, 3, 4])
        self.assertEqual(parse_task_ids("all", 3), [0, 1, 2])

    def test_gripper_transform(self) -> None:
        open_action = transform_action_for_libero(np.array([0, 0, 0, 0, 0, 0, 0.9]))
        close_action = transform_action_for_libero(np.array([0, 0, 0, 0, 0, 0, 0.1]))
        self.assertEqual(open_action[-1], -1.0)
        self.assertEqual(close_action[-1], 1.0)

    def test_summary(self) -> None:
        rows = [
            {
                "task_id": 0,
                "task_description": "task",
                "episode_idx": 0,
                "success": True,
                "error": None,
                "action_latencies_ms": [10.0, 20.0],
            },
            {
                "task_id": 0,
                "task_description": "task",
                "episode_idx": 1,
                "success": False,
                "error": None,
                "action_latencies_ms": [30.0],
            },
        ]
        summary = summarize(rows, {"backend": "bf16"}, {"backend": "bf16"})
        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["action_latency_ms"]["mean"], 20.0)


class ComparisonTest(unittest.TestCase):
    def test_paired_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = {
                "task_suite_name": "libero_spatial",
                "task_ids": [0],
                "num_trials_per_task": 2,
                "num_steps_wait": 10,
                "max_steps": 220,
                "seed": 7,
                "env_seed": 0,
                "center_crop": True,
                "preprocessing": "official",
            }
            for backend, successes, latency in (
                ("bf16", [True, False], 20.0),
                ("fp8", [True, True], 10.0),
            ):
                summary_path = root / f"{backend}.summary.json"
                summary_path.write_text(
                    json.dumps({"run_config": config, "action_latency_ms": {"mean": latency}}),
                    encoding="utf-8",
                )
                rows = [
                    {"task_id": 0, "episode_idx": index, "success": success}
                    for index, success in enumerate(successes)
                ]
                (root / f"{backend}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )

            result = compare(root / "bf16.summary.json", root / "fp8.summary.json", False)
            self.assertEqual(result["paired_episodes"], 2)
            self.assertEqual(result["success_rate_delta_percentage_points"], 50.0)
            self.assertEqual(result["latency_speedup"], 2.0)
            self.assertEqual(result["pair_outcomes"]["fp8_only"], 1)


if __name__ == "__main__":
    unittest.main()
