#!/usr/bin/env python3
"""Export the deterministic action detokenization contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from transformers import AutoConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from deploy.tensorrt.common import DEFAULT_UNNORM_KEY, EMPTY_TOKEN_ID, write_json  # noqa: E402
from runtime_env import MODEL_PATH, MODEL_REVISION  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "deploy/tensorrt/artifacts/action_meta/action_meta.json",
    )
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(
        MODEL_PATH,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )
    if args.unnorm_key not in config.norm_stats:
        raise SystemExit(f"Unknown unnorm key {args.unnorm_key!r}; choices={sorted(config.norm_stats)}")
    stats = config.norm_stats[args.unnorm_key]["action"]
    effective_vocab_size = int(config.text_config.vocab_size - config.pad_to_multiple_of)
    n_action_tokens = int(config.n_action_bins)
    low_token, high_token = effective_vocab_size - n_action_tokens, effective_vocab_size - 1
    bins = np.linspace(-1, 1, n_action_tokens)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    action_dim = len(stats["q01"])
    value = {
        "unnorm_key": args.unnorm_key,
        "empty_token_id": EMPTY_TOKEN_ID,
        "action_dim": action_dim,
        "n_action_tokens": n_action_tokens,
        "n_bin_centers": int(bin_centers.shape[0]),
        "effective_vocab_size": effective_vocab_size,
        "padded_lm_head_size": int(config.text_config.vocab_size),
        "action_token_id_min": low_token,
        "action_token_id_max": high_token,
        "bin_centers": bin_centers.astype(float).tolist(),
        "q01": list(stats["q01"]),
        "q99": list(stats["q99"]),
        "mask": list(stats.get("mask", [True] * action_dim)),
    }
    write_json(args.output, value)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
