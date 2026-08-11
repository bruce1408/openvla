#!/usr/bin/env python3
"""Extract the fine-tuned OpenVLA Llama weights for an Edge-LLM export probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from deploy.tensorrt.common import load_openvla, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "deploy/tensorrt/artifacts/hf_llama",
    )
    args = parser.parse_args()

    processor, model = load_openvla(args.device, "bf16")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.language_model.save_pretrained(args.output_dir, safe_serialization=True)
    processor.tokenizer.save_pretrained(args.output_dir)
    write_json(
        args.output_dir / "openvla_extraction.json",
        {
            "source_architecture": model.__class__.__name__,
            "language_model_architecture": model.language_model.__class__.__name__,
            "model_type": model.language_model.config.model_type,
            "effective_vocab_size": int(model.vocab_size),
            "padded_vocab_size": int(model.config.text_config.vocab_size),
            "warning": "Llama 2 is not explicitly listed in the Edge-LLM support matrix; export is a compatibility probe.",
        },
    )
    print(f"Extracted OpenVLA language model to {args.output_dir}")


if __name__ == "__main__":
    main()
