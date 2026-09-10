#!/usr/bin/env python3
"""Compare OpenVLA prompt token ids: HF harness vs C++-style Edge-LLM encode.

Does NOT run the engine. Goal: decide whether textcheck mismatch is
tokenization, before blaming harness KV/RoPE.

  python 05b_compare_prompt_tokens.py
  python 05b_compare_prompt_tokens.py --cpp-ids 512,29901,...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from deploy.tensorrt.common import prompt_for  # noqa: E402

ARTIFACTS = REPO_ROOT / "deploy/tensorrt/artifacts"
ENGINE_TOK = ARTIFACTS / "engines/openvla_llama_fp8"
GOLDEN_IDS = ARTIFACTS / "golden/sample_0001/input_ids.npy"
SMOKE = ARTIFACTS / "smoke_input.json"
DEFAULT_OUT = ARTIFACTS / "validate/prompt_token_compare.json"


def _ids(tokenizer, text: str, add_special: bool) -> list[int]:
    return tokenizer(text, add_special_tokens=add_special)["input_ids"]


def _simulate_cpp_no_metaspace(text: str) -> list[int] | None:
    """Approximate Edge-LLM encode(..., false) when Metaspace is ignored.

    C++ only implements Split/Regex pretokenizers. Llama tokenizer.json uses
    Metaspace, so Edge-LLM falls back to an empty Sequence (pass-through) and
    BPE-encodes the raw string with no BOS post-processor.
    """
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(ENGINE_TOK / "tokenizer.json"))
    except Exception:
        return None
    tok.pre_tokenizer = None
    tok.post_processor = None
    return list(tok.encode(text).ids)


def _hf_tokenizers_full(text: str) -> list[int] | None:
    """Engine tokenizer.json via HuggingFace tokenizers (Metaspace + BOS)."""
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(ENGINE_TOK / "tokenizer.json"))
    except Exception:
        return None
    return list(tok.encode(text).ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--cpp-ids",
        default="",
        help="Comma-separated ids from a C++ Tokenizer::encode dump, if you have them",
    )
    args = parser.parse_args()

    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
    text = smoke["requests"][0]["messages"][0]["content"]
    expected = prompt_for("pick up the blue object")

    import os

    from transformers import AutoTokenizer

    tok_dir = str(ENGINE_TOK)
    try:
        hf = AutoTokenizer.from_pretrained(tok_dir, use_fast=True, local_files_only=True)
    except Exception:
        tok_dir = os.environ.get(
            "OPENVLA_MODEL_ID",
            "/share_data/huggingface/models/openvla-7b-finetuned-libero-spatial",
        )
        hf = AutoTokenizer.from_pretrained(tok_dir, use_fast=True, local_files_only=True)
    harness = _ids(hf, text, True)
    no_bos = _ids(hf, text, False)
    tokenizers_full = _hf_tokenizers_full(text)
    cpp_sim = _simulate_cpp_no_metaspace(text)
    cpp_real = [int(x) for x in args.cpp_ids.split(",") if x.strip()] or None

    golden = None
    if GOLDEN_IDS.is_file():
        import numpy as np

        golden = np.load(GOLDEN_IDS).reshape(-1).astype(int).tolist()

    def same(a, b) -> bool:
        return a is not None and b is not None and list(a) == list(b)

    result = {
        "text": text,
        "text_matches_prompt_for": text == expected,
        "harness_add_special_true": harness,
        "hf_add_special_false": no_bos,
        "tokenizers_json_full": tokenizers_full,
        "cpp_simulated_no_metaspace_no_bos": cpp_sim,
        "cpp_real_encode": cpp_real,
        "golden_input_ids": golden,
        "comparisons": {
            "harness_eq_tokenizers_full": same(harness, tokenizers_full),
            "harness_eq_hf_no_bos": same(harness, no_bos),
            "harness_eq_cpp_sim": same(harness, cpp_sim),
            "hf_no_bos_eq_cpp_sim": same(no_bos, cpp_sim),
            "harness_eq_cpp_real": same(harness, cpp_real) if cpp_real else None,
            "hf_no_bos_eq_cpp_real": same(no_bos, cpp_real) if cpp_real else None,
            "cpp_sim_eq_cpp_real": same(cpp_sim, cpp_real) if cpp_real else None,
            "golden_starts_with_harness": (
                golden[: len(harness)] == harness if golden else None
            ),
        },
        "notes": [
            "Harness textcheck uses AutoTokenizer(..., add_special_tokens=True) → BOS id 1.",
            "Edge-LLM llm_inference calls encode(text, /*addBos=*/false) and does not run the HF post_processor.",
            "Llama tokenizer.json pre_tokenizer is Metaspace; Edge-LLM only handles Split/Regex → empty Sequence.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
