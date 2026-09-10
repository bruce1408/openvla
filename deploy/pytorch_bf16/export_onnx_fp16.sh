#!/usr/bin/env bash
# Export OpenVLA to FP16 ONNX: vision+projector (PyTorch) + LLM (Edge-LLM).
#
# OpenVLA 无法用一个 ONNX 文件表示完整动作推理图，标准拆分为：
#   1. vision_projector_fp16.onnx     — DINOv2 + SigLIP + projector
#   2. hf_llama_onnx_fp16/llm/        — Llama LLM（model.onnx + embedding.safetensors 等）
#
# Usage:
#   bash deploy/pytorch_bf16/export_onnx_fp16.sh              # vision + LLM
#   bash deploy/pytorch_bf16/export_onnx_fp16.sh vision       # 仅 vision
#   bash deploy/pytorch_bf16/export_onnx_fp16.sh llm          # 仅 LLM
#   SKIP_VISION=1 bash ...                                    # 跳过 vision
#   FORCE=1 bash ...                                          # 覆盖已有产物
#
# Env:
#   PY              torch270_128 python（vision 导出 / 权重提取）
#   EDGELLM_BIN     Edge-LLM 导出 CLI（默认 torch212 环境）
#   DEVICE          vision 导出 GPU，默认 cuda:0
#   ARTIFACTS_DIR   产物根目录，默认 deploy/pytorch_bf16/artifacts

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/home/bruce/miniconda3/envs/torch270_128/bin/python}"
EDGELLM_BIN="${EDGELLM_BIN:-/home/bruce/miniconda3/envs/torch212/bin/tensorrt-edgellm-export}"
DEVICE="${DEVICE:-cuda:0}"
MODE="${MODE:-combined}"
FORCE="${FORCE:-0}"
STAGE="${1:-all}"

ARTIFACTS_DIR="${ARTIFACTS_DIR:-$REPO_ROOT/deploy/pytorch_bf16/artifacts}"
VISION_ONNX="${VISION_ONNX:-$ARTIFACTS_DIR/onnx/vision_projector_fp16.onnx}"
HF_LLAMA_DIR="${HF_LLAMA_DIR:-$ARTIFACTS_DIR/hf_llama}"
LLM_ONNX_DIR="${LLM_ONNX_DIR:-$ARTIFACTS_DIR/hf_llama_onnx_fp16}"
LOG_DIR="${LOG_DIR:-$ARTIFACTS_DIR/logs}"

export PYTHONPATH="$REPO_ROOT"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"
# shellcheck disable=SC1091
source env_gpu.sh 2>/dev/null || true

mkdir -p "$ARTIFACTS_DIR/onnx" "$LOG_DIR"

log() { echo "[$(date -Is)] $*"; }

should_run_vision() {
  [[ "${SKIP_VISION:-0}" == "1" ]] && return 1
  [[ "$STAGE" == "all" || "$STAGE" == "vision" ]]
}

should_run_llm() {
  [[ "${SKIP_LLM:-0}" == "1" ]] && return 1
  [[ "$STAGE" == "all" || "$STAGE" == "llm" ]]
}

export_vision() {
  if [[ -f "$VISION_ONNX" && "$FORCE" != "1" ]]; then
    log "skip vision export (exists): $VISION_ONNX  (FORCE=1 to overwrite)"
    return 0
  fi
  log "=== [1/3] Vision + Projector -> FP16 ONNX ==="
  VERIFY_FLAG=()
  if [[ "${VERIFY:-0}" == "1" ]]; then
    VERIFY_FLAG=(--verify)
  fi
  "$PY" deploy/pytorch_bf16/export_onnx_fp16.py \
    --device "$DEVICE" \
    --mode "$MODE" \
    --output "$VISION_ONNX" \
    "${VERIFY_FLAG[@]}"
  log "vision ONNX: $VISION_ONNX"
}

extract_llm_checkpoint() {
  if [[ -f "$HF_LLAMA_DIR/config.json" && "$FORCE" != "1" ]]; then
    log "skip LLM extract (exists): $HF_LLAMA_DIR  (FORCE=1 to overwrite)"
    return 0
  fi
  log "=== [2/3] Extract Llama weights from OpenVLA ==="
  "$PY" deploy/tensorrt/pipeline/fp16/02_extract_llama_checkpoint.py \
    --device cpu \
    --output-dir "$HF_LLAMA_DIR" \
    2>&1 | tee "$LOG_DIR/extract_llama.log"
  [[ -f "$HF_LLAMA_DIR/config.json" ]] || {
    echo "LLM extract failed: missing $HF_LLAMA_DIR/config.json" >&2
    exit 1
  }
  log "hf_llama checkpoint: $HF_LLAMA_DIR"
}

ensure_chat_template() {
  local llm_dir="$1"
  local template_path="$llm_dir/processed_chat_template.json"
  if [[ -f "$template_path" ]]; then
    return 0
  fi
  log "writing fallback processed_chat_template.json"
  "$PY" - "$HF_LLAMA_DIR" "$template_path" <<'PY'
import json
import sys

fallback = {
    "model_path": sys.argv[1],
    "roles": {
        "system": {"prefix": "", "suffix": "\n"},
        "user": {"prefix": "User: ", "suffix": "\n"},
        "assistant": {"prefix": "Assistant: ", "suffix": "\n"},
    },
    "content_types": {},
    "generation_prompt": "Assistant: ",
    "default_system_prompt": "",
}
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(fallback, f, indent=2)
PY
}

export_llm_onnx() {
  local llm_out="$LLM_ONNX_DIR/llm"
  if [[ -f "$llm_out/model.onnx" && -f "$llm_out/embedding.safetensors" && "$FORCE" != "1" ]]; then
    log "skip LLM ONNX export (exists): $llm_out  (FORCE=1 to overwrite)"
    return 0
  fi
  if [[ ! -x "$EDGELLM_BIN" && ! -f "$EDGELLM_BIN" ]]; then
    echo "Edge-LLM export CLI not found: $EDGELLM_BIN" >&2
    echo "Install TensorRT-Edge-LLM in torch212, or set EDGELLM_BIN=/path/to/tensorrt-edgellm-export" >&2
    exit 1
  fi
  log "=== [3/3] LLM -> FP16 ONNX (Edge-LLM) ==="
  log "using: $EDGELLM_BIN"
  mkdir -p "$LLM_ONNX_DIR"
  "$EDGELLM_BIN" "$HF_LLAMA_DIR" "$LLM_ONNX_DIR" \
    2>&1 | tee "$LOG_DIR/edgellm_export_fp16.log"
  for f in config.json model.onnx model.onnx.data embedding.safetensors tokenizer.json; do
    if [[ ! -f "$llm_out/$f" ]]; then
      echo "LLM ONNX export failed: missing $llm_out/$f" >&2
      exit 1
    fi
  done
  ensure_chat_template "$llm_out"
  log "LLM ONNX: $llm_out"
}

write_manifest() {
  local manifest_path="$ARTIFACTS_DIR/onnx_export_manifest.json"
  "$PY" - "$manifest_path" "$VISION_ONNX" "$HF_LLAMA_DIR" "$LLM_ONNX_DIR" <<'PY'
import json
import sys
from pathlib import Path

manifest_path, vision_onnx, hf_llama, llm_onnx = sys.argv[1:5]
llm_dir = Path(llm_onnx) / "llm"
payload = {
    "precision": "fp16",
    "vision_onnx": vision_onnx,
    "vision_manifest": str(Path(vision_onnx).with_suffix(".manifest.json")),
    "hf_llama_checkpoint": hf_llama,
    "llm_onnx_dir": str(llm_dir),
    "llm_files": {
        name: str(llm_dir / name)
        for name in (
            "config.json",
            "model.onnx",
            "model.onnx.data",
            "embedding.safetensors",
            "tokenizer.json",
            "processed_chat_template.json",
        )
        if (llm_dir / name).is_file()
    },
}
Path(manifest_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"manifest: {manifest_path}")
PY
}

case "$STAGE" in
  all | vision | llm) ;;
  *)
    echo "Unknown stage: $STAGE (use: all | vision | llm)" >&2
    exit 2
    ;;
esac

if should_run_vision; then
  export_vision
fi

if should_run_llm; then
  extract_llm_checkpoint
  export_llm_onnx
fi

if should_run_vision || should_run_llm; then
  write_manifest
fi

log "export_onnx_fp16 done (stage=$STAGE)"
if should_run_vision; then
  log "  vision: $VISION_ONNX"
fi
if should_run_llm; then
  log "  llm:    $LLM_ONNX_DIR/llm/model.onnx"
fi
