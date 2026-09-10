#!/usr/bin/env bash
# =============================================================================
# 为指定 LIBERO suite 构建 FP8 TensorRT 产物（vision + LLM + action_meta）
#
# 每个 suite 写入独立目录，避免覆盖:
#   deploy/tensorrt/artifacts/suites/{spatial,object,goal,10}/
#
# 用法:
#   cd /share_data/bruce/workspace/ai/openvla && source env_gpu.sh
#   SUITE=object bash experiments/robot/libero/official_repro/build_fp8_artifacts_for_suite.sh
#
# 分阶段（省时间/debug）:
#   STAGE=vision SUITE=goal bash ...
#   STAGE=llm SUITE=goal bash ...
#   STAGE=action_meta SUITE=goal bash ...
#   STAGE=all SUITE=goal bash ...    # 默认
#
# 可选:
#   FORCE=1              覆盖已有产物
#   SKIP_LLM_BUILD=1     只 export/quantize ONNX，不跑 llm_build（需手动构建 engine）
#   DEVICE=cuda:7        整条流水线用的物理 GPU（会写入 CUDA_VISIBLE_DEVICES）
#   EDGELLM_BIN=...      Edge-LLM CLI（默认 torch212 环境）
# =============================================================================
set -euo pipefail

: "${SUITE:=spatial}"
: "${STAGE:=all}"
: "${FORCE:=0}"
: "${SKIP_LLM_BUILD:=0}"
: "${PY:=/home/bruce/miniconda3/envs/torch270_128/bin/python}"
: "${REPO:=/share_data/bruce/workspace/ai/openvla}"
: "${BUILD:=/home/bruce/TensorRT-Edge-LLM/build}"
: "${DEVICE:=cuda:0}"
: "${EDGELLM_BIN:=/home/bruce/miniconda3/envs/torch212/bin/tensorrt-edgellm-export}"
: "${EDGELLM_QUANTIZE_BIN:=/home/bruce/miniconda3/envs/torch212/bin/tensorrt-edgellm-quantize}"
: "${LLM_BUILD:=${BUILD}/examples/llm/llm_build}"
: "${NUM_SAMPLES:=512}"
: "${CALIB_DATASET:=/share_data/public/openvla_models/calib_text.jsonl}"
: "${TRT:=/share_data/bruce/software/TensorRT-11.2.1.2}"

# llm_build 需要 Edge-LLM 插件库。默认按相对路径 build/libNvInfer_edgellm_plugin.so
# 查找，会因 CWD 不同而失败；这里显式指向 BUILD 目录。
export EDGELLM_PLUGIN_PATH="${EDGELLM_PLUGIN_PATH:-$BUILD/libNvInfer_edgellm_plugin.so}"
export LD_LIBRARY_PATH="$BUILD:$TRT/lib:${LD_LIBRARY_PATH:-}"

# DEVICE=cuda:N 只对认识 --device 的 Python 脚本生效。trtexec / Edge-LLM 默认仍用
# 物理 GPU 0。这里把可见卡锁成同一张，后续所有子进程里的 cuda:0 都是这张卡。
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && "$DEVICE" =~ ^cuda:([0-9]+)$ ]]; then
  export CUDA_VISIBLE_DEVICES="${BASH_REMATCH[1]}"
  DEVICE="cuda:0"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/libero_suites.sh"
resolve_libero_suite "$SUITE"

export PYTHONPATH="$REPO"
export OPENVLA_MODEL_ID="$CKPT"
export OPENVLA_UNNORM_KEY="$UNNORM_KEY"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

PIPELINE="$REPO/deploy/tensorrt/pipeline"
VISION_FP16_ONNX="$FP8_ONNX_DIR/vision_projector_fp16.onnx"
VISION_FP8_ONNX="$FP8_ONNX_DIR/vision_projector_fp8.onnx"
LOG_DIR="$FP8_ARTIFACT_ROOT/logs"

mkdir -p "$FP8_ONNX_DIR" "$FP8_ENGINE_DIR" "$FP8_ARTIFACT_ROOT/action_meta" "$LOG_DIR"

log() { echo "[$(date -Is)] [$SUITE_SHORT] $*"; }

should_run() {
  local name="$1"
  [[ "$STAGE" == "all" || "$STAGE" == "$name" ]]
}

stage_vision() {
  if [[ -f "$FP8_VISION_ENGINE" && "$FORCE" != "1" ]]; then
    log "skip vision engine (exists): $FP8_VISION_ENGINE"
    return 0
  fi

  if [[ ! -f "$VISION_FP16_ONNX" || "$FORCE" == "1" ]]; then
    log "export vision FP16 ONNX -> $VISION_FP16_ONNX"
    "$PY" "$PIPELINE/fp16/01_export_vision_projector_onnx.py" \
      --device "$DEVICE" \
      --output "$VISION_FP16_ONNX" \
      2>&1 | tee "$LOG_DIR/01_export_vision_fp16.log"
  fi

  if [[ ! -f "$VISION_FP8_ONNX" || "$FORCE" == "1" ]]; then
    log "quantize vision -> FP8 ONNX"
    "$PY" "$PIPELINE/fp8/01b_quantize_vision_fp8.py" \
      --device "$DEVICE" \
      --input-onnx "$VISION_FP16_ONNX" \
      --output-onnx "$VISION_FP8_ONNX" \
      2>&1 | tee "$LOG_DIR/01b_quantize_vision_fp8.log"
  fi

  log "build vision FP8 TensorRT engine"
  bash "$PIPELINE/fp16/05_build_vision_engine.sh" \
    "$VISION_FP8_ONNX" \
    "$FP8_VISION_ENGINE" \
    fp8 \
    2>&1 | tee "$LOG_DIR/05_build_vision_engine.log"
}

stage_llm() {
  if [[ -f "$FP8_LLM_ENGINE_DIR/llm.engine" && "$FORCE" != "1" ]]; then
    log "skip LLM engine (exists): $FP8_LLM_ENGINE_DIR/llm.engine"
    return 0
  fi

  if [[ ! -f "$FP8_HF_LLAMA_DIR/config.json" || "$FORCE" == "1" ]]; then
    log "extract Llama checkpoint from OpenVLA"
    "$PY" "$PIPELINE/fp16/02_extract_llama_checkpoint.py" \
      --device cpu \
      --output-dir "$FP8_HF_LLAMA_DIR" \
      2>&1 | tee "$LOG_DIR/02_extract_llama.log"
  fi

  local hf_fp8="$FP8_ARTIFACT_ROOT/hf_llama_fp8"
  local onnx_fp8="$FP8_LLM_ONNX_DIR"
  local llm_onnx="$onnx_fp8/llm"

  if [[ ! -f "$hf_fp8/config.json" || "$FORCE" == "1" ]]; then
    log "quantize LLM -> FP8 weights (Edge-LLM)"
    if [[ ! -x "$EDGELLM_QUANTIZE_BIN" && ! -f "$EDGELLM_QUANTIZE_BIN" ]]; then
      echo "tensorrt-edgellm-quantize not found: $EDGELLM_QUANTIZE_BIN" >&2
      exit 1
    fi
    "$EDGELLM_QUANTIZE_BIN" llm \
      --model_dir "$FP8_HF_LLAMA_DIR" \
      --output_dir "$hf_fp8" \
      --quantization fp8 \
      --device "$DEVICE" \
      --num_samples "$NUM_SAMPLES" \
      --dataset "$CALIB_DATASET" \
      2>&1 | tee "$LOG_DIR/edgellm_quantize_fp8.log"
  fi

  if [[ ! -f "$llm_onnx/model.onnx" || "$FORCE" == "1" ]]; then
    log "export LLM FP8 -> ONNX"
    if [[ ! -x "$EDGELLM_BIN" && ! -f "$EDGELLM_BIN" ]]; then
      echo "tensorrt-edgellm-export not found: $EDGELLM_BIN" >&2
      exit 1
    fi
    mkdir -p "$onnx_fp8"
    "$EDGELLM_BIN" "$hf_fp8" "$onnx_fp8" \
      2>&1 | tee "$LOG_DIR/edgellm_export_fp8.log"
  fi

  if [[ "$SKIP_LLM_BUILD" == "1" ]]; then
    log "SKIP_LLM_BUILD=1: ONNX ready at $llm_onnx; run llm_build manually"
    return 0
  fi

  if [[ ! -x "$LLM_BUILD" ]]; then
    echo "llm_build not found: $LLM_BUILD" >&2
    exit 1
  fi

  mkdir -p "$FP8_LLM_ENGINE_DIR"
  log "build LLM FP8 TensorRT engine -> $FP8_LLM_ENGINE_DIR"
  "$LLM_BUILD" \
    --onnxDir "$llm_onnx" \
    --engineDir "$FP8_LLM_ENGINE_DIR" \
    --maxBatchSize 1 \
    --maxInputLen 1024 \
    --maxKVCacheCapacity 1024 \
    2>&1 | tee "$LOG_DIR/llm_build_fp8.log"

  [[ -f "$FP8_LLM_ENGINE_DIR/llm.engine" ]] || {
    echo "LLM engine build failed: missing $FP8_LLM_ENGINE_DIR/llm.engine" >&2
    exit 1
  }
}

stage_action_meta() {
  if [[ -f "$FP8_ACTION_META" && "$FORCE" != "1" ]]; then
    log "skip action_meta (exists): $FP8_ACTION_META"
    return 0
  fi
  log "export action_meta.json (unnorm_key=$UNNORM_KEY)"
  "$PY" "$PIPELINE/fp16/04_export_action_params.py" \
    --unnorm-key "$UNNORM_KEY" \
    --output "$FP8_ACTION_META" \
    2>&1 | tee "$LOG_DIR/04_export_action_params.log"
}

cd "$REPO"

log "======== Build FP8 artifacts for suite=$TASK_SUITE ========"
log "checkpoint: $CKPT"
log "artifact root: $FP8_ARTIFACT_ROOT"
log "stage: $STAGE"
log "device: $DEVICE  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

if should_run vision; then stage_vision; fi
if should_run llm; then stage_llm; fi
if should_run action_meta; then stage_action_meta; fi

if fp8_artifacts_ready; then
  log "FP8 artifacts ready for $SUITE_SHORT"
  log "  vision: $FP8_VISION_ENGINE"
  log "  llm:    $FP8_LLM_ENGINE_DIR/llm.engine"
  log "  meta:   $FP8_ACTION_META"
else
  log "WARNING: artifacts incomplete after stage=$STAGE"
  exit 1
fi

log "done"
