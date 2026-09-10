#!/usr/bin/env bash
# =============================================================================
# LIBERO 四个 benchmark suite 的共享配置。
#
# 用法（在其他脚本里 source）:
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   # shellcheck disable=SC1091
#   source "$SCRIPT_DIR/libero_suites.sh"
#   resolve_libero_suite "${SUITE:-spatial}"
#
# SUITE 可写短名: spatial | object | goal | 10
# 或完整名: libero_spatial | libero_object | libero_goal | libero_10
# =============================================================================

: "${REPO:=/share_data/bruce/workspace/ai/openvla}"
: "${MODEL_ROOT:=/share_data/huggingface/models}"

resolve_libero_suite() {
  local raw="${1:-spatial}"
  raw="${raw#libero_}"

  # 每次解析都重置 suite 相关变量，避免连续调用时沿用上一次值
  unset CKPT FP8_VISION_ENGINE_OVERRIDE FP8_LLM_ENGINE_DIR_OVERRIDE FP8_ACTION_META_OVERRIDE

  case "$raw" in
    spatial)
      SUITE_SHORT=spatial
      TASK_SUITE=libero_spatial
      CKPT="$MODEL_ROOT/openvla-7b-finetuned-libero-spatial"
      UNNORM_KEY=libero_spatial
      ;;
    object)
      SUITE_SHORT=object
      TASK_SUITE=libero_object
      CKPT="$MODEL_ROOT/openvla-7b-finetuned-libero-object"
      UNNORM_KEY=libero_object
      ;;
    goal)
      SUITE_SHORT=goal
      TASK_SUITE=libero_goal
      CKPT="$MODEL_ROOT/openvla-7b-finetuned-libero-goal"
      UNNORM_KEY=libero_goal
      ;;
    10)
      SUITE_SHORT=10
      TASK_SUITE=libero_10
      CKPT="$MODEL_ROOT/openvla-7b-finetuned-libero-10"
      UNNORM_KEY=libero_10
      ;;
    *)
      echo "Unknown LIBERO suite: $1 (use spatial|object|goal|10)" >&2
      return 1
      ;;
  esac

  NUM_TASKS=10
  TRIALS_PER_TASK="${TRIALS_PER_TASK:-50}"

  # 每个 suite 独立的 FP8 产物目录（避免互相覆盖）
  FP8_ARTIFACT_ROOT="$REPO/deploy/tensorrt/artifacts/suites/${SUITE_SHORT}"
  FP8_ONNX_DIR="$FP8_ARTIFACT_ROOT/onnx"
  FP8_ENGINE_DIR="$FP8_ARTIFACT_ROOT/engines"
  FP8_HF_LLAMA_DIR="$FP8_ARTIFACT_ROOT/hf_llama"
  FP8_LLM_ONNX_DIR="$FP8_ARTIFACT_ROOT/hf_llama_onnx_fp8"
  FP8_ACTION_META="$FP8_ARTIFACT_ROOT/action_meta/action_meta.json"
  FP8_VISION_ENGINE="$FP8_ENGINE_DIR/vision_projector_fp8.plan"
  FP8_LLM_ENGINE_DIR="$FP8_ENGINE_DIR/openvla_llama_fp8"

  # spatial 已有 legacy 产物时，默认仍指向旧路径（向后兼容）
  if [[ "$SUITE_SHORT" == "spatial" ]]; then
    local legacy_vision="$REPO/deploy/tensorrt/artifacts/engines/vision_projector_fp8.plan"
    local legacy_llm="$REPO/deploy/tensorrt/artifacts/engines/openvla_llama_fp8"
    local legacy_meta="$REPO/deploy/tensorrt/artifacts/action_meta/action_meta.json"
    if [[ -z "${FP8_VISION_ENGINE_OVERRIDE:-}" && -f "$legacy_vision" ]]; then
      FP8_VISION_ENGINE="$legacy_vision"
    fi
    if [[ -z "${FP8_LLM_ENGINE_DIR_OVERRIDE:-}" && -d "$legacy_llm" ]]; then
      FP8_LLM_ENGINE_DIR="$legacy_llm"
    fi
    if [[ -z "${FP8_ACTION_META_OVERRIDE:-}" && -f "$legacy_meta" ]]; then
      FP8_ACTION_META="$legacy_meta"
    fi
  fi

  FP8_VISION_ENGINE="${FP8_VISION_ENGINE_OVERRIDE:-$FP8_VISION_ENGINE}"
  FP8_LLM_ENGINE_DIR="${FP8_LLM_ENGINE_DIR_OVERRIDE:-$FP8_LLM_ENGINE_DIR}"
  FP8_ACTION_META="${FP8_ACTION_META_OVERRIDE:-$FP8_ACTION_META}"

  export SUITE_SHORT TASK_SUITE CKPT UNNORM_KEY NUM_TASKS TRIALS_PER_TASK
  export FP8_ARTIFACT_ROOT FP8_ONNX_DIR FP8_ENGINE_DIR FP8_HF_LLAMA_DIR
  export FP8_LLM_ONNX_DIR FP8_ACTION_META FP8_VISION_ENGINE FP8_LLM_ENGINE_DIR
}

official_bf16_run_tag() {
  local seed="${1:-7}"
  echo "official-${SUITE_SHORT}-seed${seed}-8gpu"
}

official_fp8_balanced_run_tag() {
  local seed="${1:-7}"
  echo "official-${SUITE_SHORT}-seed${seed}-8gpu-fp8-balanced"
}

official_compare_output() {
  local seed="${1:-7}"
  local logdir="${2:-$REPO/experiments/logs/libero_official}"
  echo "$logdir/$(official_bf16_run_tag "$seed")-bf16-vs-fp8-official.json"
}

fp8_artifacts_ready() {
  [[ -f "$FP8_VISION_ENGINE" ]] \
    && [[ -f "$FP8_LLM_ENGINE_DIR/llm.engine" ]] \
    && [[ -f "$FP8_ACTION_META" ]]
}

list_all_suite_short_names() {
  echo "spatial object goal 10"
}
