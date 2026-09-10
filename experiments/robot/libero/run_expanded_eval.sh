#!/usr/bin/env bash
# Expanded LIBERO-Spatial BF16 vs FP8 paired eval (all tasks).
set -euo pipefail

PY=/home/bruce/miniconda3/envs/torch270_128/bin/python
REPO=/share_data/bruce/workspace/ai/openvla
CKPT=/share_data/huggingface/models/openvla-7b-finetuned-libero-spatial
BUILD=/home/bruce/TensorRT-Edge-LLM/build
TRT=/share_data/bruce/software/TensorRT-11.2.1.2
LOGDIR="$REPO/experiments/logs/libero_deploy"
RUN_TAG="${RUN_TAG:-spatial-expanded-seed7}"
TRIALS="${TRIALS:-10}"

export PYTHONUNBUFFERED=1
export PATH="/home/bruce/miniconda3/envs/torch270_128/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$BUILD:$TRT/lib:/usr/local/cuda/lib64"
export PYTHONPATH="$REPO"
export OPENVLA_PREFIX="$REPO"
export OPENVLA_MODEL_ID="$CKPT"
export OPENVLA_UNNORM_KEY=libero_spatial
export EDGE_LLM_DIR=/home/bruce/TensorRT-Edge-LLM
export EDGELLM_PLUGIN_PATH="$BUILD/libNvInfer_edgellm_plugin.so"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$REPO"
# shellcheck disable=SC1091
source env_gpu.sh 2>/dev/null || true

COMMON=(
  --checkpoint "$CKPT"
  --task-suite-name libero_spatial
  --task-ids all
  --num-trials-per-task "$TRIALS"
  --preprocessing portable
  --local-files-only
  --device cuda:0
  --seed 7
  --env-seed 0
)

echo "[$(date -Is)] Expanded LIBERO eval: all tasks x ${TRIALS} trials | CUDA=${CUDA_VISIBLE_DEVICES}"

echo "[$(date -Is)] === BF16 ==="
"$PY" experiments/robot/libero/run_libero_deploy_eval.py \
  --backend bf16 \
  "${COMMON[@]}" \
  --attn-implementation sdpa \
  --run-name "${RUN_TAG}-bf16" \
  --resume

echo "[$(date -Is)] === FP8 ==="
"$PY" experiments/robot/libero/run_libero_deploy_eval.py \
  --backend fp8 \
  "${COMMON[@]}" \
  --vision-engine "$REPO/deploy/tensorrt/artifacts/engines/vision_projector_fp8.plan" \
  --llm-engine-dir "$REPO/deploy/tensorrt/artifacts/engines/openvla_llama_fp8" \
  --action-metadata "$REPO/deploy/tensorrt/artifacts/action_meta/action_meta.json" \
  --edge-llm-plugin "$BUILD/libNvInfer_edgellm_plugin.so" \
  --run-name "${RUN_TAG}-fp8" \
  --resume

echo "[$(date -Is)] === Compare ==="
"$PY" experiments/robot/libero/compare_libero_results.py \
  --bf16-summary "$LOGDIR/${RUN_TAG}-bf16.summary.json" \
  --fp8-summary "$LOGDIR/${RUN_TAG}-fp8.summary.json" \
  --output "$LOGDIR/${RUN_TAG}-bf16-vs-fp8.json"

echo "[$(date -Is)] Done. Compare: $LOGDIR/${RUN_TAG}-bf16-vs-fp8.json"
