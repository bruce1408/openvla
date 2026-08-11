#!/usr/bin/env bash
# =============================================================================
# 14_nvfp4_collect_all.sh - NVFP4 采集编排 (文档 §1.4/§3/§5 数据源)
#
# 复用 FP8 流程,为 NVFP4 LLM engine 采集三类数据,供回填
#   "OpenVLA 模型评测（NVFP4 + TensorRT）.md" 的 [待采集] 章节:
#     1. Vision trace + LLM profile/csv + 逐层精度  ->  run 09 --precision nvfp4
#     2. NVFP4 7-DoF 动作精度 (vs BF16 golden)       ->  run 14_nvfp4_action_accuracy.py
#     3. LLM kernel 级 nsys + 算子类别              ->  nsys + run 10_operator_categories.py
#
# 用法:
#   bash deploy/tensorrt/pipeline/nvfp4/14_nvfp4_collect_all.sh
# 可调:
#   TAG       输出 tag (默认时间戳)
#   WARMUP/ACTIVE  09 的 warmup/active 次数
#   SKIP_09 / SKIP_ACC / SKIP_NSYS  跳过对应阶段 (默认全跑)
# =============================================================================
set -uo pipefail

OPENVLA_DIR="${OPENVLA_DIR:-/workspace/openvla}"
EDGE_LLM_DIR="${EDGE_LLM_DIR:-/workspace/TensorRT-Edge-LLM}"
LOGS_DIR="${OPENVLA_LOGS_DIR:-/workspace/outputs/openvla}"
PIPELINE_FP8="$OPENVLA_DIR/deploy/tensorrt/pipeline/fp8"
PIPELINE_NVFP4="$OPENVLA_DIR/deploy/tensorrt/pipeline/nvfp4"
ARTIFACTS="$OPENVLA_DIR/deploy/tensorrt/artifacts"
NVFP4_ENGINE="$ARTIFACTS/engines/openvla_llama_nvfp4"
NSYS="/usr/local/cuda/bin/nsys"

TAG="${TAG:-$(date +%Y_%m%d_%H%M%S)}"
WARMUP="${WARMUP:-2}"; ACTIVE="${ACTIVE:-5}"
export EDGE_LLM_DIR OPENVLA_DIR OPENVLA_LOGS_DIR=$LOGS_DIR

export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export EDGELLM_PLUGIN_PATH="$EDGE_LLM_DIR/build/libNvInfer_edgellm_plugin.so"

echo "[NVFP4 collect] tag=$TAG"
mkdir -p "$LOGS_DIR"

# ---- 阶段 1: 09 trace + 逐层精度 ----
if [[ "${SKIP_09:-0}" != "1" ]]; then
  echo; echo "===== [1/3] 09 trace + verify-precision (nvfp4) ====="
  python "$PIPELINE_FP8/09_prof_trace_e2e.py" \
    --precision nvfp4 --warmup "$WARMUP" --active "$ACTIVE" \
    --tag "$TAG" --verify-precision --output-dir "$LOGS_DIR"
  # 09 用内部 tag;把产物名统一到 $TAG 便于下游 (09 已在文件名嵌入 tag)
fi

# ---- 阶段 2: NVFP4 动作精度 (vs BF16 golden) ----
if [[ "${SKIP_ACC:-0}" != "1" ]]; then
  echo; echo "===== [2/3] NVFP4 7-DoF 动作精度 ====="
  python "$PIPELINE_NVFP4/14_nvfp4_action_accuracy.py" --mode textcheck --output "$LOGS_DIR/nvfp4_action_accuracy_${TAG}_textcheck.json"
  python "$PIPELINE_NVFP4/14_nvfp4_action_accuracy.py" --mode action --output "$LOGS_DIR/nvfp4_action_accuracy_${TAG}_action.json"
  python "$PIPELINE_NVFP4/14_nvfp4_action_accuracy.py" --mode golden-emb --output "$LOGS_DIR/nvfp4_action_accuracy_${TAG}_golden-emb.json"
fi

# ---- 阶段 3: LLM kernel 级 nsys + 算子类别 ----
if [[ "${SKIP_NSYS:-0}" != "1" ]]; then
  echo; echo "===== [3/3] LLM kernel 级 nsys + 算子类别 ====="
  REP="$LOGS_DIR/nvfp4_nsys_${TAG}"
  "$NSYS" profile -t cuda,nvtx -o "$REP" --force-overwrite true \
    "$EDGE_LLM_DIR/build/examples/llm/llm_inference" \
    --engineDir "$NVFP4_ENGINE" \
    --inputFile "$ARTIFACTS/smoke_input.json" \
    --outputFile /tmp/nvfp4_nsys_out.json \
    --warmup 3
  python "$PIPELINE_FP8/10_operator_categories.py" \
    --llm-nsys "${REP}.nsys-rep" \
    --tag "nvfp4_${TAG}" \
    --output "$LOGS_DIR/nvfp4_operator_categories_${TAG}.json"
fi

echo; echo "[NVFP4 collect] done. tag=$TAG"
