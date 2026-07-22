#!/usr/bin/env bash
# =============================================================================
# measure_llm_latency.sh - 测量 LLM engine 的 prefill/decode 时延（支持 nvfp4 / fp8）
# =============================================================================
#
# 用途:
#   用 Edge-LLM 的 llm_bench 分别测 LLM 的 prefill 和 decode 时延。
#   通过第一个参数选择精度 (nvfp4 或 fp8)。
#
# 用法:
#   bash measure_llm_latency.sh nvfp4
#   bash measure_llm_latency.sh fp8
#
# 可选环境变量:
#   INPUT_LEN   prefill 输入长度 (默认 262 = 256 视觉 + 6 文本)
#   ACTION_DIM  动作维度 = 生成 token 数 (默认 7)
#   ITERS       正式迭代次数 (默认 20)
#   WARMUP      warmup 次数 (默认 5)
# =============================================================================
set -euo pipefail

PRECISION="${1:-nvfp4}"
if [[ "$PRECISION" != "nvfp4" && "$PRECISION" != "fp8" ]]; then
  echo "用法: bash measure_llm_latency.sh [nvfp4|fp8]" >&2
  exit 1
fi

EDGE_LLM_DIR="${EDGE_LLM_DIR:-/workspace/TensorRT-Edge-LLM}"
OPENVLA_DIR="${OPENVLA_DIR:-/workspace/openvla}"
ENGINE_DIR="$OPENVLA_DIR/deploy/tensorrt/artifacts/engines/openvla_llama_${PRECISION}"

INPUT_LEN="${INPUT_LEN:-262}"
ACTION_DIM="${ACTION_DIM:-7}"
DECODE_STEPS=$(( ACTION_DIM - 1 ))
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-5}"

# Edge-LLM 运行时必需
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export EDGELLM_PLUGIN_PATH="$EDGE_LLM_DIR/build/libNvInfer_edgellm_plugin.so"
LLM_BENCH="$EDGE_LLM_DIR/build/examples/llm/llm_bench"

if [[ ! -f "$ENGINE_DIR/llm.engine" ]]; then
  echo "错误: 找不到 engine: $ENGINE_DIR/llm.engine" >&2
  echo "请先用 llm_build 构建 $PRECISION engine。" >&2
  exit 1
fi

echo "============================================================"
echo "LLM engine 时延基准  [精度: $PRECISION]"
echo "  engine     : $ENGINE_DIR"
echo "  prefill len: $INPUT_LEN | decode steps: $DECODE_STEPS | warmup/iters: $WARMUP/$ITERS"
echo "============================================================"

echo
echo ">>> [1/2] Prefill (inputLen=$INPUT_LEN)"
"$LLM_BENCH" --engineDir "$ENGINE_DIR" --mode prefill \
  --inputLen "$INPUT_LEN" --warmup "$WARMUP" --iterations "$ITERS" 2>&1 \
  | grep -viE "AttentionPlugin|FMHA supported" \
  | grep -iE "Prefill E2E Time|Tokens/sec|InputLen"

echo
echo ">>> [2/2] Decode (osl=$DECODE_STEPS, pastKVLen=$INPUT_LEN, CUDA graph ON)"
"$LLM_BENCH" --engineDir "$ENGINE_DIR" --mode decode \
  --pastKVLen "$INPUT_LEN" --osl "$DECODE_STEPS" --warmup "$WARMUP" --iterations "$ITERS" 2>&1 \
  | grep -viE "AttentionPlugin|FMHA supported" \
  | grep -iE "E2E Time|Per-step avg|Throughput|PastKVLen"

echo
echo "============================================================"
echo "完成 [$PRECISION]。decode 每 token 时延 x $DECODE_STEPS 是主要瓶颈。"
echo "============================================================"
