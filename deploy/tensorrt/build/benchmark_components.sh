#!/usr/bin/env bash
# =============================================================================
# OpenVLA on Thor - 三组件时延基准测试脚本 (NVFP4 LLM + FP16 Vision)
# =============================================================================
#
# 用途:
#   分别测量 OpenVLA 推理链路的三个组件在 Jetson AGX Thor 上的时延:
#     1. Vision engine  (DINOv2 + SigLIP + projector, FP16)   -> trtexec
#     2. LLM prefill     (256 视觉 token + 文本 token,      NVFP4) -> llm_bench
#     3. LLM decode      (逐 token 自回归生成动作 token,    NVFP4) -> llm_bench
#   最后把三段相加,估算端到端单步动作时延。
#
# 重要说明:
#   - 这是"组件相加"估算,vision 与 LLM 两个 engine 目前是独立测的,
#     尚未真正缝合(视觉 embedding 注入 LLM 是后续工作)。
#   - 所有测量都带 warmup,取多次迭代的稳定平均值。
#   - 不验证数值正确性,只测机械执行时延。
#
# 用法:
#   bash deploy/tensorrt/build/benchmark_components.sh
#
# 可调环境变量:
#   EDGE_LLM_DIR    Edge-LLM 仓库路径 (默认 /workspace/TensorRT-Edge-LLM)
#   LLM_ENGINE_DIR  LLM engine 目录  (默认 nvfp4 engine)
#   VISION_PLAN     Vision engine plan 路径
#   PROMPT_LEN      文本 prompt token 数 (默认 6)
#   N_VISION_TOKENS 视觉 token 数 (默认 256)
#   ACTION_DIM      动作维度 = 生成 token 数 (默认 7)
#   ITERS           正式迭代次数 (默认 20)
#   WARMUP          warmup 次数 (默认 5)
# =============================================================================

set -euo pipefail

# ---- 路径与参数 ----
EDGE_LLM_DIR="${EDGE_LLM_DIR:-/workspace/TensorRT-Edge-LLM}"
OPENVLA_DIR="${OPENVLA_DIR:-/workspace/openvla}"
LLM_ENGINE_DIR="${LLM_ENGINE_DIR:-$OPENVLA_DIR/deploy/tensorrt/artifacts/engines/openvla_llama_nvfp4}"
VISION_PLAN="${VISION_PLAN:-$OPENVLA_DIR/deploy/tensorrt/artifacts/engines/vision_projector_fp16.plan}"

PROMPT_LEN="${PROMPT_LEN:-6}"
N_VISION_TOKENS="${N_VISION_TOKENS:-256}"
ACTION_DIM="${ACTION_DIM:-7}"
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-5}"

# prefill 输入长度 = 视觉 token + 文本 token
INPUT_LEN=$(( N_VISION_TOKENS + PROMPT_LEN ))
# decode 步数 = 动作 token 数 - 1 (第 1 个 token 由 prefill 产生)
DECODE_STEPS=$(( ACTION_DIM - 1 ))

# ---- Edge-LLM 运行时必需的环境变量 ----
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export EDGELLM_PLUGIN_PATH="$EDGE_LLM_DIR/build/libNvInfer_edgellm_plugin.so"

TRTEXEC="/usr/src/tensorrt/bin/trtexec"
LLM_BENCH="$EDGE_LLM_DIR/build/examples/llm/llm_bench"

echo "============================================================"
echo "OpenVLA Thor 组件时延基准"
echo "============================================================"
echo "  LLM engine   : $LLM_ENGINE_DIR"
echo "  Vision plan  : $VISION_PLAN"
echo "  prefill 长度 : $INPUT_LEN ($N_VISION_TOKENS 视觉 + $PROMPT_LEN 文本)"
echo "  decode 步数  : $DECODE_STEPS (动作维度 $ACTION_DIM,prefill 出第 1 个)"
echo "  warmup/iters : $WARMUP / $ITERS"
echo "============================================================"

# ---- 组件 1: Vision engine (trtexec) ----
echo
echo ">>> [1/3] Vision engine (FP16, 1x6x224x224)"
"$TRTEXEC" \
  --loadEngine="$VISION_PLAN" \
  --warmUp=1000 \
  --iterations="$(( ITERS * 3 ))" \
  --avgRuns="$ITERS" 2>&1 \
  | grep -iE "^\[.*\] \[I\] (Latency|GPU Compute Time|Throughput):" || true

# ---- 组件 2: LLM prefill (llm_bench) ----
echo
echo ">>> [2/3] LLM prefill (NVFP4, inputLen=$INPUT_LEN)"
"$LLM_BENCH" \
  --engineDir "$LLM_ENGINE_DIR" \
  --mode prefill \
  --inputLen "$INPUT_LEN" \
  --warmup "$WARMUP" \
  --iterations "$ITERS" 2>&1 \
  | grep -viE "AttentionPlugin|FMHA supported" \
  | grep -iE "E2E Time|Tokens/sec|InputLen" || true

# ---- 组件 3: LLM decode (llm_bench, CUDA graph) ----
echo
echo ">>> [3/3] LLM decode (NVFP4, osl=$DECODE_STEPS, pastKVLen=$INPUT_LEN, CUDA graph ON)"
"$LLM_BENCH" \
  --engineDir "$LLM_ENGINE_DIR" \
  --mode decode \
  --pastKVLen "$INPUT_LEN" \
  --osl "$DECODE_STEPS" \
  --warmup "$WARMUP" \
  --iterations "$ITERS" 2>&1 \
  | grep -viE "AttentionPlugin|FMHA supported" \
  | grep -iE "E2E Time|Per-step avg|Throughput|PastKVLen" || true

echo
echo "============================================================"
echo "说明: 端到端单步时延 ≈ Vision + prefill + (decode_per_step * $DECODE_STEPS)"
echo "     decode 是主要瓶颈 (逐 token 串行,无 action chunking)"
echo "============================================================"
