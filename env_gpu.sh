# 5090 / 共享 NFS 环境。由 env.sh 自动选中，也可手动: source env_gpu.sh
export OPENVLA_PREFIX="${OPENVLA_PREFIX:-/share_data/bruce/workspace/ai/openvla}"

SHARED_OPENVLA_MODEL="/share_data/huggingface/models/openvla-7b-finetuned-libero-spatial"
if [[ -d "${SHARED_OPENVLA_MODEL}" ]]; then
  export OPENVLA_MODEL_ID="${OPENVLA_MODEL_ID:-${SHARED_OPENVLA_MODEL}}"
else
  export OPENVLA_MODEL_ID="${OPENVLA_MODEL_ID:-openvla/openvla-7b}"
fi

export OPENVLA_DEVICE="${OPENVLA_DEVICE:-cuda:0}"
export OPENVLA_ATTN_IMPLEMENTATION="${OPENVLA_ATTN_IMPLEMENTATION:-sdpa}"
export OPENVLA_UNNORM_KEY="${OPENVLA_UNNORM_KEY:-libero_spatial}"

SHARED_HF_HOME="/share_data/huggingface/hf_cache"
if [[ -z "${HF_HOME}" ]]; then
  if [[ -d "${SHARED_HF_HOME}" ]]; then
    export HF_HOME="${SHARED_HF_HOME}"
  else
    export HF_HOME="${HOME}/.cache/huggingface"
  fi
fi
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export OPENVLA_REVISION="47a0ec7fc4ec123775a391911046cf33cf9ed83f"

# TensorRT / Edge-LLM FP8 流水线依赖（llm_build 加载 AttentionPlugin 等自定义插件）
export EDGELLM_PLUGIN_PATH="${EDGELLM_PLUGIN_PATH:-/home/bruce/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so}"
export LD_LIBRARY_PATH="/home/bruce/TensorRT-Edge-LLM/build:/share_data/bruce/software/TensorRT-11.2.1.2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 换 LIBERO 套件时改 env.local.sh，不必改本文件
_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
if [[ -f "${_ENV_DIR}/env.local.sh" ]]; then
  # shellcheck source=/dev/null
  source "${_ENV_DIR}/env.local.sh"
fi
