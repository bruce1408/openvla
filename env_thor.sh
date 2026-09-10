export OPENVLA_PREFIX="/workspace/openvla"
# 本地 snapshot 优先（离线环境；若 shell 里已有 OPENVLA_MODEL_ID=openvla/openvla-7b 也强制覆盖）
LOCAL_OPENVLA_MODEL="/data/checkpoints/openvla/hf_cache/hf_cache/transformers/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f"
if [[ -d "${LOCAL_OPENVLA_MODEL}" ]]; then
  export OPENVLA_MODEL_ID="${LOCAL_OPENVLA_MODEL}"
else
  export OPENVLA_MODEL_ID="${OPENVLA_MODEL_ID:-openvla/openvla-7b}"
fi
export OPENVLA_DEVICE="${OPENVLA_DEVICE:-cuda:0}"
export OPENVLA_ATTN_IMPLEMENTATION="${OPENVLA_ATTN_IMPLEMENTATION:-sdpa}"
export OPENVLA_UNNORM_KEY="${OPENVLA_UNNORM_KEY:-bridge_orig}"
# modules 缓存需可写；权重只读挂载在 /data
export HF_HOME="/workspace/outputs/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export OPENVLA_REVISION="47a0ec7fc4ec123775a391911046cf33cf9ed83f"
