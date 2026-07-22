export OPENVLA_PREFIX="/workspace/openvla"
export OPENVLA_MODEL_ID="${OPENVLA_MODEL_ID:-openvla/openvla-7b}"
export OPENVLA_DEVICE="${OPENVLA_DEVICE:-cuda:0}"
export OPENVLA_ATTN_IMPLEMENTATION="${OPENVLA_ATTN_IMPLEMENTATION:-sdpa}"
export OPENVLA_UNNORM_KEY="${OPENVLA_UNNORM_KEY:-bridge_orig}"
export HF_HOME="/workspace/checkpoints/openvla/hf_cache"
export TRANSFORMERS_CACHE="/workspace/checkpoints/openvla/hf_cache/transformers"
# 新版 huggingface_hub 只认 HF_HUB_CACHE(默认 HF_HOME/hub,此处为空);
# 模型实际缓存在 transformers 子目录,故显式指向它,否则离线加载会报 LocalEntryNotFoundError。
export HF_HUB_CACHE="/workspace/checkpoints/openvla/hf_cache/transformers"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export OPENVLA_REVISION="47a0ec7fc4ec123775a391911046cf33cf9ed83f"