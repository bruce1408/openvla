# 入口：按机器选择 env_thor.sh 或 env_gpu.sh，现有 `source env.sh` / runtime_env.py 不用改。
#
# 判定顺序:
#   1) 手动指定  export OPENVLA_ENV=thor|gpu
#   2) Thor:     存在 /etc/nv_tegra_release（Jetson）
#   3) GPU 机:   存在 /share_data/huggingface
#   4) 默认 thor（保持原行为）
#
# 覆盖示例:
#   OPENVLA_ENV=gpu source env.sh

_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

_detect_openvla_env() {
  if [[ -n "${OPENVLA_ENV}" ]]; then
    echo "${OPENVLA_ENV}"
    return
  fi
  if [[ -f /etc/nv_tegra_release ]]; then
    echo thor
    return
  fi
  if [[ -d /share_data/huggingface ]]; then
    echo gpu
    return
  fi
  echo thor
}

OPENVLA_ENV="$(_detect_openvla_env)"
export OPENVLA_ENV

case "${OPENVLA_ENV}" in
  thor)
    # shellcheck source=/dev/null
    source "${_ENV_DIR}/env_thor.sh"
    ;;
  gpu)
    # shellcheck source=/dev/null
    source "${_ENV_DIR}/env_gpu.sh"
    ;;
  *)
    echo "Unknown OPENVLA_ENV=${OPENVLA_ENV}; expected thor or gpu" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac
