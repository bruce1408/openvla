#!/usr/bin/env bash
# 下载 OpenVLA LIBERO 评测用 checkpoint（支持 HuggingFace 镜像与断点续传）
#
# 用法:
#   # 使用镜像（国内推荐）
#   export HF_ENDPOINT=https://hf-mirror.com
#   bash scripts/download_openvla_eval_models.sh
#
#   # 只下载 spatial 套件对应模型
#   bash scripts/download_openvla_eval_models.sh --models spatial
#
#   # 指定输出目录
#   bash scripts/download_openvla_eval_models.sh --output-dir /path/to/checkpoints

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENVLA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OPENVLA_ROOT}/checkpoints"
MODELS="all"

# repo_id -> 本地目录名
declare -A MODEL_MAP=(
    ["spatial"]="openvla/openvla-7b-finetuned-libero-spatial"
    ["object"]="openvla/openvla-7b-finetuned-libero-object"
    ["goal"]="openvla/openvla-7b-finetuned-libero-goal"
    ["10"]="openvla/openvla-7b-finetuned-libero-10"
    ["base"]="openvla/openvla-7b"
)

usage() {
    cat <<'EOF'
下载 OpenVLA 评测模型

选项:
  --models LIST     要下载的模型，逗号分隔
                    可选: spatial, object, goal, 10, base, all
                    默认: all（4 个 LIBERO 微调模型，不含 base）
  --output-dir DIR  保存目录，默认: ./checkpoints
  --with-base       与 --models all 一起用时，额外下载 openvla-7b 基座
  -h, --help        显示帮助

环境变量:
  HF_ENDPOINT       HuggingFace 镜像，例如 https://hf-mirror.com
  HF_TOKEN          可选，gated 模型或提高下载限速时使用

示例:
  export HF_ENDPOINT=https://hf-mirror.com
  bash scripts/download_openvla_eval_models.sh --models spatial
  bash scripts/download_openvla_eval_models.sh --models all --with-base
EOF
}

WITH_BASE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            MODELS="${2:-all}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="${2:?missing path}"
            shift 2
            ;;
        --with-base)
            WITH_BASE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "未知参数: $1" >&2
            usage
            exit 1
            ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"

if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "[错误] 未找到 hf / huggingface-cli，请先安装: pip install huggingface_hub" >&2
    exit 1
fi

HF_CMD="hf"
if ! command -v hf >/dev/null 2>&1; then
    HF_CMD="huggingface-cli"
fi

# 解析要下载的 key 列表
if [[ "${MODELS}" == "all" ]]; then
    SELECTED=(spatial object goal 10)
    [[ "${WITH_BASE}" == "true" ]] && SELECTED+=(base)
else
    IFS=',' read -r -a SELECTED <<< "${MODELS}"
fi

check_download_complete() {
    local dir="$1"
    local shard_count=0

    [[ -f "${dir}/config.json" ]] || return 1

    # 单文件权重
    if [[ -f "${dir}/model.safetensors" || -f "${dir}/pytorch_model.bin" ]]; then
        return 0
    fi

    # 分片权重：仅有 index.json 不够，必须存在实际 shard 文件
    if [[ -f "${dir}/model.safetensors.index.json" ]]; then
        shard_count="$(find "${dir}" -maxdepth 1 -name 'model-*-of-*.safetensors' 2>/dev/null | wc -l)"
        [[ "${shard_count}" -gt 0 ]] && return 0
    fi

    return 1
}

echo "=============================================="
echo " OpenVLA 评测模型下载"
echo " 输出目录: ${OUTPUT_DIR}"
echo " HF_ENDPOINT: ${HF_ENDPOINT:-（未设置，使用 huggingface.co）}"
echo "=============================================="

for key in "${SELECTED[@]}"; do
    key="$(echo "${key}" | xargs)"  # trim whitespace
    repo_id="${MODEL_MAP[${key}]:-}"
    if [[ -z "${repo_id}" ]]; then
        echo "[警告] 未知模型 key: ${key}，跳过（可选: spatial, object, goal, 10, base）" >&2
        continue
    fi

    local_name="$(basename "${repo_id}")"
    target_dir="${OUTPUT_DIR}/${local_name}"

    if check_download_complete "${target_dir}"; then
        echo "[跳过] ${repo_id} 已存在于 ${target_dir}"
        continue
    fi

    echo ""
    echo "[下载] ${repo_id}"
    echo "       -> ${target_dir}"
    echo "       （单个模型约 14–16 GB，请耐心等待，支持断点续传）"

    # hf download 默认支持断点续传；新版 CLI 已移除 --local-dir-use-symlinks
    ${HF_CMD} download "${repo_id}" \
        --local-dir "${target_dir}"

    if check_download_complete "${target_dir}"; then
        echo "[完成] ${local_name}"
    else
        echo "[警告] ${local_name} 下载后校验未通过，请检查网络后重新运行本脚本（会自动续传）" >&2
    fi
done

echo ""
echo "=============================================="
echo " 下载任务结束。本地路径示例:"
for key in "${SELECTED[@]}"; do
    key="$(echo "${key}" | xargs)"
    repo_id="${MODEL_MAP[${key}]:-}"
    [[ -n "${repo_id}" ]] && echo "   ${OUTPUT_DIR}/$(basename "${repo_id}")"
done
echo ""
echo " 量化评测示例（spatial 套件，4-bit）:"
echo "   conda activate openvla5090"
echo "   cd ${OPENVLA_ROOT}"
echo "   bash scripts/run_libero_quantized_benchmark.sh --suite spatial --quant 4bit"
echo "=============================================="
