#!/usr/bin/env bash
# 在 LIBERO 仿真中对比 OpenVLA 不同量化精度下的任务成功率
#
# 依赖:
#   - conda activate openvla5090
#   - pip install bitsandbytes   # 8bit/4bit 量化需要
#   - LIBERO 已安装且 ~/.libero/config.yaml 已配置
#
# 用法:
#   bash scripts/run_libero_quantized_benchmark.sh --suite spatial --quant bf16,8bit,4bit
#   bash scripts/run_libero_quantized_benchmark.sh --suite spatial --quant 4bit --trials 5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENVLA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHECKPOINT_DIR="${OPENVLA_ROOT}/checkpoints"
SUITE="spatial"
QUANT_MODES="bf16,8bit,4bit"
NUM_TRIALS=50
SEED=7
LOG_DIR="${OPENVLA_ROOT}/experiments/logs/quantized_benchmark"

declare -A SUITE_TO_CHECKPOINT=(
    ["spatial"]="openvla-7b-finetuned-libero-spatial"
    ["object"]="openvla-7b-finetuned-libero-object"
    ["goal"]="openvla-7b-finetuned-libero-goal"
    ["10"]="openvla-7b-finetuned-libero-10"
    ["libero_spatial"]="openvla-7b-finetuned-libero-spatial"
    ["libero_object"]="openvla-7b-finetuned-libero-object"
    ["libero_goal"]="openvla-7b-finetuned-libero-goal"
    ["libero_10"]="openvla-7b-finetuned-libero-10"
)

declare -A SUITE_TO_TASK=(
    ["spatial"]="libero_spatial"
    ["object"]="libero_object"
    ["goal"]="libero_goal"
    ["10"]="libero_10"
    ["libero_spatial"]="libero_spatial"
    ["libero_object"]="libero_object"
    ["libero_goal"]="libero_goal"
    ["libero_10"]="libero_10"
)

usage() {
    cat <<'EOF'
OpenVLA LIBERO 量化评测

选项:
  --suite NAME       任务套件: spatial | object | goal | 10
  --quant LIST       量化模式，逗号分隔: bf16, 8bit, 4bit（默认全部）
  --checkpoint DIR   模型本地路径（默认根据 --suite 自动选择 checkpoints/ 下目录）
  --trials N         每个任务的 rollout 次数（默认 50；快速测试可设 5）
  --seed N           随机种子（默认 7）
  --log-dir DIR      日志目录
  -h, --help         显示帮助

指标说明:
  脚本会在仿真中跑任务，最终输出各量化模式的 task success rate（任务成功率）。
  这是 OpenVLA 官方 LIBERO 评测指标，可直接对比 bf16 / 8bit / 4bit 的性能与显存占用。

示例:
  bash scripts/run_libero_quantized_benchmark.sh --suite spatial --quant bf16,4bit --trials 10
EOF
}

CHECKPOINT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite) SUITE="${2:?}"; shift 2 ;;
        --quant) QUANT_MODES="${2:?}"; shift 2 ;;
        --checkpoint) CHECKPOINT="${2:?}"; shift 2 ;;
        --trials) NUM_TRIALS="${2:?}"; shift 2 ;;
        --seed) SEED="${2:?}"; shift 2 ;;
        --log-dir) LOG_DIR="${2:?}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "未知参数: $1" >&2; usage; exit 1 ;;
    esac
done

ckpt_name="${SUITE_TO_CHECKPOINT[${SUITE}]:-}"
task_suite="${SUITE_TO_TASK[${SUITE}]:-}"

if [[ -z "${ckpt_name}" || -z "${task_suite}" ]]; then
    echo "[错误] 未知 suite: ${SUITE}（可选: spatial, object, goal, 10）" >&2
    exit 1
fi

if [[ -z "${CHECKPOINT}" ]]; then
    CHECKPOINT="${CHECKPOINT_DIR}/${ckpt_name}"
fi

if [[ ! -d "${CHECKPOINT}" || ! -f "${CHECKPOINT}/config.json" ]]; then
    echo "[错误] 找不到模型: ${CHECKPOINT}" >&2
    echo "请先运行: bash scripts/download_openvla_eval_models.sh --models ${SUITE}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"
SUMMARY_FILE="${LOG_DIR}/quant_summary_${task_suite}_$(date +%Y%m%d_%H%M%S).txt"

IFS=',' read -r -a MODES <<< "${QUANT_MODES}"

echo "=============================================="
echo " OpenVLA 量化评测"
echo " 模型: ${CHECKPOINT}"
echo " 套件: ${task_suite}"
echo " 量化: ${QUANT_MODES}"
echo " trials/task: ${NUM_TRIALS}"
echo "=============================================="

# 抑制 TensorFlow 冗余日志
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

cd "${OPENVLA_ROOT}"

run_eval() {
    local mode="$1"
    local load_8bit="False"
    local load_4bit="False"
    local note="${mode}"

    case "${mode}" in
        bf16|fp16|none) ;;
        8bit) load_8bit="True" ;;
        4bit) load_4bit="True" ;;
        *)
            echo "[警告] 未知量化模式: ${mode}，跳过" >&2
            return 0
            ;;
    esac

    if [[ "${load_8bit}" == "True" || "${load_4bit}" == "True" ]]; then
        python -c "import bitsandbytes" 2>/dev/null || {
            echo "[错误] ${mode} 需要 bitsandbytes: pip install bitsandbytes" >&2
            return 1
        }
    fi

    echo ""
    echo ">>> 运行评测: quant=${mode}"

    PYTHONPATH=. python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint "${CHECKPOINT}" \
        --task_suite_name "${task_suite}" \
        --center_crop True \
        --load_in_8bit "${load_8bit}" \
        --load_in_4bit "${load_4bit}" \
        --num_trials_per_task "${NUM_TRIALS}" \
        --seed "${SEED}" \
        --run_id_note "quant-${mode}" \
        --local_log_dir "${LOG_DIR}"

    # 从最新日志中提取总成功率
    local latest_log
    latest_log="$(ls -t "${LOG_DIR}"/EVAL-"${task_suite}"-*quant-"${mode}"*.txt 2>/dev/null | head -1 || true)"
    if [[ -n "${latest_log}" ]]; then
        local rate
        rate="$(grep -E "Current total success rate:" "${latest_log}" | tail -1 | awk '{print $NF}' || echo "N/A")"
        echo "${mode},${task_suite},${NUM_TRIALS},${rate},${latest_log}" >> "${SUMMARY_FILE}.csv"
        echo "[结果] ${mode} total success rate = ${rate}"
    fi
}

# 写 CSV 表头
echo "quant_mode,task_suite,trials_per_task,total_success_rate,log_file" > "${SUMMARY_FILE}.csv"

for mode in "${MODES[@]}"; do
    mode="$(echo "${mode}" | xargs)"
    run_eval "${mode}"
done

echo ""
echo "=============================================="
echo " 量化评测完成，汇总文件:"
echo "   ${SUMMARY_FILE}.csv"
echo ""
column -t -s',' "${SUMMARY_FILE}.csv" 2>/dev/null || cat "${SUMMARY_FILE}.csv"
echo "=============================================="
