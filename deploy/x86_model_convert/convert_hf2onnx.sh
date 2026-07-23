#!/bin/bash
# OpenVLA Llama -> Edge-LLM ONNX 导出脚本（x86 GPU 主机）
#
# 用法:
#   ./convert_hf_onnx.sh              # 只导出 BF16 基准（默认，无量化）
#   ./convert_hf_onnx.sh bf16         # 同上
#   ./convert_hf_onnx.sh fp8          # FP8 量化 + 导出
#   ./convert_hf_onnx.sh nvfp4        # NVFP4 量化 + 导出
#   ./convert_hf_onnx.sh mxfp8        # MXFP8 量化 + 导出
#   ./convert_hf_onnx.sh all          # bf16 + fp8 + nvfp4 + mxfp8 全部
#   FORCE=1 ./convert_hf_onnx.sh fp8  # 覆盖已有产物重新执行
#
# 注意:
#   1. 量化需要安装 tools 组件: pip install ".[tools]"
#   2. 量化校准默认使用本地 /share_data/public/openvla_models/calib_text.jsonl
#      （512 条 cnn_dailymail 文章，已通过 hf-mirror 下载）；可用
#      CALIB_DATASET=... 覆盖；设为 cnn_dailymail 则从 HF hub 在线加载
#   3. 按部署方案顺序：必须先用 bf16 产物在 Thor 上验证 token exact match，
#      再使用 fp8 / nvfp4 产物，不要跳过基准验证。

set -euo pipefail

# 本机无外网：禁止 HF hub 联网，全部使用本地文件
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---------------- 配置 ----------------
MODEL_DIR="${MODEL_DIR:-/share_data/public/openvla_models/hf_llama}"
OUT_BASE="${OUT_BASE:-/share_data/public/openvla_models}"
LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORCE="${FORCE:-0}"

NUM_SAMPLES="${NUM_SAMPLES:-512}"   # 校准样本数，减小可加快校准（精度风险自负）

# 校准数据：离线机器使用本地 JSONL（需含 "text" 列），默认 cnn_dailymail 需联网
CALIB_DATASET="${CALIB_DATASET:-/share_data/public/openvla_models/calib_text.jsonl}"

# ---------------- 函数 ----------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# 导出: $1=模型目录 $2=输出目录 $3=日志名
do_export() {
    local model_dir="$1" out_dir="$2" log_name="$3"
    if [[ -f "${out_dir}/llm/config.json" && "${FORCE}" != "1" ]]; then
        log "跳过导出（已存在）: ${out_dir}  (FORCE=1 可覆盖)"
        return 0
    fi
    log "导出 ONNX: ${model_dir} -> ${out_dir}"
    tensorrt-edgellm-export "${model_dir}" "${out_dir}" \
        2>&1 | tee "${LOG_DIR}/${log_name}"
    # 成功标志：config / 权重 / tokenizer / chat template 必须齐全
    for f in config.json model.onnx model.onnx.data embedding.safetensors \
             tokenizer.json processed_chat_template.json; do
        if [[ ! -f "${out_dir}/llm/${f}" ]]; then
            # chat template 缺失时补一份 fallback（OpenVLA 不使用 chat 模板）
            if [[ "${f}" == "processed_chat_template.json" ]]; then
                python3 - "${model_dir}" "${out_dir}/llm/${f}" <<'PY'
import json, sys
fallback = {
    "model_path": sys.argv[1],
    "roles": {"system": {"prefix": "", "suffix": "\n"},
              "user": {"prefix": "User: ", "suffix": "\n"},
              "assistant": {"prefix": "Assistant: ", "suffix": "\n"}},
    "content_types": {},
    "generation_prompt": "Assistant: ",
    "default_system_prompt": "",
}
json.dump(fallback, open(sys.argv[2], "w"), indent=2)
print("wrote fallback processed_chat_template.json")
PY
            else
                echo "导出失败：缺少 ${out_dir}/llm/${f}" >&2
                return 1
            fi
        fi
    done
    log "导出成功: ${out_dir}"
}

# 量化: $1=精度(fp8|nvfp4|...) $2=输出目录 $3=日志名
do_quantize() {
    local quant="$1" out_dir="$2" log_name="$3"
    if [[ -f "${out_dir}/config.json" && "${FORCE}" != "1" ]]; then
        log "跳过量化（已存在）: ${out_dir}  (FORCE=1 可覆盖)"
        return 0
    fi
    log "量化 (${quant}): ${MODEL_DIR} -> ${out_dir}"
    tensorrt-edgellm-quantize llm \
        --model_dir "${MODEL_DIR}" \
        --output_dir "${out_dir}" \
        --quantization "${quant}" \
        --num_samples "${NUM_SAMPLES}" \
        --dataset "${CALIB_DATASET}" \
        2>&1 | tee "${LOG_DIR}/${log_name}"
    [[ -f "${out_dir}/config.json" ]] || { echo "量化失败：缺少 ${out_dir}/config.json" >&2; return 1; }
    log "量化成功 (${quant}): ${out_dir}"
}

stage_bf16() {
    do_export "${MODEL_DIR}" "${OUT_BASE}/hf_llama_onnx" edgellm_export.log
}

stage_fp8() {
    do_quantize fp8 "${OUT_BASE}/hf_llama_fp8" edgellm_quantize_fp8.log
    do_export "${OUT_BASE}/hf_llama_fp8" "${OUT_BASE}/hf_llama_onnx_fp8" edgellm_export_fp8.log
}

stage_nvfp4() {
    do_quantize nvfp4 "${OUT_BASE}/hf_llama_nvfp4" edgellm_quantize_nvfp4.log
    do_export "${OUT_BASE}/hf_llama_nvfp4" "${OUT_BASE}/hf_llama_onnx_nvfp4" edgellm_export_nvfp4.log
}

stage_mxfp8() {
    do_quantize mxfp8 "${OUT_BASE}/hf_llama_mxfp8" edgellm_quantize_mxfp8.log
    do_export "${OUT_BASE}/hf_llama_mxfp8" "${OUT_BASE}/hf_llama_onnx_mxfp8" edgellm_export_mxfp8.log
}

# ---------------- 入口 ----------------
STAGE="${1:-bf16}"
[[ -d "${MODEL_DIR}" ]] || { echo "模型目录不存在: ${MODEL_DIR}" >&2; exit 1; }

case "${STAGE}" in
    bf16)  stage_bf16 ;;
    fp8)   stage_fp8 ;;
    nvfp4) stage_nvfp4 ;;
    mxfp8) stage_mxfp8 ;;
    all)   stage_bf16; stage_fp8; stage_nvfp4; stage_mxfp8 ;;
    *) echo "未知阶段: ${STAGE}（可选: bf16 | fp8 | nvfp4 | mxfp8 | all）" >&2; exit 2 ;;
esac

log "全部完成 (${STAGE})"
