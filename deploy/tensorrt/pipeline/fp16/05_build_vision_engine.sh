#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
onnx_path="${1:-${repo_root}/deploy/tensorrt/artifacts/onnx/vision_projector_fp16.onnx}"
engine_path="${2:-${repo_root}/deploy/tensorrt/artifacts/engines/vision_projector_fp16.plan}"
# 第 3 个参数: 精度 fp16 (默认) 或 fp8。fp8 需要输入是已插入 Q/DQ 的量化 ONNX
# (由 01b_quantize_vision_fp8.py 产出),并同时开 --fp16 作为非量化层的兜底精度。
precision="${3:-fp16}"

if command -v trtexec >/dev/null 2>&1; then
    trtexec_bin="$(command -v trtexec)"
elif [[ -x /usr/src/tensorrt/bin/trtexec ]]; then
    trtexec_bin=/usr/src/tensorrt/bin/trtexec
else
    echo "trtexec was not found in PATH or /usr/src/tensorrt/bin" >&2
    exit 1
fi

# TensorRT 10.x 用 --fp16/--fp8 选精度。11.x 默认 strongly-typed，
# 这两个开关已删除，精度跟 ONNX 里的 dtype / QDQ 走。
precision_flags=()
if "${trtexec_bin}" --help 2>&1 | grep -qE '^[[:space:]]+--fp16'; then
    precision_flags=(--fp16)
    if [[ "${precision}" == "fp8" ]]; then
        precision_flags=(--fp8 --fp16)
    fi
else
    echo "trtexec has no --fp16/--fp8 (TensorRT 11+ strongly-typed). Using ONNX dtypes."
    if [[ "${precision}" == "fp8" ]]; then
        echo "Expect Q/DQ FP8 nodes in ${onnx_path}"
    fi
fi

mkdir -p "$(dirname "${engine_path}")"
build_cmd=("${trtexec_bin}" --onnx="${onnx_path}" --saveEngine="${engine_path}")
if ((${#precision_flags[@]})); then
    build_cmd+=("${precision_flags[@]}")
fi
build_cmd+=(--builderOptimizationLevel=5 --profilingVerbosity=detailed --skipInference)
"${build_cmd[@]}"

echo "Built ${engine_path} (precision=${precision})"
