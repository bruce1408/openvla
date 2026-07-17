#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
onnx_path="${1:-${repo_root}/deploy/tensorrt/artifacts/onnx/vision_projector_fp16.onnx}"
engine_path="${2:-${repo_root}/deploy/tensorrt/artifacts/engines/vision_projector_fp16.plan}"

if command -v trtexec >/dev/null 2>&1; then
    trtexec_bin="$(command -v trtexec)"
elif [[ -x /usr/src/tensorrt/bin/trtexec ]]; then
    trtexec_bin=/usr/src/tensorrt/bin/trtexec
else
    echo "trtexec was not found in PATH or /usr/src/tensorrt/bin" >&2
    exit 1
fi

mkdir -p "$(dirname "${engine_path}")"
"${trtexec_bin}" \
    --onnx="${onnx_path}" \
    --saveEngine="${engine_path}" \
    --fp16 \
    --builderOptimizationLevel=5 \
    --profilingVerbosity=detailed \
    --skipInference

echo "Built ${engine_path}"
