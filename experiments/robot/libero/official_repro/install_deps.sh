#!/usr/bin/env bash
# 安装官方 LIBERO 复现所需的 TensorFlow（在 torch270_128 环境中）
set -euo pipefail

: "${PY:=/home/bruce/miniconda3/envs/torch270_128/bin/python}"

echo "Installing tensorflow==2.15.0 into $(dirname "$PY") ..."
"$PY" -m pip install 'tensorflow==2.15.0'

echo ""
bash "$(dirname "$0")/check_env.sh"
