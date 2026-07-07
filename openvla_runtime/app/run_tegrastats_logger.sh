#!/usr/bin/env bash
set -euo pipefail

PREFIX="${OPENVLA_PREFIX:-/share_data/bruce/workspace/openvla_runtime}"
mkdir -p "${PREFIX}/logs"
LOG_FILE="${1:-${PREFIX}/logs/tegrastats_$(date +%Y%m%d_%H%M%S).log}"

echo "Writing tegrastats log to ${LOG_FILE}"
echo "Stop with Ctrl-C."
tegrastats --interval 1000 2>&1 | tee -a "${LOG_FILE}"
