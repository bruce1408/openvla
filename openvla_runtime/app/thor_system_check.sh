#!/usr/bin/env bash
set -euo pipefail

echo "== system =="
uname -a
cat /etc/os-release || true

echo
echo "== nvidia packages =="
dpkg -l | grep -Ei 'nvidia|cuda|tensorrt|jetpack' || true

echo
echo "== gpu tools =="
nvidia-smi || true
tegrastats --interval 1000 --count 3 || true

echo
echo "== power mode =="
sudo nvpmodel -q || nvpmodel -q || true

echo
echo "== python / torch / tensorrt =="
python - <<'PY'
import importlib.util
import sys

print("python:", sys.version)

if importlib.util.find_spec("torch"):
    import torch
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
else:
    print("torch: not installed")

if importlib.util.find_spec("tensorrt"):
    import tensorrt as trt
    print("tensorrt:", trt.__version__)
else:
    print("tensorrt: not installed")
PY
