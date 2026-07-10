import os
import subprocess
from pathlib import Path


ENV_SCRIPT = Path(__file__).resolve().parent / "env.sh"


def source_env_script(script_path: Path) -> None:
    """Source env.sh and copy its exported variables into this process."""
    if not script_path.is_file():
        raise FileNotFoundError(f"Environment script not found: {script_path}")

    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            'source "$1" && env -0',
            "bash",
            str(script_path),
        ],
        check=True,
        capture_output=True,
    )
    for entry in result.stdout.split(b"\0"):
        if not entry or b"=" not in entry:
            continue
        key, value = entry.split(b"=", 1)
        os.environ[os.fsdecode(key)] = os.fsdecode(value)


source_env_script(ENV_SCRIPT)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_PATH = os.environ.get("OPENVLA_MODEL_ID", "openvla/openvla-7b")

# 是否为本地模型目录（本地目录不需要 revision）。
IS_LOCAL_MODEL = MODEL_PATH.startswith(("/", "./", "../", "~"))

if IS_LOCAL_MODEL:
    local_model_path = Path(MODEL_PATH).expanduser()
    if not local_model_path.is_dir():
        raise FileNotFoundError(
            f"OPENVLA_MODEL_ID must point to an existing local model directory: {local_model_path}"
        )
    MODEL_PATH = str(local_model_path)

os.environ["OPENVLA_MODEL_ID"] = MODEL_PATH

# 固定 Hugging Face 仓库的 commit，避免每次加载都重新下载/更新自定义远程代码，
# 同时保证可复现。仅对 HF 模型 ID 生效；本地目录忽略。
_revision = os.environ.get("OPENVLA_REVISION") or None
MODEL_REVISION = None if IS_LOCAL_MODEL else _revision
