import os
import subprocess
from pathlib import Path


ENV_SCRIPT = Path(__file__).resolve().parents[1] / "env.sh"


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

MODEL_PATH = Path(
    os.environ.get(
        "OPENVLA_MODEL_ID",
        "/share_data/public/models/openvla-7b",
    )
).expanduser()

if not MODEL_PATH.is_dir():
    raise FileNotFoundError(
        f"OPENVLA_MODEL_ID must point to an existing local model directory: {MODEL_PATH}"
    )

os.environ["OPENVLA_MODEL_ID"] = str(MODEL_PATH)
