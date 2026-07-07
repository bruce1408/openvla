import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from runtime_env import ENV_SCRIPT, MODEL_PATH

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


MODEL_ID = str(MODEL_PATH)
DEVICE = os.getenv("OPENVLA_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
ATTN_IMPLEMENTATION = os.getenv("OPENVLA_ATTN_IMPLEMENTATION", "sdpa")
UNNORM_KEY = os.getenv("OPENVLA_UNNORM_KEY", "bridge_orig")


def model_dtype() -> torch.dtype:
    if DEVICE.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))

print("env script:", ENV_SCRIPT)
print("model path:", MODEL_PATH)

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    attn_implementation=ATTN_IMPLEMENTATION,
    torch_dtype=model_dtype(),
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    local_files_only=True,
).to(DEVICE)
model.eval()

image = Image.new("RGB", (224, 224), color=(128, 128, 128))
instruction = "move the robot arm forward"
prompt = f"In: What action should the robot take to {instruction}?\nOut:"

inputs = processor(prompt, image).to(DEVICE, dtype=model_dtype())

with torch.inference_mode():
    action = model.predict_action(
        **inputs,
        unnorm_key=UNNORM_KEY,
        do_sample=False,
    )

print("action:", action)
