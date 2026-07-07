import json
import logging
import os
import time
import traceback
from typing import Any

from runtime_env import MODEL_PATH

import json_numpy

json_numpy.patch()

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


MODEL_ID = str(MODEL_PATH)
DEVICE = os.getenv("OPENVLA_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
ATTN_IMPLEMENTATION = os.getenv("OPENVLA_ATTN_IMPLEMENTATION", "sdpa")
DEFAULT_UNNORM_KEY = os.getenv("OPENVLA_UNNORM_KEY", "bridge_orig")
HOST = os.getenv("OPENVLA_HOST", "0.0.0.0")
PORT = int(os.getenv("OPENVLA_PORT", "8000"))


def get_prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def model_dtype() -> torch.dtype:
    if DEVICE.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


class OpenVLAServer:
    def __init__(self) -> None:
        print(f"Loading processor: {MODEL_ID}")
        self.processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )

        print(f"Loading model: {MODEL_ID}")
        print(f"Device: {DEVICE}; attention: {ATTN_IMPLEMENTATION}")
        self.model = AutoModelForVision2Seq.from_pretrained(
            MODEL_PATH,
            attn_implementation=ATTN_IMPLEMENTATION,
            torch_dtype=model_dtype(),
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
        ).to(DEVICE)
        self.model.eval()

        print("Ready.")
        print("torch:", torch.__version__)
        print("cuda runtime:", torch.version.cuda)
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("gpu:", torch.cuda.get_device_name(0))
            print("capability:", torch.cuda.get_device_capability(0))

    def predict_action(self, payload: dict[str, Any]) -> JSONResponse:
        try:
            request_start = time.perf_counter()
            if "encoded" in payload:
                payload = json.loads(payload["encoded"])

            decode_start = time.perf_counter()
            image = Image.fromarray(payload["image"]).convert("RGB")
            instruction = payload["instruction"]
            unnorm_key = payload.get("unnorm_key", DEFAULT_UNNORM_KEY)
            decode_end = time.perf_counter()

            processor_start = time.perf_counter()
            inputs = self.processor(get_prompt(instruction), image).to(
                DEVICE,
                dtype=model_dtype(),
            )
            if DEVICE.startswith("cuda"):
                torch.cuda.synchronize()
            processor_end = time.perf_counter()

            infer_start = time.perf_counter()
            with torch.inference_mode():
                action = self.model.predict_action(
                    **inputs,
                    unnorm_key=unnorm_key,
                    do_sample=False,
                )
            if DEVICE.startswith("cuda"):
                torch.cuda.synchronize()
            infer_end = time.perf_counter()

            metrics = {
                "json_image_decode_time_ms": (decode_end - decode_start) * 1000.0,
                "processor_h2d_time_ms": (processor_end - processor_start) * 1000.0,
                "predict_action_total_time_ms": (infer_end - infer_start) * 1000.0,
                "service_e2e_time_ms": (infer_end - request_start) * 1000.0,
            }

            if payload.get("return_metrics", False):
                return JSONResponse({"action": action, "metrics": metrics})
            return JSONResponse(action)
        except Exception:
            logging.error(traceback.format_exc())
            return JSONResponse({"error": traceback.format_exc()}, status_code=500)


server = OpenVLAServer()
app = FastAPI(title="OpenVLA Server")
app.post("/act")(server.predict_action)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "device": DEVICE,
        "attention": ATTN_IMPLEMENTATION,
        "cuda_available": torch.cuda.is_available(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
