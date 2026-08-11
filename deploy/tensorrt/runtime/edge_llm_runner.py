"""TensorRT Edge-LLM runner for OpenVLA embedding-prefill inference."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
from typing import Any

import numpy as np


def build_rope_cos_sin(max_len: int, rotary_dim: int, theta: float) -> np.ndarray:
    """Mirror Edge-LLM's normal RoPE cache layout: [cos, sin]."""

    half = rotary_dim // 2
    positions = np.arange(max_len, dtype=np.float64)[:, None]
    dimensions = np.arange(half, dtype=np.float64)[None, :]
    angles = positions / np.power(theta, 2.0 * dimensions / rotary_dim)
    cache = np.empty((max_len, rotary_dim), dtype=np.float32)
    cache[:, :half] = np.cos(angles)
    cache[:, half:] = np.sin(angles)
    return cache[None]


class EdgeLlmRunner:
    """Drive an Edge-LLM TensorRT engine through its explicit KV-cache contract."""

    def __init__(
        self,
        engine_dir: Path | str,
        plugin_path: Path | str,
        device: str = "cuda:0",
    ) -> None:
        import tensorrt as trt
        import torch

        self.trt = trt
        self.torch = torch
        self.device = torch.device(device)
        self.engine_dir = Path(engine_dir)
        self.engine_path = self.engine_dir / "llm.engine"
        self.config = json.loads((self.engine_dir / "config.json").read_text(encoding="utf-8"))

        self.hidden_size = int(self.config["hidden_size"])
        self.vocab_size = int(self.config["vocab_size"])
        self.num_layers = int(self.config["num_hidden_layers"])
        self.num_kv_heads = int(self.config["num_key_value_heads"])
        self.head_dim = int(self.config["head_dim"])
        self.kv_capacity = int(self.config["builder_config"]["max_kv_cache_capacity"])
        self.rope_theta = float(self.config.get("rope_theta", 10000.0))

        plugin_path = Path(plugin_path)
        if not plugin_path.is_file():
            raise FileNotFoundError(f"Edge-LLM plugin not found: {plugin_path}")
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"Edge-LLM engine not found: {self.engine_path}")

        ctypes.CDLL(str(plugin_path))
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")
        with self.engine_path.open("rb") as engine_file, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_file.read())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize Edge-LLM engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create Edge-LLM execution context")

        rope = build_rope_cos_sin(self.kv_capacity, self.head_dim, self.rope_theta)
        self.rope = torch.from_numpy(rope).to(self.device)
        self.kv = [
            torch.zeros(
                (1, 2, self.num_kv_heads, self.kv_capacity, self.head_dim),
                dtype=torch.float16,
                device=self.device,
            )
            for _ in range(self.num_layers)
        ]
        self.logits = torch.empty((1, 1, self.vocab_size), dtype=torch.float32, device=self.device)

    def load_embedding_table(self) -> Any:
        """Load the exported token embedding table onto the runner device."""

        from safetensors.numpy import load_file

        path = self.engine_dir / "embedding.safetensors"
        table = load_file(str(path))["embedding"]
        if table.shape != (self.vocab_size, self.hidden_size):
            raise ValueError(
                f"Embedding shape {table.shape} does not match engine "
                f"({self.vocab_size}, {self.hidden_size})"
            )
        return self.torch.from_numpy(table).to(self.device)

    def _run(self, embeddings: Any, start: int, valid_len: int, last_index: int) -> Any:
        torch = self.torch
        sequence_length = int(embeddings.shape[0])
        if valid_len > self.kv_capacity:
            raise ValueError(f"KV length {valid_len} exceeds engine capacity {self.kv_capacity}")

        embeddings = (
            embeddings.to(self.device, dtype=torch.float16)
            .contiguous()
            .view(1, sequence_length, self.hidden_size)
        )
        context_lengths = torch.tensor([valid_len], dtype=torch.int32, device=self.device)
        start_index = torch.tensor([start], dtype=torch.int32, device=self.device)
        last_token_ids = torch.tensor([[last_index]], dtype=torch.int64, device=self.device)

        context = self.context
        bindings = {
            "inputs_embeds": embeddings,
            "rope_rotary_cos_sin": self.rope,
            "context_lengths": context_lengths,
            "kvcache_start_index": start_index,
            "last_token_ids": last_token_ids,
        }
        for name, tensor in bindings.items():
            if not context.set_input_shape(name, tuple(tensor.shape)):
                raise RuntimeError(f"TensorRT rejected shape {tuple(tensor.shape)} for {name}")
            context.set_tensor_address(name, tensor.data_ptr())

        for layer_index, kv_buffer in enumerate(self.kv):
            pointer = kv_buffer.data_ptr()
            name = f"past_key_values_{layer_index}"
            if not context.set_input_shape(name, tuple(kv_buffer.shape)):
                raise RuntimeError(f"TensorRT rejected KV shape for layer {layer_index}")
            context.set_tensor_address(name, pointer)
            context.set_tensor_address(f"present_key_values_{layer_index}", pointer)
        context.set_tensor_address("logits", self.logits.data_ptr())

        stream = torch.cuda.current_stream(self.device)
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("Edge-LLM execute_async_v3 returned False")
        stream.synchronize()
        return self.logits.view(self.vocab_size).clone()

    def generate(self, prefill_embeddings: Any, embedding_table: Any, n_new_tokens: int) -> list[int]:
        """Greedily generate action tokens from multimodal prefill embeddings."""

        if n_new_tokens < 1:
            raise ValueError("n_new_tokens must be positive")
        sequence_length = int(prefill_embeddings.shape[0])
        if sequence_length + n_new_tokens - 1 > self.kv_capacity:
            raise ValueError(
                f"Prefill ({sequence_length}) + decode ({n_new_tokens - 1}) exceeds "
                f"KV capacity {self.kv_capacity}"
            )

        for kv_buffer in self.kv:
            kv_buffer.zero_()

        logits = self._run(
            prefill_embeddings,
            start=0,
            valid_len=sequence_length,
            last_index=sequence_length - 1,
        )
        token = int(self.torch.argmax(logits).item())
        generated = [token]
        current_length = sequence_length
        for _ in range(n_new_tokens - 1):
            token_embedding = embedding_table[token].view(1, self.hidden_size)
            logits = self._run(
                token_embedding,
                start=current_length,
                valid_len=current_length + 1,
                last_index=0,
            )
            token = int(self.torch.argmax(logits).item())
            generated.append(token)
            current_length += 1
        return generated
