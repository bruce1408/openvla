"""Small TensorRT 10 runner that uses torch tensors as CUDA buffers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class TensorRTRunner:
    def __init__(self, engine_path: Path | str, device: str = "cuda:0") -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT execution requires CUDA")
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError("Install the JetPack-provided TensorRT Python package") from exc

        self.trt = trt
        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        engine_bytes = Path(engine_path).read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create TensorRT execution context")

        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def _torch_dtype(self, trt_dtype: Any) -> torch.dtype:
        mapping = {
            self.trt.float32: torch.float32,
            self.trt.float16: torch.float16,
            self.trt.int32: torch.int32,
            self.trt.int8: torch.int8,
            self.trt.bool: torch.bool,
        }
        if hasattr(self.trt, "bfloat16"):
            mapping[self.trt.bfloat16] = torch.bfloat16
        try:
            return mapping[trt_dtype]
        except KeyError as exc:
            raise TypeError(f"Unsupported TensorRT dtype: {trt_dtype}") from exc

    def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        missing = set(self.input_names) - set(inputs)
        extra = set(inputs) - set(self.input_names)
        if missing or extra:
            raise ValueError(f"Engine inputs mismatch; missing={sorted(missing)}, extra={sorted(extra)}")

        bound_inputs: dict[str, torch.Tensor] = {}
        for name in self.input_names:
            tensor = inputs[name].to(
                device=self.device,
                dtype=self._torch_dtype(self.engine.get_tensor_dtype(name)),
            ).contiguous()
            if not self.context.set_input_shape(name, tuple(tensor.shape)):
                raise RuntimeError(f"TensorRT rejected shape {tuple(tensor.shape)} for {name}")
            self.context.set_tensor_address(name, tensor.data_ptr())
            bound_inputs[name] = tensor

        outputs: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(f"Unresolved output shape for {name}: {shape}")
            tensor = torch.empty(
                shape,
                dtype=self._torch_dtype(self.engine.get_tensor_dtype(name)),
                device=self.device,
            )
            self.context.set_tensor_address(name, tensor.data_ptr())
            outputs[name] = tensor

        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return outputs
