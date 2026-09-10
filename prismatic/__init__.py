"""Prismatic VLM utilities.

Heavy model-loading exports are resolved lazily so that
``import prismatic.extern.hf.*`` does not require the full training stack
(draccus, TensorFlow, ...).
"""

from typing import Any

__all__ = [
    "available_model_names",
    "available_models",
    "get_model_description",
    "load",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .models import (
            available_model_names,
            available_models,
            get_model_description,
            load,
        )

        mapping = {
            "available_model_names": available_model_names,
            "available_models": available_models,
            "get_model_description": get_model_description,
            "load": load,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
