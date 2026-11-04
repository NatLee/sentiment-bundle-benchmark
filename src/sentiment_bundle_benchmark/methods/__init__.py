"""Benchmark methods. Each kind exposes `run(method, data) -> MethodOutput`."""

from __future__ import annotations

from importlib import import_module

from ..config import MethodConfig
from .base import MethodInput, MethodOutput

# kind -> module that implements it; imported lazily so the CPU-only parts never import torch.
_MODULES = {"tfidf": "tfidf", "embedding": "embedding", "finetune": "finetune", "llm": "llm"}


def run_method(method: MethodConfig, data: MethodInput) -> MethodOutput:
    module = import_module(f"{__name__}.{_MODULES[method.kind]}")
    return module.run(method, data)


__all__ = ["MethodInput", "MethodOutput", "run_method"]
