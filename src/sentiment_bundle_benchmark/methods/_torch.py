"""Small helpers shared by the PyTorch-based methods."""

from __future__ import annotations

import random

import numpy as np
import torch


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    """bfloat16 on GPUs that support it, float32 otherwise (CPU smoke runs)."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def peak_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 2**20


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def weights_of(module: torch.nn.Module) -> tuple[int, float]:
    """Parameter count and their size in MB, as loaded."""
    params = list(module.parameters())
    count = sum(p.numel() for p in params)
    size = sum(p.numel() * p.element_size() for p in params) / 2**20
    return int(count), float(size)

