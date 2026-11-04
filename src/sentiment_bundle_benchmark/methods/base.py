"""What a method receives and what it hands back to the runner."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..datasets import DatasetSpec
from ..splits import Splits


@dataclass
class MethodInput:
    spec: DatasetSpec
    splits: Splits
    seed: int
    save_dir: Path | None = None  # where to keep the trained model, if the method can save one


@dataclass
class MethodOutput:
    predictions: list[int]
    train_seconds: float
    infer_seconds: float
    params: int | None = None  # parameters (or learned weights) of the model used at inference
    size_mb: float | None = None  # memory taken by those weights
    peak_vram_mb: float | None = None
    device: str = "cpu"
    details: dict[str, Any] = field(default_factory=dict)


class Stopwatch:
    def __init__(self) -> None:
        self.seconds = 0.0

    @contextmanager
    def running(self):
        start = time.perf_counter()
        try:
            yield self
        finally:
            self.seconds += time.perf_counter() - start
