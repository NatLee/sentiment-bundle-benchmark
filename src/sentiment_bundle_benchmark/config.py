"""Load a benchmark configuration (TOML) into small typed objects.

Everything that could change between runs lives in the config file: sources, file checksums,
split sizes, the seed and the list of methods with their pinned model revisions.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

METHOD_KINDS = ("tfidf", "embedding", "finetune", "llm")


class ConfigError(ValueError):
    """The configuration file is missing something or contains a value we cannot use."""


@dataclass(frozen=True)
class SplitConfig:
    max_train: int
    max_val: int
    max_test: int
    val_fraction: float
    test_fraction: float


@dataclass(frozen=True)
class SourceConfig:
    name: str
    url: str
    files: dict[str, str]  # file name -> SHA-256
    archive: str | None = None
    description: str = ""


@dataclass(frozen=True)
class MethodConfig:
    name: str
    kind: str
    label: str
    params: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.params:
            raise ConfigError(f"method {self.name!r} needs {key!r}")
        return self.params[key]


@dataclass(frozen=True)
class BenchmarkConfig:
    path: Path
    seed: int
    data_dir: Path
    results_dir: Path
    datasets: list[str]
    splits: SplitConfig
    sources: dict[str, SourceConfig]
    dataset_options: dict[str, dict[str, Any]]
    methods: list[MethodConfig]
    # Cap on CPU threads for scikit-learn / BLAS work. None = library default (all cores).
    cpu_threads: int | None = None

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def runs_dir(self) -> Path:
        return self.results_dir / "runs"

    def method(self, name: str) -> MethodConfig:
        for method in self.methods:
            if method.name == name:
                return method
        known = ", ".join(m.name for m in self.methods)
        raise ConfigError(f"unknown method {name!r}; this config has: {known}")


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path)


def load_config(path: str | Path, *, root: str | Path | None = None) -> BenchmarkConfig:
    """Read a config file. Relative directories are resolved against `root` (default: the cwd)."""
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    base = Path(root) if root is not None else Path.cwd()

    try:
        splits = SplitConfig(**raw["splits"])
        sources = {
            name: SourceConfig(
                name=name,
                url=src["url"],
                files=dict(src["files"]),
                archive=src.get("archive"),
                description=src.get("description", ""),
            )
            for name, src in raw.get("sources", {}).items()
        }
        methods = []
        for entry in raw.get("methods", []):
            entry = dict(entry)
            name, kind = entry.pop("name"), entry.pop("kind")
            label = entry.pop("label", name)
            if kind not in METHOD_KINDS:
                raise ConfigError(f"method {name!r} has unknown kind {kind!r}; use one of {METHOD_KINDS}")
            methods.append(MethodConfig(name=name, kind=kind, label=label, params=entry))
        config = BenchmarkConfig(
            path=path,
            seed=int(raw["seed"]),
            data_dir=_resolve(base, raw.get("data_dir", "data")),
            results_dir=_resolve(base, raw.get("results_dir", "results")),
            datasets=list(raw["datasets"]),
            splits=splits,
            sources=sources,
            dataset_options={k: dict(v) for k, v in raw.get("dataset_options", {}).items()},
            methods=methods,
            cpu_threads=raw.get("cpu_threads"),
        )
    except KeyError as exc:
        raise ConfigError(f"{path}: missing required key {exc}") from exc
    except TypeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc

    names = [m.name for m in config.methods]
    if len(names) != len(set(names)):
        raise ConfigError(f"{path}: method names must be unique")
    return config
