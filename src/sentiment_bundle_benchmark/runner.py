"""Glue: prepare the splits, run methods over datasets, write one JSON record per run."""

from __future__ import annotations

import gc
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from threadpoolctl import threadpool_limits

from .config import BenchmarkConfig, MethodConfig
from .datasets import get_spec, pick_layout, read_dataset
from .metrics import evaluate
from .methods import MethodInput, run_method
from .splits import build_splits, describe, load_splits, save_splits, write_manifest

MANIFEST_NAME = "splits.json"
_PACKAGES = ("numpy", "pandas", "scikit-learn", "torch", "transformers", "sentence-transformers")


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def prepare(config: BenchmarkConfig, datasets: list[str] | None = None) -> dict:
    """Build and save the splits for each dataset; record them in <results_dir>/splits.json."""
    manifest_path = config.results_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["seed"] = config.seed
    manifest["caps"] = {"max_train": config.splits.max_train, "max_val": config.splits.max_val,
                        "max_test": config.splits.max_test}
    manifest.setdefault("datasets", {})
    for name in datasets or config.datasets:
        spec = get_spec(name)
        layout, source = pick_layout(spec, config.sources)
        frame = read_dataset(spec, config.raw_dir, config.sources, config.dataset_options.get(name))
        splits = build_splits(frame, config.splits, config.seed)
        save_splits(splits, config.processed_dir / name)
        manifest["datasets"][name] = {
            "source": source.name,
            "source_files": {file: source.files[file] for file in layout.files},
            "rows_read": len(frame),
            "splits": describe(splits, spec.labels),
        }
        sizes = ", ".join(f"{part} {len(splits[part]):,}" for part in ("train", "val", "test"))
        log(f"{name}: {len(frame):,} rows read -> {sizes}")
    write_manifest(manifest_path, manifest)
    return manifest


def environment() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for package in _PACKAGES:
        try:
            info[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    if "torch" in sys.modules:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    return info


def _free_accelerator_memory() -> None:
    gc.collect()
    if "torch" in sys.modules:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def result_path(config: BenchmarkConfig, method: str, dataset: str) -> Path:
    return config.runs_dir / method / f"{dataset}.json"


def run_one(config: BenchmarkConfig, method: MethodConfig, dataset: str, save_models: Path | None = None) -> dict:
    spec = get_spec(dataset)
    splits = load_splits(config.processed_dir / dataset)
    save_dir = save_models / method.name / dataset if save_models is not None else None
    with threadpool_limits(limits=config.cpu_threads):
        output = run_method(method, MethodInput(spec=spec, splits=splits, seed=config.seed, save_dir=save_dir))
    scores = evaluate(splits.test["label"], output.predictions, len(spec.labels), seed=config.seed)
    n_test = len(splits.test)
    record = {
        "method": method.name,
        "method_label": method.label,
        "kind": method.kind,
        "dataset": dataset,
        "labels": list(spec.labels),
        "metrics": scores,
        "n_train": len(splits.train),
        "n_val": len(splits.val),
        "n_test": n_test,
        "train_seconds": round(output.train_seconds, 2),
        "infer_seconds": round(output.infer_seconds, 2),
        "throughput_per_s": round(n_test / output.infer_seconds, 1) if output.infer_seconds else None,
        "peak_vram_mb": round(output.peak_vram_mb, 1) if output.peak_vram_mb is not None else None,
        "params": output.params,
        "size_mb": round(output.size_mb, 1) if output.size_mb is not None else None,
        "device": output.device,
        "seed": config.seed,
        "cpu_threads": config.cpu_threads,
        "method_config": method.params,
        "details": output.details,
        "environment": environment(),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = result_path(config, method.name, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return record


def run(
    config: BenchmarkConfig,
    methods: list[str] | None = None,
    datasets: list[str] | None = None,
    *,
    force: bool = False,
    save_models: Path | None = None,
) -> list[dict]:
    """Run every (method, dataset) pair that has no result yet (or all of them with force=True)."""
    selected = [config.method(name) for name in methods] if methods else config.methods
    records = []
    for method in selected:
        for dataset in datasets or config.datasets:
            path = result_path(config, method.name, dataset)
            if path.exists() and not force:
                log(f"skip {method.name} on {dataset}: {path} exists (use --force to redo)")
                continue
            log(f"run {method.name} on {dataset}")
            record = run_one(config, method, dataset, save_models)
            _free_accelerator_memory()
            m = record["metrics"]
            log(f"  accuracy {m['accuracy']:.4f}, macro-F1 {m['macro_f1']:.4f}, "
                f"train {record['train_seconds']:.0f}s, {record['throughput_per_s']} examples/s")
            records.append(record)
    return records
