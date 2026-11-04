"""Turn a dataset into fixed train / val / test splits that every method shares.

The same rules apply to every dataset:
1. Strip whitespace and drop empty texts.
2. Drop exact duplicate texts. If copies disagree on the label, drop them all; otherwise keep one,
   preferring the copy in train, then val, then test, so no test text is also a training text.
3. Use the source's own test (and val) split when it has one. Otherwise carve a stratified
   test_fraction (and val_fraction of what is left), never more than the cap, with the seed.
4. Subsample each split down to its cap (stratified, same seed).
Row ids refer to positions in the reader's output, so a split can be re-checked from its id list.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import SplitConfig
from .datasets import SPLIT_NAMES

_PRIORITY = {name: rank for rank, name in enumerate(SPLIT_NAMES)}


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def __getitem__(self, name: str) -> pd.DataFrame:
        return getattr(self, name)

    def items(self):
        return ((name, self[name]) for name in SPLIT_NAMES)


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.assign(text=frame["text"].str.strip())
    frame = frame[frame["text"] != ""]
    labels_per_text = frame.groupby("text")["label"].nunique()
    conflicting = labels_per_text[labels_per_text > 1].index
    frame = frame[~frame["text"].isin(conflicting)]
    order = frame["split"].map(_PRIORITY)
    frame = frame.assign(_rank=order).sort_values(["_rank"], kind="stable")
    frame = frame.drop_duplicates("text", keep="first").drop(columns="_rank")
    return frame.sort_index()


def _carve(frame: pd.DataFrame, fraction: float, limit: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off a stratified `fraction` of `frame`, at most `limit` rows. Returns (rest, carved)."""
    size = min(round(fraction * len(frame)), limit)
    rest, carved = train_test_split(frame, test_size=size, stratify=frame["label"], random_state=seed)
    return rest.sort_index(), carved.sort_index()


def _cap(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    kept, _ = train_test_split(frame, train_size=limit, stratify=frame["label"], random_state=seed)
    return kept.sort_index()


def build_splits(frame: pd.DataFrame, config: SplitConfig, seed: int) -> Splits:
    frame = deduplicate(frame.rename_axis("id").reset_index().set_index("id", drop=False))

    test = frame[frame["split"] == "test"]
    pool = frame[frame["split"] != "test"]
    if test.empty:
        pool, test = _carve(pool, config.test_fraction, config.max_test, seed)

    val = pool[pool["split"] == "val"]
    train = pool[pool["split"] != "val"]
    if val.empty:
        train, val = _carve(train, config.val_fraction, config.max_val, seed)

    columns = ["id", "text", "label"]
    return Splits(
        train=_cap(train, config.max_train, seed)[columns].reset_index(drop=True),
        val=_cap(val, config.max_val, seed)[columns].reset_index(drop=True),
        test=_cap(test, config.max_test, seed)[columns].reset_index(drop=True),
    )


def describe(splits: Splits, labels: tuple[str, ...]) -> dict:
    """Counts, label balance and a fingerprint of the exact ids in each split."""
    out = {}
    for name, part in splits.items():
        ids = ",".join(str(i) for i in sorted(part["id"].tolist()))
        counts = part["label"].value_counts().reindex(range(len(labels)), fill_value=0)
        out[name] = {
            "size": len(part),
            "labels": {labels[i]: int(n) for i, n in counts.items()},
            "ids_sha256": hashlib.sha256(ids.encode()).hexdigest(),
        }
    return out


def save_splits(splits: Splits, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_json(directory / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)


def load_splits(directory: Path) -> Splits:
    parts = {}
    for name in SPLIT_NAMES:
        path = directory / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run `sbb prepare` first")
        parts[name] = pd.read_json(path, orient="records", lines=True, dtype={"id": int, "text": str, "label": int})
    return Splits(**parts)


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
