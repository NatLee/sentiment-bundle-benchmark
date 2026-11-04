import pandas as pd
import pytest

from sentiment_bundle_benchmark.config import SplitConfig
from sentiment_bundle_benchmark.splits import build_splits, deduplicate, describe, load_splits, save_splits

CAPS = SplitConfig(max_train=1000, max_val=1000, max_test=1000, val_fraction=0.1, test_fraction=0.2)


def frame(n=200, split="train", offset=0):
    return pd.DataFrame(
        {"text": [f"text number {i + offset}" for i in range(n)], "label": [i % 2 for i in range(n)], "split": split}
    )


def ids(part):
    return set(part["id"])


def test_duplicates_keep_the_training_copy_and_drop_conflicts():
    data = pd.DataFrame(
        {
            "text": ["same", " same ", "clash", "clash", "only test", ""],
            "label": [1, 1, 0, 1, 0, 1],
            "split": ["test", "train", "train", "train", "test", "train"],
        }
    )
    out = deduplicate(data)
    assert out["text"].tolist() == ["same", "only test"]
    assert out["split"].tolist() == ["train", "test"]


def test_carves_test_and_val_when_the_source_has_none():
    splits = build_splits(frame(), CAPS, seed=1)
    assert (len(splits.train), len(splits.val), len(splits.test)) == (144, 16, 40)
    assert not ids(splits.train) & ids(splits.test) and not ids(splits.train) & ids(splits.val)
    assert splits.test["label"].mean() == pytest.approx(0.5)  # stratified


def test_keeps_official_splits_and_applies_caps():
    data = pd.concat([frame(300), frame(50, "val", 1000), frame(400, "test", 2000)], ignore_index=True)
    caps = SplitConfig(max_train=100, max_val=1000, max_test=120, val_fraction=0.1, test_fraction=0.2)
    splits = build_splits(data, caps, seed=3)
    assert (len(splits.train), len(splits.val), len(splits.test)) == (100, 50, 120)
    assert splits.val["text"].str.startswith("text number 1").all()
    assert splits.test["id"].min() >= 350  # only rows that the source marked as test


def test_same_seed_same_split_other_seed_other_split():
    a, b, c = (build_splits(frame(), CAPS, seed) for seed in (7, 7, 8))
    assert ids(a.test) == ids(b.test)
    assert ids(a.test) != ids(c.test)


def test_round_trip_keeps_texts_that_look_like_numbers(tmp_path):
    data = pd.DataFrame({"text": ["123", "nan", "null", "1e5", "true", "ok"] * 10, "label": [0, 1] * 30})
    data["text"] = data["text"] + [f" {i}" if i >= 6 else "" for i in range(60)]
    data["split"] = "train"
    splits = build_splits(data, CAPS, seed=0)
    save_splits(splits, tmp_path / "set")
    loaded = load_splits(tmp_path / "set")
    for name, part in splits.items():
        pd.testing.assert_frame_equal(part, loaded[name], check_dtype=False)
    assert describe(splits, ("neg", "pos")) == describe(loaded, ("neg", "pos"))


def test_describe_fingerprints_the_ids():
    splits = build_splits(frame(), CAPS, seed=1)
    info = describe(splits, ("negative", "positive"))
    assert info["test"]["size"] == 40
    assert info["test"]["labels"] == {"negative": 20, "positive": 20}
    assert len(info["test"]["ids_sha256"]) == 64
