"""Tiny, synthetic stand-ins for the raw files, written in the same formats as the real ones."""

from __future__ import annotations

import csv
import pickle
import random
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

POSITIVE_WORDS = "great wonderful loved brilliant fun moving superb charming".split()
NEGATIVE_WORDS = "awful boring hated dull mess tedious weak painful".split()
FILLER = "the story cast scene ending music plot characters episode season".split()


def fake_review(rng: random.Random, label: int, length: int) -> str:
    mood = POSITIVE_WORDS if label == 1 else NEGATIVE_WORDS
    return " ".join(rng.choice(mood) if i % 3 == 0 else rng.choice(FILLER) for i in range(length))


def write_imdb_pickle(path: Path, per_class: int = 40, seed: int = 0) -> None:
    """(x_test, y_test, x_train, y_train) where each training class is [real half, generated half]."""
    rng = random.Random(seed)
    x_train, y_train = [], []
    for label in (1, 0):  # the bundle stores positives first
        real = [fake_review(rng, label, rng.randint(5, 220)) for _ in range(per_class)]
        generated = [fake_review(rng, label, 100 + rng.randint(-3, 3)) for _ in range(per_class)]
        x_train += real + generated
        y_train += [label] * (2 * per_class)
    x_test = [fake_review(rng, label, rng.randint(5, 220)) for label in (1, 0) for _ in range(per_class)]
    y_test = [label for label in (1, 0) for _ in range(per_class)]
    with open(path, "wb") as handle:
        pickle.dump((x_test, y_test, x_train, y_train), handle, protocol=3)


MALCOD_COLUMNS = ["id", "workId", "reviewId", "workName", "postTime", "episodesSeen", "author",
                  "peopleFoundUseful", "overallRating", "storyRating", "animationRating", "soundRating",
                  "characterRating", "enjoymentRating", "review"]


def write_malcod_csv(path: Path, rows: list[tuple[str, int, str]]) -> None:
    """rows: (postTime, overallRating, review). Quotes are escaped with a backslash, like the real file."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL, escapechar="\\", doublequote=False)
        writer.writerow(MALCOD_COLUMNS)
        for i, (posted, rating, review) in enumerate(rows):
            writer.writerow([i, 1, i, "Some_Show", posted, 12, f"user{i}", 3, rating, 5, 5, 5, 5, 5, review])


def malcod_rows(n: int = 120, seed: int = 0) -> list[tuple[str, int, str]]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        rating = rng.randint(1, 10)
        year = 2012 + i % 8  # 2012-2019, so both sides of a 2018 cut-off are filled
        text = fake_review(rng, int(rating >= 6), rng.randint(20, 80))
        rows.append((f"{year}-05-0{1 + i % 9} 12:00:00", rating, text))
    return rows


def write_bundle_parquet(root: Path, config: str, frame) -> list[Path]:
    """The Parquet bundle layout: <config>/<split>.parquet with text and label columns."""
    names = {"train": "train", "val": "validation", "test": "test"}
    paths = []
    for split in ("train", "val", "test"):
        part = frame[frame["split"] == split]
        if len(part):
            path = root / config / f"{names[split]}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            part[["text", "label"]].to_parquet(path, index=False)
            paths.append(path)
    return paths


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """data/raw with the two fixture-friendly datasets (IMDB pickle and MALCoD CSV)."""
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    write_imdb_pickle(raw / "aclImdb.pkl")
    write_malcod_csv(raw / "animeReviewsOrderByTime.csv", malcod_rows())
    return raw
