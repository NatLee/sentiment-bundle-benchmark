"""The six datasets: where each file comes from, how to read it, and what the labels mean.

Every reader returns a DataFrame with the columns
  text  - the input text
  label - an integer class id (an index into `DatasetSpec.labels`)
  split - "train", "val" or "test" when the source defines it, otherwise "train"
so the split builder can treat all datasets the same way.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .safe_pickle import load_pickle

SPLIT_NAMES = ("train", "val", "test")


class DataFormatError(ValueError):
    """A raw file does not look the way the reader expects."""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    short: str  # display name in tables and charts
    title: str
    source: str  # key of a [sources.*] table in the config
    file: str
    reader: Callable[..., pd.DataFrame]
    labels: tuple[str, ...]
    unit: str  # what one example is: "review", "sentence", "phrase"
    task: str  # one-line task description, used in LLM and embedding prompts
    about: str
    homepage: str
    citation: str
    licence: str


# ---------------------------------------------------------------- readers

_SPLIT_ALIASES = {"train": "train", "dev": "val", "valid": "val", "validation": "val", "test": "test"}


def read_senteval_frame(path: Path, **_: Any) -> pd.DataFrame:
    """MR, SUBJ, MPQA and SST2: a pickled DataFrame with sentence / label / split columns."""
    return parse_senteval_frame(load_pickle(path), path.name)


def parse_senteval_frame(frame: Any, source: str = "data") -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not {"sentence", "label", "split"} <= set(frame.columns):
        raise DataFormatError(f"{source}: expected a DataFrame with sentence, label and split columns")
    unknown = set(frame["split"]) - set(_SPLIT_ALIASES)
    if unknown:
        raise DataFormatError(f"{source}: unknown split names {sorted(unknown)}")
    return pd.DataFrame(
        {
            "text": frame["sentence"].astype(str),
            "label": frame["label"].astype(int),
            "split": frame["split"].map(_SPLIT_ALIASES),
        }
    )


# In the bundle's IMDB file every training class is stored as [12,500 real reviews, 12,500 fake ones].
# The fake half was produced by GURA's data augmentation (random word spans glued together), which
# makes every fake review almost exactly the corpus' average length. Real reviews vary a lot.
IMDB_REAL_MIN_STDEV = 30.0  # words; real halves measure about 82-90
IMDB_FAKE_MAX_STDEV = 10.0  # words; fake halves measure about 4.5


def _word_count_stdev(texts: list[str]) -> float:
    return statistics.pstdev(len(text.split()) for text in texts)


def read_imdb_bundle(path: Path, **_: Any) -> pd.DataFrame:
    """aclImdb.pkl: (test_texts, test_labels, train_texts, train_labels), with fake training reviews removed."""
    return parse_imdb_tuple(load_pickle(path), path.name)


def parse_imdb_tuple(data: Any, source: str = "data") -> pd.DataFrame:
    if not (isinstance(data, tuple) and len(data) == 4):
        raise DataFormatError(f"{source}: expected a 4-tuple (x_test, y_test, x_train, y_train)")
    x_test, y_test, x_train, y_train = data

    real_x: list[str] = []
    real_y: list[int] = []
    for label in sorted(set(y_train)):
        block = [x for x, y in zip(x_train, y_train) if y == label]
        half = len(block) // 2
        real, fake = block[:half], block[half:]
        if not (
            len(block) % 2 == 0
            and _word_count_stdev(real) >= IMDB_REAL_MIN_STDEV
            and _word_count_stdev(fake) <= IMDB_FAKE_MAX_STDEV
        ):
            raise DataFormatError(
                f"{source}: training class {label} is not laid out as [real reviews, generated reviews]; "
                "the file may have been re-uploaded, so check it before trusting the IMDB numbers"
            )
        real_x.extend(real)
        real_y.extend([label] * half)

    return pd.DataFrame(
        {
            "text": list(real_x) + list(x_test),
            "label": [int(y) for y in real_y] + [int(y) for y in y_test],
            "split": ["train"] * len(real_x) + ["test"] * len(x_test),
        }
    )


def read_malcod(path: Path, *, positive_min_rating: int, test_from: str, **_: Any) -> pd.DataFrame:
    """MALCoD reviews: full review text, overall rating 1-10 turned into negative/positive, split by post date."""
    frame = pd.read_csv(path, escapechar="\\", dtype=str, keep_default_na=False)
    needed = {"review", "overallRating", "postTime"}
    if not needed <= set(frame.columns):
        raise DataFormatError(f"{path.name}: missing columns {sorted(needed - set(frame.columns))}")
    rating = pd.to_numeric(frame["overallRating"], errors="coerce")
    valid = rating.between(1, 10)  # a few rows carry 0 or 11, which the site's 1-10 scale cannot produce
    frame, rating = frame[valid], rating[valid]
    posted = pd.to_datetime(frame["postTime"], errors="coerce")
    if posted.isna().any():
        raise DataFormatError(f"{path.name}: {int(posted.isna().sum())} rows have an unreadable postTime")
    return pd.DataFrame(
        {
            "text": frame["review"].str.split().str.join(" "),  # the scrape kept runs of layout spaces
            "label": (rating >= positive_min_rating).astype(int),
            "split": (posted >= pd.Timestamp(test_from)).map({True: "test", False: "train"}),
        }
    ).reset_index(drop=True)


# ---------------------------------------------------------------- registry

DATASETS: dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in (
        DatasetSpec(
            name="myanimelist",
            short="MyAnimeList",
            title="MyAnimeList reviews (MALCoD)",
            source="malcod",
            file="animeReviewsOrderByTime.csv",
            reader=read_malcod,
            labels=("negative", "positive"),
            unit="review",
            task="Classify the sentiment of an anime review as positive or negative.",
            about="135K MyAnimeList.net reviews (2006-2019) with the reviewer's overall 1-10 rating.",
            homepage="https://www.kaggle.com/datasets/natlee/myanimelist-comment-dataset",
            citation='N. Lee, "MyAnimeList Comment Dataset (MALCoD)," Kaggle, 2019.',
            licence="ODbL 1.0 (database) / DbCL 1.0 (contents), as listed on Kaggle",
        ),
        DatasetSpec(
            name="imdb",
            short="IMDB",
            title="IMDB movie reviews (aclImdb)",
            source="bundle",
            file="aclImdb.pkl",
            reader=read_imdb_bundle,
            labels=("negative", "positive"),
            unit="review",
            task="Classify the sentiment of a movie review as positive or negative.",
            about="50K polar IMDB reviews, 25K train / 25K test (Maas et al., 2011).",
            homepage="https://ai.stanford.edu/~amaas/data/sentiment/",
            citation="A. L. Maas et al., \"Learning Word Vectors for Sentiment Analysis,\" ACL 2011.",
            licence="No licence stated; free for research, cite the paper",
        ),
        DatasetSpec(
            name="sst2",
            short="SST-2",
            title="Stanford Sentiment Treebank, binary (SST-2)",
            source="bundle",
            file="SST2.pkl",
            reader=read_senteval_frame,
            labels=("negative", "positive"),
            unit="sentence",
            task="Classify the sentiment of a movie review sentence as positive or negative.",
            about="Movie-review sentences and phrases from Rotten Tomatoes (Socher et al., 2013).",
            homepage="https://nlp.stanford.edu/sentiment/",
            citation="R. Socher et al., \"Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank,\" EMNLP 2013.",
            licence="No licence stated; free for research, cite the paper",
        ),
        DatasetSpec(
            name="mr",
            short="MR",
            title="Movie Review sentence polarity (MR)",
            source="bundle",
            file="MR.pkl",
            reader=read_senteval_frame,
            labels=("negative", "positive"),
            unit="sentence",
            task="Classify the sentiment of a movie review snippet as positive or negative.",
            about="5,331 positive and 5,331 negative Rotten Tomatoes snippets (Pang and Lee, 2005).",
            homepage="https://www.cs.cornell.edu/people/pabo/movie-review-data/",
            citation="B. Pang and L. Lee, \"Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales,\" ACL 2005.",
            licence="No licence stated; free for research, cite the paper",
        ),
        DatasetSpec(
            name="subj",
            short="SUBJ",
            title="Subjectivity (SUBJ)",
            source="bundle",
            file="SUBJ.pkl",
            reader=read_senteval_frame,
            labels=("subjective", "objective"),
            unit="sentence",
            task="Classify a movie sentence as subjective (an opinion) or objective (a plot fact).",
            about="5,000 subjective review snippets and 5,000 objective plot sentences (Pang and Lee, 2004).",
            homepage="https://www.cs.cornell.edu/people/pabo/movie-review-data/",
            citation="B. Pang and L. Lee, \"A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts,\" ACL 2004.",
            licence="No licence stated; free for research, cite the paper",
        ),
        DatasetSpec(
            name="mpqa",
            short="MPQA",
            title="MPQA opinion polarity",
            source="bundle",
            file="MPQA.pkl",
            reader=read_senteval_frame,
            labels=("negative", "positive"),
            unit="phrase",
            task="Classify the opinion in a short news phrase as positive or negative.",
            about="10,606 short opinion phrases from news articles (Wiebe et al., 2005).",
            homepage="https://mpqa.cs.pitt.edu/",
            citation="J. Wiebe, T. Wilson and C. Cardie, \"Annotating Expressions of Opinions and Emotions in Language,\" LRE 2005.",
            licence="Research and academic use only (the news texts); see the homepage",
        ),
    )
}

def get_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; known: {', '.join(DATASETS)}") from None


def read_dataset(spec: DatasetSpec, raw_dir: Path, options: dict[str, Any] | None = None) -> pd.DataFrame:
    path = raw_dir / spec.file
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `sbb download` first")
    frame = spec.reader(path, **(options or {}))
    bad = set(frame["label"]) - set(range(len(spec.labels)))
    if bad:
        raise DataFormatError(f"{spec.name}: labels {sorted(bad)} are outside 0..{len(spec.labels) - 1}")
    return frame
