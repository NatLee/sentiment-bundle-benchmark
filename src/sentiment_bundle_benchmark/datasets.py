"""The six datasets: where each file comes from, how to read it, and what the labels mean.

A dataset can be stored in more than one layout: the Parquet files of the current Hugging Face bundle,
the 2023 pickles (still reachable through their pinned revision), or the MALCoD CSV on Kaggle. The
config's [sources.*] tables list which files exist; the first layout whose files a source lists wins.

Every reader returns a DataFrame with the columns
  text  - the input text
  label - an integer class id (an index into `DatasetSpec.labels`)
  split - "train", "val" or "test" when the source defines it, otherwise "train"
so the split builder can treat all datasets the same way. All layouts of one dataset return the same
rows in the same order, so the split fingerprints do not depend on where the data came from.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from .safe_pickle import load_pickle

if TYPE_CHECKING:
    from .config import SourceConfig

SPLIT_NAMES = ("train", "val", "test")


class DataFormatError(ValueError):
    """A raw file does not look the way the reader expects."""


@dataclass(frozen=True)
class Layout:
    """One way a dataset is stored: the files (relative to the source) and the function that reads them."""

    files: tuple[str, ...]
    reader: Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    short: str  # display name in tables and charts
    title: str
    layouts: tuple[Layout, ...]  # in order of preference
    labels: tuple[str, ...]
    unit: str  # what one example is: "review", "sentence", "phrase"
    task: str  # one-line task description, used in LLM and embedding prompts
    about: str
    homepage: str
    citation: str
    licence: str


# ---------------------------------------------------------------- readers

_SPLIT_ALIASES = {"train": "train", "dev": "val", "valid": "val", "validation": "val", "test": "test"}


def read_parquet_splits(paths: list[Path], **_: Any) -> pd.DataFrame:
    """Bundle Parquet layout: one file per split, named after it, each with text and label columns."""
    parts = []
    for path in paths:
        split = _SPLIT_ALIASES.get(path.stem)
        if split is None:
            raise DataFormatError(f"{path}: the file name should be a split name (train, validation, test)")
        frame = pd.read_parquet(path, columns=["text", "label"])
        parts.append(pd.DataFrame({"text": frame["text"].astype(str), "label": frame["label"].astype(int),
                                   "split": split}))
    return pd.concat(parts, ignore_index=True)


def read_senteval_frame(paths: list[Path], **_: Any) -> pd.DataFrame:
    """MR, SUBJ, MPQA and SST2 in the 2023 bundle: a pickled DataFrame with sentence / label / split columns."""
    return parse_senteval_frame(load_pickle(paths[0]), paths[0].name)


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


# In the 2023 bundle's IMDB file every training class is stored as [12,500 real reviews, 12,500 fake
# ones]. The fake half was produced by GURA's data augmentation (random word spans glued together),
# which makes every fake review almost exactly the corpus' average length. Real reviews vary a lot.
# The Parquet bundle no longer contains the fake half.
IMDB_REAL_MIN_STDEV = 30.0  # words; real halves measure about 82-90
IMDB_FAKE_MAX_STDEV = 10.0  # words; fake halves measure about 4.5


def _word_count_stdev(texts: list[str]) -> float:
    return statistics.pstdev(len(text.split()) for text in texts)


def read_imdb_bundle(paths: list[Path], **_: Any) -> pd.DataFrame:
    """aclImdb.pkl: (test_texts, test_labels, train_texts, train_labels), with fake training reviews removed."""
    return parse_imdb_tuple(load_pickle(paths[0]), paths[0].name)


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


def _reviews(text: pd.Series, rating: pd.Series, posted: pd.Series, source: str, *,
             positive_min_rating: int, test_from: str, **_: Any) -> pd.DataFrame:
    """MyAnimeList rows as the benchmark uses them, whichever file they came from."""
    rating = pd.to_numeric(rating, errors="coerce")
    valid = rating.between(1, 10)  # a few rows carry 0 or 11, which the site's 1-10 scale cannot produce
    posted = pd.to_datetime(posted[valid], errors="coerce")
    if posted.isna().any():
        raise DataFormatError(f"{source}: {int(posted.isna().sum())} rows have an unreadable post time")
    return pd.DataFrame(
        {
            "text": text[valid].astype(str).str.split().str.join(" "),  # the scrape kept runs of layout spaces
            "label": (rating[valid] >= positive_min_rating).astype(int),
            "split": (posted >= pd.Timestamp(test_from)).map({True: "test", False: "train"}),
        }
    ).reset_index(drop=True)


def read_malcod(paths: list[Path], **options: Any) -> pd.DataFrame:
    """MALCoD CSV from Kaggle: full reviews, overall rating 1-10 turned into negative/positive, split by date."""
    path = paths[0]
    frame = pd.read_csv(path, escapechar="\\", dtype=str, keep_default_na=False)
    needed = {"review", "overallRating", "postTime"}
    if not needed <= set(frame.columns):
        raise DataFormatError(f"{path.name}: missing columns {sorted(needed - set(frame.columns))}")
    return _reviews(frame["review"], frame["overallRating"], frame["postTime"], path.name, **options)


def read_myanimelist_parquet(paths: list[Path], **options: Any) -> pd.DataFrame:
    """The bundle's myanimelist config: the same MALCoD reviews with rating and post_time columns."""
    frame = pd.read_parquet(paths[0], columns=["text", "rating", "post_time"])
    return _reviews(frame["text"], frame["rating"], frame["post_time"], paths[0].name, **options)


def _parquet(config: str, *splits: str) -> Layout:
    return Layout(tuple(f"{config}/{split}.parquet" for split in splits), read_parquet_splits)


# ---------------------------------------------------------------- registry

DATASETS: dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in (
        DatasetSpec(
            name="myanimelist",
            short="MyAnimeList",
            title="MyAnimeList reviews (MALCoD)",
            layouts=(
                Layout(("myanimelist/train.parquet",), read_myanimelist_parquet),
                Layout(("animeReviewsOrderByTime.csv",), read_malcod),
            ),
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
            layouts=(_parquet("aclImdb", "train", "test"), Layout(("aclImdb.pkl",), read_imdb_bundle)),
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
            layouts=(_parquet("SST2", "train", "validation", "test"), Layout(("SST2.pkl",), read_senteval_frame)),
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
            layouts=(_parquet("MR", "train"), Layout(("MR.pkl",), read_senteval_frame)),
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
            layouts=(_parquet("SUBJ", "train"), Layout(("SUBJ.pkl",), read_senteval_frame)),
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
            layouts=(_parquet("MPQA", "train"), Layout(("MPQA.pkl",), read_senteval_frame)),
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


def pick_layout(spec: DatasetSpec, sources: Mapping[str, SourceConfig]) -> tuple[Layout, SourceConfig]:
    """The first layout of `spec` whose files are all listed (with checksums) by one of the sources."""
    for layout in spec.layouts:
        for source in sources.values():
            if all(name in source.files for name in layout.files):
                return layout, source
    wanted = " or ".join("+".join(layout.files) for layout in spec.layouts)
    raise DataFormatError(f"{spec.name}: no source in the config lists {wanted}")


def read_dataset(spec: DatasetSpec, raw_dir: Path, sources: Mapping[str, SourceConfig],
                 options: dict[str, Any] | None = None) -> pd.DataFrame:
    layout, _ = pick_layout(spec, sources)
    paths = [raw_dir / name for name in layout.files]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{', '.join(missing)} missing; run `sbb download` first")
    frame = layout.reader(paths, **(options or {}))
    bad = set(frame["label"]) - set(range(len(spec.labels)))
    if bad:
        raise DataFormatError(f"{spec.name}: labels {sorted(bad)} are outside 0..{len(spec.labels) - 1}")
    return frame
