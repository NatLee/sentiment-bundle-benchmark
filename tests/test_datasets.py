import pandas as pd
import pytest
from conftest import malcod_rows, write_bundle_parquet, write_malcod_csv

from sentiment_bundle_benchmark.config import SourceConfig
from sentiment_bundle_benchmark.datasets import (
    DATASETS,
    DataFormatError,
    get_spec,
    parse_imdb_tuple,
    parse_senteval_frame,
    pick_layout,
    read_dataset,
    read_imdb_bundle,
    read_malcod,
    read_myanimelist_parquet,
    read_parquet_splits,
)


def test_senteval_frame_maps_labels_and_split_names():
    frame = pd.DataFrame(
        {"sentence": ["good", "bad", "fine"], "label": ["1", "0", "1"], "split": ["train", "dev", "test"]}
    )
    out = parse_senteval_frame(frame)
    assert out["label"].tolist() == [1, 0, 1]
    assert out["split"].tolist() == ["train", "val", "test"]
    assert out["text"].tolist() == ["good", "bad", "fine"]


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"text": ["x"], "label": ["1"], "split": ["train"]}),
        pd.DataFrame({"sentence": ["x"], "label": ["1"], "split": ["holdout"]}),
        ["not", "a", "frame"],
    ],
)
def test_senteval_frame_rejects_unexpected_shapes(frame):
    with pytest.raises(DataFormatError):
        parse_senteval_frame(frame)


def test_imdb_reader_drops_the_generated_half(raw_dir):
    out = read_imdb_bundle([raw_dir / "aclImdb.pkl"])
    train, test = out[out["split"] == "train"], out[out["split"] == "test"]
    assert len(train) == 80 and len(test) == 80  # 40 real per class kept, 40 generated per class dropped
    assert sorted(train["label"].value_counts().tolist()) == [40, 40]
    assert train["text"].str.split().str.len().std() > 30  # only the varied-length real reviews remain


def test_imdb_reader_refuses_an_unexpected_layout():
    texts = [" ".join(["word"] * n) for n in range(10, 210, 5)]  # varied lengths in both halves
    with pytest.raises(DataFormatError, match="not laid out"):
        parse_imdb_tuple((texts, [1] * len(texts), texts, [1] * len(texts)))


def test_malcod_reader_labels_splits_and_cleans(tmp_path):
    rows = [
        ("2017-12-31 23:59:00", 6, 'Good show, I said "wow"   twice.'),
        ("2018-01-01 00:00:00", 5, "Meh."),
        ("2016-03-03 10:00:00", 0, "Broken rating."),
        ("2019-02-02 10:00:00", 11, "Also broken."),
        ("2015-01-01 10:00:00", 1, "Terrible\nwith a line break."),
    ]
    path = tmp_path / "reviews.csv"
    write_malcod_csv(path, rows)
    out = read_malcod([path], positive_min_rating=6, test_from="2018-01-01")
    assert out["text"].tolist() == ['Good show, I said "wow" twice.', "Meh.", "Terrible with a line break."]
    assert out["label"].tolist() == [1, 0, 0]
    assert out["split"].tolist() == ["train", "test", "train"]


def source(name, *files):
    return SourceConfig(name=name, url="unused/{file}", files={f: "0" * 64 for f in files})


OPTIONS = {"positive_min_rating": 6, "test_from": "2018-01-01"}


def test_read_dataset_checks_files_and_labels(tmp_path):
    write_malcod_csv(tmp_path / "animeReviewsOrderByTime.csv", malcod_rows(20))
    sources = {"malcod": source("malcod", "animeReviewsOrderByTime.csv"), "bundle": source("bundle", "aclImdb.pkl")}
    frame = read_dataset(get_spec("myanimelist"), tmp_path, sources, OPTIONS)
    assert set(frame["label"]) <= {0, 1}
    with pytest.raises(FileNotFoundError, match="sbb download"):
        read_dataset(get_spec("imdb"), tmp_path, sources)


def test_layout_is_picked_from_the_files_the_config_lists():
    parquet = source("bundle", "aclImdb/train.parquet", "aclImdb/test.parquet", "myanimelist/train.parquet")
    pickles = source("bundle-2023", "aclImdb.pkl")
    csv = source("malcod", "animeReviewsOrderByTime.csv")
    layout, chosen = pick_layout(get_spec("imdb"), {"old": pickles, "new": parquet})
    assert layout.files == ("aclImdb/train.parquet", "aclImdb/test.parquet") and chosen is parquet
    assert pick_layout(get_spec("imdb"), {"old": pickles})[0].files == ("aclImdb.pkl",)
    assert pick_layout(get_spec("myanimelist"), {"csv": csv})[1] is csv
    assert pick_layout(get_spec("myanimelist"), {"csv": csv, "new": parquet})[1] is parquet
    with pytest.raises(DataFormatError, match="no source"):
        pick_layout(get_spec("mr"), {"old": pickles})


def test_parquet_layout_gives_the_same_rows_as_the_pickle(raw_dir, tmp_path):
    from_pickle = read_imdb_bundle([raw_dir / "aclImdb.pkl"])
    write_bundle_parquet(tmp_path, "aclImdb", from_pickle)
    from_parquet = read_parquet_splits([tmp_path / "aclImdb" / "train.parquet", tmp_path / "aclImdb" / "test.parquet"])
    pd.testing.assert_frame_equal(from_pickle, from_parquet, check_dtype=False)


def test_parquet_layout_maps_split_names_and_rejects_others(tmp_path):
    frame = pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 1, 1], "split": ["train", "val", "test"]})
    paths = write_bundle_parquet(tmp_path, "SST2", frame)
    assert read_parquet_splits(paths)["split"].tolist() == ["train", "val", "test"]
    (tmp_path / "odd.parquet").write_bytes(paths[0].read_bytes())
    with pytest.raises(DataFormatError, match="split name"):
        read_parquet_splits([tmp_path / "odd.parquet"])


def test_myanimelist_parquet_matches_the_csv_reader(tmp_path):
    rows = malcod_rows(60)
    write_malcod_csv(tmp_path / "reviews.csv", rows + [("2015-01-01 10:00:00", 0, "Broken rating.")])
    from_csv = read_malcod([tmp_path / "reviews.csv"], **OPTIONS)
    published = pd.DataFrame({
        "text": [text for _, _, text in rows], "rating": pd.array([rating for _, rating, _ in rows], dtype="int8"),
        "post_time": pd.to_datetime([posted for posted, _, _ in rows]),
    })
    published.to_parquet(tmp_path / "train.parquet")
    pd.testing.assert_frame_equal(from_csv, read_myanimelist_parquet([tmp_path / "train.parquet"], **OPTIONS))


def test_registry_is_complete():
    assert set(DATASETS) == {"myanimelist", "imdb", "sst2", "mr", "subj", "mpqa"}
    for spec in DATASETS.values():
        assert len(spec.labels) == 2 and spec.citation and spec.homepage and spec.licence
        assert all(label.lower() in spec.task.lower() for label in spec.labels)
