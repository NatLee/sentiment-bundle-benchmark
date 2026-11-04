import pandas as pd
import pytest
from conftest import malcod_rows, write_malcod_csv

from sentiment_bundle_benchmark.datasets import (
    DATASETS,
    DataFormatError,
    get_spec,
    parse_imdb_tuple,
    parse_senteval_frame,
    read_dataset,
    read_imdb_bundle,
    read_malcod,
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
    out = read_imdb_bundle(raw_dir / "aclImdb.pkl")
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
    out = read_malcod(path, positive_min_rating=6, test_from="2018-01-01")
    assert out["text"].tolist() == ['Good show, I said "wow" twice.', "Meh.", "Terrible with a line break."]
    assert out["label"].tolist() == [1, 0, 0]
    assert out["split"].tolist() == ["train", "test", "train"]


def test_read_dataset_checks_files_and_labels(tmp_path):
    write_malcod_csv(tmp_path / "animeReviewsOrderByTime.csv", malcod_rows(20))
    frame = read_dataset(get_spec("myanimelist"), tmp_path, {"positive_min_rating": 6, "test_from": "2018-01-01"})
    assert set(frame["label"]) <= {0, 1}
    with pytest.raises(FileNotFoundError, match="sbb download"):
        read_dataset(get_spec("imdb"), tmp_path)


def test_registry_is_complete():
    assert set(DATASETS) == {"myanimelist", "imdb", "sst2", "mr", "subj", "mpqa"}
    for spec in DATASETS.values():
        assert len(spec.labels) == 2 and spec.citation and spec.homepage and spec.licence
        assert all(label.lower() in spec.task.lower() for label in spec.labels)
