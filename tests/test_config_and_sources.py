import hashlib
import zipfile

import pytest
from conftest import REPO

from sentiment_bundle_benchmark.config import ConfigError, SourceConfig, load_config
from sentiment_bundle_benchmark.datasets import get_spec, pick_layout
from sentiment_bundle_benchmark.sources import ChecksumError, fetch_source


@pytest.mark.parametrize("name", ["benchmark.toml", "smoke.toml"])
def test_shipped_configs_load_and_cover_every_dataset_file(name):
    config = load_config(REPO / "configs" / name, root=REPO)
    assert config.methods and config.datasets
    for dataset in config.datasets:
        pick_layout(get_spec(dataset), config.sources)  # raises if no source lists the files it needs


def test_full_config_reads_every_dataset_from_the_parquet_bundle():
    config = load_config(REPO / "configs" / "benchmark.toml", root=REPO)
    for dataset in config.datasets:
        layout, source = pick_layout(get_spec(dataset), config.sources)
        assert source.name == "bundle" and all(f.endswith(".parquet") for f in layout.files)
    old = {k: v for k, v in config.sources.items() if k != "bundle"}  # the 2023 files still work as a fallback
    assert {pick_layout(get_spec(d), old)[1].name for d in config.datasets} == {"bundle-2023", "malcod"}


def test_full_config_pins_every_model_revision():
    config = load_config(REPO / "configs" / "benchmark.toml", root=REPO)
    assert {m.kind for m in config.methods} == {"tfidf", "embedding", "finetune", "llm"}
    for method in config.methods:
        if method.kind != "tfidf":
            assert len(method.require("revision")) == 40


def test_config_errors_are_readable(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text('seed = 1\ndatasets = ["imdb"]\n[splits]\nmax_train = 1\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.toml")
    kinds = (REPO / "configs" / "smoke.toml").read_text(encoding="utf-8").replace('kind = "tfidf"', 'kind = "magic"')
    (tmp_path / "kinds.toml").write_text(kinds, encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown kind"):
        load_config(tmp_path / "kinds.toml")


def _source(tmp_path, payload: bytes, *, archive: bool, digest: str | None = None) -> SourceConfig:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    if archive:
        with zipfile.ZipFile(upstream / "bundle.zip", "w") as bundle:
            bundle.writestr("data.csv", payload)
        url = (upstream / "bundle.zip").as_uri()
    else:
        (upstream / "data.csv").write_bytes(payload)
        url = upstream.as_uri() + "/{file}"
    return SourceConfig(name="test", url=url, files={"data.csv": digest or hashlib.sha256(payload).hexdigest()},
                        archive="bundle.zip" if archive else None)


@pytest.mark.parametrize("archive", [False, True])
def test_download_checks_the_sha256(tmp_path, archive):
    source = _source(tmp_path, b"text,label\nhi,1\n", archive=archive)
    raw = tmp_path / "raw"
    [path] = fetch_source(source, raw, ["data.csv"], quiet=True)
    assert path.read_bytes() == b"text,label\nhi,1\n"
    assert sorted(p.name for p in raw.iterdir()) == ["data.csv"]  # the archive is removed after extraction


def test_download_rejects_a_changed_file(tmp_path):
    source = _source(tmp_path, b"changed upstream", archive=False, digest="0" * 64)
    with pytest.raises(ChecksumError, match="expected"):
        fetch_source(source, tmp_path / "raw", ["data.csv"], quiet=True)
