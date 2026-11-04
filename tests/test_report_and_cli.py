import json
import shutil

import pytest
from conftest import REPO

from sentiment_bundle_benchmark.cli import main
from sentiment_bundle_benchmark.config import load_config
from sentiment_bundle_benchmark.report import README_END, README_START, leaderboard_table, load_results, report


def fake_run(runs_dir, method, dataset, f1, speed=100.0):
    record = {
        "method": method, "dataset": dataset, "kind": "tfidf",
        "metrics": {"accuracy": f1 + 0.01, "macro_f1": f1, "macro_f1_ci95": [f1 - 0.02, f1 + 0.02]},
        "n_train": 10, "n_val": 2, "n_test": 5, "train_seconds": 3.0, "infer_seconds": 1.0,
        "throughput_per_s": speed, "peak_vram_mb": None, "params": 1234,
    }
    path = runs_dir / method / f"{dataset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


@pytest.fixture
def config(tmp_path):
    shutil.copy(REPO / "configs" / "smoke.toml", tmp_path / "smoke.toml")
    config = load_config(tmp_path / "smoke.toml", root=tmp_path)
    fake_run(config.runs_dir, "tfidf-logreg", "myanimelist", 0.80, speed=5000)
    fake_run(config.runs_dir, "tfidf-logreg", "imdb", 0.90, speed=4000)
    fake_run(config.runs_dir, "tiny-modernbert-ft", "myanimelist", 0.85, speed=300)
    fake_run(config.runs_dir, "tiny-modernbert-ft", "imdb", 0.88, speed=200)
    fake_run(config.runs_dir, "not-in-config", "imdb", 0.99)
    historical = config.results_dir / "historical" / "gura.json"
    historical.parent.mkdir(parents=True)
    historical.write_text(json.dumps({"note": "Old numbers.", "values": {"imdb": {"chart_2023": 0.73}}}), encoding="utf-8")
    return config


def test_leaderboard_orders_by_mean_and_marks_the_best(config):
    results = load_results(config)
    assert results.methods == ["tiny-modernbert-ft", "tfidf-logreg"]  # 86.5 mean beats 85.0; unknown runs ignored
    table = leaderboard_table(results)
    assert "**85.0**" in table and "**90.0**" in table
    assert table.splitlines()[2].startswith("| tiny-modernbert-ft | **85.0** | 88.0 | 86.5 |")


def test_report_writes_tables_charts_and_readme_block(config, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{README_START}\nstale\n{README_END}\noutro\n", encoding="utf-8")
    markdown = report(config, image_dir=tmp_path / "docs" / "images", readme=readme)
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("intro\n") and text.endswith("outro\n") and "stale" not in text
    assert 'srcset="docs/images/leaderboard-dark.png"' in text
    assert "GURA, for the record" in markdown and "≈ 73" in markdown
    for stem in ("leaderboard", "speed-vs-quality", "gura-then-now"):
        assert (tmp_path / "docs" / "images" / f"{stem}.png").stat().st_size > 1000


def test_readme_without_markers_is_an_error(config, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("no markers here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="results:start"):
        report(config, image_dir=tmp_path / "img", readme=readme, charts=False)


def test_cli_help_and_list(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "sbb download" in capsys.readouterr().out
    monkeypatch.chdir(REPO)
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "myanimelist" in out and "qwen3.5-2b-8-shot" in out


def test_cli_reports_errors_without_a_traceback(tmp_path, capsys):
    assert main(["-c", str(tmp_path / "nope.toml"), "list"]) == 2
    assert "config file not found" in capsys.readouterr().err
    shutil.copy(REPO / "configs" / "smoke.toml", tmp_path / "smoke.toml")
    assert main(["-c", str(tmp_path / "smoke.toml"), "run", "--methods", "nope"]) == 2
    assert "unknown method" in capsys.readouterr().err
