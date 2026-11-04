"""End to end on fixture data with tiny random models: prepare -> run every kind -> report -> predict.

Needs the [models] extra and network access to the Hugging Face Hub. Run with `pytest -m smoke`.
"""

import json
import shutil

import pytest
from conftest import REPO

from sentiment_bundle_benchmark.cli import main

pytestmark = pytest.mark.smoke


def test_whole_pipeline(raw_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    shutil.copy(REPO / "configs" / "smoke.toml", tmp_path / "smoke.toml")
    readme = tmp_path / "README.md"
    readme.write_text("# demo\n<!-- results:start -->\nold\n<!-- results:end -->\n", encoding="utf-8")

    assert main(["-c", "smoke.toml", "prepare"]) == 0
    assert main(["-c", "smoke.toml", "run", "--save-models", "models"]) == 0

    runs = sorted((tmp_path / "results-smoke" / "runs").glob("*/*.json"))
    assert len(runs) == 4 * 2  # four method kinds, two datasets
    for path in runs:
        record = json.loads(path.read_text(encoding="utf-8"))
        assert 0.0 <= record["metrics"]["macro_f1"] <= 1.0
        test_file = tmp_path / "data" / "processed" / record["dataset"] / "test.jsonl"
        assert record["n_test"] == len(test_file.read_text(encoding="utf-8").splitlines())

    assert main(["-c", "smoke.toml", "report", "--readme", "README.md"]) == 0
    assert "Leaderboard" in readme.read_text(encoding="utf-8")
    assert (tmp_path / "docs" / "images" / "leaderboard-dark.png").exists()

    capsys.readouterr()
    model_dir = tmp_path / "models" / "tiny-modernbert-ft" / "myanimelist"
    assert main(["predict", "--model-dir", str(model_dir), "What a great show", "Boring and awful"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2 and all(line.split()[0] in {"negative", "positive"} for line in lines)
