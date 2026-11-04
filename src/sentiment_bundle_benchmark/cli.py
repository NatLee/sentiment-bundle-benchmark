"""Command line: sbb download | prepare | run | report | list | predict."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import ConfigError, load_config
from .datasets import DATASETS, DataFormatError
from .sources import ChecksumError

DEFAULT_CONFIG = "configs/benchmark.toml"

EPILOG = """\
typical run:
  sbb download                      fetch the raw files (about 300 MB) into data/raw
  sbb prepare                       build the fixed splits into data/processed
  sbb run --methods tfidf-logreg    run one method on every dataset (results/runs/...)
  sbb run                           run everything that has no result yet
  sbb report --readme README.md     rebuild tables and charts
"""


def _names(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sbb",
        description="Benchmark sentiment classifiers on NatLee's dataset bundle and MyAnimeList reviews.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG, help=f"benchmark config (default: {DEFAULT_CONFIG})")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def with_datasets(p):
        p.add_argument("-d", "--datasets", type=_names, metavar="A,B",
                       help=f"comma-separated subset (default: all in the config; known: {', '.join(DATASETS)})")
        return p

    with_datasets(sub.add_parser("download", help="download the raw dataset files and check their SHA-256"))
    with_datasets(sub.add_parser("prepare", help="build the train/val/test splits and results/splits.json"))

    run = with_datasets(sub.add_parser("run", help="run methods on datasets and save one JSON per run"))
    run.add_argument("-m", "--methods", type=_names, metavar="A,B", help="comma-separated method names (see `sbb list`)")
    run.add_argument("--force", action="store_true", help="redo runs that already have a result file")
    run.add_argument("--save-models", type=Path, metavar="DIR",
                     help="keep fine-tuned models under DIR/<method>/<dataset> for `sbb predict`")

    rep = sub.add_parser("report", help="write results/leaderboard.md and the charts")
    rep.add_argument("--images", type=Path, default=Path("docs/images"), help="chart folder (default: docs/images)")
    rep.add_argument("--readme", type=Path, help="also refresh the results block of this README")
    rep.add_argument("--no-charts", action="store_true", help="tables only")

    sub.add_parser("list", help="show the datasets and methods in the config")

    pred = sub.add_parser("predict", help="label your own text with a model saved by `run --save-models`")
    pred.add_argument("--model-dir", type=Path, required=True, help="e.g. models/modernbert-base-ft/myanimelist")
    pred.add_argument("texts", nargs="+", help="one or more texts (quote each one)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "predict":
            from .predict import predict_texts

            for text, (label, probability) in zip(args.texts, predict_texts(args.model_dir, args.texts)):
                print(f"{label:<10} {probability:6.1%}  {text}")
            return 0

        config = load_config(args.config)
        if args.command == "download":
            from .sources import download

            for path in download(config, args.datasets):
                print(f"ok  {path}")
        elif args.command == "prepare":
            from .runner import prepare

            prepare(config, args.datasets)
        elif args.command == "run":
            from .runner import run

            run(config, args.methods, args.datasets, force=args.force, save_models=args.save_models)
        elif args.command == "report":
            from .report import report

            print(report(config, image_dir=args.images, readme=args.readme, charts=not args.no_charts))
        elif args.command == "list":
            print("datasets:")
            for name in config.datasets:
                spec = DATASETS[name]
                print(f"  {name:<12} {spec.title}  [{', '.join(spec.labels)}]")
            print("methods:")
            for method in config.methods:
                print(f"  {method.name:<28} {method.kind:<9} {method.label}")
    except (ConfigError, ChecksumError, DataFormatError, FileNotFoundError, KeyError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"sbb: error: {message}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
