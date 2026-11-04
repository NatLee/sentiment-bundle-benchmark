"""Turn the run records into Markdown tables and charts (light and dark PNGs)."""

from __future__ import annotations

import json
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import BenchmarkConfig
from .datasets import get_spec

README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"
HISTORICAL_FILE = "historical/gura.json"

# Chart theme: one accent hue plus gray for context, and a one-hue blue ramp for magnitude.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "accent": "#2a78d6", "context": "#898781"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "accent": "#3987e5", "context": "#898781"},
}
HEATMAP_LOW_PERCENTILE = 10
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


@dataclass
class Results:
    records: list[dict]
    datasets: list[str]
    methods: list[str]  # ordered best first by mean macro-F1
    labels: dict[str, str]

    def get(self, method: str, dataset: str) -> dict | None:
        for record in self.records:
            if record["method"] == method and record["dataset"] == dataset:
                return record
        return None

    def mean(self, method: str, metric: str) -> float | None:
        values = [r["metrics"][metric] for d in self.datasets if (r := self.get(method, d))]
        return statistics.fmean(values) if len(values) == len(self.datasets) else None


def load_results(config: BenchmarkConfig) -> Results:
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(config.runs_dir.glob("*/*.json"))]
    known = {m.name for m in config.methods}
    records = [r for r in records if r["method"] in known and r["dataset"] in config.datasets]
    datasets = [d for d in config.datasets if any(r["dataset"] == d for r in records)]
    labels = {m.name: m.label for m in config.methods}
    results = Results(records, datasets, [m.name for m in config.methods if any(r["method"] == m.name for r in records)], labels)
    results.methods.sort(key=lambda m: -(results.mean(m, "macro_f1") or -1.0))
    return results


# ---------------------------------------------------------------- formatting helpers

def pct(value: float | None, digits: int = 1) -> str:
    return "–" if value is None else f"{100 * value:.{digits}f}"


def human_count(n: int | None) -> str:
    if n is None:
        return "–"
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= size:
            return f"{n / size:.1f}{unit}".replace(".0", "")
    return str(n)


def human_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def short_name(dataset: str) -> str:
    return get_spec(dataset).short


def _table(header: list[str], rows: list[list[str]], align: list[str]) -> str:
    sep = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------- tables

def leaderboard_table(results: Results, metric: str = "macro_f1") -> str:
    best = {d: max((r["metrics"][metric] for r in results.records if r["dataset"] == d), default=None)
            for d in results.datasets}
    rows = []
    for method in results.methods:
        cells = []
        for dataset in results.datasets:
            record = results.get(method, dataset)
            value = record["metrics"][metric] if record else None
            text = pct(value)
            cells.append(f"**{text}**" if value is not None and value == best[dataset] else text)
        rows.append([results.labels[method], *cells, pct(results.mean(method, metric))])
    header = ["Method", *map(short_name, results.datasets), "Mean"]
    return _table(header, rows, ["l"] + ["r"] * (len(header) - 1))


def cost_table(results: Results) -> str:
    rows = []
    for method in results.methods:
        runs = [r for d in results.datasets if (r := results.get(method, d))]
        speeds = [r["throughput_per_s"] for r in runs if r["throughput_per_s"]]
        vram = [r["peak_vram_mb"] for r in runs if r["peak_vram_mb"] is not None]
        rows.append([
            results.labels[method],
            human_count(runs[0]["params"]) if runs else "–",
            human_seconds(sum(r["train_seconds"] for r in runs)) if any(r["train_seconds"] for r in runs) else "none",
            f"{statistics.median(speeds):,.0f}" if speeds else "–",
            f"{max(vram) / 1024:.1f} GB" if vram else "CPU only",
        ])
    header = ["Method", "Parameters", "Training (all sets)", "Inference, examples/s (median)", "Peak VRAM"]
    return _table(header, rows, ["l", "r", "r", "r", "r"])


def dataset_tables(results: Results) -> str:
    parts = []
    for dataset in results.datasets:
        spec = get_spec(dataset)
        runs = sorted((r for r in results.records if r["dataset"] == dataset), key=lambda r: -r["metrics"]["macro_f1"])
        if not runs:
            continue
        sizes = f"{runs[0]['n_train']:,} train / {runs[0]['n_val']:,} val / {runs[0]['n_test']:,} test"
        rows = []
        for r in runs:
            m = r["metrics"]
            low, high = m["macro_f1_ci95"]
            rows.append([
                results.labels[r["method"]], pct(m["accuracy"]), pct(m["macro_f1"]),
                f"{pct(low)}–{pct(high)}", human_seconds(r["train_seconds"]) if r["train_seconds"] else "none",
                f"{r['throughput_per_s']:,.0f}" if r["throughput_per_s"] else "–",
            ])
        table = _table(["Method", "Accuracy", "Macro-F1", "95% CI (F1)", "Training", "Examples/s"], rows,
                       ["l", "r", "r", "r", "r", "r"])
        parts.append(f"#### {spec.title}\n\n{sizes}; labels: {' / '.join(spec.labels)}.\n\n{table}")
    return "\n\n".join(parts)


def historical_table(results: Results, historical: dict) -> str:
    rows = []
    for dataset in results.datasets:
        old = historical["values"].get(dataset, {})
        runs = [r for r in results.records if r["dataset"] == dataset]
        best = max(runs, key=lambda r: r["metrics"]["accuracy"]) if runs else None
        rows.append([
            short_name(dataset),
            pct(old.get("claim_2019")),
            ("≈ " + pct(old.get("chart_2023"), 0)) if old.get("chart_2023") is not None else "–",
            f"{pct(best['metrics']['accuracy'])} ({results.labels[best['method']]})" if best else "–",
        ])
    return _table(["Dataset", "GURA 2019 claim", "GURA 2023 charts", "Best now, test accuracy"], rows,
                  ["l", "r", "r", "l"])


def render_markdown(results: Results, historical: dict | None, image_dir: str) -> str:
    sections = [
        "### Leaderboard: macro-F1 on the held-out test splits (%)\n\n"
        "Bold marks the best score per dataset. Macro-F1 averages the F1 of both classes, so a model "
        "cannot look good by always guessing the bigger class.\n\n" + leaderboard_table(results),
        _picture(image_dir, "leaderboard", "Heatmap of macro-F1 per method and dataset"),
        "### Accuracy (%)\n\n" + leaderboard_table(results, "accuracy"),
        "### What it costs\n\n" + cost_table(results),
        _picture(image_dir, "speed-vs-quality", "Mean macro-F1 against inference speed for each method"),
    ]
    if historical:
        sections.append(
            "### GURA, for the record\n\n" + historical["note"] + "\n\n" + historical_table(results, historical)
        )
        sections.append(_picture(image_dir, "gura-then-now", "GURA's reported accuracy next to the best accuracy today"))
    sections.append("### Per-dataset breakdown\n\n" + dataset_tables(results))
    return "\n\n".join(sections) + "\n"


def _picture(image_dir: str, stem: str, alt: str) -> str:
    return (
        "<picture>\n"
        f'  <source media="(prefers-color-scheme: dark)" srcset="{image_dir}/{stem}-dark.png">\n'
        f'  <img alt="{alt}" src="{image_dir}/{stem}.png" width="100%">\n'
        "</picture>"
    )


def update_readme(readme: Path, block: str) -> None:
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.S)
    if not pattern.search(text):
        raise ValueError(f"{readme} has no {README_START} ... {README_END} block")
    readme.write_text(pattern.sub(lambda _: f"{README_START}\n{block}{README_END}", text), encoding="utf-8")


# ---------------------------------------------------------------- charts

def _style(ax, theme):
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
    ax.tick_params(colors=theme["ink2"], labelsize=10, length=0)
    ax.grid(color=theme["grid"], linewidth=1, linestyle="-")
    ax.set_axisbelow(True)


def _save(fig, path: Path, theme) -> None:
    fig.savefig(path, dpi=160, facecolor=theme["surface"], bbox_inches="tight", pad_inches=0.25)


def chart_heatmap(results: Results, path: Path, theme: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    values = [[(r["metrics"]["macro_f1"] if (r := results.get(m, d)) else float("nan")) for d in results.datasets]
              for m in results.methods]
    flat = [v for row in values for v in row if v == v]
    # One failed run (e.g. a zero-shot score near chance) would wash every other cell into the same
    # dark blue, so the colour scale starts at a low percentile and anything below it gets the lightest step.
    low, high = float(np.percentile(flat, HEATMAP_LOW_PERCENTILE)), max(flat)
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(results.methods) + 1.6))
    fig.patch.set_facecolor(theme["surface"])
    ax.imshow(values, cmap=cmap, vmin=low, vmax=high, aspect="auto")
    ax.set_xticks(range(len(results.datasets)), [short_name(d) for d in results.datasets])
    ax.set_yticks(range(len(results.methods)), [results.labels[m] for m in results.methods])
    ax.xaxis.tick_top()
    ax.tick_params(colors=theme["ink"], labelsize=10, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(results.datasets))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(results.methods))], minor=True)
    ax.grid(which="minor", color=theme["surface"], linewidth=2)
    ax.tick_params(which="minor", length=0)
    span = (high - low) or 1.0
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            if value == value:
                dark_cell = (value - low) / span > 0.55
                ax.text(j, i, f"{100 * value:.1f}", ha="center", va="center", fontsize=10,
                        color="#ffffff" if dark_cell else "#0b0b0b")
    ax.set_title("Macro-F1 on held-out test sets (%) · darker is better", loc="left", color=theme["ink"],
                 fontsize=12, pad=28)
    _save(fig, path, theme)
    plt.close(fig)


def chart_speed(results: Results, path: Path, theme: dict) -> None:
    import matplotlib.pyplot as plt

    points = []
    for method in results.methods:
        speeds = [r["throughput_per_s"] for d in results.datasets if (r := results.get(method, d)) and r["throughput_per_s"]]
        f1 = results.mean(method, "macro_f1")
        if speeds and f1 is not None:
            points.append((statistics.median(speeds), 100 * f1, results.labels[method]))
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    fig.patch.set_facecolor(theme["surface"])
    _style(ax, theme)
    ax.set_xscale("log")
    ax.scatter([p[0] for p in points], [p[1] for p in points], s=70, color=theme["accent"],
               edgecolors=theme["surface"], linewidths=2, zorder=3)
    for x, y, label in points:
        ax.annotate(label, (x, y), xytext=(8, 6), textcoords="offset points", fontsize=9.5, color=theme["ink"])
    ax.set_xlabel("Inference speed, examples per second (median over datasets, log scale)", color=theme["ink2"])
    ax.set_ylabel("Mean macro-F1 (%)", color=theme["ink2"])
    ax.set_title("Quality against speed · up and to the right is better", loc="left", color=theme["ink"], fontsize=12)
    xs = [p[0] for p in points]
    ax.set_xlim(min(xs) / 1.6, max(xs) * 6)
    _save(fig, path, theme)
    plt.close(fig)


def chart_then_now(results: Results, historical: dict, path: Path, theme: dict) -> None:
    import matplotlib.pyplot as plt

    rows = []
    for dataset in results.datasets:
        old = historical["values"].get(dataset, {}).get("chart_2023")
        runs = [r for r in results.records if r["dataset"] == dataset]
        if old is None or not runs:
            continue
        rows.append((short_name(dataset), 100 * old, 100 * max(r["metrics"]["accuracy"] for r in runs)))
    fig, ax = plt.subplots(figsize=(9.5, 0.6 * len(rows) + 1.8))
    fig.patch.set_facecolor(theme["surface"])
    _style(ax, theme)
    ax.grid(axis="y", visible=False)
    for i, (_, old, new) in enumerate(rows):
        ax.plot([old, new], [i, i], color=theme["axis"], linewidth=2, zorder=1, solid_capstyle="round")
    ax.scatter([r[1] for r in rows], range(len(rows)), s=70, color=theme["context"], edgecolors=theme["surface"],
               linewidths=2, zorder=3, label="GURA, 2023 training charts (validation, best epoch)")
    ax.scatter([r[2] for r in rows], range(len(rows)), s=70, color=theme["accent"], edgecolors=theme["surface"],
               linewidths=2, zorder=3, label="Best method today (held-out test)")
    for i, (_, _, new) in enumerate(rows):
        ax.annotate(f"{new:.1f}", (new, i), xytext=(9, -4), textcoords="offset points", fontsize=9.5, color=theme["ink"])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(min(r[1] for r in rows) - 5, 103)
    ax.set_xlabel("Accuracy (%)", color=theme["ink2"])
    ax.set_title("GURA then, best method now · accuracy", loc="left", color=theme["ink"], fontsize=12, pad=30)
    legend = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9.5,
                       labelcolor=theme["ink2"], borderaxespad=0.2)
    for handle in legend.legend_handles:
        handle.set_sizes([60])
    _save(fig, path, theme)
    plt.close(fig)


def build_charts(results: Results, historical: dict | None, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for mode, theme in THEMES.items():
        suffix = "" if mode == "light" else "-dark"
        jobs = [("leaderboard", chart_heatmap), ("speed-vs-quality", chart_speed)]
        for stem, draw in jobs:
            path = out_dir / f"{stem}{suffix}.png"
            draw(results, path, theme)
            written.append(path)
        if historical:
            path = out_dir / f"gura-then-now{suffix}.png"
            chart_then_now(results, historical, path, theme)
            written.append(path)
    return written


def load_historical(config: BenchmarkConfig) -> dict | None:
    path = config.results_dir / HISTORICAL_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def report(config: BenchmarkConfig, *, image_dir: Path, readme: Path | None = None, charts: bool = True) -> str:
    results = load_results(config)
    if not results.records:
        raise FileNotFoundError(f"no run results under {config.runs_dir}; run `sbb run` first")
    historical = load_historical(config)
    if charts:
        build_charts(results, historical, image_dir)
    markdown = render_markdown(results, historical, _relative(image_dir, config.results_dir))
    (config.results_dir / "leaderboard.md").write_text(markdown, encoding="utf-8")
    if readme is not None:
        update_readme(readme, render_markdown(results, historical, _relative(image_dir, readme.parent)))
    return markdown


def _relative(target: Path, start: Path) -> str:
    """Path of `target` as written in a Markdown file that lives in `start`."""
    return Path(os.path.relpath(target.resolve(), start.resolve())).as_posix()
