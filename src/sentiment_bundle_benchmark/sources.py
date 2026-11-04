"""Download the raw dataset files and check them against the SHA-256 sums in the config.

Nothing here is ever committed: files land in <data_dir>/raw, which is git-ignored.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .config import BenchmarkConfig, SourceConfig
from .datasets import get_spec

USER_AGENT = "sentiment-bundle-benchmark (+https://github.com/NatLee/sentiment-bundle-benchmark)"
CHUNK = 1 << 20


class ChecksumError(RuntimeError):
    """A downloaded file does not match the checksum pinned in the config."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    actual = sha256_of(path)
    if actual != expected:
        raise ChecksumError(
            f"{path.name}: SHA-256 is {actual}, expected {expected}. "
            "The upstream file changed; check it and update the config on purpose."
        )


def _fetch(url: str, target: Path, *, quiet: bool = False) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                for block in iter(lambda: response.read(CHUNK), b""):
                    tmp.write(block)
                    done += len(block)
                    if not quiet and total:
                        print(f"\r  {target.name}: {done / total:6.1%} of {total / 1e6:.1f} MB", end="", file=sys.stderr)
            if not quiet:
                print(file=sys.stderr)
        except BaseException:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise
    shutil.move(tmp_path, target)


def fetch_source(source: SourceConfig, raw_dir: Path, names: list[str], *, quiet: bool = False) -> list[Path]:
    """Make sure each named file of `source` is in raw_dir with the right checksum. Returns the paths."""
    unknown = set(names) - set(source.files)
    if unknown:
        raise KeyError(f"source {source.name!r} has no checksum for {sorted(unknown)}; add it to the config")
    wanted = {name: raw_dir / name for name in names}
    missing = [name for name, path in wanted.items() if not path.exists()]

    if missing and source.archive:
        archive = raw_dir / source.archive
        if not quiet:
            print(f"downloading {source.name} ({source.description or source.url})", file=sys.stderr)
        _fetch(source.url, archive, quiet=quiet)
        with zipfile.ZipFile(archive) as bundle:
            for name in missing:
                with bundle.open(name) as src, open(wanted[name], "wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
        archive.unlink()
    else:
        for name in missing:
            if not quiet:
                print(f"downloading {name} from {source.name}", file=sys.stderr)
            _fetch(source.url.format(file=name), wanted[name], quiet=quiet)

    for name, path in wanted.items():
        verify(path, source.files[name])
    return list(wanted.values())


def download(config: BenchmarkConfig, datasets: list[str] | None = None, *, quiet: bool = False) -> list[Path]:
    """Fetch the raw files behind `datasets` (default: all datasets in the config)."""
    needed: dict[str, list[str]] = {}
    for name in datasets or config.datasets:
        spec = get_spec(name)
        needed.setdefault(spec.source, []).append(spec.file)
    paths: list[Path] = []
    for source_name, names in needed.items():
        paths.extend(fetch_source(config.sources[source_name], config.raw_dir, names, quiet=quiet))
    return paths
