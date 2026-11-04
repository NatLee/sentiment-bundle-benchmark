"""Load the bundle's pickle files without letting them run arbitrary code.

A pickle can call any importable function while it loads. The bundle files only need plain
Python containers plus the handful of pandas/numpy classes that rebuild a DataFrame, so every
other global is refused. If pandas ever moves one of these classes, loading fails loudly here
(see tests/test_safe_pickle.py) instead of silently doing something else.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

# Exactly the globals referenced by the bundle files (found with `pickletools.genops`).
ALLOWED_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        ("builtins", "slice"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("pandas.core.frame", "DataFrame"),
        ("pandas.core.internals.managers", "BlockManager"),
        ("pandas.core.indexes.base", "Index"),
        ("pandas.core.indexes.base", "_new_Index"),
        ("pandas.core.indexes.range", "RangeIndex"),
    }
)


class UnsafePickleError(pickle.UnpicklingError):
    """The file asked for a Python object that is not on the allow-list."""


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in ALLOWED_GLOBALS:
            raise UnsafePickleError(f"refusing to load {module}.{name} from a pickle file")
        return super().find_class(module, name)


def load_pickle(path: str | Path) -> Any:
    """Unpickle `path`, allowing only the globals in ALLOWED_GLOBALS."""
    with open(path, "rb") as handle:
        return _RestrictedUnpickler(handle).load()
