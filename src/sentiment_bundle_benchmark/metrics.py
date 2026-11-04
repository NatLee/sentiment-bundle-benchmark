"""Scores reported for every run: accuracy, macro-F1, the confusion matrix and bootstrap intervals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

BOOTSTRAP_ROUNDS = 1000


def confusion(y_true: Sequence[int], y_pred: Sequence[int], n_labels: int) -> np.ndarray:
    """matrix[i, j] = how many examples of true class i were predicted as class j."""
    matrix = np.zeros((n_labels, n_labels), dtype=np.int64)
    np.add.at(matrix, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return matrix


def _scores(matrix: np.ndarray) -> tuple[float, float]:
    """Accuracy and macro-F1 from one confusion matrix (or a stack of them on axis 0)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)
    predicted = matrix.sum(axis=-2)
    actual = matrix.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(predicted + actual > 0, 2 * tp / (predicted + actual), 0.0)
    accuracy = tp.sum(axis=-1) / matrix.sum(axis=(-2, -1))
    return accuracy, f1.mean(axis=-1)


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], n_labels: int) -> float:
    return float(_scores(confusion(y_true, y_pred, n_labels))[1])


def evaluate(y_true: Sequence[int], y_pred: Sequence[int], n_labels: int, *, seed: int = 0) -> dict:
    """Point scores plus 95% bootstrap intervals (resampling test examples with replacement)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("y_true and y_pred must be non-empty and the same length")
    matrix = confusion(y_true, y_pred, n_labels)
    acc, f1 = _scores(matrix)

    rng = np.random.default_rng(seed)
    cell = y_true * n_labels + y_pred
    draws = rng.integers(0, len(cell), size=(BOOTSTRAP_ROUNDS, len(cell)))
    counts = np.apply_along_axis(np.bincount, 1, cell[draws], minlength=n_labels * n_labels)
    boot_acc, boot_f1 = _scores(counts.reshape(BOOTSTRAP_ROUNDS, n_labels, n_labels))

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "accuracy_ci95": [float(x) for x in np.percentile(boot_acc, [2.5, 97.5])],
        "macro_f1_ci95": [float(x) for x in np.percentile(boot_f1, [2.5, 97.5])],
        "confusion_matrix": matrix.tolist(),
        "n": int(len(y_true)),
    }
