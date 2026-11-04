"""Logistic regression on fixed features, with C picked on the validation split."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..metrics import macro_f1


def fit_logreg(x_train, y_train, x_val, y_val, *, c_grid: Sequence[float], n_labels: int, seed: int,
               solver: str = "lbfgs"):
    """Fit one model per C, keep the one with the best validation macro-F1. Returns (model, details)."""
    best, best_score, scores = None, -1.0, {}
    for c in c_grid:
        model = LogisticRegression(C=c, solver=solver, max_iter=3000, random_state=seed)
        model.fit(x_train, np.asarray(y_train))
        score = macro_f1(y_val, model.predict(x_val), n_labels)
        scores[str(c)] = round(score, 4)
        if score > best_score:
            best, best_score = model, score
    return best, {"c": float(best.C), "val_macro_f1_by_c": scores}
