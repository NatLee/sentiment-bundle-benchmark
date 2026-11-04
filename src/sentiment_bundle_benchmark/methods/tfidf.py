"""Word and bigram TF-IDF features with logistic regression: the classic CPU baseline."""

from __future__ import annotations

import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import MethodConfig
from .base import MethodInput, MethodOutput, Stopwatch
from .linear import fit_logreg


def run(method: MethodConfig, data: MethodInput) -> MethodOutput:
    train, val, test = data.splits.train, data.splits.val, data.splits.test
    train_clock, infer_clock = Stopwatch(), Stopwatch()

    with train_clock.running():
        vectorizer = TfidfVectorizer(
            ngram_range=(1, int(method.get("ngram_max", 2))),
            min_df=int(method.get("min_df", 2)),
            max_features=method.get("max_features"),
            sublinear_tf=True,
        )
        x_train = vectorizer.fit_transform(train["text"])
        model, details = fit_logreg(
            x_train, train["label"], vectorizer.transform(val["text"]), val["label"],
            c_grid=method.require("c_grid"), n_labels=len(data.spec.labels), seed=data.seed,
            solver=method.get("solver", "lbfgs"),
        )

    with infer_clock.running():
        predictions = model.predict(vectorizer.transform(test["text"])).tolist()

    weights = model.coef_.size + model.intercept_.size
    return MethodOutput(
        predictions=predictions,
        train_seconds=train_clock.seconds,
        infer_seconds=infer_clock.seconds,
        params=int(weights),
        size_mb=len(pickle.dumps((vectorizer, model))) / 2**20,
        details={**details, "vocabulary": len(vectorizer.vocabulary_)},
    )
