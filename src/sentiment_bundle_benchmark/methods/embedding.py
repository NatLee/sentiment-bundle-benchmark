"""Frozen sentence embeddings from a pretrained model, then logistic regression on top."""

from __future__ import annotations

from ..config import MethodConfig
from ._torch import peak_mb, pick_device, pick_dtype, reset_peak, seed_everything, sync, weights_of
from .base import MethodInput, MethodOutput, Stopwatch
from .linear import fit_logreg


def run(method: MethodConfig, data: MethodInput) -> MethodOutput:
    from sentence_transformers import SentenceTransformer

    seed_everything(data.seed)
    device = pick_device()
    encoder = SentenceTransformer(
        method.require("model"),
        revision=method.get("revision"),
        device=str(device),
        model_kwargs={"dtype": pick_dtype(device)},
    )
    encoder.max_seq_length = int(method.get("max_length", 512))
    prompt = method.get("prompt", "").format(task=data.spec.task) or None
    batch_size = int(method.get("batch_size", 32))

    def encode(texts):
        return encoder.encode(
            list(texts), batch_size=batch_size, prompt=prompt, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype("float32")

    train, val, test = data.splits.train, data.splits.val, data.splits.test
    train_clock, infer_clock = Stopwatch(), Stopwatch()
    reset_peak(device)

    with train_clock.running():
        x_train, x_val = encode(train["text"]), encode(val["text"])
        sync(device)
        classifier, details = fit_logreg(
            x_train, train["label"], x_val, val["label"],
            c_grid=method.require("c_grid"), n_labels=len(data.spec.labels), seed=data.seed,
            solver=method.get("solver", "lbfgs"),
        )

    with infer_clock.running():
        predictions = classifier.predict(encode(test["text"])).tolist()
        sync(device)

    params, size_mb = weights_of(encoder)
    return MethodOutput(
        predictions=predictions,
        train_seconds=train_clock.seconds,
        infer_seconds=infer_clock.seconds,
        params=params + classifier.coef_.size + classifier.intercept_.size,
        size_mb=size_mb,
        peak_vram_mb=peak_mb(device),
        device=str(device),
        details={**details, "embedding_dim": int(x_train.shape[1]), "max_length": encoder.max_seq_length},
    )
