"""Fine-tune a pretrained encoder end to end, keeping the epoch with the best validation macro-F1."""

from __future__ import annotations

import math

import numpy as np
import torch

from ..config import MethodConfig
from ..metrics import evaluate
from ._torch import peak_mb, pick_device, pick_dtype, reset_peak, seed_everything, sync, weights_of
from .base import MethodInput, MethodOutput, Stopwatch

# Batches are formed from windows of this many batches, sorted by length inside each window,
# so a batch holds texts of similar length (less padding) while the order stays random.
SORT_WINDOW_BATCHES = 50


def _batches(lengths: list[int], batch_size: int, rng: np.random.Generator | None) -> list[list[int]]:
    order = rng.permutation(len(lengths)) if rng is not None else np.arange(len(lengths))
    window = batch_size * SORT_WINDOW_BATCHES
    batches: list[list[int]] = []
    for start in range(0, len(order), window):
        chunk = sorted(order[start : start + window], key=lambda i: lengths[i])
        batches.extend([list(map(int, chunk[i : i + batch_size])) for i in range(0, len(chunk), batch_size)])
    if rng is not None:
        rng.shuffle(batches)
    return batches


def _collate(tokenizer, encodings: list[list[int]], index: list[int], device: torch.device) -> dict[str, torch.Tensor]:
    padded = tokenizer.pad({"input_ids": [encodings[i] for i in index]}, return_tensors="pt")
    return {key: value.to(device) for key, value in padded.items()}


@torch.inference_mode()
def predict(model, tokenizer, encodings, batch_size, device, dtype) -> np.ndarray:
    model.eval()
    lengths = [len(e) for e in encodings]
    predictions = np.empty(len(encodings), dtype=np.int64)
    for index in _batches(lengths, batch_size * 2, rng=None):
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            logits = model(**_collate(tokenizer, encodings, index, device)).logits
        predictions[index] = logits.argmax(-1).cpu().numpy()
    return predictions


def run(method: MethodConfig, data: MethodInput) -> MethodOutput:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    seed_everything(data.seed)
    device = pick_device()
    dtype = pick_dtype(device)
    labels = data.spec.labels
    name, revision = method.require("model"), method.get("revision")
    batch_size = int(method.get("batch_size", 16))
    epochs = int(method.get("epochs", 2))

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    # Weights stay in float32 while training (some checkpoints, e.g. DeBERTa-v3, are stored in
    # float16 and would otherwise be trained in it, which overflows to NaN); autocast runs the maths in bf16.
    model = AutoModelForSequenceClassification.from_pretrained(
        name,
        revision=revision,
        dtype=torch.float32,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)},
    ).to(device)

    def encode(texts) -> list[list[int]]:
        return tokenizer(list(texts), truncation=True, max_length=int(method.get("max_length", 512)))["input_ids"]

    train, val, test = data.splits.train, data.splits.val, data.splits.test
    train_clock, infer_clock = Stopwatch(), Stopwatch()
    reset_peak(device)
    rng = np.random.default_rng(data.seed)
    history = []

    with train_clock.running():
        x_train, x_val = encode(train["text"]), encode(val["text"])
        y_train = torch.tensor(train["label"].to_numpy())
        lengths = [len(e) for e in x_train]
        steps = epochs * math.ceil(len(x_train) / batch_size)
        decay = [p for p in model.parameters() if p.ndim >= 2]
        no_decay = [p for p in model.parameters() if p.ndim < 2]
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": float(method.get("weight_decay", 0.01))},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=float(method.require("learning_rate")),
        )
        scheduler = get_linear_schedule_with_warmup(optimizer, int(float(method.get("warmup_ratio", 0.06)) * steps), steps)
        best_state, best_score = None, -1.0

        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for index in _batches(lengths, batch_size, rng):
                batch = _collate(tokenizer, x_train, index, device)
                with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                    logits = model(**batch).logits
                loss = torch.nn.functional.cross_entropy(logits.float(), y_train[index].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(loss.item())
            scores = evaluate(val["label"], predict(model, tokenizer, x_val, batch_size, device, dtype), len(labels))
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                            "val_accuracy": scores["accuracy"], "val_macro_f1": scores["macro_f1"]})
            print(f"    epoch {epoch}/{epochs}: loss {history[-1]['train_loss']:.4f}, val macro-F1 {scores['macro_f1']:.4f}", flush=True)
            if scores["macro_f1"] > best_score:
                best_score = scores["macro_f1"]
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        sync(device)

    with infer_clock.running():
        predictions = predict(model, tokenizer, encode(test["text"]), batch_size, device, dtype).tolist()
        sync(device)

    if data.save_dir is not None:
        model.save_pretrained(data.save_dir)
        tokenizer.save_pretrained(data.save_dir)

    params, size_mb = weights_of(model)
    best_epoch = max(history, key=lambda h: h["val_macro_f1"])["epoch"]
    return MethodOutput(
        predictions=predictions,
        train_seconds=train_clock.seconds,
        infer_seconds=infer_clock.seconds,
        params=params,
        size_mb=size_mb,
        peak_vram_mb=peak_mb(device),
        device=str(device),
        details={"history": history, "best_epoch": best_epoch, "train_steps": steps,
                 "longest_input_tokens": max(lengths)},
    )
