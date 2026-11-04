"""Label your own sentences with a model saved by `sbb run --save-models`."""

from __future__ import annotations

from pathlib import Path


def predict_texts(model_dir: str | Path, texts: list[str], max_length: int = 512) -> list[tuple[str, float]]:
    """Return (label, probability) for each text, using a saved fine-tuned encoder."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(model_dir)
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"{model_dir} has no saved model; run `sbb run --save-models DIR` with a finetune method")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    with torch.inference_mode():
        batch = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        probabilities = model(**batch).logits.softmax(-1)
    best = probabilities.argmax(-1)
    return [(model.config.id2label[int(i)], float(probabilities[row, i])) for row, i in enumerate(best)]
