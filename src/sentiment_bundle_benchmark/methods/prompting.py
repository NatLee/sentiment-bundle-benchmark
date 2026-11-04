"""Prompt building for the LLM method. Pure Python, so it is testable without a GPU stack."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..datasets import DatasetSpec


class PromptError(ValueError):
    """The labels cannot be told apart by their first token with this tokenizer."""


def label_token_ids(tokenizer, labels: tuple[str, ...]) -> list[list[int]]:
    """For each label, the first-token ids of its spellings ("positive", "Positive")."""
    ids = []
    for label in labels:
        spellings = {label, label.capitalize()}
        ids.append(sorted({tokenizer.encode(s, add_special_tokens=False)[0] for s in spellings}))
    flat = [i for group in ids for i in group]
    if len(flat) != len(set(flat)):
        raise PromptError(f"labels {labels} share a first token; pick label words that start differently")
    return ids


def truncate(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return text if len(ids) <= max_tokens else tokenizer.decode(ids[:max_tokens]).strip() + " ..."


def pick_demos(train: pd.DataFrame, shots: int, n_labels: int, seed: int) -> list[tuple[str, int]]:
    """`shots` training examples, as balanced across labels as possible, in a fixed shuffled order."""
    if shots == 0:
        return []
    rng = np.random.default_rng(seed)
    per_label = [shots // n_labels + (1 if i < shots % n_labels else 0) for i in range(n_labels)]
    picked = []
    for label, count in enumerate(per_label):
        pool = train.index[train["label"] == label].to_numpy()
        for i in rng.choice(pool, size=count, replace=False):
            picked.append((train.at[i, "text"], label))
    order = rng.permutation(len(picked))
    return [picked[i] for i in order]


def build_prompt(spec: DatasetSpec, text: str, demos: list[tuple[str, str]]) -> str:
    """The user message. `demos` holds (already truncated text, label word) pairs."""
    lines = [f"{spec.task} Answer with one word: {' or '.join(spec.labels)}.", ""]
    if demos:
        lines += ["Examples:", ""]
        for demo_text, label in demos:
            lines += [f"Text: {demo_text}", f"Answer: {label}", ""]
    lines += [f"Text: {text}", "Answer:"]
    return "\n".join(lines)


def prompt_template(spec: DatasetSpec, demo_labels: list[str]) -> str:
    """The user message with placeholders instead of texts, for the run record."""
    demos = [(f"{{example {i} from the training split}}", label) for i, label in enumerate(demo_labels, 1)]
    return build_prompt(spec, "{text from the test split}", demos)
