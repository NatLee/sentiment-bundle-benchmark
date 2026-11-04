"""The CPU-only pieces of the methods: TF-IDF end to end, and LLM prompt building."""

import random

import pandas as pd
import pytest
from conftest import fake_review

from sentiment_bundle_benchmark.config import MethodConfig
from sentiment_bundle_benchmark.datasets import get_spec
from sentiment_bundle_benchmark.methods import MethodInput, run_method
from sentiment_bundle_benchmark.methods.prompting import PromptError, build_prompt, label_token_ids, pick_demos
from sentiment_bundle_benchmark.splits import Splits


def toy_splits(n=60, seed=0):
    rng = random.Random(seed)

    def part(size, start):
        labels = [i % 2 for i in range(size)]
        return pd.DataFrame({"id": range(start, start + size),
                             "text": [fake_review(rng, y, 12) for y in labels], "label": labels})

    return Splits(train=part(n, 0), val=part(20, n), test=part(30, n + 20))


def test_tfidf_learns_an_easy_task():
    method = MethodConfig("tfidf", "tfidf", "TF-IDF", {"c_grid": [1.0, 4.0], "min_df": 1, "solver": "liblinear"})
    out = run_method(method, MethodInput(spec=get_spec("imdb"), splits=toy_splits(), seed=0))
    test = toy_splits().test
    assert len(out.predictions) == len(test)
    assert sum(p == y for p, y in zip(out.predictions, test["label"])) / len(test) > 0.9
    assert out.params > 0 and out.size_mb > 0 and out.details["c"] in (1.0, 4.0)


class WordTokenizer:
    """Stands in for a Hugging Face tokenizer: one token per word, ids from a fixed vocabulary."""

    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.setdefault(word, len(self.vocab)) for word in text.split()]

    def decode(self, ids):
        reverse = {i: w for w, i in self.vocab.items()}
        return " ".join(reverse[i] for i in ids)


def test_label_tokens_cover_both_spellings():
    ids = label_token_ids(WordTokenizer(), ("negative", "positive"))
    assert [len(group) for group in ids] == [2, 2]


def test_label_tokens_must_differ():
    class FirstLetter(WordTokenizer):
        def encode(self, text, add_special_tokens=False):
            return [ord(text[0].lower())]

    with pytest.raises(PromptError):
        label_token_ids(FirstLetter(), ("subjective", "smart"))


def test_demos_are_balanced_and_reproducible():
    train = toy_splits().train
    demos = pick_demos(train, 8, 2, seed=3)
    assert len(demos) == 8 and sum(label for _, label in demos) == 4
    assert demos == pick_demos(train, 8, 2, seed=3)
    assert pick_demos(train, 0, 2, seed=3) == []


def test_prompt_lists_examples_then_the_question():
    spec = get_spec("subj")
    prompt = build_prompt(spec, "The film follows a detective.", [("I adored it.", "subjective")])
    assert prompt.startswith(spec.task) and "subjective or objective" in prompt
    assert prompt.index("Answer: subjective") < prompt.index("Text: The film follows a detective.")
    assert prompt.endswith("Answer:")
