import pytest
from sklearn.metrics import accuracy_score, f1_score

from sentiment_bundle_benchmark.metrics import accuracy, confusion, evaluate, macro_f1

Y_TRUE = [0, 0, 1, 1, 1, 0, 1, 0, 0, 1]
Y_PRED = [0, 1, 1, 1, 0, 0, 1, 0, 1, 1]


def test_hand_computed_example():
    # Class 0: tp=3, fp=1, fn=2 -> F1 = 6/9. Class 1: tp=4, fp=2, fn=1 -> F1 = 8/11.
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(0.7)
    assert macro_f1(Y_TRUE, Y_PRED, 2) == pytest.approx((6 / 9 + 8 / 11) / 2)
    assert confusion(Y_TRUE, Y_PRED, 2).tolist() == [[3, 2], [1, 4]]


def test_matches_scikit_learn():
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(accuracy_score(Y_TRUE, Y_PRED))
    assert macro_f1(Y_TRUE, Y_PRED, 2) == pytest.approx(f1_score(Y_TRUE, Y_PRED, average="macro"))


def test_majority_guess_gets_low_macro_f1():
    y_true = [1] * 80 + [0] * 20
    scores = evaluate(y_true, [1] * 100, 2)
    assert scores["accuracy"] == pytest.approx(0.8)
    assert scores["macro_f1"] == pytest.approx((0 + 2 * 80 / 180) / 2)


def test_bootstrap_interval_is_reproducible_and_brackets_the_score():
    first = evaluate(Y_TRUE * 20, Y_PRED * 20, 2, seed=5)
    second = evaluate(Y_TRUE * 20, Y_PRED * 20, 2, seed=5)
    assert first == second
    low, high = first["macro_f1_ci95"]
    assert low <= first["macro_f1"] <= high
    assert first["n"] == 200


def test_rejects_mismatched_input():
    with pytest.raises(ValueError):
        evaluate([0, 1], [0], 2)
