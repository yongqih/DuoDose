import pandas as pd

from duodose.thresholds import expected_rate_threshold, resolve_threshold


def test_expected_rate_threshold() -> None:
    scores = pd.Series([0.1, 0.2, 0.3, 0.4, 0.9])
    threshold = expected_rate_threshold(scores, 0.2)
    assert threshold == 0.4


def test_fixed_probability_and_continuous() -> None:
    scores = pd.Series([0.1, 0.8])
    assert resolve_threshold(scores, strategy="probability", expected_doublet_rate=0.1, probability_threshold=0.5) == 0.5
    assert resolve_threshold(scores, strategy=None, expected_doublet_rate=0.1, probability_threshold=0.5) is None
