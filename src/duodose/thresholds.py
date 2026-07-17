"""Prediction threshold strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def expected_rate_threshold(scores: pd.Series, expected_doublet_rate: float) -> float:
    values = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot threshold scores because none are finite")
    if not 0.0 < float(expected_doublet_rate) < 1.0:
        raise ValueError("expected_doublet_rate must be between 0 and 1")
    return float(np.quantile(finite, 1.0 - float(expected_doublet_rate), method="lower"))


def resolve_threshold(
    scores: pd.Series,
    *,
    strategy: str | None,
    expected_doublet_rate: float,
    probability_threshold: float,
) -> float | None:
    if strategy is None:
        return None
    if strategy == "expected_rate":
        return expected_rate_threshold(scores, expected_doublet_rate)
    if strategy == "probability":
        if not 0.0 <= float(probability_threshold) <= 1.0:
            raise ValueError("probability threshold must be between 0 and 1")
        return float(probability_threshold)
    raise ValueError("Unknown threshold strategy")
