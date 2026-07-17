"""Frozen sample-weight contract for the public calibrated-RF model."""

from __future__ import annotations

import pandas as pd


FORMAL_HIGH_RNA_NEGATIVE_WEIGHT = 2.0


def formal_rf_sample_weights(train_scores: pd.DataFrame) -> pd.Series:
    """Return the one fixed RF weighting rule used by DuoDose.

    Ordinary singlets and all other training rows retain unit weight. Only
    constructed high-RNA singlets receive the frozen factor of two.
    """

    if "true_label" not in train_scores:
        raise ValueError("formal RF sample weights require the semi-real true_label column")
    weights = pd.Series(1.0, index=train_scores.index, name="sample_weight", dtype=float)
    weights.loc[train_scores["true_label"].astype(str).eq("high_RNA_singlet")] = FORMAL_HIGH_RNA_NEGATIVE_WEIGHT
    return weights

