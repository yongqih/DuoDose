"""Small, model-agnostic metric helpers used by DuoDose workflows."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def safe_auroc(y_true, y_score) -> float:
    """Return AUROC, or NaN when the requested comparison is undefined."""

    try:
        value = roc_auc_score(np.asarray(y_true), np.asarray(y_score, dtype=float))
    except (TypeError, ValueError):
        return float("nan")
    return float(value) if np.isfinite(value) else float("nan")


def safe_auprc(y_true, y_score) -> float:
    """Return average precision, or NaN when the comparison is undefined."""

    try:
        value = average_precision_score(np.asarray(y_true), np.asarray(y_score, dtype=float))
    except (TypeError, ValueError):
        return float("nan")
    return float(value) if np.isfinite(value) else float("nan")


__all__ = ["safe_auroc", "safe_auprc"]
