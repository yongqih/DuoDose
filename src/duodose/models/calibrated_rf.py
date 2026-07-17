"""Calibrated random-forest backend for the default public DuoDose method."""

from __future__ import annotations

from typing import Any

INTERNAL_METHOD_NAME = "DuoDose-ML-CalibratedRF-SafeFeatures"
_TORCH_ONLY = {"max_epochs", "patience", "device", "use_amp", "batch_size", "num_workers", "diagnostic_only"}


def train_predict(train_scores, test_scores, **kwargs: Any):
    from ..net import train_predict_diagnostic_model

    sklearn_kwargs = {key: value for key, value in kwargs.items() if key not in _TORCH_ONLY}
    return train_predict_diagnostic_model(train_scores, test_scores, method=INTERNAL_METHOD_NAME, **sklearn_kwargs)
