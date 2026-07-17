"""Conditional multitask neural-network backend."""

from __future__ import annotations

from typing import Any

INTERNAL_METHOD_NAME = "DuoDose-DL-ConditionalMultiTaskMLP-SafeFeatures"


def train_predict(train_scores, test_scores, **kwargs: Any):
    from ..net import train_predict_diagnostic_model

    return train_predict_diagnostic_model(
        train_scores,
        test_scores,
        method=INTERNAL_METHOD_NAME,
        **{key: value for key, value in kwargs.items() if key != "diagnostic_only"},
    )
