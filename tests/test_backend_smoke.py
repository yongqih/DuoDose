from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from duodose.models.registry import DUODOSE_BACKENDS
from duodose.net import _capped_ratio, train_predict_diagnostic_model


def test_capped_ratio() -> None:
    assert _capped_ratio(10, 2) == 5.0
    assert _capped_ratio(100, 1, cap=20) == 20.0
    assert _capped_ratio(1, 0, default=1.0) == 1.0
    assert _capped_ratio(float("nan"), 1, default=1.0) == 1.0


def _cell_scores() -> tuple[pd.DataFrame, pd.Index, pd.Index]:
    rng = np.random.default_rng(4)
    labels = np.repeat(["clean", "high_RNA_singlet", "homotypic_doublet", "heterotypic_doublet"], 18)
    frame = pd.DataFrame(
        {
            "true_label": labels,
            "nCount": rng.uniform(1000, 5000, len(labels)),
            "log_nCount": rng.normal(8, 0.5, len(labels)),
            "nFeature": rng.uniform(500, 2000, len(labels)),
            "log_nFeature": rng.normal(7, 0.4, len(labels)),
            "dosage_outlier_score": rng.random(len(labels)),
            "identity_inlier_score": rng.random(len(labels)),
            "duodose_score": rng.random(len(labels)),
            "homotypic_score": rng.random(len(labels)),
            "heterotypic_score": rng.random(len(labels)),
            "cluster_nCount_z": rng.normal(size=len(labels)),
            "benchmark_cluster": np.tile(["a", "b", "c"], 24),
            "sample_id": "sample",
        },
        index=[f"row_{i}" for i in range(len(labels))],
    )
    train_positions = [position for start in range(0, len(labels), 18) for position in range(start, start + 14)]
    validation_positions = [position for start in range(0, len(labels), 18) for position in range(start + 14, start + 18)]
    train_index = frame.index[train_positions]
    validation_index = frame.index[validation_positions]
    return frame, train_index, validation_index


def test_rf_backend_smoke() -> None:
    pytest.importorskip("sklearn")
    frame, train_index, validation_index = _cell_scores()
    result = train_predict_diagnostic_model(
        frame,
        frame.iloc[:8],
        method=DUODOSE_BACKENDS["rf"],
        train_index=train_index,
        validation_index=validation_index,
        random_state=0,
    )
    assert result["summary"]["status"] == "success"
    assert result["fitted_backend"] is not None


def test_rf_backend_is_deterministic() -> None:
    pytest.importorskip("sklearn")
    frame, train_index, validation_index = _cell_scores()
    outputs = []
    for _ in range(2):
        result = train_predict_diagnostic_model(
            frame,
            frame.iloc[:8],
            method=DUODOSE_BACKENDS["rf"],
            train_index=train_index,
            validation_index=validation_index,
            random_state=11,
        )
        outputs.append(result["test_probabilities"].to_numpy(dtype=float))
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=0.0, atol=1e-14)


def test_dl_backend_cpu_smoke() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    frame, train_index, validation_index = _cell_scores()
    result = train_predict_diagnostic_model(
        frame,
        frame.iloc[:8],
        method=DUODOSE_BACKENDS["dl"],
        train_index=train_index,
        validation_index=validation_index,
        random_state=0,
        net_train_seed=0,
        device="cpu",
        max_epochs=2,
        patience=1,
        batch_size=16,
    )
    assert result["summary"]["status"] == "success"
    assert result["summary"]["training_device"] == "cpu"
    assert result["fitted_backend"] is not None
