"""Tests for the single frozen calibrated-RF high-RNA weighting rule."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from duodose.config import DuoDoseConfig, SemiRealConfig
from duodose.protocol import FINAL_HIGH_RNA_NEGATIVE_WEIGHT, load_final_protocol
from duodose.rf_weighting import FORMAL_HIGH_RNA_NEGATIVE_WEIGHT, formal_rf_sample_weights
from duodose.net import FULL_FEATURE_COLUMNS, train_predict_diagnostic_model
from reproducibility.lib.common import formal_backend_training_kwargs


def _training_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {"true_label": ["clean", "singlet", "high_RNA_singlet", "homotypic_doublet"]},
        index=["ordinary_a", "ordinary_b", "high_rna", "doublet"],
    )


def test_formal_weight_is_exactly_two() -> None:
    protocol = load_final_protocol()
    assert FORMAL_HIGH_RNA_NEGATIVE_WEIGHT == 2.0
    assert FINAL_HIGH_RNA_NEGATIVE_WEIGHT == 2.0
    assert protocol["models"]["high_rna_negative_weight"] == 2.0
    assert DuoDoseConfig().semireal is not None
    assert DuoDoseConfig().semireal.high_rna_negative_weight == 2.0


def test_fixed_sample_weights_preserve_ordinary_and_upweight_high_rna() -> None:
    weights = formal_rf_sample_weights(_training_rows())
    assert weights.loc["ordinary_a"] == 1.0
    assert weights.loc["ordinary_b"] == 1.0
    assert weights.loc["high_rna"] == 2.0
    assert weights.loc["doublet"] == 1.0


def test_public_config_rejects_alternate_high_rna_weight() -> None:
    invalid = SemiRealConfig(high_rna_negative_weight=4.0)
    try:
        DuoDoseConfig(semireal=invalid)
    except ValueError as exc:
        assert "high_rna_negative_weight" in str(exc)
    else:  # pragma: no cover - makes the frozen contract explicit
        raise AssertionError("public DuoDose accepted a non-formal high-RNA sample weight")


def test_formal_rf_receives_one_fixed_training_configuration() -> None:
    kwargs = formal_backend_training_kwargs("rf", _training_rows())
    assert set(kwargs) == {"sample_weight", "high_rna_negative_weight"}
    assert kwargs["high_rna_negative_weight"] == 2.0
    assert formal_backend_training_kwargs("dl", _training_rows()) == {}


def test_obsolete_weight_search_workflow_is_absent() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "src" / "duodose" / "background_clean_pilot.py").exists()
    assert not (root / "reproducibility" / "run_background_clean_hard_negative_pilot.py").exists()
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (root / "src", root / "reproducibility")
        for path in path.rglob("*.py")
    )
    for forbidden in ("selected_hard_negative_weight", "hard_negative_weights", "--weights"):
        assert forbidden not in source


def test_serialized_fitted_rf_records_fixed_high_rna_weight(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    labels = np.repeat(["clean", "high_RNA_singlet", "homotypic_doublet"], 12)
    scores = pd.DataFrame(rng.normal(size=(len(labels), len(FULL_FEATURE_COLUMNS))), columns=FULL_FEATURE_COLUMNS)
    scores["true_label"] = labels
    scores.index = [f"cell_{position}" for position in range(len(scores))]
    fitted = train_predict_diagnostic_model(
        scores,
        scores.iloc[:6].copy(),
        method="DuoDose-ML-CalibratedRF-SafeFeatures",
        sample_weight=formal_rf_sample_weights(scores),
        high_rna_negative_weight=2.0,
        random_state=0,
    )["fitted_backend"]
    assert fitted is not None
    path = tmp_path / "duodose_rf.joblib"
    joblib.dump(fitted, path)
    restored = joblib.load(path)
    assert restored.training_summary["high_rna_negative_weight"] == 2.0
