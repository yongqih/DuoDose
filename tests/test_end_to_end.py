import numpy as np

from duodose import DuoDose
from duodose.config import DLConfig, DuoDoseConfig, FeatureConfig, SemiRealConfig


SCORE_COLUMNS = [
    "duodose_score",
    "duodose_homotypic_score",
    "duodose_heterotypic_score",
    "predicted_doublet",
    "predicted_subtype",
    "subtype_confidence",
]


def _small_dl_config(backend: str) -> DuoDoseConfig:
    return DuoDoseConfig(
        backend=backend,
        expected_doublet_rate=0.08,
        random_state=0,
        layer="counts",
        device="cpu",
        training_preset="fast",
        feature=FeatureConfig(n_hvgs=24, n_pcs=6, n_simulated_doublets=80),
        dl=DLConfig(max_epochs=2, patience=1, batch_size=16),
        semireal=SemiRealConfig.from_preset("fast"),
    )


def _assert_dl_result(result, n_obs: int, backend: str) -> None:
    assert result.backend == backend
    assert result.scores.shape[0] == n_obs
    assert set(SCORE_COLUMNS).issubset(result.scores.columns)
    probability_columns = SCORE_COLUMNS[:3]
    probabilities = result.scores[probability_columns].to_numpy(dtype=float)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    np.testing.assert_allclose(
        result.scores["duodose_score"].to_numpy(dtype=float),
        (
            result.scores["duodose_homotypic_score"]
            + result.scores["duodose_heterotypic_score"]
        ).to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-7,
    )
    assert result.training_summary["status"] == "success"
    assert result.training_summary["training_device"] == "cpu"


def test_small_end_to_end_rf(protocol_adata) -> None:
    config = DuoDoseConfig(
        backend="rf",
        expected_doublet_rate=0.1,
        random_state=0,
        layer="counts",
        training_preset="fast",
        feature=FeatureConfig(n_hvgs=24, n_pcs=6, n_simulated_doublets=80),
        semireal=SemiRealConfig.from_preset("fast"),
    )
    result = DuoDose(config=config).fit_predict(protocol_adata)
    assert result.backend == "rf"
    assert result.scores.shape[0] == protocol_adata.n_obs
    assert result.parent_audit["parent_leakage_audit_status"] == "passed"
    assert result.model_metadata["construction_variant"] == "raw_sum_parents_removed"
    assert result.model_metadata["safe_feature_mode"] == "fitted_reference"
    assert result.scores["duodose_score"].between(0, 1).all()


def test_dl_backend_fit_predict_smoke(protocol_adata) -> None:
    result = DuoDose(config=_small_dl_config("dl")).fit_predict(protocol_adata)
    _assert_dl_result(result, protocol_adata.n_obs, "dl")
