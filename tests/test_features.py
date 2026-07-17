import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from duodose.predict import REQUIRED_OBS_COLUMNS, assign_labels, run_duodose


def make_small_adata(seed=0):
    rng = np.random.default_rng(seed)
    n_cells = 36
    n_genes = 80
    clusters = np.array(["A"] * 18 + ["B"] * 18)
    X = rng.poisson(2, size=(n_cells, n_genes))
    X[:18, :10] += rng.poisson(12, size=(18, 10))
    X[18:, 10:20] += rng.poisson(12, size=(18, 10))
    obs = pd.DataFrame(
        {"sample_id": ["lib1"] * 18 + ["lib2"] * 18},
        index=[f"cell{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    adata = AnnData(X=sparse.csr_matrix(X), obs=obs, var=var)
    adata.layers["counts"] = adata.X.copy()
    return adata


def test_run_duodose_outputs_required_columns(monkeypatch):
    adata = make_small_adata()

    def assign_test_clusters(work, **_kwargs):
        midpoint = work.n_obs // 2
        work.obs["duodose_cluster"] = pd.Categorical(
            ["A"] * midpoint + ["B"] * (work.n_obs - midpoint)
        )

    monkeypatch.setattr("duodose.predict.preliminary_clustering", assign_test_clusters)
    result = run_duodose(
        adata,
        library_key="sample_id",
        expected_doublet_rate=0.1,
        n_simulated_doublets=60,
        n_hvgs=40,
        n_pcs=8,
        min_counts=1,
        min_genes=1,
        random_state=0,
        high_confidence_threshold=0.5,
        uncertain_threshold=0.4,
    )
    for column in REQUIRED_OBS_COLUMNS:
        assert column in result.obs
    assert set(result.obs["duodose_label"].astype(str)).issubset(
        {"clean", "heterotypic_doublet", "homotypic_doublet", "low_quality", "uncertain"}
    )
    expected_raw_score = 1.0 - (1.0 - result.obs["duodose_heterotypic_score"]) * (1.0 - result.obs["duodose_homotypic_score"])
    assert np.allclose(result.obs["duodose_score_raw_union"], expected_raw_score)
    assert np.allclose(result.obs["duodose_score"], result.obs["duodose_score_tail_calibrated"])
    assert result.obs["duodose_score_rank_calibrated"].between(0.0, 1.0).all()
    assert result.obs["duodose_heterotypic_tail_score"].between(0.0, 1.0).all()
    assert result.obs["duodose_homotypic_tail_score"].between(0.0, 1.0).all()
    assert result.obs["duodose_score"].between(0.0, 1.0).all()


def test_label_assignment_uses_subtype_thresholds():
    probs = pd.DataFrame(
        {
            "clean": [0.01] * 10,
            "heterotypic_doublet": [0.95] * 10,
            "homotypic_doublet": [0.01] * 10,
            "low_quality": [0.0] * 10,
        },
        index=[f"cell{i}" for i in range(10)],
    )
    labels = assign_labels(probs, expected_doublet_rate=0.2, high_confidence_threshold=0.9)
    assert (labels == "heterotypic_doublet").all()


def test_label_assignment_uses_union_score():
    probs = pd.DataFrame(
        {
            "clean": [0.1],
            "heterotypic_doublet": [0.55],
            "homotypic_doublet": [0.55],
            "low_quality": [0.0],
        },
        index=["cell0"],
    )
    labels = assign_labels(probs, expected_doublet_rate=1.0, high_confidence_threshold=0.9, uncertain_threshold=0.6)
    assert labels.loc["cell0"] == "uncertain"
