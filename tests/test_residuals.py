import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from duodose.residuals import compute_dosage_residuals


def test_dosage_residuals_are_computed_per_cluster():
    X = sparse.csr_matrix(
        np.array(
            [
                [100, 10, 0],
                [110, 9, 0],
                [95, 11, 0],
                [1000, 100, 0],
                [0, 8, 80],
                [0, 9, 85],
                [0, 7, 75],
                [0, 80, 800],
            ],
            dtype=int,
        )
    )
    obs = pd.DataFrame(
        {"duodose_cluster": ["A", "A", "A", "A", "B", "B", "B", "B"], "sample_id": ["lib"] * 8},
        index=[f"cell{i}" for i in range(8)],
    )
    adata = AnnData(X=X, obs=obs)
    adata.layers["counts"] = X.copy()
    residuals = compute_dosage_residuals(adata, library_key="sample_id")
    assert "duodose_dosage_residual" in residuals
    medians = adata.obs.groupby("duodose_cluster")["duodose_count_residual"].median().abs()
    assert np.all(medians < 1e-8)
    assert adata.obs.loc["cell3", "duodose_count_residual"] > 1
    assert adata.obs.loc["cell7", "duodose_count_residual"] > 1

