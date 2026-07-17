import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from duodose.simulate import simulate_doublets


def make_adata():
    X = sparse.csr_matrix(
        np.array(
            [
                [10, 0, 1, 0],
                [9, 1, 0, 0],
                [0, 8, 0, 1],
                [0, 7, 1, 0],
                [5, 0, 0, 2],
                [4, 0, 1, 1],
                [0, 5, 2, 0],
                [0, 4, 1, 2],
            ],
            dtype=int,
        )
    )
    obs = pd.DataFrame(
        {
            "duodose_cluster": ["A", "A", "B", "B", "A", "A", "B", "B"],
            "sample_id": ["lib1", "lib1", "lib1", "lib1", "lib2", "lib2", "lib2", "lib2"],
        },
        index=[f"cell{i}" for i in range(8)],
    )
    adata = AnnData(X=X, obs=obs)
    adata.layers["counts"] = X.copy()
    return adata


def test_doublets_are_not_simulated_across_libraries():
    adata = make_adata()
    sim = simulate_doublets(adata, library_key="sample_id", n_doublets=100, random_state=1)
    libraries = adata.obs["sample_id"].to_numpy()
    assert np.all(libraries[sim.parent_index_1] == libraries[sim.parent_index_2])
    assert set(sim.library).issubset({"lib1", "lib2"})


def test_simulated_parent_labels_match_type():
    adata = make_adata()
    sim = simulate_doublets(adata, library_key="sample_id", n_doublets=100, homotypic_fraction=0.5, random_state=2)
    homotypic = sim.doublet_type == "homotypic"
    heterotypic = sim.doublet_type == "heterotypic"
    assert np.all(sim.parent1[homotypic] == sim.parent2[homotypic])
    assert np.all(sim.parent1[heterotypic] != sim.parent2[heterotypic])

