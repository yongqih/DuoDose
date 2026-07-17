from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_adata():
    anndata = pytest.importorskip("anndata")
    rng = np.random.default_rng(7)
    counts = rng.poisson(1.5, size=(72, 36)).astype(np.float32)
    counts[:12] += rng.poisson(2.0, size=(12, 36)).astype(np.float32)
    adata = anndata.AnnData(
        X=counts,
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(counts.shape[0])]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(counts.shape[1])]),
    )
    adata.layers["counts"] = counts.copy()
    return adata


@pytest.fixture
def protocol_adata():
    """Small but viable parent-disjoint dataset for public API tests."""

    anndata = pytest.importorskip("anndata")
    rng = np.random.default_rng(17)
    counts = rng.poisson(1.4, size=(240, 48)).astype(np.float32)
    for group in range(6):
        rows = slice(group * 40, (group + 1) * 40)
        genes = slice(group * 6, (group + 1) * 6)
        counts[rows, genes] += rng.poisson(2.5, size=(40, 6)).astype(np.float32)
    adata = anndata.AnnData(
        X=counts,
        obs=pd.DataFrame(index=[f"protocol_cell_{i}" for i in range(counts.shape[0])]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(counts.shape[1])]),
    )
    adata.layers["counts"] = counts.copy()
    return adata
