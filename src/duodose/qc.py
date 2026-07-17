"""Loose quality control for doublet-aware analysis."""

from __future__ import annotations

from typing import Sequence
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData

from .data import get_counts_matrix, row_nnz, row_sums, safe_divide


def loose_qc(
    adata: AnnData,
    min_counts: int = 500,
    min_genes: int = 200,
    max_mito_fraction: float = 0.3,
    mito_prefix: Sequence[str] = ("MT-", "mt-"),
    counts_layer: str = "counts",
) -> AnnData:
    """Remove obvious low-quality cells while retaining high-RNA doublet signal.

    High count or high feature cells are never removed by this function.
    QC metrics are stored in ``adata.obs`` before filtering.
    """

    X = get_counts_matrix(adata, counts_layer=counts_layer)
    n_counts = row_sums(X).astype(float)
    n_genes = row_nnz(X).astype(float)
    adaptive_gene_cap = max(1, int(np.ceil(0.1 * adata.n_vars)))
    effective_min_genes = min(int(min_genes), adaptive_gene_cap)

    var_names = pd.Index(adata.var_names.astype(str))
    mito_mask = np.zeros(adata.n_vars, dtype=bool)
    for prefix in mito_prefix:
        mito_mask |= np.asarray(var_names.str.startswith(prefix), dtype=bool)
    mito_counts = row_sums(X[:, mito_mask]) if mito_mask.any() else np.zeros(adata.n_obs)
    mito_fraction = safe_divide(mito_counts, n_counts)

    count_failure = np.clip(1.0 - safe_divide(n_counts, np.full_like(n_counts, min_counts)), 0.0, 1.0)
    gene_failure = np.clip(1.0 - safe_divide(n_genes, np.full_like(n_genes, effective_min_genes)), 0.0, 1.0)
    mito_failure = np.clip(
        safe_divide(mito_fraction - max_mito_fraction, np.full_like(mito_fraction, 1.0 - max_mito_fraction)),
        0.0,
        1.0,
    )
    low_quality_score = np.maximum.reduce([count_failure, gene_failure, mito_failure])

    adata.obs["n_counts"] = n_counts
    adata.obs["n_genes"] = n_genes
    adata.obs["mito_fraction"] = mito_fraction
    adata.obs["duodose_low_quality_score"] = low_quality_score

    keep = (n_counts >= min_counts) & (n_genes >= effective_min_genes) & (mito_fraction <= max_mito_fraction)
    adata.obs["duodose_qc_pass"] = keep

    if keep.sum() == 0:
        warnings.warn(
            "Loose QC would remove every cell; returning the unfiltered AnnData with QC metrics.",
            RuntimeWarning,
            stacklevel=2,
        )
        return adata.copy()

    if keep.sum() < 2 and adata.n_obs >= 2:
        warnings.warn(
            "Loose QC left fewer than two cells; returning the unfiltered AnnData for downstream stability.",
            RuntimeWarning,
            stacklevel=2,
        )
        return adata.copy()

    return adata[keep].copy()
