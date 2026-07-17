"""Cluster-specific conditional RNA dosage residuals."""

from __future__ import annotations

from typing import Optional, Sequence
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.linear_model import HuberRegressor, Ridge

from .data import get_counts_matrix, get_group_key_frame, grouped_robust_z, row_nnz, row_sums


def _stable_gene_score(adata: AnnData, counts_layer: str = "counts", n_genes: int = 100) -> np.ndarray:
    X = get_counts_matrix(adata, counts_layer=counts_layer)
    totals = row_sums(X).astype(float)
    means = np.asarray(X.mean(axis=0)).ravel()
    if hasattr(X, "power"):
        variances = np.asarray(X.power(2).mean(axis=0)).ravel() - means**2
    else:
        variances = np.var(np.asarray(X), axis=0)
    cv = np.sqrt(np.maximum(variances, 0.0)) / np.maximum(means, 1e-8)
    expressed = means > 0
    if not expressed.any():
        return np.log1p(totals)
    stable_idx = np.argsort(np.where(expressed, cv, np.inf))[: min(n_genes, expressed.sum())]
    stable_counts = row_sums(X[:, stable_idx]).astype(float)
    return np.log1p(stable_counts)


def compute_dosage_residuals(
    adata: AnnData,
    cluster_key: str = "duodose_cluster",
    library_key: Optional[str] = None,
    covariates: Sequence[str] = ("cell_cycle_score", "mito_fraction"),
    counts_layer: str = "counts",
) -> pd.DataFrame:
    """Compute cluster/library-specific RNA dosage residual features.

    The first version uses robust within-group z-scores and, when enough
    covariate information is available, an optional robust regression residual
    for ``log1p(n_counts)``.
    """

    groups = get_group_key_frame(adata, cluster_key=cluster_key, library_key=library_key)
    X = get_counts_matrix(adata, counts_layer=counts_layer)
    if "n_counts" not in adata.obs:
        adata.obs["n_counts"] = row_sums(X).astype(float)
    if "n_genes" not in adata.obs:
        adata.obs["n_genes"] = row_nnz(X).astype(float)

    log_counts = pd.Series(np.log1p(adata.obs["n_counts"].to_numpy(dtype=float)), index=adata.obs_names)
    log_genes = pd.Series(np.log1p(adata.obs["n_genes"].to_numpy(dtype=float)), index=adata.obs_names)
    stable_score = pd.Series(_stable_gene_score(adata, counts_layer=counts_layer), index=adata.obs_names)
    adata.obs["stable_gene_module_score"] = stable_score

    count_z = grouped_robust_z(log_counts, groups)
    gene_z = grouped_robust_z(log_genes, groups)
    stable_z = grouped_robust_z(stable_score, groups)
    dosage = 0.5 * count_z + 0.3 * gene_z + 0.2 * stable_z

    regression_residual = pd.Series(0.0, index=adata.obs_names)
    available_covariates = [c for c in covariates if c in adata.obs]
    if available_covariates:
        key = groups.astype(str).agg("||".join, axis=1)
        for _, labels in key.groupby(key).groups.items():
            idx = list(labels)
            if len(idx) < max(8, len(available_covariates) + 3):
                regression_residual.loc[idx] = count_z.loc[idx]
                continue
            X_cov = adata.obs.loc[idx, available_covariates].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
            y = log_counts.loc[idx].to_numpy()
            try:
                model = HuberRegressor().fit(X_cov, y)
            except Exception:
                model = Ridge(alpha=1.0).fit(X_cov, y)
            residual = y - model.predict(X_cov)
            med = np.median(residual)
            mad = 1.4826 * np.median(np.abs(residual - med))
            if not np.isfinite(mad) or mad <= 1e-12:
                mad = np.std(residual) or 1.0
            regression_residual.loc[idx] = (residual - med) / mad
    else:
        regression_residual = count_z.copy()

    residuals = pd.DataFrame(
        {
            "duodose_count_residual": count_z,
            "duodose_gene_residual": gene_z,
            "duodose_stable_dosage_residual": stable_z,
            "duodose_dosage_residual": dosage,
            "duodose_regression_dosage_residual": regression_residual,
        },
        index=adata.obs_names,
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for column in residuals.columns:
        adata.obs[column] = residuals[column]
    if residuals.empty:
        warnings.warn("No dosage residuals were computed.", RuntimeWarning, stacklevel=2)
    return residuals
