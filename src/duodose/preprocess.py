"""Normalization, HVG selection, and PCA-style latent representation."""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
from anndata import AnnData
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

from .data import ArrayLikeMatrix, ensure_counts_layer, get_counts_matrix, row_sums


def _normalize_log_counts(X: ArrayLikeMatrix, target_sum: float = 1e4) -> ArrayLikeMatrix:
    totals = row_sums(X).astype(float)
    scale = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
    if sparse.issparse(X):
        norm = sparse.diags(scale).dot(X.tocsr().astype(np.float32))
        norm.data = np.log1p(norm.data)
        return norm.tocsr()
    norm = np.asarray(X, dtype=np.float32) * scale[:, None]
    return np.log1p(norm)


def _column_variance(X: ArrayLikeMatrix) -> np.ndarray:
    if sparse.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
        return np.maximum(mean_sq - mean**2, 0.0)
    return np.var(np.asarray(X), axis=0)


def normalize_hvg_pca(
    adata: AnnData,
    counts_layer: str = "counts",
    n_hvgs: int = 2000,
    n_pcs: int = 50,
    random_state: int = 0,
) -> AnnData:
    """Normalize counts, select highly variable genes, and store ``adata.obsm['X_pca']``.

    The raw counts are preserved in ``adata.layers[counts_layer]``. For sparse
    matrices this uses a TruncatedSVD representation on log-normalized HVGs,
    which avoids densifying the full expression matrix.
    """

    ensure_counts_layer(adata, counts_layer=counts_layer)
    counts = get_counts_matrix(adata, counts_layer=counts_layer)
    log_norm = _normalize_log_counts(counts)
    adata.X = log_norm.copy() if sparse.issparse(log_norm) else np.asarray(log_norm, dtype=np.float32)

    n_hvgs_eff = int(min(max(1, n_hvgs), adata.n_vars))
    variances = _column_variance(log_norm)
    hvg_order = np.argsort(variances)[::-1][:n_hvgs_eff]
    hvg_mask = np.zeros(adata.n_vars, dtype=bool)
    hvg_mask[hvg_order] = True
    adata.var["highly_variable"] = hvg_mask
    adata.var["duodose_hvg_variance"] = variances

    X_hvg = log_norm[:, hvg_mask]
    max_components = max(1, min(n_pcs, adata.n_obs - 1 if adata.n_obs > 1 else 1, X_hvg.shape[1] - 1 if X_hvg.shape[1] > 1 else 1))

    if adata.n_obs < 2 or X_hvg.shape[1] < 2:
        adata.obsm["X_pca"] = np.zeros((adata.n_obs, 1), dtype=np.float32)
        adata.uns["duodose_preprocess"] = {
            "target_sum": 1e4,
            "hvg_indices": np.flatnonzero(hvg_mask),
            "center": False,
            "scale": np.ones(int(hvg_mask.sum()), dtype=np.float32),
            "mean": np.zeros(int(hvg_mask.sum()), dtype=np.float32),
            "components": np.zeros((1, int(hvg_mask.sum())), dtype=np.float32),
            "method": "constant",
        }
        return adata

    if sparse.issparse(X_hvg):
        scaler = StandardScaler(with_mean=False)
        X_scaled = scaler.fit_transform(X_hvg)
        reducer: Any = TruncatedSVD(n_components=max_components, random_state=random_state)
        X_pca = reducer.fit_transform(X_scaled)
        center = False
        mean = np.zeros(X_hvg.shape[1], dtype=np.float32)
    else:
        scaler = StandardScaler(with_mean=True)
        X_scaled = scaler.fit_transform(np.asarray(X_hvg, dtype=np.float32))
        reducer = PCA(n_components=max_components, random_state=random_state)
        X_pca = reducer.fit_transform(X_scaled)
        center = True
        mean = scaler.mean_.astype(np.float32)

    scale = np.asarray(getattr(scaler, "scale_", np.ones(X_hvg.shape[1])), dtype=np.float32)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    components = np.asarray(getattr(reducer, "components_", np.zeros((max_components, X_hvg.shape[1]))), dtype=np.float32)

    if X_pca.shape[1] < n_pcs:
        padded = np.zeros((adata.n_obs, n_pcs), dtype=np.float32)
        padded[:, : X_pca.shape[1]] = X_pca
        X_pca = padded

    adata.obsm["X_pca"] = np.asarray(X_pca, dtype=np.float32)
    adata.uns["duodose_preprocess"] = {
        "target_sum": 1e4,
        "hvg_indices": np.flatnonzero(hvg_mask),
        "center": center,
        "scale": scale,
        "mean": mean,
        "components": components,
        "method": "pca" if center else "truncated_svd",
    }
    explained = getattr(reducer, "explained_variance_ratio_", None)
    if explained is not None:
        adata.uns["duodose_pca_variance_ratio"] = np.asarray(explained, dtype=float)
    return adata


def transform_counts_to_pca(
    adata: AnnData,
    counts: ArrayLikeMatrix,
    batch_size: int = 4096,
) -> np.ndarray:
    """Project raw count rows into the DuoDose PCA/SVD space."""

    if "duodose_preprocess" not in adata.uns:
        raise KeyError("DuoDose preprocessing metadata not found; run normalize_hvg_pca first.")
    params = adata.uns["duodose_preprocess"]
    hvg_indices = np.asarray(params["hvg_indices"], dtype=int)
    scale = np.asarray(params["scale"], dtype=np.float32)
    mean = np.asarray(params["mean"], dtype=np.float32)
    components = np.asarray(params["components"], dtype=np.float32)
    center = bool(params.get("center", False))
    n_rows = counts.shape[0]
    n_components = components.shape[0]
    out = np.zeros((n_rows, n_components), dtype=np.float32)

    for start in range(0, n_rows, batch_size):
        stop = min(start + batch_size, n_rows)
        block = counts[start:stop]
        log_block = _normalize_log_counts(block, target_sum=float(params.get("target_sum", 1e4)))
        X_hvg = log_block[:, hvg_indices]
        if sparse.issparse(X_hvg):
            if center:
                dense = X_hvg.toarray().astype(np.float32)
                dense = (dense - mean) / scale
                out[start:stop] = dense @ components.T
            else:
                X_scaled = X_hvg.multiply(1.0 / scale)
                out[start:stop] = X_scaled @ components.T
        else:
            dense = np.asarray(X_hvg, dtype=np.float32)
            if center:
                dense = dense - mean
            dense = dense / scale
            out[start:stop] = dense @ components.T

    if out.shape[1] < adata.obsm["X_pca"].shape[1]:
        padded = np.zeros((n_rows, adata.obsm["X_pca"].shape[1]), dtype=np.float32)
        padded[:, : out.shape[1]] = out
        out = padded
    elif out.shape[1] > adata.obsm["X_pca"].shape[1]:
        out = out[:, : adata.obsm["X_pca"].shape[1]]
    if not np.all(np.isfinite(out)):
        warnings.warn("Non-finite values encountered while projecting counts; replacing with zeros.", stacklevel=2)
        out = np.nan_to_num(out)
    return out

