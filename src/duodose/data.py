"""Shared data helpers for DuoDose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse


ArrayLikeMatrix = sparse.spmatrix | np.ndarray


@dataclass
class SimulatedDoublets:
    """Container for artificial doublets and their provenance."""

    X: ArrayLikeMatrix
    parent1: np.ndarray
    parent2: np.ndarray
    doublet_type: np.ndarray
    library: np.ndarray
    parent_index_1: np.ndarray
    parent_index_2: np.ndarray

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def obs_frame(self, prefix: str = "sim") -> pd.DataFrame:
        """Return simulation metadata as an observation-like table."""

        index = [f"{prefix}_{i}" for i in range(len(self))]
        return pd.DataFrame(
            {
                "parent1": self.parent1,
                "parent2": self.parent2,
                "doublet_type": self.doublet_type,
                "library": self.library,
                "parent_index_1": self.parent_index_1,
                "parent_index_2": self.parent_index_2,
            },
            index=index,
        )


def get_counts_matrix(adata: AnnData, counts_layer: str = "counts") -> ArrayLikeMatrix:
    """Return the raw count matrix from ``adata.layers[counts_layer]`` or ``adata.X``."""

    if counts_layer and counts_layer in adata.layers:
        X = adata.layers[counts_layer]
    else:
        X = adata.X
    if sparse.issparse(X):
        return X.tocsr()
    return np.asarray(X)


def ensure_counts_layer(adata: AnnData, counts_layer: str = "counts") -> AnnData:
    """Ensure a raw counts layer exists, using ``adata.X`` when needed."""

    if counts_layer and counts_layer not in adata.layers:
        adata.layers[counts_layer] = adata.X.copy()
    return adata


def row_sums(X: ArrayLikeMatrix) -> np.ndarray:
    """Compute matrix row sums as a flat NumPy array."""

    return np.asarray(X.sum(axis=1)).ravel()


def col_sums(X: ArrayLikeMatrix) -> np.ndarray:
    """Compute matrix column sums as a flat NumPy array."""

    return np.asarray(X.sum(axis=0)).ravel()


def row_nnz(X: ArrayLikeMatrix) -> np.ndarray:
    """Compute detected features per row."""

    if sparse.issparse(X):
        return np.diff(X.tocsr().indptr)
    return np.count_nonzero(np.asarray(X), axis=1)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Elementwise division with stable handling of zero denominators."""

    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.full_like(numerator, fill, dtype=float)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


def get_library_series(adata: AnnData, library_key: Optional[str] = None) -> pd.Series:
    """Return per-cell library labels, using a single synthetic library if absent."""

    if library_key is None:
        return pd.Series("__all__", index=adata.obs_names, dtype="object")
    if library_key not in adata.obs:
        raise KeyError(f"library_key={library_key!r} not found in adata.obs")
    return adata.obs[library_key].astype(str)


def get_group_key_frame(
    adata: AnnData,
    cluster_key: str,
    library_key: Optional[str] = None,
) -> pd.DataFrame:
    """Return cluster and library labels indexed by observations."""

    if cluster_key not in adata.obs:
        raise KeyError(f"cluster_key={cluster_key!r} not found in adata.obs")
    return pd.DataFrame(
        {
            "cluster": adata.obs[cluster_key].astype(str).to_numpy(),
            "library": get_library_series(adata, library_key).to_numpy(),
        },
        index=adata.obs_names,
    )


def matrix_to_dense(X: ArrayLikeMatrix) -> np.ndarray:
    """Convert a small matrix or vector to a dense NumPy array."""

    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def subset_rows(X: ArrayLikeMatrix, indices: np.ndarray | list[int]) -> ArrayLikeMatrix:
    """Subset matrix rows preserving sparse format when present."""

    if sparse.issparse(X):
        return X.tocsr()[indices]
    return np.asarray(X)[indices]


def downsample_count_row(row: ArrayLikeMatrix, fraction: float, rng: np.random.Generator) -> sparse.csr_matrix:
    """Binomially downsample one count row and return it as CSR."""

    fraction = float(np.clip(fraction, 0.0, 1.0))
    if sparse.issparse(row):
        csr = row.tocsr(copy=True)
        if csr.nnz:
            counts = np.rint(csr.data).clip(min=0).astype(np.int64)
            csr.data = rng.binomial(counts, fraction).astype(np.float32)
            csr.eliminate_zeros()
        return csr

    dense = np.asarray(row).reshape(1, -1)
    counts = np.rint(dense).clip(min=0).astype(np.int64)
    sampled = rng.binomial(counts, fraction).astype(np.float32)
    return sparse.csr_matrix(sampled)


def robust_mad(values: np.ndarray) -> tuple[float, float]:
    """Return median and a non-zero robust scale estimate."""

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        q75, q25 = np.percentile(x, [75, 25])
        scale = float((q75 - q25) / 1.349) if q75 > q25 else float(np.std(x))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return median, scale


def robust_z(values: np.ndarray) -> np.ndarray:
    """Compute robust z-scores using median and MAD."""

    median, scale = robust_mad(values)
    return (np.asarray(values, dtype=float) - median) / scale


def grouped_robust_z(values: pd.Series, groups: pd.DataFrame) -> pd.Series:
    """Compute robust z-scores within cluster/library groups."""

    result = pd.Series(index=values.index, dtype=float)
    key = groups.astype(str).agg("||".join, axis=1)
    for _, idx in key.groupby(key).groups.items():
        result.loc[idx] = robust_z(values.loc[idx].to_numpy())
    return result.fillna(0.0)


def save_results(adata: AnnData, path: str) -> None:
    """Write a DuoDose-annotated AnnData object to disk."""

    if not path.endswith(".h5ad"):
        warnings.warn("DuoDose results are usually saved as .h5ad files.", stacklevel=2)
    adata.write(path)
