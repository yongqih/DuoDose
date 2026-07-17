"""Preliminary clustering and centroid utilities."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


def preliminary_clustering(
    adata: AnnData,
    n_neighbors: int = 15,
    resolution: float = 0.6,
    use_rep: str = "X_pca",
    cluster_key: str = "duodose_cluster",
    random_state: int = 0,
) -> AnnData:
    """Compute preliminary clusters using Scanpy Leiden with a KMeans fallback."""

    if use_rep not in adata.obsm:
        raise KeyError(f"use_rep={use_rep!r} not found in adata.obsm")
    X = np.asarray(adata.obsm[use_rep])

    if adata.n_obs < 3:
        adata.obs[cluster_key] = pd.Categorical(["0"] * adata.n_obs)
        return adata

    try:
        import scanpy as sc

        sc.pp.neighbors(adata, n_neighbors=min(n_neighbors, adata.n_obs - 1), use_rep=use_rep, random_state=random_state)
        sc.tl.leiden(adata, resolution=resolution, key_added=cluster_key, random_state=random_state)
        adata.obs[cluster_key] = adata.obs[cluster_key].astype(str).astype("category")
        return adata
    except Exception as exc:  # pragma: no cover - depends on optional leiden stack
        warnings.warn(
            f"Scanpy Leiden clustering failed ({exc!r}); falling back to KMeans.",
            RuntimeWarning,
            stacklevel=2,
        )

    n_clusters = int(np.clip(round(np.sqrt(adata.n_obs / 2)), 2, min(20, adata.n_obs)))
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit_predict(X)
    adata.obs[cluster_key] = pd.Categorical(labels.astype(str))
    return adata


def compute_cluster_centroids(
    adata: AnnData,
    cluster_key: str,
    use_rep: str = "X_pca",
) -> pd.DataFrame:
    """Return cluster centroids in the selected representation."""

    if cluster_key not in adata.obs:
        raise KeyError(f"cluster_key={cluster_key!r} not found in adata.obs")
    X = np.asarray(adata.obsm[use_rep])
    clusters = adata.obs[cluster_key].astype(str)
    rows = []
    index = []
    cluster_values = clusters.to_numpy()
    for cluster in pd.unique(clusters):
        idx = np.flatnonzero(cluster_values == cluster)
        rows.append(X[idx].mean(axis=0))
        index.append(str(cluster))
    return pd.DataFrame(np.vstack(rows), index=index)


def compute_cluster_distances(centroids: pd.DataFrame) -> pd.DataFrame:
    """Return pairwise Euclidean distances between cluster centroids."""

    distances = pairwise_distances(centroids.to_numpy(), metric="euclidean")
    return pd.DataFrame(distances, index=centroids.index.astype(str), columns=centroids.index.astype(str))
