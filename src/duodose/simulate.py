"""Saturation-aware artificial doublet simulation."""

from __future__ import annotations

from typing import Optional
import warnings

import numpy as np
from anndata import AnnData
from scipy import sparse

from .data import SimulatedDoublets, downsample_count_row, get_counts_matrix, get_library_series


def _weighted_choice(rng: np.random.Generator, values: np.ndarray, weights: np.ndarray) -> str:
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0 or not np.all(np.isfinite(weights)):
        weights = np.ones_like(weights, dtype=float)
    weights = weights / weights.sum()
    return str(rng.choice(values, p=weights))


def simulate_doublets(
    adata: AnnData,
    cluster_key: str = "duodose_cluster",
    library_key: Optional[str] = None,
    counts_layer: str = "counts",
    n_doublets: int = 50000,
    homotypic_fraction: float = 0.5,
    saturation_range: tuple[float, float] = (0.6, 1.0),
    random_state: int = 0,
) -> SimulatedDoublets:
    """Simulate homotypic and heterotypic doublets within each library.

    Artificial doublets are generated from raw counts by addition followed by
    binomial downsampling using a saturation factor sampled uniformly from
    ``saturation_range``.
    """

    if cluster_key not in adata.obs:
        raise KeyError(f"cluster_key={cluster_key!r} not found in adata.obs")
    if n_doublets < 0:
        raise ValueError("n_doublets must be non-negative")

    rng = np.random.default_rng(random_state)
    counts = get_counts_matrix(adata, counts_layer=counts_layer)
    if sparse.issparse(counts):
        counts = counts.tocsr()
    clusters = adata.obs[cluster_key].astype(str).to_numpy()
    libraries = get_library_series(adata, library_key).to_numpy()

    unique_libraries, library_counts = np.unique(libraries, return_counts=True)
    valid_library_mask = library_counts > 0
    unique_libraries = unique_libraries[valid_library_mask]
    library_counts = library_counts[valid_library_mask]
    if unique_libraries.size == 0 or n_doublets == 0:
        return SimulatedDoublets(
            X=sparse.csr_matrix((0, adata.n_vars), dtype=np.float32),
            parent1=np.array([], dtype=object),
            parent2=np.array([], dtype=object),
            doublet_type=np.array([], dtype=object),
            library=np.array([], dtype=object),
            parent_index_1=np.array([], dtype=int),
            parent_index_2=np.array([], dtype=int),
        )

    allocation = rng.multinomial(int(n_doublets), library_counts / library_counts.sum())
    rows: list[sparse.csr_matrix] = []
    parent1: list[str] = []
    parent2: list[str] = []
    doublet_type: list[str] = []
    source_library: list[str] = []
    parent_index_1: list[int] = []
    parent_index_2: list[int] = []
    low, high = sorted((float(saturation_range[0]), float(saturation_range[1])))
    low = max(0.0, low)
    high = min(1.0, high)
    if high <= 0:
        warnings.warn("saturation_range produces all-zero doublets.", RuntimeWarning, stacklevel=2)

    for lib, n_for_lib in zip(unique_libraries, allocation):
        lib_indices = np.flatnonzero(libraries == lib)
        if lib_indices.size == 0 or n_for_lib == 0:
            continue

        lib_clusters = clusters[lib_indices]
        cluster_values, cluster_sizes = np.unique(lib_clusters, return_counts=True)
        cluster_to_indices = {cluster: lib_indices[lib_clusters == cluster] for cluster in cluster_values}
        can_heterotypic = cluster_values.size >= 2

        for _ in range(int(n_for_lib)):
            make_homotypic = (rng.random() < homotypic_fraction) or not can_heterotypic
            if make_homotypic:
                c1 = _weighted_choice(rng, cluster_values, cluster_sizes)
                c2 = c1
                idx_pool = cluster_to_indices[c1]
                a = int(rng.choice(idx_pool))
                b = int(rng.choice(idx_pool))
                dtype = "homotypic"
            else:
                probs = cluster_sizes / cluster_sizes.sum()
                c1, c2 = rng.choice(cluster_values, size=2, replace=False, p=probs)
                c1 = str(c1)
                c2 = str(c2)
                a = int(rng.choice(cluster_to_indices[c1]))
                b = int(rng.choice(cluster_to_indices[c2]))
                dtype = "heterotypic"

            if sparse.issparse(counts):
                raw = counts.getrow(a) + counts.getrow(b)
            else:
                raw = np.asarray(counts[a]).reshape(1, -1) + np.asarray(counts[b]).reshape(1, -1)
            saturation = float(rng.uniform(low, high)) if high > low else high
            rows.append(downsample_count_row(raw, saturation, rng))
            parent1.append(c1)
            parent2.append(c2)
            doublet_type.append(dtype)
            source_library.append(str(lib))
            parent_index_1.append(a)
            parent_index_2.append(b)

    X_sim = sparse.vstack(rows, format="csr", dtype=np.float32) if rows else sparse.csr_matrix((0, adata.n_vars), dtype=np.float32)
    return SimulatedDoublets(
        X=X_sim,
        parent1=np.asarray(parent1, dtype=object),
        parent2=np.asarray(parent2, dtype=object),
        doublet_type=np.asarray(doublet_type, dtype=object),
        library=np.asarray(source_library, dtype=object),
        parent_index_1=np.asarray(parent_index_1, dtype=int),
        parent_index_2=np.asarray(parent_index_2, dtype=int),
    )

