"""Population-level doublet propensity estimation."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData

from .data import get_group_key_frame


def estimate_population_propensity(
    heterotypic_scores: pd.Series | np.ndarray,
    adata: AnnData,
    cluster_key: str = "duodose_cluster",
    library_key: Optional[str] = None,
    expected_doublet_rate: float = 0.06,
) -> pd.DataFrame:
    """Estimate cluster-level physical/capture doublet propensity.

    This simplified first-version estimator uses high heterotypic-like observed
    cells to estimate cluster participation, adjusts by abundance, and normalizes
    a non-negative propensity ``q_i`` within each library. Pair-specific residuals
    are intentionally not used to infer homotypic burden.
    """

    groups = get_group_key_frame(adata, cluster_key=cluster_key, library_key=library_key)
    scores = pd.Series(np.asarray(heterotypic_scores, dtype=float), index=adata.obs_names).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rows = []

    for library, lib_labels in groups.groupby("library").groups.items():
        lib_idx = list(lib_labels)
        lib_groups = groups.loc[lib_idx]
        lib_scores = scores.loc[lib_idx]
        if len(lib_scores) == 0:
            continue
        threshold = float(lib_scores.quantile(0.9)) if len(lib_scores) >= 10 else float(lib_scores.max())
        high = lib_scores >= threshold
        cluster_counts = lib_groups["cluster"].value_counts()
        abundance = cluster_counts / cluster_counts.sum()
        high_counts = lib_groups.loc[high, "cluster"].value_counts()
        high_rate = (high_counts / cluster_counts).reindex(cluster_counts.index).fillna(0.0)

        raw_q = (high_rate + 1e-3) / (abundance + 1e-6)
        raw_q = raw_q.clip(lower=0.0)
        weighted_mean = float((raw_q * abundance).sum())
        propensity_q = raw_q / weighted_mean if weighted_mean > 0 else pd.Series(1.0, index=raw_q.index)

        burden_raw = np.square(abundance) * np.square(propensity_q)
        burden_sum = float(burden_raw.sum())
        if burden_sum > 0:
            expected_homotypic_fraction = expected_doublet_rate * burden_raw / burden_sum
        else:
            expected_homotypic_fraction = pd.Series(0.0, index=cluster_counts.index)

        for cluster in cluster_counts.index:
            rows.append(
                {
                    "cluster": str(cluster),
                    "library": str(library),
                    "abundance": float(abundance.loc[cluster]),
                    "propensity_q": float(propensity_q.loc[cluster]),
                    "expected_homotypic_fraction": float(expected_homotypic_fraction.loc[cluster]),
                }
            )

    return pd.DataFrame(rows, columns=["cluster", "library", "abundance", "propensity_q", "expected_homotypic_fraction"])

