"""Matplotlib plotting helpers for DuoDose outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from .plotting_style import apply_manuscript_style


def _get_ax(ax=None):
    import matplotlib.pyplot as plt

    apply_manuscript_style()

    if ax is None:
        _, ax = plt.subplots()
    return ax


def plot_score_histograms(adata: AnnData, ax=None):
    """Plot DuoDose union and subtype score histograms."""

    ax = _get_ax(ax)
    for column, label in [
        ("duodose_score", "union"),
        ("duodose_heterotypic_score", "heterotypic"),
        ("duodose_homotypic_score", "homotypic"),
    ]:
        if column in adata.obs:
            ax.hist(adata.obs[column].to_numpy(dtype=float), bins=40, alpha=0.45, label=label)
    ax.set_xlabel("DuoDose score")
    ax.set_ylabel("Cells")
    ax.legend(frameon=False)
    return ax


def plot_umap_scores(adata: AnnData, score_key: str = "duodose_score", ax=None):
    """Scatter UMAP coordinates colored by a DuoDose score."""

    ax = _get_ax(ax)
    if "X_umap" not in adata.obsm:
        raise KeyError("adata.obsm['X_umap'] is required for UMAP plotting.")
    score = adata.obs[score_key].to_numpy(dtype=float)
    coords = adata.obsm["X_umap"]
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=score, s=6, cmap="viridis", linewidths=0)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.figure.colorbar(scatter, ax=ax, label=score_key)
    return ax


def plot_precision_recall_curves(results: dict[str, tuple[np.ndarray, np.ndarray]], ax=None):
    """Plot precomputed precision-recall curves."""

    ax = _get_ax(ax)
    for name, (precision, recall) in results.items():
        ax.plot(recall, precision, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=False)
    return ax


def plot_homotypic_vs_heterotypic_performance(results: pd.DataFrame, ax=None):
    """Bar plot comparing homotypic and heterotypic AUPRC."""

    ax = _get_ax(ax)
    columns = [c for c in ["homotypic_AUPRC", "heterotypic_AUPRC"] if c in results]
    results[columns].plot(kind="bar", ax=ax)
    ax.set_ylabel("AUPRC")
    ax.legend(frameon=False)
    return ax


def plot_cluster_dosage_residuals(adata: AnnData, cluster_key: str = "duodose_cluster", ax=None):
    """Plot dosage residual distributions by preliminary cluster."""

    ax = _get_ax(ax)
    if "duodose_dosage_residual" not in adata.obs:
        raise KeyError("duodose_dosage_residual not found in adata.obs.")
    data = [
        adata.obs.loc[adata.obs[cluster_key].astype(str) == cluster, "duodose_dosage_residual"].to_numpy(dtype=float)
        for cluster in sorted(adata.obs[cluster_key].astype(str).unique())
    ]
    ax.boxplot(data, labels=sorted(adata.obs[cluster_key].astype(str).unique()), showfliers=False)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Dosage residual")
    return ax
