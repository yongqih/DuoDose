"""Lightweight benchmark-only DuoDose-Hybrid score helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def empirical_percentile_score(values: pd.Series | np.ndarray) -> pd.Series:
    """Convert a score vector to empirical percentile ranks in [0, 1]."""

    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan)
    if series.dropna().empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return series.rank(method="average", pct=True).fillna(0.5).clip(0.0, 1.0)


def tail_calibrate(percentile_score: pd.Series | np.ndarray, center: float = 0.5) -> pd.Series:
    """Keep only the upper tail of an empirical percentile score."""

    series = pd.Series(percentile_score, dtype=float).replace([np.inf, -np.inf], np.nan)
    center = float(np.clip(center, 0.0, 0.99))
    denom = max(1.0 - center, 1e-8)
    return ((series - center) / denom).clip(lower=0.0, upper=1.0)


def duodose_hybrid_scores(
    scrublet_score: pd.Series | np.ndarray,
    duodose_homotypic_score: pd.Series | np.ndarray,
    center: float = 0.5,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Fuse Scrublet heterotypic evidence with DuoDose homotypic evidence."""

    scrublet_percentile = empirical_percentile_score(scrublet_score)
    homotypic_percentile = empirical_percentile_score(duodose_homotypic_score)
    hybrid_heterotypic = tail_calibrate(scrublet_percentile, center=center)
    hybrid_homotypic = tail_calibrate(homotypic_percentile, center=center)
    hybrid_overall = 1.0 - (1.0 - hybrid_heterotypic) * (1.0 - hybrid_homotypic)
    return hybrid_overall.astype(float), hybrid_homotypic.astype(float), hybrid_heterotypic.astype(float)


def dosage_informativeness_score(
    homotypic_score: pd.Series | np.ndarray,
    n_count: pd.Series | np.ndarray | None = None,
) -> float:
    """Estimate whether homotypic dosage evidence has a useful upper tail.

    This unsupervised diagnostic intentionally uses only score/count
    distributions. It does not inspect truth labels or benchmark metrics.
    """

    hom = pd.Series(homotypic_score, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if hom.empty or hom.nunique() <= 1:
        return 0.0
    q25, q50, q75, q95 = hom.quantile([0.25, 0.50, 0.75, 0.95])
    iqr = max(float(q75 - q25), 1e-8)
    tail_separation = float(np.clip(((q95 - q50) / iqr - 1.0) / 3.0, 0.0, 1.0))
    spread = float(np.clip(iqr / 0.25, 0.0, 1.0))
    corr_penalty = 1.0
    if n_count is not None:
        counts = pd.Series(n_count, dtype=float).replace([np.inf, -np.inf], np.nan)
        aligned = pd.concat([hom, np.log1p(counts)], axis=1).dropna()
        if len(aligned) >= 3 and aligned.iloc[:, 0].nunique() > 1 and aligned.iloc[:, 1].nunique() > 1:
            corr = abs(float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])))
            corr_penalty = float(np.clip(1.0 - 0.35 * corr, 0.45, 1.0))
    return float(np.clip((0.65 * tail_separation + 0.35 * spread) * corr_penalty, 0.0, 1.0))


def weighted_hybrid_scores(
    scrublet_score: pd.Series | np.ndarray,
    duodose_homotypic_score: pd.Series | np.ndarray,
    homotypic_weight: float,
    center: float = 0.5,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Fuse calibrated Scrublet evidence with a weighted homotypic branch."""

    scrublet_percentile = empirical_percentile_score(scrublet_score)
    homotypic_percentile = empirical_percentile_score(duodose_homotypic_score)
    hybrid_heterotypic = tail_calibrate(scrublet_percentile, center=center)
    hybrid_homotypic = (float(np.clip(homotypic_weight, 0.0, 1.0)) * tail_calibrate(homotypic_percentile, center=center)).clip(0.0, 1.0)
    hybrid_overall = 1.0 - (1.0 - hybrid_heterotypic) * (1.0 - hybrid_homotypic)
    return hybrid_overall.astype(float), hybrid_homotypic.astype(float), hybrid_heterotypic.astype(float)
