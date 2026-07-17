"""Training set construction for DuoDose classifiers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SimulatedDoublets


def _rank01(values: pd.Series) -> pd.Series:
    values = pd.Series(values, index=values.index).replace([np.inf, -np.inf], np.nan).fillna(values.median())
    if values.nunique(dropna=True) <= 1:
        return pd.Series(0.0, index=values.index)
    return values.rank(method="average", pct=True)


def build_training_data(
    adata,
    simulated_doublets: SimulatedDoublets,
    observed_features: pd.DataFrame,
    simulated_features: pd.DataFrame,
    reliable_negative_quantile: float = 0.3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build first-version supervised training data.

    Simulated heterotypic/homotypic doublets provide positives. Low-risk observed
    cells provide pseudo-negative clean examples. Obvious low-quality observed
    cells are included when present.
    """

    feature_columns = observed_features.select_dtypes(include=[np.number]).columns
    observed = observed_features[feature_columns].copy()
    simulated = simulated_features.reindex(columns=feature_columns, fill_value=0.0).copy()

    risk = (
        _rank01(observed["heterotypic_similarity"])
        + _rank01(observed["homotypic_similarity"])
        + _rank01(observed["artificial_neighbor_fraction"])
        + _rank01(observed["dosage_residual"].clip(lower=0.0))
        + _rank01(observed.get("duodose_low_quality_score", pd.Series(0.0, index=observed.index)))
    ) / 5.0
    threshold = float(risk.quantile(np.clip(reliable_negative_quantile, 0.01, 0.9)))
    clean_idx = risk.index[risk <= threshold]
    if len(clean_idx) == 0 and len(risk):
        clean_idx = risk.sort_values().index[: max(1, len(risk) // 10)]

    low_quality_score = observed.get("duodose_low_quality_score", pd.Series(0.0, index=observed.index))
    low_quality_idx = low_quality_score.index[low_quality_score >= 0.95]
    low_quality_idx = low_quality_idx.difference(clean_idx)

    X_parts = []
    y_parts = []
    if len(clean_idx):
        X_parts.append(observed.loc[clean_idx])
        y_parts.append(pd.Series("clean", index=clean_idx))
    if len(low_quality_idx):
        X_parts.append(observed.loc[low_quality_idx])
        y_parts.append(pd.Series("low_quality", index=low_quality_idx))
    if len(simulated):
        labels = pd.Series(
            np.where(simulated_doublets.doublet_type == "heterotypic", "heterotypic_doublet", "homotypic_doublet"),
            index=simulated.index,
        )
        X_parts.append(simulated)
        y_parts.append(labels)

    if not X_parts:
        raise ValueError("No training examples were constructed.")
    X_train = pd.concat(X_parts, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = pd.concat(y_parts, axis=0)
    return X_train, y_train

