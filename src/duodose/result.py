"""Public result object returned by DuoDose."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DuoDoseResult:
    scores: pd.DataFrame
    threshold: float | None
    backend: str
    training_summary: dict[str, Any] = field(default_factory=dict)
    feature_audit: dict[str, Any] = field(default_factory=dict)
    parent_audit: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    model_metadata: dict[str, Any] = field(default_factory=dict)

    def add_to_adata(self, adata) -> None:
        aligned = self.scores.reindex(adata.obs_names)
        mapping = {
            "duodose_score": "duodose_score",
            "duodose_homotypic_score": "duodose_homotypic_score",
            "duodose_heterotypic_score": "duodose_heterotypic_score",
            "predicted_doublet": "duodose_prediction",
            "predicted_subtype": "duodose_subtype",
            "subtype_confidence": "duodose_subtype_confidence",
        }
        for source, destination in mapping.items():
            adata.obs[destination] = aligned[source].to_numpy()
        adata.uns["duodose"] = {
            "backend": self.backend,
            "threshold": self.threshold,
            "training_summary": self.training_summary,
            "feature_audit": self.feature_audit,
            "parent_audit": self.parent_audit,
            "config": self.config,
            "model_metadata": self.model_metadata,
        }

    def to_csv(self, path: str | Path) -> None:
        self.scores.to_csv(path, index=True)

    def write_h5ad(self, adata, path: str | Path) -> None:
        output = adata.copy()
        self.add_to_adata(output)
        output.write_h5ad(path)
