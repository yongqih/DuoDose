"""Configuration objects for public DuoDose workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from .models.registry import DEFAULT_DUODOSE_BACKEND, get_backend_spec
from .rf_weighting import FORMAL_HIGH_RNA_NEGATIVE_WEIGHT


@dataclass
class FeatureConfig:
    n_hvgs: int = 2000
    n_pcs: int = 40
    n_simulated_doublets: int | None = None


@dataclass
class DLConfig:
    max_epochs: int = 200
    patience: int = 20
    batch_size: int | None = None
    num_workers: int = 0
    amp: bool = False


@dataclass
class SemiRealConfig:
    n_singlets: int = 5000
    n_train_homotypic_doublets: int = 500
    n_train_heterotypic_doublets: int = 500
    n_validation_homotypic_doublets: int = 125
    n_validation_heterotypic_doublets: int = 125
    n_clusters: int = 12
    test_parent_fraction: float = 0.40
    validation_parent_fraction: float = 0.25
    high_rna_quantile: float = 0.90
    saturation_low: float = 0.60
    saturation_high: float = 1.00
    min_cluster_size: int = 10
    minimum_singlets: int = 200
    construction_variant: str = "raw_sum_parents_removed"
    safe_feature_mode: str = "fitted_reference"
    parent_disjoint: bool = True
    high_rna_negative_weight: float = FORMAL_HIGH_RNA_NEGATIVE_WEIGHT

    @classmethod
    def from_preset(cls, preset: Literal["fast", "default", "robust"]) -> "SemiRealConfig":
        if preset == "fast":
            return cls(
                n_singlets=500,
                n_train_homotypic_doublets=80,
                n_train_heterotypic_doublets=80,
                n_validation_homotypic_doublets=20,
                n_validation_heterotypic_doublets=20,
                n_clusters=6,
                min_cluster_size=5,
                minimum_singlets=200,
            )
        if preset == "robust":
            return cls(
                n_singlets=10000,
                n_train_homotypic_doublets=1000,
                n_train_heterotypic_doublets=1000,
                n_validation_homotypic_doublets=250,
                n_validation_heterotypic_doublets=250,
                n_clusters=16,
                min_cluster_size=15,
                minimum_singlets=300,
            )
        if preset != "default":
            raise ValueError("training_preset must be 'fast', 'default', or 'robust'")
        return cls()


@dataclass
class DuoDoseConfig:
    backend: str = DEFAULT_DUODOSE_BACKEND
    expected_doublet_rate: float = 0.08
    random_state: int = 0
    layer: str | None = None
    device: str = "auto"
    threshold_strategy: Literal["expected_rate", "probability"] | None = "expected_rate"
    threshold: float = 0.5
    training_preset: Literal["fast", "default", "robust"] = "default"
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    dl: DLConfig = field(default_factory=DLConfig)
    semireal: SemiRealConfig | None = None

    def __post_init__(self) -> None:
        self.backend = get_backend_spec(self.backend).alias
        if not 0.0 < float(self.expected_doublet_rate) < 1.0:
            raise ValueError("expected_doublet_rate must be between 0 and 1")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        if self.threshold_strategy not in {"expected_rate", "probability", None}:
            raise ValueError("threshold_strategy must be 'expected_rate', 'probability', or None")
        if not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if self.semireal is None:
            self.semireal = SemiRealConfig.from_preset(self.training_preset)
        if self.semireal.construction_variant != "raw_sum_parents_removed":
            raise ValueError("the public DuoDose protocol requires construction_variant='raw_sum_parents_removed'")
        if self.semireal.safe_feature_mode != "fitted_reference":
            raise ValueError("the public DuoDose protocol requires safe_feature_mode='fitted_reference'")
        if not self.semireal.parent_disjoint:
            raise ValueError("the public DuoDose protocol requires parent-disjoint semi-real splits")
        if float(self.semireal.high_rna_negative_weight) != FORMAL_HIGH_RNA_NEGATIVE_WEIGHT:
            raise ValueError(
                "the public DuoDose RF protocol fixes high_rna_negative_weight="
                f"{FORMAL_HIGH_RNA_NEGATIVE_WEIGHT}"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
