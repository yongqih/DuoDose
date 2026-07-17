"""Loading and validation for the frozen manuscript protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .rf_weighting import FORMAL_HIGH_RNA_NEGATIVE_WEIGHT


FINAL_CONSTRUCTION_VARIANT = "raw_sum_parents_removed"
FINAL_SAFE_FEATURE_MODE = "fitted_reference"
FINAL_MAIN_BACKEND = "rf"
FINAL_ABLATION_BACKEND = "dl"
FINAL_HIGH_RNA_NEGATIVE_WEIGHT = FORMAL_HIGH_RNA_NEGATIVE_WEIGHT


def default_protocol_path() -> Path:
    """Return the source-checkout protocol path or raise a useful error."""

    path = Path(__file__).resolve().parents[2] / "reproducibility" / "configs" / "final_protocol.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            "final_protocol.yaml is not available in this installation; pass its explicit path to load_final_protocol"
        )
    return path


def load_final_protocol(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the version-controlled final manuscript protocol."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency metadata covers normal installs
        raise ImportError("PyYAML is required to read DuoDose manuscript protocol files") from exc

    source = Path(path) if path is not None else default_protocol_path()
    with source.open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    if not isinstance(protocol, dict):
        raise ValueError(f"protocol file must contain a YAML mapping: {source}")
    _validate_final_protocol(protocol)
    protocol["_protocol_path"] = str(source.resolve())
    return protocol


def _validate_final_protocol(protocol: dict[str, Any]) -> None:
    models = protocol.get("models", {})
    construction = protocol.get("construction", {})
    if models.get("main_backend") != FINAL_MAIN_BACKEND:
        raise ValueError("final protocol main_backend must be 'rf'")
    if models.get("ablation_backend") != FINAL_ABLATION_BACKEND:
        raise ValueError("final protocol ablation_backend must be 'dl'")
    if float(models.get("high_rna_negative_weight", float("nan"))) != FINAL_HIGH_RNA_NEGATIVE_WEIGHT:
        raise ValueError(
            "final protocol high_rna_negative_weight must be "
            f"{FINAL_HIGH_RNA_NEGATIVE_WEIGHT}"
        )
    if construction.get("construction_variant") != FINAL_CONSTRUCTION_VARIANT:
        raise ValueError("final protocol construction_variant must be 'raw_sum_parents_removed'")
    if construction.get("safe_feature_mode") != FINAL_SAFE_FEATURE_MODE:
        raise ValueError("final protocol safe_feature_mode must be 'fitted_reference'")
    if construction.get("parent_disjoint") is not True:
        raise ValueError("final protocol must require parent-disjoint construction")
    methods = protocol.get("external_methods", {}).get("methods", [])
    expected = ["Scrublet", "scDblFinder", "DoubletFinder", "scds"]
    if list(methods) != expected:
        raise ValueError(f"final protocol external methods must be {expected}")
    application = protocol.get("real_application", {})
    if application.get("backend") != "rf":
        raise ValueError("real_application backend must be 'rf'")
    if application.get("internal_method_name") != "DuoDose-ML-CalibratedRF-SafeFeatures":
        raise ValueError("real_application must expose the frozen calibrated-RF implementation")
    if list(application.get("external_methods", [])) != ["Scrublet", "DoubletFinder", "scDblFinder", "scds"]:
        raise ValueError("real_application external panel order is frozen")
    if list(protocol.get("seeds", {}).get("real_application", [])) != [0]:
        raise ValueError("real_application must use seed 0")
    feature_allowlist = list(protocol.get("features", {}).get("allowlist", []))
    if "library_complexity_balance" not in feature_allowlist:
        raise ValueError("final protocol must include library_complexity_balance in the SafeFeature allowlist")
    evaluation = protocol.get("evaluation", {})
    primary_fpr = evaluation.get("primary_high_rna_fpr", {})
    if primary_fpr.get("metric") != "high_RNA_singlet_FPR_at_matched_50pct_homotypic_recall":
        raise ValueError("main-text high-RNA FPR must use matched 50% homotypic recall")
    if float(primary_fpr.get("target_homotypic_recall", float("nan"))) != 0.50:
        raise ValueError("main-text high-RNA FPR target_homotypic_recall must be 0.50")
    supplementary_fpr = evaluation.get("supplementary_high_rna_fpr", {})
    if float(supplementary_fpr.get("fixed_candidate_fraction", float("nan"))) != 0.20:
        raise ValueError("supplementary fixed candidate fraction must be 0.20")
