"""Shared producer/aggregator/completion contract for strict domain audits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


PRIMARY_ANALYSIS = "matched_heterotypic_safe_features"
PRIMARY_ANALYSIS_ALIASES = frozenset(
    {
        PRIMARY_ANALYSIS,
        "matched_safe_features",
        "matched_mechanism_features",
        "matched_raw_mechanism_features",
        "matched_heterotypic_mechanism_features",
    }
)
SUCCESS_STATUSES = frozenset({"PASS", "COMPLETED", "SUCCESS"})
KNOWN_NON_SUCCESS_STATUSES = frozenset(
    {
        "ANALYSIS_ERROR",
        "FAILED",
        "INCOMPLETE",
        "INSUFFICIENT_PARENT_DISJOINT_DATA",
        "NOT_AVAILABLE",
        "SKIPPED",
        "SKIPPED_INSUFFICIENT_DATA",
    }
)


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"true", "1", "yes"}


def normalize_primary_analysis(value: object) -> str:
    analysis = _text(value)
    return PRIMARY_ANALYSIS if analysis in PRIMARY_ANALYSIS_ALIASES else analysis


def normalize_success_status(value: object) -> str | None:
    status = _text(value).upper()
    if status in SUCCESS_STATUSES:
        return "COMPLETED"
    return None


def _support_has_primary_rows(frame: pd.DataFrame | None) -> bool:
    if frame is None or frame.empty or "analysis" not in frame.columns:
        return False
    analyses = frame["analysis"].map(normalize_primary_analysis)
    return bool(analyses.eq(PRIMARY_ANALYSIS).any())


@dataclass(frozen=True)
class PrimaryAuditValidation:
    completed: bool
    audit_status: str
    reason: str
    primary: Mapping[str, Any] | None


def validate_primary_audit(
    summary: pd.DataFrame,
    fold_metrics: pd.DataFrame | None,
    predictions: pd.DataFrame | None,
) -> PrimaryAuditValidation:
    """Validate one saved dataset-level primary audit without recomputation."""

    if summary.empty or "analysis" not in summary.columns or "is_primary" not in summary.columns:
        return PrimaryAuditValidation(False, "CONTRACT_ERROR", "dataset summary lacks analysis/is_primary columns", None)
    normalized = summary.copy()
    normalized["analysis"] = normalized["analysis"].map(normalize_primary_analysis)
    primary_rows = normalized.loc[normalized["is_primary"].map(_truthy) & normalized["analysis"].eq(PRIMARY_ANALYSIS)]
    if len(primary_rows) != 1:
        return PrimaryAuditValidation(False, "CONTRACT_ERROR", f"expected exactly one primary matched row; found {len(primary_rows)}", None)
    primary = primary_rows.iloc[0].to_dict()
    status = _text(primary.get("status")).upper()
    if status not in SUCCESS_STATUSES | KNOWN_NON_SUCCESS_STATUSES:
        return PrimaryAuditValidation(False, "CONTRACT_ERROR", f"unknown dataset-level primary status {primary.get('status')!r}", primary)
    if status not in SUCCESS_STATUSES:
        message = _text(primary.get("message")) or f"dataset-level primary status is {status}"
        normalized_status = "SKIPPED_INSUFFICIENT_DATA" if status in {"INSUFFICIENT_PARENT_DISJOINT_DATA", "SKIPPED_INSUFFICIENT_DATA"} else status
        return PrimaryAuditValidation(False, normalized_status, message, primary)
    metrics = pd.to_numeric(
        pd.Series([primary.get("pooled_oof_auroc"), primary.get("pooled_oof_auprc")]), errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(metrics).all():
        return PrimaryAuditValidation(False, "CONTRACT_ERROR", "successful primary row has non-finite pooled OOF AUROC/AUPRC", primary)
    overlap = pd.to_numeric(pd.Series([primary.get("parent_overlap_across_folds")]), errors="coerce").iloc[0]
    if not np.isfinite(overlap) or float(overlap) != 0.0:
        return PrimaryAuditValidation(False, "CONTRACT_ERROR", "successful primary row requires parent_overlap_across_folds == 0", primary)
    if _text(primary.get("fold_balance_status")).upper() != "PASS":
        return PrimaryAuditValidation(False, "CONTRACT_ERROR", "successful primary row requires fold_balance_status == PASS", primary)
    if _text(primary.get("transformer_reference_provenance_status")).upper() != "PASS":
        return PrimaryAuditValidation(False, "CONTRACT_ERROR", "successful primary row requires transformer_reference_provenance_status == PASS", primary)
    if not _support_has_primary_rows(fold_metrics):
        return PrimaryAuditValidation(False, "CONTRACT_ERROR", "primary fold metrics are missing", primary)
    if not _support_has_primary_rows(predictions):
        return PrimaryAuditValidation(False, "CONTRACT_ERROR", "primary predictions are missing", primary)
    primary["analysis"] = PRIMARY_ANALYSIS
    primary["status"] = "COMPLETED"
    return PrimaryAuditValidation(True, "COMPLETED", "", primary)


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def validate_primary_audit_files(dataset_output: str | Path) -> PrimaryAuditValidation:
    output = Path(dataset_output)
    summary = _read_csv(output / "domain_audit_summary.csv")
    if summary is None:
        return PrimaryAuditValidation(False, "CONTRACT_ERROR", "domain_audit_summary.csv is missing or unreadable", None)
    return validate_primary_audit(
        summary,
        _read_csv(output / "domain_audit_fold_metrics.csv"),
        _read_csv(output / "domain_audit_predictions.csv"),
    )


__all__ = [
    "KNOWN_NON_SUCCESS_STATUSES",
    "PRIMARY_ANALYSIS",
    "PRIMARY_ANALYSIS_ALIASES",
    "PrimaryAuditValidation",
    "SUCCESS_STATUSES",
    "normalize_primary_analysis",
    "normalize_success_status",
    "validate_primary_audit",
    "validate_primary_audit_files",
]
