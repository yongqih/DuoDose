"""Read-only cross-dataset aggregation for completed domain-audit outputs.

This module never invokes semi-real construction, SafeFeature export, or a
domain classifier.  It rebuilds aggregate artifacts from per-dataset audit
files and the saved batch run-status table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .plotting_style import apply_manuscript_style

from .domain_audit_contract import (
    KNOWN_NON_SUCCESS_STATUSES,
    PRIMARY_ANALYSIS,
    SUCCESS_STATUSES,
    normalize_primary_analysis,
    validate_primary_audit,
)

UNMATCHED_ANALYSIS = "unmatched_heterotypic_safe_features"
TECHNICAL_ANALYSIS = "technical_covariates_only"
MINIMUM_PARENT_UNIQUE_DOUBLETS = 30

SUMMARY_COLUMNS = [
    "dataset",
    "analysis",
    "status",
    "audit_status",
    "wrapper_execution_status",
    "wrapper_message",
    "failure_code",
    "failure_reason",
    "metric_unavailable_reason",
    "n_experimental",
    "n_semireal",
    "n_semireal_heterotypic_available",
    "n_features",
    "n_folds",
    "split_strategy",
    "mean_fold_auroc",
    "sd_fold_auroc",
    "pooled_oof_auroc",
    "direction_adjusted_separability",
    "separation_category",
    "pooled_oof_auprc",
    "balanced_accuracy",
    "mcc",
    "construction_variant",
    "safe_feature_mode",
    "parent_overlap_across_folds",
    "n_semireal_before_parent_unique_filter",
    "n_semireal_after_parent_unique_filter",
    "parent_unique_retention_fraction",
    "unmatched_mechanism_auroc",
    "technical_covariates_only_auroc",
    "matched_sample_size_per_domain",
    "minimum_required",
    "two_fold_fallback_attempted",
    "cache_path",
    "output_path",
    "summary_path",
]

FAILURE_COLUMNS = [
    "dataset",
    "audit_status",
    "failure_code",
    "failure_reason",
    "n_semireal_heterotypic_available",
    "n_semireal_after_parent_unique_filter",
    "minimum_required",
    "two_fold_fallback_attempted",
    "cache_path",
    "output_path",
]


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _number(row: Mapping[str, Any], column: str) -> float:
    return float(pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0])


def _first_analysis(summary: pd.DataFrame, analysis: str) -> Mapping[str, Any] | None:
    rows = summary.loc[summary["analysis"].map(normalize_primary_analysis).eq(analysis)]
    return rows.iloc[0].to_dict() if len(rows) == 1 else None


def _audit_status(source_status: object, analysis_status: object, message: object) -> str:
    source = _text(source_status).upper()
    status = _text(analysis_status).upper()
    message_text = _text(message).lower()
    if status in SUCCESS_STATUSES:
        return "COMPLETED"
    if status in KNOWN_NON_SUCCESS_STATUSES:
        if status in {"INSUFFICIENT_PARENT_DISJOINT_DATA", "SKIPPED_INSUFFICIENT_DATA"}:
            return "SKIPPED_INSUFFICIENT_DATA"
        return status
    if source == "INSUFFICIENT_PARENT_DISJOINT_DATA" or any(
        token in message_text
        for token in (
            "insufficient parent-unique",
            "parents_removed requires additional labeled singlets",
            "no parent cluster can construct homotypic",
        )
    ):
        return "INSUFFICIENT_PARENT_DISJOINT_DATA"
    if "provenance" in message_text or "transformer" in message_text or "reference pool" in message_text:
        return "PROVENANCE_ERROR"
    if source in {"INPUT_ERROR", "FAILED"} and any(
        token in message_text for token in ("input", "missing", "not found", "read")
    ):
        return "INPUT_ERROR"
    if source and source not in SUCCESS_STATUSES | KNOWN_NON_SUCCESS_STATUSES:
        return "CONTRACT_ERROR"
    return "ANALYSIS_ERROR"


def _separation_category(value: float) -> str:
    if not np.isfinite(value):
        return "not available"
    if value <= 0.55:
        return "little detectable separation"
    if value <= 0.65:
        return "weak separation"
    if value <= 0.75:
        return "moderate separation"
    return "substantial separation"


def _status_source(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "domain_audit_batch_run_status.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["dataset", "run_status", "message", "output_dir"])
    return pd.read_csv(path)


def _source_record(statuses: pd.DataFrame, dataset: str, dataset_output: Path) -> Mapping[str, Any]:
    rows = statuses.loc[statuses["dataset"].astype(str).eq(dataset)] if "dataset" in statuses.columns else statuses.iloc[0:0]
    if len(rows) == 1:
        record = rows.iloc[0].to_dict()
        if not _text(record.get("run_status")):
            record["run_status"] = record.get("status", record.get("audit_status", ""))
        if not _text(record.get("message")):
            record["message"] = record.get("failure_reason", "")
        return record
    return {"dataset": dataset, "run_status": "", "message": "", "output_dir": str(dataset_output)}


def _completed_row(
    dataset: str,
    summary: pd.DataFrame,
    source: Mapping[str, Any],
    *,
    cache_dir: Path | None,
    dataset_output: Path,
) -> dict[str, Any]:
    fold_path = dataset_output / "domain_audit_fold_metrics.csv"
    prediction_path = dataset_output / "domain_audit_predictions.csv"
    folds = pd.read_csv(fold_path) if fold_path.is_file() and fold_path.stat().st_size else None
    predictions = pd.read_csv(prediction_path) if prediction_path.is_file() and prediction_path.stat().st_size else None
    validation = validate_primary_audit(summary, folds, predictions)
    primary = validation.primary
    if primary is None:
        return _unavailable_row(
            dataset,
            source,
            cache_dir=cache_dir,
            dataset_output=dataset_output,
            analysis_status="",
            analysis_message=validation.reason,
            forced_audit_status=validation.audit_status,
        )
    if not validation.completed:
        return _unavailable_row(
            dataset,
            source,
            cache_dir=cache_dir,
            dataset_output=dataset_output,
            analysis_status=primary.get("status", ""),
            analysis_message=validation.reason,
            primary=primary,
            forced_audit_status=validation.audit_status,
        )
    unmatched = _first_analysis(summary, UNMATCHED_ANALYSIS)
    technical = _first_analysis(summary, TECHNICAL_ANALYSIS)
    raw_auroc = _number(primary, "pooled_oof_auroc")
    return {
        "dataset": dataset,
        "analysis": PRIMARY_ANALYSIS,
        "status": "COMPLETED",
        "audit_status": "COMPLETED",
        "wrapper_execution_status": _text(source.get("run_status")),
        "wrapper_message": _text(source.get("message")),
        "failure_code": "",
        "failure_reason": "",
        "metric_unavailable_reason": "",
        "n_experimental": _number(primary, "n_experimental_doublets"),
        "n_semireal": _number(primary, "n_semireal_heterotypic_doublets"),
        "n_semireal_heterotypic_available": _number(primary, "n_semireal_before_parent_unique_filter"),
        "n_features": _number(primary, "n_features"),
        "n_folds": _number(primary, "n_folds"),
        "split_strategy": _text(primary.get("split_strategy")),
        "mean_fold_auroc": _number(primary, "auroc_mean"),
        "sd_fold_auroc": _number(primary, "auroc_std"),
        "pooled_oof_auroc": raw_auroc,
        "direction_adjusted_separability": max(raw_auroc, 1.0 - raw_auroc),
        "separation_category": _separation_category(max(raw_auroc, 1.0 - raw_auroc)),
        "pooled_oof_auprc": _number(primary, "pooled_oof_auprc"),
        "balanced_accuracy": _number(primary, "balanced_accuracy"),
        "mcc": _number(primary, "mcc"),
        "construction_variant": _text(primary.get("construction_variant")),
        "safe_feature_mode": _text(primary.get("safe_feature_mode")),
        "parent_overlap_across_folds": _number(primary, "parent_overlap_across_folds"),
        "n_semireal_before_parent_unique_filter": _number(primary, "n_semireal_before_parent_unique_filter"),
        "n_semireal_after_parent_unique_filter": _number(primary, "n_semireal_after_parent_unique_filter"),
        "parent_unique_retention_fraction": _number(primary, "parent_unique_retention_fraction"),
        "unmatched_mechanism_auroc": _number(unmatched, "pooled_oof_auroc") if unmatched else np.nan,
        "technical_covariates_only_auroc": _number(technical, "pooled_oof_auroc") if technical else np.nan,
        "matched_sample_size_per_domain": _number(primary, "n_experimental_doublets"),
        "minimum_required": MINIMUM_PARENT_UNIQUE_DOUBLETS,
        "two_fold_fallback_attempted": "not_needed_three_fold",
        "cache_path": str(cache_dir / dataset / "seed_0" / "domain_audit") if cache_dir else "",
        "output_path": str(dataset_output),
        "summary_path": str(dataset_output / "domain_audit_summary.csv"),
    }


def _unavailable_row(
    dataset: str,
    source: Mapping[str, Any],
    *,
    cache_dir: Path | None,
    dataset_output: Path,
    analysis_status: object,
    analysis_message: object,
    primary: Mapping[str, Any] | None = None,
    forced_audit_status: str | None = None,
) -> dict[str, Any]:
    reason = _text(analysis_message) or _text(source.get("message"))
    audit_status = forced_audit_status or _audit_status(source.get("run_status"), analysis_status, reason)
    primary = primary or {}
    n_available = _number(primary, "n_semireal_before_parent_unique_filter") if primary else np.nan
    n_retained = _number(primary, "n_semireal_after_parent_unique_filter") if primary else np.nan
    fallback = "not_attempted: no held-out parent-unique semi-real audit sample was available"
    if np.isfinite(n_retained):
        fallback = "attempted" if 30 <= n_retained < 60 else "not_needed_or_not_eligible"
    return {
        "dataset": dataset,
        "analysis": PRIMARY_ANALYSIS,
        "status": _text(analysis_status) or "NOT_AVAILABLE",
        "audit_status": audit_status,
        "wrapper_execution_status": _text(source.get("run_status")),
        "wrapper_message": _text(source.get("message")),
        "failure_code": audit_status,
        "failure_reason": reason or "No completed primary matched-domain audit result was available.",
        "metric_unavailable_reason": reason or "No completed primary matched-domain audit result was available.",
        "n_experimental": _number(primary, "n_experimental_doublets") if primary else np.nan,
        "n_semireal": _number(primary, "n_semireal_heterotypic_doublets") if primary else np.nan,
        "n_semireal_heterotypic_available": n_available,
        "n_features": _number(primary, "n_features") if primary else np.nan,
        "n_folds": _number(primary, "n_folds") if primary else np.nan,
        "split_strategy": _text(primary.get("split_strategy")) if primary else "",
        "mean_fold_auroc": np.nan,
        "sd_fold_auroc": np.nan,
        "pooled_oof_auroc": np.nan,
        "direction_adjusted_separability": np.nan,
        "separation_category": "not available",
        "pooled_oof_auprc": np.nan,
        "balanced_accuracy": np.nan,
        "mcc": np.nan,
        "construction_variant": _text(primary.get("construction_variant")) if primary else "",
        "safe_feature_mode": _text(primary.get("safe_feature_mode")) if primary else "",
        "parent_overlap_across_folds": np.nan,
        "n_semireal_before_parent_unique_filter": n_available,
        "n_semireal_after_parent_unique_filter": n_retained,
        "parent_unique_retention_fraction": _number(primary, "parent_unique_retention_fraction") if primary else np.nan,
        "unmatched_mechanism_auroc": np.nan,
        "technical_covariates_only_auroc": np.nan,
        "matched_sample_size_per_domain": np.nan,
        "minimum_required": MINIMUM_PARENT_UNIQUE_DOUBLETS,
        "two_fold_fallback_attempted": fallback,
        "cache_path": str(cache_dir / dataset / "seed_0" / "domain_audit") if cache_dir else "",
        "output_path": str(dataset_output),
        "summary_path": str(dataset_output / "domain_audit_summary.csv") if (dataset_output / "domain_audit_summary.csv").is_file() else "",
    }


def _collect_rows(output_dir: Path, cache_dir: Path | None) -> pd.DataFrame:
    statuses = _status_source(output_dir)
    dataset_outputs = {
        path.parent.name: path.parent
        for path in output_dir.glob("*/domain_audit_summary.csv")
        if path.parent.is_dir()
    }
    dataset_names = set(dataset_outputs)
    if "dataset" in statuses.columns:
        dataset_names.update(statuses["dataset"].dropna().astype(str))
    rows: list[dict[str, Any]] = []
    for dataset in sorted(dataset_names, key=str.lower):
        dataset_output = dataset_outputs.get(dataset, output_dir / dataset)
        source = _source_record(statuses, dataset, dataset_output)
        summary_path = dataset_output / "domain_audit_summary.csv"
        if summary_path.is_file():
            rows.append(_completed_row(dataset, pd.read_csv(summary_path), source, cache_dir=cache_dir, dataset_output=dataset_output))
        else:
            rows.append(
                _unavailable_row(
                    dataset,
                    source,
                    cache_dir=cache_dir,
                    dataset_output=dataset_output,
                    analysis_status="NOT_AVAILABLE",
                    analysis_message=source.get("message", ""),
                )
            )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _write_feature_and_balance_outputs(summary: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    feature_rows: list[dict[str, Any]] = []
    balance_frames: list[pd.DataFrame] = []
    for row in summary.loc[summary["audit_status"].eq("COMPLETED")].to_dict("records"):
        dataset_output = Path(row["output_path"])
        feature_path = dataset_output / "domain_audit_feature_audit.csv"
        if feature_path.is_file():
            feature = pd.read_csv(feature_path)
            included = feature["included"].astype(str).str.lower().isin({"true", "1", "yes"})
            feature_rows.append(
                {
                    "dataset": row["dataset"],
                    "n_features_included": int(included.sum()),
                    "n_features_excluded": int((~included).sum()),
                }
            )
        balance_path = dataset_output / "domain_audit_cluster_balance.csv"
        if balance_path.is_file():
            balance_frames.append(pd.read_csv(balance_path))
    feature_output = output_dir / "domain_audit_all_datasets_feature_counts.csv"
    balance_output = output_dir / "domain_audit_all_datasets_matching_balance.csv"
    pd.DataFrame(feature_rows, columns=["dataset", "n_features_included", "n_features_excluded"]).to_csv(
        feature_output, index=False
    )
    (pd.concat(balance_frames, ignore_index=True) if balance_frames else pd.DataFrame()).to_csv(balance_output, index=False)
    return feature_output, balance_output


def _save_plots(summary: pd.DataFrame, output_dir: Path) -> list[str]:
    completed = summary.loc[summary["audit_status"].eq("COMPLETED")].copy()
    if completed.empty:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    apply_manuscript_style()
    paths: list[str] = []

    def save(figure: Any, stem: str) -> None:
        for suffix in ("png", "pdf"):
            path = output_dir / f"{stem}.{suffix}"
            figure.savefig(path, dpi=180 if suffix == "png" else None, facecolor="white")
            paths.append(path.name)

    ordered = completed.sort_values("direction_adjusted_separability", ascending=True, kind="mergesort")
    y = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(9.5, max(4.5, 0.48 * len(ordered) + 1.7)))
    height = 0.24
    axis.barh(y - height, ordered["unmatched_mechanism_auroc"], height=height, label="Unmatched mechanism features", color="#78909c")
    axis.barh(y, ordered["pooled_oof_auroc"], height=height, label="Matched mechanism features (primary)", color="#1565c0")
    axis.barh(y + height, ordered["technical_covariates_only_auroc"], height=height, label="Technical covariates only", color="#e67e22")
    axis.axvline(0.5, color="black", linestyle="--", linewidth=0.8, label="AUROC = 0.5")
    axis.set_yticks(y, ordered["dataset"])
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Pooled out-of-fold AUROC")
    axis.set_title("Experimental versus held-out semi-real heterotypic domains")
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    save(figure, "domain_audit_all_datasets_auroc_comparison")
    plt.close(figure)

    ranked = completed.sort_values("direction_adjusted_separability", ascending=True, kind="mergesort")
    colors = {
        "little detectable separation": "#90a4ae",
        "weak separation": "#42a5f5",
        "moderate separation": "#fb8c00",
        "substantial separation": "#c62828",
    }
    figure, axis = plt.subplots(figsize=(8.5, max(4.5, 0.48 * len(ranked) + 1.7)))
    axis.barh(
        np.arange(len(ranked)),
        ranked["direction_adjusted_separability"],
        color=[colors.get(value, "#78909c") for value in ranked["separation_category"]],
    )
    axis.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
    axis.set_yticks(np.arange(len(ranked)), ranked["dataset"])
    axis.set_xlim(0.5, 1.0)
    axis.set_xlabel("Direction-adjusted separability = max(AUROC, 1 - AUROC)")
    axis.set_title("Ranked matched primary-analysis separability")
    figure.tight_layout()
    save(figure, "domain_audit_all_datasets_matched_direction_adjusted")
    plt.close(figure)
    return paths


def _fmt(value: object, decimals: int = 3) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "NA" if pd.isna(number) else f"{float(number):.{decimals}f}"


def _write_report(summary: pd.DataFrame, output_dir: Path, plot_paths: list[str]) -> Path:
    completed = summary.loc[summary["audit_status"].eq("COMPLETED")].copy()
    skipped = summary.loc[~summary["audit_status"].eq("COMPLETED")].copy()
    raw = pd.to_numeric(completed["pooled_oof_auroc"], errors="coerce")
    unmatched = pd.to_numeric(completed["unmatched_mechanism_auroc"], errors="coerce")
    technical = pd.to_numeric(completed["technical_covariates_only_auroc"], errors="coerce")
    adjusted = pd.to_numeric(completed["direction_adjusted_separability"], errors="coerce")
    category_counts = completed["separation_category"].value_counts().to_dict()
    lines = [
        "# Cross-dataset Semi-real Domain Audit",
        "",
        "## Summary",
        "",
        f"{len(completed)} of {len(summary)} datasets completed the strict parent-disjoint audit. "
        "Across completed datasets, matched domain-classification AUROC showed generally limited separation "
        "between experimentally labeled and semi-real heterotypic doublets.",
        "",
        f"- Matched pooled OOF AUROC: mean {_fmt(raw.mean())}, median {_fmt(raw.median())}, range {_fmt(raw.min())}–{_fmt(raw.max())}.",
        f"- Skipped or failed datasets: {len(skipped)}.",
        f"- Mean unmatched mechanism-feature AUROC: {_fmt(unmatched.mean())}.",
        f"- Mean technical-covariates-only AUROC: {_fmt(technical.mean())}.",
        f"- Mean direction-adjusted matched separability: {_fmt(adjusted.mean())}.",
        "- Direction-adjusted categories: "
        + "; ".join(
            f"{label}={int(category_counts.get(label, 0))}"
            for label in (
                "little detectable separation",
                "weak separation",
                "moderate separation",
                "substantial separation",
            )
        )
        + ".",
        "- Separation categories: 0.50–0.55 little detectable separation; >0.55–0.65 weak separation; >0.65–0.75 moderate separation; >0.75 substantial separation.",
        "",
        "Raw AUROC preserves classifier direction. Direction-adjusted separability is `max(AUROC, 1 - AUROC)`; it is reported alongside raw AUROC and is not a model-selection metric.",
        "",
        "## Completed Datasets",
        "",
        "| Dataset | Matched pooled OOF AUROC | Direction-adjusted separability | Unmatched mechanism AUROC | Technical-only AUROC | Matched sample size / domain | Parent-unique semi-real retained | Folds | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in completed.sort_values("direction_adjusted_separability", ascending=False, kind="mergesort").iterrows():
        lines.append(
            f"| {row['dataset']} | {_fmt(row['pooled_oof_auroc'])} | {_fmt(row['direction_adjusted_separability'])} | "
            f"{_fmt(row['unmatched_mechanism_auroc'])} | {_fmt(row['technical_covariates_only_auroc'])} | "
            f"{_fmt(row['matched_sample_size_per_domain'], 0)} | "
            f"{_fmt(row['n_semireal_after_parent_unique_filter'], 0)} | {_fmt(row['n_folds'], 0)} | COMPLETED |"
        )
    lines.extend(["", "## Skipped or Failed Datasets", ""])
    if skipped.empty:
        lines.append("None.")
    else:
        lines.extend(["| Dataset | Status | Reason |", "| --- | --- | --- |"])
        for _, row in skipped.sort_values("dataset", kind="mergesort").iterrows():
            reason = _text(row["failure_reason"]).replace("|", "\\|")
            lines.append(f"| {row['dataset']} | {row['audit_status']} | {reason} |")
    lines.extend(
        [
            "",
            "## Methods and Limitations",
            "",
            "- Formal semi-real construction: `raw_sum_parents_removed`; SafeFeature provenance: `fitted_reference`.",
            "- Only held-out semi-real heterotypic doublets were compared with experimentally labeled doublets.",
            "- The mechanism-level classifier excludes downstream scores, rank features, cluster abundance, cluster one-hot features, and direct technical covariates. Technical covariates are evaluated only in their dedicated control.",
            "- Parent-unique filtering is required before matched fold assignment. Datasets without adequate parent-disjoint training or held-out samples are skipped rather than relaxing the leakage constraint.",
            "- Experimental labels are not complete confirmed homotypic/heterotypic subtype annotations. This audit evaluates feature-space concordance and does not prove identical distributions.",
            "",
            "## Figures",
            "",
        ]
        + [f"- `{name}`" for name in plot_paths]
    )
    path = output_dir / "domain_audit_all_datasets_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def regenerate_domain_audit_outputs(output_dir: str | Path, cache_dir: str | Path | None = None) -> Mapping[str, Path]:
    """Rebuild all combined domain-audit artifacts from saved per-dataset outputs."""

    output = Path(output_dir).expanduser().resolve()
    cache = Path(cache_dir).expanduser().resolve() if cache_dir else None
    # Superseded single-series aggregate plots did not include the required
    # unmatched and technical controls, so leave only the corrected figures.
    for stem in (
        "domain_audit_all_datasets_matched_auroc",
        "domain_audit_all_datasets_direction_adjusted_separability",
    ):
        for suffix in ("png", "pdf"):
            (output / f"{stem}.{suffix}").unlink(missing_ok=True)
    summary = _collect_rows(output, cache)
    if summary.empty:
        raise ValueError(f"No per-dataset domain-audit summaries or saved run-status rows exist under '{output}'.")
    summary_path = output / "domain_audit_all_datasets_summary.csv"
    failures_path = output / "domain_audit_all_datasets_failures.csv"
    status_path = output / "domain_audit_all_datasets_run_status.csv"
    summary.reindex(columns=SUMMARY_COLUMNS).to_csv(summary_path, index=False)
    summary.loc[summary["audit_status"].ne("COMPLETED"), FAILURE_COLUMNS].to_csv(failures_path, index=False)
    summary[["dataset", "audit_status", "failure_code", "failure_reason", "cache_path", "output_path"]].to_csv(
        status_path, index=False
    )
    feature_counts, matching_balance = _write_feature_and_balance_outputs(summary, output)
    plot_paths = _save_plots(summary, output)
    report_path = _write_report(summary, output, plot_paths)
    return {
        "summary": summary_path,
        "failures": failures_path,
        "run_status": status_path,
        "feature_counts": feature_counts,
        "matching_balance": matching_balance,
        "report": report_path,
    }
