"""Validate completion of the frozen DuoDose manuscript result tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from duodose.protocol import load_final_protocol  # noqa: E402
from duodose.rf_weighting import FORMAL_HIGH_RNA_NEGATIVE_WEIGHT  # noqa: E402
from duodose.domain_audit_contract import validate_primary_audit_files  # noqa: E402
from duodose.runtime_completeness import CANONICAL_RUNTIME_METHODS  # noqa: E402

COMPLETE, INCOMPLETE, FAILED, NOT_RUN = "COMPLETE", "INCOMPLETE", "FAILED", "NOT_RUN"
INTERNAL_METHODS = {"DuoDose", "DuoDose-DL"}
EXTERNAL_METHODS = {"Scrublet", "scDblFinder", "DoubletFinder", "scds"}
ALL_METHODS = INTERNAL_METHODS | EXTERNAL_METHODS
REAL_APPLICATION_METHODS = {"DuoDose", "Scrublet", "DoubletFinder", "scDblFinder", "scds"}
REAL_APPLICATION_FILES = (
    "real_application_umap_coordinates.csv.gz",
    "real_application_method_scores.csv.gz",
    "real_application_candidate_calls.csv.gz",
    "real_application_method_status.csv",
    "real_application_label_usage_audit.csv",
    "real_application_reference_audit.csv",
    "real_application_shared_embedding_audit.csv",
    "real_application_candidate_display_audit.csv",
    "real_application_candidate_summary.csv",
    "real_application_group_diagnostics.csv",
    "real_application_cross_method_umap.png",
    "real_application_cross_method_umap.pdf",
    "real_application_duodose_diagnostics.png",
    "real_application_duodose_diagnostics.pdf",
    "run_manifest.json",
    "figure_manifest.json",
)
CONTROLLED_METRICS = {
    "AUROC", "overall_AUPRC", "homotypic_AUPRC", "heterotypic_AUPRC",
    "macro_subtype_AUPRC", "homotypic_vs_high_RNA_singlet_AUPRC",
    "high_RNA_singlet_FPR",
    "high_RNA_singlet_FPR_at_matched_50pct_homotypic_recall",
    "high_RNA_singlet_FPR_at_matched_70pct_homotypic_recall",
    "high_RNA_singlet_FPR_at_matched_80pct_homotypic_recall",
    "high_RNA_singlet_FPR_at_fixed_20pct_candidate_budget",
    "high_RNA_singlet_FPR_at_true_doublet_budget",
    "precision_at_K", "recall_at_K", "status", "message",
}


@dataclass
class AnalysisStatus:
    analysis: str
    status: str
    completed_units: int
    expected_units: int
    failed_units: int
    missing_units: int
    message: str
    output_path: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _csv(path: Path) -> pd.DataFrame | None:
    if not _nonempty(path):
        return None
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def _json(path: Path) -> dict | None:
    if not _nonempty(path):
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _manifest(path: Path, protocol_hash: str, workflow: str, dataset: str, seed: int) -> bool:
    value = _json(path)
    if not value:
        return False
    try:
        return bool(value.get("workflow") == workflow and value.get("protocol_config_sha256") == protocol_hash and str(value.get("dataset")) == dataset and int(value.get("seed", -1)) == int(seed))
    except (TypeError, ValueError):
        return False


def _result(name: str, root: Path, expected: int, complete: int, failed: int, missing: int, messages: list[str]) -> AnalysisStatus:
    status = NOT_RUN if not root.exists() else FAILED if failed else COMPLETE if complete == expected and not missing else INCOMPLETE
    message = "; ".join(messages[:8])
    if len(messages) > 8:
        message += f"; plus {len(messages) - 8} additional issue(s)"
    return AnalysisStatus(name, status, complete, expected, failed, missing, message, str(root))


def _metrics_valid(run_dir: Path, filename: str, methods: set[str], dataset: str, seed: int, required: tuple[str, ...]) -> bool:
    if any(not _nonempty(run_dir / name) for name in required):
        return False
    frame = _csv(run_dir / filename)
    if frame is None or set(frame.get("method", pd.Series(dtype=str)).astype(str)) != methods:
        return False
    required_columns = {"dataset", "seed", "method", "status", "message", "AUROC"}
    required_columns |= CONTROLLED_METRICS
    if not required_columns.issubset(frame.columns):
        return False
    return bool(not frame.duplicated(["dataset", "seed", "method"]).any() and frame["dataset"].astype(str).eq(dataset).all() and pd.to_numeric(frame["seed"], errors="coerce").eq(seed).all() and frame.get("status", pd.Series(dtype=str)).astype(str).str.lower().eq("success").all())


def _formal_rf_weight_valid(run_dir: Path) -> bool:
    """Require the one fixed high-RNA weighting rule in formal RF outputs."""

    summary = _csv(run_dir / "training_summaries.csv")
    if summary is None:
        return False
    rf = summary.loc[summary.get("public_method_name", pd.Series(index=summary.index, dtype=str)).astype(str).eq("DuoDose")]
    if len(rf) != 1:
        return False
    weight = pd.to_numeric(rf.get("high_rna_negative_weight", pd.Series(index=rf.index, dtype=float)), errors="coerce")
    if len(weight) != 1 or not weight.eq(FORMAL_HIGH_RNA_NEGATIVE_WEIGHT).all():
        return False
    return bool(
        rf.get("sample_weight_used", pd.Series(index=rf.index, dtype=bool)).astype(bool).all()
        and pd.to_numeric(rf.get("sample_weight_min", pd.Series(index=rf.index, dtype=float)), errors="coerce").eq(1.0).all()
        and pd.to_numeric(rf.get("sample_weight_max", pd.Series(index=rf.index, dtype=float)), errors="coerce").eq(FORMAL_HIGH_RNA_NEGATIVE_WEIGHT).all()
    )


def check_controlled(results: Path, protocol: dict, protocol_hash: str) -> AnalysisStatus:
    root = results / "controlled"
    datasets, seeds = protocol["datasets"]["real_doublet_enriched"], protocol["seeds"]["controlled_benchmark"]
    required = ("controlled_metrics.csv", "training_summaries.csv", "semireal_parent_map.csv.gz", "construction_report.json", "controlled_test_predictions.csv.gz", "controlled_high_RNA_operating_points.csv", "run_manifest.json", "output_manifest.json")
    complete = failed = missing = 0
    messages: list[str] = []
    for dataset in datasets:
        for seed in seeds:
            run_dir = root / dataset / f"seed_{seed}"
            valid = (
                _metrics_valid(run_dir, "controlled_metrics.csv", INTERNAL_METHODS, dataset, seed, required)
                and _formal_rf_weight_valid(run_dir)
                and _manifest(run_dir / "run_manifest.json", protocol_hash, "controlled_benchmark", dataset, seed)
            )
            if valid:
                complete += 1
            elif _nonempty(run_dir / "failure.json"):
                failed += 1; messages.append(f"{dataset}/seed_{seed}: explicit failure")
            else:
                missing += 1; messages.append(f"{dataset}/seed_{seed}: missing or invalid output")
    aggregate = _csv(root / "controlled_metrics_all_runs.csv")
    if complete == len(datasets) * len(seeds) and (aggregate is None or len(aggregate) != complete * 2):
        missing += 1; messages.append("controlled_metrics_all_runs.csv does not contain the complete grid")
    return _result("controlled_benchmark", root, len(datasets) * len(seeds), complete, failed, missing, messages)


def _check_method_grid(results: Path, protocol: dict, protocol_hash: str, *, name: str, subdir: str, workflow: str, metric_file: str, expected_methods: set[str]) -> AnalysisStatus:
    root = results / subdir
    datasets = protocol["datasets"]["real_doublet_enriched"]
    seeds = protocol["seeds"]["controlled_benchmark"]
    required = (metric_file, "external_method_status.csv", "external_high_RNA_operating_points.csv", "run_manifest.json", "output_manifest.json")
    complete = failed = missing = 0
    messages: list[str] = []
    for dataset in datasets:
        for seed in seeds:
            run_dir = root / dataset / f"seed_{seed}"
            status = _csv(run_dir / "external_method_status.csv")
            status_ok = bool(status is not None and set(status.get("method", pd.Series(dtype=str)).astype(str)) == EXTERNAL_METHODS and status["status"].astype(str).str.lower().eq("success").all())
            valid = _metrics_valid(run_dir, metric_file, expected_methods, dataset, seed, required) and status_ok and _manifest(run_dir / "run_manifest.json", protocol_hash, workflow, dataset, seed)
            if valid:
                complete += 1
            elif status is not None and status.get("status", pd.Series(dtype=str)).astype(str).str.lower().eq("failed").any():
                failed += 1; messages.append(f"{dataset}/seed_{seed}: external method failure")
            else:
                missing += 1; messages.append(f"{dataset}/seed_{seed}: missing or invalid output")
    return _result(name, root, len(datasets) * len(seeds), complete, failed, missing, messages)


def check_external(results: Path, protocol: dict, protocol_hash: str) -> AnalysisStatus:
    return _check_method_grid(results, protocol, protocol_hash, name="external_baselines", subdir="external", workflow="external_benchmark", metric_file="external_controlled_metrics.csv", expected_methods=EXTERNAL_METHODS)


def _false_value(value: object) -> bool:
    return str(value).strip().lower() in {"false", "0", "no"}


def check_real_application(results: Path, protocol: dict, protocol_hash: str) -> AnalysisStatus:
    root = results / "real_application"
    datasets = list(protocol["datasets"]["real_application"])
    seeds = list(protocol["seeds"]["real_application"])
    complete = failed = missing = 0
    messages: list[str] = []
    for dataset in datasets:
        for seed in seeds:
            run_dir = root / dataset / f"seed_{seed}"
            absent = [name for name in REAL_APPLICATION_FILES if not _nonempty(run_dir / name)]
            status = _csv(run_dir / "real_application_method_status.csv")
            label = _csv(run_dir / "real_application_label_usage_audit.csv")
            reference = _csv(run_dir / "real_application_reference_audit.csv")
            embedding = _csv(run_dir / "real_application_shared_embedding_audit.csv")
            candidate_display = _csv(run_dir / "real_application_candidate_display_audit.csv")
            figure = _json(run_dir / "figure_manifest.json") or {}
            status_ok = bool(
                status is not None
                and set(status.get("method", pd.Series(dtype=str)).astype(str)) == REAL_APPLICATION_METHODS
                and status.get("status", pd.Series(dtype=str)).astype(str).str.lower().eq("success").all()
            )
            expected_label_checks = {
                "experimental_label_used_in_training",
                "experimental_label_used_in_reference_selection",
                "experimental_label_used_in_threshold_selection",
                "experimental_label_used_in_embedding",
            }
            label_ok = bool(
                label is not None
                and expected_label_checks.issubset(set(label.get("check", pd.Series(dtype=str)).astype(str)))
                and label.loc[label["check"].isin(expected_label_checks), "value"].map(_false_value).all()
                and label.get("status", pd.Series(dtype=str)).astype(str).eq("PASS").all()
            )
            reference_ok = bool(
                reference is not None
                and len(reference) == 1
                and reference.iloc[0].get("backend") == "rf"
                and reference.iloc[0].get("construction_variant") == "raw_sum_parents_removed"
                and reference.iloc[0].get("safe_feature_mode") == "fitted_reference"
                and str(reference.iloc[0].get("status")) == "PASS"
            )
            embedding_ok = bool(
                embedding is not None
                and len(embedding) == 9
                and embedding.get("status", pd.Series(dtype=str)).astype(str).eq("PASS").all()
                and embedding.get("coordinate_hash", pd.Series(dtype=str)).astype(str).nunique() == 1
                and embedding.get("cell_id_hash", pd.Series(dtype=str)).astype(str).nunique() == 1
            )
            candidate_display_ok = bool(
                candidate_display is not None
                and len(candidate_display) == 1
                and str(candidate_display.iloc[0].get("status")) == "PASS"
                and str(candidate_display.iloc[0].get("candidate_class_sum_equals_k")).lower() == "true"
                and int(candidate_display.iloc[0].get("candidate_class_sum", -1)) == int(candidate_display.iloc[0].get("common_display_top_k", -2))
            )
            figure_ok = bool(
                figure.get("layout_rows") == 3
                and figure.get("layout_columns") == 3
                and figure.get("internal_methods") == ["DuoDose"]
                and "DuoDose-DL" not in json.dumps(figure)
                and len(figure.get("panel_order", [])) == 9
            )
            manifest_ok = _manifest(run_dir / "run_manifest.json", protocol_hash, "real_data_application", dataset, seed)
            valid = not absent and status_ok and label_ok and reference_ok and embedding_ok and candidate_display_ok and figure_ok and manifest_ok
            if valid:
                complete += 1
            elif _nonempty(run_dir / "failure.json") or (status is not None and status.get("status", pd.Series(dtype=str)).astype(str).str.lower().eq("failed").any()):
                failed += 1
                messages.append(f"{dataset}/seed_{seed}: explicit method or pipeline failure")
            else:
                missing += 1
                details = []
                if absent: details.append("missing " + ", ".join(absent[:3]))
                if not status_ok: details.append("required method status incomplete")
                if not label_ok: details.append("label-usage audit invalid")
                if not reference_ok: details.append("frozen RF reference audit invalid")
                if not embedding_ok: details.append("shared embedding audit invalid")
                if not candidate_display_ok: details.append("common-budget candidate display audit invalid")
                if not figure_ok: details.append("3x3 figure contract invalid")
                if not manifest_ok: details.append("run manifest invalid")
                messages.append(f"{dataset}/seed_{seed}: {'; '.join(details)}")
    if not _nonempty(root / "real_application_method_status_all.csv"):
        missing += 1; messages.append("combined method-status ledger is missing")
    if not _nonempty(root / "real_application_failures.csv"):
        missing += 1; messages.append("combined failure ledger is missing")
    return _result("real_data_application", root, len(datasets) * len(seeds), complete, failed, missing, messages)


def check_domain(results: Path, protocol: dict, _protocol_hash: str) -> AnalysisStatus:
    root = results / "domain_audit"
    datasets = set(protocol["datasets"]["real_doublet_enriched"])
    required = (
        "domain_audit_all_datasets_summary.csv", "domain_audit_all_datasets_failures.csv",
        "domain_audit_all_datasets_run_status.csv", "domain_audit_all_datasets_report.md",
        "domain_audit_all_datasets_auroc_comparison.png", "domain_audit_all_datasets_auroc_comparison.pdf",
        "domain_audit_all_datasets_matched_direction_adjusted.png", "domain_audit_all_datasets_matched_direction_adjusted.pdf",
        "output_manifest.json",
    )
    if not root.exists():
        return _result("domain_audit", root, len(datasets), 0, 0, len(datasets), [])
    missing_files = [name for name in required if not _nonempty(root / name)]
    status = _csv(root / "domain_audit_all_datasets_run_status.csv")
    if status is None:
        return _result("domain_audit", root, len(datasets), 0, 0, len(datasets), ["all-dataset run-status ledger is missing"])
    rows = status.loc[status.get("dataset", pd.Series(dtype=str)).astype(str).isin(datasets)].copy()
    represented = set(rows.get("dataset", pd.Series(dtype=str)).astype(str))
    combined_by_dataset = rows.set_index("dataset", drop=False) if not rows.duplicated("dataset").any() else pd.DataFrame()
    complete = failed = 0
    messages = [f"missing {name}" for name in missing_files]
    skip_statuses = {"SKIPPED_INSUFFICIENT_DATA", "INSUFFICIENT_PARENT_DISJOINT_DATA", "SKIPPED"}
    for dataset in sorted(datasets):
        validation = validate_primary_audit_files(root / dataset)
        combined_status = ""
        if not combined_by_dataset.empty and dataset in combined_by_dataset.index:
            combined_status = str(combined_by_dataset.loc[dataset].get("audit_status", "")).upper()
        if validation.completed:
            if combined_status == "COMPLETED":
                complete += 1
            else:
                failed += 1
                messages.append(f"{dataset}: valid dataset-level primary audit is not normalized to COMPLETED in combined status")
        elif validation.audit_status in skip_statuses and combined_status in skip_statuses:
            complete += 1
        else:
            failed += 1
            messages.append(f"{dataset}: {validation.audit_status}: {validation.reason}")
    missing = len(datasets - represented) + len(missing_files)
    if rows.duplicated("dataset").any():
        failed += 1; messages.append("duplicate datasets in all-dataset status ledger")
    if failed:
        messages.append(f"{failed} dataset audit(s) failed the canonical primary-audit contract")
    return _result("domain_audit", root, len(datasets), complete, failed, missing, messages)


def check_validation(results: Path, _protocol: dict, _protocol_hash: str) -> AnalysisStatus:
    root = results / "validation_suite"
    required = (
        ".validation_suite_complete.json", "validation_suite_config.json", "validation_suite_checks.csv",
        "validation_suite_manifest.json", "validation_suite_report.md", "subtype_permutation_results.csv",
        "full_label_permutation_results.csv", "domain_audit_contract_check.csv",
    )
    if not root.exists():
        return _result("validation_suite", root, 1, 0, 0, 1, [])
    missing_files = [name for name in required if not _nonempty(root / name)]
    config = _json(root / "validation_suite_config.json") or {}
    checks = _csv(root / "validation_suite_checks.csv")
    subtype = _csv(root / "subtype_permutation_results.csv")
    full = _csv(root / "full_label_permutation_results.csv")
    domain = _csv(root / "domain_audit_contract_check.csv")
    incomplete: list[str] = []
    if config.get("mode") != "full": incomplete.append("suite was not run in full mode")
    if subtype is None or len(subtype) != 100: incomplete.append("subtype control does not contain 100 permutations")
    if full is None or len(full) != 100: incomplete.append("full-label control does not contain 100 permutations")
    hard_failure = bool(checks is not None and (checks.get("required", False).astype(bool) & checks.get("status", "").astype(str).eq("FAIL")).any())
    if hard_failure: incomplete.append("required validation check failed")
    if domain is None or not domain.get("status", pd.Series(dtype=str)).astype(str).eq("PASS").all(): incomplete.append("strict domain-audit contract did not pass")
    complete = int(not missing_files and not incomplete)
    return _result("validation_suite", root, 1, complete, int(hard_failure), int(bool(missing_files or (incomplete and not hard_failure))), [*missing_files, *incomplete])


def check_runtime(results: Path, protocol: dict, _protocol_hash: str) -> AnalysisStatus:
    root = results / "runtime"
    required = ("runtime_scaling_by_run.csv", "runtime_scaling_summary.csv", "runtime_method_completeness_audit.csv", "run_manifest.json", "output_manifest.json")
    if not root.exists():
        return _result("runtime_benchmark", root, 1, 0, 0, 1, [])
    missing_files = [name for name in required if not _nonempty(root / name)]
    frame, manifest = _csv(root / "runtime_scaling_by_run.csv"), _json(root / "run_manifest.json") or {}
    audit = _csv(root / "runtime_method_completeness_audit.csv")
    problems: list[str] = []
    methods, repetitions = set(protocol["runtime"]["methods"]), int(protocol["runtime"]["repetitions"])
    counts = {int(value) for value in manifest.get("cell_counts", []) if str(value).isdigit()}
    expected_rows = len(counts) * len(methods) * repetitions
    if frame is None:
        problems.append("runtime by-run table is unreadable")
    else:
        if not set(frame.get("method", pd.Series(dtype=str)).astype(str)).issubset(methods): problems.append("runtime rows contain methods outside the frozen protocol")
        if not frame.get("status", pd.Series(dtype=str)).astype(str).str.lower().eq("success").all(): problems.append("one or more measured runtime rows failed")
        required_columns = {"data_loading_seconds", "safe_feature_construction_seconds", "model_training_seconds", "prediction_seconds", "total_wall_clock_seconds", "peak_ram_bytes", "peak_gpu_memory_bytes"}
        if not required_columns.issubset(frame.columns): problems.append("runtime component or memory columns are missing")
    if manifest.get("n_jobs_requested") is None or manifest.get("thread_configuration") is None: problems.append("thread configuration is not recorded")
    if audit is None:
        problems.append("runtime method-completeness audit is unreadable")
    else:
        if list(audit.get("method", pd.Series(dtype=str)).astype(str)) != list(CANONICAL_RUNTIME_METHODS): problems.append("runtime audit method order/scope is invalid")
        valid_status = audit.get("status", pd.Series(dtype=str)).astype(str).isin({"COMPLETED", "FAILED", "UNAVAILABLE", "INCOMPLETE", "NOT_RUN"})
        accounted = audit.get("plotted", False).astype(bool) | audit.get("omission_reason", "").astype(str).str.strip().ne("")
        if not valid_status.all() or not accounted.all(): problems.append("one or more formal runtime methods are silently unaccounted for")
        completed = audit.loc[audit["status"].astype(str).eq("COMPLETED")]
        if not completed.get("plotted", False).astype(bool).all(): problems.append("a completed runtime method is not plotted")
    complete = int(not missing_files and not problems)
    return _result("runtime_benchmark", root, 1, complete, 0, int(bool(missing_files or problems)), [*missing_files, *problems])


def check_figure_style(results: Path, _protocol: dict, _protocol_hash: str) -> AnalysisStatus:
    root = results
    audit = _csv(root / "figure_style_contract_audit.csv")
    problems: list[str] = []
    if audit is None:
        problems.append("figure_style_contract_audit.csv is missing or unreadable")
    elif not audit.get("contract_status", pd.Series(dtype=str)).astype(str).eq("PASS").all():
        problems.append("one or more formal plotting entry points fail the Arial style contract")
    complete = int(not problems)
    return _result("figure_style_contract", root, 1, complete, 0, int(bool(problems)), problems)


def check_sensitivity(results: Path, protocol: dict, _protocol_hash: str) -> AnalysisStatus:
    root = results / "sensitivity"
    required = ("parameter_sensitivity_by_run.csv", "parameter_sensitivity_summary.csv", "run_manifest.json", "output_manifest.json")
    if not root.exists():
        return _result("parameter_sensitivity", root, 1, 0, 0, 1, [])
    missing_files = [name for name in required if not _nonempty(root / name)]
    frame = _csv(root / "parameter_sensitivity_by_run.csv")
    expected = len(protocol["seeds"]["parameter_sensitivity"]) * len(protocol["parameter_sensitivity"]["semi_real_size_factors"]) * len(protocol["parameter_sensitivity"]["expected_doublet_rates"])
    problems: list[str] = []
    if frame is None or len(frame) != expected: problems.append(f"sensitivity grid does not contain {expected} rows")
    elif set(frame.get("method", pd.Series(dtype=str)).astype(str)) != {"DuoDose"}: problems.append("sensitivity model is not frozen RF DuoDose")
    complete = int(not missing_files and not problems)
    return _result("parameter_sensitivity", root, 1, complete, 0, int(bool(missing_files or problems)), [*missing_files, *problems])


def _artifacts(results: Path, analysis: str, names: tuple[str, ...], subdir: str) -> AnalysisStatus:
    root = results / subdir
    missing = [name for name in names if not _nonempty(root / name)]
    status = NOT_RUN if not root.exists() else INCOMPLETE if missing else COMPLETE
    return AnalysisStatus(analysis, status, len(names) - len(missing), len(names), 0, len(missing), "; ".join(f"missing {name}" for name in missing), str(root))


def check_tables(results: Path, _protocol: dict, _protocol_hash: str) -> AnalysisStatus:
    names = (
        "final_controlled_comparison_by_run.csv", "final_controlled_comparison_summary.csv",
        "final_real_application_method_status.csv", "final_real_application_candidate_summary.csv",
        "domain_audit_all_datasets_summary.csv", "domain_audit_all_datasets_failures.csv",
        "runtime_scaling_summary.csv", "parameter_sensitivity_summary.csv", "final_method_status_ledger.csv",
        "final_analysis_status_ledger.csv", "final_table_manifest.json", "../reports/final_validation_report.md", "../final_artifact_manifest.json",
    )
    return _artifacts(results, "final_tables", names, "tables")


def check_figures(results: Path, _protocol: dict, _protocol_hash: str) -> AnalysisStatus:
    stems = ("final_controlled_overall_auprc", "final_real_application_cross_method_umap", "final_runtime_scaling", "final_parameter_sensitivity", "domain_audit_all_datasets_auroc_comparison", "domain_audit_all_datasets_matched_direction_adjusted")
    names = tuple(f"{stem}.{suffix}" for stem in stems for suffix in ("png", "pdf")) + ("final_figure_manifest.json",)
    return _artifacts(results, "final_figures", names, "figures")


CHECKS: dict[str, Callable[[Path, dict, str], AnalysisStatus]] = {
    "controlled_benchmark": check_controlled,
    "external_baselines": check_external,
    "real_data_application": check_real_application,
    "domain_audit": check_domain,
    "validation_suite": check_validation,
    "runtime_benchmark": check_runtime,
    "figure_style_contract": check_figure_style,
    "parameter_sensitivity": check_sensitivity,
    "final_tables": check_tables,
    "final_figures": check_figures,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/final_v1")
    parser.add_argument("--protocol", default="reproducibility/configs/final_protocol.yaml")
    parser.add_argument("--stage", choices=tuple(CHECKS), default=None)
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero unless every selected analysis is COMPLETE.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = Path(args.results_dir).resolve()
    protocol = load_final_protocol(args.protocol)
    protocol_hash = _sha256(Path(protocol["_protocol_path"]))
    selected = [args.stage] if args.stage else list(CHECKS)
    rows = [CHECKS[name](results, protocol, protocol_hash) for name in selected]
    overall = COMPLETE if all(row.status == COMPLETE for row in rows) else FAILED if any(row.status == FAILED for row in rows) else INCOMPLETE
    if args.status_only:
        print(rows[0].status if len(rows) == 1 else overall)
    else:
        print(pd.DataFrame([asdict(row) for row in rows]).to_string(index=False))
    payload = {
        "schema_version": 1,
        "code_completion_status": "NOT_EVALUATED_SEPARATELY",
        "formal_result_completion_status": overall,
        "protocol": str(Path(args.protocol)),
        "results_dir": str(Path(args.results_dir)),
        "analyses": [asdict(row) for row in rows],
    }
    if not args.no_write:
        results.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(row) for row in rows]).to_csv(results / "formal_completion_status.csv", index=False)
        (results / "formal_completion_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.strict and overall != COMPLETE:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
