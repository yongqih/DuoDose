"""Run the complete DuoDose validation and audit suite with one command."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from duodose.net import probabilities_to_scores  # noqa: E402
from duodose.progress import ProgressReporter, ProgressSettings, add_progress_arguments, migrate_progress_artifacts, progress_paths  # noqa: E402
from duodose.protocol import load_final_protocol  # noqa: E402
from duodose.plotting_style import apply_manuscript_style  # noqa: E402
from duodose.validation_suite import (  # noqa: E402
    HARD_AUDITS,
    REQUIRED_OUTPUTS,
    SuiteCheck,
    ValidationSuiteError,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    audit_chunking_invariance,
    audit_frozen_reference,
    audit_model_serialization,
    audit_order_invariance,
    audit_parent_disjoint,
    audit_run_status,
    audit_same_cell_features,
    audit_transformer_serialization,
    canonical_json_hash,
    deterministic_subset,
    domain_audit_contract_check,
    load_validation_config,
    make_fixture_adata,
    metric_contract_table,
    protocol_override_for_mode,
    schema_audit,
    stale_temporary_outputs,
    train_rf_with_fit_labels,
    validate_feature_names,
    validate_probability_frame,
    verify_required_outputs,
    compare_frames,
    completed_run_is_reusable,
)
from reproducibility.lib.common import (  # noqa: E402
    LoadedDataset,
    controlled_metric_row,
    environment_record,
    evaluate_internal_controlled,
    load_dataset_exact,
    run_protocol_models,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="reproducibility/configs/final_validation_suite.yaml")
    parser.add_argument("--data-dir", default="data/xili_real/real_datasets")
    parser.add_argument("--output-dir", default="results/final_v1/validation_suite")
    parser.add_argument("--existing-domain-audit-dir", default=None, help="Read-only path to an already completed strict domain audit")
    parser.add_argument("--formal-results-dir", default=None, help="Formal result root inspected by the run-status audit.")
    parser.add_argument("--mode", choices=["smoke", "quick", "full"], default="quick")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--n-subtype-permutations", type=int, default=None)
    parser.add_argument("--n-full-label-permutations", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--n-jobs", type=int, default=1)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--overwrite", action="store_true")
    group.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--refresh-schema-report-only",
        action="store_true",
        help="Migrate progress paths and refresh only the existing schema contract and report.",
    )
    parser.add_argument(
        "--refresh-run-status-report-only",
        action="store_true",
        help="Refresh only the formal run-status contract and existing validation report.",
    )
    add_progress_arguments(parser)
    return parser


def _resolve_repo_path(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _relative_public_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def _implementation_hash() -> str:
    paths = [
        Path(__file__).resolve(),
        SRC / "duodose" / "validation_suite.py",
        SRC / "duodose" / "safe_feature_transformer.py",
        SRC / "duodose" / "net.py",
        ROOT / "reproducibility" / "lib" / "common.py",
    ]
    payload = {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return canonical_json_hash(payload)


def _absolute_path_matches(output: Path) -> list[str]:
    pattern = re.compile(r"[A-Za-z]:[\\/]")
    matches: list[str] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".md"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if pattern.search(text):
            matches.append(str(path.relative_to(output)).replace("\\", "/"))
    return matches


def _existing_report_runtime(report_path: Path) -> float:
    if not report_path.is_file():
        return 0.0
    match = re.search(r"^- Runtime: ([0-9.]+) seconds$", report_path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    return float(match.group(1)) if match else 0.0


def _refresh_schema_report_only(output: Path) -> None:
    """Refresh path portability status from existing artifacts only."""

    required = (
        "schema_audit.csv",
        "validation_suite_checks.csv",
        "validation_suite_config.json",
        "parent_disjoint_audit.csv",
        "subtype_permutation_summary.csv",
        "full_label_permutation_summary.csv",
        "run_status_audit.csv",
        "domain_audit_contract_check.csv",
        "validation_suite_report.md",
    )
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise ValidationSuiteError("schema/report-only refresh requires existing outputs: " + ", ".join(missing))

    results_dir = output.parent
    migrate_progress_artifacts(
        ledger_path=output / "runtime_ledger.csv",
        snapshot_path=output / "formal_progress.json",
        results_dir=results_dir,
    )
    absolute_paths = _absolute_path_matches(output)

    schema = pd.read_csv(output / "schema_audit.csv")
    schema = schema.loc[schema["check"].astype(str).ne("no_local_absolute_paths_in_public_outputs")].copy()
    schema = pd.concat(
        [
            schema,
            pd.DataFrame(
                [
                    {
                        "check": "no_local_absolute_paths_in_public_outputs",
                        "value": len(absolute_paths),
                        "status": "PASS" if not absolute_paths else "FAIL",
                        "message": "" if not absolute_paths else ", ".join(absolute_paths),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    atomic_write_csv(output / "schema_audit.csv", schema)

    checks = pd.read_csv(output / "validation_suite_checks.csv")
    checks = checks.loc[checks["audit"].astype(str).ne("schema_contract")].copy()
    refreshed = [
        _aggregate_frame_check("schema_contract", schema, required=True).as_dict(),
        _check(
            "schema_contract",
            "no_local_absolute_paths_in_public_outputs",
            not absolute_paths,
            required=True,
            value=len(absolute_paths),
            message="" if not absolute_paths else ", ".join(absolute_paths),
        ).as_dict(),
    ]
    checks = pd.concat([checks, pd.DataFrame(refreshed)], ignore_index=True)
    required_mask = checks["required"].astype(str).str.lower().isin({"true", "1"})
    hard_failure_count = int((required_mask & checks["status"].astype(str).eq("FAIL")).sum())
    code_completion_status = "COMPLETE" if hard_failure_count == 0 else "INCOMPLETE"

    run_status = pd.read_csv(output / "run_status_audit.csv")
    formal_analysis_status = "COMPLETE" if run_status["status"].astype(str).eq("COMPLETED").all() else "INCOMPLETE"
    config = json.loads((output / "validation_suite_config.json").read_text(encoding="utf-8"))
    metadata = {
        "mode": config.get("mode", "unknown"),
        "dataset": config.get("dataset", "unknown"),
        "config_hash": config.get("config_hash", "unknown"),
        "runtime_seconds": _existing_report_runtime(output / "validation_suite_report.md"),
        "code_completion_status": code_completion_status,
        "formal_analysis_status": formal_analysis_status,
    }
    summary = checks.groupby("status", sort=False).size().rename("count").reset_index()
    summary["code_completion_status"] = code_completion_status
    summary["formal_analysis_status"] = formal_analysis_status
    parent = pd.read_csv(output / "parent_disjoint_audit.csv")
    subtype = pd.read_csv(output / "subtype_permutation_summary.csv")
    full = pd.read_csv(output / "full_label_permutation_summary.csv")
    domain = pd.read_csv(output / "domain_audit_contract_check.csv")
    atomic_write_csv(output / "validation_suite_checks.csv", checks)
    atomic_write_csv(output / "validation_suite_summary.csv", summary)
    atomic_write_text(output / "validation_suite_report.md", _report(output, metadata, checks, parent, subtype, full, run_status, domain))

    manifest_path = output / "validation_suite_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {"schema_version": 1}
    manifest["code_completion_status"] = code_completion_status
    manifest["formal_analysis_status"] = formal_analysis_status
    manifest["files"] = [
        {"path": path.relative_to(output).as_posix(), "size_bytes": path.stat().st_size}
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in {"validation_suite_manifest.json", ".validation_suite_complete.json"} and not path.name.endswith(".tmp")
    ]
    atomic_write_json(manifest_path, manifest)

    if absolute_paths:
        raise ValidationSuiteError("local absolute paths remain after migration: " + ", ".join(absolute_paths))


def _refresh_run_status_report_only(output: Path, formal_results_dir: Path, protocol: dict[str, Any]) -> None:
    """Refresh the formal run-status contract from completed artifacts only."""

    required = (
        "validation_suite_checks.csv",
        "validation_suite_config.json",
        "parent_disjoint_audit.csv",
        "subtype_permutation_summary.csv",
        "full_label_permutation_summary.csv",
        "domain_audit_contract_check.csv",
        "validation_suite_report.md",
    )
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise ValidationSuiteError("run-status/report-only refresh requires existing outputs: " + ", ".join(missing))

    run_status = audit_run_status(formal_results_dir, protocol)
    expected_controlled = len(protocol["datasets"]["real_doublet_enriched"]) * len(protocol["seeds"]["controlled_benchmark"]) * 6
    expected_application = len(protocol["datasets"]["real_application"]) * len(protocol["seeds"]["real_application"]) * 5
    expected_rows = expected_controlled + expected_application
    duplicate = run_status.duplicated(["workflow", "dataset", "seed", "method"]).any()
    incomplete_count = int(run_status["status"].astype(str).ne("COMPLETED").sum())

    checks = pd.read_csv(output / "validation_suite_checks.csv")
    checks = checks.loc[checks["audit"].astype(str).ne("run_status_contract")].copy()
    refreshed = [
        _check(
            "run_status_contract",
            "complete_requested_grid",
            len(run_status) == expected_rows and not duplicate,
            required=False,
            value=len(run_status),
            message=f"expected {expected_controlled} controlled and {expected_application} real-application rows",
        ).as_dict(),
        SuiteCheck(
            "run_status_contract",
            "formal_run_completeness",
            "PASS" if incomplete_count == 0 else "INCOMPLETE",
            False,
            "current controlled-benchmark and real-data-application formal units",
            incomplete_count,
        ).as_dict(),
    ]
    checks = pd.concat([checks, pd.DataFrame(refreshed)], ignore_index=True)
    required_mask = checks["required"].astype(str).str.lower().isin({"true", "1"})
    hard_failure_count = int((required_mask & checks["status"].astype(str).eq("FAIL")).sum())
    code_completion_status = "COMPLETE" if hard_failure_count == 0 else "INCOMPLETE"
    formal_analysis_status = "COMPLETE" if incomplete_count == 0 else "INCOMPLETE"

    config = json.loads((output / "validation_suite_config.json").read_text(encoding="utf-8"))
    metadata = {
        "mode": config.get("mode", "unknown"),
        "dataset": config.get("dataset", "unknown"),
        "config_hash": config.get("config_hash", "unknown"),
        "runtime_seconds": _existing_report_runtime(output / "validation_suite_report.md"),
        "code_completion_status": code_completion_status,
        "formal_analysis_status": formal_analysis_status,
    }
    summary = checks.groupby("status", sort=False).size().rename("count").reset_index()
    summary["code_completion_status"] = code_completion_status
    summary["formal_analysis_status"] = formal_analysis_status
    parent = pd.read_csv(output / "parent_disjoint_audit.csv")
    subtype = pd.read_csv(output / "subtype_permutation_summary.csv")
    full = pd.read_csv(output / "full_label_permutation_summary.csv")
    domain = pd.read_csv(output / "domain_audit_contract_check.csv")

    atomic_write_csv(output / "run_status_audit.csv", run_status)
    atomic_write_csv(output / "validation_suite_checks.csv", checks)
    atomic_write_csv(output / "validation_suite_summary.csv", summary)
    atomic_write_text(output / "validation_suite_report.md", _report(output, metadata, checks, parent, subtype, full, run_status, domain))

    manifest_path = output / "validation_suite_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {"schema_version": 1}
    manifest["code_completion_status"] = code_completion_status
    manifest["formal_analysis_status"] = formal_analysis_status
    manifest["files"] = [
        {"path": path.relative_to(output).as_posix(), "size_bytes": path.stat().st_size}
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in {"validation_suite_manifest.json", ".validation_suite_complete.json"} and not path.name.endswith(".tmp")
    ]
    atomic_write_json(manifest_path, manifest)


def _validate_cli(args: argparse.Namespace, config: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be at least 1")
    mode = dict(config["modes"][args.mode])
    if args.max_cells is not None:
        if args.max_cells < 240:
            raise ValueError("--max-cells must be at least 240")
        mode["max_cells"] = int(args.max_cells)
    subtype = int(args.n_subtype_permutations if args.n_subtype_permutations is not None else mode["subtype_permutations"])
    full = int(args.n_full_label_permutations if args.n_full_label_permutations is not None else mode["full_label_permutations"])
    if subtype < 1 or full < 1:
        raise ValueError("permutation counts must be positive")
    if args.mode == "smoke" and (subtype > 2 or full > 2):
        raise ValueError("smoke mode permits at most 2 permutations per control")
    if args.mode == "quick" and (subtype > 5 or full > 5):
        raise ValueError("quick mode permits at most 5 permutations per control")
    return mode, subtype, full


def _check(audit: str, check: str, passed: bool, *, required: bool | None = None, message: str = "", value: object = "", maximum: float = float("nan")) -> SuiteCheck:
    return SuiteCheck(audit, check, "PASS" if passed else "FAIL", audit in HARD_AUDITS if required is None else required, message, value, maximum)


def _aggregate_frame_check(audit: str, frame: pd.DataFrame, *, status_column: str = "status", required: bool | None = None) -> SuiteCheck:
    statuses = frame[status_column].astype(str) if status_column in frame else pd.Series(["FAIL"])
    passed = not statuses.eq("FAIL").any()
    maximum = float(pd.to_numeric(frame.get("maximum_absolute_difference", pd.Series(dtype=float)), errors="coerce").max()) if "maximum_absolute_difference" in frame else float("nan")
    return _check(audit, "aggregate", passed, required=required, message="" if passed else "one or more audit rows failed", value=int(len(frame)), maximum=maximum)


def _metric_from_probabilities(run: Any, probabilities: pd.DataFrame) -> dict[str, Any]:
    overall, homotypic, heterotypic = probabilities_to_scores(probabilities)
    return controlled_metric_row(
        dataset=run.dataset,
        seed=run.seed,
        method="DuoDose",
        labels=run.test_scores["true_label"].astype(str),
        obs=run.bundle.test_adata.obs,
        overall_score=overall,
        homotypic_score=homotypic,
        heterotypic_score=heterotypic,
    )


PERMUTATION_METRIC_DIRECTIONS = {
    "AUROC": "higher_is_better",
    "overall_AUPRC": "higher_is_better",
    "homotypic_AUPRC": "higher_is_better",
    "heterotypic_AUPRC": "higher_is_better",
    "macro_subtype_AUPRC": "higher_is_better",
    "homotypic_vs_high_RNA_singlet_AUPRC": "higher_is_better",
    "high_RNA_singlet_FPR": "lower_is_better",
}


def _permutation_summary(results: pd.DataFrame, observed: dict[str, Any], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        if metric not in PERMUTATION_METRIC_DIRECTIONS:
            raise ValidationSuiteError(f"permutation metric direction is not declared: {metric}")
        direction = PERMUTATION_METRIC_DIRECTIONS[metric]
        null = pd.to_numeric(results[metric], errors="coerce").dropna().to_numpy(dtype=float)
        value = float(observed.get(metric, np.nan))
        if len(null) and np.isfinite(value):
            tail_count = np.sum(null >= value) if direction == "higher_is_better" else np.sum(null <= value)
            empirical_p = float((1 + tail_count) / (1 + len(null)))
            percentile = float(100.0 * np.mean(null <= value))
        else:
            empirical_p = float("nan")
            percentile = float("nan")
        execution_status = "COMPLETED" if len(null) == len(results) and len(null) > 0 else "INCOMPLETE"
        contract_status = "PASS" if not results.empty and results.get("contract_status", pd.Series("FAIL", index=results.index)).astype(str).eq("PASS").all() else "FAIL"
        expected_side = bool(value > np.mean(null)) if direction == "higher_is_better" and len(null) else bool(value < np.mean(null)) if len(null) else False
        if execution_status != "COMPLETED" or contract_status != "PASS":
            evidence_status = "NOT_APPLICABLE"
            evidence_reason = "permutation execution or contract is incomplete"
        elif len(null) < 20:
            evidence_status = "INCONCLUSIVE"
            evidence_reason = "fewer than 20 permutations; smoke/quick output tests implementation, not formal evidence"
        elif expected_side and empirical_p <= 0.05:
            evidence_status = "SUPPORTED"
            evidence_reason = f"observed metric is separated from the null in the {direction} direction"
        else:
            evidence_status = "NOT_SUPPORTED"
            evidence_reason = "observed metric is not significantly separated from the null in the expected direction"
        rows.append(
            {
                "metric": metric,
                "metric_direction": direction,
                "observed_value": value,
                "n_permutations": int(len(null)),
                "null_mean": float(np.mean(null)) if len(null) else float("nan"),
                "null_standard_deviation": float(np.std(null, ddof=1)) if len(null) > 1 else 0.0 if len(null) == 1 else float("nan"),
                "null_median": float(np.median(null)) if len(null) else float("nan"),
                "null_025_quantile": float(np.quantile(null, 0.025)) if len(null) else float("nan"),
                "null_975_quantile": float(np.quantile(null, 0.975)) if len(null) else float("nan"),
                "empirical_p_value": empirical_p,
                "observed_percentile": percentile,
                "execution_status": execution_status,
                "contract_status": contract_status,
                "evidence_status": evidence_status,
                "evidence_reason": evidence_reason,
            }
        )
    return pd.DataFrame(rows)


def _run_permutations(
    run: Any,
    n_subtype: int,
    n_full: int,
    *,
    settings: ProgressSettings,
    ledger_path: Path,
    snapshot_path: Path,
    config_hash: str,
    output_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed = evaluate_internal_controlled(run).set_index("method").loc["DuoDose"].to_dict()
    original = run.fit_scores["true_label"].astype(str)
    subtype_rows = []
    doublet_mask = original.isin(["homotypic_doublet", "heterotypic_doublet"])
    subtype_progress = ProgressReporter(stage="validation_subtype_label_permutations", total_units=n_subtype, settings=settings, ledger_path=ledger_path, snapshot_path=snapshot_path, config_hash=config_hash, output_path=output_path)
    try:
        for permutation in range(n_subtype):
            unit_started = subtype_progress.start_unit(dataset=run.dataset, seed=run.seed, method="subtype_label_permutation", output_path=output_path, prefix=f"subtype-label permutation {permutation + 1}/{n_subtype}")
            try:
                rng = np.random.default_rng(10_000 + int(run.seed) * 1_000 + permutation)
                permuted = original.copy()
                permuted.loc[doublet_mask] = rng.permutation(original.loc[doublet_mask].to_numpy())
                _, probabilities, _ = train_rf_with_fit_labels(run, permuted)
                row = _metric_from_probabilities(run, probabilities)
                counts_preserved = original.value_counts().sort_index().equals(permuted.value_counts().sort_index())
                binary_unchanged = original.isin(["homotypic_doublet", "heterotypic_doublet"]).equals(permuted.isin(["homotypic_doublet", "heterotypic_doublet"]))
                contract_ok = counts_preserved and binary_unchanged and original.loc[~doublet_mask].equals(permuted.loc[~doublet_mask])
                row.update(permutation=permutation, permutation_seed=10_000 + int(run.seed) * 1_000 + permutation, strategy="subtype labels permuted among fit-split doublets", model="DuoDose", n_changed=int((permuted != original).sum()), class_counts_preserved=counts_preserved, overall_binary_target_unchanged=binary_unchanged, fit_membership_preserved=permuted.index.equals(original.index), validation_test_labels_unchanged=True, execution_status="COMPLETED", contract_status="PASS" if contract_ok else "FAIL", evidence_status="NOT_APPLICABLE")
                subtype_rows.append(row)
                subtype_progress.complete_unit(unit_started, message=f"subtype permutation {permutation + 1}: {'PASS' if contract_ok else 'FAIL'}")
            except Exception as exc:
                subtype_progress.fail_unit(unit_started, exc)
                raise
    finally:
        subtype_progress.close()
    subtype_results = pd.DataFrame(subtype_rows)
    subtype_summary = _permutation_summary(subtype_results, observed, ["homotypic_AUPRC", "heterotypic_AUPRC", "macro_subtype_AUPRC"])

    full_rows = []
    full_progress = ProgressReporter(stage="validation_full_label_permutations", total_units=n_full, settings=settings, ledger_path=ledger_path, snapshot_path=snapshot_path, config_hash=config_hash, output_path=output_path)
    try:
        for permutation in range(n_full):
            unit_started = full_progress.start_unit(dataset=run.dataset, seed=run.seed, method="full_label_permutation", output_path=output_path, prefix=f"full-label permutation {permutation + 1}/{n_full}")
            try:
                rng = np.random.default_rng(20_000 + int(run.seed) * 1_000 + permutation)
                permuted = pd.Series(rng.permutation(original.to_numpy()), index=original.index)
                _, probabilities, _ = train_rf_with_fit_labels(run, permuted)
                row = _metric_from_probabilities(run, probabilities)
                counts_preserved = original.value_counts().sort_index().equals(permuted.value_counts().sort_index())
                contract_ok = counts_preserved and permuted.index.equals(original.index)
                row.update(permutation=permutation, permutation_seed=20_000 + int(run.seed) * 1_000 + permutation, strategy="complete multiclass labels permuted across fit-split rows", model="DuoDose", n_changed=int((permuted != original).sum()), class_counts_preserved=counts_preserved, fit_membership_preserved=permuted.index.equals(original.index), validation_test_labels_unchanged=True, execution_status="COMPLETED", contract_status="PASS" if contract_ok else "FAIL", evidence_status="NOT_APPLICABLE")
                full_rows.append(row)
                full_progress.complete_unit(unit_started, message=f"full-label permutation {permutation + 1}: {'PASS' if contract_ok else 'FAIL'}")
            except Exception as exc:
                full_progress.fail_unit(unit_started, exc)
                raise
    finally:
        full_progress.close()
    full_results = pd.DataFrame(full_rows)
    full_summary = _permutation_summary(
        full_results,
        observed,
        ["AUROC", "overall_AUPRC", "homotypic_AUPRC", "heterotypic_AUPRC", "macro_subtype_AUPRC", "homotypic_vs_high_RNA_singlet_AUPRC", "high_RNA_singlet_FPR"],
    )
    return subtype_results, subtype_summary, full_results, full_summary


def _deterministic_rerun(loaded: LoadedDataset, run: Any, protocol_override: dict[str, Any], args: argparse.Namespace, tolerance: float) -> pd.DataFrame:
    rerun = run_protocol_models(
        loaded,
        protocol_path=run.protocol.get("_protocol_path"),
        protocol_override=protocol_override,
        seed=int(run.seed),
        backends=("rf",),
        device=args.device,
        amp=False,
        dl_max_epochs=1,
        dl_patience=1,
    )
    rows = []
    plan_columns = ["synthetic_cell_id", "split", "synthetic_subtype", "parent_1_id", "parent_2_id"]
    left_plan = run.bundle.parent_map.loc[:, plan_columns].sort_values("synthetic_cell_id").reset_index(drop=True)
    right_plan = rerun.bundle.parent_map.loc[:, plan_columns].sort_values("synthetic_cell_id").reset_index(drop=True)
    rows.append({"context": "parent_map", "maximum_absolute_difference": 0.0 if left_plan.equals(right_plan) else float("inf"), "status": "PASS" if left_plan.equals(right_plan) else "FAIL", "message": ""})
    for split, left_ids, right_ids in (
        ("fit", run.fit_scores.index, rerun.fit_scores.index),
        ("validation", run.validation_scores.index, rerun.validation_scores.index),
        ("test", run.test_scores.index, rerun.test_scores.index),
    ):
        same = list(map(str, left_ids)) == list(map(str, right_ids))
        rows.append({"context": f"{split}_membership", "maximum_absolute_difference": 0.0 if same else float("inf"), "status": "PASS" if same else "FAIL", "message": ""})
    feature_comparison = compare_frames(run.test_features, rerun.test_features, audit="deterministic_rf_rerun", context="raw_safe_features", tolerance=tolerance)
    rows.extend(feature_comparison.rename(columns={"context": "detail"}).assign(context=lambda frame: frame["detail"] + ":" + frame["feature_or_output"])[["context", "maximum_absolute_difference", "status"]].assign(message="").to_dict("records"))
    probability_comparison = compare_frames(run.method_probabilities_test["DuoDose"], rerun.method_probabilities_test["DuoDose"], audit="deterministic_rf_rerun", context="rf_probabilities", tolerance=tolerance)
    rows.extend(probability_comparison.assign(context=lambda frame: "rf_probabilities:" + frame["feature_or_output"])[["context", "maximum_absolute_difference", "status"]].assign(message="").to_dict("records"))
    left_metrics = evaluate_internal_controlled(run).set_index("method").loc["DuoDose"]
    right_metrics = evaluate_internal_controlled(rerun).set_index("method").loc["DuoDose"]
    for metric in ["AUROC", "overall_AUPRC", "homotypic_AUPRC", "heterotypic_AUPRC", "macro_subtype_AUPRC"]:
        difference = abs(float(left_metrics[metric]) - float(right_metrics[metric]))
        rows.append({"context": f"metric:{metric}", "maximum_absolute_difference": difference, "status": "PASS" if difference < tolerance else "FAIL", "message": ""})
    return pd.DataFrame(rows)


def _plot_outputs(output: Path, invariance: pd.DataFrame, parent: pd.DataFrame, subtype_results: pd.DataFrame, subtype_summary: pd.DataFrame, full_results: pd.DataFrame, full_summary: pd.DataFrame, probabilities: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    apply_manuscript_style()

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    grouped = invariance.groupby("audit", sort=False)["maximum_absolute_difference"].max().replace([np.inf, -np.inf], np.nan).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(grouped.index, grouped.values, color="#38686A")
    ax.axhline(1e-6, color="#A23B3B", linestyle="--")
    ax.set_ylabel("Maximum absolute error")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"maximum_invariance_error.{suffix}", dpi=180 if suffix == "png" else None)
    plt.close(fig)

    overlap = parent.loc[parent["check"].str.contains("overlap|duplicated|remain|reference", case=False, regex=True)].copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(overlap["check"], pd.to_numeric(overlap["value"], errors="coerce").fillna(0), color="#5C946E")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"parent_overlap_summary.{suffix}", dpi=180 if suffix == "png" else None)
    plt.close(fig)

    metric = "macro_subtype_AUPRC"
    observed = float(subtype_summary.set_index("metric").loc[metric, "observed_value"])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(subtype_results[metric], bins=max(2, len(subtype_results)), color="#7A6F9B")
    ax.axvline(observed, color="#A23B3B", linewidth=2, label="Observed")
    ax.legend()
    ax.set_xlabel("Macro subtype AUPRC")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"subtype_permutation_observed_vs_null.{suffix}", dpi=180 if suffix == "png" else None)
    plt.close(fig)

    for metric_name, stem, label in (("AUROC", "full_label_permutation_auroc", "AUROC"), ("overall_AUPRC", "full_label_permutation_auprc", "Overall AUPRC")):
        observed = float(full_summary.set_index("metric").loc[metric_name, "observed_value"])
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.hist(full_results[metric_name], bins=max(2, len(full_results)), color="#4C78A8")
        ax.axvline(observed, color="#A23B3B", linewidth=2, label="Observed")
        ax.legend()
        ax.set_xlabel(label)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(figures / f"{stem}.{suffix}", dpi=180 if suffix == "png" else None)
        plt.close(fig)

    counts = probabilities["status"].value_counts().reindex(["PASS", "FAIL"], fill_value=0)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(counts.index, counts.values, color=["#38686A", "#A23B3B"])
    ax.set_ylabel("Probability-contract checks")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"probability_contract_status.{suffix}", dpi=180 if suffix == "png" else None)
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    display = frame.copy().replace([np.inf, -np.inf], ["inf", "-inf"]).fillna("")
    columns = [str(column) for column in display.columns]
    def clean(value: object) -> str:
        if isinstance(value, float):
            value = f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in display.itertuples(index=False, name=None))
    return "\n".join(lines)


def _report(output: Path, metadata: dict[str, Any], checks: pd.DataFrame, parent: pd.DataFrame, subtype_summary: pd.DataFrame, full_summary: pd.DataFrame, run_status: pd.DataFrame, domain: pd.DataFrame) -> str:
    counts = checks["status"].value_counts()
    lines = [
        "# DuoDose validation suite report",
        "",
        f"- Mode: `{metadata['mode']}`",
        f"- Dataset: `{metadata['dataset']}`",
        f"- Configuration hash: `{metadata['config_hash']}`",
        f"- Runtime: {metadata['runtime_seconds']:.2f} seconds",
        f"- Code completion status: `{metadata['code_completion_status']}`",
        f"- Formal analysis status: `{metadata['formal_analysis_status']}`",
        "",
        "## Status summary",
        "",
    ]
    for status in ("PASS", "FAIL", "NOT_APPLICABLE", "NOT_RUN", "INCOMPLETE"):
        lines.append(f"- {status}: {int(counts.get(status, 0))}")
    lines += ["", "## Parent/reference contract", "", _markdown_table(parent), "", "## Subtype permutation sanity control", "", _markdown_table(subtype_summary), "", "## Full-label permutation sanity control", "", _markdown_table(full_summary), "", "## Formal run-status contract", "", f"Incomplete current formal method rows: {int(run_status['status'].ne('COMPLETED').sum())}", "", "## Existing domain audit", "", _markdown_table(domain), "", "Smoke and quick permutation distributions are implementation sanity checks, not formal statistical evidence. The real-versus-semi-real domain audit is not rerun and contains no permutation workflow."]
    return "\n".join(lines) + "\n"


def _print_summary(output: Path, runtime: float, checks: pd.DataFrame, parent: pd.DataFrame, subtype_summary: pd.DataFrame, full_summary: pd.DataFrame) -> None:
    counts = checks["status"].value_counts()
    print(f"output directory: {_relative_public_path(output)}")
    print(f"total runtime: {runtime:.2f} seconds")
    for status in ("PASS", "FAIL", "NOT_APPLICABLE", "NOT_RUN", "INCOMPLETE"):
        print(f"{status}: {int(counts.get(status, 0))}")
    for audit in ("same_cell_feature_invariance", "cell_order_invariance", "chunking_invariance", "transformer_serialization", "rf_serialization", "dl_serialization", "deterministic_rf_rerun"):
        match = checks.loc[checks["audit"].eq(audit), "maximum_absolute_error"]
        print(f"maximum error {audit}: {float(pd.to_numeric(match, errors='coerce').max()) if len(match) else float('nan'):.3g}")
    for check in ("train_validation_parent_overlap", "train_test_parent_overlap", "validation_test_parent_overlap", "reference_parent_overlap"):
        match = parent.loc[parent["check"].eq(check), "value"]
        print(f"{check}: {int(match.iloc[0]) if len(match) else -1}")
    print(f"subtype permutation execution: {subtype_summary['execution_status'].iloc[0]}; contract: {subtype_summary['contract_status'].iloc[0]}; evidence: {','.join(sorted(subtype_summary['evidence_status'].unique()))}")
    print(f"full-label permutation execution: {full_summary['execution_status'].iloc[0]}; contract: {full_summary['contract_status'].iloc[0]}; evidence: {','.join(sorted(full_summary['evidence_status'].unique()))}")
    print(f"report: {_relative_public_path(output / 'validation_suite_report.md')}")


def main() -> None:
    started = time.perf_counter()
    args = build_parser().parse_args()
    config_path = _resolve_repo_path(args.config)
    config = load_validation_config(config_path)
    output = _resolve_repo_path(args.output_dir)
    if args.refresh_schema_report_only and args.refresh_run_status_report_only:
        raise ValidationSuiteError("choose only one report-only refresh mode")
    if args.refresh_schema_report_only:
        if args.overwrite:
            raise ValidationSuiteError("--refresh-schema-report-only cannot be combined with --overwrite")
        _refresh_schema_report_only(output)
        checks = pd.read_csv(output / "validation_suite_checks.csv")
        match = checks.loc[
            checks["audit"].astype(str).eq("schema_contract")
            & checks["check"].astype(str).eq("no_local_absolute_paths_in_public_outputs")
        ]
        print(f"schema_contract / no_local_absolute_paths_in_public_outputs = {match.iloc[0]['status']}")
        print(f"report: {_relative_public_path(output / 'validation_suite_report.md')}")
        return
    if args.refresh_run_status_report_only:
        if args.overwrite:
            raise ValidationSuiteError("--refresh-run-status-report-only cannot be combined with --overwrite")
        protocol_path = (config_path.parent / str(config["protocol"])).resolve()
        protocol = load_final_protocol(protocol_path)
        formal_results_dir = _resolve_repo_path(args.formal_results_dir or config["run_status"]["final_results_relative_dir"])
        _refresh_run_status_report_only(output, formal_results_dir, protocol)
        refreshed = pd.read_csv(output / "run_status_audit.csv")
        print(f"formal run-status rows: {len(refreshed)}")
        print(f"incomplete current formal rows: {int(refreshed['status'].ne('COMPLETED').sum())}")
        print(f"report: {_relative_public_path(output / 'validation_suite_report.md')}")
        return
    mode_config, n_subtype, n_full = _validate_cli(args, config)
    completion_path = output / ".validation_suite_complete.json"

    if args.overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    stale = stale_temporary_outputs(output)
    if stale:
        if args.resume or args.overwrite:
            for path in stale:
                path.unlink()
        else:
            raise ValidationSuiteError("interrupted temporary outputs detected; use --resume or --overwrite: " + ", ".join(path.name for path in stale))

    override_record = {
        "config": _relative_public_path(config_path),
        "data_dir": _relative_public_path(_resolve_repo_path(args.data_dir)),
        "output_dir": _relative_public_path(output),
        "existing_domain_audit_dir": _relative_public_path(_resolve_repo_path(args.existing_domain_audit_dir or config.get("existing_domain_audit_dir"))) if (args.existing_domain_audit_dir or config.get("existing_domain_audit_dir")) else None,
        "formal_results_dir": _relative_public_path(_resolve_repo_path(args.formal_results_dir or config["run_status"]["final_results_relative_dir"])),
        "mode": args.mode,
        "dataset": args.dataset,
        "max_cells": args.max_cells,
        "n_subtype_permutations": args.n_subtype_permutations,
        "n_full_label_permutations": args.n_full_label_permutations,
        "device": args.device,
        "n_jobs": args.n_jobs,
        "overwrite": args.overwrite,
        "resume": args.resume,
    }
    implementation_hash = _implementation_hash()
    scientific_parameters = {key: value for key, value in override_record.items() if key not in {"output_dir", "overwrite", "resume"}}
    run_hash = canonical_json_hash({"config_hash": config["_config_hash"], "implementation_hash": implementation_hash, "parameters": scientific_parameters})
    if args.resume and completed_run_is_reusable(output, run_hash):
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        checks = pd.read_csv(output / "validation_suite_checks.csv")
        parent = pd.read_csv(output / "parent_disjoint_audit.csv")
        subtype_summary = pd.read_csv(output / "subtype_permutation_summary.csv")
        full_summary = pd.read_csv(output / "full_label_permutation_summary.csv")
        _print_summary(output, float(completion.get("runtime_seconds", 0.0)), checks, parent, subtype_summary, full_summary)
        return

    protocol_path = (config_path.parent / str(config["protocol"])).resolve()
    protocol = load_final_protocol(protocol_path)
    protocol_override = protocol_override_for_mode(protocol, mode_config)
    dataset = str(args.dataset or config["representative_dataset"])
    if args.mode == "smoke":
        adata = make_fixture_adata(int(mode_config["fixture_cells"]), int(mode_config["fixture_genes"]), seed=0)
        loaded = LoadedDataset("validation_fixture", adata, Path("fixture_generated"), "generated", "fixture", "not_applicable")
        dataset = "validation_fixture"
    else:
        conversion_dir = output.parent / "data"
        loaded = load_dataset_exact(_resolve_repo_path(args.data_dir), dataset, conversion_dir=conversion_dir, convert_rds=True)
        loaded.adata = deterministic_subset(loaded.adata, mode_config.get("max_cells"), seed=0)

    run = run_protocol_models(
        loaded,
        protocol_path=protocol_path,
        protocol_override=protocol_override,
        seed=0,
        backends=("rf", "dl"),
        device=args.device,
        amp=args.device in {"auto", "cuda"},
        dl_max_epochs=int(mode_config["dl_max_epochs"]),
        dl_patience=int(mode_config["dl_patience"]),
    )
    tolerance = float(config["contract"]["numerical_tolerance"])
    checks: list[SuiteCheck] = []

    parent_audit, parent_membership, parent_maps = audit_parent_disjoint(run)
    checks.append(_aggregate_frame_check("parent_disjoint", parent_audit, required=True))
    parent_removal_rows = parent_audit.loc[parent_audit["check"].isin(["generated_parent_retained_singlet_overlap", "reference_parent_overlap", "parents_marked_in_reference"])]
    checks.append(_aggregate_frame_check("parent_removal", parent_removal_rows, required=True))

    same_cell = audit_same_cell_features(run, tolerance)
    checks.append(_aggregate_frame_check("same_cell_feature_invariance", same_cell, required=True))
    order = audit_order_invariance(run, tolerance)
    checks.append(_aggregate_frame_check("cell_order_invariance", order, required=True))
    chunking = audit_chunking_invariance(run, config["invariance"]["chunk_sizes"], tolerance)
    checks.append(_aggregate_frame_check("chunking_invariance", chunking, required=True))

    state_dir = output / ".audit_state"
    transformer_serialization = audit_transformer_serialization(run, state_dir, tolerance)
    checks.append(_aggregate_frame_check("transformer_serialization", transformer_serialization, required=True))
    model_serialization = audit_model_serialization(run, state_dir, tolerance)
    for method, frame in model_serialization.groupby("context", sort=False):
        audit_name = "rf_serialization" if method == "DuoDose" else "dl_serialization"
        checks.append(_aggregate_frame_check(audit_name, frame, required=True))

    frozen_reference = audit_frozen_reference(run)
    checks.append(_aggregate_frame_check("frozen_reference", frozen_reference, required=True))
    deterministic = _deterministic_rerun(loaded, run, protocol_override, args, tolerance)
    checks.append(_aggregate_frame_check("deterministic_rf_rerun", deterministic, required=True))
    checks.append(SuiteCheck("dl_training_reproducibility", "independent_training", "NOT_APPLICABLE", False, "fixed-model DL inference is audited; independent CUDA training is not claimed bitwise deterministic"))

    feature_audit = validate_feature_names(run.transformer.model_feature_columns_, protocol_override["features"]["allowlist"])
    checks.append(_aggregate_frame_check("feature_leakage", feature_audit, required=True))

    ledger_path, snapshot_path = progress_paths(output, args)
    subtype_results, subtype_summary, full_results, full_summary = _run_permutations(
        run,
        n_subtype,
        n_full,
        settings=ProgressSettings.from_args(args),
        ledger_path=ledger_path,
        snapshot_path=snapshot_path,
        config_hash=str(args.progress_config_hash or run_hash),
        output_path=output,
    )
    checks.append(_check("subtype_permutation_execution", "completed", len(subtype_results) == n_subtype and subtype_summary["execution_status"].eq("COMPLETED").all(), required=True, value=len(subtype_results)))
    checks.append(_check("subtype_permutation_contract", "contract", subtype_summary["contract_status"].eq("PASS").all(), required=True, value=subtype_summary["contract_status"].iloc[0]))
    checks.append(_check("subtype_permutation_evidence", "scientific_evidence", True, required=False, value=",".join(sorted(subtype_summary["evidence_status"].unique())), message="evidence status does not determine software validity"))
    checks.append(_check("full_label_permutation_execution", "completed", len(full_results) == n_full and full_summary["execution_status"].eq("COMPLETED").all(), required=True, value=len(full_results)))
    checks.append(_check("full_label_permutation_contract", "contract", full_summary["contract_status"].eq("PASS").all(), required=True, value=full_summary["contract_status"].iloc[0]))
    checks.append(_check("full_label_permutation_evidence", "scientific_evidence", True, required=False, value=",".join(sorted(full_summary["evidence_status"].unique())), message="evidence status does not determine software validity"))

    probability_frames = []
    for method, frame in run.method_probabilities_test.items():
        probability_frames.append(validate_probability_frame(frame, method, tolerance))
    probability_audit = pd.concat(probability_frames, ignore_index=True)
    checks.append(_aggregate_frame_check("probability_contract", probability_audit, required=True))

    metrics = metric_contract_table()
    metrics["status"] = "PASS"
    checks.append(_aggregate_frame_check("metric_contract", metrics, required=False))
    schema = schema_audit(loaded.adata, run)
    checks.append(_aggregate_frame_check("schema_contract", schema, required=True))

    run_status_root = _resolve_repo_path(args.formal_results_dir or config["run_status"]["final_results_relative_dir"])
    run_status = audit_run_status(run_status_root, protocol)
    expected_rows = (
        len(protocol["datasets"]["real_doublet_enriched"]) * len(protocol["seeds"]["controlled_benchmark"]) * 6
        + len(protocol["datasets"]["real_application"]) * len(protocol["seeds"]["real_application"]) * 5
    )
    checks.append(_check("run_status_contract", "complete_requested_grid", len(run_status) == expected_rows and not run_status.duplicated(["workflow", "dataset", "seed", "method"]).any(), required=False, value=len(run_status)))
    incomplete_count = int(run_status["status"].ne("COMPLETED").sum())
    checks.append(SuiteCheck("run_status_contract", "formal_run_completeness", "INCOMPLETE" if incomplete_count else "PASS", False, "current controlled-benchmark and real-data-application formal units", incomplete_count))
    configured_domain_dir = args.existing_domain_audit_dir or config.get("existing_domain_audit_dir")
    domain_contract = domain_audit_contract_check(_resolve_repo_path(configured_domain_dir) if configured_domain_dir else None)
    domain_status = "PASS" if domain_contract["status"].eq("PASS").all() else "NOT_RUN" if domain_contract["status"].eq("NOT_RUN").any() else "INCOMPLETE"
    checks.append(SuiteCheck("existing_domain_audit_contract", "existing_outputs", domain_status, False, str(domain_contract.iloc[0].get("reason", "")), len(domain_contract)))

    atomic_write_csv(output / "parent_disjoint_audit.csv", parent_audit)
    atomic_write_csv(output / "parent_split_membership.csv.gz", parent_membership)
    atomic_write_csv(output / "parent_maps.csv.gz", parent_maps)
    atomic_write_csv(output / "same_cell_feature_invariance.csv", same_cell)
    atomic_write_csv(output / "cell_order_invariance.csv", order)
    atomic_write_csv(output / "chunking_invariance.csv", chunking)
    atomic_write_csv(output / "transformer_save_load_invariance.csv", transformer_serialization)
    atomic_write_csv(output / "model_save_load_invariance.csv", model_serialization)
    atomic_write_csv(output / "frozen_reference_audit.csv", frozen_reference)
    atomic_write_csv(output / "deterministic_rerun_audit.csv", deterministic)
    atomic_write_csv(output / "domain_feature_audit.csv", feature_audit)
    atomic_write_csv(output / "subtype_permutation_results.csv", subtype_results)
    atomic_write_csv(output / "subtype_permutation_summary.csv", subtype_summary)
    atomic_write_csv(output / "full_label_permutation_results.csv", full_results)
    atomic_write_csv(output / "full_label_permutation_summary.csv", full_summary)
    atomic_write_csv(output / "probability_contract_audit.csv", probability_audit)
    atomic_write_csv(output / "run_status_audit.csv", run_status)
    atomic_write_csv(output / "metric_contract_audit.csv", metrics)
    atomic_write_csv(output / "schema_audit.csv", schema)
    atomic_write_csv(output / "domain_audit_contract_check.csv", domain_contract)

    invariance = pd.concat(
        [same_cell, order, chunking, transformer_serialization, model_serialization.assign(audit="model_serialization"), deterministic.rename(columns={"context": "feature_or_output"}).assign(audit="deterministic_rf_rerun")],
        ignore_index=True,
        sort=False,
    )
    _plot_outputs(output, invariance, parent_audit, subtype_results, subtype_summary, full_results, full_summary, probability_audit)
    if state_dir.exists():
        shutil.rmtree(state_dir)

    runtime = time.perf_counter() - started
    environment = environment_record()
    environment.update({"validation_mode": args.mode, "n_jobs_requested": int(args.n_jobs), "device_requested": args.device})
    config_record = {
        "schema_version": 1,
        "suite_name": config["suite_name"],
        "config_file": _relative_public_path(config_path),
        "config_hash": config["_config_hash"],
        "run_hash": run_hash,
        "implementation_hash": implementation_hash,
        "mode": args.mode,
        "dataset": dataset,
        "overrides": override_record,
        "frozen_contract": config["contract"],
        "protocol_file": _relative_public_path(protocol_path),
    }
    atomic_write_json(output / "validation_suite_config.json", config_record)
    atomic_write_json(output / "validation_suite_environment.json", environment)
    absolute_paths = _absolute_path_matches(output)
    schema = pd.concat(
        [
            schema,
            pd.DataFrame(
                [
                    {
                        "check": "no_local_absolute_paths_in_public_outputs",
                        "value": len(absolute_paths),
                        "status": "PASS" if not absolute_paths else "FAIL",
                        "message": "" if not absolute_paths else ", ".join(absolute_paths),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    atomic_write_csv(output / "schema_audit.csv", schema)
    checks.append(_check("schema_contract", "no_local_absolute_paths_in_public_outputs", not absolute_paths, required=True, value=len(absolute_paths), message="" if not absolute_paths else ", ".join(absolute_paths)))

    checks_frame = pd.DataFrame([item.as_dict() for item in checks])
    hard_failure_count = int((checks_frame["required"].astype(bool) & checks_frame["status"].eq("FAIL")).sum())
    code_completion_status = "COMPLETE" if hard_failure_count == 0 else "INCOMPLETE"
    formal_analysis_status = "COMPLETE" if run_status["status"].eq("COMPLETED").all() else "INCOMPLETE"
    metadata = {"mode": args.mode, "dataset": dataset, "config_hash": config["_config_hash"], "runtime_seconds": runtime, "code_completion_status": code_completion_status, "formal_analysis_status": formal_analysis_status}
    summary = checks_frame.groupby("status", sort=False).size().rename("count").reset_index()
    summary["code_completion_status"] = code_completion_status
    summary["formal_analysis_status"] = formal_analysis_status
    atomic_write_csv(output / "validation_suite_checks.csv", checks_frame)
    atomic_write_csv(output / "validation_suite_summary.csv", summary)
    atomic_write_text(output / "validation_suite_report.md", _report(output, metadata, checks_frame, parent_audit, subtype_summary, full_summary, run_status, domain_contract))

    manifest_files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"validation_suite_manifest.json", ".validation_suite_complete.json"} and not path.name.endswith(".tmp"):
            manifest_files.append({"path": str(path.relative_to(output)).replace("\\", "/"), "size_bytes": path.stat().st_size})
    atomic_write_json(output / "validation_suite_manifest.json", {"schema_version": 1, "run_hash": run_hash, "config_hash": config["_config_hash"], "code_completion_status": code_completion_status, "formal_analysis_status": formal_analysis_status, "files": manifest_files})
    required = verify_required_outputs(output)
    if not required["status"].eq("PASS").all():
        missing = required.loc[required["status"].ne("PASS"), "output"].tolist()
        checks_frame = pd.concat([checks_frame, pd.DataFrame([SuiteCheck("required_outputs", name, "INCOMPLETE", True, "required output missing or empty").as_dict() for name in missing])], ignore_index=True)
        atomic_write_csv(output / "validation_suite_checks.csv", checks_frame)
        incomplete_summary = checks_frame.groupby("status", sort=False).size().rename("count").reset_index()
        incomplete_summary["code_completion_status"] = "INCOMPLETE"
        incomplete_summary["formal_analysis_status"] = formal_analysis_status
        atomic_write_csv(output / "validation_suite_summary.csv", incomplete_summary)
        raise ValidationSuiteError("required validation outputs are incomplete: " + ", ".join(missing))

    hard_failures = checks_frame.loc[checks_frame["required"].astype(bool) & checks_frame["status"].eq("FAIL")]
    if hard_failures.empty:
        atomic_write_json(completion_path, {"run_hash": run_hash, "runtime_seconds": runtime, "completed": True})
    _print_summary(output, runtime, checks_frame, parent_audit, subtype_summary, full_summary)
    if not hard_failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
