from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from duodose.domain_audit_aggregate import regenerate_domain_audit_outputs
from duodose.domain_audit_contract import (
    PRIMARY_ANALYSIS,
    normalize_primary_analysis,
    validate_primary_audit,
    validate_primary_audit_files,
)


def _summary_row(*, status: str = "PASS", analysis: str = PRIMARY_ANALYSIS, auroc: float = 0.61, auprc: float = 0.59, overlap: float = 0.0) -> dict[str, object]:
    return {
        "dataset": "cline-ch",
        "source_run_id": "domain_audit",
        "analysis": analysis,
        "is_primary": True,
        "status": status,
        "message": "" if status.upper() in {"PASS", "COMPLETED", "SUCCESS"} else "scientific analysis failed",
        "n_experimental_doublets": 60,
        "n_semireal_heterotypic_doublets": 60,
        "n_semireal_before_parent_unique_filter": 80,
        "n_semireal_after_parent_unique_filter": 60,
        "parent_unique_retention_fraction": 0.75,
        "n_unique_parents_retained": 120,
        "parent_overlap_across_folds": overlap,
        "n_features": 7,
        "n_folds": 3,
        "auroc_mean": auroc,
        "auroc_std": 0.02,
        "auprc_mean": auprc,
        "auprc_std": 0.03,
        "pooled_oof_auroc": auroc,
        "pooled_oof_auprc": auprc,
        "balanced_accuracy": 0.58,
        "mcc": 0.16,
        "split_strategy": "matched_parent_unique_round_robin_stratified_greedy",
        "fold_balance_status": "PASS",
        "transformer_reference_provenance_status": "PASS",
        "construction_variant": "raw_sum_parents_removed",
        "safe_feature_mode": "fitted_reference",
        "semireal_split_used": "test",
    }


def _support_frame(analysis: str = PRIMARY_ANALYSIS) -> pd.DataFrame:
    return pd.DataFrame({"dataset": ["cline-ch"], "analysis": [analysis], "fold": [1]})


def _write_dataset(root: Path, row: dict[str, object]) -> Path:
    dataset = str(row["dataset"])
    output = root / dataset
    output.mkdir(parents=True)
    pd.DataFrame([row]).to_csv(output / "domain_audit_summary.csv", index=False)
    _support_frame(str(row["analysis"])).to_csv(output / "domain_audit_fold_metrics.csv", index=False)
    _support_frame(str(row["analysis"])).assign(stable_cell_id="cell_1").to_csv(output / "domain_audit_predictions.csv", index=False)
    pd.DataFrame({"feature": ["identity_inlier_score"], "included": [True]}).to_csv(output / "domain_audit_feature_audit.csv", index=False)
    pd.DataFrame({"dataset": [dataset], "canonical_cluster": ["cluster_1"]}).to_csv(output / "domain_audit_cluster_balance.csv", index=False)
    return output


@pytest.mark.parametrize("status", ["PASS", "COMPLETED", "SUCCESS", "success"])
def test_valid_primary_success_statuses_are_completed(status: str) -> None:
    validation = validate_primary_audit(pd.DataFrame([_summary_row(status=status)]), _support_frame(), _support_frame())
    assert validation.completed
    assert validation.audit_status == "COMPLETED"
    assert validation.primary is not None and validation.primary["status"] == "COMPLETED"


@pytest.mark.parametrize(
    "alias",
    ["matched_safe_features", "matched_mechanism_features", "matched_raw_mechanism_features", "matched_heterotypic_mechanism_features"],
)
def test_primary_analysis_aliases_are_normalized(alias: str) -> None:
    assert normalize_primary_analysis(alias) == PRIMARY_ANALYSIS
    validation = validate_primary_audit(pd.DataFrame([_summary_row(analysis=alias)]), _support_frame(alias), _support_frame(alias))
    assert validation.completed
    assert validation.primary is not None and validation.primary["analysis"] == PRIMARY_ANALYSIS


def test_pass_with_missing_metrics_is_rejected() -> None:
    validation = validate_primary_audit(pd.DataFrame([_summary_row(auroc=np.nan)]), _support_frame(), _support_frame())
    assert not validation.completed
    assert validation.audit_status == "CONTRACT_ERROR"
    assert "non-finite" in validation.reason


def test_pass_with_parent_overlap_is_rejected() -> None:
    validation = validate_primary_audit(pd.DataFrame([_summary_row(overlap=1)]), _support_frame(), _support_frame())
    assert not validation.completed
    assert validation.audit_status == "CONTRACT_ERROR"
    assert "parent_overlap" in validation.reason


def test_unknown_primary_status_is_a_clear_contract_error() -> None:
    validation = validate_primary_audit(pd.DataFrame([_summary_row(status="MYSTERY")]), _support_frame(), _support_frame())
    assert not validation.completed
    assert validation.audit_status == "CONTRACT_ERROR"
    assert "unknown dataset-level primary status" in validation.reason


def test_wrapper_success_does_not_override_true_analysis_error(tmp_path: Path) -> None:
    root = tmp_path / "domain_audit"
    row = _summary_row(status="ANALYSIS_ERROR")
    _write_dataset(root, row)
    pd.DataFrame({"dataset": ["cline-ch"], "status": ["SUCCESS"], "message": [""]}).to_csv(root / "domain_audit_batch_run_status.csv", index=False)
    regenerate_domain_audit_outputs(root)
    combined = pd.read_csv(root / "domain_audit_all_datasets_summary.csv")
    assert combined.iloc[0]["wrapper_execution_status"] == "SUCCESS"
    assert combined.iloc[0]["audit_status"] == "ANALYSIS_ERROR"


def test_existing_valid_primary_generates_completed_summary_and_four_plots(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    root = tmp_path / "domain_audit"
    output = _write_dataset(root, _summary_row())
    pd.DataFrame({"dataset": ["cline-ch"], "status": ["SUCCESS"], "message": [""]}).to_csv(root / "domain_audit_batch_run_status.csv", index=False)
    assert validate_primary_audit_files(output).completed
    regenerate_domain_audit_outputs(root)
    combined = pd.read_csv(root / "domain_audit_all_datasets_summary.csv")
    assert combined.iloc[0]["analysis"] == PRIMARY_ANALYSIS
    assert combined.iloc[0]["status"] == "COMPLETED"
    assert combined.iloc[0]["audit_status"] == "COMPLETED"
    for name in (
        "domain_audit_all_datasets_auroc_comparison.png",
        "domain_audit_all_datasets_auroc_comparison.pdf",
        "domain_audit_all_datasets_matched_direction_adjusted.png",
        "domain_audit_all_datasets_matched_direction_adjusted.pdf",
    ):
        assert (root / name).is_file() and (root / name).stat().st_size > 0
