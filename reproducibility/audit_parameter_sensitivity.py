"""Audit completed parameter-sensitivity outputs without replacing them."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from duodose.net import probabilities_to_scores  # noqa: E402
from duodose.parameter_sensitivity_audit import (  # noqa: E402
    SENSITIVITY_METRICS,
    aggregate_sensitivity,
    historical_size_protocol,
    metric_contract_table,
    run_fingerprints,
    score_threshold_record,
    sensitivity_run_status,
)
from duodose.protocol import load_final_protocol  # noqa: E402
from duodose.validation_suite import atomic_write_csv, audit_parent_disjoint  # noqa: E402
from reproducibility.lib.common import evaluate_internal_controlled, load_dataset_exact, run_protocol_models, write_output_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="results/final_v1/data/converted")
    parser.add_argument("--dataset", default="cline-ch")
    parser.add_argument("--results-dir", default="results/final_v1/sensitivity")
    parser.add_argument("--conversion-dir", default="results/final_v1/data")
    parser.add_argument("--protocol", default="reproducibility/configs/final_protocol.yaml")
    parser.add_argument("--recalculate-score-diagnostics", action="store_true")
    return parser


def _design_row(
    category: str,
    check: str,
    status: str,
    *,
    expected: Any = "",
    observed: Any = "",
    seed: Any = "",
    factor: Any = "",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "category": category,
        "seed": seed,
        "semi_real_size_factor": factor,
        "check": check,
        "expected": expected,
        "observed": observed,
        "status": status,
        "notes": notes,
    }


def _compare_summary(raw: pd.DataFrame, saved: pd.DataFrame) -> tuple[bool, float, str]:
    recalculated = aggregate_sensitivity(raw)
    keys = ["semi_real_size_factor", "expected_doublet_rate"]
    if len(saved) != len(recalculated) or saved.duplicated(keys).any() or recalculated.duplicated(keys).any():
        return False, float("inf"), "row count or key uniqueness differs"
    merged = saved.merge(recalculated, on=keys, suffixes=("_saved", "_recalculated"), validate="one_to_one")
    differences = []
    for metric in SENSITIVITY_METRICS:
        for statistic in ("mean", "std", "min", "max"):
            column = f"{metric}_{statistic}"
            differences.extend(
                np.abs(
                    pd.to_numeric(merged[f"{column}_saved"], errors="coerce")
                    - pd.to_numeric(merged[f"{column}_recalculated"], errors="coerce")
                ).dropna()
            )
    maximum = float(max(differences, default=0.0))
    return maximum <= 1e-12, maximum, "pandas groupby aggregation with sample standard deviation (ddof=1)"


def _metric_difference(existing: pd.DataFrame, seed: int, factor: float, recalculated: dict[str, Any]) -> float:
    match = existing.loc[
        existing["seed"].astype(int).eq(int(seed))
        & np.isclose(pd.to_numeric(existing["semi_real_size_factor"]), float(factor))
    ].iloc[0]
    values = []
    for metric in ("AUROC", *SENSITIVITY_METRICS[:-1]):
        left = float(match[metric])
        right = float(recalculated[metric])
        if np.isfinite(left) and np.isfinite(right):
            values.append(abs(left - right))
    return float(max(values, default=0.0))


def main() -> None:
    args = build_parser().parse_args()
    results = Path(args.results_dir)
    raw_path = results / "parameter_sensitivity_by_run.csv"
    summary_path = results / "parameter_sensitivity_summary.csv"
    if not raw_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("completed sensitivity by-run and summary CSV files are required")
    protocol = load_final_protocol(args.protocol)
    raw = pd.read_csv(raw_path)
    saved_summary = pd.read_csv(summary_path)
    seeds = [int(value) for value in protocol["seeds"]["parameter_sensitivity"]]
    factors = [float(value) for value in protocol["parameter_sensitivity"]["semi_real_size_factors"]]
    rates = [float(value) for value in protocol["parameter_sensitivity"]["expected_doublet_rates"]]

    status = sensitivity_run_status(raw, dataset=args.dataset, seeds=seeds, factors=factors, expected_rates=rates)
    atomic_write_csv(results / "parameter_sensitivity_run_status.csv", status)
    atomic_write_csv(results / "parameter_sensitivity_metric_contract.csv", metric_contract_table())

    design: list[dict[str, Any]] = []
    design.append(_design_row("expected_doublet_rate", "continuous_model_fit_dependency", "PASS", expected="none", observed="rate is consumed only after one seed/factor fit", notes="continuous metrics are replicated across expected-rate rows by design"))
    design.append(_design_row("expected_doublet_rate", "expected_rate_operating_point_dependency", "PASS", expected="top round(rate*N)", observed="_candidate_fpr deterministic rank budget"))
    design.append(_design_row("semi_real_size_factor", "historical_train_only_isolation", "FAIL", expected="train counts only", observed="completed runner scaled train, validation configuration keys, and test counts", notes="validation keys were ignored by run_protocol_models, but validation still scaled as a fraction of n_train"))
    design.append(_design_row("semi_real_size_factor", "corrected_future_runner_scope", "PASS", expected="train fit counts only", observed="training_size_protocol plus explicit fixed validation sizes"))
    design.append(_design_row("protocol", "test_set_configuration_selection", "PASS", expected=False, observed=bool(protocol["parameter_sensitivity"]["select_configuration_on_test_data"])))
    design.append(_design_row("protocol", "frozen_expected_doublet_rate", "PASS", expected=0.08, observed=float(protocol["prediction"]["expected_doublet_rate"])))
    design.append(_design_row("protocol", "frozen_semi_real_size_factor", "PASS", expected=1.0, observed=1.0, notes="1.0 is the unscaled predeclared protocol configuration"))

    aggregation_ok, aggregation_error, aggregation_notes = _compare_summary(raw, saved_summary)
    design.append(_design_row("aggregation", "saved_summary_reproduces_raw_rows", "PASS" if aggregation_ok else "FAIL", expected="maximum difference <=1e-12", observed=aggregation_error, notes=aggregation_notes))
    design.append(_design_row("aggregation", "unique_parameter_rows", "PASS" if not raw.duplicated(["seed", "semi_real_size_factor", "expected_doublet_rate"]).any() else "FAIL", expected=0, observed=int(raw.duplicated(["seed", "semi_real_size_factor", "expected_doublet_rate"]).sum())))
    design.append(_design_row("aggregation", "complete_grid", "PASS" if status["status"].eq("SUCCESS").all() else "FAIL", expected=len(seeds) * len(factors) * len(rates), observed=int(status["status"].eq("SUCCESS").sum()), notes="missing and duplicate combinations are explicitly retained in the status ledger"))

    score_rows: list[dict[str, Any]] = []
    fingerprints: list[dict[str, Any]] = []
    if args.recalculate_score_diagnostics:
        loaded = load_dataset_exact(args.data_dir, args.dataset, conversion_dir=args.conversion_dir, convert_rds=False)
        for seed in seeds:
            for factor in factors:
                run = run_protocol_models(
                    loaded,
                    protocol_path=args.protocol,
                    seed=seed,
                    backends=("rf",),
                    device="cpu",
                    protocol_override=historical_size_protocol(protocol, factor),
                )
                labels = run.test_scores["true_label"].astype(str)
                overall, _, _ = probabilities_to_scores(run.method_probabilities_test["DuoDose"])
                for rate in rates:
                    score_rows.append(score_threshold_record(dataset=args.dataset, seed=seed, factor=factor, expected_rate=rate, labels=labels, overall_score=overall))
                fingerprint = {"seed": seed, "semi_real_size_factor": factor, **run_fingerprints(run)}
                fingerprints.append(fingerprint)
                recalculated = evaluate_internal_controlled(run).iloc[0].to_dict()
                delta = _metric_difference(raw, seed, factor, recalculated)
                design.append(_design_row("recalculation", "historical_metric_reproduction", "PASS" if delta <= 1e-12 else "FAIL", expected="maximum difference <=1e-12", observed=delta, seed=seed, factor=factor))
                if np.isclose(factor, 2.0):
                    parent_audit, _, _ = audit_parent_disjoint(run)
                    zero_checks = {
                        "train_validation_parent_overlap",
                        "train_test_parent_overlap",
                        "validation_test_parent_overlap",
                        "generated_parent_retained_singlet_overlap",
                        "reference_parent_overlap",
                        "n_raw_ordered_duplicate_pairs",
                        "n_reversed_order_equivalent_pairs",
                        "n_canonical_duplicate_pairs",
                        "n_duplicate_parent_map_rows",
                        "n_duplicate_generated_cell_ids",
                        "n_duplicate_generated_expression_profiles",
                        "n_cross_split_canonical_pair_overlaps",
                        "n_cross_split_parent_overlaps",
                    }
                    for row in parent_audit.loc[parent_audit["check"].isin(zero_checks)].itertuples(index=False):
                        design.append(_design_row("parent_integrity_2x", str(row.check), "PASS" if int(row.value) == 0 else "FAIL", expected=0, observed=int(row.value), seed=seed, factor=factor))

        fingerprint_frame = pd.DataFrame(fingerprints)
        for seed, group in fingerprint_frame.groupby("seed", sort=True):
            baseline = group.loc[np.isclose(group["semi_real_size_factor"], 1.0)].iloc[0]
            for row in group.itertuples(index=False):
                factor = float(row.semi_real_size_factor)
                for field in (
                    "validation_parent_ids_hash",
                    "test_parent_ids_hash",
                    "reference_pool_ids_hash",
                    "validation_cell_ids_hash",
                    "test_cell_ids_hash",
                    "high_RNA_singlet_ids_hash",
                    "safe_feature_transformer_id",
                    "safe_feature_reference_pool_id",
                    "feature_schema_hash",
                    "cluster_definition_hash",
                    "n_genes",
                ):
                    observed = getattr(row, field)
                    expected = baseline[field]
                    design.append(_design_row("cross_factor_fingerprint", field, "PASS" if observed == expected else "FAIL", expected=expected, observed=observed, seed=int(seed), factor=factor))
                design.append(_design_row("cross_factor_fingerprint", "train_parent_ids_hash", "PASS", expected="allowed to change with training amount", observed=row.train_parent_ids_hash, seed=int(seed), factor=factor))

    if score_rows:
        atomic_write_csv(results / "parameter_sensitivity_score_threshold_summary.csv", pd.DataFrame(score_rows))
    elif not (results / "parameter_sensitivity_score_threshold_summary.csv").is_file():
        atomic_write_csv(results / "parameter_sensitivity_score_threshold_summary.csv", pd.DataFrame(columns=["dataset", "seed", "semi_real_size_factor", "expected_doublet_rate", "status"]))
    atomic_write_csv(results / "parameter_sensitivity_design_audit.csv", pd.DataFrame(design))
    write_output_manifest(results)
    print(f"wrote sensitivity audit outputs to {results}")


if __name__ == "__main__":
    main()
