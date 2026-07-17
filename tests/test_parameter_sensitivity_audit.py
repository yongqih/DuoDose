from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from duodose.parameter_sensitivity_audit import (
    aggregate_sensitivity,
    canonical_top_true_doublet_budget,
    deterministic_top_fraction,
    metric_contract_table,
    sensitivity_run_status,
    training_size_protocol,
)
from duodose.safe_feature_transformer import SafeFeatureTransformer
from duodose.semireal_bundle import make_parent_disjoint_semireal_bundle
from duodose.validation_suite import audit_parent_disjoint, make_fixture_adata


def _protocol() -> dict:
    return {
        "semi_real": {
            "n_train_homotypic_doublets": 20,
            "n_train_heterotypic_doublets": 20,
            "n_validation_homotypic_doublets": 6,
            "n_validation_heterotypic_doublets": 6,
            "n_test_homotypic_doublets": 8,
            "n_test_heterotypic_doublets": 8,
        },
        "parameter_sensitivity": {"select_configuration_on_test_data": False},
    }


def test_expected_rate_changes_rank_calls_not_continuous_scores() -> None:
    labels = pd.Series(
        ["clean", "high_RNA_singlet", "homotypic_doublet", "heterotypic_doublet"],
        index=["a", "b", "c", "d"],
    )
    scores = pd.Series([0.1, 0.8, 0.9, 0.7], index=labels.index)
    before = scores.copy()
    low = deterministic_top_fraction(labels, scores, 0.25)
    high = deterministic_top_fraction(labels, scores, 0.50)
    pd.testing.assert_series_equal(scores, before)
    assert low["number_selected_cells"] == 1
    assert high["number_selected_cells"] == 2
    assert low["threshold"] != high["threshold"]


def test_high_rna_fpr_operating_points_use_documented_budgets() -> None:
    labels = pd.Series(
        ["high_RNA_singlet", "clean", "homotypic_doublet", "heterotypic_doublet", "clean"],
        index=["b", "a", "c", "d", "e"],
    )
    scores = pd.Series([0.8, 0.8, 0.9, 0.7, 0.1], index=labels.index)
    canonical = canonical_top_true_doublet_budget(labels, scores)
    expected = deterministic_top_fraction(labels, scores, 0.20)
    assert canonical["number_selected_cells"] == 2
    assert canonical["high_RNA_singlet_FPR"] == 1.0
    assert expected["number_selected_cells"] == 1
    assert expected["high_RNA_singlet_FPR"] == 0.0


def test_metric_contract_declares_both_fprs_lower_is_better() -> None:
    contract = metric_contract_table().set_index("metric")
    assert set(contract["higher_or_lower_is_better"]) == {"lower"}
    assert "fixed" not in contract.loc["high_RNA_singlet_FPR", "threshold_rule"]
    assert "round" in contract.loc["high_RNA_singlet_FPR_at_expected_rate", "threshold_rule"]


def test_training_size_protocol_changes_only_train_counts() -> None:
    protocol = _protocol()
    varied = training_size_protocol(protocol, 2.0)
    assert varied["semi_real"]["n_train_homotypic_doublets"] == 40
    assert varied["semi_real"]["n_train_heterotypic_doublets"] == 40
    for name in (
        "n_validation_homotypic_doublets",
        "n_validation_heterotypic_doublets",
        "n_test_homotypic_doublets",
        "n_test_heterotypic_doublets",
    ):
        assert varied["semi_real"][name] == protocol["semi_real"][name]
    assert varied["parameter_sensitivity"]["select_configuration_on_test_data"] is False
    assert protocol == _protocol()


def _bundle(train_count: int):
    adata = make_fixture_adata(700, 80, seed=41)
    adata.obs["experimental_doublet"] = 0
    return make_parent_disjoint_semireal_bundle(
        adata,
        dataset="sensitivity_fixture",
        seed=3,
        n_singlets=300,
        n_train_homotypic_doublets=train_count,
        n_train_heterotypic_doublets=train_count,
        n_test_homotypic_doublets=8,
        n_test_heterotypic_doublets=8,
        n_validation_homotypic_doublets=6,
        n_validation_heterotypic_doublets=6,
        n_clusters=4,
        test_parent_fraction=0.40,
        validation_parent_fraction=0.25,
        high_rna_quantile=0.90,
        min_cluster_size=5,
        construction_variant="raw_sum_parents_removed",
    )


def test_evaluation_cells_are_identical_when_only_training_size_changes() -> None:
    small = _bundle(10)
    large = _bundle(40)
    assert small.val_adata.obs_names.equals(large.val_adata.obs_names)
    assert small.test_adata.obs_names.equals(large.test_adata.obs_names)
    assert small.reference_cell_ids.equals(large.reference_cell_ids)
    for split in ("validation", "test"):
        left = small.parent_map.loc[small.parent_map["split"].eq(split)].reset_index(drop=True)
        right = large.parent_map.loc[large.parent_map["split"].eq(split)].reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)


def test_fitted_reference_transformer_is_identical_across_training_sizes() -> None:
    small = _bundle(10)
    large = _bundle(40)

    def fit_transformer(bundle):
        origin = bundle.fit_adata.obs["semireal_origin"].astype(str)
        reference = bundle.fit_adata[origin.isin({"observed_background", "real_labeled_singlet"}).to_numpy(), :].copy()
        return SafeFeatureTransformer(random_state=3, reference_seed=3, n_components=8, n_clusters=4, n_neighbors=5).fit(
            reference,
            reference_pool_id="sensitivity_fixture|seed=3|fit_split_clean_singlets",
            dataset="sensitivity_fixture",
        )

    small_transformer = fit_transformer(small)
    large_transformer = fit_transformer(large)
    assert small_transformer.reference_cell_ids_.equals(large_transformer.reference_cell_ids_)
    assert small_transformer.reference_pool_id_ == large_transformer.reference_pool_id_
    assert small_transformer.transformer_id_ == large_transformer.transformer_id_
    assert small_transformer.model_feature_columns_ == large_transformer.model_feature_columns_


def test_parent_disjointness_and_duplicates_remain_zero_at_2x() -> None:
    bundle = _bundle(40)
    fit_reference = bundle.fit_adata.obs.get("semireal_origin", pd.Series("", index=bundle.fit_adata.obs_names)).astype(str).isin({"observed_background", "real_labeled_singlet"})
    reference_ids = bundle.fit_adata.obs_names[fit_reference]
    run = type("Run", (), {"bundle": bundle, "transformer": type("Transformer", (), {"reference_cell_ids_": reference_ids})()})()
    audit, _, _ = audit_parent_disjoint(run)
    required_zero = audit.loc[audit["required_value"].astype(str).eq("0")]
    assert np.all(pd.to_numeric(required_zero["value"]) == 0)
    duplicate_expression = audit.loc[audit["check"].eq("n_duplicate_generated_expression_profiles"), "value"]
    assert int(duplicate_expression.iloc[0]) == 0


def test_aggregation_reproduces_sample_statistics_and_status_surfaces_missing() -> None:
    rows = []
    for seed, value in ((0, 0.2), (1, 0.4), (2, 0.6)):
        row = {
            "dataset": "fixture",
            "seed": seed,
            "status": "success",
            "message": "",
            "semi_real_size_factor": 1.0,
            "expected_doublet_rate": 0.1,
        }
        for metric in (
            "overall_AUPRC",
            "homotypic_AUPRC",
            "heterotypic_AUPRC",
            "homotypic_vs_high_RNA_singlet_AUPRC",
            "high_RNA_singlet_FPR",
            "high_RNA_singlet_FPR_at_expected_rate",
        ):
            row[metric] = value
        rows.append(row)
    frame = pd.DataFrame(rows)
    summary = aggregate_sensitivity(frame)
    assert summary.loc[0, "overall_AUPRC_mean"] == pytest.approx(0.4)
    assert summary.loc[0, "overall_AUPRC_std"] == pytest.approx(0.2)
    status = sensitivity_run_status(frame, dataset="fixture", seeds=[0, 1, 2], factors=[1.0], expected_rates=[0.1, 0.2])
    assert int(status["status"].eq("SUCCESS").sum()) == 3
    assert int(status["status"].eq("NOT_RUN").sum()) == 3
