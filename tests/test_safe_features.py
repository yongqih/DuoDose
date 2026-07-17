import pandas as pd

from duodose.net import split_safe_feature_columns
from duodose.safe_feature_manifest import LEGACY_FEATURE_NAME_MAP, migrate_legacy_feature_names, safe_feature_provenance


def test_safe_features_exclude_truth_and_external_scores() -> None:
    included, excluded = split_safe_feature_columns(
        ["dosage_outlier_score", "true_label", "benchmark_seed", "scrublet_score", "hybrid_overall_score", "external_method_score"]
    )
    assert included == ["dosage_outlier_score"]
    assert set(excluded) == {"true_label", "benchmark_seed", "scrublet_score", "hybrid_overall_score", "external_method_score"}


def test_historical_pre_model_feature_names_migrate_without_value_changes() -> None:
    historical = pd.DataFrame({"duodose_score": [0.2, 0.8], "legacy_heterotypic_score": [0.1, 0.7]}, index=["a", "b"])
    migrated = migrate_legacy_feature_names(historical)
    assert list(migrated.columns) == ["handcrafted_combined_score", "handcrafted_artificial_doublet_compatible_score"]
    assert migrated["handcrafted_combined_score"].tolist() == [0.2, 0.8]
    assert LEGACY_FEATURE_NAME_MAP["duodose_score"] == "handcrafted_combined_score"
    provenance = safe_feature_provenance("handcrafted_combined_score")
    assert provenance["computed_before_model_fit"] is True
    assert provenance["uses_truth_labels"] is False
    assert provenance["uses_model_output"] is False
    assert provenance["uses_outcome_calibration"] is False
    assert provenance["uses_dataset_rank"] is False


def test_library_complexity_balance_is_safe_row_local_dosage_feature() -> None:
    provenance = safe_feature_provenance("library_complexity_balance")
    assert provenance["computed_before_model_fit"] is True
    assert provenance["uses_truth_labels"] is False
    included, excluded = split_safe_feature_columns(["library_complexity_balance"])
    assert included == ["library_complexity_balance"]
    assert excluded == []
