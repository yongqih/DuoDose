from duodose.models.registry import BACKEND_SPECS
from duodose.protocol import load_final_protocol


def test_final_protocol_matches_public_registry() -> None:
    protocol = load_final_protocol()
    assert protocol["construction"] == {
        "construction_variant": "raw_sum_parents_removed",
        "count_construction_mode": "raw_sum",
        "parent_reference_mode": "removed",
        "safe_feature_mode": "fitted_reference",
        "parent_disjoint": True,
    }
    assert list(BACKEND_SPECS) == ["rf", "dl"]
    assert protocol["models"]["main_internal_name"] == BACKEND_SPECS["rf"].internal_name
    assert protocol["models"]["ablation_internal_name"] == BACKEND_SPECS["dl"].internal_name
    assert protocol["real_application"]["backend"] == "rf"
    assert protocol["real_application"]["internal_method_name"] == BACKEND_SPECS["rf"].internal_name
    assert protocol["real_application"]["required_real_label_metrics"] == []
    assert protocol["seeds"]["real_application"] == [0]
    assert protocol["models"]["model_revision"] == "complexity_balance_v1"
    assert "library_complexity_balance" in protocol["features"]["allowlist"]
    assert protocol["evaluation"]["primary_high_rna_fpr"]["target_homotypic_recall"] == 0.50
    assert protocol["evaluation"]["supplementary_high_rna_fpr"]["fixed_candidate_fraction"] == 0.20
