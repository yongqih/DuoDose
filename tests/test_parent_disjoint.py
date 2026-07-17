import numpy as np
import pandas as pd
import pytest

from duodose.semireal_bundle import _pair_plan_identity, canonical_parent_pair, make_parent_disjoint_semireal_bundle


def test_canonical_parent_pair_is_order_independent() -> None:
    assert canonical_parent_pair("A", "B") == ("A", "B")
    assert canonical_parent_pair("B", "A") == ("A", "B")


def test_shared_plan_rejects_true_canonical_duplicate_pair() -> None:
    plan = pd.DataFrame(
        {
            "synthetic_cell_id": ["d0", "d1"],
            "split": ["train", "train"],
            "synthetic_subtype": ["heterotypic", "heterotypic"],
            "parent_1_id": ["A", "B"],
            "parent_2_id": ["B", "A"],
        }
    )
    with pytest.raises(ValueError, match="duplicate canonical unordered parent pair"):
        _pair_plan_identity(plan)


def test_parent_disjoint_bundle_has_zero_overlap_and_raw_sums(protocol_adata) -> None:
    work = protocol_adata.copy()
    work.obs["experimental_doublet"] = 0
    bundle = make_parent_disjoint_semireal_bundle(
        work,
        dataset="test",
        seed=0,
        n_singlets=120,
        n_train_homotypic_doublets=12,
        n_train_heterotypic_doublets=12,
        n_test_homotypic_doublets=6,
        n_test_heterotypic_doublets=6,
        n_clusters=4,
        test_parent_fraction=0.3,
        validation_parent_fraction=0.25,
        high_rna_quantile=0.9,
        saturation_range=(0.8, 1.0),
        min_cluster_size=3,
        construction_variant="raw_sum_parents_removed",
    )
    assert bundle.parent_audit["parent_leakage_audit_status"] == "passed"
    assert bundle.parent_audit["train_test_parent_overlap_fraction"] == 0.0
    assert bundle.construction_report["construction_variant"] == "raw_sum_parents_removed"
    parent_ids = set(bundle.parent_map["parent_1_id"]) | set(bundle.parent_map["parent_2_id"])
    assert parent_ids.isdisjoint(set(bundle.reference_cell_ids.astype(str)))

    first = bundle.parent_map.iloc[0]
    split = {"train": bundle.fit_adata, "validation": bundle.val_adata, "test": bundle.test_adata}[first["split"]]
    synthetic = np.asarray(split[first["synthetic_cell_id"], :].layers["counts"].toarray()).ravel()
    parent_1 = np.asarray(work[first["parent_1_id"], :].layers["counts"]).ravel()
    parent_2 = np.asarray(work[first["parent_2_id"], :].layers["counts"]).ravel()
    np.testing.assert_array_equal(synthetic, parent_1 + parent_2)
