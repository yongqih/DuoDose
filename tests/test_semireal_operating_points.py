import numpy as np
import pandas as pd

from duodose.semireal_metrics import (
    HIGH_RNA_METRIC_VERSION,
    high_rna_metric_bundle,
    high_rna_operating_point_metrics,
)


def _toy_data() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.Index([f"c{i}" for i in range(10)])
    labels = [
        "homotypic_doublet",
        "high_RNA_singlet",
        "homotypic_doublet",
        "clean",
        "clean",
        "homotypic_doublet",
        "high_RNA_singlet",
        "clean",
        "homotypic_doublet",
        "clean",
    ]
    obs = pd.DataFrame(
        {
            "true_label": labels,
            "is_high_rna_singlet": [label == "high_RNA_singlet" for label in labels],
        },
        index=index,
    )
    score = pd.Series(np.linspace(1.0, 0.1, len(index)), index=index)
    return obs, score


def test_primary_fpr_uses_matched_50pct_homotypic_recall() -> None:
    obs, score = _toy_data()
    bundle = high_rna_metric_bundle(
        obs,
        score,
        dataset="toy",
        source_dataset="toy",
        seed=0,
        method="DuoDose",
    )
    # The smallest prefix recovering 2 of 4 homotypic doublets contains c0,c1,c2.
    assert bundle["high_RNA_singlet_FPR"] == 0.5
    assert bundle["high_RNA_singlet_FPR_at_matched_50pct_homotypic_recall"] == 0.5
    assert bundle["primary_FPR_number_selected_cells"] == 3
    assert bundle["primary_FPR_actual_homotypic_recall"] == 0.5
    assert bundle["high_RNA_singlet_FPR_metric_version"] == HIGH_RNA_METRIC_VERSION


def test_fixed_candidate_budget_is_20pct_for_every_method() -> None:
    obs, score = _toy_data()
    table = high_rna_operating_point_metrics(
        obs,
        {"DuoDose": score, "Scrublet": score[::-1].set_axis(score.index)},
        dataset="toy",
        source_dataset="toy",
        seed=0,
    )
    fixed = table.loc[table["operating_point_type"].eq("fixed_candidate_fraction")]
    assert set(fixed["number_selected_cells"]) == {2}
    assert set(fixed["target_candidate_fraction"]) == {0.20}
    assert set(fixed["candidate_fraction_source"]) == {"frozen_20pct_candidate_budget"}
