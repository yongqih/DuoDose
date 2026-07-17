from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import duodose.real_application as application
from duodose import DuoDose
from reproducibility.run_real_application import ROOT, validate_output_root


def _coordinates(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {"umap_1": np.linspace(-1, 1, len(index)), "umap_2": np.sin(np.linspace(0, 2, len(index)))},
        index=index,
    )


def _scores(index: pd.Index) -> pd.DataFrame:
    overall = np.linspace(0.05, 0.95, len(index))
    q_hom = np.linspace(0.1, 0.9, len(index))
    return pd.DataFrame(
        {
            "duodose_score": overall,
            "duodose_homotypic_score": overall * q_hom,
            "duodose_heterotypic_score": overall * (1.0 - q_hom),
        },
        index=index,
    )


def test_clean_output_boundary_preserves_legacy_tree() -> None:
    legacy = ROOT.parent / "DuoDose"
    script = legacy / "examples" / "run_real_application_figure.py"
    before = script.read_bytes() if script.is_file() else b""
    with pytest.raises(ValueError, match="clean repository"):
        validate_output_root(legacy / "results" / "forbidden")
    after = script.read_bytes() if script.is_file() else b""
    assert before == after
    assert validate_output_root(ROOT / "results" / "allowed").is_relative_to(ROOT)


def test_main_figure_method_scope_is_rf_only() -> None:
    assert application.INTERNAL_FIGURE_METHODS == ("DuoDose",)
    assert application.EXTERNAL_FIGURE_METHODS == ("Scrublet", "DoubletFinder", "scDblFinder", "scds")
    source = inspect.getsource(application.plot_cross_method_umap)
    assert "DuoDose-DL" not in source
    for historical in ("DuoDose-identity", "DuoDose-dosage", "DuoDose-combined", "DuoDose-max", "Hybrid"):
        assert historical not in source


def test_public_rf_frozen_protocol() -> None:
    detector = DuoDose(backend="rf", training_preset="fast", layer="counts")
    assert detector.config.backend == "rf"
    assert detector.config.semireal.construction_variant == "raw_sum_parents_removed"
    assert detector.config.semireal.safe_feature_mode == "fitted_reference"
    assert detector.config.semireal.parent_disjoint is True


def test_masking_or_permuting_experimental_labels_cannot_change_scores(monkeypatch, small_adata) -> None:
    class FakeTransformer:
        def transform(self, adata, **_kwargs):
            assert adata.obs["experimental_doublet"].eq(0).all()
            return pd.DataFrame({"safe": np.asarray(adata.layers["counts"].sum(axis=1)).ravel()}, index=adata.obs_names)

    class FakeDuoDose:
        def __init__(self, **kwargs):
            assert kwargs["backend"] == "rf"
            self.safe_feature_transformer_ = FakeTransformer()

        def fit_predict(self, adata):
            assert adata.obs["experimental_doublet"].eq(0).all()
            values = np.asarray(adata.layers["counts"].sum(axis=1)).ravel()
            values = values / values.max()
            return SimpleNamespace(scores=_scores(adata.obs_names).assign(duodose_score=values), threshold=0.5)

    monkeypatch.setattr(application, "DuoDose", FakeDuoDose)
    variants = []
    for labels in (
        np.arange(small_adata.n_obs) % 2,
        np.arange(small_adata.n_obs)[::-1] % 2,
        None,
    ):
        adata = small_adata.copy()
        if labels is None:
            adata.obs.drop(columns=["experimental_doublet"], errors="ignore", inplace=True)
        else:
            adata.obs["experimental_doublet"] = labels
        _, result, _ = application.fit_public_rf_label_free(
            adata, expected_doublet_rate=0.08, random_state=0, training_preset="fast"
        )
        variants.append(result.scores)
    pd.testing.assert_frame_equal(variants[0], variants[1])
    pd.testing.assert_frame_equal(variants[0], variants[2])


def test_label_usage_and_parent_disjoint_audits() -> None:
    label = application.label_usage_audit().set_index("check")
    for check in (
        "experimental_label_used_in_training",
        "experimental_label_used_in_reference_selection",
        "experimental_label_used_in_threshold_selection",
        "experimental_label_used_in_embedding",
    ):
        assert bool(label.loc[check, "value"]) is False
    assert bool(label.loc["experimental_labels_joined_after_scores_frozen", "value"]) is True
    result = SimpleNamespace(
        backend="rf",
        feature_audit={"safe_feature_mode": "fitted_reference", "safe_feature_transformer_id": "t", "safe_feature_reference_pool_id": "r"},
        parent_audit={"parent_leakage_audit_status": "passed", "reference_parent_overlap_count": 0},
        model_metadata={"internal_method_name": "DuoDose-ML-CalibratedRF-SafeFeatures", "construction_variant": "raw_sum_parents_removed", "parent_disjoint": True},
    )
    assert application.reference_audit(SimpleNamespace(), result).iloc[0]["status"] == "PASS"


def test_shared_embedding_and_score_alignment() -> None:
    index = pd.Index([f"cell_{i}" for i in range(20)])
    coordinates = _coordinates(index)
    scores = pd.DataFrame({method: np.linspace(0, 1, len(index)) for method in application.SCORING_METHODS}, index=index)
    audit = application.shared_embedding_audit(coordinates, scores)
    assert len(audit) == 9
    assert audit["status"].eq("PASS").all()
    assert audit["coordinate_hash"].nunique() == 1
    assert audit["cell_id_hash"].nunique() == 1
    assert scores.index.equals(coordinates.index)


def test_candidate_classes_use_probability_contract_and_like_names() -> None:
    index = pd.Index(["non", "amb", "het", "hom"])
    scores = pd.DataFrame(
        {
            "duodose_score": [0.1, 0.9, 0.9, 0.9],
            "duodose_homotypic_score": [0.05, 0.45, 0.18, 0.72],
            "duodose_heterotypic_score": [0.05, 0.45, 0.72, 0.18],
        },
        index=index,
    )
    calls = application.candidate_calls(scores, overall_threshold=0.5, homotypic_threshold=0.6, heterotypic_threshold=0.4)
    assert calls["model_inferred_subtype_class"].tolist() == [
        "non_candidate", "subtype_ambiguous", "heterotypic_like", "homotypic_like"
    ]
    assert set(calls["model_inferred_subtype_class"]) == set(application.CANDIDATE_CLASSES)
    labels, summary = application.candidate_class_display_labels(calls, class_column="model_inferred_subtype_class")
    assert labels == {
        "non_candidate": "non candidate (n = 1; 25.0%)",
        "subtype_ambiguous": "subtype ambiguous (n = 1; 25.0%)",
        "heterotypic_like": "heterotypic-like (n = 1; 25.0%)",
        "homotypic_like": "homotypic-like (n = 1; 25.0%)",
    }
    assert all(item["count"] == 1 for item in summary.values())


def test_common_budget_candidate_classes_are_exact_stable_and_transparent() -> None:
    index = pd.Index([f"cell_{i:02d}" for i in range(12)], name="cell_id")
    raw_scores = pd.DataFrame(
        {
            "duodose_score": np.repeat(0.5, len(index)),
            "duodose_homotypic_score": np.array([0.25, 0.10, 0.40, 0.25, 0.10, 0.40] * 2),
            "duodose_heterotypic_score": np.array([0.25, 0.40, 0.10, 0.25, 0.40, 0.10] * 2),
        },
        index=index,
    )
    raw_calls = application.candidate_calls(raw_scores, overall_threshold=0.6, homotypic_threshold=0.6, heterotypic_threshold=0.4)
    original_scores = raw_scores.copy(deep=True)
    original_q = raw_calls[["duodose_q_homotypic_given_doublet", "duodose_q_heterotypic_given_doublet"]].copy(deep=True)
    displayed = application.apply_common_budget_candidate_classes(raw_calls, raw_scores["duodose_score"], top_k=5)
    selected = set(displayed.index[displayed["common_display_candidate"]])
    assert selected == {"cell_00", "cell_01", "cell_02", "cell_03", "cell_04"}
    assert int(displayed["common_display_candidate"].sum()) == 5
    assert displayed["duodose_common_budget_candidate_class"].ne("non_candidate").sum() == 5
    assert displayed["duodose_common_budget_candidate_class"].eq("non_candidate").sum() == len(index) - 5
    assert displayed["duodose_raw_model_candidate_class"].equals(raw_calls["model_inferred_subtype_class"].astype(str))

    permuted_ids = index[[7, 2, 11, 0, 5, 9, 1, 10, 4, 6, 3, 8]]
    permuted = application.apply_common_budget_candidate_classes(
        raw_calls.reindex(permuted_ids), raw_scores["duodose_score"].reindex(permuted_ids), top_k=5
    )
    assert set(permuted.index[permuted["common_display_candidate"]]) == selected

    smaller_budget = application.apply_common_budget_candidate_classes(raw_calls, raw_scores["duodose_score"], top_k=3)
    pd.testing.assert_frame_equal(raw_scores, original_scores)
    pd.testing.assert_frame_equal(
        smaller_budget[["duodose_q_homotypic_given_doublet", "duodose_q_heterotypic_given_doublet"]],
        original_q,
    )
    assert int(smaller_budget["common_display_candidate"].sum()) == 3

    budget = {
        "labeled_singlet_count": 7,
        "labeled_doublet_count": 5,
        "labeled_doublet_fraction": 5 / 12,
        "common_display_top_k": 5,
    }
    audit = application.candidate_display_audit("fixture", displayed, budget).iloc[0]
    assert bool(audit["candidate_class_sum_equals_k"]) is True
    assert int(audit["candidate_class_sum"]) == 5
    assert audit["status"] == "PASS"


def test_figure_uses_common_budget_class_column() -> None:
    source = inspect.getsource(application.plot_cross_method_umap)
    assert 'calls["duodose_common_budget_candidate_class"]' in source
    candidate_panel = source[source.index("candidate_labels") :]
    assert 'calls["model_inferred_subtype_class"]' not in candidate_panel


def test_common_display_budget_is_posthoc_and_deterministic() -> None:
    index = pd.Index([f"cell_{i:02d}" for i in range(10)])
    labels = pd.Series([0] * 8 + [1] * 2, index=index)
    budget = application.experimental_display_budget(labels, n_cells=len(index))
    assert budget["labeled_singlet_count"] == 8
    assert budget["labeled_doublet_count"] == 2
    assert budget["labeled_doublet_fraction"] == pytest.approx(0.2)
    assert budget["common_display_top_k"] == 2
    assert budget["display_budget_used_for_model_fitting"] is False
    scores = pd.DataFrame({method: np.arange(10, dtype=float) for method in (*application.EXTERNAL_FIGURE_METHODS, "DuoDose")}, index=index)
    masks = application.common_top_k_masks(scores, top_k=2)
    assert masks.sum().eq(2).all()
    assert masks.loc[["cell_08", "cell_09"]].all().all()


def test_exact_three_by_three_panel_contract() -> None:
    assert application.PANEL_ORDER == (
        "clusters_or_annotations",
        "experimental_singlet_doublet_labels",
        "Scrublet",
        "DoubletFinder",
        "scDblFinder",
        "scds",
        "DuoDose overall doublet probability",
        "DuoDose subtype evidence",
        "DuoDose candidate classes at common top-K budget",
    )
    assert len(application.PANEL_ORDER) == 3 * 3


def test_png_pdf_and_missing_external_scores_are_explicit(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    index = pd.Index([f"cell_{i}" for i in range(24)])
    coordinates = _coordinates(index)
    method_scores = pd.DataFrame(
        {
            "Scrublet": np.linspace(0, 1, len(index)),
            "DoubletFinder": np.nan,
            "scDblFinder": np.linspace(1, 0, len(index)),
            "scds": np.linspace(0.2, 0.8, len(index)),
            "DuoDose": np.linspace(0.05, 0.95, len(index)),
        },
        index=index,
    )
    status = pd.DataFrame(
        [
            {"method": method, "status": "failed" if method == "DoubletFinder" else "success", "message": "missing package" if method == "DoubletFinder" else ""}
            for method in ("DuoDose", *application.EXTERNAL_FIGURE_METHODS)
        ]
    )
    calls = application.candidate_calls(_scores(index), overall_threshold=0.5, homotypic_threshold=0.6, heterotypic_threshold=0.4)
    budget = application.experimental_display_budget(pd.Series(np.arange(len(index)) % 2, index=index), n_cells=len(index))
    masks = application.common_top_k_masks(method_scores, top_k=budget["common_display_top_k"])
    calls = application.apply_common_budget_candidate_classes(calls, method_scores["DuoDose"], top_k=budget["common_display_top_k"])
    png, pdf = tmp_path / "figure.png", tmp_path / "figure.pdf"
    application.plot_cross_method_umap(
        png,
        pdf,
        dataset="fixture",
        coordinates=coordinates,
        annotation=pd.Series(["cluster_0"] * len(index), index=index),
        annotation_name="clusters",
        experimental_labels=pd.Series(np.arange(len(index)) % 2, index=index),
        method_scores=method_scores,
        calls=calls,
        status=status,
        display_budget=budget,
        top_k_masks=masks,
    )
    assert png.stat().st_size > 0
    assert pdf.stat().st_size > 0
    assert status.loc[status["method"].eq("DoubletFinder"), "message"].iloc[0] == "missing package"


def test_percentiles_are_not_model_inputs() -> None:
    source = inspect.getsource(application)
    assert "percentile" not in source.lower()
    assert "quantile" not in inspect.getsource(application.candidate_calls).lower()


def test_manuscript_palette_and_font_contract() -> None:
    assert application.SCORE_PALETTE[0] == "#F7F7F7"
    assert application.SCORE_PALETTE[-1] == "#7F0000"
    assert application.SUBTYPE_PALETTE[0] == "#2166AC"
    assert application.SUBTYPE_PALETTE[-1] == "#B2182B"
    assert application.CANDIDATE_PALETTE["non_candidate"] == "#ECEFF1"
    assert application.INTENDED_FONT == "Arial"
