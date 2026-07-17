from __future__ import annotations

import json
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import duodose.manuscript_materials as manuscript_materials
from duodose.manuscript_materials import (
    CATEGORICAL_TICK_ROTATION,
    HIGH_RNA_FPR_DEFINITION,
    MAIN_METHODS,
    _assert_row_axes_aligned,
    _document_model_hyperparameters,
    _frame_spec,
    _runtime_manuscript_frame,
    _set_categorical_xticks,
)


def test_main_method_scope_excludes_dl() -> None:
    assert MAIN_METHODS == ["DuoDose", "Scrublet", "scDblFinder", "DoubletFinder", "scds"]
    assert "DuoDose-DL" not in MAIN_METHODS


def test_high_rna_definition_is_matched_homotypic_recall() -> None:
    assert "at least 50%" in HIGH_RNA_FPR_DEFINITION
    assert "same homotypic recall" in HIGH_RNA_FPR_DEFINITION
    assert "lower is better" in HIGH_RNA_FPR_DEFINITION


def test_workbook_spec_preserves_numeric_types() -> None:
    spec = _frame_spec(pd.DataFrame({"method": ["DuoDose"], "mean": [0.75], "n": [15]}))
    assert spec["columns"] == ["method", "mean", "n"]
    assert spec["rows"] == [["DuoDose", 0.75, 15]]
    json.dumps(spec)


def test_clean_results_contract_is_generated_not_checked_in() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "reproducibility/generate_final_artifacts.py").is_file()
    assert (root / "reproducibility/generate_manuscript_materials.py").is_file()
    placeholder = root / "manuscript_materials/README.md"
    assert placeholder.is_file()
    assert "Regenerate" in placeholder.read_text(encoding="utf-8")


def test_categorical_tick_contract_is_exactly_45_degrees() -> None:
    fig, ax = plt.subplots()
    _set_categorical_xticks(ax, [0, 1], ["one", "two"])
    assert CATEGORICAL_TICK_ROTATION == 45
    assert all(tick.get_rotation() == 45 for tick in ax.get_xticklabels())
    assert all(tick.get_ha() == "right" for tick in ax.get_xticklabels())
    assert all(tick.get_rotation_mode() == "anchor" for tick in ax.get_xticklabels())
    plt.close(fig)


def test_composite_generator_has_no_figure_level_supertitles() -> None:
    source = inspect.getsource(manuscript_materials)
    assert ".suptitle(" not in source


def test_row_alignment_contract_checks_actual_axes_regions() -> None:
    fig = plt.figure()
    grid = fig.add_gridspec(1, 2)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    _assert_row_axes_aligned([axes])
    shifted = axes[1].get_position()
    axes[1].set_position([shifted.x0, shifted.y0 + 0.01, shifted.width, shifted.height])
    try:
        _assert_row_axes_aligned([axes])
    except RuntimeError as exc:
        assert "not vertically aligned" in str(exc)
    else:
        raise AssertionError("misaligned plotting areas were not rejected")
    plt.close(fig)


def test_runtime_manuscript_frame_excludes_memory_measurements() -> None:
    frame = pd.DataFrame(
        {
            "total_wall_clock_seconds_mean": [1.0],
            "peak_ram_bytes_mean": [2.0],
            "peak_gpu_memory_bytes_mean": [3.0],
        }
    )
    public = _runtime_manuscript_frame(frame)
    assert list(public.columns) == ["total_wall_clock_seconds_mean"]


def test_figure3_does_not_plot_redundant_recall_at_k() -> None:
    source = inspect.getsource(manuscript_materials._figure3)
    assert '"precision_at_K"' in source
    assert '"recall_at_K"' not in source
    assert "DuoDose advantage vs best external" in source

def test_rf_table_s2_hyperparameters_are_complete_and_explicit() -> None:
    frame = pd.DataFrame(
        {
            "public_method_name": ["DuoDose", "DuoDose-DL"],
            "method": ["DuoDose-ML-CalibratedRF-SafeFeatures", "DuoDose-DL-ConditionalMultiTaskMLP-SafeFeatures"],
            "backend": ["rf", "dl"],
            "estimator": ["CalibratedRF", "ConditionalMultiTaskMLP"],
            "lambda_highrna": [0.0, 0.5],
        }
    )
    documented = _document_model_hyperparameters(frame)
    rf = documented.iloc[0]
    assert rf["n_estimators"] == 240
    assert rf["max_features"] == "sqrt"
    assert rf["min_samples_leaf"] == 2
    assert rf["calibration_method"] == "sigmoid"
    assert "prefit" in rf["calibration_scheme"]
    assert str(rf["calibration_folds"]).startswith("NOT_APPLICABLE")
    assert rf["lambda_highrna"] == "NOT_APPLICABLE"
    dl = documented.iloc[1]
    assert dl["lambda_highrna"] == 0.5


def test_table_s3_manifest_text_matches_eight_operating_point_rows() -> None:
    source = inspect.getsource(manuscript_materials._supplementary_tables)
    assert "eight rows per method-run" in source
    assert "not a separate operating-point row" in source


def test_writing_index_lists_csv_components_and_domain_boundary() -> None:
    source = inspect.getsource(manuscript_materials._write_reports)
    assert "Table S2 CSV components" in source
    assert "Table S6 CSV components" in source
    assert "do not claim that semi-real and experimental domains are indistinguishable" in source

