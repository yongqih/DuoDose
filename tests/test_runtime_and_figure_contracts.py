from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from duodose.plotting_style import apply_manuscript_style, audit_figure_style_contract
from duodose.runtime_completeness import CANONICAL_RUNTIME_METHODS, build_runtime_method_completeness_audit


def _runtime(methods: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"method": methods, "status": "success", "n_cells": 100, "total_wall_clock_seconds": 1.0})


def test_every_runtime_method_is_plotted_or_explicitly_accounted_for() -> None:
    audit = build_runtime_method_completeness_audit(
        _runtime(list(CANONICAL_RUNTIME_METHODS[:4])),
        expected_methods=CANONICAL_RUNTIME_METHODS,
        requested_methods=CANONICAL_RUNTIME_METHODS[:4],
        expected_successful_rows=1,
    )
    assert (audit["plotted"] | audit["omission_reason"].str.len().gt(0)).all()


def test_missing_runtime_method_cannot_disappear_silently() -> None:
    audit = build_runtime_method_completeness_audit(
        _runtime(["DuoDose"]), expected_methods=CANONICAL_RUNTIME_METHODS, requested_methods=["DuoDose"]
    )
    row = audit.set_index("method").loc["scds"]
    assert row["status"] == "NOT_RUN" and row["omission_reason"]


def test_runtime_aliases_do_not_cause_method_loss() -> None:
    audit = build_runtime_method_completeness_audit(
        _runtime(["DuoDose-ML-CalibratedRF-SafeFeatures", "scdblfinder", "doubletfinder"]),
        expected_methods=CANONICAL_RUNTIME_METHODS,
        requested_methods=["DuoDose", "scDblFinder", "DoubletFinder"],
    ).set_index("method")
    assert audit.loc[["DuoDose", "scDblFinder", "DoubletFinder"], "plotted"].all()


def test_incomplete_runtime_grid_still_plots_valid_measurements() -> None:
    audit = build_runtime_method_completeness_audit(
        _runtime(["DoubletFinder"]),
        expected_methods=CANONICAL_RUNTIME_METHODS,
        requested_methods=CANONICAL_RUNTIME_METHODS,
        expected_successful_rows=3,
    ).set_index("method")
    assert audit.loc["DoubletFinder", "status"] == "INCOMPLETE"
    assert bool(audit.loc["DoubletFinder", "plotted"])


def test_formal_plotting_modules_apply_shared_arial_style() -> None:
    root = Path(__file__).resolve().parents[1]
    audit = audit_figure_style_contract(root)
    assert audit["shared_style_applied"].all()
    assert audit["requested_font"].eq("Arial").all()


def test_conflicting_local_font_override_is_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import duodose.plotting_style as style

    path = tmp_path / "bad.py"
    path.write_text("apply_manuscript_style()\nfont.family = 'Times New Roman'\n", encoding="utf-8")
    monkeypatch.setattr(style, "FORMAL_PLOTTING_ENTRY_POINTS", (("bad", "bad.py"),))
    audit = style.audit_figure_style_contract(tmp_path)
    assert bool(audit.iloc[0]["conflicting_override_found"])
    assert audit.iloc[0]["contract_status"] == "FAIL"


def test_png_and_pdf_export_with_shared_style(tmp_path: Path) -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    apply_manuscript_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    for suffix in ("png", "pdf"):
        fig.savefig(tmp_path / f"figure.{suffix}")
        assert (tmp_path / f"figure.{suffix}").stat().st_size > 0
    plt.close(fig)
