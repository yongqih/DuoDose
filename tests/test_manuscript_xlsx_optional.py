from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import duodose.manuscript_materials as manuscript


def _state(tmp_path: Path) -> manuscript.BuildState:
    output = tmp_path / "manuscript_materials"
    (output / "table_components").mkdir(parents=True)
    (output / "reports").mkdir(parents=True)
    return manuscript.BuildState(
        results_dir=tmp_path / "results",
        output_dir=output,
        repository_root=Path(__file__).resolve().parents[1],
    )


def test_xlsx_is_optional_when_node_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(tmp_path)
    monkeypatch.delenv("DUODOSE_NODE", raising=False)
    monkeypatch.setattr(manuscript.shutil, "which", lambda _: None)

    outputs = manuscript._build_workbooks(
        state,
        {},
        {"Display": pd.DataFrame({"metric": ["AUPRC"], "value": [0.9]})},
        require_xlsx=False,
    )

    assert outputs == []
    assert (state.output_dir / "table_components/workbook_spec.json").is_file()
    assert (state.output_dir / "reports/xlsx_generation_status.md").is_file()
    assert any(item["artifact"].endswith(".xlsx") for item in state.omitted)


def test_require_xlsx_fails_when_node_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(tmp_path)
    monkeypatch.delenv("DUODOSE_NODE", raising=False)
    monkeypatch.setattr(manuscript.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="Node.js was not found"):
        manuscript._build_workbooks(
            state,
            {},
            {"Display": pd.DataFrame({"metric": ["AUPRC"], "value": [0.9]})},
            require_xlsx=True,
        )
