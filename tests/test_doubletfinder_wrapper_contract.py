from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "benchmarks" / "external" / "run_doubletfinder.R"


def test_doubletfinder_uses_frozen_pk_and_official_pann_orientation() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert 'pK <- 0.09' in source
    assert 'pK_source <- "frozen_wrapper_parameter"' in source
    assert "paramSweep" not in source
    assert "larger values are more doublet-like" in source


def test_doubletfinder_selects_new_pann_and_aligns_by_barcode() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "new_pann_cols <- setdiff(all_pann_cols, stale_pann_cols)" in source
    assert "length(new_pann_cols) != 1" in source
    assert "seu@meta.data[cell_ids, pann_col" in source
    assert "identical(final_seurat_ids, cell_ids)" in source
    assert "one finite pANN score for every input barcode" in source
