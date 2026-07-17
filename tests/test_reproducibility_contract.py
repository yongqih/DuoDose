from pathlib import Path

import pandas as pd
import pytest

from reproducibility.generate_final_artifacts import ALLOWED_METHODS, _assert_methods
from reproducibility.lib.common import discover_dataset_manifest


def test_dataset_discovery_uses_bundle_directory_not_sidecar_stems(tmp_path: Path) -> None:
    bundle = tmp_path / "cline-ch"
    bundle.mkdir()
    for name in ("matrix.mtx", "genes.tsv", "barcodes.tsv", "labels.tsv", "metadata.tsv"):
        (bundle / name).touch()
    manifest = discover_dataset_manifest(tmp_path)
    assert manifest["dataset"].tolist() == ["cline-ch"]
    assert manifest.iloc[0]["discovery_status"] == "valid"
    assert not set(manifest["dataset"]).intersection({"matrix", "genes", "barcodes", "labels", "metadata"})


def test_final_artifacts_allow_only_frozen_methods() -> None:
    assert ALLOWED_METHODS == ["DuoDose", "DuoDose-DL", "Scrublet", "scDblFinder", "DoubletFinder", "scds"]


def test_manuscript_ledger_rejects_deprecated_internal_method() -> None:
    with pytest.raises(ValueError, match="non-frozen methods"):
        _assert_methods(pd.DataFrame({"method": ["DuoDose-Hybrid"], "AUROC": [0.5]}))
