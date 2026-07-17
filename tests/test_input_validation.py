import numpy as np
import pytest

from duodose.validation import validate_adata


def test_input_validation_accepts_counts(small_adata) -> None:
    validate_adata(small_adata, layer="counts", expected_doublet_rate=0.08, device="cpu")


def test_input_validation_rejects_negative_counts(small_adata) -> None:
    small_adata.X[0, 0] = -1
    with pytest.raises(ValueError, match="negative"):
        validate_adata(small_adata, layer=None, expected_doublet_rate=0.08, device="cpu")


def test_explicit_cuda_is_strict(small_adata, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        validate_adata(small_adata, layer=None, expected_doublet_rate=0.08, device="cuda")
