"""Input validation for public DuoDose workflows."""

from __future__ import annotations

import numpy as np


def validate_adata(adata, *, layer: str | None, expected_doublet_rate: float, device: str) -> None:
    try:
        from anndata import AnnData
    except ImportError as exc:
        raise ImportError("anndata is required to run DuoDose") from exc
    if not isinstance(adata, AnnData):
        raise TypeError("DuoDose expects an anndata.AnnData object")
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("AnnData must contain at least one cell and one gene")
    if adata.n_obs < 20:
        raise ValueError("DuoDose requires at least 20 cells")
    if adata.n_vars < 10:
        raise ValueError("DuoDose requires at least 10 genes")
    if not adata.obs_names.is_unique:
        raise ValueError("AnnData cell names must be unique")
    if layer is not None and layer not in adata.layers:
        raise KeyError(f"layer={layer!r} was not found in adata.layers")
    from scipy import sparse

    matrix = adata.X if layer is None else adata.layers[layer]
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if values.size == 0:
        raise ValueError("Count matrix is empty")
    if not np.isfinite(values).all():
        raise ValueError("Count matrix contains non-finite values")
    if np.any(values < 0):
        raise ValueError("Count matrix contains negative values")
    if not 0.0 < float(expected_doublet_rate) < 1.0:
        raise ValueError("expected_doublet_rate must be between 0 and 1")
    fractional = np.abs(values - np.round(values))
    if np.mean(fractional > 1e-6) > 0.05:
        raise ValueError("Input does not appear count-like; provide raw counts through adata.X or layer='counts'")
    if device == "cuda":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("device='cuda' requested but PyTorch is unavailable") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is unavailable")


def counts_copy(adata, *, layer: str | None):
    work = adata.copy()
    source = work.X if layer is None else work.layers[layer]
    work.X = source.copy()
    work.layers["counts"] = source.copy()
    return work
