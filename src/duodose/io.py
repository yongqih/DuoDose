"""Small public I/O helpers."""

from __future__ import annotations

from pathlib import Path


def read_h5ad(path: str | Path):
    from anndata import read_h5ad as _read_h5ad

    return _read_h5ad(path)


def write_result_h5ad(result, adata, path: str | Path) -> None:
    result.write_h5ad(adata, path)
