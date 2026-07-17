"""Minimal DuoDose Python API example."""

from anndata import read_h5ad

from duodose import DuoDose


adata = read_h5ad("input.h5ad")
result = DuoDose(backend="rf", expected_doublet_rate=0.08, device="cpu").fit_predict(adata)
result.add_to_adata(adata)
adata.write_h5ad("output.h5ad")
