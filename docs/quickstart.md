# Quickstart

This guide takes an H5AD file from raw counts to an annotated result. For method details, see [Method](method.md); for every public option, see [Parameters](parameters.md).

## Install and check DuoDose

From the repository root:

```powershell
python -m pip install .
duodose info
```

For an editable development installation:

```powershell
python -m pip install -e ".[dev]"
```

The default `rf` backend is CPU compatible and does not need PyTorch. The optional `dl` ablation needs PyTorch; install a CUDA-enabled build matching the local driver for GPU training.

## Check the input matrix

DuoDose expects an `AnnData` object with cells in rows and genes in columns. The selected matrix must contain finite, non-negative, count-like values. It must not contain normalized, scaled, or log-transformed expression.

Raw counts can be in either location:

- `adata.X`: construct `DuoDose(layer=None)`, which is the default.
- `adata.layers["counts"]`: construct `DuoDose(layer="counts")`.

An H5AD may safely retain normalized values in `adata.X` as long as DuoDose is pointed at a separate raw-count layer.

```python
from anndata import read_h5ad
import numpy as np

adata = read_h5ad("input.h5ad")
counts = adata.layers["counts"]
values = counts.data if hasattr(counts, "data") and not isinstance(counts, np.ndarray) else np.asarray(counts)

assert counts.shape == (adata.n_obs, adata.n_vars)
assert adata.obs_names.is_unique
assert np.isfinite(values).all()
assert (values >= 0).all()
print(f"{adata.n_obs:,} cells x {adata.n_vars:,} genes")
```

Basic matrix validation requires 20 cells and 10 genes, but the public parent-disjoint training workflow requires at least 200 usable cells (`robust` requires 300). These are still practical floors, not recommended study sizes. Retain hundreds to thousands of informative genes and several well-populated biological groups.

## Run DuoDose in Python

```python
from anndata import read_h5ad
from duodose import DuoDose

adata = read_h5ad("input.h5ad")

detector = DuoDose(
    backend="rf",
    layer="counts",
    expected_doublet_rate=0.08,
    training_preset="default",
    device="auto",
    random_state=0,
)
result = detector.fit_predict(adata)
result.add_to_adata(adata)

print(result.scores.sort_values("duodose_score", ascending=False).head(20))
adata.write_h5ad("input_duodose.h5ad")
```

Use `layer=None` if raw counts are already in `adata.X`.

The six added `adata.obs` columns are:

- `duodose_score`
- `duodose_homotypic_score`
- `duodose_heterotypic_score`
- `duodose_prediction`
- `duodose_subtype`
- `duodose_subtype_confidence`

See [Outputs](outputs.md) before interpreting subtype calls.

## Run DuoDose from the CLI

```powershell
duodose run input.h5ad `
  --output input_duodose.h5ad `
  --layer counts `
  --backend rf `
  --expected-doublet-rate 0.08 `
  --training-preset default `
  --device auto `
  --seed 0
```

Run `duodose run --help` to inspect the options supported by the installed version.

## Use continuous scores only

Binary calls are optional. To rank cells without applying a threshold:

```python
detector = DuoDose(layer="counts", threshold_strategy=None)
result = detector.fit_predict(adata)
ranked = result.scores.sort_values("duodose_score", ascending=False)
```

With thresholding disabled, `predicted_doublet`, `predicted_subtype`, and `subtype_confidence` are left unassigned. The three probability-derived scores remain available.

## Next steps

- Read [Method](method.md) for the semi-real training workflow and leakage controls.
- Read [Parameters](parameters.md) to choose a preset, backend, device, or threshold.
- Read [Troubleshooting](troubleshooting.md) if input validation or model training fails.
- Read [Reproducibility](reproducibility.md) before comparing runs.
