# Troubleshooting

## Invalid, normalized, or log-transformed count matrix

**Symptoms:** validation reports negative, non-finite, or non-count-like values; dosage behavior appears implausible.

**Cause:** the selected matrix contains normalized, scaled, log-transformed, or otherwise processed expression instead of raw counts.

**Actions:**

1. Inspect `adata.X` and `adata.layers`.
2. Locate the raw non-negative count matrix.
3. Set `layer="counts"` or the actual raw-count layer name.
4. Confirm shape, finiteness, non-negativity, and count-like values before fitting.

Sparse matrices expose stored values through `.data`; dense matrices can be checked with `numpy.asarray`.

## Missing layer

**Symptom:** `KeyError` reports that the requested count layer does not exist.

```python
print(list(adata.layers.keys()))
```

Correct the spelling and capitalization of `layer`, create the intended raw-count layer upstream, or use `layer=None` only when `adata.X` truly contains raw counts. DuoDose does not silently substitute another matrix.

## Cell names are not unique

**Symptom:** validation reports duplicate observation names.

Cell names must be unique for unambiguous score alignment. If duplicates are accidental, make them unique before fitting and preserve the mapping to original identifiers:

```python
adata.obs["original_cell_id"] = adata.obs_names.astype(str)
adata.obs_names_make_unique()
```

## CUDA unavailable

**Symptom:** `device="cuda"` raises an error.

Explicit CUDA is strict and never falls back to CPU. Check the installation:

```powershell
duodose info
```

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
```

Install a CUDA-enabled PyTorch build compatible with the local driver, or use `device="auto"` or `device="cpu"`. The default `rf` backend is the CPU-friendly manuscript method.

## GPU out of memory

**Symptoms:** PyTorch reports CUDA out of memory during feature transfer or training.

Try, in order:

1. Reduce `dl_batch_size`.
2. Use `training_preset="fast"`.
3. Close other GPU processes and rerun in a fresh Python process.
4. Disable `amp` only when diagnosing numerical issues; AMP usually reduces CUDA memory use.
5. Use `backend="rf"` on CPU when GPU memory remains insufficient.

Changing the preset changes training sample sizes, so record the change when comparing analyses.

## Too few cells or clusters

**Symptoms:** validation rejects the input size, clustering produces too few usable groups, or semi-real pair construction cannot find enough parents.

Basic input validation requires 20 cells and 10 genes. Practical preset minima are 40, 200, and 300 usable cells for `fast`, `default`, and `robust`, respectively. A dataset can exceed these totals but remain unsuitable if filtering leaves one very small or nearly homogeneous population.

Check that cells are rows, filtering did not remove most observations, and multiple expected biological groups remain. Use `fast` for small pilot data. Do not combine biologically unrelated samples merely to satisfy a numeric minimum without considering batch structure.

## Missing optional dependencies

**Symptoms:** imports fail for PyTorch, AnnData, scikit-learn, or benchmark-only R methods.

Reinstall the project in the active environment and run `duodose info`. Confirm that the `python`, `pip`, and `duodose` commands resolve to the same environment:

```powershell
python -c "import sys; print(sys.executable)"
python -m pip show duodose
Get-Command duodose
```

DoubletFinder, scDblFinder, and scds are optional paper-benchmark dependencies and are not required for normal DuoDose prediction.

## Reproducibility differences

Fix `random_state`, backend, preset, threshold settings, package versions, and hardware. GPU kernels and different PyTorch/CUDA releases may not be bitwise identical. Compare rankings and metrics with reasonable numeric tolerances instead of requiring byte-identical floating-point output across hardware.

Archive `result.config`, `result.training_summary`, the DuoDose version, and the input-file checksum. See [Reproducibility](reproducibility.md).

## Output file permissions

**Symptoms:** `write_h5ad` reports permission denied, access denied, a read-only path, or a locked file.

Write to a directory owned by the current user and choose a new output filename. Close applications that may hold the H5AD file open. Confirm enough free disk space exists, especially because H5AD writes may create a substantial temporary file. Avoid overwriting the only input copy until the annotated output has been validated.

## Training failure without a clear input error

Inspect the full traceback and rerun with `training_preset="fast"` to distinguish a resource problem from a data problem. Record `duodose info`, Python version, package versions, matrix shape, sparse/dense format, backend, preset, and device when reporting an issue. Do not include private cell metadata or expression data in a public issue.
