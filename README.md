<p align="center">
  <img src="./assets/duodose-logo.png" alt="DuoDose" width="520">
</p>

# DuoDose

DuoDose is a homotypic-aware doublet detector for single-cell RNA sequencing. Its default method is a calibrated random forest trained on leakage-safe, fitted-reference features with a fixed high-RNA-singlet training weight of `2.0`; an optional conditional multitask neural-network ablation is available as `DuoDose-DL`.

## Why Homotypic Doublets Are Difficult

Heterotypic doublets often look like mixtures in cell-identity space. Homotypic doublets can remain identity-space inliers while showing abnormal RNA dosage or library complexity. DuoDose combines identity-mixture and dosage evidence, including a row-local library-complexity balance feature, then learns overall doublet probability, doublet subtype, and high-RNA rejection behavior jointly.

## Installation

From a local checkout:

```powershell
python -m pip install .
```

For development and manuscript reproduction:

```powershell
python -m pip install -e ".[dev,manuscript]"
```

The default `rf` backend is CPU compatible and does not require PyTorch. Install `.[dl]` plus a CUDA-enabled PyTorch build appropriate for the machine to use `DuoDose-DL` on a GPU. DoubletFinder, scDblFinder, and scds are optional paper-benchmark tools and are not required for normal prediction.

Confirm the installation and accelerator status with:

```powershell
duodose info
```

## Thirty-Second Quick Start

```python
from duodose import DuoDose

detector = DuoDose(
    expected_doublet_rate=0.08,
    random_state=0,
)
result = detector.fit_predict(adata)
result.add_to_adata(adata)
```

`adata` must be an `AnnData` object containing raw, non-negative counts in `adata.X`, or in a selected layer such as `adata.layers["counts"]`.

## Preparing Input Data

### AnnData input

DuoDose accepts an [`AnnData`](https://anndata.readthedocs.io/) object. Cells are rows and genes are columns. Cell names must be unique because scores and parent-disjoint audit records are aligned by `adata.obs_names`.

### Raw counts are required

The selected matrix must contain finite, non-negative, count-like values. Raw UMI counts are the expected input. Normalized, scaled, or log-transformed expression matrices are not accepted as count input because dosage and library-complexity features depend on the original count scale. If `adata.X` contains normalized or log-transformed values, keep those values in `adata.X` and point DuoDose to a raw-count layer.

Use `adata.X` when it already contains raw counts:

```python
detector = DuoDose(layer=None)
```

Use a layer when raw counts are stored separately:

```python
detector = DuoDose(layer="counts")
```

Verify the selected matrix before fitting:

```python
import numpy as np

counts = adata.layers["counts"]  # use adata.X when layer=None
values = counts.data if hasattr(counts, "data") and not isinstance(counts, np.ndarray) else np.asarray(counts)

assert counts.shape == (adata.n_obs, adata.n_vars)
assert np.isfinite(values).all()
assert (values >= 0).all()
assert adata.obs_names.is_unique
print(adata.n_obs, "cells x", adata.n_vars, "genes")
```

If the named layer is absent, DuoDose raises a clear `KeyError`. It never silently switches to another matrix.

### Size requirements

Input validation enforces at least 20 cells and 10 genes. Those are structural checks, not recommended analysis sizes. In practice, retain hundreds to thousands of informative genes after ordinary gene filtering; a matrix with only 10 genes is unlikely to support meaningful clustering or identity features. Semi-real training also needs more cells:

- `fast` requires at least 200 usable cells;
- `default` requires at least 200 usable cells;
- `robust` requires at least 300 usable cells.

More cells are strongly recommended so clustering can identify several adequately sized groups. Very small or homogeneous datasets may pass the basic matrix check but still fail semi-real construction because there are too few cells per cluster.

See [the detailed quickstart](docs/quickstart.md) and [parameter reference](docs/parameters.md) for additional preparation examples.

## Complete End-to-End Python Example

```python
from anndata import read_h5ad
from duodose import DuoDose

# The file contains normalized values in adata.X and raw counts in this layer.
adata = read_h5ad("input.h5ad")

detector = DuoDose(
    backend="rf",
    layer="counts",
    expected_doublet_rate=0.08,
    training_preset="default",
    device="cpu",
    random_state=0,
    threshold_strategy="expected_rate",
)

result = detector.fit_predict(adata)
result.add_to_adata(adata)

print(result.scores.head())
output_columns = [
    "duodose_score",
    "duodose_homotypic_score",
    "duodose_heterotypic_score",
    "duodose_prediction",
    "duodose_subtype",
    "duodose_subtype_confidence",
]
print(adata.obs[output_columns].head())

adata.write_h5ad("input_duodose.h5ad")
```

For a dataset whose raw counts are in `adata.X`, omit `layer="counts"` or set `layer=None`.

## Command-Line Usage

The CLI performs the same public workflow and writes the result columns into a new H5AD file:

```powershell
duodose run input.h5ad `
  --output input_duodose.h5ad `
  --layer counts `
  --backend rf `
  --expected-doublet-rate 0.08 `
  --training-preset default `
  --device cpu `
  --seed 0
```

Use `duodose run --help` for the installed CLI options. In the CLI, `--threshold-strategy none` corresponds to `threshold_strategy=None` in Python.

## How DuoDose Works

The normal prediction workflow is:

```text
observed count matrix
  -> preprocessing and clustering
  -> same-cluster homotypic and different-cluster heterotypic semi-real doublets
  -> parent-disjoint training and validation construction
  -> leakage-safe identity, dosage, neighborhood, and library-complexity features
     (including log1p(nFeature) - 0.5 * log1p(nCount))
  -> calibrated random forest (or the optional conditional multitask DL ablation)
  -> overall doublet, subtype, and high-RNA rejection outputs
  -> prediction on the original observed cells
```

Semi-real training is necessary because reliable experimental homotypic-doublet labels are usually unavailable. DuoDose therefore creates controlled homotypic and heterotypic states from the observed expression background. Same-cluster parent pairs model homotypic doublets, while different-cluster pairs model heterotypic doublets. Parent cells are separated across training and validation pools so the model cannot gain an advantage from seeing the same biological parent in both splits.

The model learns from SafeFeatures derived without using experimental doublet labels. One transformer is fitted once on the clean fit-split singlet reference and reused unchanged for validation and observed cells. Internal identity and dosage evidence are model inputs; they are not exposed as separate public methods. See [How DuoDose works](docs/method.md) for the technical details.

## Main Parameters

| Parameter | Default | Accepted values | Meaning |
|---|---:|---|---|
| `backend` | `"rf"` | `"rf"`, `"dl"` | `rf` is the public DuoDose method; `dl` is the sole internal ablation. |
| `layer` | `None` | `None` or an existing layer name | Count matrix source. `None` uses `adata.X`. |
| `expected_doublet_rate` | `0.08` | Float strictly between 0 and 1 | Expected fraction used by expected-rate thresholding. |
| `training_preset` | `"default"` | `"fast"`, `"default"`, `"robust"` | Semi-real training size and clustering preset. |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"cuda"` | Training device for neural-network backends. |
| `random_state` | `0` | Integer | Seed for sampling, splitting, clustering, and model initialization. |
| `threshold_strategy` | `"expected_rate"` | `"expected_rate"`, `"probability"`, `None` | Converts continuous scores to binary calls, or disables calls. |
| `threshold` | `0.5` | Float from 0 to 1 | Fixed cutoff used only when `threshold_strategy="probability"`. |

The constructor also supports `amp`, `dl_batch_size`, `dl_max_epochs`, `dl_patience`, and an advanced `config` object. Their exact defaults and interactions are documented in [Parameters](docs/parameters.md).

## Training Presets

Presets control the amount of semi-real training data. They do not change the score definitions.

| Preset | Semi-real background | Constructed training doublets | Clusters | Practical use |
|---|---:|---:|---:|---|
| `fast` | Up to 500 reference cells | 80 homotypic + 80 heterotypic | 6 | Smoke tests, iteration, limited memory |
| `default` | Up to 5,000 reference cells | 500 homotypic + 500 heterotypic | 12 | Normal analysis |
| `robust` | Up to 10,000 reference cells | 1,000 homotypic + 1,000 heterotypic | 16 | Larger, heterogeneous datasets and stability checks |

Runtime depends on cell count, gene count, backend, device, and early stopping. As a rough relative guide, `fast` is the baseline, `default` is usually several times more expensive, and `robust` can be roughly two to four times the cost of `default`. These are relative planning estimates, not wall-clock guarantees.

## Expected Doublet Rate

`expected_doublet_rate` is used by the default `"expected_rate"` threshold strategy to convert the continuous ranking into binary predictions. It does **not** alter `duodose_score`, `duodose_homotypic_score`, or `duodose_heterotypic_score`.

When the expected rate is uncertain, run a sensitivity analysis with several plausible rates and compare the binary calls. The continuous score ordering should remain the primary evidence for ranking candidates.

Use a fixed probability cutoff instead:

```python
detector = DuoDose(
    threshold_strategy="probability",
    threshold=0.5,
)
```

Or request continuous scores without binary or subtype calls:

```python
detector = DuoDose(threshold_strategy=None)
result = detector.fit_predict(adata)
candidates = result.scores.sort_values("duodose_score", ascending=False)
```

## Understanding the Results

Before adding results to AnnData, `result.scores` contains:

- `duodose_score` = `P(homotypic_doublet) + P(heterotypic_doublet)`
- `duodose_homotypic_score` = `P(homotypic_doublet)`
- `duodose_heterotypic_score` = `P(heterotypic_doublet)`
- `predicted_doublet`: binary call under the selected threshold strategy
- `predicted_subtype`: `homotypic_doublet` or `heterotypic_doublet`, assigned only to predicted doublets
- `subtype_confidence`: the winning subtype probability divided by total doublet probability, assigned only to predicted doublets

After `result.add_to_adata(adata)`, the last three columns are named `duodose_prediction`, `duodose_subtype`, and `duodose_subtype_confidence` in `adata.obs`.

Rank candidates by `duodose_score` for overall doublet evidence. Use the two subtype scores to understand which inferred state contributes to that evidence. A high homotypic-like score is especially useful for candidates that remain identity-space inliers but show dosage-related evidence.

Homotypic and heterotypic predictions are **model-inferred states**, not experimentally confirmed subtype labels. They should be interpreted alongside expression profiles, cell-type annotations, neighborhood context, quality-control measurements, and experimental design.

For all result fields and persistence helpers, see [Outputs](docs/outputs.md).

## Choosing a Backend

- `rf` is the default public **DuoDose** method. It uses the frozen calibrated random-forest SafeFeatures implementation and runs on CPU.
- `dl` is **DuoDose-DL**, the only retained internal ablation. It uses the conditional multitask neural network and can run on CPU or CUDA.

Logistic, plain MLP, Net, Hybrid, and DL-regularized historical variants are not public backends and are not part of manuscript-facing comparisons.

## Runtime And Hardware

DuoDose supports CPU execution. The default `rf` backend is CPU-only; `dl` can train on CPU or CUDA.

- `device="auto"` selects CUDA when PyTorch reports it available, otherwise CPU.
- `device="cpu"` always uses CPU.
- `device="cuda"` is strict. If CUDA is unavailable, DuoDose raises an error and does not silently fall back to CPU or another backend.
- `amp=True` enables mixed precision only for CUDA neural-network training.

For lower memory use, choose `training_preset="fast"`, reduce `dl_batch_size`, close other GPU processes, or use the default `backend="rf"` on CPU. The advanced configuration objects can reduce feature and semi-real training sizes; see [Parameters](docs/parameters.md).

## Troubleshooting

Common failures are summarized here; detailed remedies are in [Troubleshooting](docs/troubleshooting.md).

- **Invalid or normalized matrix:** supply raw non-negative counts, usually with `layer="counts"`.
- **Missing layer:** inspect `list(adata.layers.keys())` and correct `layer`.
- **CUDA unavailable:** use `device="auto"` or `device="cpu"`, or install a matching CUDA-enabled PyTorch build.
- **GPU out of memory:** use `fast`, lower `dl_batch_size`, or switch to `rf`.
- **Too few cells or clusters:** use a larger dataset, try `fast`, or verify that filtering did not remove most cells.
- **Missing dependencies:** reinstall the package and inspect `duodose info`; R tools are needed only for benchmark baselines.
- **Reproducibility differences:** fix `random_state`, package versions, backend, preset, and hardware; GPU kernels may not be bitwise identical.
- **Output permission errors:** write to a directory where the current user has permission and confirm that the target file is not locked by another program.

## Further Documentation

- [Quickstart](docs/quickstart.md)
- [Method and semi-real training](docs/method.md)
- [Parameters and advanced configuration](docs/parameters.md)
- [Outputs and interpretation](docs/outputs.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Reproducibility](docs/reproducibility.md)

## Reproducing Paper Workflows

Normal users do not need the manuscript workflows. They are isolated under `reproducibility/` and all read the frozen `reproducibility/configs/final_protocol.yaml`.

Parent-disjoint controlled semi-real benchmark:

```powershell
python reproducibility/run_controlled_benchmark.py --data-dir PATH_TO_DATA --datasets all --output-dir results/final_v1/controlled --conversion-dir results/final_v1/data --backends rf,dl --device auto --convert-rds --continue-on-error
```

Real doublet-enriched validation for one dataset:

```powershell
python reproducibility/run_real_validation.py --data-dir PATH_TO_DATA --dataset cline-ch --output-dir results/final_v1/real_validation/cline-ch/seed_0 --conversion-dir results/final_v1/data --backends rf,dl --external-methods Scrublet,scDblFinder,DoubletFinder,scds --device auto --convert-rds
```

The main manuscript benchmark is the parent-disjoint semi-real transfer analysis across formal seeds `0, 1, 2, 3, 4`; seed 0 remains the default application seed. Its primary high-RNA false-positive metric is evaluated at matched 50% homotypic recall; fixed 20% candidate-budget and historical true-doublet-budget values are supplementary. The real workflow evaluates experimentally detectable doublets and is not a strict homotypic ground-truth benchmark. Experimental labels are not used to fit the model. Domain matching reduced but did not eliminate semi-real-versus-experimental separability, so the audit is not evidence of indistinguishable domains. Data and generated results are not included in this repository. See [the complete reproduction guide](reproducibility/README.md) for domain-audit, runtime, sensitivity, and final-artifact commands.

## Citation

See `CITATION.cff`. 

## License

MIT License.
