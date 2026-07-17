# Parameters

This page documents the current public `DuoDose` constructor. Run `help(DuoDose)` or inspect `DuoDose().get_params()` in the installed version when pinning a production workflow.

## Constructor

```python
DuoDose(
    backend="rf",
    expected_doublet_rate=0.08,
    random_state=0,
    device="auto",
    *,
    layer=None,
    threshold_strategy="expected_rate",
    threshold=0.5,
    training_preset="default",
    amp=False,
    dl_batch_size=None,
    dl_max_epochs=200,
    dl_patience=20,
    config=None,
)
```

## Main parameters

| Parameter | Default | Accepted values | Behavior |
|---|---:|---|---|
| `backend` | `"rf"` | `"rf"`, `"dl"` | Selects the frozen main method or its sole internal ablation. |
| `layer` | `None` | `None` or an existing layer name | `None` reads `adata.X`; a string reads `adata.layers[layer]`. |
| `expected_doublet_rate` | `0.08` | Float strictly between 0 and 1 | Expected fraction used only by expected-rate thresholding. |
| `training_preset` | `"default"` | `"fast"`, `"default"`, `"robust"` | Controls semi-real sample sizes and clustering. |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"cuda"` | Device for neural-network training. Explicit CUDA is strict. |
| `random_state` | `0` | Integer | Seed for sampling, splitting, clustering, and supported model initialization. |
| `threshold_strategy` | `"expected_rate"` | `"expected_rate"`, `"probability"`, `None` | Selects expected-rate calls, a fixed cutoff, or no calls. |
| `threshold` | `0.5` | Float from 0 to 1 | Cutoff used only by `threshold_strategy="probability"`. |

## Training presets

| Setting | `fast` | `default` | `robust` |
|---|---:|---:|---:|
| Maximum singlet background | 500 | 5,000 | 10,000 |
| Training homotypic doublets | 80 | 500 | 1,000 |
| Training heterotypic doublets | 80 | 500 | 1,000 |
| Validation homotypic doublets | 20 | 125 | 250 |
| Validation heterotypic doublets | 20 | 125 | 250 |
| Requested clusters | 6 | 12 | 16 |
| Minimum cluster size | 5 | 10 | 15 |
| Minimum usable singlets | 200 | 200 | 300 |

- `fast` is intended for smoke tests, iteration, and constrained memory.
- `default` is the normal analysis preset and balances runtime with training diversity.
- `robust` increases semi-real sample sizes and cluster resolution for larger heterogeneous datasets and stability analysis.

Presets do not change the definitions of the output scores. Runtime is data-dependent. As a rough planning guide, `default` is usually several times more expensive than `fast`, and `robust` can cost roughly two to four times as much as `default`.

## Backends

### `rf`

The default public DuoDose method. It uses `DuoDose-ML-CalibratedRF-SafeFeatures`, runs on CPU, and does not require PyTorch. Its formal high-RNA-singlet negative sample weight is fixed at `2.0` (ordinary singlets retain weight `1.0`); this is not a tunable public parameter. The final SafeFeature set includes the row-local `library_complexity_balance = log1p(nFeature) - 0.5 * log1p(nCount)` feature; it is computed from the same raw-count cell and does not use benchmark labels.

Frozen RF and calibration settings used in the formal benchmark:

| Setting | Value |
|---|---|
| Trees | 240 |
| Criterion | `gini` |
| `max_features` | `sqrt` |
| `min_samples_leaf` | 2 |
| Bootstrap | `True` |
| Class weight | `balanced_subsample` |
| Parallel jobs | `-1` |
| Calibration method | sigmoid |
| Calibration scheme | prefit calibration on the complete held-out validation split |
| Calibration folds | `NOT_APPLICABLE` (no cross-validation folds are used for the prefit calibrator) |

Neural-network-only quantities such as hidden width, dropout, epoch count, and multitask loss weights are `NOT_APPLICABLE` to the RF backend.

### `dl`

The sole retained internal ablation, publicly reported as DuoDose-DL. It uses `DuoDose-DL-ConditionalMultiTaskMLP-SafeFeatures`; CUDA acceleration and AMP are supported.

Both backends return the same complete public score schema. Historical Logistic, plain MLP, Net, Hybrid, and DL-regularized variants are not accepted backend values.

## Thresholding

### Expected-rate threshold

With `threshold_strategy="expected_rate"`, the top fraction specified by `expected_doublet_rate` is called doublet. The expected rate changes only binary and subtype calls; it does not retrain the model or alter the continuous scores.

Sensitivity analysis is appropriate when the expected capture rate is uncertain:

```python
for rate in (0.04, 0.08, 0.12):
    detector = DuoDose(expected_doublet_rate=rate, random_state=0)
    result = detector.fit_predict(adata)
```

Each loop fits a new detector. For a strict comparison of thresholds on one fitted score vector, retain the continuous scores and apply the desired ranking cutoffs downstream.

### Probability threshold

With `threshold_strategy="probability"`, cells whose `duodose_score` meets the fixed `threshold` are called doublet.

### No threshold

With `threshold_strategy=None`, DuoDose returns continuous scores and leaves prediction and subtype fields unassigned. The CLI spelling is `--threshold-strategy none`.

## Device and deep-learning controls

| Parameter | Default | Behavior |
|---|---:|---|
| `device` | `"auto"` | Uses CUDA when PyTorch reports it available; otherwise CPU. |
| `amp` | `False` | Enables automatic mixed precision for CUDA neural-network training. |
| `dl_batch_size` | `None` | Uses the backend's automatic batch-size choice. Set a positive integer to override it. |
| `dl_max_epochs` | `200` | Maximum neural-network epochs. Early stopping may finish sooner. |
| `dl_patience` | `20` | Validation epochs tolerated without improvement before early stopping. |

`device="cuda"` never silently falls back. It raises an error if PyTorch or CUDA support is unavailable. `amp=True` has an effect only for a CUDA neural-network backend.

Lowering `dl_batch_size` is the first adjustment for GPU memory pressure. The dataset's feature matrix and semi-real data also consume memory, so `training_preset="fast"` can have a larger effect.

## Advanced configuration

`config` accepts a `DuoDoseConfig` object for advanced, version-coupled control of semi-real construction, features, and deep-learning settings. When supplied, it is the authoritative configuration; the convenience constructor values are not merged into it.

```python
from duodose import DuoDose, DuoDoseConfig

config = DuoDoseConfig(
    backend="rf",
    layer="counts",
    expected_doublet_rate=0.08,
    random_state=0,
    device="cpu",
    training_preset="default",
)
detector = DuoDose(config=config)
```

Advanced configuration classes are part of the public package but are more tightly coupled to a DuoDose release than the main constructor. Save `result.config` with an analysis and pin the package version.

## Convenience function

`detect_doublets` constructs a detector and calls `fit_predict`:

```python
from duodose import detect_doublets

result = detect_doublets(
    adata,
    backend="rf",
    layer="counts",
    expected_doublet_rate=0.08,
    random_state=0,
)
```
