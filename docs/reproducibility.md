# Reproducibility

## Record the analysis contract

At minimum, record:

- DuoDose version and source revision;
- Python, NumPy, pandas, scikit-learn, AnnData, PyTorch, and CUDA versions;
- backend and training preset;
- `random_state`;
- count layer name and input-file checksum;
- expected doublet rate and threshold strategy;
- CPU or GPU model and operating system.

The result object retains the effective configuration and training metadata:

```python
print(result.config)
print(result.training_summary)
print(result.model_metadata)
print(result.feature_audit)
print(result.parent_audit)
```

Serialize these dictionaries with the surrounding analysis metadata. The annotated H5AD alone does not capture every software and hardware detail.

## Random seeds

`random_state` seeds semi-real sampling, parent-disjoint splitting, clustering, and supported model initialization. Use the same integer for matched backend comparisons.

A fixed seed improves repeatability but does not measure stability. For a scientific comparison, run a predeclared set of seeds and report the distribution of metrics or rank agreement.

## CPU and GPU differences

CPU runs with the same software stack and seed should be closely reproducible. Different BLAS implementations, thread counts, or library releases can still produce small numeric differences.

CUDA runs can vary across GPU models, drivers, PyTorch builds, and nondeterministic kernels. AMP changes floating-point precision and can also change the optimization path. Preserve `device`, GPU name, CUDA version, and `amp` in the run record. Use tolerances and ranking stability for cross-hardware checks rather than byte-identical outputs.

Explicit `device="cuda"` raises an error when CUDA is unavailable. This avoids silently producing a CPU run under a GPU-labeled command.

## Input provenance

Preserve the raw-count source and preprocessing history. Record whether counts came from `adata.X` or a named layer and verify that cell and gene ordering did not change between scoring and downstream analysis.

A practical checksum record in PowerShell is:

```powershell
Get-FileHash input.h5ad -Algorithm SHA256
```

Store the checksum next to the configuration, not only in a transient terminal log.

## Expected-rate sensitivity

The default expected-rate threshold converts a continuous ranking into a fixed number of binary calls. It does not alter the continuous score. When the capture rate is uncertain, report results across a plausible range and distinguish changes in calls from changes in score ordering.

For a threshold-free analysis, use `threshold_strategy=None` and preserve all continuous scores.

The formal sensitivity metric contracts and completed-run audit are documented
in [parameter_sensitivity_metric_definitions.md](parameter_sensitivity_metric_definitions.md)
and [parameter_sensitivity_audit.md](parameter_sensitivity_audit.md).


## High-RNA FPR operating points

The primary manuscript FPR is the label-relative high-RNA singlet FPR at matched 50% homotypic recall. The supplementary standardized budget selects exactly 20% of each test set for every method and dataset. The historical budget with K equal to the number of constructed test doublets is retained only for continuity because K/N differs across datasets.

## Matched backend comparisons

Use the same input matrix, parent-disjoint split seed, preset, expected rate, and evaluation cells when comparing the default `rf` method with the `dl` ablation. Do not tune a backend on experimental evaluation labels.

The public result audits help verify that parents remained disjoint and unsafe features were excluded. Treat a failed parent or feature audit as a failed run, not as a result eligible for selection.

## Normal use versus paper reproduction

Normal users should use the Python API or `duodose run`. The scripts under `reproducibility/` reproduce the parent-disjoint semi-real benchmark, qualitative real-data application, domain audit, runtime analysis, sensitivity analysis, and manuscript artifacts. They can be much slower or require optional external tools.

Benchmark commands, datasets, and generated results are deliberately separate from the normal API documentation. In the real-data application, experimental labels are a descriptive overlay joined only after scores and the shared embedding are frozen. They do not supply homotypic/heterotypic ground truth and are not required for AUROC/AUPRC completion metrics.

The complete command sequence, external R setup, data manifest, and final artifact generator are documented in [`reproducibility/README.md`](../reproducibility/README.md).

## Environment snapshot

Capture a package snapshot after a successful run:

```powershell
python -m pip freeze > duodose_environment.txt
duodose info
```

For long-lived analyses, keep the environment file, run configuration, input checksum, and output H5AD together.

## Validation status boundaries

An already completed strict domain audit can be checked without rerunning it by passing `--existing-domain-audit-dir <path>` to `reproducibility/run_validation_suite.py`. The suite imports only contract and summary information; it does not copy raw audit caches. If the path is omitted the check is explicitly `NOT_RUN`; an incomplete configured directory is `INCOMPLETE`.

Formal benchmark execution status is independent of code completion. Interrupted, unavailable, and skipped rows remain visible in `run_status_audit.csv`; they do not turn a passing software validation into a claim that manuscript analyses are complete.
