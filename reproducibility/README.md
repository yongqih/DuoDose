# Manuscript Reproduction

Normal users should use the `DuoDose` Python API or `duodose run`. This directory contains the frozen manuscript workflows. Every workflow reads `configs/final_protocol.yaml`, writes explicit failure/status files, and records its command, environment, seed, Git state, runtime, input provenance, and output manifest.

The canonical, CLI-verified formal commands and execution order are in [`FORMAL_RUNBOOK.md`](FORMAL_RUNBOOK.md). Run the complete formal workflow with `run_all_formal_analyses.ps1`; inspect result completion separately with `check_formal_completion.py`.

## Unified Validation Suite

The validation suite is a single command. It performs the canonical parent-disjoint construction, fitted-reference and inference invariance checks, RF and fixed-DL serialization checks, leakage/probability/metric/schema contracts, both fixed-split RF permutation controls, interrupted-run accounting, existing domain-audit contract inspection, and final report/figure generation.

Quick validation on the representative dataset:

```powershell
python reproducibility/run_validation_suite.py --config reproducibility/configs/final_validation_suite.yaml --data-dir data/xili_real/real_datasets --output-dir results/final_v1/validation_suite --mode quick --dataset cline-ch --device cuda --n-jobs 4
```

Formal validation, after benchmark artifacts are finalized:

```powershell
python reproducibility/run_validation_suite.py --config reproducibility/configs/final_validation_suite.yaml --data-dir data/xili_real/real_datasets --output-dir results/final_v1/validation_suite --existing-domain-audit-dir results/final_v1/domain_audit --formal-results-dir results/final_v1 --mode full --dataset cline-ch --device cuda --n-jobs 4 --overwrite
```

Use `--resume` to reuse a complete result set only when its configuration and implementation hashes match. Full mode automatically runs every applicable audit and generates `validation_suite_report.md`; no audit-specific command or separate report step is required.

## Data

Download `real_datasets.zip` from the [Xi and Li Zenodo record](https://zenodo.org/records/4062232), extract it outside the repository, and retain the original RDS filenames listed in `data/datasets.yaml`.

```powershell
python reproducibility/prepare_data.py `
  --data-dir PATH_TO_REAL_DATASETS `
  --output-dir results/final_v1/data `
  --datasets all `
  --convert-rds
```

The conversion uses `scripts/convert_xili_rds.R`. Set `R_SCRIPT` to an explicit `Rscript.exe` when it is not on `PATH`; standard Windows R installations are also discovered automatically.

## Controlled Semi-Real Benchmark

```powershell
python reproducibility/run_controlled_benchmark.py `
  --data-dir PATH_TO_REAL_DATASETS `
  --datasets all `
  --output-dir results/final_v1/controlled `
  --conversion-dir results/final_v1/data `
  --backends rf,dl `
  --device auto `
  --convert-rds `
  --continue-on-error
```

## External Methods

Install R dependencies once:

```powershell
Rscript reproducibility/environment/install_r_dependencies.R
```

Run one exact dataset/seed at a time so failures remain attributable:

```powershell
python reproducibility/run_external_benchmark.py `
  --data-dir PATH_TO_REAL_DATASETS `
  --dataset cline-ch `
  --seed 0 `
  --output-dir results/final_v1/external/cline-ch/seed_0 `
  --conversion-dir results/final_v1/data `
  --methods Scrublet,scDblFinder,DoubletFinder,scds `
  --convert-rds
```

Each method writes a status and message. A missing R package or failed wrapper is never converted into a successful row. Each controlled run also writes a full high-RNA operating-point table. The primary manuscript value is matched at 50% homotypic recall; top 20%, matched 70%/80%, and the historical true-doublet budget remain supplementary.

## Real-Data Application And Domain Audit

```powershell
python reproducibility/run_real_application.py `
  --data-dir PATH_TO_REAL_DATASETS `
  --datasets all `
  --output-dir results/final_v1/real_application `
  --conversion-dir results/final_v1/data `
  --external-methods Scrublet,DoubletFinder,scDblFinder,scds `
  --convert-rds

python reproducibility/run_domain_audit.py `
  --data-dir PATH_TO_REAL_DATASETS `
  --datasets all `
  --output-dir results/final_v1/domain_audit `
  --cache-dir results/final_v1/domain_audit_inputs `
  --conversion-dir results/final_v1/data `
  --convert-rds `
  --continue-on-error
```

The application figure is a qualitative shared-UMAP comparison. Its only internal method is calibrated-RF `DuoDose`; experimental labels are a post-hoc overlay and are not used by any model or embedding step.

The strict audit compares experimental doublets with held-out semi-real heterotypic doublets using raw mechanism features, parent-unique selection, parent-disjoint balanced folds, and logistic regression only.

## Runtime And Sensitivity

```powershell
python reproducibility/run_runtime_scaling.py --data-dir PATH_TO_REAL_DATASETS --dataset HMEC-orig-MULTI --output-dir results/final_v1/runtime --conversion-dir results/final_v1/data --protocol reproducibility/configs/final_protocol.yaml --cell-counts 5000,10000,20000,largest_feasible --methods DuoDose,DuoDose-DL,Scrublet,scDblFinder --repetitions 3 --device auto --n-jobs 4 --convert-rds

python reproducibility/run_parameter_sensitivity.py --data-dir PATH_TO_REAL_DATASETS --dataset cline-ch --output-dir results/final_v1/sensitivity --conversion-dir results/final_v1/data --convert-rds
```

## Final Artifacts

```powershell
python reproducibility/generate_final_artifacts.py --results-dir results/final_v1
```

Generated reports use “doublet-enriched datasets” and do not infer heterotypic parity or method superiority unless the clean numerical outputs support it.
