# DuoDose Formal Manuscript Runbook

This runbook is for Windows PowerShell opened at the `DuoDose_github` repository root. It describes the frozen clean rerun, not normal package use. Formal result completion is separate from code completion and must be established with `check_formal_completion.py`.

## Frozen Contract

- Main internal method: `DuoDose` (`rf`, implemented by `DuoDose-ML-CalibratedRF-SafeFeatures`).
- RF training assigns ordinary singlets weight `1.0` and constructed high-RNA singlets the single frozen weight `2.0`.
- The final SafeFeature allowlist includes the row-local `library_complexity_balance = log1p(nFeature) - 0.5 * log1p(nCount)` feature.
- The main-text label-relative high-RNA FPR is measured at matched 50% homotypic recall. Fixed top-20%, matched 70%/80%, and the historical true-doublet-budget FPR are supplementary.
- Internal ablation: `DuoDose-DL` (`dl`, implemented by `DuoDose-DL-ConditionalMultiTaskMLP-SafeFeatures`).
- External methods: `Scrublet`, `scDblFinder`, `DoubletFinder`, and `scds`.
- Construction: `raw_sum_parents_removed`, `fitted_reference`, parent-disjoint.
- Real collections are **doublet-enriched datasets**. Experimental labels are evaluation-only and are not complete homotypic/heterotypic ground truth.
- The domain audit is the strict lightweight logistic-regression audit. It has no domain-label permutations or Random Forest path.
- Validation permutations are the 100 subtype-label and 100 full-label controls specified by `reproducibility/configs/final_validation_suite.yaml`.

No formal command below invokes Logistic, plain MLP, Net, Hybrid, DL-regularized, or other historical internal variants.

## Prerequisites

Create or update the Conda environment:

```powershell
conda env create -f environment.yml
conda activate duodose
```

For an existing environment, use:

```powershell
conda env update -f environment.yml --prune
conda activate duodose
python -m pip install -e ".[dev,manuscript]"
```

Install the R dependencies used by the current wrappers:

```powershell
Rscript reproducibility/environment/install_r_dependencies.R
```

This installs/verifies `Matrix`, `Seurat`, `SingleCellExperiment`, `SummarizedExperiment`, `scDblFinder`, `scds`, and `DoubletFinder`; package versions are written to `reproducibility/environment/r_package_versions.csv`. Set `R_SCRIPT` to `Rscript.exe` when R is not on `PATH`.

Verify Python, R, and CUDA:

```powershell
python -c "import anndata,numpy,pandas,scipy,sklearn,scrublet,torch,yaml; print('Python dependencies: OK'); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
Rscript -e "packages <- c('Matrix','Seurat','SingleCellExperiment','SummarizedExperiment','scDblFinder','scds','DoubletFinder'); print(sapply(packages, packageVersion))"
```

`-Device cuda` fails when CUDA is unavailable. `-Device auto` permits CPU fallback. CUDA is strongly recommended for every command that fits or validates `DuoDose-DL`.

## Input Data

Download `real_datasets.zip` from the [Xi and Li Zenodo record](https://zenodo.org/records/4062232) and extract the original RDS files into one directory. The current machine has the inputs at this repository-relative sibling path:

```text
..\DuoDose\data\xili_real\real_datasets\
  cline-ch.rds
  HEK-HMEC-MULTI.rds
  hm-12k.rds
  hm-6k.rds
  HMEC-orig-MULTI.rds
  HMEC-rep-MULTI.rds
  J293t-dm.rds
  mkidney-ch.rds
  nuc-MULTI.rds
  pbmc-1A-dm.rds
  pbmc-1B-dm.rds
  pbmc-1C-dm.rds
  pbmc-2ctrl-dm.rds
  pbmc-2stim-dm.rds
  pbmc-ch.rds
  pdx-MULTI.rds
```

The formal YAML configures 15 datasets and explicitly excludes `J293t-dm` because it cannot meet the frozen parent-disjoint minimum.

Prepare, convert, checksum, and manifest every configured dataset:

```powershell
python reproducibility/prepare_data.py `
  --data-dir ..\DuoDose\data\xili_real\real_datasets `
  --output-dir results\final_v1\data `
  --protocol reproducibility\configs\final_protocol.yaml `
  --datasets all `
  --convert-rds
```

Expected outputs are `dataset_discovery_manifest.csv`, `dataset_input_manifest.csv`, converted dataset bundles, and `data_preparation_manifest.json`. The command fails if any configured formal dataset or prepared input is missing.

## Execution Order

| Order | Analysis | Typical runtime | Hardware |
|---:|---|---|---|
| 1 | Data preparation | minutes to an hour | CPU, R |
| 2 | Controlled internal benchmark | many hours to days | CUDA strongly recommended |
| 3 | Controlled external baselines | many hours | CPU, R |
| 4 | Real-data application and cross-method UMAP comparison | many hours | CPU, R |
| 5 | Strict domain audit | hours | CPU |
| 6 | Full validation suite | several hours | CUDA strongly recommended |
| 7 | Runtime and scalability | several hours | CPU and CUDA timing environments |
| 8 | Parameter sensitivity | several hours | CPU |
| 9 | Final aggregation | minutes | CPU |

Runtime depends strongly on dataset sizes, R package startup, GPU model, storage, and CPU count.

## Controlled Benchmark

The internal and external parts are separate stages because the external methods have independent status and failure records.

Internal `DuoDose` and `DuoDose-DL`, all 15 configured datasets and seeds 0-4:

```powershell
python reproducibility/run_controlled_benchmark.py `
  --data-dir ..\DuoDose\data\xili_real\real_datasets `
  --datasets all `
  --output-dir results\final_v1\controlled `
  --conversion-dir results\final_v1\data `
  --protocol reproducibility\configs\final_protocol.yaml `
  --backends rf,dl `
  --device cuda `
  --amp `
  --dl-max-epochs 200 `
  --dl-patience 20 `
  --convert-rds `
  --continue-on-error `
  --resume
python reproducibility/check_formal_completion.py --results-dir results\final_v1 --protocol reproducibility\configs\final_protocol.yaml --stage controlled_benchmark --strict
```

External baselines are unchanged by `library_complexity_balance`. After the internal controlled rerun, an existing external score cache can be migrated to the revised operating-point contract without refitting external methods. The migration accepts a cache only when its cell IDs match the fresh controlled test rows exactly:

```powershell
python reproducibility/recompute_semireal_operating_points.py `
  --controlled-dir results\final_v1\controlled `
  --external-dir results\final_v1\external `
  --protocol reproducibility\configs\final_protocol.yaml `
  --external-only `
  --skip-missing `
  --update-manifests
```

The master script performs this migration automatically under `-Resume`; missing or ambiguous caches are rerun normally.

External baselines, all configured dataset/seed combinations:

```powershell
$datasets = @('cline-ch','HEK-HMEC-MULTI','hm-12k','hm-6k','HMEC-orig-MULTI','HMEC-rep-MULTI','mkidney-ch','nuc-MULTI','pbmc-1A-dm','pbmc-1B-dm','pbmc-1C-dm','pbmc-2ctrl-dm','pbmc-2stim-dm','pbmc-ch','pdx-MULTI')
foreach ($dataset in $datasets) {
  foreach ($seed in 0..4) {
    python reproducibility/run_external_benchmark.py `
      --data-dir ..\DuoDose\data\xili_real\real_datasets `
      --dataset $dataset --seed $seed `
      --output-dir "results\final_v1\external\$dataset\seed_$seed" `
      --conversion-dir results\final_v1\data `
      --protocol reproducibility\configs\final_protocol.yaml `
      --methods Scrublet,scDblFinder,DoubletFinder,scds `
      --convert-rds
    if ($LASTEXITCODE -ne 0) { throw "External benchmark failed: $dataset seed $seed" }
    $status = @(Import-Csv "results\final_v1\external\$dataset\seed_$seed\external_method_status.csv")
    if ($status.Count -ne 4 -or @($status | Where-Object { $_.status.ToLowerInvariant() -ne 'success' }).Count) { throw "Required external method failed: $dataset seed $seed" }
  }
}
python reproducibility/check_formal_completion.py --results-dir results\final_v1 --protocol reproducibility\configs\final_protocol.yaml --stage external_baselines --strict
```

Per-run metric files contain AUROC, overall/homotypic/heterotypic/macro AUPRC, homotypic-versus-high-RNA AUPRC, the primary matched-50%-homotypic-recall high-RNA FPR, supplementary fixed-20% and matched-70%/80% FPRs, the historical true-doublet-budget FPR, precision@K, recall@K, status, and failure message. Each run also writes a full deterministic operating-point table. External failures remain explicit and make formal completion fail.

## Real-Data Application And Cross-Method UMAP Comparison

This manuscript-facing stage is qualitative, descriptive, and diagnostic. It uses only the public calibrated-RF `DuoDose` backend internally. Experimental singlet/doublet labels are joined after scores, candidate calls, clustering, PCA, and UMAP are frozen; they are an overlay and are not used for fitting, reference selection, thresholds, embeddings, or external methods. Model-inferred homotypic-like and heterotypic-like calls are not experimental subtype ground truth.

```powershell
python reproducibility/run_real_application.py `
  --data-dir ..\DuoDose\data\xili_real\real_datasets `
  --datasets all `
  --output-dir results\final_v1\real_application `
  --conversion-dir results\final_v1\data `
  --protocol reproducibility\configs\final_protocol.yaml `
  --external-methods Scrublet,DoubletFinder,scDblFinder,scds `
  --convert-rds `
  --continue-on-error
python reproducibility/check_formal_completion.py --results-dir results\final_v1 --protocol reproducibility\configs\final_protocol.yaml --stage real_data_application --strict
```

Each configured dataset uses seed 0 and writes the exact 3 x 3 PNG used for manuscript aggregation, raw aligned method scores, candidate calls, label/reference/shared-embedding audits, diagnostics, and manifests under `real_application/<dataset>/seed_0`. No real-label AUROC/AUPRC and no DuoDose-DL output are required by this stage.

The version-controlled manuscript contracts are `configs/final_table_manifest.json` and `configs/final_figure_manifest.json`. Final aggregation writes resolved manifests beside the generated tables and figures.

## Strict Domain Audit

One command prepares missing canonical bundles, validates shared transformer/reference provenance, and runs every configured dataset:

```powershell
python reproducibility/run_domain_audit.py `
  --data-dir ..\DuoDose\data\xili_real\real_datasets `
  --datasets all `
  --output-dir results\final_v1\domain_audit `
  --cache-dir results\final_v1\domain_audit_inputs `
  --conversion-dir results\final_v1\data `
  --protocol reproducibility\configs\final_protocol.yaml `
  --max-cells-per-domain 2000 `
  --convert-rds `
  --continue-on-error `
  --resume
python reproducibility/check_formal_completion.py --results-dir results\final_v1 --protocol reproducibility\configs\final_protocol.yaml --stage domain_audit --strict
```

The implementation enforces `raw_sum_parents_removed`, `fitted_reference`, held-out semi-real heterotypic doublets, raw mechanism features, deterministic parent-unique filtering, balanced parent-disjoint folds with zero fold-parent overlap, matched/unmatched/technical-only analyses, and logistic regression. No domain-label permutation option exists.

## Unified Validation Suite

Run this only after the strict domain audit exists so its contract can be checked:

```powershell
python reproducibility/run_validation_suite.py `
  --config reproducibility\configs\final_validation_suite.yaml `
  --data-dir ..\DuoDose\data\xili_real\real_datasets `
  --output-dir results\final_v1\validation_suite `
  --existing-domain-audit-dir results\final_v1\domain_audit `
  --formal-results-dir results\final_v1 `
  --mode full `
  --dataset cline-ch `
  --device cuda `
  --n-jobs 4 `
  --resume
```

Full mode runs all required parent, invariance, serialization, deterministic rerun, leakage, probability, metric, schema, run-status, domain-contract, and permutation-control audits. The frozen YAML supplies 100 subtype-label permutations and 100 full-label permutations. It generates its figures, manifest, and `validation_suite_report.md` in the same command.

Use `--overwrite` instead of `--resume` to deliberately replace a validation result whose run/implementation hash differs.

## Runtime And Scalability

The frozen dataset is `HMEC-orig-MULTI`; scales are 5,000, 10,000, 20,000, and the largest feasible size, deduplicated when a requested scale equals the largest feasible size. Each method/scale is repeated three times.

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
python reproducibility/run_runtime_scaling.py `
  --data-dir ..\DuoDose\data\xili_real\real_datasets `
  --dataset HMEC-orig-MULTI `
  --output-dir results\final_v1\runtime `
  --conversion-dir results\final_v1\data `
  --protocol reproducibility\configs\final_protocol.yaml `
  --cell-counts 5000,10000,20000,largest_feasible `
  --methods DuoDose,DuoDose-DL,Scrublet,scDblFinder,DoubletFinder,scds `
  --repetitions 3 `
  --device cuda --amp --n-jobs 4 `
  --convert-rds
```

`runtime_scaling_by_run.csv` records loading/preprocessing, semi-real construction, SafeFeature construction, model training, prediction, total wall-clock time, status, and message. Manuscript-facing summaries use elapsed-time fields only. `run_manifest.json` records hardware, requested device, AMP, logical CPUs, requested threads, relevant environment variables, and fixed method-level job settings.

`runtime_method_completeness_audit.csv` accounts for all six formal methods. A method without valid measurements remains explicit as `FAILED`, `UNAVAILABLE`, `INCOMPLETE`, or `NOT_RUN`; it is never silently dropped from the runtime contract or given fabricated values.

All manuscript-facing plotters call `duodose.plotting_style.apply_manuscript_style()`. The Arial-first plotting contract and its static/generated-output audit are documented in `docs/figure_style_contract.md`.

## Parameter Sensitivity

The compact frozen RF analysis uses seeds 0-2, expected rates 0.05/0.10/0.15, and semi-real size factors 0.5/1.0/2.0:

```powershell
python reproducibility/run_parameter_sensitivity.py `
  --data-dir ..\DuoDose\data\xili_real\real_datasets `
  --dataset cline-ch `
  --output-dir results\final_v1\sensitivity `
  --conversion-dir results\final_v1\data `
  --protocol reproducibility\configs\final_protocol.yaml `
  --convert-rds
```

Cluster count is fixed at 12 in the formal protocol and is **not** a configured sensitivity dimension. This analysis does not select a model or threshold on test results.

## Main-Text Figure Plan

- Figure 1: manual concept and method schematic.
- Figure 2: controlled semi-real overall, homotypic, heterotypic, and macro-subtype performance.
- Figure 3: homotypic-versus-high-RNA separation, matched-50%-homotypic-recall high-RNA FPR, precision at the common true-doublet budget, and paired DuoDose advantage.
- Figure 4: `cline-ch` cross-method real-application UMAP.
- Figure 5: robustness and practicality.


## Final Aggregation

This single command recursively aggregates controlled, external, real, domain, runtime, and sensitivity outputs; writes method/analysis status ledgers; and regenerates every final table and figure defined by the clean artifact generator:

```powershell
python reproducibility/generate_final_artifacts.py `
  --results-dir results\final_v1 `
  --output-dir results\final_v1
```

Final tables are under `results\final_v1\tables`, final figures under `results\final_v1\figures`, and the conservative generated report under `results\final_v1\reports`.

## Manuscript Materials

After final aggregation, build the publication-facing PNG figures, CSV tables, optional XLSX workbooks, provenance manifests, and writing index with one lightweight command:

```powershell
python reproducibility/generate_manuscript_materials.py `
  --results-dir results\final_v1 `
  --output-dir manuscript_materials `
  --overwrite
```

This command reads only clean cached outputs under `results\final_v1`; it does not fit models or rerun scientific analyses. Figure 1 is intentionally excluded, all generated figures are PNG-only, and the finalized cline-ch 3 × 3 UMAP is reused with standardized panel labels. CSV tables and per-sheet CSV components are always generated. XLSX workbook authoring is optional: when Node.js or `@oai/artifact-tool` is unavailable, the command completes normally and records the omission in `manuscript_materials\reports\xlsx_generation_status.md`. Add `--require-xlsx` only when XLSX output must be mandatory. Use `--main-only` or `--supplement-only` for focused rebuilds, and reserve `--skip-missing-noncritical` for explicitly recorded optional supplementary omissions.

The manuscript-material stage is deliberately downstream of the formal analysis master script so publishing changes cannot alter or invalidate the frozen scientific run.

## One-Command Formal Run

On the current machine:

```powershell
powershell -ExecutionPolicy Bypass -File reproducibility\run_all_formal_analyses.ps1 `
  -DataDir ..\DuoDose\data\xili_real\real_datasets `
  -OutputDir results\final_v1 `
  -Device cuda `
  -NJobs 4 `
  -Resume
```

The master script uses the frozen YAML, verifies Python/R/CUDA preconditions, prints every command, preserves timestamped logs, validates every required method, and stops when a command or output contract fails.

## Progress And Live Monitoring

Progress is enabled by default. Interactive terminals use `tqdm`; redirected output uses periodic plain text without ANSI control sequences. The master command reports all 8 formal stages, while benchmark children report the configured dataset, seed, and method hierarchy. `DuoDose-DL` prints one line per epoch; `--verbose-progress` additionally enables batch-level messages. Long child and R processes emit PID heartbeats every 45 seconds when their normal logs are quiet.

The shared machine-readable files are updated atomically after meaningful events:

```text
results\final_v1\runtime_ledger.csv
results\final_v1\formal_progress.json
```

From another PowerShell window, inspect the latest structured status:

```powershell
Get-Content results\final_v1\formal_progress.json -Raw | ConvertFrom-Json | Format-List
```

Follow the most recent master log:

```powershell
$log = Get-ChildItem results\final_v1\logs\*.log | Sort-Object LastWriteTime | Select-Object -Last 1
Get-Content $log.FullName -Wait
```

Inspect completed, cached, failed, or interrupted work units:

```powershell
Import-Csv results\final_v1\runtime_ledger.csv | Group-Object analysis_stage,status | Sort-Object Name | Format-Table Count,Name
```

The child CLIs consistently accept `--progress`, `--no-progress`, `--progress-refresh-seconds`, and `--verbose-progress`. The PowerShell master equivalents are `-Progress`, `-NoProgress`, `-ProgressRefreshSeconds`, and `-VerboseProgress`. Normal redirected runs should leave progress enabled: they automatically choose plain-text messages instead of terminal control codes.

Representative redirected output is intentionally concise:

```text
[Stage 2/8] controlled_benchmark | RUNNING | elapsed 00:18:42 | overall ETA 03:41:20
[Dataset 3/15] hm-12k
[Seed 2/5] 1
[Method 2/2] DuoDose-DL | ETA 00:27:15 (same_method_similar_size_rolling_median)
DL epoch 14/200 | train=0.41320 | val=0.43891 | validation_AUPRC=0.88210 | best=0.88403 | patience=2/20
[scDblFinder | hm-12k] still running | PID 18420 | elapsed 00:12:30 | last log activity 00:00:18 ago
```

With `-Resume`, output contracts are scanned before execution. Valid units are recorded as `SKIPPED_VALID_CACHE`, invalid or incomplete units remain eligible for rebuilding, and compatible completed history supplies rolling-median ETAs. Cached durations are not added to the current run's elapsed time.

## Completion Check

Inspect status without claiming that code completion implies result completion:

```powershell
python reproducibility/check_formal_completion.py `
  --results-dir results\final_v1 `
  --protocol reproducibility\configs\final_protocol.yaml
```

Require every category to be complete:

```powershell
python reproducibility/check_formal_completion.py `
  --results-dir results\final_v1 `
  --protocol reproducibility\configs\final_protocol.yaml `
  --strict
```

The checker writes `formal_completion_status.csv` and `formal_completion_status.json` and reports `COMPLETE`, `INCOMPLETE`, `FAILED`, or `NOT_RUN` for controlled internal results, external baselines, the real-data application, domain audit, validation suite, runtime, sensitivity, final tables, and final figures.

## Resume And Overwrite

- `run_all_formal_analyses.ps1 -Resume` skips only outputs that pass their formal contract. Controlled and domain scripts also support direct `--resume`; the real-data application is validated from its per-dataset audits and combined ledgers.
- Runtime and sensitivity resume at stage granularity because their current direct CLIs do not expose partial-grid resume.
- Validation `--resume` requires matching configuration and implementation hashes.
- `run_all_formal_analyses.ps1 -Overwrite` deletes only known stage directories below the selected output directory, then reruns them. It is mutually exclusive with `-Resume`.
- External and real direct commands do not have a resume flag; use the master script for safe per-run resume.

## Common Failures

- **Configured formal datasets were not discovered**: check the exact data root and filenames; do not point `--data-dir` at a single RDS file.
- **Rscript was not found**: install R, add it to `PATH`, or set `R_SCRIPT` to `Rscript.exe`.
- **Missing R packages**: rerun `Rscript reproducibility/environment/install_r_dependencies.R` and inspect `r_package_versions.csv`.
- **CUDA requested but unavailable**: install a CUDA-enabled PyTorch build/driver or deliberately use `-Device cpu`; the latter is much slower for DL stages.
- **Required external method failed**: inspect the exact dataset/seed log and `external_method_status.csv`. Required methods are never silently omitted.
- **Dataset too small for frozen parent-disjoint construction**: only the protocol-declared `J293t-dm` exclusion is accepted. A new failure requires review, not automatic threshold relaxation.
- **Domain bundle provenance mismatch**: remove only the affected cache directory or use master `-Overwrite`; never bypass transformer/reference validation.
- **Validation resume hash mismatch**: use validation `--overwrite` or master `-Overwrite` only when a fresh run is intended.
- **Completion is INCOMPLETE after a successful command**: inspect the named missing/invalid unit in `formal_completion_status.csv`; a process exit code alone is not evidence of formal completion.

## Progress bridge hotfix (v2.1)

The master PowerShell runner now omits blank optional arguments when calling
`progress_stage.py`. This prevents PowerShell from turning an empty
`--log-path ""` value into a bare `--log-path` flag. The Python bridge also
accepts a valueless blankable option defensively.

If an older checkout reports `progress_stage.py: error: argument --log-path:
expected one argument`, overwrite these two files from the v2.1 package and
rerun the same command with `-Resume`:

- `reproducibility/run_all_formal_analyses.ps1`
- `reproducibility/progress_stage.py`

No scientific output is invalidated by this orchestration-only failure; it
occurs before the first formal analysis stage starts.


### Windows transient progress-file lock recovery

The progress snapshot is operational telemetry only. Version 2.2 retries
atomic replacement of `formal_progress.json` when Windows Defender, indexing,
or a file preview briefly locks the destination; if the lock persists, the
snapshot update is skipped with a warning instead of failing RF/DL training.

For a run that previously failed only with `[WinError 5] Access is denied` on
`formal_progress.json`, keep all existing outputs and rerun the master command
with `-Resume`. The controlled benchmark will reuse the 74 valid runs and
rebuild only the failed dataset/seed unit.
## Release documentation contract

Before tagging a public release or submitting the manuscript, verify the following documentation-only items after manuscript-material generation:

1. Table S2 records the frozen RF settings: 240 trees, `max_features=sqrt`, `min_samples_leaf=2`, balanced-subsample class weights, sigmoid calibration, and a held-out prefit calibration scheme. RF-inapplicable DL fields must read `NOT_APPLICABLE`.
2. The formal controlled benchmark is explicitly documented as seeds `0, 1, 2, 3, 4`; seed 0 remains the default application seed.
3. Table S3 operating-point documentation states that the operating-point file has eight rows per method-run. The historical true-doublet-budget FPR is a column in the full benchmark table, not a ninth operating-point row.
4. When optional XLSX generation is skipped, `reports/manuscript_writing_index.md` must enumerate every CSV component for Tables S2, S4, S5, and S6.
5. Domain-audit language must say that matching reduced, but did not eliminate, separability. Do not use “indistinguishable.”

These checks require regeneration of manuscript materials only. They do not require rerunning model fitting, external methods, the controlled benchmark, or real-data inference.

