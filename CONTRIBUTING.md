# Contributing

Thank you for helping improve DuoDose.

## Development setup

```powershell
conda env create -f environment.yml
conda activate duodose
python -m pip install -e ".[dev,manuscript]"
```

Run the lightweight suite before opening a change:

```powershell
python -m pytest tests
python -m compileall src reproducibility examples
```

## Scientific guardrails

- Do not use experimental doublet labels for fitting, feature construction, threshold selection, or model selection.
- Keep `raw_sum_parents_removed`, `fitted_reference`, and parent-disjoint construction unchanged unless a protocol revision is explicitly proposed and documented.
- Public backends are limited to `rf` (DuoDose) and `dl` (DuoDose-DL).
- Do not add external-method scores to SafeFeatures.
- Preserve exact method names and stable result schemas used by manuscript workflows.

Changes to model definitions, simulation, feature provenance, or evaluation semantics should include a focused test and an entry in `CHANGELOG.md`.

## Data and results

Do not commit raw datasets, RDS/H5AD files, large caches, model debug logs, or generated result directories. Add small manifests, checksums, and reproduction instructions instead. Never include private cell metadata in an issue or pull request.
