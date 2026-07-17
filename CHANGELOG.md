# Changelog

## 0.1.0

- Added the public `DuoDose` class, `detect_doublets` convenience function, result object, and CLI.
- Frozen `DuoDose-ML-CalibratedRF-SafeFeatures` as the default public `rf` backend.
- Retained `DuoDose-DL-ConditionalMultiTaskMLP-SafeFeatures` as the sole internal `dl` ablation.
- Frozen raw parent-sum construction with exact parent removal, parent-disjoint splits, and fitted-reference SafeFeatures.
- Separated normal package usage from manuscript reproduction workflows.

## Final v2 manuscript contract

- Added the row-local `library_complexity_balance = log1p(nFeature) - 0.5 * log1p(nCount)` SafeFeature to the frozen calibrated-RF model.
- Replaced the manuscript-facing high-RNA FPR with the value at matched 50% homotypic recall.
- Retained fixed top-20%, matched 70%/80%, and historical true-doublet-budget FPRs as supplementary sensitivity analyses.
- Added complete per-run operating-point outputs, aggregation tables, cache migration with exact cell-ID validation, and revised completion contracts.
- Updated Figure 3, Table 1, supplementary operating-point tables, manifests, documentation, and the formal runbook.
- Removed the standalone fully controlled synthetic benchmark code, tests, configs, examples, reports, and generated outputs from the manuscript workflow.
## Documentation release cleanup

- Completed frozen RF and calibration metadata in Table S2 generation.
- Added explicit formal benchmark seeds 0–4 to generated protocol materials.
- Corrected Table S3 operating-point documentation.
- Indexed CSV components for Tables S2/S4/S5/S6 when optional XLSX authoring is unavailable.
- Added conservative domain-audit claim boundaries.

