# Parameter-sensitivity metric definitions

This document defines the operating-point metrics used by the formal
parameter-sensitivity analysis. Both metrics are evaluated on the held-out
semi-real test split and use the continuous overall DuoDose score

```text
P(homotypic_doublet) + P(heterotypic_doublet).
```

Neither metric uses a fixed probability threshold, a 0.5 cutoff, multiclass
argmax, the training split, or the calibration split.

## `high_RNA_singlet_FPR`

- Population: every cell in the held-out semi-real test split.
- High-RNA subset: rows where `is_high_rna_singlet == True`.
- Selection budget: `K = number of true homotypic + heterotypic doublets in
  the test split`.
- Numerator: selected high-RNA singlets among the top `K` overall scores.
- Denominator: all high-RNA singlets in that test split.
- Formula: `selected high-RNA singlets / all test high-RNA singlets`.
- Threshold source: test scores and the true test-doublet prevalence.
- Direction: lower is better.
- Undefined cases: `NOT_AVAILABLE` with `NaN` when there are no high-RNA
  singlets or no true test doublets.
- Implementation: `high_rna_singlet_top_budget_fpr` in
  `src/duodose/semireal_metrics.py`.

The historical implementation maps non-finite scores to the bottom of the
ranking and uses NumPy `argsort`. It does not declare a cell-ID tie break.
This is recorded as a contract warning because the original saved outputs did
not record boundary-tie diagnostics.

## `high_RNA_singlet_FPR_at_expected_rate`

- Population: every cell in the held-out semi-real test split.
- High-RNA subset: rows whose controlled label is `high_RNA_singlet`.
- Selection budget: `K = round(expected_doublet_rate * n_test_cells)`.
- Numerator: selected high-RNA singlets among exactly the top `K` overall
  scores.
- Denominator: all high-RNA singlets in that test split.
- Formula: `selected high-RNA singlets / all test high-RNA singlets`.
- Threshold source: the test score distribution.
- Tie handling: score descending, then stable cell ID ascending.
- Direction: lower is better.
- Undefined cases: `NaN` when no high-RNA singlets are available.
- Implementation: `_candidate_fpr` in
  `reproducibility/run_parameter_sensitivity.py`, now delegated to
  `deterministic_top_fraction` in
  `src/duodose/parameter_sensitivity_audit.py`.

This metric is an exact rank-budget diagnostic. The public API instead obtains
an expected-rate score threshold and applies `score >= threshold`; ties at the
threshold can therefore produce more calls than the exact rank budget. That
difference must be retained when comparing these outputs with public API
binary calls.

## Continuous metrics

AUROC and AUPRC are calculated from continuous scores. The sensitivity
analysis fits one model for each dataset, seed, and semi-real size factor, then
reuses that same score vector for all expected-rate rows. Identical continuous
metrics across expected rates are therefore required by design.
