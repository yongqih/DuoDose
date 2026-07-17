# Outputs

`DuoDose.fit_predict` returns a `DuoDoseResult`. It keeps scores, threshold information, audits, configuration, and model metadata together.

## Continuous scores

The three public continuous scores are:

```text
duodose_score =
    P(homotypic_doublet) + P(heterotypic_doublet)

duodose_homotypic_score =
    P(homotypic_doublet)

duodose_heterotypic_score =
    P(heterotypic_doublet)
```

`duodose_score` is the primary overall ranking. A high homotypic score indicates dosage- or complexity-supported doublet evidence without requiring an identity mixture. A high heterotypic score indicates evidence more consistent with a mixed identity state.

Homotypic and heterotypic scores describe model-inferred states. They are not experimentally confirmed subtype labels.

## Columns in `result.scores`

| Column | Type | Interpretation |
|---|---|---|
| `duodose_score` | Float | Overall doublet probability, equal to the sum of subtype probabilities. |
| `duodose_homotypic_score` | Float | Inferred homotypic-doublet probability. |
| `duodose_heterotypic_score` | Float | Inferred heterotypic-doublet probability. |
| `predicted_doublet` | Nullable boolean | Thresholded overall call, or unassigned when thresholding is disabled. |
| `predicted_subtype` | String/category-like | Winning subtype for predicted doublets; unassigned for other cells. |
| `subtype_confidence` | Float | Winning subtype probability divided by total doublet probability; assigned only to predicted doublets. |

`subtype_confidence` is conditional on the model's doublet-like evidence. A cell can have high subtype confidence but a modest overall score, so always inspect it together with `duodose_score`.

## Columns added to AnnData

```python
result.add_to_adata(adata)
```

adds:

| `adata.obs` column | Source |
|---|---|
| `duodose_score` | `result.scores["duodose_score"]` |
| `duodose_homotypic_score` | `result.scores["duodose_homotypic_score"]` |
| `duodose_heterotypic_score` | `result.scores["duodose_heterotypic_score"]` |
| `duodose_prediction` | `result.scores["predicted_doublet"]` |
| `duodose_subtype` | `result.scores["predicted_subtype"]` |
| `duodose_subtype_confidence` | `result.scores["subtype_confidence"]` |

The index is aligned to `adata.obs_names`. Unique cell names are required to make that alignment unambiguous.

## Rank and review candidates

```python
ranked = result.scores.sort_values("duodose_score", ascending=False)
print(ranked.head(50))
```

The continuous ranking is useful when a hard expected doublet rate is unavailable. Candidate review can combine this ranking with cell type annotations and quality-control fields:

```python
review = adata.obs[
    [
        "cell_type",
        "total_counts",
        "duodose_score",
        "duodose_homotypic_score",
        "duodose_heterotypic_score",
        "duodose_subtype",
    ]
].sort_values("duodose_score", ascending=False)
```

Replace `cell_type` and `total_counts` with columns available in the dataset.

## Work without thresholding

```python
detector = DuoDose(threshold_strategy=None)
result = detector.fit_predict(adata)
```

The score columns are populated, while prediction and subtype call fields are unassigned. This is appropriate for downstream ranking, custom decision rules, or sensitivity analysis.

## Result metadata and audits

Useful `DuoDoseResult` attributes include:

- `threshold`: the resolved numeric cutoff, or `None` when thresholding is disabled;
- `backend`: the selected public backend key;
- `training_summary`: backend, device, optimization, and fit status information;
- `feature_audit`: included features and excluded unsafe features;
- `parent_audit`: parent-disjoint split checks;
- `config`: serialized settings used for the run;
- `model_metadata`: public and internal backend metadata.

Inspect these fields before archiving a run:

```python
print(result.threshold)
print(result.training_summary)
print(result.feature_audit)
print(result.parent_audit)
print(result.config)
```

## Save an annotated H5AD

```python
result.add_to_adata(adata)
adata.write_h5ad("input_duodose.h5ad")
```

Save to a new path until the analysis has been checked. H5AD serialization preserves the added `adata.obs` columns. Store the package version and run configuration separately or in analysis metadata for reproducibility.
