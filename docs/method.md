# Method

## The detection problem

A doublet contains RNA from two captured cells. Heterotypic doublets combine sufficiently different cell states and often appear between populations in expression space. Homotypic doublets combine similar cells and can remain inside the identity distribution, making identity-based detection alone insufficient.

DuoDose combines identity-mixture evidence with dosage, neighborhood, and library-complexity evidence. Its default backend is the frozen calibrated random forest; `DuoDose-DL` is the sole retained conditional multitask neural-network ablation.

## Why semi-real training is required

Reliable experimental labels for homotypic doublets are usually unavailable. Experimental demultiplexing and cross-species designs preferentially identify distinguishable or heterotypic events, while same-state pairs may remain unlabeled.

DuoDose therefore does not fit the public model to experimental labels. Instead, it creates controlled homotypic and heterotypic states from the expression background of the observed dataset. This adapts training to the dataset's own count depth, cell states, and variability while preserving a known construction label for each artificial pair.

The generated subtype labels describe the semi-real construction. Predictions on observed cells remain model-inferred states, not experimentally confirmed subtype labels.

## End-to-end workflow

```text
observed count matrix
  -> preprocessing and clustering
  -> same-cluster homotypic and different-cluster heterotypic semi-real doublets
  -> parent-disjoint training and validation construction
  -> leakage-safe identity, dosage, neighborhood, and library-complexity features
  -> calibrated random forest or conditional multitask DL ablation
  -> overall doublet, subtype, and high-RNA rejection outputs
  -> prediction on the original observed cells
```

### 1. Validate and copy counts

DuoDose validates the selected `adata.X` or layer as a finite, non-negative, count-like matrix and works from a copy. Experimental annotations in `adata.obs` are not consumed as training labels.

### 2. Preprocess and cluster

The observed expression background is transformed into a compact representation and partitioned into clusters. Clusters supply a dataset-adaptive operational definition for constructing similar-state and different-state cell pairs.

### 3. Generate semi-real doublets

- Homotypic semi-real doublets combine parents sampled from the same cluster.
- Heterotypic semi-real doublets combine parents sampled from different clusters.

The frozen construction is `raw_sum_parents_removed`: each constructed count vector is the direct raw-count sum of its two parents, and every exact parent cell is removed from the clean reference pool. No library-size downsampling is applied. The pair construction is controlled and seeded; it models recognizable dosage and identity-mixture states without claiming that the clusters are experimentally verified cell types.

### 4. Keep parents disjoint

Parent cells are assigned to fit, validation, and held-out test pools before constructed doublets are evaluated. The same biological parent is not allowed to appear across those splits, and exact constructed-doublet parents do not enter clean-singlet training rows or the reference pool. This prevents a model from benefiting from parent-specific expression patterns shared across training and validation.

The returned `DuoDoseResult.parent_audit` records the parent-disjoint audit summary.

### 5. Build SafeFeatures

One `SafeFeatureTransformer` is fitted once on clean fit-split reference singlets. Its selected genes, normalization, PCA, cluster coordinate system, neighborhood state, categorical mapping, and feature-column order are then reused unchanged for validation, held-out, and original observed cells. The model uses leakage-safe evidence derived from the count matrix and this frozen reference state. Feature families include:

- identity mixture and identity-inlier evidence;
- RNA dosage and residual evidence;
- artificial-doublet neighborhood evidence;
- library size and detected-gene complexity evidence, including `library_complexity_balance = log1p(nFeature) - 0.5 * log1p(nCount)`;
- high-RNA singlet protection signals.

Unsafe benchmark truth fields, experimental labels, and external method scores are not eligible public-model features. Internal identity and dosage components are features, not public complete-method scores.

The returned `DuoDoseResult.feature_audit` lists included and excluded features.

### 6. Train the conditional multitask model

The default `rf` backend predicts calibrated probabilities for singlet, homotypic-doublet, and heterotypic-doublet states. Its single frozen training rule assigns unit sample weight to ordinary singlets and a fixed weight of `2.0` to constructed high-RNA singlets; it does not search or select alternative weights. The optional `dl` backend uses conditional tasks to help distinguish high-RNA singlets from dosage-driven doublets. Both use the same parent-disjoint splits and fitted-reference feature contract; DL additionally uses held-out validation for early stopping.

Both retained backends expose the same public output contract. See [Parameters](parameters.md#backends).

### 7. Predict observed cells

The fitted model scores the original observed cells using the same SafeFeatures. The primary continuous scores are:

```text
duodose_score =
    P(homotypic_doublet) + P(heterotypic_doublet)

duodose_homotypic_score =
    P(homotypic_doublet)

duodose_heterotypic_score =
    P(heterotypic_doublet)
```

Thresholding converts the continuous overall score into `predicted_doublet`. It does not change any of the three continuous scores.

## Interpreting subtype predictions

For a predicted doublet, the larger subtype probability determines `predicted_subtype`. `subtype_confidence` is the larger subtype probability divided by total doublet probability. It therefore expresses confidence conditional on the cell being doublet-like; it is not a separate probability that the cell is a true doublet.

Homotypic and heterotypic calls are model-inferred expression states. They should be checked against cell annotations, markers, neighborhood structure, quality-control measurements, and the experimental design. See [Outputs](outputs.md) for practical interpretation.

## Scope of experimental labels

Experimental labels can be used for evaluation in a suitable benchmark, but they are not used for public-model fitting, threshold selection, feature construction, or model selection. The real validation collections are described as doublet-enriched datasets and must not be presented as complete homotypic ground truth.

## Pre-model feature provenance

The supervised backends consume mechanism and technical features computed before model fitting from the count matrix and one frozen singlet reference. Pre-model composite scores use the `handcrafted_` prefix so they cannot be confused with public RF or DL probabilities. They use no truth labels, fitted model outputs, outcome-label calibration, or query-dataset ranks. The frozen allowlist is checked together with semantic provenance metadata; either check can reject a feature.

Historical cached SafeFeature frames are migrated by the explicit map in [feature_name_migration.csv](feature_name_migration.csv). The public prediction field remains `duodose_score`; only the internal pre-model feature formerly sharing that name was renamed.

## High-RNA false-positive evaluation

The manuscript-facing high-RNA singlet FPR is measured at matched 50% homotypic recall. For each method, cells are ranked by the method's overall doublet score and the smallest deterministic prefix recovering at least half of held-out homotypic doublets is selected. The fraction of held-out high-RNA singlets in that candidate set is then reported. This compares methods at equal homotypic sensitivity instead of rewarding methods that retrieve few homotypic doublets. A fixed top-20% candidate budget and the historical top-true-doublet-count budget are retained only as supplementary sensitivity analyses.
## Domain-audit interpretation

The strict domain audit evaluates residual separation between matched semi-real and experimental doublets. Matching reduced domain separability, but it did not eliminate it: pooled out-of-fold AUROC values remain in a weak-to-moderate range. The manuscript should therefore describe partial alignment or reduced separability, not claim that the two domains are indistinguishable.

