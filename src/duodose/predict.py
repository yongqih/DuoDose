"""End-to-end DuoDose prediction API."""

from __future__ import annotations

from typing import Optional
import math
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData

from .clustering import preliminary_clustering
from .data import ensure_counts_layer, get_group_key_frame
from .features import extract_features, extract_simulated_features, infer_parent_clusters
from .model import CLASS_LABELS, DuoDoseBaseClassifier
from .preprocess import normalize_hvg_pca
from .propensity import estimate_population_propensity
from .qc import loose_qc
from .residuals import compute_dosage_residuals
from .simulate import simulate_doublets
from .train import build_training_data


REQUIRED_OBS_COLUMNS = (
    "duodose_score",
    "duodose_homotypic_score",
    "duodose_heterotypic_score",
    "duodose_low_quality_score",
    "duodose_uncertainty",
    "duodose_label",
    "duodose_predicted_parent_1",
    "duodose_predicted_parent_2",
    "duodose_library",
)


def assign_labels(
    probabilities: pd.DataFrame,
    expected_doublet_rate: float = 0.06,
    max_remove_fraction: Optional[float] = None,
    high_confidence_threshold: float = 0.9,
    uncertain_threshold: float = 0.6,
) -> pd.Series:
    """Assign DuoDose labels from subtype-specific scores."""

    probs = probabilities.copy()
    for label in CLASS_LABELS:
        if label not in probs:
            probs[label] = 0.0
    probs = probs[list(CLASS_LABELS)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n = len(probs)
    if n == 0:
        return pd.Series(dtype=object)

    labels = pd.Series("clean", index=probs.index, dtype=object)

    low_quality_mask = probs["low_quality"] >= high_confidence_threshold
    labels.loc[low_quality_mask] = "low_quality"

    heterotypic_score = probs["heterotypic_doublet"].clip(0.0, 1.0)
    homotypic_score = probs["homotypic_doublet"].clip(0.0, 1.0)
    doublet_score = _union_score(heterotypic_score, homotypic_score)

    high_heterotypic = (heterotypic_score >= high_confidence_threshold) & (heterotypic_score > homotypic_score)
    high_homotypic = (homotypic_score >= high_confidence_threshold) & (homotypic_score >= heterotypic_score)
    labels.loc[high_heterotypic & ~low_quality_mask] = "heterotypic_doublet"
    labels.loc[high_homotypic & ~low_quality_mask] = "homotypic_doublet"

    medium_doublet = (
        ((heterotypic_score >= uncertain_threshold) | (homotypic_score >= uncertain_threshold) | (doublet_score >= uncertain_threshold))
        & ~labels.isin(["heterotypic_doublet", "homotypic_doublet", "low_quality"])
    )
    labels.loc[medium_doublet] = "uncertain"
    return labels


def _apply_propensity_features(
    adata: AnnData,
    features: pd.DataFrame,
    propensity: pd.DataFrame,
    cluster_key: str,
    library_key: Optional[str],
) -> pd.DataFrame:
    if propensity.empty:
        return features
    groups = get_group_key_frame(adata, cluster_key=cluster_key, library_key=library_key)
    lookup = propensity.set_index(["cluster", "library"])
    q = []
    burden = []
    for _, row in groups.iterrows():
        key = (str(row["cluster"]), str(row["library"]))
        if key in lookup.index:
            q.append(float(lookup.loc[key, "propensity_q"]))
            burden.append(float(lookup.loc[key, "expected_homotypic_fraction"]))
        else:
            q.append(1.0)
            burden.append(0.0)
    features = features.copy()
    features["population_propensity_prior"] = q
    features["cluster_level_expected_homotypic_burden"] = burden
    return features


def _rank01(values: pd.Series | np.ndarray, *, high_is_good: bool = True) -> pd.Series:
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan)
    finite = series[np.isfinite(series)]
    fill = float(finite.median()) if len(finite) else 0.0
    series = series.fillna(fill)
    if series.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    ranked = series.rank(method="average", pct=True)
    if not high_is_good:
        ranked = 1.0 - ranked
    return ranked.clip(0.0, 1.0)


def _feature_series(features: pd.DataFrame, name: str, index: pd.Index, default: float = 0.0) -> pd.Series:
    if name in features:
        return pd.Series(features[name], index=index, dtype=float)
    return pd.Series(default, index=index, dtype=float)


def _sigmoid_gate(values: pd.Series, center: float = 1.0, scale: float = 0.75) -> pd.Series:
    scaled = ((pd.Series(values, dtype=float) - center) / max(scale, 1e-6)).clip(-60.0, 60.0)
    return pd.Series(1.0 / (1.0 + np.exp(-scaled)), index=values.index, dtype=float).clip(0.0, 1.0)


def _legacy_refine_base_scores(
    probabilities: pd.DataFrame,
    features: pd.DataFrame,
    penalize_heterotypic: bool = True,
    use_propensity_prior: bool = True,
) -> pd.DataFrame:
    """Make DuoDose-Base subtype scores obey simple biological gates.

    The random forest provides broad class probabilities. This post-calibration
    delegates homotypic discrimination to an interpretable module: homotypic
    doublets should be identity-space inliers, dosage-space outliers, uniformly
    inflated across dosage modules, and unlike heterotypic mixtures.
    """

    probs = probabilities.copy()
    for label in CLASS_LABELS:
        if label not in probs:
            probs[label] = 0.0
    probs = probs[list(CLASS_LABELS)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features = features.reindex(probs.index)

    dosage_residual = _feature_series(features, "dosage_residual", probs.index).clip(lower=0.0)
    count_residual = _feature_series(features, "cluster_count_robust_z", probs.index).clip(lower=0.0)
    gene_residual = _feature_series(features, "cluster_gene_robust_z", probs.index).clip(lower=0.0)
    stable_residual = _feature_series(features, "cluster_stable_dosage_robust_z", probs.index).clip(lower=0.0)
    homotypic_similarity_raw = _feature_series(features, "homotypic_similarity", probs.index)
    heterotypic_similarity_raw = _feature_series(features, "heterotypic_similarity", probs.index)
    distance_to_centroid = _feature_series(features, "distance_to_cluster_centroid", probs.index)
    propensity_prior_raw = _feature_series(features, "population_propensity_prior", probs.index, default=1.0)

    count_gate = _sigmoid_gate(count_residual, center=1.0, scale=0.75)
    gene_gate = _sigmoid_gate(gene_residual, center=0.75, scale=0.75)
    stable_gate = _sigmoid_gate(stable_residual, center=0.75, scale=0.75)
    residual_gate = _sigmoid_gate(dosage_residual, center=1.0, scale=0.75)
    conditional_dosage = np.minimum(count_gate, gene_gate)
    dosage_evidence = (
        0.45 * conditional_dosage
        + 0.30 * np.sqrt((gene_gate * stable_gate).clip(0.0, 1.0))
        + 0.25 * residual_gate
    ).clip(0.0, 1.0)
    homotypic_similarity = _rank01(homotypic_similarity_raw)
    heterotypic_similarity = _rank01(heterotypic_similarity_raw)
    cluster_centeredness = _rank01(distance_to_centroid, high_is_good=False)
    identity_inlier = _feature_series(features, "identity_inlier_score", probs.index, default=0.0)
    if identity_inlier.nunique(dropna=True) <= 1:
        identity_inlier = cluster_centeredness
    identity_inlier = identity_inlier.clip(0.0, 1.0)
    dosage_outlier = _feature_series(features, "dosage_outlier_score", probs.index, default=0.0)
    if dosage_outlier.nunique(dropna=True) <= 1:
        dosage_outlier = dosage_evidence
    dosage_outlier = dosage_outlier.clip(0.0, 1.0)
    uniform_dosage = _feature_series(features, "uniform_dosage_inflation_score", probs.index, default=0.5).clip(0.0, 1.0)
    biological_program = _feature_series(features, "biological_program_coherence_score", probs.index, default=0.0).clip(0.0, 1.0)
    homotypic_candidate = _feature_series(features, "homotypic_candidate_score", probs.index, default=0.0)
    if homotypic_candidate.nunique(dropna=True) <= 1:
        homotypic_candidate = np.sqrt((dosage_outlier * homotypic_similarity * identity_inlier).clip(0.0, 1.0))
    homotypic_candidate = homotypic_candidate.clip(0.0, 1.0)
    if use_propensity_prior and propensity_prior_raw.nunique(dropna=True) > 1:
        propensity_component = (0.75 + 0.25 * _rank01(propensity_prior_raw)).clip(0.75, 1.0)
    else:
        propensity_component = 1.0
    heterotypic_margin = _rank01(
        (heterotypic_similarity_raw - homotypic_similarity_raw).clip(lower=0.0)
    )
    off_centroid = _rank01(distance_to_centroid)

    heterotypic_mixture = (
        0.65 * probs["heterotypic_doublet"].clip(0.0, 1.0)
        + 0.15 * heterotypic_similarity
        + 0.15 * heterotypic_margin
        + 0.05 * off_centroid
    ).clip(0.0, 1.0)
    module_heterotypic_mixture = _feature_series(features, "homotypic_heterotypic_mixture_penalty", probs.index, default=0.0)
    if module_heterotypic_mixture.nunique(dropna=True) > 1:
        heterotypic_mixture = (0.55 * heterotypic_mixture + 0.45 * module_heterotypic_mixture.clip(0.0, 1.0)).clip(0.0, 1.0)
    homotypic_gate = np.power(
        (dosage_outlier * homotypic_similarity * identity_inlier).clip(0.0, 1.0),
        1.0 / 3.0,
    )
    heterotypic_gate = (
        0.55 * heterotypic_similarity
        + 0.25 * off_centroid
        + 0.20 * heterotypic_margin
    ).clip(0.0, 1.0)

    refined = probs.copy()
    refined["homotypic_doublet"] = (
        probs["homotypic_doublet"].clip(0.0, 1.0)
        * (0.10 + 0.90 * homotypic_gate)
        * np.power(1.0 - heterotypic_mixture, 3.0)
    )
    refined["heterotypic_doublet"] = (
        probs["heterotypic_doublet"].clip(0.0, 1.0)
        * (0.35 + 0.65 * heterotypic_gate)
    )
    doublet_before = probs["homotypic_doublet"] + probs["heterotypic_doublet"]
    heterotypic_penalty = np.square(1.0 - heterotypic_mixture) if penalize_heterotypic else 1.0
    biological_program_penalty = np.square(1.0 - 0.75 * biological_program)
    uniform_component = (0.35 + 0.65 * uniform_dosage).clip(0.35, 1.0)
    module_final = (
        homotypic_candidate
        * uniform_component
        * biological_program_penalty
        * heterotypic_penalty
        * propensity_component
    ).clip(0.0, 1.0)
    homotypic_evidence_score = (
        0.62 * module_final
        + 0.18 * (dosage_outlier * uniform_component * biological_program_penalty).clip(0.0, 1.0)
        + 0.12 * probs["homotypic_doublet"].clip(0.0, 1.0)
        + 0.08 * doublet_before.clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    refined["homotypic_doublet"] = 0.20 * refined["homotypic_doublet"] + 0.80 * homotypic_evidence_score
    if penalize_heterotypic:
        homotypic_cap_base = np.maximum(doublet_before.clip(0.0, 1.0), module_final)
        refined["homotypic_doublet"] = np.minimum(
            refined["homotypic_doublet"],
            homotypic_cap_base * np.square(1.0 - 0.85 * heterotypic_mixture),
        )
    refined["homotypic_doublet"] = refined["homotypic_doublet"].clip(0.0, 1.0)
    refined["heterotypic_doublet"] = refined["heterotypic_doublet"].clip(0.0, 1.0)

    doublet_after = refined["homotypic_doublet"] + refined["heterotypic_doublet"]
    released_mass = (doublet_before - doublet_after).clip(lower=0.0)
    refined["clean"] = (refined["clean"] + released_mass).clip(0.0, 1.0)

    row_sum = refined[list(CLASS_LABELS)].sum(axis=1).replace(0.0, 1.0)
    return refined[list(CLASS_LABELS)].div(row_sum, axis=0)


def _union_score(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> pd.Series:
    left_series = pd.Series(left, dtype=float).clip(0.0, 1.0)
    right_series = pd.Series(right, index=left_series.index, dtype=float).clip(0.0, 1.0)
    return (1.0 - (1.0 - left_series) * (1.0 - right_series)).clip(0.0, 1.0)


def _rank_calibrated_scores(values: pd.Series | np.ndarray, groups: Optional[pd.Series] = None) -> pd.Series:
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan)
    if groups is None:
        if series.nunique(dropna=True) <= 1:
            return pd.Series(0.5, index=series.index, dtype=float)
        return series.rank(method="average", pct=True).fillna(0.5).clip(0.0, 1.0)

    group_series = pd.Series(groups, index=series.index).astype(str)
    ranked = pd.Series(index=series.index, dtype=float)
    for _, idx in group_series.groupby(group_series).groups.items():
        values_in_group = series.loc[idx]
        if values_in_group.nunique(dropna=True) <= 1:
            ranked.loc[idx] = 0.5
        else:
            ranked.loc[idx] = values_in_group.rank(method="average", pct=True).fillna(0.5)
    return ranked.fillna(0.5).clip(0.0, 1.0)


def _rank_calibrated_union(
    heterotypic_score: pd.Series | np.ndarray,
    homotypic_score: pd.Series | np.ndarray,
    groups: Optional[pd.Series] = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    heterotypic_rank = _rank_calibrated_scores(heterotypic_score, groups=groups)
    homotypic_rank = _rank_calibrated_scores(homotypic_score, groups=groups)
    return _union_score(heterotypic_rank, homotypic_rank), heterotypic_rank, homotypic_rank


def _tail_calibrated_scores(values: pd.Series | np.ndarray, groups: Optional[pd.Series] = None) -> pd.Series:
    percentile = _rank_calibrated_scores(values, groups=groups)
    return ((percentile - 0.5) / 0.5).clip(0.0, 1.0)


def _tail_calibrated_union(
    heterotypic_score: pd.Series | np.ndarray,
    homotypic_score: pd.Series | np.ndarray,
    groups: Optional[pd.Series] = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    heterotypic_tail = _tail_calibrated_scores(heterotypic_score, groups=groups)
    homotypic_tail = _tail_calibrated_scores(homotypic_score, groups=groups)
    return _union_score(heterotypic_tail, homotypic_tail), heterotypic_tail, homotypic_tail


def _refine_base_scores(
    probabilities: pd.DataFrame,
    features: pd.DataFrame,
    penalize_heterotypic: bool = True,
    use_propensity_prior: bool = True,
) -> pd.DataFrame:
    """Fuse subtype-specific DuoDose-Base scores without subtype competition.

    Heterotypic and homotypic evidence are treated as separate scores. The
    homotypic score is driven primarily by the interpretable homotypic module,
    with only soft damping from strong heterotypic or biological-program
    evidence. Overall doublet risk is computed downstream as a monotonic union.
    """

    probs = probabilities.copy()
    for label in CLASS_LABELS:
        if label not in probs:
            probs[label] = 0.0
    probs = probs[list(CLASS_LABELS)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features = features.reindex(probs.index)

    homotypic_similarity_raw = _feature_series(features, "homotypic_similarity", probs.index)
    heterotypic_similarity_raw = _feature_series(features, "heterotypic_similarity", probs.index)
    distance_to_centroid = _feature_series(features, "distance_to_cluster_centroid", probs.index)
    homotypic_module = _feature_series(features, "homotypic_final_score", probs.index, default=0.0).clip(0.0, 1.0)
    homotypic_candidate = _feature_series(features, "homotypic_candidate_score", probs.index, default=0.0).clip(0.0, 1.0)
    dosage_outlier = _feature_series(features, "dosage_outlier_score", probs.index, default=0.0).clip(0.0, 1.0)
    uniform_dosage = _feature_series(features, "uniform_dosage_inflation_score", probs.index, default=0.5).clip(0.0, 1.0)
    biological_program = _feature_series(features, "biological_program_coherence_score", probs.index, default=0.0).clip(0.0, 1.0)
    identity_inlier = _feature_series(features, "identity_inlier_score", probs.index, default=0.0).clip(0.0, 1.0)
    propensity_prior_raw = _feature_series(features, "population_propensity_prior", probs.index, default=1.0)

    homotypic_similarity = _rank01(homotypic_similarity_raw)
    heterotypic_similarity = _rank01(heterotypic_similarity_raw)
    heterotypic_margin = _rank01((heterotypic_similarity_raw - homotypic_similarity_raw).clip(lower=0.0))
    off_centroid = _rank01(distance_to_centroid)
    cluster_centeredness = _rank01(distance_to_centroid, high_is_good=False)
    if identity_inlier.nunique(dropna=True) <= 1:
        identity_inlier = cluster_centeredness

    heterotypic_gate = (
        0.55 * heterotypic_similarity
        + 0.25 * off_centroid
        + 0.20 * heterotypic_margin
    ).clip(0.0, 1.0)
    heterotypic_score = (
        probs["heterotypic_doublet"].clip(0.0, 1.0)
        * (0.35 + 0.65 * heterotypic_gate)
    ).clip(0.0, 1.0)

    module_mixture = _feature_series(features, "homotypic_heterotypic_mixture_penalty", probs.index, default=0.0).clip(0.0, 1.0)
    heterotypic_mixture = (
        0.45 * heterotypic_score
        + 0.20 * heterotypic_similarity
        + 0.20 * heterotypic_margin
        + 0.10 * off_centroid
        + 0.05 * module_mixture
    ).clip(0.0, 1.0)
    heterotypic_soft_gate = (1.0 - 0.40 * np.square(heterotypic_mixture)).clip(0.60, 1.0) if penalize_heterotypic else 1.0

    biological_soft_gate = (1.0 - 0.45 * biological_program).clip(0.55, 1.0)
    uniform_component = (0.25 + 0.75 * uniform_dosage).clip(0.25, 1.0)
    fallback_module = (
        homotypic_candidate
        * uniform_component
        * biological_soft_gate
        * (0.30 + 0.70 * identity_inlier)
    ).clip(0.0, 1.0)
    if homotypic_module.nunique(dropna=True) <= 1:
        homotypic_module = fallback_module

    if use_propensity_prior and propensity_prior_raw.nunique(dropna=True) > 1:
        propensity_component = (0.85 + 0.15 * _rank01(propensity_prior_raw)).clip(0.85, 1.0)
    else:
        propensity_component = 1.0

    raw_support = (
        0.65 * probs["homotypic_doublet"].clip(0.0, 1.0)
        + 0.20 * probs["heterotypic_doublet"].clip(0.0, 1.0)
        + 0.15 * dosage_outlier
    ).clip(0.0, 1.0)
    homotypic_score = (
        0.72 * homotypic_module
        + 0.15 * fallback_module
        + 0.08 * homotypic_candidate
        + 0.05 * raw_support
    ).clip(0.0, 1.0)
    homotypic_score = (
        homotypic_score
        * heterotypic_soft_gate
        * biological_soft_gate
        * propensity_component
    ).clip(0.0, 1.0)

    doublet_union = _union_score(heterotypic_score, homotypic_score)
    low_quality = probs["low_quality"].clip(0.0, 1.0)
    refined = pd.DataFrame(index=probs.index)
    refined["heterotypic_doublet"] = heterotypic_score
    refined["homotypic_doublet"] = homotypic_score
    refined["low_quality"] = low_quality
    refined["clean"] = ((1.0 - doublet_union) * (1.0 - low_quality)).clip(0.0, 1.0)
    return refined[list(CLASS_LABELS)].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def run_duodose(
    adata: AnnData,
    library_key: Optional[str] = "sample_id",
    expected_doublet_rate: float = 0.06,
    n_simulated_doublets: int = 50000,
    n_hvgs: int = 2000,
    n_pcs: int = 50,
    clustering_resolution: float = 0.6,
    random_state: int = 0,
    counts_layer: str = "counts",
    min_counts: int = 500,
    min_genes: int = 200,
    max_mito_fraction: float = 0.3,
    homotypic_fraction: float = 0.5,
    saturation_range: tuple[float, float] = (0.6, 1.0),
    max_remove_fraction: Optional[float] = None,
    high_confidence_threshold: float = 0.9,
    uncertain_threshold: float = 0.6,
) -> AnnData:
    """Run the full DuoDose-Base pipeline on an AnnData object."""

    if library_key is not None and library_key not in adata.obs:
        warnings.warn(f"library_key={library_key!r} not found; treating all cells as one library.", RuntimeWarning, stacklevel=2)
        library_key = None

    work = adata.copy()
    ensure_counts_layer(work, counts_layer=counts_layer)
    work = loose_qc(
        work,
        min_counts=min_counts,
        min_genes=min_genes,
        max_mito_fraction=max_mito_fraction,
        counts_layer=counts_layer,
    )
    ensure_counts_layer(work, counts_layer=counts_layer)
    normalize_hvg_pca(work, counts_layer=counts_layer, n_hvgs=n_hvgs, n_pcs=n_pcs, random_state=random_state)
    preliminary_clustering(work, resolution=clustering_resolution, use_rep="X_pca", random_state=random_state)
    compute_dosage_residuals(work, cluster_key="duodose_cluster", library_key=library_key, counts_layer=counts_layer)

    simulated = simulate_doublets(
        work,
        cluster_key="duodose_cluster",
        library_key=library_key,
        counts_layer=counts_layer,
        n_doublets=n_simulated_doublets,
        homotypic_fraction=homotypic_fraction,
        saturation_range=saturation_range,
        random_state=random_state,
    )
    observed_features = extract_features(work, simulated, cluster_key="duodose_cluster", library_key=library_key, use_rep="X_pca")
    propensity = estimate_population_propensity(
        observed_features["heterotypic_similarity"],
        work,
        cluster_key="duodose_cluster",
        library_key=library_key,
        expected_doublet_rate=expected_doublet_rate,
    )
    observed_features = _apply_propensity_features(work, observed_features, propensity, "duodose_cluster", library_key)
    simulated_features = extract_simulated_features(
        work,
        simulated,
        observed_features,
        cluster_key="duodose_cluster",
        library_key=library_key,
        use_rep="X_pca",
    )

    X_train, y_train = build_training_data(work, simulated, observed_features, simulated_features)
    classifier = DuoDoseBaseClassifier(random_state=random_state)
    classifier.fit(X_train, y_train)
    raw_probabilities = classifier.predict_proba(observed_features)
    probabilities = _refine_base_scores(raw_probabilities, observed_features)
    labels = assign_labels(
        probabilities,
        expected_doublet_rate=expected_doublet_rate,
        max_remove_fraction=max_remove_fraction,
        high_confidence_threshold=high_confidence_threshold,
        uncertain_threshold=uncertain_threshold,
    )
    parent1, parent2 = infer_parent_clusters(work, simulated, use_rep="X_pca")

    work.obs["duodose_heterotypic_score"] = probabilities["heterotypic_doublet"].to_numpy()
    work.obs["duodose_homotypic_score"] = probabilities["homotypic_doublet"].to_numpy()
    work.obs["duodose_low_quality_score"] = np.maximum(
        work.obs.get("duodose_low_quality_score", pd.Series(0.0, index=work.obs_names)).to_numpy(dtype=float),
        probabilities["low_quality"].to_numpy(),
    )
    raw_union_score = _union_score(
        work.obs["duodose_heterotypic_score"],
        work.obs["duodose_homotypic_score"],
    )
    if library_key is None:
        rank_groups = pd.Series("__all__", index=work.obs_names, dtype=object)
    else:
        rank_groups = work.obs[library_key].astype(str)
    rank_union_score, heterotypic_rank_score, homotypic_rank_score = _rank_calibrated_union(
        work.obs["duodose_heterotypic_score"],
        work.obs["duodose_homotypic_score"],
        groups=rank_groups,
    )
    tail_union_score, heterotypic_tail_score, homotypic_tail_score = _tail_calibrated_union(
        work.obs["duodose_heterotypic_score"],
        work.obs["duodose_homotypic_score"],
        groups=rank_groups,
    )
    work.obs["duodose_score_raw_union"] = raw_union_score.to_numpy()
    work.obs["duodose_heterotypic_rank_score"] = heterotypic_rank_score.to_numpy()
    work.obs["duodose_homotypic_rank_score"] = homotypic_rank_score.to_numpy()
    work.obs["duodose_score_rank_calibrated"] = rank_union_score.to_numpy()
    work.obs["duodose_heterotypic_tail_score"] = heterotypic_tail_score.to_numpy()
    work.obs["duodose_homotypic_tail_score"] = homotypic_tail_score.to_numpy()
    work.obs["duodose_score_tail_calibrated"] = tail_union_score.to_numpy()
    work.obs["duodose_score"] = tail_union_score.to_numpy()
    work.obs["duodose_uncertainty"] = 1.0 - probabilities.max(axis=1).to_numpy()
    for feature_column, obs_column in {
        "identity_inlier_score": "duodose_identity_inlier_score",
        "dosage_outlier_score": "duodose_dosage_outlier_score",
        "uniform_dosage_inflation_score": "duodose_uniform_dosage_inflation_score",
        "biological_program_coherence_score": "duodose_biological_program_coherence_score",
        "homotypic_candidate_score": "duodose_homotypic_candidate_score",
        "homotypic_final_score": "duodose_homotypic_module_score",
    }.items():
        if feature_column in observed_features:
            work.obs[obs_column] = observed_features[feature_column].to_numpy(dtype=float)
    work.obs["duodose_label"] = pd.Categorical(
        labels.loc[work.obs_names],
        categories=["clean", "heterotypic_doublet", "homotypic_doublet", "low_quality", "uncertain"],
    )
    work.obs["duodose_predicted_parent_1"] = parent1
    work.obs["duodose_predicted_parent_2"] = parent2
    if library_key is None:
        work.obs["duodose_library"] = "__all__"
    else:
        work.obs["duodose_library"] = work.obs[library_key].astype(str)

    work.uns["duodose_features"] = observed_features
    work.uns["duodose_raw_probabilities"] = raw_probabilities
    work.uns["duodose_refined_probabilities"] = probabilities
    work.uns["duodose_propensity"] = propensity
    work.uns["duodose_training_summary"] = {
        "n_training_examples": int(len(X_train)),
        "training_class_counts": y_train.value_counts().to_dict(),
        "expected_doublet_rate": float(expected_doublet_rate),
        "n_simulated_doublets": int(len(simulated)),
    }
    return work
