"""Feature extraction for observed cells and artificial doublets."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from .clustering import compute_cluster_centroids, compute_cluster_distances
from .data import SimulatedDoublets, get_counts_matrix, get_group_key_frame, get_library_series, grouped_robust_z, robust_mad, row_nnz, row_sums, safe_divide
from .preprocess import transform_counts_to_pca


def _fraction_for_prefix(adata: AnnData, prefixes: tuple[str, ...], counts_layer: str = "counts") -> np.ndarray:
    X = get_counts_matrix(adata, counts_layer=counts_layer)
    names = pd.Index(adata.var_names.astype(str))
    mask = np.zeros(adata.n_vars, dtype=bool)
    for prefix in prefixes:
        mask |= np.asarray(names.str.startswith(prefix), dtype=bool)
    total = row_sums(X).astype(float)
    selected = row_sums(X[:, mask]).astype(float) if mask.any() else np.zeros(adata.n_obs)
    return safe_divide(selected, total)


def _count_metrics_from_matrix(X) -> tuple[np.ndarray, np.ndarray]:
    return row_sums(X).astype(float), row_nnz(X).astype(float)


def _nearest_distance(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if reference.shape[0] == 0:
        return np.full(query.shape[0], np.nan, dtype=float)
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(reference)
    dist, _ = nn.kneighbors(query)
    return dist[:, 0]


def _artificial_neighbor_fraction(observed_rep: np.ndarray, simulated_rep: np.ndarray, k: int = 25) -> np.ndarray:
    if simulated_rep.shape[0] == 0:
        return np.zeros(observed_rep.shape[0], dtype=float)
    combined = np.vstack([observed_rep, simulated_rep])
    n_neighbors = min(k + 1, combined.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(combined)
    indices = nn.kneighbors(observed_rep, return_distance=False)
    artificial = indices >= observed_rep.shape[0]
    return artificial.sum(axis=1) / max(1, n_neighbors - 1)


def _sigmoid_array(values: np.ndarray, center: float = 0.0, scale: float = 1.0) -> np.ndarray:
    scaled = ((np.asarray(values, dtype=float) - center) / max(scale, 1e-6)).clip(-60.0, 60.0)
    return np.clip(1.0 / (1.0 + np.exp(-scaled)), 0.0, 1.0)


def _rank01_array(values: np.ndarray, high_is_good: bool = True) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float)).replace([np.inf, -np.inf], np.nan)
    finite = series[np.isfinite(series)]
    fill = float(finite.median()) if len(finite) else 0.0
    series = series.fillna(fill)
    if series.nunique(dropna=True) <= 1:
        ranked = pd.Series(0.5, index=series.index, dtype=float)
    else:
        ranked = series.rank(method="average", pct=True)
    if not high_is_good:
        ranked = 1.0 - ranked
    return ranked.clip(0.0, 1.0).to_numpy(dtype=float)


def _normalized_representation(rep: np.ndarray, reference: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    reference = rep if reference is None else reference
    scale = np.nanstd(np.asarray(reference, dtype=float), axis=0)
    scale[~np.isfinite(scale) | (scale <= 1e-8)] = 1.0
    return np.asarray(rep, dtype=float) / scale, scale


def _local_neighbor_consistency(rep: np.ndarray, clusters: np.ndarray, k: int = 15) -> np.ndarray:
    if rep.shape[0] <= 1:
        return np.ones(rep.shape[0], dtype=float)
    n_neighbors = min(k + 1, rep.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(rep)
    indices = nn.kneighbors(rep, return_distance=False)
    neighbor_clusters = clusters[indices[:, 1:]]
    return (neighbor_clusters == clusters[:, None]).mean(axis=1)


def _reference_neighbor_consistency(
    query_rep: np.ndarray,
    reference_rep: np.ndarray,
    reference_clusters: np.ndarray,
    query_clusters: np.ndarray,
    k: int = 15,
) -> np.ndarray:
    if reference_rep.shape[0] == 0:
        return np.ones(query_rep.shape[0], dtype=float)
    n_neighbors = min(k, reference_rep.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(reference_rep)
    indices = nn.kneighbors(query_rep, return_distance=False)
    neighbor_clusters = reference_clusters[indices]
    return (neighbor_clusters == query_clusters[:, None]).mean(axis=1)


def _cluster_abundance(adata: AnnData, cluster_key: str, library_key: Optional[str]) -> pd.Series:
    groups = get_group_key_frame(adata, cluster_key=cluster_key, library_key=library_key)
    abundance = pd.Series(index=adata.obs_names, dtype=float)
    for library, lib_idx in groups.groupby("library").groups.items():
        frame = groups.loc[lib_idx]
        counts = frame["cluster"].value_counts(normalize=True)
        abundance.loc[lib_idx] = frame["cluster"].map(counts).astype(float)
    return abundance.fillna(0.0)


def _cluster_min_distance(centroid_distances: pd.DataFrame, clusters: pd.Series) -> pd.Series:
    values = {}
    for cluster in centroid_distances.index:
        row = centroid_distances.loc[cluster].drop(index=cluster, errors="ignore")
        values[cluster] = float(row.min()) if len(row) else 0.0
    return clusters.astype(str).map(values).fillna(0.0)


def _dosage_z_against_observed(
    values: np.ndarray,
    clusters: np.ndarray,
    libraries: np.ndarray,
    observed_groups: pd.DataFrame,
    observed_values: pd.Series,
) -> np.ndarray:
    out = np.zeros(len(values), dtype=float)
    cache: dict[tuple[str, str], tuple[float, float]] = {}
    for i, (cluster, library) in enumerate(zip(clusters.astype(str), libraries.astype(str))):
        key = (cluster, library)
        if key not in cache:
            mask = (observed_groups["cluster"].astype(str) == cluster) & (observed_groups["library"].astype(str) == library)
            if not mask.any():
                mask = observed_groups["cluster"].astype(str) == cluster
            cache[key] = robust_mad(observed_values.loc[mask].to_numpy())
        median, scale = cache[key]
        out[i] = (values[i] - median) / scale
    return np.nan_to_num(out)


def _cluster_z_against_observed(
    values: np.ndarray,
    clusters: np.ndarray,
    observed_clusters: pd.Series,
    observed_values: pd.Series,
) -> np.ndarray:
    out = np.zeros(len(values), dtype=float)
    cache: dict[str, tuple[float, float]] = {}
    observed_clusters = observed_clusters.astype(str)
    for i, cluster in enumerate(clusters.astype(str)):
        if cluster not in cache:
            mask = observed_clusters == cluster
            cache[cluster] = robust_mad(observed_values.loc[mask].to_numpy()) if mask.any() else robust_mad(observed_values.to_numpy())
        median, scale = cache[cluster]
        out[i] = (values[i] - median) / scale
    return np.nan_to_num(out)


def _build_homotypic_discrimination_features(
    index: pd.Index,
    count_z: np.ndarray,
    gene_z: np.ndarray,
    stable_z: np.ndarray,
    marker_z: np.ndarray,
    distance_z: np.ndarray,
    neighbor_consistency: np.ndarray,
    homotypic_similarity: np.ndarray,
    heterotypic_similarity: np.ndarray,
    global_count_rank: np.ndarray | None = None,
) -> pd.DataFrame:
    """Construct interpretable homotypic-specific evidence scores."""

    residuals = pd.DataFrame(
        {
            "count": np.maximum(np.asarray(count_z, dtype=float), 0.0),
            "gene": np.maximum(np.asarray(gene_z, dtype=float), 0.0),
            "stable": np.maximum(np.asarray(stable_z, dtype=float), 0.0),
            "marker": np.maximum(np.asarray(marker_z, dtype=float), 0.0),
        },
        index=index,
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    residual_ranks = residuals.rank(method="average", pct=True).fillna(0.5).clip(0.0, 1.0)

    count_gate = _sigmoid_array(residuals["count"].to_numpy(), center=1.50, scale=0.60)
    gene_gate = _sigmoid_array(residuals["gene"].to_numpy(), center=1.25, scale=0.60)
    stable_gate = _sigmoid_array(residuals["stable"].to_numpy(), center=1.25, scale=0.60)
    marker_gate = _sigmoid_array(residuals["marker"].to_numpy(), center=1.25, scale=0.60)
    dosage_outlier = np.clip(0.35 * count_gate + 0.20 * gene_gate + 0.25 * stable_gate + 0.20 * marker_gate, 0.0, 1.0)

    gate_frame = pd.DataFrame({"count": count_gate, "gene": gene_gate, "stable": stable_gate, "marker": marker_gate}, index=index)
    rank_min = residual_ranks.min(axis=1).to_numpy(dtype=float)
    rank_max = residual_ranks.max(axis=1).to_numpy(dtype=float)
    rank_spread = np.clip(rank_max - rank_min, 0.0, 1.0)
    rank_std = residual_ranks.std(axis=1, ddof=0).to_numpy(dtype=float)
    rank_balance = np.clip(1.0 - 2.5 * rank_std, 0.0, 1.0)
    gate_joint = gate_frame.min(axis=1).to_numpy(dtype=float)
    uniform_dosage = np.sqrt(np.clip(gate_joint * rank_balance, 0.0, 1.0))

    module_high = np.maximum(rank_max, gate_frame.max(axis=1).to_numpy(dtype=float))
    module_imbalance = np.clip(0.65 * rank_spread + 0.35 * (1.0 - rank_balance), 0.0, 1.0)
    centroid_inlier = _sigmoid_array(-np.asarray(distance_z, dtype=float), center=-1.25, scale=0.75)
    neighbor_consistency = np.clip(np.asarray(neighbor_consistency, dtype=float), 0.0, 1.0)
    identity_inlier = np.sqrt(np.clip(centroid_inlier * (0.25 + 0.75 * neighbor_consistency), 0.0, 1.0))

    imbalance_program = np.clip(module_high * module_imbalance * _sigmoid_array(dosage_outlier, center=0.45, scale=0.20), 0.0, 1.0)
    if global_count_rank is None:
        coherent_high_rna_program = np.zeros(len(index), dtype=float)
    else:
        coherent_high_rna_program = np.clip(
            np.asarray(global_count_rank, dtype=float)
            * identity_inlier
            * (1.0 - uniform_dosage)
            * (1.0 - 0.65 * dosage_outlier),
            0.0,
            1.0,
        )
    biological_program = np.maximum(imbalance_program, coherent_high_rna_program)

    homotypic_rank = _rank01_array(homotypic_similarity)
    heterotypic_rank = _rank01_array(heterotypic_similarity)
    heterotypic_margin = _rank01_array(np.maximum(np.asarray(heterotypic_similarity) - np.asarray(homotypic_similarity), 0.0))
    heterotypic_mixture = np.clip(0.55 * heterotypic_rank + 0.35 * heterotypic_margin + 0.10 * (1.0 - identity_inlier), 0.0, 1.0)

    sensitive_evidence = np.clip(0.55 * dosage_outlier + 0.25 * homotypic_rank + 0.20 * identity_inlier, 0.0, 1.0)
    homotypic_candidate = np.sqrt(np.clip(sensitive_evidence * (0.10 + 0.90 * dosage_outlier) * (0.35 + 0.65 * identity_inlier), 0.0, 1.0))
    homotypic_final = np.clip(
        homotypic_candidate
        * (0.35 + 0.65 * uniform_dosage)
        * (1.0 - 0.55 * biological_program)
        * (1.0 - 0.35 * heterotypic_mixture),
        0.0,
        1.0,
    )

    return pd.DataFrame(
        {
            "identity_centroid_inlier_score": centroid_inlier,
            "local_neighbor_consistency": neighbor_consistency,
            "identity_inlier_score": identity_inlier,
            "cluster_marker_dosage_robust_z": marker_z,
            "dosage_outlier_score": dosage_outlier,
            "uniform_dosage_inflation_score": uniform_dosage,
            "biological_program_coherence_score": biological_program,
            "homotypic_candidate_score": homotypic_candidate,
            "homotypic_final_score": homotypic_final,
            "homotypic_heterotypic_mixture_penalty": heterotypic_mixture,
            "module_residual_rank_mean": residual_ranks.mean(axis=1).to_numpy(dtype=float),
            "module_residual_rank_spread": rank_spread,
        },
        index=index,
    )


def extract_features(
    adata: AnnData,
    simulated_doublets: SimulatedDoublets,
    cluster_key: str = "duodose_cluster",
    library_key: Optional[str] = None,
    use_rep: str = "X_pca",
) -> pd.DataFrame:
    """Extract observed-cell DuoDose features indexed by cell barcode."""

    if use_rep not in adata.obsm:
        raise KeyError(f"use_rep={use_rep!r} not found in adata.obsm")
    observed_rep = np.asarray(adata.obsm[use_rep], dtype=float)
    observed_rep_norm, rep_scale = _normalized_representation(observed_rep)
    simulated_rep = transform_counts_to_pca(adata, simulated_doublets.X) if len(simulated_doublets) else np.zeros((0, observed_rep.shape[1]))
    simulated_rep_norm = simulated_rep / rep_scale

    centroids = compute_cluster_centroids(adata, cluster_key=cluster_key, use_rep=use_rep)
    centroids_norm = pd.DataFrame(centroids.to_numpy(dtype=float) / rep_scale, index=centroids.index, columns=centroids.columns)
    centroid_distances = compute_cluster_distances(centroids)
    adata.uns["duodose_cluster_centroids"] = centroids.to_numpy()
    adata.uns["duodose_cluster_labels"] = centroids.index.to_numpy(dtype=object)
    adata.uns["duodose_cluster_distances"] = centroid_distances

    clusters = adata.obs[cluster_key].astype(str)
    centroid_lookup = centroids.loc[clusters].to_numpy()
    distance_to_centroid = np.linalg.norm(observed_rep - centroid_lookup, axis=1)
    centroid_lookup_norm = centroids_norm.loc[clusters].to_numpy()
    normalized_distance_to_centroid = np.linalg.norm(observed_rep_norm - centroid_lookup_norm, axis=1)
    distance_z = grouped_robust_z(pd.Series(normalized_distance_to_centroid, index=adata.obs_names), get_group_key_frame(adata, cluster_key=cluster_key, library_key=None)).to_numpy(dtype=float)
    neighbor_consistency = _local_neighbor_consistency(observed_rep_norm, clusters.to_numpy(dtype=str))

    heter_mask = simulated_doublets.doublet_type == "heterotypic"
    homo_mask = simulated_doublets.doublet_type == "homotypic"
    nearest_heter = _nearest_distance(observed_rep_norm, simulated_rep_norm[heter_mask]) if len(simulated_doublets) else np.full(adata.n_obs, np.nan)
    nearest_homo = _nearest_distance(observed_rep_norm, simulated_rep_norm[homo_mask]) if len(simulated_doublets) else np.full(adata.n_obs, np.nan)
    finite_dist = np.concatenate([nearest_heter[np.isfinite(nearest_heter)], nearest_homo[np.isfinite(nearest_homo)]])
    fill_dist = float(np.nanmax(finite_dist)) if finite_dist.size else 0.0
    nearest_heter = np.nan_to_num(nearest_heter, nan=fill_dist)
    nearest_homo = np.nan_to_num(nearest_homo, nan=fill_dist)

    artificial_fraction = _artificial_neighbor_fraction(observed_rep_norm, simulated_rep_norm)
    X = get_counts_matrix(adata)
    n_counts, n_genes = _count_metrics_from_matrix(X)
    if "mito_fraction" in adata.obs:
        mito_fraction = adata.obs["mito_fraction"].to_numpy(dtype=float)
    else:
        mito_fraction = _fraction_for_prefix(adata, ("MT-", "mt-"))
    ribosomal_fraction = _fraction_for_prefix(adata, ("RPL", "RPS", "rpl", "rps"))
    complexity = safe_divide(n_genes, n_counts)
    abundance = _cluster_abundance(adata, cluster_key, library_key)
    groups = get_group_key_frame(adata, cluster_key=cluster_key, library_key=library_key)
    marker_dosage = pd.Series(_cluster_marker_dosage_score(adata, cluster_key), index=adata.obs_names)
    marker_z = grouped_robust_z(marker_dosage, groups).to_numpy(dtype=float)

    features = pd.DataFrame(index=adata.obs_names)
    n_rep = observed_rep.shape[1]
    for i in range(n_rep):
        features[f"pca_{i}"] = observed_rep[:, i]
    features["distance_to_cluster_centroid"] = distance_to_centroid
    features["normalized_distance_to_cluster_centroid"] = normalized_distance_to_centroid
    features["nearest_heterotypic_doublet_distance"] = nearest_heter
    features["nearest_homotypic_doublet_distance"] = nearest_homo
    features["heterotypic_similarity"] = 1.0 / (1.0 + nearest_heter)
    features["homotypic_similarity"] = 1.0 / (1.0 + nearest_homo)
    features["artificial_neighbor_fraction"] = artificial_fraction
    features["log1p_n_counts"] = np.log1p(n_counts)
    features["log1p_n_genes"] = np.log1p(n_genes)
    features["complexity"] = complexity
    features["mito_fraction"] = mito_fraction
    features["ribosomal_fraction"] = ribosomal_fraction
    features["stable_gene_module_score"] = adata.obs.get("stable_gene_module_score", pd.Series(np.log1p(n_counts), index=adata.obs_names)).to_numpy(dtype=float)
    features["cluster_marker_dosage_score"] = marker_dosage.to_numpy(dtype=float)
    features["cluster_count_robust_z"] = adata.obs.get("duodose_count_residual", pd.Series(0.0, index=adata.obs_names)).to_numpy(dtype=float)
    features["cluster_gene_robust_z"] = adata.obs.get("duodose_gene_residual", pd.Series(0.0, index=adata.obs_names)).to_numpy(dtype=float)
    features["cluster_stable_dosage_robust_z"] = adata.obs.get("duodose_stable_dosage_residual", pd.Series(0.0, index=adata.obs_names)).to_numpy(dtype=float)
    features["dosage_residual"] = adata.obs.get("duodose_dosage_residual", pd.Series(0.0, index=adata.obs_names)).to_numpy(dtype=float)
    features["cluster_abundance"] = abundance.to_numpy(dtype=float)
    features["cluster_level_expected_homotypic_burden"] = np.square(features["cluster_abundance"])
    features["population_propensity_prior"] = 1.0
    features["cluster_distance_to_likely_parent"] = _cluster_min_distance(centroid_distances, clusters).to_numpy(dtype=float)
    features["duodose_low_quality_score"] = adata.obs.get("duodose_low_quality_score", pd.Series(0.0, index=adata.obs_names)).to_numpy(dtype=float)
    homotypic_features = _build_homotypic_discrimination_features(
        features.index,
        features["cluster_count_robust_z"].to_numpy(dtype=float),
        features["cluster_gene_robust_z"].to_numpy(dtype=float),
        features["cluster_stable_dosage_robust_z"].to_numpy(dtype=float),
        marker_z,
        distance_z,
        neighbor_consistency,
        features["homotypic_similarity"].to_numpy(dtype=float),
        features["heterotypic_similarity"].to_numpy(dtype=float),
        _rank01_array(features["log1p_n_counts"].to_numpy(dtype=float)),
    )
    features = pd.concat([features, homotypic_features], axis=1)

    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _cluster_marker_dosage_score(adata: AnnData, cluster_key: str, top_n: int = 25, counts_layer: str = "counts") -> np.ndarray:
    marker_basis = adata.X
    if sparse.issparse(marker_basis):
        X_dense = marker_basis.toarray() if adata.n_obs * adata.n_vars <= 5_000_000 else None
    else:
        X_dense = np.asarray(marker_basis)
    if X_dense is None:
        return np.log1p(np.asarray(row_sums(get_counts_matrix(adata, counts_layer=counts_layer)), dtype=float))

    counts = get_counts_matrix(adata, counts_layer=counts_layer)
    clusters = adata.obs[cluster_key].astype(str).to_numpy()
    unique = np.unique(clusters)
    scores = np.zeros(adata.n_obs, dtype=float)
    global_mean = X_dense.mean(axis=0)
    for cluster in unique:
        idx = np.flatnonzero(clusters == cluster)
        if idx.size == 0:
            continue
        cluster_mean = X_dense[idx].mean(axis=0)
        markers = np.argsort(cluster_mean - global_mean)[::-1][: min(top_n, adata.n_vars)]
        scores[idx] = np.log1p(row_sums(counts[idx][:, markers]).astype(float))
    return scores


def extract_simulated_features(
    adata: AnnData,
    simulated_doublets: SimulatedDoublets,
    observed_features: pd.DataFrame,
    cluster_key: str = "duodose_cluster",
    library_key: Optional[str] = None,
    use_rep: str = "X_pca",
) -> pd.DataFrame:
    """Extract classifier features for simulated doublets."""

    if len(simulated_doublets) == 0:
        return pd.DataFrame(columns=observed_features.columns)

    observed_rep = np.asarray(adata.obsm[use_rep], dtype=float)
    observed_rep_norm, rep_scale = _normalized_representation(observed_rep)
    sim_rep = transform_counts_to_pca(adata, simulated_doublets.X)
    sim_rep_norm = sim_rep / rep_scale
    centroids = compute_cluster_centroids(adata, cluster_key=cluster_key, use_rep=use_rep)
    centroids_norm = pd.DataFrame(centroids.to_numpy(dtype=float) / rep_scale, index=centroids.index, columns=centroids.columns)
    centroid_matrix = centroids.to_numpy()
    centroid_matrix_norm = centroids_norm.to_numpy()
    centroid_labels = centroids.index.astype(str).to_numpy()
    nearest_centroid = pairwise_distances(sim_rep_norm, centroid_matrix_norm).argmin(axis=1)
    assigned_cluster = centroid_labels[nearest_centroid]
    distance_to_centroid = np.linalg.norm(sim_rep - centroid_matrix[nearest_centroid], axis=1)
    normalized_distance_to_centroid = np.linalg.norm(sim_rep_norm - centroid_matrix_norm[nearest_centroid], axis=1)
    observed_clusters = adata.obs[cluster_key].astype(str)
    observed_distance = pd.Series(observed_features["normalized_distance_to_cluster_centroid"], index=adata.obs_names)
    distance_z = _cluster_z_against_observed(normalized_distance_to_centroid, assigned_cluster, observed_clusters, observed_distance)
    neighbor_consistency = _reference_neighbor_consistency(sim_rep_norm, observed_rep_norm, observed_clusters.to_numpy(dtype=str), assigned_cluster)

    heter_mask = simulated_doublets.doublet_type == "heterotypic"
    homo_mask = simulated_doublets.doublet_type == "homotypic"
    nearest_heter = _nearest_distance(sim_rep_norm, sim_rep_norm[heter_mask])
    nearest_homo = _nearest_distance(sim_rep_norm, sim_rep_norm[homo_mask])
    n_counts, n_genes = _count_metrics_from_matrix(simulated_doublets.X)
    complexity = safe_divide(n_genes, n_counts)

    groups = get_group_key_frame(adata, cluster_key=cluster_key, library_key=library_key)
    observed_log_counts = pd.Series(observed_features["log1p_n_counts"], index=adata.obs_names)
    observed_log_genes = pd.Series(observed_features["log1p_n_genes"], index=adata.obs_names)
    sim_log_counts = np.log1p(n_counts)
    sim_log_genes = np.log1p(n_genes)
    sim_libraries = simulated_doublets.library.astype(str)
    count_z = _dosage_z_against_observed(sim_log_counts, assigned_cluster, sim_libraries, groups, observed_log_counts)
    gene_z = _dosage_z_against_observed(sim_log_genes, assigned_cluster, sim_libraries, groups, observed_log_genes)
    stable_z = count_z.copy()
    marker_z = _dosage_z_against_observed(sim_log_counts, assigned_cluster, sim_libraries, groups, pd.Series(observed_features["cluster_marker_dosage_score"], index=adata.obs_names))

    cluster_abundance_map = observed_features.groupby(adata.obs[cluster_key].astype(str))["cluster_abundance"].median()
    abundance = pd.Series(assigned_cluster).map(cluster_abundance_map).fillna(cluster_abundance_map.median() if len(cluster_abundance_map) else 0.0).to_numpy()

    features = pd.DataFrame(index=[f"sim_{i}" for i in range(len(simulated_doublets))])
    for i in range(observed_features.filter(regex=r"^pca_").shape[1]):
        features[f"pca_{i}"] = sim_rep[:, i] if i < sim_rep.shape[1] else 0.0
    features["distance_to_cluster_centroid"] = distance_to_centroid
    features["normalized_distance_to_cluster_centroid"] = normalized_distance_to_centroid
    features["nearest_heterotypic_doublet_distance"] = nearest_heter
    features["nearest_homotypic_doublet_distance"] = nearest_homo
    features["heterotypic_similarity"] = 1.0 / (1.0 + nearest_heter)
    features["homotypic_similarity"] = 1.0 / (1.0 + nearest_homo)
    features["artificial_neighbor_fraction"] = 1.0
    features["log1p_n_counts"] = sim_log_counts
    features["log1p_n_genes"] = sim_log_genes
    features["complexity"] = complexity
    features["mito_fraction"] = 0.0
    features["ribosomal_fraction"] = 0.0
    features["stable_gene_module_score"] = sim_log_counts
    features["cluster_marker_dosage_score"] = sim_log_counts
    features["cluster_count_robust_z"] = count_z
    features["cluster_gene_robust_z"] = gene_z
    features["cluster_stable_dosage_robust_z"] = stable_z
    features["cluster_marker_dosage_robust_z"] = marker_z
    features["dosage_residual"] = 0.5 * count_z + 0.3 * gene_z + 0.2 * stable_z
    features["cluster_abundance"] = abundance
    features["cluster_level_expected_homotypic_burden"] = np.square(abundance)
    features["population_propensity_prior"] = 1.0
    features["cluster_distance_to_likely_parent"] = 0.0
    features["duodose_low_quality_score"] = 0.0
    homotypic_features = _build_homotypic_discrimination_features(
        features.index,
        features["cluster_count_robust_z"].to_numpy(dtype=float),
        features["cluster_gene_robust_z"].to_numpy(dtype=float),
        features["cluster_stable_dosage_robust_z"].to_numpy(dtype=float),
        marker_z,
        distance_z,
        neighbor_consistency,
        features["homotypic_similarity"].to_numpy(dtype=float),
        features["heterotypic_similarity"].to_numpy(dtype=float),
        _rank01_array(sim_log_counts),
    )
    for column in homotypic_features:
        features[column] = homotypic_features[column].to_numpy(dtype=float)

    return features.reindex(columns=observed_features.columns, fill_value=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def infer_parent_clusters(
    adata: AnnData,
    simulated_doublets: SimulatedDoublets,
    use_rep: str = "X_pca",
) -> tuple[np.ndarray, np.ndarray]:
    """Infer likely parent clusters from the nearest simulated doublet."""

    if len(simulated_doublets) == 0:
        empty = np.array([""] * adata.n_obs, dtype=object)
        return empty, empty.copy()
    observed_rep = np.asarray(adata.obsm[use_rep], dtype=float)
    sim_rep = transform_counts_to_pca(adata, simulated_doublets.X)
    nn = NearestNeighbors(n_neighbors=1).fit(sim_rep)
    _, indices = nn.kneighbors(observed_rep)
    idx = indices[:, 0]
    return simulated_doublets.parent1[idx].astype(object), simulated_doublets.parent2[idx].astype(object)
