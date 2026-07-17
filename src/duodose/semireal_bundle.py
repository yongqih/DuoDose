"""Parent-disjoint semi-real data construction with configurable parent handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from duodose.semireal_construction import (
    DEFAULT_CONSTRUCTION_VARIANT,
    EmpiricalUpperTailLibrarySampler,
    construct_semireal_doublet,
    resolve_construction_config,
)


def canonical_parent_pair(parent_1_id: object, parent_2_id: object) -> tuple[str, str]:
    """Return a deterministic unordered representation of one parent pair."""

    parent_1 = str(parent_1_id)
    parent_2 = str(parent_2_id)
    return (parent_1, parent_2) if parent_1 <= parent_2 else (parent_2, parent_1)


def _draw_unused_parent_pair(
    draw_pair,
    used_pairs: set[tuple[str, str]],
    parent_ids: np.ndarray,
    *,
    context: str,
    max_attempts: int = 10_000,
) -> tuple[int, int]:
    """Draw one unused unordered pair without changing the sampling rule."""

    for _ in range(int(max_attempts)):
        parent_1, parent_2 = map(int, draw_pair())
        pair = canonical_parent_pair(parent_ids[parent_1], parent_ids[parent_2])
        if pair not in used_pairs:
            used_pairs.add(pair)
            return parent_1, parent_2
    raise ValueError(f"{context}: could not draw a unique canonical parent pair after {max_attempts} attempts")


@dataclass
class SemiRealClusterProjection:
    gene_names: pd.Index
    pca_model: object
    scaler: StandardScaler
    cluster_model: KMeans


@dataclass
class SemiRealSplitBundle:
    dataset: str
    seed: int
    fit_adata: AnnData
    val_adata: AnnData
    test_adata: AnnData
    construction_report: dict[str, object]
    parent_audit: dict[str, object]
    parent_map: pd.DataFrame
    reference_cell_ids: pd.Index
    synthetic_parent_cell_ids: pd.Index
    cluster_projection: SemiRealClusterProjection


def _counts_matrix(adata: AnnData) -> sparse.csr_matrix:
    matrix = adata.layers["counts"] if "counts" in adata.layers else adata.X
    return matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(matrix)


def _ensure_counts_layer(adata: AnnData) -> AnnData:
    work = adata.copy()
    if "counts" not in work.layers:
        work.layers["counts"] = work.X.copy()
    return work


def _normalized_log_matrix(counts: sparse.csr_matrix, target_sum: float = 1e4):
    matrix = counts.tocsr().astype(np.float32).copy()
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.divide(target_sum, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)
    matrix = sparse.diags(scale).dot(matrix).tocsr()
    matrix.data = np.log1p(matrix.data)
    return matrix


def _stratified_index_split(indices: np.ndarray, clusters: np.ndarray, *, test_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 2:
        raise ValueError("semi-real parent/reference pool is too small to split")
    labels = np.asarray(clusters)[indices]
    counts = pd.Series(labels).value_counts()
    stratify = labels if len(counts) > 1 and int(counts.min()) >= 2 else None
    try:
        first, second = train_test_split(indices, test_size=float(test_size), random_state=int(random_state), stratify=stratify)
    except ValueError:
        first, second = train_test_split(indices, test_size=float(test_size), random_state=int(random_state), stratify=None)
    return np.asarray(first, dtype=int), np.asarray(second, dtype=int)


def _cluster_reference_and_project_parents(
    reference: AnnData,
    parents: AnnData,
    *,
    n_clusters: int,
    random_state: int,
) -> tuple[AnnData, AnnData, SemiRealClusterProjection]:
    """Fit normalization/PCA/scaling/KMeans on reference singlets only."""

    reference = reference.copy()
    parents = parents.copy()
    ref_counts = _counts_matrix(reference)
    parent_counts = _counts_matrix(parents)
    if reference.n_obs < 3 or reference.n_vars < 3:
        raise ValueError("too few reference singlets or genes for semi-real clustering")
    n_components = int(max(2, min(30, reference.n_obs - 1, reference.n_vars - 1)))
    ref_normalized = _normalized_log_matrix(ref_counts)
    parent_normalized = _normalized_log_matrix(parent_counts)
    if sparse.issparse(ref_normalized):
        pca = TruncatedSVD(n_components=n_components, random_state=int(random_state))
    else:  # pragma: no cover - _counts_matrix normalizes to CSR
        pca = PCA(n_components=n_components, random_state=int(random_state))
    pca.fit(ref_normalized)
    ref_rep = pca.transform(ref_normalized)
    parent_rep = pca.transform(parent_normalized)
    scaler = StandardScaler().fit(ref_rep)
    ref_rep = scaler.transform(ref_rep)
    parent_rep = scaler.transform(parent_rep)
    k = int(min(max(2, int(n_clusters)), max(2, reference.n_obs // 25)))
    clusterer = KMeans(n_clusters=k, n_init=20, random_state=int(random_state)).fit(ref_rep)
    reference.obs["semireal_cluster"] = [f"cluster_{value}" for value in clusterer.predict(ref_rep)]
    parents.obs["semireal_cluster"] = [f"cluster_{value}" for value in clusterer.predict(parent_rep)]
    projection = SemiRealClusterProjection(
        gene_names=reference.var_names.copy(),
        pca_model=pca,
        scaler=scaler,
        cluster_model=clusterer,
    )
    return reference, parents, projection


def project_fully_real_cells(
    adata: AnnData,
    projection: SemiRealClusterProjection,
    *,
    dataset: str,
) -> AnnData:
    """Project original real cells into the reference-fitted cluster system."""

    work = _ensure_counts_layer(adata)
    if not work.var_names.equals(projection.gene_names):
        raise ValueError("fully real cells do not match the reference-fitted gene order")
    normalized = _normalized_log_matrix(_counts_matrix(work))
    representation = projection.scaler.transform(projection.pca_model.transform(normalized))
    labels = pd.Series(
        [f"cluster_{value}" for value in projection.cluster_model.predict(representation)],
        index=work.obs_names,
        dtype=str,
    )
    for column in ("semireal_cluster", "benchmark_cluster", "duodose_cluster"):
        work.obs[column] = labels
    if "sample_id" not in work.obs:
        work.obs["sample_id"] = dataset
    if "library" not in work.obs:
        work.obs["library"] = work.obs["sample_id"].astype(str)
    return work


def _prepare_split_adata(
    reference_background: AnnData,
    synthetic_parent_background: AnnData,
    reference_indices: np.ndarray,
    parent_indices: np.ndarray,
    *,
    dataset: str,
    split_name: str,
    n_homotypic_doublets: int,
    n_heterotypic_doublets: int,
    high_rna_quantile: float,
    min_cluster_size: int,
    construction_mode: str,
    parent_reference_mode: str,
    downsampled_library_lower_quantile: float,
    downsampled_library_upper_quantile: float,
    random_state: int,
    shared_parent_pairs: pd.DataFrame | None = None,
) -> tuple[AnnData, pd.DataFrame]:
    rng = np.random.default_rng(int(random_state))
    reference_indices = np.asarray(reference_indices, dtype=int)
    parent_indices = np.asarray(parent_indices, dtype=int)
    reference_counts = _counts_matrix(reference_background)
    parent_counts = _counts_matrix(synthetic_parent_background)
    reference_obs = reference_background.obs.iloc[reference_indices].copy()
    parent_clusters_all = synthetic_parent_background.obs["semireal_cluster"].astype(str).to_numpy()
    parent_ids_all = synthetic_parent_background.obs_names.astype(str).to_numpy()
    reference_ids = set(reference_background.obs_names.astype(str))

    split_counts = reference_counts[reference_indices, :].tocsr()
    split_obs = reference_obs.copy()
    split_obs["semireal_split"] = split_name
    split_obs["semireal_origin"] = "observed_background"
    split_obs["experimental_doublet"] = 0
    split_obs["parent_cell_id"] = split_obs.index.astype(str)
    split_obs["parent1_id"] = ""
    split_obs["parent2_id"] = ""
    split_obs["parent_cluster1"] = ""
    split_obs["parent_cluster2"] = ""
    split_obs["doublet_subtype"] = ""
    split_obs["true_doublet_label"] = "clean"
    split_obs["count_construction_mode"] = construction_mode
    split_obs["parent_reference_mode"] = parent_reference_mode
    split_obs["raw_parent_sum_library_size"] = np.nan
    split_obs["target_library_size"] = np.nan
    split_obs["retention_fraction"] = np.nan
    split_obs["construction_random_seed"] = np.nan
    split_obs["benchmark_cluster"] = split_obs["semireal_cluster"].astype(str)
    split_obs["duodose_cluster"] = split_obs["semireal_cluster"].astype(str)
    if "sample_id" not in split_obs:
        split_obs["sample_id"] = dataset
    if "library" not in split_obs:
        split_obs["library"] = split_obs["sample_id"].astype(str)

    n_counts = np.asarray(split_counts.sum(axis=1)).ravel().astype(float)
    clusters = split_obs["semireal_cluster"].astype(str).to_numpy()
    high_rna = np.zeros(len(split_obs), dtype=bool)
    for cluster in sorted(pd.unique(clusters)):
        local = np.flatnonzero(clusters == cluster)
        if len(local):
            cutoff = float(np.quantile(np.log1p(n_counts[local]), float(high_rna_quantile)))
            high_rna[local] = np.log1p(n_counts[local]) >= cutoff
    split_obs["is_high_rna_singlet"] = high_rna
    split_obs["benchmark_cell_type"] = np.where(high_rna, "high_RNA_singlet", "singlet")
    split_obs["true_label"] = np.where(high_rna, "high_RNA_singlet", "clean")

    planned_pairs = (
        _pair_plan_identity(shared_parent_pairs).loc[lambda frame: frame["split"].eq(split_name)].copy()
        if shared_parent_pairs is not None
        else pd.DataFrame(columns=PARENT_PAIR_PLAN_COLUMNS)
    )
    parent_index_by_id = {str(cell_id): index for index, cell_id in enumerate(parent_ids_all)}
    if not planned_pairs.empty:
        planned_ids = set(planned_pairs["parent_1_id"]) | set(planned_pairs["parent_2_id"])
        missing = sorted(planned_ids - set(parent_index_by_id))
        if missing:
            raise AssertionError(f"{dataset}/{split_name}: shared plan parents are absent from the synthetic parent pool")
        pools: dict[str, np.ndarray] = {}
        valid_homo = np.array([], dtype=object)
        valid_hetero = np.array([], dtype=object)
    else:
        pools = {}
        for cluster in sorted(pd.unique(parent_clusters_all[parent_indices])):
            pools[str(cluster)] = np.asarray([idx for idx in parent_indices if parent_clusters_all[idx] == cluster], dtype=int)
        valid_homo = np.array([cluster for cluster, values in pools.items() if len(values) >= max(2, int(min_cluster_size))])
        valid_hetero = np.array([cluster for cluster, values in pools.items() if len(values) >= 1])
        if len(valid_homo) == 0 and int(n_homotypic_doublets) > 0:
            raise ValueError(f"{dataset}/{split_name}: no parent cluster can construct homotypic doublets")
        if len(valid_hetero) < 2 and int(n_heterotypic_doublets) > 0:
            raise ValueError(f"{dataset}/{split_name}: fewer than two parent clusters can construct heterotypic doublets")

    sampler = None
    if construction_mode == "downsampled":
        sampler = EmpiricalUpperTailLibrarySampler.from_reference_counts(
            split_counts,
            lower_quantile=float(downsampled_library_lower_quantile),
            upper_quantile=float(downsampled_library_upper_quantile),
        )

    doublet_rows: list[sparse.csr_matrix] = []
    obs_rows: list[dict[str, object]] = []
    map_rows: list[dict[str, object]] = []
    used_pairs: set[tuple[str, str]] = set()

    def add_doublet(parent_i: int, parent_j: int, subtype: str, synthetic_id: str | None = None) -> None:
        synthetic, metadata = construct_semireal_doublet(
            parent_counts.getrow(parent_i),
            parent_counts.getrow(parent_j),
            construction_mode,
            sampler,
            rng,
            random_seed=int(random_state),
        )
        doublet_rows.append(synthetic)
        cell_number = len(obs_rows)
        synthetic_id = synthetic_id or f"{dataset}_{split_name}_semireal_doublet_{cell_number:05d}"
        parent_1_id = str(parent_ids_all[parent_i])
        parent_2_id = str(parent_ids_all[parent_j])
        cluster_1 = str(parent_clusters_all[parent_i])
        cluster_2 = str(parent_clusters_all[parent_j])
        parent_1_ncount = int(np.asarray(parent_counts.getrow(parent_i).sum()).item())
        parent_2_ncount = int(np.asarray(parent_counts.getrow(parent_j).sum()).item())
        parent_row = {
            "synthetic_cell_id": synthetic_id,
            "split": split_name,
            "synthetic_subtype": subtype,
            "parent_1_id": parent_1_id,
            "parent_2_id": parent_2_id,
            "parent_1_cluster": cluster_1,
            "parent_2_cluster": cluster_2,
            "parent_1_nCount": parent_1_ncount,
            "parent_2_nCount": parent_2_ncount,
            "parent_1_in_reference": bool(parent_1_id in reference_ids),
            "parent_2_in_reference": bool(parent_2_id in reference_ids),
            "parent_reference_mode": parent_reference_mode,
            **metadata,
        }
        map_rows.append(parent_row)
        obs_rows.append(
            {
                "experimental_doublet": 1,
                "semireal_split": split_name,
                "semireal_origin": "constructed_doublet",
                "benchmark_cell_type": "semireal_doublet",
                "is_high_rna_singlet": False,
                "semireal_cluster": cluster_1,
                "benchmark_cluster": cluster_1 if subtype == "homotypic" else f"{cluster_1}|{cluster_2}",
                "duodose_cluster": cluster_1,
                "true_label": f"{subtype}_doublet",
                "true_doublet_label": f"{subtype}_doublet",
                "doublet_subtype": subtype,
                "parent_cell_id": "",
                "parent1_id": parent_1_id,
                "parent2_id": parent_2_id,
                "parent_cluster1": cluster_1,
                "parent_cluster2": cluster_2,
                "doublet_parent_cluster_1": cluster_1,
                "doublet_parent_cluster_2": cluster_2,
                "doublet_saturation": metadata["retention_fraction"],
                "count_construction_mode": construction_mode,
                "parent_reference_mode": parent_reference_mode,
                "raw_parent_sum_library_size": metadata["raw_parent_sum_library_size"],
                "target_library_size": metadata["target_library_size"],
                "retention_fraction": metadata["retention_fraction"],
                "construction_random_seed": metadata["random_seed"],
                "sample_id": str(split_obs.iloc[0].get("sample_id", dataset)) if len(split_obs) else dataset,
                "library": str(split_obs.iloc[0].get("library", dataset)) if len(split_obs) else dataset,
            }
        )

    if not planned_pairs.empty:
        for row in planned_pairs.itertuples(index=False):
            add_doublet(
                parent_index_by_id[str(row.parent_1_id)],
                parent_index_by_id[str(row.parent_2_id)],
                str(row.synthetic_subtype),
                synthetic_id=str(row.synthetic_cell_id),
            )
    else:
        for _ in range(int(n_homotypic_doublets)):
            def draw_homotypic() -> tuple[int, int]:
                cluster = str(rng.choice(valid_homo))
                parent_1, parent_2 = rng.choice(pools[cluster], size=2, replace=False)
                return int(parent_1), int(parent_2)

            parent_1, parent_2 = _draw_unused_parent_pair(
                draw_homotypic,
                used_pairs,
                parent_ids_all,
                context=f"{dataset}/{split_name}/homotypic",
            )
            add_doublet(int(parent_1), int(parent_2), "homotypic")
        for _ in range(int(n_heterotypic_doublets)):
            def draw_heterotypic() -> tuple[int, int]:
                cluster_1, cluster_2 = rng.choice(valid_hetero, size=2, replace=False)
                return int(rng.choice(pools[str(cluster_1)])), int(rng.choice(pools[str(cluster_2)]))

            parent_1, parent_2 = _draw_unused_parent_pair(
                draw_heterotypic,
                used_pairs,
                parent_ids_all,
                context=f"{dataset}/{split_name}/heterotypic",
            )
            add_doublet(parent_1, parent_2, "heterotypic")

    if obs_rows:
        doublet_obs = pd.DataFrame(obs_rows, index=[row["synthetic_cell_id"] for row in map_rows])
        output_counts = sparse.vstack([split_counts, sparse.vstack(doublet_rows, format="csr")], format="csr")
        output_obs = pd.concat([split_obs, doublet_obs], axis=0, sort=False)
    else:
        output_counts = split_counts
        output_obs = split_obs
    output_obs["experimental_doublet"] = output_obs["experimental_doublet"].astype(int)
    output_obs["is_high_rna_singlet"] = output_obs["is_high_rna_singlet"].fillna(False).astype(bool)
    output = AnnData(X=output_counts, obs=output_obs, var=reference_background.var.copy())
    output.layers["counts"] = output.X.copy()
    return output, pd.DataFrame(map_rows)


def _parent_ids(parent_map: pd.DataFrame, split: str) -> set[str]:
    rows = parent_map.loc[parent_map["split"].eq(split)] if not parent_map.empty else parent_map
    return set(rows.get("parent_1_id", pd.Series(dtype=str)).astype(str)) | set(rows.get("parent_2_id", pd.Series(dtype=str)).astype(str))


PARENT_PAIR_PLAN_COLUMNS = [
    "synthetic_cell_id",
    "split",
    "synthetic_subtype",
    "parent_1_id",
    "parent_2_id",
]


def _pair_plan_identity(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in PARENT_PAIR_PLAN_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"shared parent-pair plan is missing columns: {', '.join(missing)}")
    out = frame.loc[:, PARENT_PAIR_PLAN_COLUMNS].copy()
    for column in PARENT_PAIR_PLAN_COLUMNS:
        out[column] = out[column].astype(str)
    if out["synthetic_cell_id"].duplicated().any():
        raise ValueError("shared parent-pair plan contains duplicate synthetic_cell_id values")
    if ~out["split"].isin(["train", "validation", "test"]).all():
        raise ValueError("shared parent-pair plan contains an unsupported split")
    if ~out["synthetic_subtype"].isin(["homotypic", "heterotypic"]).all():
        raise ValueError("shared parent-pair plan contains an unsupported synthetic subtype")
    canonical = out.apply(
        lambda row: canonical_parent_pair(row["parent_1_id"], row["parent_2_id"]),
        axis=1,
        result_type="expand",
    )
    canonical.columns = ["canonical_parent_1", "canonical_parent_2"]
    out = pd.concat([out, canonical], axis=1)
    if out.duplicated(["canonical_parent_1", "canonical_parent_2"]).any():
        raise ValueError("shared parent-pair plan contains a duplicate canonical unordered parent pair")
    split_memberships: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        rows = out.loc[out["split"].eq(split)]
        split_memberships[split] = set(rows["parent_1_id"]) | set(rows["parent_2_id"])
    if any(
        split_memberships[left] & split_memberships[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("shared parent-pair plan contains cross-split parent overlap")
    out = out.drop(columns=["canonical_parent_1", "canonical_parent_2"])
    return out.sort_values("synthetic_cell_id", kind="stable").reset_index(drop=True)


def generate_shared_parent_pair_plan(
    adata: AnnData,
    *,
    dataset: str,
    seed: int,
    n_singlets: int,
    n_train_homotypic_doublets: int,
    n_train_heterotypic_doublets: int,
    n_test_homotypic_doublets: int,
    n_test_heterotypic_doublets: int,
    n_clusters: int,
    test_parent_fraction: float,
    validation_parent_fraction: float,
    min_cluster_size: int,
) -> pd.DataFrame:
    """Generate one deterministic parent-pair plan shared by all variants."""

    work = _ensure_counts_layer(adata)
    if "experimental_doublet" not in work.obs:
        raise ValueError("input AnnData needs experimental_doublet labels")
    singlet_indices = np.flatnonzero(work.obs["experimental_doublet"].astype(int).eq(0).to_numpy())
    if len(singlet_indices) < max(200, 3 * int(min_cluster_size)):
        raise ValueError(f"too few real singlets for shared parent-pair planning: {len(singlet_indices)}")
    selection_rng = np.random.default_rng(int(seed))
    selected = singlet_indices
    if len(selected) > int(n_singlets):
        selected = selection_rng.choice(selected, size=int(n_singlets), replace=False)
    planning_background = work[np.asarray(selected, dtype=int), :].copy()
    planning_background, _, _ = _cluster_reference_and_project_parents(
        planning_background,
        planning_background,
        n_clusters=int(n_clusters),
        random_state=int(seed),
    )
    clusters = planning_background.obs["semireal_cluster"].astype(str).to_numpy()
    all_indices = np.arange(planning_background.n_obs, dtype=int)
    fit_indices, test_indices = _stratified_index_split(
        all_indices,
        clusters,
        test_size=float(test_parent_fraction),
        random_state=int(seed),
    )
    fit_indices, validation_indices = _stratified_index_split(
        fit_indices,
        clusters,
        test_size=float(validation_parent_fraction),
        random_state=int(seed) + 1009,
    )
    n_validation_homo = max(1, int(round(float(n_train_homotypic_doublets) * float(validation_parent_fraction))))
    n_validation_hetero = max(1, int(round(float(n_train_heterotypic_doublets) * float(validation_parent_fraction))))
    split_specs = [
        ("train", fit_indices, max(1, int(n_train_homotypic_doublets) - n_validation_homo), max(1, int(n_train_heterotypic_doublets) - n_validation_hetero)),
        ("validation", validation_indices, n_validation_homo, n_validation_hetero),
        ("test", test_indices, int(n_test_homotypic_doublets), int(n_test_heterotypic_doublets)),
    ]
    pair_rng = np.random.default_rng(int(seed) + 7001)
    rows: list[dict[str, object]] = []
    ids = planning_background.obs_names.astype(str).to_numpy()
    used_pairs: set[tuple[str, str]] = set()
    for split_name, split_indices, n_homo, n_hetero in split_specs:
        pools = {
            str(cluster): np.asarray([index for index in split_indices if clusters[index] == cluster], dtype=int)
            for cluster in sorted(pd.unique(clusters[split_indices]))
        }
        valid_homo = np.array([cluster for cluster, values in pools.items() if len(values) >= max(2, int(min_cluster_size))])
        valid_hetero = np.array([cluster for cluster, values in pools.items() if len(values) >= 1])
        if len(valid_homo) == 0 and n_homo > 0:
            raise ValueError(f"{dataset}/{split_name}: no parent cluster can create homotypic pair-plan rows")
        if len(valid_hetero) < 2 and n_hetero > 0:
            raise ValueError(f"{dataset}/{split_name}: fewer than two parent clusters can create heterotypic pair-plan rows")

        def append_row(parent_1: int, parent_2: int, subtype: str) -> None:
            rows.append(
                {
                    "synthetic_cell_id": f"{dataset}_{split_name}_semireal_doublet_{len([row for row in rows if row['split'] == split_name]):05d}",
                    "split": split_name,
                    "synthetic_subtype": subtype,
                    "parent_1_id": str(ids[parent_1]),
                    "parent_2_id": str(ids[parent_2]),
                    "dataset": dataset,
                    "seed": int(seed),
                }
            )

        for _ in range(n_homo):
            def draw_homotypic() -> tuple[int, int]:
                cluster = str(pair_rng.choice(valid_homo))
                parent_1, parent_2 = pair_rng.choice(pools[cluster], size=2, replace=False)
                return int(parent_1), int(parent_2)

            parent_1, parent_2 = _draw_unused_parent_pair(
                draw_homotypic,
                used_pairs,
                ids,
                context=f"{dataset}/{split_name}/shared-plan/homotypic",
            )
            append_row(parent_1, parent_2, "homotypic")
        for _ in range(n_hetero):
            def draw_heterotypic() -> tuple[int, int]:
                cluster_1, cluster_2 = pair_rng.choice(valid_hetero, size=2, replace=False)
                return int(pair_rng.choice(pools[str(cluster_1)])), int(pair_rng.choice(pools[str(cluster_2)]))

            parent_1, parent_2 = _draw_unused_parent_pair(
                draw_heterotypic,
                used_pairs,
                ids,
                context=f"{dataset}/{split_name}/shared-plan/heterotypic",
            )
            append_row(parent_1, parent_2, "heterotypic")
    return _pair_plan_identity(pd.DataFrame(rows))


def make_parent_disjoint_semireal_bundle(
    adata: AnnData,
    *,
    dataset: str,
    seed: int,
    n_singlets: int,
    n_train_homotypic_doublets: int,
    n_train_heterotypic_doublets: int,
    n_test_homotypic_doublets: int,
    n_test_heterotypic_doublets: int,
    n_clusters: int,
    test_parent_fraction: float,
    validation_parent_fraction: float,
    high_rna_quantile: float,
    saturation_range: tuple[float, float] = (0.6, 1.0),
    min_cluster_size: int = 10,
    construction_variant: str = DEFAULT_CONSTRUCTION_VARIANT,
    n_validation_homotypic_doublets: int | None = None,
    n_validation_heterotypic_doublets: int | None = None,
    count_construction_mode: str | None = None,
    parent_reference_mode: str | None = None,
    downsampled_library_lower_quantile: float = 0.70,
    downsampled_library_upper_quantile: float = 0.995,
    shared_parent_pair_plan: pd.DataFrame | None = None,
) -> SemiRealSplitBundle:
    """Create parent-disjoint semi-real splits from raw experimental singlets."""

    config = resolve_construction_config(construction_variant, count_construction_mode, parent_reference_mode)
    rng = np.random.default_rng(int(seed))
    work = _ensure_counts_layer(adata)
    if "experimental_doublet" not in work.obs:
        raise ValueError("input AnnData needs experimental_doublet labels")
    singlet_indices = np.flatnonzero(work.obs["experimental_doublet"].astype(int).eq(0).to_numpy())
    if len(singlet_indices) < max(200, 3 * int(min_cluster_size)):
        raise ValueError(f"too few real singlets for semi-real construction: {len(singlet_indices)}")
    shared_plan = _pair_plan_identity(shared_parent_pair_plan) if shared_parent_pair_plan is not None else None
    cell_index_by_id = {str(cell_id): index for index, cell_id in enumerate(work.obs_names.astype(str))}
    planned_parent_ids: list[str] = []
    if shared_plan is not None:
        planned_parent_ids = list(dict.fromkeys([*shared_plan["parent_1_id"], *shared_plan["parent_2_id"]]))
        missing = sorted(set(planned_parent_ids) - set(work.obs_names.astype(str)))
        if missing:
            raise ValueError("shared parent-pair plan contains cell IDs absent from the input AnnData")
        planned_indices = np.asarray([cell_index_by_id[cell_id] for cell_id in planned_parent_ids], dtype=int)
        if not np.all(work.obs.iloc[planned_indices]["experimental_doublet"].astype(int).eq(0)):
            raise ValueError("shared parent-pair plan contains an experimentally annotated doublet parent")
    else:
        planned_indices = np.array([], dtype=int)

    if config.parent_reference_mode == "retained":
        if shared_plan is None:
            selected = singlet_indices
            if len(selected) > int(n_singlets):
                selected = rng.choice(selected, size=int(n_singlets), replace=False)
        else:
            target = max(int(n_singlets), len(planned_indices))
            extras = np.setdiff1d(singlet_indices, planned_indices, assume_unique=False)
            fill_count = min(max(0, target - len(planned_indices)), len(extras))
            fill = rng.choice(extras, size=fill_count, replace=False) if fill_count else np.array([], dtype=int)
            selected = np.concatenate([planned_indices, fill])
        reference_background = work[np.asarray(selected, dtype=int), :].copy()
        synthetic_parent_background = reference_background
    else:
        if shared_plan is None:
            requested_reference = min(int(n_singlets), len(singlet_indices))
            requested_total = min(len(singlet_indices), 2 * requested_reference)
            if requested_total <= requested_reference:
                raise ValueError("parents_removed requires additional labeled singlets beyond the reference pool")
            selected = rng.choice(singlet_indices, size=requested_total, replace=False)
            reference_indices = selected[:requested_reference]
            parent_indices_for_background = selected[requested_reference:]
        else:
            reference_candidates = np.setdiff1d(singlet_indices, planned_indices, assume_unique=False)
            requested_reference = min(int(n_singlets), len(reference_candidates))
            if requested_reference < max(3 * int(min_cluster_size), 3):
                raise ValueError("parents_removed lacks a disjoint reference singlet pool for the shared parent-pair plan")
            reference_indices = rng.choice(reference_candidates, size=requested_reference, replace=False)
            parent_candidates = np.setdiff1d(reference_candidates, reference_indices, assume_unique=False)
            target_parent_count = max(len(planned_indices), min(int(n_singlets), len(planned_indices) + len(parent_candidates)))
            fill_count = max(0, target_parent_count - len(planned_indices))
            fill = rng.choice(parent_candidates, size=fill_count, replace=False) if fill_count else np.array([], dtype=int)
            parent_indices_for_background = np.concatenate([planned_indices, fill])
        reference_background = work[reference_indices, :].copy()
        synthetic_parent_background = work[parent_indices_for_background, :].copy()
        if synthetic_parent_background.n_obs < max(3 * int(min_cluster_size), 6):
            raise ValueError("parents_removed has too few disjoint synthetic-parent singlets; lower --semireal-n-singlets")

    reference_background, synthetic_parent_background, cluster_projection = _cluster_reference_and_project_parents(
        reference_background,
        synthetic_parent_background,
        n_clusters=int(n_clusters),
        random_state=int(seed),
    )
    reference_clusters = reference_background.obs["semireal_cluster"].astype(str).to_numpy()
    parent_clusters = synthetic_parent_background.obs["semireal_cluster"].astype(str).to_numpy()
    all_reference_indices = np.arange(reference_background.n_obs, dtype=int)
    all_parent_indices = np.arange(synthetic_parent_background.n_obs, dtype=int)
    fit_ref, test_ref = _stratified_index_split(all_reference_indices, reference_clusters, test_size=float(test_parent_fraction), random_state=int(seed))
    fit_ref, val_ref = _stratified_index_split(fit_ref, reference_clusters, test_size=float(validation_parent_fraction), random_state=int(seed) + 1009)
    if shared_plan is not None:
        fit_parent = all_parent_indices
        val_parent = all_parent_indices
        test_parent = all_parent_indices
    elif config.parent_reference_mode == "retained":
        fit_parent, val_parent, test_parent = fit_ref, val_ref, test_ref
    else:
        fit_parent, test_parent = _stratified_index_split(all_parent_indices, parent_clusters, test_size=float(test_parent_fraction), random_state=int(seed) + 2003)
        fit_parent, val_parent = _stratified_index_split(fit_parent, parent_clusters, test_size=float(validation_parent_fraction), random_state=int(seed) + 3011)

    explicit_validation = n_validation_homotypic_doublets is not None or n_validation_heterotypic_doublets is not None
    if explicit_validation and (n_validation_homotypic_doublets is None or n_validation_heterotypic_doublets is None):
        raise ValueError("explicit validation sizing requires both homotypic and heterotypic counts")
    if explicit_validation:
        n_val_homo = max(1, int(n_validation_homotypic_doublets))
        n_val_hetero = max(1, int(n_validation_heterotypic_doublets))
        n_fit_homo = max(1, int(n_train_homotypic_doublets))
        n_fit_hetero = max(1, int(n_train_heterotypic_doublets))
    else:
        n_val_homo = max(1, int(round(float(n_train_homotypic_doublets) * float(validation_parent_fraction))))
        n_val_hetero = max(1, int(round(float(n_train_heterotypic_doublets) * float(validation_parent_fraction))))
        n_fit_homo = max(1, int(n_train_homotypic_doublets) - n_val_homo)
        n_fit_hetero = max(1, int(n_train_heterotypic_doublets) - n_val_hetero)
    shared = dict(
        dataset=dataset,
        high_rna_quantile=float(high_rna_quantile),
        min_cluster_size=int(min_cluster_size),
        construction_mode=config.count_construction_mode,
        parent_reference_mode=config.parent_reference_mode,
        downsampled_library_lower_quantile=float(downsampled_library_lower_quantile),
        downsampled_library_upper_quantile=float(downsampled_library_upper_quantile),
    )
    fit_adata, fit_map = _prepare_split_adata(reference_background, synthetic_parent_background, fit_ref, fit_parent, split_name="train", n_homotypic_doublets=n_fit_homo, n_heterotypic_doublets=n_fit_hetero, random_state=int(seed) + 11, shared_parent_pairs=shared_plan, **shared)
    val_adata, val_map = _prepare_split_adata(reference_background, synthetic_parent_background, val_ref, val_parent, split_name="validation", n_homotypic_doublets=n_val_homo, n_heterotypic_doublets=n_val_hetero, random_state=int(seed) + 23, shared_parent_pairs=shared_plan, **shared)
    test_adata, test_map = _prepare_split_adata(reference_background, synthetic_parent_background, test_ref, test_parent, split_name="test", n_homotypic_doublets=int(n_test_homotypic_doublets), n_heterotypic_doublets=int(n_test_heterotypic_doublets), random_state=int(seed) + 37, shared_parent_pairs=shared_plan, **shared)
    parent_map = pd.concat([fit_map, val_map, test_map], axis=0, ignore_index=True)
    canonical_pairs = parent_map.apply(
        lambda row: canonical_parent_pair(row["parent_1_id"], row["parent_2_id"]),
        axis=1,
        result_type="expand",
    )
    canonical_pairs.columns = ["canonical_parent_1", "canonical_parent_2"]
    parent_map = pd.concat([parent_map, canonical_pairs], axis=1)
    if parent_map["synthetic_cell_id"].astype(str).duplicated().any():
        raise AssertionError("constructed parent map contains duplicate generated-cell IDs")
    if parent_map.duplicated(["canonical_parent_1", "canonical_parent_2"]).any():
        raise AssertionError("constructed parent map contains duplicate canonical unordered parent pairs")
    if shared_plan is not None and not _pair_plan_identity(parent_map).equals(shared_plan):
        raise AssertionError("constructed parent map does not exactly match the shared parent-pair plan")

    if config.parent_reference_mode == "removed":
        reference_ids = set(reference_background.obs_names.astype(str))
        used_parents = _parent_ids(parent_map, "train") | _parent_ids(parent_map, "validation") | _parent_ids(parent_map, "test")
        if used_parents & reference_ids:
            raise AssertionError("parents_removed leaked an exact synthetic parent into the reference singlet pool")
        if not parent_map.empty and parent_map[["parent_1_in_reference", "parent_2_in_reference"]].to_numpy(dtype=bool).any():
            raise AssertionError("parents_removed parent map reports a parent in the reference pool")

    fit_parent_ids = _parent_ids(parent_map, "train")
    val_parent_ids = _parent_ids(parent_map, "validation")
    test_parent_ids = _parent_ids(parent_map, "test")
    parent_audit = {
        "dataset": dataset,
        "seed": int(seed),
        "parent_reference_mode": config.parent_reference_mode,
        "n_fit_parent_cells_used": int(len(fit_parent_ids)),
        "n_validation_parent_cells_used": int(len(val_parent_ids)),
        "n_test_parent_cells_used": int(len(test_parent_ids)),
        "train_validation_parent_overlap_fraction": float(len(fit_parent_ids & val_parent_ids) / max(1, len(fit_parent_ids | val_parent_ids))),
        "train_test_parent_overlap_fraction": float(len(fit_parent_ids & test_parent_ids) / max(1, len(fit_parent_ids | test_parent_ids))),
        "validation_test_parent_overlap_fraction": float(len(val_parent_ids & test_parent_ids) / max(1, len(val_parent_ids | test_parent_ids))),
        "reference_parent_overlap_count": int(len(set(reference_background.obs_names.astype(str)) & (fit_parent_ids | val_parent_ids | test_parent_ids))),
        "parent_leakage_audit_status": "passed" if not (fit_parent_ids & val_parent_ids or fit_parent_ids & test_parent_ids or val_parent_ids & test_parent_ids) else "failed",
    }
    construction_report = {
        "dataset": dataset,
        "seed": int(seed),
        "construction_variant": config.variant,
        "count_construction_mode": config.count_construction_mode,
        "parent_reference_mode": config.parent_reference_mode,
        "n_observed_background_cells_available": int(len(singlet_indices)),
        "n_reference_background_cells_used": int(reference_background.n_obs),
        "n_synthetic_parent_pool": int(synthetic_parent_background.n_obs),
        "n_genes": int(reference_background.n_vars),
        "n_clusters_requested": int(n_clusters),
        "n_reference_clusters_observed": int(pd.Series(reference_clusters).nunique()),
        "n_parent_clusters_observed": int(pd.Series(parent_clusters).nunique()),
        "n_fit_cells": int(fit_adata.n_obs),
        "n_validation_cells": int(val_adata.n_obs),
        "n_test_cells": int(test_adata.n_obs),
        "n_fit_homotypic_doublets": int(n_fit_homo),
        "n_fit_heterotypic_doublets": int(n_fit_hetero),
        "n_validation_homotypic_doublets": int(n_val_homo),
        "n_validation_heterotypic_doublets": int(n_val_hetero),
        "explicit_validation_sizes": bool(explicit_validation),
        "n_test_homotypic_doublets": int(n_test_homotypic_doublets),
        "n_test_heterotypic_doublets": int(n_test_heterotypic_doublets),
        "downsampled_library_lower_quantile": float(downsampled_library_lower_quantile),
        "downsampled_library_upper_quantile": float(downsampled_library_upper_quantile),
        "legacy_saturation_range_ignored": bool(config.count_construction_mode == "raw_sum"),
        "label_definition": (
            "positive=constructed homotypic/heterotypic doublets; negative=label-blinded observed background cells "
            "including high-RNA cells; experimental doublet contamination is possible"
        ),
        "high_rna_singlet_definition": (
            "generator-provided ground_truth_high_rna_singlet"
            if "ground_truth_high_rna_singlet" in reference_background.obs
            else "within-cluster observed-background upper RNA quantile"
        ),
    }
    return SemiRealSplitBundle(
        dataset,
        int(seed),
        fit_adata,
        val_adata,
        test_adata,
        construction_report,
        parent_audit,
        parent_map,
        reference_background.obs_names.copy(),
        synthetic_parent_background.obs_names.copy(),
        cluster_projection,
    )
