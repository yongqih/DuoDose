"""Parent-disjoint supervised semi-real benchmark for DuoDose.

This script uses real Xi & Li-style labeled singlets as the expression
background, constructs controlled homotypic/heterotypic doublets from
parent-disjoint train/validation/test parent pools, trains supervised
DuoDose SafeFeatures models on the train split, and evaluates only on the
held-out test split.

It intentionally does not use run_real_ml_dl_ablation_models from
run_xili_realdata_validation.py. That path is label-free real-data
diagnostics; this script is supervised semi-real validation.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from . import realdata as xili
from .methods import (
    DUODOSE_METHODS,
    DUODOSE_PUBLIC_METHOD_NAMES,
    resolve_duodose_cli_methods,
)
from .net import probabilities_to_scores, train_predict_diagnostic_model
from .semireal_metrics import high_rna_metric_bundle


SUPERVISED_METHODS = list(DUODOSE_METHODS)
PUBLIC_METHOD_NAME = DUODOSE_PUBLIC_METHOD_NAMES
DOUBLET_LABELS = {"homotypic_doublet", "heterotypic_doublet"}


@dataclass
class SemiRealSplitBundle:
    dataset: str
    seed: int
    fit_adata: AnnData
    val_adata: AnnData
    test_adata: AnnData
    construction_report: dict[str, object]
    parent_audit: dict[str, object]


def _split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]



def _as_csr(X) -> sparse.csr_matrix:
    return X.tocsr() if sparse.issparse(X) else sparse.csr_matrix(X)


def _counts_matrix(adata: AnnData) -> sparse.csr_matrix:
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    return _as_csr(X)


def _rank01(score: pd.Series) -> pd.Series:
    values = pd.to_numeric(score, errors="coerce").replace([np.inf, -np.inf], np.nan)
    fill = float(values.dropna().median()) if values.notna().any() else 0.0
    values = values.fillna(fill)
    if values.nunique(dropna=True) <= 1:
        return pd.Series(0.0, index=values.index, dtype=float)
    return values.rank(method="average", pct=True).clip(0.0, 1.0)


def _tail_rank01(score: pd.Series) -> pd.Series:
    ranks = _rank01(score)
    return pd.Series(np.maximum(0.0, (ranks.to_numpy(dtype=float) - 0.5) / 0.5), index=ranks.index, dtype=float).clip(0.0, 1.0)


def _calibrated_union(score_a: pd.Series, score_b: pd.Series) -> pd.Series:
    a = _tail_rank01(score_a)
    b = _tail_rank01(score_b)
    values = 1.0 - (1.0 - a.to_numpy(dtype=float)) * (1.0 - b.to_numpy(dtype=float))
    return pd.Series(values, index=a.index, dtype=float).clip(0.0, 1.0)


def _robust_z_by_group(values: pd.Series, groups: pd.Series) -> pd.Series:
    out = pd.Series(0.0, index=values.index, dtype=float)
    for _, idx in groups.groupby(groups).groups.items():
        local = values.loc[idx].astype(float)
        median = float(local.median())
        mad = float(np.median(np.abs(local.to_numpy(dtype=float) - median)))
        scale = 1.4826 * mad if mad > 0 else float(local.std(ddof=0) or 1.0)
        out.loc[idx] = (local - median) / scale
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _safe_metric(metric_fn, y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    valid = np.isfinite(s)
    if valid.sum() < 2 or len(np.unique(y[valid])) < 2:
        return float("nan")
    try:
        value = metric_fn(y[valid], s[valid])
    except Exception:
        return float("nan")
    return float(value) if np.isfinite(value) else float("nan")


def _top_k_metrics(y_true: np.ndarray, score: np.ndarray, k: int) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    valid = np.isfinite(s)
    if k <= 0 or valid.sum() == 0 or y.sum() == 0:
        return float("nan"), float("nan")
    order = np.argsort(-np.where(valid, s, -np.inf))[: min(k, int(valid.sum()))]
    tp = float(y[order].sum())
    precision = tp / float(len(order)) if len(order) else float("nan")
    recall = tp / float(y.sum()) if y.sum() else float("nan")
    return precision, recall


def _controlled_metrics(
    dataset: str,
    method: str,
    public_method_name: str,
    obs: pd.DataFrame,
    score: pd.Series,
    *,
    status: str = "success",
    message: str = "",
    runtime_sec: float = 0.0,
    seed: int,
    source_dataset: str,
) -> dict[str, object]:
    labels = obs["true_label"].astype(str)
    y = labels.isin(DOUBLET_LABELS).astype(int).to_numpy()
    score_array = pd.Series(score, index=obs.index).reindex(obs.index).to_numpy(dtype=float)
    n_doublets = int(y.sum())
    precision_at_k, recall_at_k = _top_k_metrics(y, score_array, n_doublets)

    clean_mask = ~labels.isin(DOUBLET_LABELS)
    hom_mask = labels.eq("homotypic_doublet")
    het_mask = labels.eq("heterotypic_doublet")
    high_mask = obs.get("is_high_rna_singlet", pd.Series(False, index=obs.index)).fillna(False).astype(bool)

    def subtype_auprc(pos_mask: pd.Series) -> float:
        mask = pos_mask | clean_mask
        yy = pos_mask.loc[mask].astype(int).to_numpy()
        ss = score_array[mask.to_numpy()]
        return _safe_metric(average_precision_score, yy, ss)

    hom_auprc = subtype_auprc(hom_mask)
    het_auprc = subtype_auprc(het_mask)
    finite_sub = [v for v in [hom_auprc, het_auprc] if np.isfinite(v)]
    macro_sub = float(np.mean(finite_sub)) if finite_sub else float("nan")

    hv_mask = hom_mask | high_mask
    if hom_mask.sum() > 0 and high_mask.sum() > 0:
        hom_vs_highrna = _safe_metric(
            average_precision_score,
            hom_mask.loc[hv_mask].astype(int).to_numpy(),
            score_array[hv_mask.to_numpy()],
        )
    else:
        hom_vs_highrna = float("nan")

    fpr_bundle = high_rna_metric_bundle(
        obs,
        pd.Series(score_array, index=obs.index),
        dataset=dataset,
        source_dataset=source_dataset,
        seed=int(seed),
        method=public_method_name,
    )

    return {
        "dataset": dataset,
        "source_dataset": source_dataset,
        "seed": int(seed),
        "method": method,
        "public_method_name": public_method_name,
        "AUROC": _safe_metric(roc_auc_score, y, score_array),
        "AUPRC": _safe_metric(average_precision_score, y, score_array),
        "average_precision": _safe_metric(average_precision_score, y, score_array),
        "precision_at_K": precision_at_k,
        "recall_at_K": recall_at_k,
        "homotypic_AUPRC": hom_auprc,
        "heterotypic_AUPRC": het_auprc,
        "macro_subtype_AUPRC": macro_sub,
        "homotypic_vs_high_RNA_AUPRC": hom_vs_highrna,
        "high_RNA_singlet_FPR": fpr_bundle["high_RNA_singlet_FPR"],
        "high_RNA_singlet_FPR_at_matched_50pct_homotypic_recall": fpr_bundle["high_RNA_singlet_FPR_at_matched_50pct_homotypic_recall"],
        "high_RNA_singlet_FPR_at_matched_70pct_homotypic_recall": fpr_bundle["high_RNA_singlet_FPR_at_matched_70pct_homotypic_recall"],
        "high_RNA_singlet_FPR_at_matched_80pct_homotypic_recall": fpr_bundle["high_RNA_singlet_FPR_at_matched_80pct_homotypic_recall"],
        "high_RNA_singlet_FPR_at_fixed_20pct_candidate_budget": fpr_bundle["high_RNA_singlet_FPR_at_fixed_20pct_candidate_budget"],
        "high_RNA_singlet_FPR_at_true_doublet_budget": fpr_bundle["high_RNA_singlet_FPR_at_true_doublet_budget"],
        "n_cells": int(len(obs)),
        "n_doublets": n_doublets,
        "n_singlets": int(clean_mask.sum()),
        "n_homotypic_doublets": int(hom_mask.sum()),
        "n_heterotypic_doublets": int(het_mask.sum()),
        "n_high_rna_singlets": int(high_mask.sum()),
        "status": status,
        "message": message,
        "runtime_sec": float(runtime_sec),
    }


def _downsample_row(row: sparse.csr_matrix, fraction: float, rng: np.random.Generator) -> sparse.csr_matrix:
    row = row.tocsr(copy=True)
    if row.nnz:
        row.data = rng.binomial(
            np.rint(row.data).clip(min=0).astype(np.int64),
            float(np.clip(fraction, 0.0, 1.0)),
        ).astype(np.float32)
        row.eliminate_zeros()
    return row


def _stratified_index_split(
    indices: np.ndarray,
    clusters: np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    labels = np.asarray(clusters)[indices]
    value_counts = pd.Series(labels).value_counts()
    stratify = labels if len(value_counts) > 1 and int(value_counts.min()) >= 2 else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=stratify,
    )
    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


def _cluster_real_singlets(background: AnnData, n_clusters: int, random_state: int) -> AnnData:
    work = background.copy()
    counts = _counts_matrix(work)
    rep = xili._pca_representation(counts, n_components=30, random_state=random_state)
    n_cells = work.n_obs
    k = int(min(max(2, int(n_clusters)), max(2, n_cells // 25)))
    labels = KMeans(n_clusters=k, n_init=20, random_state=random_state).fit_predict(rep)
    work.obs["semireal_cluster"] = pd.Series([f"cluster_{i}" for i in labels], index=work.obs_names, dtype=str)
    return work


def _make_split_adata(
    background: AnnData,
    parent_indices: np.ndarray,
    *,
    dataset: str,
    split_name: str,
    n_homotypic_doublets: int,
    n_heterotypic_doublets: int,
    high_rna_quantile: float,
    saturation_range: tuple[float, float],
    min_cluster_size: int,
    random_state: int,
) -> AnnData:
    rng = np.random.default_rng(int(random_state))
    parent_indices = np.asarray(parent_indices, dtype=int)
    counts = _counts_matrix(background)
    clusters_all = background.obs["semireal_cluster"].astype(str).to_numpy()

    split_counts = counts[parent_indices, :].tocsr()
    split_obs = background.obs.iloc[parent_indices].copy()
    split_obs["semireal_split"] = split_name
    split_obs["semireal_origin"] = "real_labeled_singlet"
    split_obs["experimental_doublet"] = 0
    split_obs["parent_cell_id"] = split_obs.index.astype(str)
    split_obs["parent1_id"] = ""
    split_obs["parent2_id"] = ""
    split_obs["parent_cluster1"] = ""
    split_obs["parent_cluster2"] = ""
    split_obs["doublet_subtype"] = ""
    split_obs["true_doublet_label"] = "clean"
    split_obs["doublet_saturation"] = np.nan
    split_obs["benchmark_cluster"] = split_obs["semireal_cluster"].astype(str)
    split_obs["duodose_cluster"] = split_obs["semireal_cluster"].astype(str)
    if "sample_id" not in split_obs:
        split_obs["sample_id"] = dataset
    if "library" not in split_obs:
        split_obs["library"] = split_obs["sample_id"].astype(str)

    # High-RNA hard negatives are defined within the split and within clusters.
    n_counts = np.asarray(split_counts.sum(axis=1)).ravel().astype(float)
    log_counts = np.log1p(n_counts)
    split_clusters = split_obs["semireal_cluster"].astype(str).to_numpy()
    high_rna = np.zeros(len(split_obs), dtype=bool)
    for cluster in sorted(pd.unique(split_clusters)):
        local = np.flatnonzero(split_clusters == cluster)
        if len(local) == 0:
            continue
        cutoff = float(np.quantile(log_counts[local], float(high_rna_quantile)))
        high_rna[local] = log_counts[local] >= cutoff
    split_obs["is_high_rna_singlet"] = high_rna
    split_obs["benchmark_cell_type"] = np.where(high_rna, "high_RNA_singlet", "singlet")
    split_obs["true_label"] = np.where(high_rna, "high_RNA_singlet", "clean")

    # Parent pools in absolute background coordinates.
    split_parent_set = set(map(int, parent_indices))
    pools: dict[str, np.ndarray] = {}
    for cluster in sorted(pd.unique(clusters_all[parent_indices])):
        local_abs = np.array([idx for idx in parent_indices if clusters_all[idx] == cluster], dtype=int)
        pools[str(cluster)] = local_abs
    valid_homo_clusters = np.array([c for c, idx in pools.items() if len(idx) >= max(2, int(min_cluster_size))])
    valid_hetero_clusters = np.array([c for c, idx in pools.items() if len(idx) >= 1])
    if len(valid_homo_clusters) == 0 and n_homotypic_doublets > 0:
        raise ValueError(f"{dataset}/{split_name}: no cluster has enough parents for homotypic doublets")
    if len(valid_hetero_clusters) < 2 and n_heterotypic_doublets > 0:
        raise ValueError(f"{dataset}/{split_name}: fewer than two clusters available for heterotypic doublets")

    low, high = sorted(saturation_range)
    doublet_rows: list[sparse.csr_matrix] = []
    doublet_obs_rows: list[dict[str, object]] = []

    def add_doublet(parent_i: int, parent_j: int, subtype: str) -> None:
        if parent_i not in split_parent_set or parent_j not in split_parent_set:
            raise AssertionError("parent leakage inside split construction")
        saturation = float(rng.uniform(low, high))
        row = counts.getrow(parent_i) + counts.getrow(parent_j)
        row = _downsample_row(row, saturation, rng)
        doublet_rows.append(row)
        p1 = str(background.obs_names[parent_i])
        p2 = str(background.obs_names[parent_j])
        c1 = str(clusters_all[parent_i])
        c2 = str(clusters_all[parent_j])
        doublet_obs_rows.append(
            {
                "experimental_doublet": 1,
                "semireal_split": split_name,
                "semireal_origin": "constructed_doublet",
                "benchmark_cell_type": "semireal_doublet",
                "is_high_rna_singlet": False,
                "semireal_cluster": c1,
                "benchmark_cluster": c1 if subtype == "homotypic" else f"{c1}|{c2}",
                "duodose_cluster": c1,
                "true_label": f"{subtype}_doublet",
                "true_doublet_label": f"{subtype}_doublet",
                "doublet_subtype": subtype,
                "parent_cell_id": "",
                "parent1_id": p1,
                "parent2_id": p2,
                "parent_cluster1": c1,
                "parent_cluster2": c2,
                "doublet_parent_cluster_1": c1,
                "doublet_parent_cluster_2": c2,
                "doublet_saturation": saturation,
                "sample_id": str(split_obs.iloc[0].get("sample_id", dataset)) if len(split_obs) else dataset,
                "library": str(split_obs.iloc[0].get("library", dataset)) if len(split_obs) else dataset,
            }
        )

    for _ in range(int(n_homotypic_doublets)):
        cluster = str(rng.choice(valid_homo_clusters))
        p1, p2 = rng.choice(pools[cluster], size=2, replace=False)
        add_doublet(int(p1), int(p2), "homotypic")

    for _ in range(int(n_heterotypic_doublets)):
        c1, c2 = rng.choice(valid_hetero_clusters, size=2, replace=False)
        p1 = int(rng.choice(pools[str(c1)]))
        p2 = int(rng.choice(pools[str(c2)]))
        add_doublet(p1, p2, "heterotypic")

    if doublet_rows:
        doublet_obs = pd.DataFrame(
            doublet_obs_rows,
            index=[f"{dataset}_{split_name}_semireal_doublet_{i:05d}" for i in range(len(doublet_obs_rows))],
        )
        X = sparse.vstack([split_counts, sparse.vstack(doublet_rows, format="csr")], format="csr")
        obs = pd.concat([split_obs, doublet_obs], axis=0, sort=False)
    else:
        X = split_counts
        obs = split_obs

    obs["experimental_doublet"] = obs["experimental_doublet"].astype(int)
    obs["is_high_rna_singlet"] = obs["is_high_rna_singlet"].fillna(False).astype(bool)
    out = AnnData(X=X, obs=obs, var=background.var.copy())
    out.layers["counts"] = out.X.copy()
    return out


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
    saturation_range: tuple[float, float],
    min_cluster_size: int,
    minimum_singlets: int = 200,
) -> SemiRealSplitBundle:
    rng = np.random.default_rng(int(seed))
    work = xili._ensure_counts_layer(adata)
    if "experimental_doublet" not in work.obs:
        raise ValueError("input AnnData needs experimental_doublet labels")
    singlet_mask = work.obs["experimental_doublet"].astype(int).eq(0).to_numpy()
    singlet_indices = np.flatnonzero(singlet_mask)
    if len(singlet_indices) < max(int(minimum_singlets), 3 * int(min_cluster_size)):
        raise ValueError(f"too few real singlets for semi-real construction: {len(singlet_indices)}")
    if len(singlet_indices) > int(n_singlets):
        singlet_indices = rng.choice(singlet_indices, size=int(n_singlets), replace=False)

    background = work[singlet_indices, :].copy()
    background = _cluster_real_singlets(background, n_clusters=n_clusters, random_state=seed)
    clusters = background.obs["semireal_cluster"].astype(str).to_numpy()
    all_parent_indices = np.arange(background.n_obs, dtype=int)

    trainval_idx, test_idx = _stratified_index_split(
        all_parent_indices,
        clusters,
        test_size=float(test_parent_fraction),
        random_state=int(seed),
    )
    fit_idx, val_idx = _stratified_index_split(
        trainval_idx,
        clusters,
        test_size=float(validation_parent_fraction),
        random_state=int(seed) + 1009,
    )

    n_val_homo = max(1, int(round(float(n_train_homotypic_doublets) * float(validation_parent_fraction))))
    n_val_hetero = max(1, int(round(float(n_train_heterotypic_doublets) * float(validation_parent_fraction))))
    n_fit_homo = max(1, int(n_train_homotypic_doublets) - n_val_homo)
    n_fit_hetero = max(1, int(n_train_heterotypic_doublets) - n_val_hetero)

    fit_adata = _make_split_adata(
        background,
        fit_idx,
        dataset=dataset,
        split_name="train",
        n_homotypic_doublets=n_fit_homo,
        n_heterotypic_doublets=n_fit_hetero,
        high_rna_quantile=high_rna_quantile,
        saturation_range=saturation_range,
        min_cluster_size=min_cluster_size,
        random_state=int(seed) + 11,
    )
    val_adata = _make_split_adata(
        background,
        val_idx,
        dataset=dataset,
        split_name="validation",
        n_homotypic_doublets=n_val_homo,
        n_heterotypic_doublets=n_val_hetero,
        high_rna_quantile=high_rna_quantile,
        saturation_range=saturation_range,
        min_cluster_size=min_cluster_size,
        random_state=int(seed) + 23,
    )
    test_adata = _make_split_adata(
        background,
        test_idx,
        dataset=dataset,
        split_name="test",
        n_homotypic_doublets=n_test_homotypic_doublets,
        n_heterotypic_doublets=n_test_heterotypic_doublets,
        high_rna_quantile=high_rna_quantile,
        saturation_range=saturation_range,
        min_cluster_size=min_cluster_size,
        random_state=int(seed) + 37,
    )

    def parent_names(indices: Iterable[int]) -> set[str]:
        return set(background.obs_names[np.asarray(list(indices), dtype=int)].astype(str).tolist())

    fit_parent_names = parent_names(fit_idx)
    val_parent_names = parent_names(val_idx)
    test_parent_names = parent_names(test_idx)
    parent_audit = {
        "dataset": dataset,
        "seed": int(seed),
        "n_fit_parent_cells": int(len(fit_parent_names)),
        "n_validation_parent_cells": int(len(val_parent_names)),
        "n_test_parent_cells": int(len(test_parent_names)),
        "train_validation_parent_overlap_fraction": float(len(fit_parent_names & val_parent_names) / max(1, len(fit_parent_names | val_parent_names))),
        "train_test_parent_overlap_fraction": float(len(fit_parent_names & test_parent_names) / max(1, len(fit_parent_names | test_parent_names))),
        "validation_test_parent_overlap_fraction": float(len(val_parent_names & test_parent_names) / max(1, len(val_parent_names | test_parent_names))),
        "parent_leakage_audit_status": "passed" if not (fit_parent_names & val_parent_names or fit_parent_names & test_parent_names or val_parent_names & test_parent_names) else "failed",
    }
    construction_report = {
        "dataset": dataset,
        "seed": int(seed),
        "n_real_labeled_singlets_available": int(singlet_mask.sum()),
        "n_real_singlets_used": int(background.n_obs),
        "n_genes": int(background.n_vars),
        "n_clusters_requested": int(n_clusters),
        "n_clusters_observed": int(pd.Series(clusters).nunique()),
        "n_fit_cells": int(fit_adata.n_obs),
        "n_validation_cells": int(val_adata.n_obs),
        "n_test_cells": int(test_adata.n_obs),
        "n_fit_homotypic_doublets": int(n_fit_homo),
        "n_fit_heterotypic_doublets": int(n_fit_hetero),
        "n_validation_homotypic_doublets": int(n_val_homo),
        "n_validation_heterotypic_doublets": int(n_val_hetero),
        "n_test_homotypic_doublets": int(n_test_homotypic_doublets),
        "n_test_heterotypic_doublets": int(n_test_heterotypic_doublets),
        "test_parent_fraction": float(test_parent_fraction),
        "validation_parent_fraction": float(validation_parent_fraction),
        "high_rna_quantile": float(high_rna_quantile),
        "saturation_low": float(sorted(saturation_range)[0]),
        "saturation_high": float(sorted(saturation_range)[1]),
        "min_cluster_size": int(min_cluster_size),
        "label_definition": "positive=constructed homotypic/heterotypic doublets; negative=real labeled singlets including high-RNA singlets",
    }
    return SemiRealSplitBundle(dataset, int(seed), fit_adata, val_adata, test_adata, construction_report, parent_audit)


def _score_split(
    adata: AnnData,
    *,
    dataset_id: str,
    random_state: int,
    outdir: Path,
    cache_dir: Path,
    expected_doublet_rate: float,
    n_simulated_doublets: int | None,
    n_hvgs: int,
    n_pcs: int,
    external_methods: list[str],
    refresh_cache: bool,
    quiet_external: bool,
    include_all_external: bool,
) -> tuple[pd.DataFrame, dict[str, pd.Series], list[dict[str, object]]]:
    method_scores: dict[str, pd.Series] = {}
    status_rows: list[dict[str, object]] = []

    duodose_scores, status, message, runtime, _ = xili.run_duodose_score(
        adata,
        dataset_id,
        random_state,
        refresh_cache,
        cache_dir,
        expected_doublet_rate=expected_doublet_rate,
        n_simulated_doublets=n_simulated_doublets,
        n_hvgs=n_hvgs,
        n_pcs=n_pcs,
    )
    for method, score in duodose_scores.items():
        method_scores[method] = pd.Series(score, index=adata.obs_names, dtype=float).reindex(adata.obs_names)
        status_rows.append({"dataset": dataset_id, "method": method, "status": status, "message": message, "runtime_sec": runtime if method == "DuoDose" else 0.0})

    ncount = pd.Series(np.asarray(_counts_matrix(adata).sum(axis=1)).ravel().astype(float), index=adata.obs_names, dtype=float)
    method_scores["nCount"] = ncount
    status_rows.append({"dataset": dataset_id, "method": "nCount", "status": "success", "message": "total UMI count per cell", "runtime_sec": 0.0})

    start = time.perf_counter()
    cluster_score, cluster_status, cluster_message = xili.cluster_specific_ncount_score(adata, random_state=random_state)
    method_scores["cluster-specific nCount"] = pd.Series(cluster_score, index=adata.obs_names, dtype=float).reindex(adata.obs_names)
    status_rows.append({"dataset": dataset_id, "method": "cluster-specific nCount", "status": cluster_status, "message": cluster_message, "runtime_sec": time.perf_counter() - start})

    # Scrublet is both a baseline and a SafeFeatures input. R methods are only needed on test.
    methods_to_run = ["Scrublet"]
    if include_all_external:
        for method in external_methods:
            if method not in methods_to_run:
                methods_to_run.append(method)
    for method in methods_to_run:
        score, status, message, runtime = xili.run_external_score(
            adata,
            dataset_id,
            method,
            random_state,
            refresh_cache,
            cache_dir,
            quiet_external,
            expected_doublet_rate,
        )
        method_scores[method] = pd.Series(score, index=adata.obs_names, dtype=float).reindex(adata.obs_names)
        status_rows.append({"dataset": dataset_id, "method": method, "status": status, "message": message, "runtime_sec": runtime})

    cell_frame = _build_cell_score_frame(adata, method_scores, dataset_id=dataset_id, random_state=random_state)
    return cell_frame, method_scores, status_rows


def _build_cell_score_frame(adata: AnnData, method_scores: dict[str, pd.Series], *, dataset_id: str, random_state: int) -> pd.DataFrame:
    obs = adata.obs.copy()
    obs.index = adata.obs_names
    frame = pd.DataFrame(index=adata.obs_names)
    frame["cell_id"] = adata.obs_names.astype(str)
    frame["dataset"] = dataset_id
    frame["seed"] = int(random_state)
    frame["design"] = "real_singlet_background"
    frame["propensity_setting"] = "semireal_parent_disjoint"
    frame["subtype_strategy"] = "balanced_homotypic_heterotypic"
    frame["mode"] = "semireal_singlet_derived"
    frame["semireal_split"] = obs.get("semireal_split", pd.Series("", index=obs.index)).astype(str)
    frame["true_label"] = obs.get("true_label", pd.Series("clean", index=obs.index)).astype(str)
    frame["true_doublet_label"] = obs.get("true_doublet_label", frame["true_label"]).astype(str)
    frame["doublet_subtype"] = obs.get("doublet_subtype", pd.Series("", index=obs.index)).astype(str)
    frame["is_high_rna_singlet"] = obs.get("is_high_rna_singlet", pd.Series(False, index=obs.index)).fillna(False).astype(bool)
    frame["parent_cell_id"] = obs.get("parent_cell_id", pd.Series("", index=obs.index)).astype(str)
    frame["parent1_id"] = obs.get("parent1_id", pd.Series("", index=obs.index)).astype(str)
    frame["parent2_id"] = obs.get("parent2_id", pd.Series("", index=obs.index)).astype(str)
    frame["parent_cluster1"] = obs.get("parent_cluster1", pd.Series("", index=obs.index)).astype(str)
    frame["parent_cluster2"] = obs.get("parent_cluster2", pd.Series("", index=obs.index)).astype(str)
    frame["benchmark_cluster"] = obs.get("benchmark_cluster", obs.get("semireal_cluster", pd.Series("cluster_0", index=obs.index))).astype(str)
    frame["duodose_cluster"] = obs.get("duodose_cluster", frame["benchmark_cluster"]).astype(str)
    frame["sample_id"] = obs.get("sample_id", pd.Series("lib1", index=obs.index)).astype(str)

    ncount = method_scores.get("nCount", pd.Series(np.asarray(_counts_matrix(adata).sum(axis=1)).ravel(), index=adata.obs_names, dtype=float)).reindex(adata.obs_names)
    frame["nCount"] = ncount.to_numpy(dtype=float)
    frame["log_nCount"] = np.log1p(frame["nCount"].astype(float))
    frame["nFeature"] = np.diff(_counts_matrix(adata).indptr).astype(float)
    frame["log_nFeature"] = np.log1p(frame["nFeature"].astype(float))
    frame["cluster_nCount_z"] = method_scores.get("cluster-specific nCount", _robust_z_by_group(pd.Series(frame["log_nCount"], index=frame.index), frame["benchmark_cluster"])).reindex(adata.obs_names).to_numpy(dtype=float)

    scrub = method_scores.get("Scrublet", pd.Series(np.nan, index=adata.obs_names, dtype=float)).reindex(adata.obs_names).astype(float)
    duo = method_scores.get("DuoDose", pd.Series(np.nan, index=adata.obs_names, dtype=float)).reindex(adata.obs_names).astype(float)
    identity = method_scores.get("DuoDose-identity", scrub).reindex(adata.obs_names).astype(float)
    dosage = method_scores.get("DuoDose-dosage", duo).reindex(adata.obs_names).astype(float)
    combined = method_scores.get("DuoDose-combined", duo).reindex(adata.obs_names).astype(float)
    gated = method_scores.get("DuoDose-gated-inlier", dosage).reindex(adata.obs_names).astype(float)
    gated_max = method_scores.get("DuoDose-gated-max", dosage).reindex(adata.obs_names).astype(float)
    max_score = method_scores.get("DuoDose-max", combined).reindex(adata.obs_names).astype(float)

    hom_rank = _rank01(gated)
    het_rank = _rank01(scrub.fillna(identity))
    duo_rank = _rank01(duo)
    hom_tail = _tail_rank01(gated)
    het_tail = _tail_rank01(scrub.fillna(identity))
    hybrid = _calibrated_union(scrub.fillna(identity), gated)

    columns = {
        "scrublet_score": scrub,
        "duodose_homotypic_score": gated,
        "duodose_heterotypic_score": identity,
        "duodose_score": duo,
        "duodose_sensitive_score": gated_max,
        "duodose_conservative_raw_score": dosage,
        "duodose_conservative_rank_score": _rank01(dosage),
        "duodose_conservative_tail_score": _tail_rank01(dosage),
        "hybrid_overall_score": hybrid,
        "hybrid_homotypic_score": hom_tail,
        "hybrid_heterotypic_score": het_tail,
        "homotypic_score": gated,
        "heterotypic_score": scrub.fillna(identity),
        "homotypic_rank_score": hom_rank,
        "heterotypic_rank_score": het_rank,
        "homotypic_tail_score": hom_tail,
        "heterotypic_tail_score": het_tail,
        "duodose_score_raw": duo,
        "duodose_score_rank_calibrated": duo_rank,
        "duodose_score_tail_calibrated": _tail_rank01(duo),
        "legacy_heterotypic_score": scrub.fillna(identity),
        "dosage_outlier_score": dosage,
        "identity_inlier_score": identity,
        "uniform_dosage_inflation_score": dosage,
        "biological_program_coherence_score": 1.0 - _rank01(identity),
        "homotypic_candidate_score": gated_max,
        "homotypic_final_score": gated,
        "module_residual_rank_mean": _rank01(dosage),
        "module_residual_rank_spread": (hom_rank - het_rank).abs(),
        "cluster_count_robust_z": frame["cluster_nCount_z"],
        "cluster_gene_robust_z": _robust_z_by_group(pd.Series(frame["log_nFeature"], index=frame.index), frame["benchmark_cluster"]),
        "cluster_stable_dosage_robust_z": frame["cluster_nCount_z"],
        "cluster_marker_dosage_robust_z": frame["cluster_nCount_z"],
        "dosage_residual": dosage,
        "cluster_abundance": frame["benchmark_cluster"].map(frame["benchmark_cluster"].value_counts(normalize=True)).astype(float),
        "cluster_level_expected_homotypic_burden": frame["benchmark_cluster"].map(frame["benchmark_cluster"].value_counts(normalize=True)).astype(float) ** 2,
        "benchmark_cluster_frequency": frame["benchmark_cluster"].map(frame["benchmark_cluster"].value_counts(normalize=True)).astype(float),
        "duodose_score_combined": combined,
        "duodose_score_max": max_score,
        "duodose_gated_inlier_score": gated,
    }
    for column, values in columns.items():
        frame[column] = pd.Series(values, index=frame.index).to_numpy(dtype=float)
    return frame.copy()


def _parent_set(frame: pd.DataFrame) -> set[str]:
    parents: set[str] = set()
    for column in ["parent_cell_id", "parent1_id", "parent2_id"]:
        if column in frame:
            values = frame[column].fillna("").astype(str)
            parents.update(v for v in values if v and v.lower() not in {"nan", "none"})
    return parents


def _call_train_predict(
    train_scores: pd.DataFrame,
    test_scores: pd.DataFrame,
    method: str,
    *,
    random_state: int,
    net_train_seed: int,
    train_index: Iterable[object],
    validation_index: Iterable[object],
    max_epochs: int,
    patience: int,
    device: str,
    batch_size: int | None,
    use_amp: bool,
) -> dict[str, object]:
    kwargs = {
        "train_cell_scores": train_scores,
        "test_cell_scores": test_scores,
        "method": method,
        "random_state": int(random_state),
        "net_train_seed": int(net_train_seed),
        "train_index": train_index,
        "validation_index": validation_index,
        "max_epochs": int(max_epochs),
        "patience": int(patience),
    }
    sig = inspect.signature(train_predict_diagnostic_model)
    optional = {
        "device": device,
        "batch_size": batch_size,
        "use_amp": use_amp,
        "dl_batch_size": batch_size,
        "dl_max_epochs": max_epochs,
        "dl_patience": patience,
    }
    for name, value in optional.items():
        if name in sig.parameters:
            kwargs[name] = value
    return train_predict_diagnostic_model(**kwargs)


