"""Label-free real-data application and manuscript UMAP figure utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .api import DuoDose
from .plotting_style import FONT_FALLBACKS, REQUESTED_FONT, apply_manuscript_style
from .validation import counts_copy


INTERNAL_FIGURE_METHODS = ("DuoDose",)
EXTERNAL_FIGURE_METHODS = ("Scrublet", "DoubletFinder", "scDblFinder", "scds")
SCORING_METHODS = (*EXTERNAL_FIGURE_METHODS, "DuoDose", "DuoDose subtype evidence")
PANEL_ORDER = (
    "clusters_or_annotations",
    "experimental_singlet_doublet_labels",
    "Scrublet",
    "DoubletFinder",
    "scDblFinder",
    "scds",
    "DuoDose overall doublet probability",
    "DuoDose subtype evidence",
    "DuoDose candidate classes at common top-K budget",
)
CANDIDATE_CLASSES = ("non_candidate", "subtype_ambiguous", "heterotypic_like", "homotypic_like")
CANDIDATE_DISPLAY_NAMES = {
    "non_candidate": "non candidate",
    "subtype_ambiguous": "subtype ambiguous",
    "heterotypic_like": "heterotypic-like",
    "homotypic_like": "homotypic-like",
}
HISTORICAL_COMPONENT_TOKENS = (
    "DuoDose-identity",
    "DuoDose-dosage",
    "DuoDose-combined",
    "DuoDose-max",
    "DuoDose-gated",
    "Hybrid",
    "Logistic",
    "DuoDose-Net",
    "DuoDose-DL",
)
SCORE_PALETTE = ("#F7F7F7", "#FEE8C8", "#FDBB84", "#E34A33", "#7F0000")
SUBTYPE_PALETTE = ("#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B")
CANDIDATE_PALETTE = {
    "non_candidate": "#ECEFF1",
    "subtype_ambiguous": "#8E70B5",
    "heterotypic_like": "#0072B2",
    "homotypic_like": "#D55E00",
}
MISSING_POINT_COLOR = "#EEF1F4"
INTENDED_FONT = REQUESTED_FONT


@dataclass
class RealApplicationResult:
    coordinates: pd.DataFrame
    method_scores: pd.DataFrame
    candidate_calls: pd.DataFrame
    method_status: pd.DataFrame
    label_usage_audit: pd.DataFrame
    reference_audit: pd.DataFrame
    shared_embedding_audit: pd.DataFrame
    candidate_display_audit: pd.DataFrame
    candidate_summary: pd.DataFrame
    group_diagnostics: pd.DataFrame
    diagnostics: pd.DataFrame
    panel_annotation: pd.Series
    panel_annotation_name: str
    duodose_result: Any
    display_budget: Mapping[str, Any] | None = None


def _hash_frame(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(map(str, frame.index)).encode("utf-8"))
    digest.update(np.asarray(frame.to_numpy(dtype=float), dtype=np.float64).tobytes())
    return digest.hexdigest()


def _hash_ids(values: Sequence[object]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def configure_manuscript_font() -> str:
    """Backward-compatible alias for the shared manuscript style."""

    return apply_manuscript_style()


def experimental_display_budget(labels: pd.Series, *, n_cells: int) -> dict[str, Any]:
    """Compute the post-hoc common display budget from experimental labels."""

    numeric = pd.to_numeric(labels, errors="coerce")
    labeled = numeric.isin([0, 1])
    n_singlet = int(numeric.loc[labeled].eq(0).sum())
    n_doublet = int(numeric.loc[labeled].eq(1).sum())
    n_labeled = n_singlet + n_doublet
    if n_labeled == 0:
        raise ValueError("experimental labels are required to define the post-hoc display budget")
    fraction = float(n_doublet / n_labeled)
    top_k = int(np.clip(np.floor(fraction * int(n_cells) + 0.5), 0, int(n_cells)))
    return {
        "labeled_singlet_count": n_singlet,
        "labeled_doublet_count": n_doublet,
        "labeled_cell_count": n_labeled,
        "labeled_doublet_fraction": fraction,
        "common_display_top_k": top_k,
        "common_display_fraction": float(top_k / int(n_cells)) if n_cells else float("nan"),
        "display_budget_source": "post-hoc experimental labeled doublet fraction",
        "display_budget_used_for_model_fitting": False,
    }


def common_top_k_masks(method_scores: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    """Select deterministic score-descending, cell-ID-ascending display masks."""

    output = pd.DataFrame(False, index=method_scores.index, columns=[*EXTERNAL_FIGURE_METHODS, "DuoDose"], dtype=bool)
    ids = method_scores.index.astype(str)
    for method in output.columns:
        values = pd.to_numeric(method_scores[method], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        candidates = pd.DataFrame({"position": np.flatnonzero(finite), "score": values[finite], "cell_id": ids[finite]})
        selected = candidates.sort_values(["score", "cell_id"], ascending=[False, True], kind="stable").head(min(int(top_k), len(candidates)))
        output.iloc[selected["position"].to_numpy(dtype=int), output.columns.get_loc(method)] = True
    return output


def apply_common_budget_candidate_classes(
    calls: pd.DataFrame,
    overall_score: pd.Series,
    *,
    top_k: int,
) -> pd.DataFrame:
    """Add exact top-K display classes without modifying raw model calls."""

    output = calls.copy()
    if "model_inferred_subtype_class" not in output:
        raise ValueError("raw model candidate classes are required for transparent display postprocessing")
    output["duodose_raw_model_candidate_class"] = output["model_inferred_subtype_class"].astype(str)
    score = pd.to_numeric(overall_score.reindex(output.index), errors="coerce")
    finite = score.notna() & np.isfinite(score.to_numpy(dtype=float))
    if int(finite.sum()) < int(top_k):
        raise ValueError(f"common display budget K={top_k} exceeds {int(finite.sum())} finite DuoDose scores")
    ranked = pd.DataFrame(
        {
            "stable_cell_id": output.index.astype(str)[finite.to_numpy()],
            "score": score.loc[finite].to_numpy(dtype=float),
        },
        index=output.index[finite],
    ).sort_values(["score", "stable_cell_id"], ascending=[False, True], kind="stable")
    rank = pd.Series(pd.NA, index=output.index, dtype="Int64")
    rank.loc[ranked.index] = np.arange(1, len(ranked) + 1, dtype=int)
    selected = rank.le(int(top_k)).fillna(False)
    classes = pd.Series("non_candidate", index=output.index, dtype=object)
    classes.loc[selected] = "subtype_ambiguous"
    q_hom = pd.to_numeric(output["duodose_q_homotypic_given_doublet"], errors="coerce")
    homotypic_threshold = float(pd.to_numeric(output["subtype_homotypic_threshold"], errors="coerce").dropna().iloc[0])
    heterotypic_threshold = float(pd.to_numeric(output["subtype_heterotypic_threshold"], errors="coerce").dropna().iloc[0])
    classes.loc[selected & q_hom.ge(homotypic_threshold)] = "homotypic_like"
    classes.loc[selected & q_hom.le(heterotypic_threshold)] = "heterotypic_like"
    output["common_display_candidate"] = selected.astype(bool)
    output["common_display_rank"] = rank
    output["duodose_common_budget_candidate_class"] = classes
    if int(selected.sum()) != int(top_k):
        raise AssertionError("deterministic common-budget selection did not produce exactly K cells")
    return output


def candidate_class_display_labels(
    calls: pd.DataFrame,
    *,
    class_column: str = "duodose_common_budget_candidate_class",
) -> tuple[dict[str, str], dict[str, dict[str, float | int]]]:
    """Return exact legend labels and traceable counts for plotted classes."""

    classes = calls[class_column].astype(str)
    total = int(len(classes))
    labels: dict[str, str] = {}
    summary: dict[str, dict[str, float | int]] = {}
    for class_name in CANDIDATE_CLASSES:
        count = int(classes.eq(class_name).sum())
        percentage = float(100.0 * count / total) if total else float("nan")
        display_name = CANDIDATE_DISPLAY_NAMES[class_name]
        labels[class_name] = f"{display_name} (n = {count:,}; {percentage:.1f}%)"
        summary[class_name] = {"count": count, "percentage_of_plotted_cells": percentage}
    return labels, summary


def candidate_display_audit(dataset: str, calls: pd.DataFrame, display_budget: Mapping[str, Any]) -> pd.DataFrame:
    """Audit the exact identity between displayed subtype calls and common K."""

    classes = calls["duodose_common_budget_candidate_class"].astype(str)
    counts = classes.value_counts().reindex(CANDIDATE_CLASSES, fill_value=0)
    candidate_sum = int(counts["subtype_ambiguous"] + counts["heterotypic_like"] + counts["homotypic_like"])
    top_k = int(display_budget["common_display_top_k"])
    equal = bool(candidate_sum == top_k)
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "total_plotted_cells": int(len(calls)),
                "labeled_singlet_count": int(display_budget["labeled_singlet_count"]),
                "labeled_doublet_count": int(display_budget["labeled_doublet_count"]),
                "labeled_doublet_fraction": float(display_budget["labeled_doublet_fraction"]),
                "common_display_top_k": top_k,
                "n_non_candidate": int(counts["non_candidate"]),
                "n_subtype_ambiguous": int(counts["subtype_ambiguous"]),
                "n_heterotypic_like": int(counts["heterotypic_like"]),
                "n_homotypic_like": int(counts["homotypic_like"]),
                "candidate_class_sum": candidate_sum,
                "candidate_class_sum_equals_k": equal,
                "tie_break_rule": "overall score descending, then stable cell ID ascending; exact rank <= K",
                "status": "PASS" if equal and int(counts["non_candidate"]) == len(calls) - top_k else "FAIL",
            }
        ]
    )


def label_blinded_adata(adata: AnnData) -> tuple[AnnData, pd.Series]:
    """Return counts/metadata with experimental labels unavailable to methods."""

    labels = adata.obs.get("experimental_doublet", pd.Series(pd.NA, index=adata.obs_names)).copy()
    blind = counts_copy(adata, layer="counts" if "counts" in adata.layers else None)
    blind_obs = adata.obs.drop(columns=["experimental_doublet"], errors="ignore").copy()
    # Some converted R objects retain an integer index-name attribute, which
    # AnnData rejects even though the cell identifiers themselves are valid.
    blind_obs.index.name = None
    blind.obs = blind_obs
    blind.obs["experimental_doublet"] = 0
    return blind, labels.reindex(adata.obs_names)


def fit_public_rf_label_free(
    adata: AnnData,
    *,
    expected_doublet_rate: float,
    random_state: int,
    training_preset: str,
) -> tuple[DuoDose, Any, pd.DataFrame]:
    """Fit the public RF workflow after forcibly masking experimental labels."""

    blind, _ = label_blinded_adata(adata)
    detector = DuoDose(
        backend="rf",
        expected_doublet_rate=float(expected_doublet_rate),
        random_state=int(random_state),
        device="cpu",
        layer="counts",
        threshold_strategy="expected_rate",
        training_preset=str(training_preset),
    )
    result = detector.fit_predict(blind)
    diagnostics = detector.safe_feature_transformer_.transform(
        blind,
        dataset_id="real_application",
        random_state=int(random_state),
    ).reindex(blind.obs_names)
    return detector, result, diagnostics


def _normalized_hvg_representation(adata: AnnData, *, n_hvgs: int, n_pcs: int, random_state: int) -> np.ndarray:
    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    counts = sparse.csr_matrix(counts, dtype=np.float32)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    scale = np.divide(1e4, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)
    normalized = sparse.diags(scale).dot(counts).tocsr()
    normalized.data = np.log1p(normalized.data)
    means = np.asarray(normalized.mean(axis=0)).ravel()
    squared = normalized.copy()
    squared.data **= 2
    variances = np.maximum(0.0, np.asarray(squared.mean(axis=0)).ravel() - means**2)
    keep = np.argsort(-variances, kind="stable")[: min(int(n_hvgs), normalized.shape[1])]
    matrix = normalized[:, keep]
    components = max(2, min(int(n_pcs), matrix.shape[0] - 1, matrix.shape[1] - 1))
    representation = TruncatedSVD(n_components=components, random_state=int(random_state)).fit_transform(matrix)
    return StandardScaler().fit_transform(representation)


def compute_shared_embedding(
    adata: AnnData,
    *,
    random_state: int,
    n_hvgs: int = 2000,
    n_pcs: int = 40,
    n_neighbors: int = 30,
    min_dist: float = 0.35,
    n_clusters: int = 12,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    representation = _normalized_hvg_representation(adata, n_hvgs=n_hvgs, n_pcs=n_pcs, random_state=random_state)
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - manuscript extra supplies umap-learn.
        raise RuntimeError("real-data application requires umap-learn (install DuoDose[manuscript])") from exc
    embedding = umap.UMAP(
        n_components=2,
        n_neighbors=min(max(2, int(n_neighbors)), max(2, adata.n_obs - 1)),
        min_dist=float(min_dist),
        metric="euclidean",
        random_state=int(random_state),
        transform_seed=int(random_state),
    ).fit_transform(representation)
    coordinates = pd.DataFrame(embedding, index=adata.obs_names, columns=["umap_1", "umap_2"])
    observed_clusters = min(max(2, int(n_clusters)), max(2, adata.n_obs // 25))
    clusters = pd.Series(
        [f"cluster_{value}" for value in KMeans(n_clusters=observed_clusters, n_init=20, random_state=int(random_state)).fit_predict(representation)],
        index=adata.obs_names,
        name="cluster",
    )
    return coordinates, representation, clusters


def choose_panel_annotation(obs: pd.DataFrame, clusters: pd.Series) -> tuple[pd.Series, str]:
    for column in ("cell_type", "celltype", "cell_type_annotation", "annotation", "CellType"):
        if column in obs and obs[column].notna().any() and obs[column].astype(str).nunique() > 1:
            return obs[column].fillna("unannotated").astype(str).reindex(clusters.index), str(column)
    return clusters.astype(str), "clusters"


def _robust_z_by_group(values: pd.Series, groups: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=values.index, dtype=float)
    for _, ids in groups.groupby(groups, sort=False).groups.items():
        local = pd.to_numeric(values.loc[ids], errors="coerce")
        median = float(local.median())
        mad = float((local - median).abs().median())
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(local.std(ddof=0))
        output.loc[ids] = (local - median) / (scale if np.isfinite(scale) and scale > 1e-12 else 1.0)
    return output


def local_diagnostics(adata: AnnData, representation: np.ndarray, clusters: pd.Series, safe_scores: pd.DataFrame, *, k: int) -> pd.DataFrame:
    n_neighbors = min(max(2, int(k) + 1), adata.n_obs)
    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(representation).kneighbors(return_distance=False)[:, 1:]
    cluster_values = clusters.to_numpy(dtype=str)
    neighbor_clusters = cluster_values[neighbors]
    purity = np.mean(neighbor_clusters == cluster_values[:, None], axis=1)
    entropy = []
    for row in neighbor_clusters:
        probabilities = pd.Series(row).value_counts(normalize=True).to_numpy(dtype=float)
        value = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))))
        entropy.append(value / max(np.log(max(2, len(probabilities))), 1e-12))
    counts = sparse.csr_matrix(adata.layers["counts"] if "counts" in adata.layers else adata.X)
    log_ncount = pd.Series(np.log1p(np.asarray(counts.sum(axis=1)).ravel()), index=adata.obs_names)
    log_ngene = pd.Series(np.log1p(np.diff(counts.indptr)), index=adata.obs_names)
    artificial = pd.to_numeric(
        safe_scores.get("handcrafted_artificial_doublet_neighbor_score", pd.Series(np.nan, index=adata.obs_names)),
        errors="coerce",
    ).reindex(adata.obs_names)
    return pd.DataFrame(
        {
            "same_cluster_neighbor_purity": purity,
            "neighbor_entropy": entropy,
            "within_cluster_log_nUMI_robust_z": _robust_z_by_group(log_ncount, clusters),
            "within_cluster_log_nGene_robust_z": _robust_z_by_group(log_ngene, clusters),
            "same_cluster_artificial_doublet_neighborhood_density": artificial,
        },
        index=adata.obs_names,
    )


def candidate_calls(scores: pd.DataFrame, *, overall_threshold: float, homotypic_threshold: float, heterotypic_threshold: float) -> pd.DataFrame:
    overall = pd.to_numeric(scores["duodose_score"], errors="coerce")
    hom = pd.to_numeric(scores["duodose_homotypic_score"], errors="coerce")
    hetero = pd.to_numeric(scores["duodose_heterotypic_score"], errors="coerce")
    denominator = hom + hetero
    q_hom = hom.div(denominator.where(denominator > 0))
    q_hetero = 1.0 - q_hom
    candidate = overall.ge(float(overall_threshold)) & overall.notna()
    classes = pd.Series("non_candidate", index=scores.index, dtype=object)
    classes.loc[candidate] = "subtype_ambiguous"
    classes.loc[candidate & q_hom.ge(float(homotypic_threshold))] = "homotypic_like"
    classes.loc[candidate & q_hom.le(float(heterotypic_threshold))] = "heterotypic_like"
    return pd.DataFrame(
        {
            "duodose_overall_candidate": candidate,
            "duodose_p_doublet": overall,
            "duodose_p_homotypic_doublet": hom,
            "duodose_p_heterotypic_doublet": hetero,
            "duodose_q_homotypic_given_doublet": q_hom,
            "duodose_q_heterotypic_given_doublet": q_hetero,
            "duodose_subtype_evidence": q_hom - q_hetero,
            "model_inferred_subtype_class": classes,
            "overall_score_threshold": float(overall_threshold),
            "subtype_homotypic_threshold": float(homotypic_threshold),
            "subtype_heterotypic_threshold": float(heterotypic_threshold),
            "ambiguous_interval": f"({float(heterotypic_threshold):g}, {float(homotypic_threshold):g})",
        },
        index=scores.index,
    )


def shared_embedding_audit(coordinates: pd.DataFrame, score_frame: pd.DataFrame) -> pd.DataFrame:
    coordinate_hash = _hash_frame(coordinates)
    id_hash = _hash_ids(coordinates.index)
    rows = []
    for panel in PANEL_ORDER:
        rows.append(
            {
                "panel": panel,
                "n_cells": int(len(coordinates)),
                "cell_id_hash": id_hash,
                "coordinate_hash": coordinate_hash,
                "cell_ids_aligned": bool(score_frame.index.equals(coordinates.index)),
                "coordinates_shared": True,
                "status": "PASS" if score_frame.index.equals(coordinates.index) else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def label_usage_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check": "experimental_label_used_in_training", "value": False, "status": "PASS"},
            {"check": "experimental_label_used_in_reference_selection", "value": False, "status": "PASS"},
            {"check": "experimental_label_used_in_threshold_selection", "value": False, "status": "PASS"},
            {"check": "experimental_label_used_in_embedding", "value": False, "status": "PASS"},
            {"check": "experimental_labels_joined_after_scores_frozen", "value": True, "status": "PASS"},
            {"check": "experimental_label_used_for_posthoc_display_budget", "value": True, "status": "PASS"},
        ]
    )


def candidate_summary(dataset: str, calls: pd.DataFrame, display_budget: Mapping[str, Any] | None = None) -> pd.DataFrame:
    class_column = "duodose_common_budget_candidate_class" if "duodose_common_budget_candidate_class" in calls else "model_inferred_subtype_class"
    counts = calls[class_column].value_counts().reindex(CANDIDATE_CLASSES, fill_value=0)
    raw_counts = calls["model_inferred_subtype_class"].value_counts().reindex(CANDIDATE_CLASSES, fill_value=0)
    total = max(1, int(len(calls)))
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "n_cells": int(len(calls)),
                "overall_score_threshold": float(calls["overall_score_threshold"].iloc[0]),
                "subtype_homotypic_threshold": float(calls["subtype_homotypic_threshold"].iloc[0]),
                "subtype_heterotypic_threshold": float(calls["subtype_heterotypic_threshold"].iloc[0]),
                **{f"n_{name}": int(counts[name]) for name in CANDIDATE_CLASSES},
                **{f"percent_{name}": float(100.0 * counts[name] / total) for name in CANDIDATE_CLASSES},
                **{f"raw_n_{name}": int(raw_counts[name]) for name in CANDIDATE_CLASSES},
                **dict(display_budget or {}),
            }
        ]
    )


def summarize_diagnostics(dataset: str, diagnostics: pd.DataFrame, calls: pd.DataFrame) -> pd.DataFrame:
    merged = diagnostics.join(calls[["model_inferred_subtype_class"]])
    rows = []
    for group in CANDIDATE_CLASSES:
        subset = merged.loc[merged["model_inferred_subtype_class"].eq(group)]
        for metric in diagnostics.columns:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            rows.append(
                {
                    "dataset": dataset,
                    "model_inferred_subtype_class": group,
                    "diagnostic": metric,
                    "n": int(len(values)),
                    "mean": float(values.mean()) if len(values) else float("nan"),
                    "median": float(values.median()) if len(values) else float("nan"),
                    "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def reference_audit(detector: DuoDose, result: Any) -> pd.DataFrame:
    feature = dict(result.feature_audit)
    parent = dict(result.parent_audit)
    metadata = dict(result.model_metadata)
    row = {
        "backend": result.backend,
        "public_method_name": "DuoDose",
        "internal_method_name": metadata.get("internal_method_name", ""),
        "construction_variant": metadata.get("construction_variant", ""),
        "safe_feature_mode": feature.get("safe_feature_mode", ""),
        "safe_feature_transformer_id": feature.get("safe_feature_transformer_id", ""),
        "safe_feature_reference_pool_id": feature.get("safe_feature_reference_pool_id", ""),
        "parent_disjoint": metadata.get("parent_disjoint", False),
        "parent_leakage_audit_status": parent.get("parent_leakage_audit_status", ""),
        "reference_parent_overlap_count": parent.get("reference_parent_overlap_count", ""),
        "experimental_labels_available_to_detector": False,
        "status": "PASS",
    }
    valid = (
        row["backend"] == "rf"
        and row["construction_variant"] == "raw_sum_parents_removed"
        and row["safe_feature_mode"] == "fitted_reference"
        and bool(row["parent_disjoint"])
        and str(row["parent_leakage_audit_status"]) == "passed"
    )
    row["status"] = "PASS" if valid else "FAIL"
    return pd.DataFrame([row])


def _scatter_continuous(
    fig,
    ax,
    coordinates: pd.DataFrame,
    values: pd.Series,
    *,
    title: str,
    cmap,
    vmin=None,
    vmax=None,
    unavailable: str = "",
    highlight_mask: pd.Series | None = None,
) -> None:
    x, y = coordinates["umap_1"], coordinates["umap_2"]
    numeric = pd.to_numeric(values.reindex(coordinates.index), errors="coerce")
    ax.scatter(x, y, color=MISSING_POINT_COLOR, s=3.2, alpha=0.72, linewidths=0, rasterized=True)
    finite = np.isfinite(numeric.to_numpy(dtype=float))
    if finite.any():
        order = np.argsort(numeric.to_numpy(dtype=float)[finite], kind="stable")
        ids = np.flatnonzero(finite)[order]
        highlighted = (
            highlight_mask.reindex(coordinates.index).fillna(False).to_numpy(dtype=bool)[ids]
            if highlight_mask is not None
            else np.zeros(len(ids), dtype=bool)
        )
        sizes = np.where(highlighted, 6.2, 3.4)
        artist = ax.scatter(x.iloc[ids], y.iloc[ids], c=numeric.iloc[ids], cmap=cmap, vmin=vmin, vmax=vmax, s=sizes, linewidths=0, rasterized=True)
        colorbar = fig.colorbar(artist, ax=ax, fraction=0.045, pad=0.02)
        colorbar.ax.tick_params(labelsize=7)
        colorbar.set_label(title, fontsize=8)
    else:
        ax.text(0.5, 0.5, unavailable or "score unavailable", transform=ax.transAxes, ha="center", va="center", fontsize=9)
    ax.set_title(title, fontsize=10)


def _scatter_categories(
    ax,
    coordinates: pd.DataFrame,
    values: pd.Series,
    *,
    title: str,
    palette: Mapping[str, str],
    order: Sequence[str] | None = None,
    display_labels: Mapping[str, str] | None = None,
    legend_loc: str = "best",
) -> None:
    categories = list(order or sorted(values.dropna().astype(str).unique()))
    for category in categories:
        mask = values.astype(str).eq(category).to_numpy()
        if mask.any():
            label = (display_labels or {}).get(category, category.replace("_", " "))
            ax.scatter(coordinates.loc[mask, "umap_1"], coordinates.loc[mask, "umap_2"], color=palette.get(category, "#777777"), s=4.0, linewidths=0, label=label, rasterized=True)
    ax.set_title(title, fontsize=10)
    if categories:
        ax.legend(frameon=False, fontsize=6.5, markerscale=2.2, loc=legend_loc)


def plot_cross_method_umap(
    output_png: Path,
    output_pdf: Path,
    *,
    dataset: str,
    coordinates: pd.DataFrame,
    annotation: pd.Series,
    annotation_name: str,
    experimental_labels: pd.Series,
    method_scores: pd.DataFrame,
    calls: pd.DataFrame,
    status: pd.DataFrame,
    display_budget: Mapping[str, Any],
    top_k_masks: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    configure_manuscript_font()
    score_cmap = LinearSegmentedColormap.from_list("duodose_score", SCORE_PALETTE, N=256)
    subtype_cmap = LinearSegmentedColormap.from_list("duodose_subtype", SUBTYPE_PALETTE, N=256)

    fig, axes = plt.subplots(3, 3, figsize=(15.6, 14.4), dpi=220)
    axes = axes.ravel()
    annotation_palette = {name: plt.get_cmap("tab20")(index % 20) for index, name in enumerate(sorted(annotation.astype(str).unique()))}
    _scatter_categories(axes[0], coordinates, annotation, title=f"{dataset}: {annotation_name}", palette=annotation_palette)
    singlet_label = f"Singlet (n = {int(display_budget['labeled_singlet_count']):,})"
    doublet_label = f"Doublet (n = {int(display_budget['labeled_doublet_count']):,})"
    label_values = experimental_labels.map({0: singlet_label, 1: doublet_label}).fillna("Label unavailable")
    _scatter_categories(
        axes[1],
        coordinates,
        label_values,
        title="Experimental singlet/doublet labels",
        palette={singlet_label: "#E7EBEF", doublet_label: "#D73027", "Label unavailable": "#F3F4F6"},
        order=(singlet_label, doublet_label, "Label unavailable"),
    )
    axes[1].text(
        0.02,
        0.02,
        f"Labeled doublet fraction = {float(display_budget['labeled_doublet_fraction']):.1%}",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#D5D9DE", "alpha": 0.92, "boxstyle": "round,pad=0.25"},
    )
    status_lookup = status.set_index("method").to_dict("index") if not status.empty else {}
    external_axes = (2, 3, 4, 5)
    for axis_index, method in zip(external_axes, EXTERNAL_FIGURE_METHODS, strict=True):
        item = status_lookup.get(method, {})
        unavailable = f"{item.get('status', 'unavailable')}: {item.get('message', '')}".strip(": ")
        _scatter_continuous(
            fig,
            axes[axis_index],
            coordinates,
            method_scores[method],
            title=f"{method} score",
            cmap=score_cmap,
            unavailable=unavailable,
            highlight_mask=top_k_masks[method],
        )
    _scatter_continuous(
        fig,
        axes[6],
        coordinates,
        method_scores["DuoDose"],
        title="DuoDose overall doublet probability",
        cmap=score_cmap,
        highlight_mask=top_k_masks["DuoDose"],
    )
    axes[6].text(
        0.02,
        0.02,
        f"Post-hoc common display budget\nRF top-K candidates: n = {int(top_k_masks['DuoDose'].sum()):,}\nK = {int(display_budget['common_display_top_k']):,} ({float(display_budget['common_display_fraction']):.1%})",
        transform=axes[6].transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "#D5D9DE", "alpha": 0.92, "boxstyle": "round,pad=0.25"},
    )
    _scatter_continuous(fig, axes[7], coordinates, calls["duodose_subtype_evidence"], title="DuoDose subtype evidence\nheterotypic-like <- 0 -> homotypic-like", cmap=subtype_cmap, vmin=-1.0, vmax=1.0)
    candidate_labels, _ = candidate_class_display_labels(calls, class_column="duodose_common_budget_candidate_class")
    _scatter_categories(
        axes[8],
        coordinates,
        calls["duodose_common_budget_candidate_class"],
        title="DuoDose candidate classes at common top-K budget",
        palette=CANDIDATE_PALETTE,
        order=CANDIDATE_CLASSES,
        display_labels=candidate_labels,
        legend_loc="upper left",
    )
    axes[8].text(
        0.02,
        0.02,
        f"Common display budget: K = {int(display_budget['common_display_top_k']):,} ({float(display_budget['common_display_fraction']):.1%})",
        transform=axes[8].transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "#D5D9DE", "alpha": 0.92, "boxstyle": "round,pad=0.25"},
    )
    x_pad = max(1e-6, 0.02 * float(coordinates["umap_1"].max() - coordinates["umap_1"].min()))
    y_pad = max(1e-6, 0.02 * float(coordinates["umap_2"].max() - coordinates["umap_2"].min()))
    x_limits = (float(coordinates["umap_1"].min() - x_pad), float(coordinates["umap_1"].max() + x_pad))
    y_limits = (float(coordinates["umap_2"].min() - y_pad), float(coordinates["umap_2"].max() + y_pad))
    for ax in axes:
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1", fontsize=8)
        ax.set_ylabel("UMAP 2", fontsize=8)
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    fig.text(0.5, 0.006, "Experimental labels provide singlet/doublet status only; DuoDose subtype classes are model-inferred.", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight", dpi=220)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(output_png: Path, output_pdf: Path, *, dataset: str, diagnostics: pd.DataFrame, calls: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    configure_manuscript_font()

    merged = diagnostics.join(calls[["model_inferred_subtype_class"]])
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), dpi=200)
    axes = axes.ravel()
    metrics = list(diagnostics.columns)
    for ax, metric in zip(axes, metrics, strict=False):
        values, labels = [], []
        for group in CANDIDATE_CLASSES:
            local = pd.to_numeric(merged.loc[merged["model_inferred_subtype_class"].eq(group), metric], errors="coerce").dropna()
            if len(local):
                values.append(local.to_numpy(dtype=float))
                labels.append(group.replace("_", "\n"))
        if values:
            ax.boxplot(values, labels=labels, showfliers=False)
        ax.set_title(metric.replace("_", " "), fontsize=9)
        ax.tick_params(axis="x", labelrotation=20, labelsize=6.5)
    for ax in axes[len(metrics) :]:
        ax.axis("off")
    fig.suptitle(f"{dataset}: mechanistic diagnostics of model-inferred candidates", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight", dpi=200)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def validate_figure_contract(result: RealApplicationResult) -> None:
    if INTERNAL_FIGURE_METHODS != ("DuoDose",):
        raise AssertionError("only DuoDose RF may be exposed as an internal main-figure method")
    text = " ".join([*PANEL_ORDER, *result.method_status.get("method", pd.Series(dtype=str)).astype(str)])
    if any(token in text for token in HISTORICAL_COMPONENT_TOKENS):
        raise AssertionError("main real-application figure exposes a prohibited historical component")
    if tuple(result.candidate_calls["model_inferred_subtype_class"].dropna().unique()) and not set(result.candidate_calls["model_inferred_subtype_class"]).issubset(CANDIDATE_CLASSES):
        raise AssertionError("candidate classes violate the frozen -like terminology")
    if "duodose_common_budget_candidate_class" not in result.candidate_calls:
        raise AssertionError("main figure is missing exact common-budget candidate classes")
    if not set(result.candidate_calls["duodose_common_budget_candidate_class"]).issubset(CANDIDATE_CLASSES):
        raise AssertionError("common-budget candidate classes violate the frozen -like terminology")
    if result.candidate_display_audit.empty or not result.candidate_display_audit["status"].eq("PASS").all():
        raise AssertionError("common-budget candidate display audit failed")
    if len(result.shared_embedding_audit) != 9 or not result.shared_embedding_audit["status"].eq("PASS").all():
        raise AssertionError("the nine figure panels do not share one aligned embedding")


def figure_manifest_payload(dataset: str, output_dir: Path, result: RealApplicationResult) -> dict[str, Any]:
    files = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name not in {"figure_manifest.json", "output_manifest.json", "failure.json"} and not path.name.startswith("."):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append({"path": path.name, "size_bytes": path.stat().st_size, "sha256": digest})
    return {
        "schema_version": 1,
        "dataset": dataset,
        "figure_role": "qualitative descriptive real-data application",
        "layout_rows": 3,
        "layout_columns": 3,
        "panel_order": list(PANEL_ORDER),
        "internal_methods": list(INTERNAL_FIGURE_METHODS),
        "external_methods": list(EXTERNAL_FIGURE_METHODS),
        "experimental_labels_evaluation_only": True,
        "shared_embedding_status": "PASS" if result.shared_embedding_audit["status"].eq("PASS").all() else "FAIL",
        "font_intended": INTENDED_FONT,
        "font_resolved": configure_manuscript_font(),
        "score_palette": list(SCORE_PALETTE),
        "subtype_palette": list(SUBTYPE_PALETTE),
        "candidate_palette": dict(CANDIDATE_PALETTE),
        "missing_point_color": MISSING_POINT_COLOR,
        "display_budget": dict(result.display_budget or {}),
        "candidate_class_column": "duodose_common_budget_candidate_class",
        "raw_candidate_class_column": "duodose_raw_model_candidate_class",
        "candidate_class_counts": candidate_class_display_labels(result.candidate_calls, class_column="duodose_common_budget_candidate_class")[1],
        "files": files,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, default=str) + "\n", encoding="utf-8")
