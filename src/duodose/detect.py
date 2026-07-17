"""Publication-ready DuoDose inference for real scRNA-seq datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional
import json
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .clustering import preliminary_clustering
from .data import ensure_counts_layer, get_counts_matrix, row_nnz, row_sums
from .features import extract_features, extract_simulated_features
from .preprocess import _column_variance, _normalize_log_counts, normalize_hvg_pca
from .qc import loose_qc
from .residuals import compute_dosage_residuals
from .simulate import simulate_doublets


PUBLIC_MODEL_NAME = "DuoDose"
PUBLIC_MODEL_KIND = "logistic"
DETECTION_CLUSTER_KEY = "duodose_cluster"
DOSAGE_RESCUE_GATE_QUANTILE = 0.95
IDENTITY_INLIER_GATE_QUANTILE = 0.75
SAFE_FEATURE_EXCLUDE_TOKENS = (
    "benchmark",
    "true",
    "label",
    "doublet_type",
    "simulated",
    "expected",
    "y_true",
    "is_",
)
SAFE_FEATURE_EXCLUDE_EXACT = {
    "artificial_neighbor_fraction",
    "nearest_heterotypic_doublet_distance",
    "nearest_homotypic_doublet_distance",
    "heterotypic_similarity",
    "homotypic_similarity",
}
SCORE_COLUMNS = [
    "cell_id",
    "duodose_score",
    "duodose_identity_score",
    "duodose_dosage_score",
    "duodose_score_combined",
    "duodose_score_max",
    "duodose_gated_025_score",
    "duodose_gated_050_score",
    "duodose_gated_max_score",
    "duodose_gated_inlier_score",
    "doublet_probability",
    "predicted_doublet",
    "predicted_doublet_type",
    "homotypic_score",
    "heterotypic_score",
    "highRNA_singlet_risk",
    "nCount",
    "cluster",
    "threshold_used",
    "expected_doublet_rate",
]


def _safe_feature_columns(columns: pd.Index) -> list[str]:
    selected: list[str] = []
    for column in columns:
        name = str(column)
        lower = name.lower()
        if lower in SAFE_FEATURE_EXCLUDE_EXACT:
            continue
        if lower == "benchmark_cluster_frequency" or lower.startswith("benchmark_cluster_"):
            continue
        if any(token in lower for token in SAFE_FEATURE_EXCLUDE_TOKENS):
            continue
        selected.append(name)
    return selected


def _safe_numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    selected = _safe_feature_columns(numeric.columns)
    if not selected:
        return pd.DataFrame({"constant_feature": np.ones(len(frame), dtype=float)}, index=frame.index)
    out = numeric.loc[:, selected].copy()
    for column in out.columns:
        if out[column].isna().all():
            out[column] = 0.0
    return out.fillna(out.median(numeric_only=True)).fillna(0.0)


def _default_n_simulated_doublets(n_obs: int) -> int:
    return int(np.clip(max(1000, 2 * int(n_obs)), 1, 50000))


def _maybe_filter_cells(
    adata: AnnData,
    *,
    min_counts: Optional[int],
    min_genes: Optional[int],
    max_mito_fraction: Optional[float],
    counts_layer: str,
) -> AnnData:
    if min_counts is None and min_genes is None and max_mito_fraction is None:
        return adata.copy()
    return loose_qc(
        adata,
        min_counts=1 if min_counts is None else int(min_counts),
        min_genes=1 if min_genes is None else int(min_genes),
        max_mito_fraction=1.0 if max_mito_fraction is None else float(max_mito_fraction),
        counts_layer=counts_layer,
    )


def _compute_basic_count_metrics(adata: AnnData, counts_layer: str) -> None:
    counts = get_counts_matrix(adata, counts_layer=counts_layer)
    adata.obs["n_counts"] = row_sums(counts).astype(float)
    adata.obs["n_genes"] = row_nnz(counts).astype(float)


def _prepare_clusters(
    adata: AnnData,
    *,
    cluster_key: Optional[str],
    clustering_resolution: float,
    random_state: int,
) -> str:
    if cluster_key is not None:
        if cluster_key not in adata.obs:
            raise KeyError(f"cluster_key={cluster_key!r} not found in adata.obs")
        if cluster_key != DETECTION_CLUSTER_KEY:
            adata.obs[DETECTION_CLUSTER_KEY] = adata.obs[cluster_key].astype(str).astype("category")
            return DETECTION_CLUSTER_KEY
        adata.obs[DETECTION_CLUSTER_KEY] = adata.obs[DETECTION_CLUSTER_KEY].astype(str).astype("category")
        return DETECTION_CLUSTER_KEY
    preliminary_clustering(
        adata,
        resolution=clustering_resolution,
        use_rep="X_pca",
        cluster_key=DETECTION_CLUSTER_KEY,
        random_state=random_state,
    )
    return DETECTION_CLUSTER_KEY


def _fit_public_logistic_model(
    observed_features: pd.DataFrame,
    simulated_features: pd.DataFrame,
    simulated_doublet_types: np.ndarray,
    *,
    random_state: int,
) -> tuple[Pipeline, list[str], pd.Series]:
    observed = _safe_numeric_features(observed_features)
    simulated = _safe_numeric_features(simulated_features).reindex(columns=observed.columns, fill_value=0.0)
    X = pd.concat([observed, simulated], axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_observed = pd.Series("clean", index=observed.index, dtype=object)
    y_simulated = pd.Series(
        np.where(np.asarray(simulated_doublet_types, dtype=object) == "heterotypic", "heterotypic_doublet", "homotypic_doublet"),
        index=simulated.index,
        dtype=object,
    )
    y = pd.concat([y_observed, y_simulated], axis=0)
    if y.nunique() < 2:
        raise ValueError("DuoDose inference needs observed cells and at least one simulated doublet class.")
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model, list(observed.columns), y


def _predict_class_probabilities(model: Pipeline, feature_names: list[str], observed_features: pd.DataFrame) -> pd.DataFrame:
    observed = _safe_numeric_features(observed_features).reindex(columns=feature_names, fill_value=0.0)
    raw = model.predict_proba(observed)
    classes = np.asarray(model.named_steps["classifier"].classes_, dtype=object)
    probs = pd.DataFrame(0.0, index=observed.index, columns=["clean", "homotypic_doublet", "heterotypic_doublet"])
    for idx, label in enumerate(classes):
        probs[str(label)] = raw[:, idx]
    row_sum = probs.sum(axis=1).replace(0.0, 1.0)
    return probs.div(row_sum, axis=0)


def _safe_binary_metric(metric_fn, y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(s)
    if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
        return float("nan")
    try:
        return float(metric_fn(y[mask], s[mask]))
    except Exception:
        return float("nan")


def _score_quantiles(values: np.ndarray) -> dict[str, float]:
    series = pd.Series(np.asarray(values, dtype=float)).replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return {key: float("nan") for key in ["min", "q01", "q05", "q10", "q25", "median", "q75", "q90", "q95", "q99", "max"]}
    quantiles = series.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "min": float(series.min()),
        "q01": float(quantiles.loc[0.01]),
        "q05": float(quantiles.loc[0.05]),
        "q10": float(quantiles.loc[0.10]),
        "q25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.50]),
        "q75": float(quantiles.loc[0.75]),
        "q90": float(quantiles.loc[0.90]),
        "q95": float(quantiles.loc[0.95]),
        "q99": float(quantiles.loc[0.99]),
        "max": float(series.max()),
    }


def _tail_rank_calibrate(scores: pd.Series, groups: pd.Series | None = None) -> pd.Series:
    score = pd.to_numeric(scores, errors="coerce").replace([np.inf, -np.inf], np.nan)
    fill = float(score.dropna().median()) if score.notna().any() else 0.0
    score = score.fillna(fill)
    if groups is None:
        groups = pd.Series("__all__", index=score.index, dtype=object)
    groups = pd.Series(groups, index=score.index).astype(str)
    out = pd.Series(0.0, index=score.index, dtype=float)
    for _, idx in groups.groupby(groups).groups.items():
        values = score.loc[idx]
        if values.nunique(dropna=True) <= 1:
            out.loc[idx] = 0.0
            continue
        ranks = values.rank(method="average", pct=True)
        out.loc[idx] = np.maximum(0.0, (ranks.to_numpy(dtype=float) - 0.5) / 0.5)
    return out.clip(0.0, 1.0)


def _percentile_rank_calibrate(scores: pd.Series) -> pd.Series:
    score = pd.to_numeric(scores, errors="coerce").replace([np.inf, -np.inf], np.nan)
    fill = float(score.dropna().median()) if score.notna().any() else 0.0
    score = score.fillna(fill)
    if score.nunique(dropna=True) <= 1:
        return pd.Series(0.0, index=score.index, dtype=float)
    return score.rank(method="average", pct=True).clip(0.0, 1.0).astype(float)


def _gated_duodose_scores(
    identity_score: pd.Series,
    dosage_score: pd.Series,
    *,
    dose_gate_quantile: float = DOSAGE_RESCUE_GATE_QUANTILE,
    inlier_gate_quantile: float = IDENTITY_INLIER_GATE_QUANTILE,
) -> dict[str, pd.Series]:
    s_id_rank = _percentile_rank_calibrate(identity_score)
    s_dose_rank = _percentile_rank_calibrate(dosage_score)
    gate = float(np.clip(dose_gate_quantile, 0.0, 0.999999))
    dose_rescue_values = np.maximum(0.0, (s_dose_rank.to_numpy(dtype=float) - gate) / max(1e-12, 1.0 - gate))
    dose_rescue = pd.Series(dose_rescue_values, index=s_dose_rank.index, dtype=float).clip(0.0, 1.0)
    id_values = s_id_rank.to_numpy(dtype=float)
    rescue_values = dose_rescue.to_numpy(dtype=float)
    gated_025 = id_values + 0.25 * (1.0 - id_values) * rescue_values
    gated_050 = id_values + 0.50 * (1.0 - id_values) * rescue_values
    gated_max = np.maximum(id_values, 0.50 * rescue_values)
    inlier_rescue = rescue_values.copy()
    inlier_rescue[id_values > float(inlier_gate_quantile)] = 0.0
    gated_inlier = id_values + 0.50 * (1.0 - id_values) * inlier_rescue
    return {
        "identity_rank": s_id_rank,
        "dosage_rank": s_dose_rank,
        "dose_rescue": dose_rescue,
        "gated_025": pd.Series(np.clip(gated_025, 0.0, 1.0), index=s_id_rank.index, dtype=float),
        "gated_050": pd.Series(np.clip(gated_050, 0.0, 1.0), index=s_id_rank.index, dtype=float),
        "gated_max": pd.Series(np.clip(gated_max, 0.0, 1.0), index=s_id_rank.index, dtype=float),
        "gated_inlier": pd.Series(np.clip(gated_inlier, 0.0, 1.0), index=s_id_rank.index, dtype=float),
    }


def _classifier_coefficients(model: Pipeline, feature_names: list[str], branch: str, max_features: int = 200) -> list[dict[str, object]]:
    classifier = model.named_steps.get("classifier")
    if classifier is None or not hasattr(classifier, "coef_"):
        return []
    coefs = np.asarray(classifier.coef_, dtype=float)
    classes = np.asarray(getattr(classifier, "classes_", []), dtype=object)
    if coefs.ndim == 1:
        coefs = coefs.reshape(1, -1)
    rows: list[dict[str, object]] = []
    for row_idx, coef_row in enumerate(coefs):
        label = str(classes[row_idx]) if row_idx < len(classes) else f"class_{row_idx}"
        order = np.argsort(-np.abs(coef_row))[: min(max_features, len(coef_row))]
        for feature_idx in order:
            rows.append(
                {
                    "branch": branch,
                    "class": label,
                    "feature": str(feature_names[feature_idx]),
                    "coefficient": float(coef_row[feature_idx]),
                    "abs_coefficient": float(abs(coef_row[feature_idx])),
                }
            )
    return rows


def _joint_pca_representation(
    observed_X,
    simulated_X,
    *,
    n_hvgs: int,
    n_pcs: int,
    random_state: int,
) -> np.ndarray:
    if sparse.issparse(observed_X) or sparse.issparse(simulated_X):
        combined = sparse.vstack([sparse.csr_matrix(observed_X), sparse.csr_matrix(simulated_X)], format="csr")
    else:
        combined = np.vstack([np.asarray(observed_X), np.asarray(simulated_X)])
    log_norm = _normalize_log_counts(combined)
    n_hvgs_eff = int(min(max(1, n_hvgs), log_norm.shape[1]))
    variances = _column_variance(log_norm)
    hvg_order = np.argsort(variances)[::-1][:n_hvgs_eff]
    hvg_mask = np.zeros(log_norm.shape[1], dtype=bool)
    hvg_mask[hvg_order] = True
    X_hvg = log_norm[:, hvg_mask]
    max_components = int(max(1, min(n_pcs, X_hvg.shape[0] - 1 if X_hvg.shape[0] > 1 else 1, X_hvg.shape[1] - 1 if X_hvg.shape[1] > 1 else 1)))
    if X_hvg.shape[0] < 2 or X_hvg.shape[1] < 2:
        return np.zeros((X_hvg.shape[0], 1), dtype=np.float32)
    if sparse.issparse(X_hvg):
        scaled = StandardScaler(with_mean=False).fit_transform(X_hvg)
        rep = TruncatedSVD(n_components=max_components, random_state=random_state).fit_transform(scaled)
    else:
        scaled = StandardScaler(with_mean=True).fit_transform(np.asarray(X_hvg, dtype=np.float32))
        rep = PCA(n_components=max_components, random_state=random_state).fit_transform(scaled)
    rep = np.asarray(rep, dtype=np.float32)
    if rep.shape[1] > 1:
        rep = StandardScaler().fit_transform(rep).astype(np.float32)
    return np.nan_to_num(rep, copy=False)


def _artificial_neighbor_identity_score(
    adata: AnnData,
    simulated_X,
    *,
    counts_layer: str,
    n_hvgs: int,
    n_pcs: int,
    random_state: int,
    k_neighbors: int = 50,
) -> tuple[pd.Series, pd.DataFrame, dict[str, object]]:
    observed_X = get_counts_matrix(adata, counts_layer=counts_layer)
    n_observed = int(adata.n_obs)
    n_artificial = int(simulated_X.shape[0])
    rep = _joint_pca_representation(
        observed_X,
        simulated_X,
        n_hvgs=n_hvgs,
        n_pcs=n_pcs,
        random_state=random_state,
    )
    total = rep.shape[0]
    if total <= 1 or n_artificial == 0:
        score = pd.Series(0.0, index=adata.obs_names, dtype=float)
        diagnostics = pd.DataFrame(index=adata.obs_names)
        diagnostics["artificial_neighbor_fraction"] = 0.0
        diagnostics["distance_weighted_artificial_neighbor_fraction"] = 0.0
        diagnostics["local_artificial_enrichment"] = 0.0
        diagnostics["mean_distance_to_artificial_neighbors"] = np.nan
        return score, diagnostics, {"status": "skipped_no_artificial_neighbors"}

    k = int(max(1, min(k_neighbors, total - 1)))
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(rep)
    distances, indices = nn.kneighbors(rep, return_distance=True)
    artificial_flags = np.arange(total) >= n_observed
    expected_fraction = n_artificial / max(1, total - 1)
    rows: list[dict[str, float]] = []
    raw_scores = np.zeros(total, dtype=float)
    for row in range(total):
        neighbor_idx = indices[row]
        neighbor_dist = distances[row]
        keep = neighbor_idx != row
        neighbor_idx = neighbor_idx[keep][:k]
        neighbor_dist = neighbor_dist[keep][:k]
        if neighbor_idx.size == 0:
            artificial_fraction = 0.0
            weighted_fraction = 0.0
            mean_artificial_distance = float("nan")
        else:
            is_artificial_neighbor = artificial_flags[neighbor_idx]
            artificial_fraction = float(is_artificial_neighbor.mean())
            weights = 1.0 / np.maximum(neighbor_dist, 1e-6)
            weighted_fraction = float(weights[is_artificial_neighbor].sum() / max(weights.sum(), 1e-12))
            mean_artificial_distance = float(neighbor_dist[is_artificial_neighbor].mean()) if is_artificial_neighbor.any() else float("nan")
        enrichment = (artificial_fraction - expected_fraction) / max(1e-6, 1.0 - expected_fraction)
        weighted_enrichment = (weighted_fraction - expected_fraction) / max(1e-6, 1.0 - expected_fraction)
        distance_component = 0.0 if not np.isfinite(mean_artificial_distance) else 1.0 / (1.0 + mean_artificial_distance)
        raw = 0.45 * max(0.0, enrichment) + 0.35 * max(0.0, weighted_enrichment) + 0.20 * distance_component
        raw_scores[row] = raw
        if row < n_observed:
            rows.append(
                {
                    "artificial_neighbor_fraction": artificial_fraction,
                    "distance_weighted_artificial_neighbor_fraction": weighted_fraction,
                    "local_artificial_enrichment": float(enrichment),
                    "mean_distance_to_artificial_neighbors": mean_artificial_distance,
                    "raw_identity_enrichment_score": float(raw),
                }
            )
    observed_raw = pd.Series(raw_scores[:n_observed], index=adata.obs_names, dtype=float)
    observed_score = observed_raw.rank(method="average", pct=True).fillna(0.0).clip(0.0, 1.0)
    diagnostics_frame = pd.DataFrame(rows, index=adata.obs_names)
    diagnostics_frame["duodose_identity_raw_score"] = observed_raw.to_numpy(dtype=float)
    diagnostics_frame["duodose_identity_score"] = observed_score.to_numpy(dtype=float)
    train_y = np.concatenate([np.zeros(n_observed, dtype=int), np.ones(n_artificial, dtype=int)])
    diagnostics = {
        "status": "success",
        "k_neighbors": int(k),
        "expected_artificial_fraction": float(expected_fraction),
        "training_AUROC": _safe_binary_metric(roc_auc_score, train_y, raw_scores),
        "training_AUPRC": _safe_binary_metric(average_precision_score, train_y, raw_scores),
        "observed_score_quantiles": _score_quantiles(observed_score.to_numpy(dtype=float)),
        "artificial_raw_score_quantiles": _score_quantiles(raw_scores[n_observed:]),
        "observed_fraction_below_0p01": float(np.mean(observed_score.to_numpy(dtype=float) < 0.01)),
        "artificial_fraction_below_0p01": float(np.mean(raw_scores[n_observed:] < 0.01)),
    }
    return observed_score, diagnostics_frame, diagnostics


def _robust_auto_threshold(scores: pd.Series) -> float:
    values = pd.to_numeric(scores, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return 1.0
    median = float(values.median())
    mad = float(np.median(np.abs(values.to_numpy(dtype=float) - median)))
    threshold = median + 3.0 * 1.4826 * mad
    if not np.isfinite(threshold) or threshold >= float(values.max()):
        threshold = float(values.quantile(0.95))
    return float(np.clip(threshold, 0.0, 1.0))


def _expected_rate_threshold(scores: pd.Series, expected_doublet_rate: Optional[float]) -> float:
    if expected_doublet_rate is None:
        return _robust_auto_threshold(scores)
    rate = float(np.clip(expected_doublet_rate, 0.0, 1.0))
    if rate <= 0.0:
        return 1.0 + 1e-9
    if rate >= 1.0:
        return -1e-9
    return float(pd.to_numeric(scores, errors="coerce").quantile(1.0 - rate))


def _threshold_scores(
    scores: pd.Series,
    *,
    expected_doublet_rate: Optional[float],
    threshold_method: Literal["expected_rate", "auto"],
) -> float:
    if threshold_method == "auto":
        return _robust_auto_threshold(scores)
    if threshold_method != "expected_rate":
        raise ValueError("threshold_method must be 'expected_rate' or 'auto'")
    return _expected_rate_threshold(scores, expected_doublet_rate)


def _score_table_from_obs(adata: AnnData) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            "duodose_score": adata.obs["duodose_score"].to_numpy(dtype=float),
            "duodose_identity_score": adata.obs.get("duodose_identity_score", pd.Series(np.nan, index=adata.obs_names)).to_numpy(dtype=float),
            "duodose_dosage_score": adata.obs.get("duodose_dosage_score", pd.Series(np.nan, index=adata.obs_names)).to_numpy(dtype=float),
            "duodose_score_combined": adata.obs.get("duodose_score_combined", pd.Series(np.nan, index=adata.obs_names)).to_numpy(dtype=float),
            "duodose_score_max": adata.obs.get("duodose_score_max", pd.Series(np.nan, index=adata.obs_names)).to_numpy(dtype=float),
            "duodose_gated_025_score": adata.obs.get("duodose_gated_025_score", pd.Series(np.nan, index=adata.obs_names)).to_numpy(dtype=float),
            "duodose_gated_050_score": adata.obs.get("duodose_gated_050_score", pd.Series(np.nan, index=adata.obs_names)).to_numpy(dtype=float),
            "duodose_gated_max_score": adata.obs.get("duodose_gated_max_score", pd.Series(np.nan, index=adata.obs_names)).to_numpy(dtype=float),
            "duodose_gated_inlier_score": adata.obs.get("duodose_gated_inlier_score", pd.Series(np.nan, index=adata.obs_names)).to_numpy(dtype=float),
            "doublet_probability": adata.obs["duodose_doublet_probability"].to_numpy(dtype=float),
            "predicted_doublet": adata.obs["duodose_predicted_doublet"].to_numpy(dtype=bool),
            "predicted_doublet_type": adata.obs["duodose_predicted_doublet_type"].astype(str).to_numpy(),
            "homotypic_score": adata.obs["duodose_homotypic_score"].to_numpy(dtype=float),
            "heterotypic_score": adata.obs["duodose_heterotypic_score"].to_numpy(dtype=float),
            "highRNA_singlet_risk": adata.obs.get("duodose_highRNA_singlet_risk", pd.Series(np.nan, index=adata.obs_names)).to_numpy(dtype=float),
            "nCount": adata.obs["n_counts"].to_numpy(dtype=float),
            "cluster": adata.obs[DETECTION_CLUSTER_KEY].astype(str).to_numpy(),
            "threshold_used": adata.obs["duodose_threshold_used"].to_numpy(dtype=float),
            "expected_doublet_rate": adata.obs["duodose_expected_doublet_rate"].to_numpy(dtype=float),
        },
        index=adata.obs_names,
    )
    return frame.loc[:, SCORE_COLUMNS]


def detect(
    adata: AnnData,
    cluster_key: Optional[str] = None,
    expected_doublet_rate: Optional[float] = 0.08,
    model: str = PUBLIC_MODEL_KIND,
    random_state: int = 0,
    *,
    library_key: Optional[str] = None,
    counts_layer: str = "counts",
    n_simulated_doublets: Optional[int] = None,
    homotypic_fraction: float = 0.5,
    saturation_range: tuple[float, float] = (0.6, 1.0),
    n_hvgs: int = 2000,
    n_pcs: int = 50,
    clustering_resolution: float = 0.6,
    threshold_method: Literal["expected_rate", "auto"] = "expected_rate",
    min_counts: Optional[int] = None,
    min_genes: Optional[int] = None,
    max_mito_fraction: Optional[float] = None,
    return_debug: bool = False,
) -> AnnData:
    """Run publication DuoDose doublet detection on an AnnData object.

    This public inference path trains the primary DuoDose model, a logistic
    SafeFeatures classifier, using observed cells as presumed singlets and
    synthetic homotypic/heterotypic doublets as positives.
    """

    if model.lower() not in {"logistic", "duodose"}:
        raise ValueError("The publication-ready inference API currently supports model='logistic' only.")
    if library_key is not None and library_key not in adata.obs:
        warnings.warn(f"library_key={library_key!r} not found; treating all cells as one library.", RuntimeWarning, stacklevel=2)
        library_key = None

    work = adata.copy()
    ensure_counts_layer(work, counts_layer=counts_layer)
    work = _maybe_filter_cells(
        work,
        min_counts=min_counts,
        min_genes=min_genes,
        max_mito_fraction=max_mito_fraction,
        counts_layer=counts_layer,
    )
    ensure_counts_layer(work, counts_layer=counts_layer)
    _compute_basic_count_metrics(work, counts_layer)

    normalize_hvg_pca(work, counts_layer=counts_layer, n_hvgs=n_hvgs, n_pcs=n_pcs, random_state=random_state)
    internal_cluster_key = _prepare_clusters(
        work,
        cluster_key=cluster_key,
        clustering_resolution=clustering_resolution,
        random_state=random_state,
    )
    compute_dosage_residuals(work, cluster_key=internal_cluster_key, library_key=library_key, counts_layer=counts_layer)

    n_doublets = _default_n_simulated_doublets(work.n_obs) if n_simulated_doublets is None else int(n_simulated_doublets)
    simulated = simulate_doublets(
        work,
        cluster_key=internal_cluster_key,
        library_key=library_key,
        counts_layer=counts_layer,
        n_doublets=n_doublets,
        homotypic_fraction=homotypic_fraction,
        saturation_range=saturation_range,
        random_state=random_state,
    )
    if len(simulated) == 0:
        raise ValueError("No synthetic doublets were generated; increase n_simulated_doublets or check input cells.")

    observed_features = extract_features(work, simulated, cluster_key=internal_cluster_key, library_key=library_key, use_rep="X_pca")
    simulated_features = extract_simulated_features(
        work,
        simulated,
        observed_features,
        cluster_key=internal_cluster_key,
        library_key=library_key,
        use_rep="X_pca",
    )
    fitted, feature_names, y_train = _fit_public_logistic_model(
        observed_features,
        simulated_features,
        simulated.doublet_type,
        random_state=random_state,
    )
    probabilities = _predict_class_probabilities(fitted, feature_names, observed_features)
    simulated_probabilities = _predict_class_probabilities(fitted, feature_names, simulated_features)
    homotypic_score = probabilities["homotypic_doublet"].clip(0.0, 1.0)
    heterotypic_score = probabilities["heterotypic_doublet"].clip(0.0, 1.0)
    dosage_raw_probability = (homotypic_score + heterotypic_score).clip(0.0, 1.0)
    library_series = pd.Series("__all__", index=work.obs_names, dtype=object) if library_key is None else work.obs[library_key].astype(str)
    dosage_score = _tail_rank_calibrate(dosage_raw_probability, library_series)
    identity_score, identity_diagnostics, identity_training = _artificial_neighbor_identity_score(
        work,
        simulated.X,
        counts_layer=counts_layer,
        n_hvgs=int(min(max(1, n_hvgs), work.n_vars)),
        n_pcs=int(max(2, min(n_pcs, work.n_obs + len(simulated) - 1, work.n_vars - 1))),
        random_state=random_state,
    )
    combined_score = 1.0 - (1.0 - identity_score) * (1.0 - dosage_score)
    max_score = pd.Series(np.maximum(identity_score.to_numpy(dtype=float), dosage_score.to_numpy(dtype=float)), index=work.obs_names, dtype=float)
    gated_scores = _gated_duodose_scores(identity_score, dosage_score)
    doublet_probability = combined_score.clip(0.0, 1.0)
    threshold = _threshold_scores(
        doublet_probability,
        expected_doublet_rate=expected_doublet_rate,
        threshold_method=threshold_method,
    )
    predicted_doublet = doublet_probability >= threshold
    predicted_type = pd.Series("clean", index=work.obs_names, dtype=object)
    subtype = np.where(homotypic_score >= heterotypic_score, "homotypic_doublet", "heterotypic_doublet")
    predicted_type.loc[predicted_doublet] = subtype[predicted_doublet.to_numpy()]

    work.obs["duodose_score"] = combined_score.to_numpy(dtype=float)
    work.obs["duodose_doublet_probability"] = doublet_probability.to_numpy(dtype=float)
    work.obs["duodose_identity_score"] = identity_score.to_numpy(dtype=float)
    work.obs["duodose_dosage_score"] = dosage_score.to_numpy(dtype=float)
    work.obs["duodose_score_combined"] = combined_score.to_numpy(dtype=float)
    work.obs["duodose_score_max"] = max_score.to_numpy(dtype=float)
    work.obs["duodose_identity_rank_score"] = gated_scores["identity_rank"].to_numpy(dtype=float)
    work.obs["duodose_dosage_rank_score"] = gated_scores["dosage_rank"].to_numpy(dtype=float)
    work.obs["duodose_dose_rescue_score"] = gated_scores["dose_rescue"].to_numpy(dtype=float)
    work.obs["duodose_gated_025_score"] = gated_scores["gated_025"].to_numpy(dtype=float)
    work.obs["duodose_gated_050_score"] = gated_scores["gated_050"].to_numpy(dtype=float)
    work.obs["duodose_gated_max_score"] = gated_scores["gated_max"].to_numpy(dtype=float)
    work.obs["duodose_gated_inlier_score"] = gated_scores["gated_inlier"].to_numpy(dtype=float)
    work.obs["duodose_dosage_raw_probability"] = dosage_raw_probability.to_numpy(dtype=float)
    work.obs["duodose_homotypic_score"] = homotypic_score.to_numpy(dtype=float)
    work.obs["duodose_heterotypic_score"] = heterotypic_score.to_numpy(dtype=float)
    for column in identity_diagnostics.columns:
        work.obs[column] = identity_diagnostics[column].to_numpy(dtype=float)
    work.obs["duodose_predicted_doublet"] = predicted_doublet.to_numpy(dtype=bool)
    work.obs["duodose_predicted_doublet_type"] = pd.Categorical(
        predicted_type,
        categories=["clean", "homotypic_doublet", "heterotypic_doublet"],
    )
    highrna_risk = observed_features.get("biological_program_coherence_score", pd.Series(np.nan, index=work.obs_names))
    work.obs["duodose_highRNA_singlet_risk"] = pd.Series(highrna_risk, index=work.obs_names).to_numpy(dtype=float)
    work.obs["duodose_threshold_used"] = float(threshold)
    work.obs["duodose_expected_doublet_rate"] = np.nan if expected_doublet_rate is None else float(expected_doublet_rate)
    work.obs["duodose_library"] = library_series.to_numpy(dtype=object)

    simulated_dosage_probability = (
        simulated_probabilities["homotypic_doublet"].clip(0.0, 1.0)
        + simulated_probabilities["heterotypic_doublet"].clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    dosage_train_score = np.concatenate([dosage_raw_probability.to_numpy(dtype=float), simulated_dosage_probability.to_numpy(dtype=float)])
    dosage_train_y = np.concatenate([np.zeros(work.n_obs, dtype=int), np.ones(len(simulated), dtype=int)])
    synthetic_train_check = {
        "n_observed_singlet_like_training_cells": int(work.n_obs),
        "n_artificial_doublets": int(len(simulated)),
        "n_training_examples": int(len(y_train)),
        "dosage_training_AUROC": _safe_binary_metric(roc_auc_score, dosage_train_y, dosage_train_score),
        "dosage_training_AUPRC": _safe_binary_metric(average_precision_score, dosage_train_y, dosage_train_score),
        "identity_training_AUROC": float(identity_training.get("training_AUROC", np.nan)),
        "identity_training_AUPRC": float(identity_training.get("training_AUPRC", np.nan)),
        "dosage_observed_score_quantiles": _score_quantiles(dosage_raw_probability.to_numpy(dtype=float)),
        "dosage_artificial_score_quantiles": _score_quantiles(simulated_dosage_probability.to_numpy(dtype=float)),
        "identity_observed_score_quantiles": identity_training.get("observed_score_quantiles", {}),
        "identity_artificial_score_quantiles": identity_training.get("artificial_raw_score_quantiles", {}),
        "dosage_observed_fraction_below_0p01": float(np.mean(dosage_raw_probability.to_numpy(dtype=float) < 0.01)),
        "dosage_artificial_fraction_below_0p01": float(np.mean(simulated_dosage_probability.to_numpy(dtype=float) < 0.01)),
        "identity_observed_fraction_below_0p01": float(identity_training.get("observed_fraction_below_0p01", np.nan)),
        "identity_artificial_fraction_below_0p01": float(identity_training.get("artificial_fraction_below_0p01", np.nan)),
        "identity_branch_status": str(identity_training.get("status", "")),
        "identity_k_neighbors": int(identity_training.get("k_neighbors", 0) or 0),
        "identity_expected_artificial_fraction": float(identity_training.get("expected_artificial_fraction", np.nan)),
        "combined_observed_score_quantiles": _score_quantiles(combined_score.to_numpy(dtype=float)),
        "max_observed_score_quantiles": _score_quantiles(max_score.to_numpy(dtype=float)),
        "gated_025_observed_score_quantiles": _score_quantiles(gated_scores["gated_025"].to_numpy(dtype=float)),
        "gated_050_observed_score_quantiles": _score_quantiles(gated_scores["gated_050"].to_numpy(dtype=float)),
        "gated_max_observed_score_quantiles": _score_quantiles(gated_scores["gated_max"].to_numpy(dtype=float)),
        "gated_inlier_observed_score_quantiles": _score_quantiles(gated_scores["gated_inlier"].to_numpy(dtype=float)),
        "combined_fraction_below_0p01": float(np.mean(combined_score.to_numpy(dtype=float) < 0.01)),
    }
    coefficient_rows = _classifier_coefficients(fitted, feature_names, "dosage")

    work.uns["duodose_detection_summary"] = {
        "method": PUBLIC_MODEL_NAME,
        "model": "logistic_safe_features_plus_identity_neighbor_enrichment",
        "n_cells": int(work.n_obs),
        "n_genes": int(work.n_vars),
        "n_simulated_doublets": int(len(simulated)),
        "n_training_examples": int(len(y_train)),
        "training_class_counts": {str(k): int(v) for k, v in y_train.value_counts().items()},
        "n_safe_features": int(len(feature_names)),
        "safe_features": list(map(str, feature_names)),
        "threshold_method": str(threshold_method),
        "threshold_used": float(threshold),
        "expected_doublet_rate": None if expected_doublet_rate is None else float(expected_doublet_rate),
        "predicted_doublets": int(predicted_doublet.sum()),
        "predicted_doublet_rate": float(predicted_doublet.mean()) if len(predicted_doublet) else 0.0,
        "public_score": "duodose_score_combined",
        "identity_branch": "joint_pca_artificial_neighbor_enrichment",
        "dosage_branch": "logistic_safe_features_observed_vs_artificial",
        "gated_score_variants": ["duodose_gated_025_score", "duodose_gated_050_score", "duodose_gated_max_score", "duodose_gated_inlier_score"],
        "dose_gate_quantile": float(DOSAGE_RESCUE_GATE_QUANTILE),
        "identity_inlier_gate_quantile": float(IDENTITY_INLIER_GATE_QUANTILE),
        "cluster_key": str(cluster_key) if cluster_key is not None else DETECTION_CLUSTER_KEY,
        "library_key": str(library_key) if library_key is not None else None,
        "n_clusters": int(work.obs[internal_cluster_key].astype(str).nunique()),
        "n_hvgs": int(n_hvgs),
        "n_pcs": int(n_pcs),
        "random_state": int(random_state),
    }
    if return_debug:
        debug_features = observed_features.copy()
        debug_features["duodose_identity_score"] = identity_score.to_numpy(dtype=float)
        debug_features["duodose_dosage_score"] = dosage_score.to_numpy(dtype=float)
        debug_features["duodose_dosage_raw_probability"] = dosage_raw_probability.to_numpy(dtype=float)
        debug_features["duodose_score_combined"] = combined_score.to_numpy(dtype=float)
        debug_features["duodose_score_max"] = max_score.to_numpy(dtype=float)
        debug_features["duodose_identity_rank_score"] = gated_scores["identity_rank"].to_numpy(dtype=float)
        debug_features["duodose_dosage_rank_score"] = gated_scores["dosage_rank"].to_numpy(dtype=float)
        debug_features["duodose_dose_rescue_score"] = gated_scores["dose_rescue"].to_numpy(dtype=float)
        debug_features["duodose_gated_025_score"] = gated_scores["gated_025"].to_numpy(dtype=float)
        debug_features["duodose_gated_050_score"] = gated_scores["gated_050"].to_numpy(dtype=float)
        debug_features["duodose_gated_max_score"] = gated_scores["gated_max"].to_numpy(dtype=float)
        debug_features["duodose_gated_inlier_score"] = gated_scores["gated_inlier"].to_numpy(dtype=float)
        work.uns["duodose_debug_observed_features"] = debug_features
        work.uns["duodose_synthetic_train_check"] = synthetic_train_check
        work.uns["duodose_synthetic_train_coefficients"] = coefficient_rows
    return work


def scores_dataframe(adata: AnnData) -> pd.DataFrame:
    """Return a publication-facing DuoDose score table from an annotated AnnData."""

    return _score_table_from_obs(adata)


def save_detection_outputs(
    adata: AnnData,
    output_dir: str | Path,
    *,
    input_name: str = "sample",
    write_h5ad: bool = True,
) -> dict[str, Path]:
    """Save DuoDose detection CSV/JSON outputs and optionally an annotated h5ad."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    scores_path = out / "duodose_scores.csv"
    summary_path = out / "duodose_summary.json"
    scores_dataframe(adata).to_csv(scores_path, index=False)
    summary = dict(adata.uns.get("duodose_detection_summary", {}))
    summary["output_scores"] = str(scores_path)
    paths["scores"] = scores_path
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    paths["summary"] = summary_path
    if write_h5ad:
        stem = Path(input_name).stem or "sample"
        h5ad_path = out / f"{stem}_duodose.h5ad"
        adata.write(h5ad_path)
        paths["h5ad"] = h5ad_path
    return paths
