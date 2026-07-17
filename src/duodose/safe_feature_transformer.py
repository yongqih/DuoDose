"""Reference-fitted SafeFeature construction for semi-real DuoDose models.

The historical SafeFeatures path intentionally remains available as
``context_local``.  This module implements the separate ``fitted_reference``
path: all contextual state is fitted once on clean fit-split singlets and is
then reused unchanged for validation, semi-real test, and fully real cells.
"""

from __future__ import annotations

import hashlib
import json
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from duodose.net import FULL_FEATURE_COLUMNS, split_safe_feature_columns
from duodose.safe_feature_manifest import LEGACY_FEATURE_NAME_MAP, build_safe_feature_manifest, migrate_legacy_feature_names


SAFE_FEATURE_MODE_CONTEXT_LOCAL = "context_local"
SAFE_FEATURE_MODE_FITTED_REFERENCE = "fitted_reference"
SAFE_FEATURE_MODES = (SAFE_FEATURE_MODE_CONTEXT_LOCAL, SAFE_FEATURE_MODE_FITTED_REFERENCE)


def _as_csr(matrix) -> sparse.csr_matrix:
    return matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(matrix)


def _counts_matrix(adata: AnnData) -> sparse.csr_matrix:
    matrix = adata.layers["counts"] if "counts" in adata.layers else adata.X
    return _as_csr(matrix)


def _normalize_log_counts(counts: sparse.csr_matrix, target_sum: float) -> sparse.csr_matrix:
    work = counts.astype(np.float32).tocsr(copy=True)
    totals = np.asarray(work.sum(axis=1)).ravel()
    scale = np.divide(float(target_sum), totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)
    work = sparse.diags(scale).dot(work).tocsr()
    work.data = np.log1p(work.data)
    return work


def _stable_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(values, ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return median, scale


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class SafeFeatureTransformer:
    """Fit and reuse context-independent SafeFeature state.

    ``fit`` accepts only the designated clean fit/reference singlet pool.  The
    class deliberately does not inspect truth labels or experimental labels.
    ``transform`` returns the score frame consumed by the existing supervised
    backends, while ``build_model_matrix`` supplies the fixed categorical
    mapping and feature-column order for the final model matrix.
    """

    VERSION = "fitted_reference_safe_features_v1"

    def __init__(
        self,
        *,
        random_state: int = 0,
        reference_seed: int | None = None,
        n_components: int = 30,
        n_clusters: int = 12,
        n_neighbors: int = 30,
        n_artificial_doublets: int | None = None,
        target_sum: float = 1e4,
        model_feature_allowlist: Iterable[str] | None = None,
    ) -> None:
        self.random_state = int(random_state)
        self.reference_seed = int(reference_seed if reference_seed is not None else random_state)
        self.n_components = int(n_components)
        self.n_clusters = int(n_clusters)
        self.n_neighbors = int(n_neighbors)
        self.n_artificial_doublets = n_artificial_doublets
        self.target_sum = float(target_sum)
        self.model_feature_allowlist = tuple(str(value) for value in (model_feature_allowlist or ()))
        self._is_fitted = False

    def fit(
        self,
        reference_adata: AnnData,
        *,
        reference_pool_id: str | None = None,
        dataset: str | None = None,
    ) -> "SafeFeatureTransformer":
        """Fit all state using only clean fit/reference singlets."""

        if reference_adata.n_obs < 3 or reference_adata.n_vars < 3:
            raise ValueError("fitted-reference SafeFeatures require at least three reference cells and genes")
        if reference_adata.obs_names.duplicated().any():
            raise ValueError("fitted-reference SafeFeatures require unique reference cell IDs")
        if reference_adata.var_names.duplicated().any():
            raise ValueError("fitted-reference SafeFeatures require unique reference gene names")

        self.selected_genes_ = reference_adata.var_names.astype(str).copy()
        self.reference_cell_ids_ = reference_adata.obs_names.astype(str).copy()
        self.reference_dataset_ = str(dataset or "")
        self.reference_pool_id_ = str(reference_pool_id or self._reference_pool_id(self.reference_cell_ids_))
        counts = _counts_matrix(reference_adata)
        self.reference_n_cells_ = int(counts.shape[0])
        self.reference_n_genes_ = int(counts.shape[1])

        normalized = _normalize_log_counts(counts, self.target_sum)
        components = int(max(2, min(self.n_components, counts.shape[0] - 1, counts.shape[1] - 1)))
        self.pca_model_ = TruncatedSVD(n_components=components, random_state=self.random_state)
        raw_embedding = self.pca_model_.fit_transform(normalized)
        self.embedding_scaler_ = StandardScaler().fit(raw_embedding)
        self.reference_embedding_ = self.embedding_scaler_.transform(raw_embedding)

        n_clusters = int(min(max(2, self.n_clusters), max(2, counts.shape[0] // 25)))
        self.cluster_model_ = KMeans(n_clusters=n_clusters, n_init=20, random_state=self.random_state).fit(self.reference_embedding_)
        self.reference_clusters_ = np.asarray([f"cluster_{value}" for value in self.cluster_model_.labels_], dtype=object)
        self.cluster_labels_ = tuple(sorted(pd.unique(self.reference_clusters_)))

        self._fit_cluster_statistics(counts, self.reference_embedding_, self.reference_clusters_)
        self.reference_neighbor_index_ = NearestNeighbors(
            n_neighbors=min(max(1, self.n_neighbors + 1), self.reference_n_cells_), metric="euclidean"
        ).fit(self.reference_embedding_)
        self._fit_artificial_bank(counts)
        self._fit_ecdf_calibrators(counts)
        self._fit_categorical_mappings(reference_adata)

        # ``transform`` is used here only to materialize the fit-only fixed
        # categorical mapping and feature order after every reference state is
        # available.
        self._is_fitted = True
        reference_frame = self.transform(reference_adata, dataset_id=self.reference_dataset_ or None)
        self.model_feature_columns_ = list(self._model_matrix_from_frame(reference_frame, freeze_columns=False).columns)
        self.manifest_ = self._build_manifest()
        self.transformer_id_ = self._transformer_id()
        return self

    def fit_transform(
        self,
        reference_adata: AnnData,
        *,
        reference_pool_id: str | None = None,
        dataset: str | None = None,
        dataset_id: str | None = None,
    ) -> pd.DataFrame:
        """Fit on a reference singlet pool and transform those same cells."""

        return self.fit(reference_adata, reference_pool_id=reference_pool_id, dataset=dataset).transform(
            reference_adata,
            dataset_id=dataset_id or dataset,
        )

    def transform(
        self,
        query_adata: AnnData,
        *,
        dataset_id: str | None = None,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """Transform query cells using only state fitted in :meth:`fit`."""

        self._require_fitted()
        if query_adata.obs_names.duplicated().any():
            raise ValueError("fitted-reference SafeFeature transform requires unique query cell IDs")
        counts = self._aligned_counts(query_adata)
        normalized = _normalize_log_counts(counts, self.target_sum)
        embedding = self.embedding_scaler_.transform(self.pca_model_.transform(normalized))
        cluster_numbers = self.cluster_model_.predict(embedding)
        clusters = np.asarray([f"cluster_{value}" for value in cluster_numbers], dtype=object)
        raw = self._raw_descriptors(counts, embedding, clusters, query_adata.obs_names.astype(str).to_numpy())
        return self._feature_frame(query_adata, raw, clusters, dataset_id=dataset_id, random_state=random_state)

    def build_model_matrix(self, score_frame: pd.DataFrame) -> pd.DataFrame:
        """Return the fixed-column SafeFeatures matrix for a transformed frame."""

        self._require_fitted()
        return self._model_matrix_from_frame(migrate_legacy_feature_names(score_frame), freeze_columns=True)

    def metadata(self) -> dict[str, object]:
        """Return compact fitted-reference metadata for audits and exports."""

        self._require_fitted()
        return {
            "safe_feature_mode": SAFE_FEATURE_MODE_FITTED_REFERENCE,
            "safe_feature_transformer_id": self.transformer_id_,
            "safe_feature_reference_pool_id": self.reference_pool_id_,
            "safe_feature_reference_n_cells": int(self.reference_n_cells_),
            "safe_feature_reference_seed": int(self.reference_seed),
        }

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        """Persist the fitted transformer plus readable state summaries."""

        self._require_fitted()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        model_path = output / "safe_feature_transformer.joblib"
        config_path = output / "safe_feature_transformer_config.json"
        stats_path = output / "safe_feature_transformer_reference_statistics.csv"
        manifest_path = output / "safe_feature_transformer_manifest.csv"
        joblib.dump(self, model_path)
        config_path.write_text(json.dumps(self._config_summary(), indent=2), encoding="utf-8")
        self.reference_statistics_.to_csv(stats_path, index=False)
        self.manifest_.to_csv(manifest_path, index=False)
        return {
            "joblib": model_path,
            "config": config_path,
            "reference_statistics": stats_path,
            "manifest": manifest_path,
        }

    @classmethod
    def load(cls, input_dir: str | Path) -> "SafeFeatureTransformer":
        """Load a transformer saved by :meth:`save`."""

        path = Path(input_dir)
        model_path = path if path.suffix == ".joblib" else path / "safe_feature_transformer.joblib"
        value = joblib.load(model_path)
        if not isinstance(value, cls):
            raise TypeError(f"{model_path} does not contain a {cls.__name__}")
        value._require_fitted()
        legacy_columns = list(getattr(value, "model_feature_columns_", ()))
        migrated_columns = [LEGACY_FEATURE_NAME_MAP.get(str(column), str(column)) for column in legacy_columns]
        if migrated_columns != legacy_columns:
            value.model_feature_columns_ = migrated_columns
            value.model_feature_allowlist = tuple(LEGACY_FEATURE_NAME_MAP.get(str(column), str(column)) for column in value.model_feature_allowlist)
            value.manifest_ = value._build_manifest()
            value.loaded_feature_schema_migration_ = "legacy_pre_model_names_to_handcrafted_v1"
        return value

    def _reference_pool_id(self, cell_ids: Iterable[str]) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.reference_seed).encode("utf-8"))
        for cell_id in cell_ids:
            digest.update(str(cell_id).encode("utf-8"))
            digest.update(b"\n")
        return f"reference_{digest.hexdigest()[:16]}"

    def _transformer_id(self) -> str:
        payload = {
            "version": self.VERSION,
            "reference_pool_id": self.reference_pool_id_,
            "reference_seed": self.reference_seed,
            "selected_genes": list(self.selected_genes_),
            "clusters": list(self.cluster_labels_),
            "model_feature_columns": list(self.model_feature_columns_),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"safeft_{hashlib.sha256(encoded).hexdigest()[:16]}"

    def _config_summary(self) -> dict[str, object]:
        return {
            "version": self.VERSION,
            **self.metadata(),
            "reference_dataset": self.reference_dataset_,
            "reference_n_genes": int(self.reference_n_genes_),
            "selected_genes": list(self.selected_genes_),
            "n_components": int(getattr(self.pca_model_, "n_components", self.n_components)),
            "n_clusters": int(len(self.cluster_labels_)),
            "n_neighbors": int(self.n_neighbors),
            "n_artificial_doublets": int(self.artificial_embedding_.shape[0]),
            "target_sum": float(self.target_sum),
            "feature_columns": list(self.model_feature_columns_),
            "model_feature_allowlist": list(self.model_feature_allowlist),
            "categorical_levels": self.categorical_levels_,
            "scrublet_feature": "fixed_reference_scrublet_compatible_knn",
        }

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("SafeFeatureTransformer must be fitted before use")

    def _aligned_counts(self, adata: AnnData) -> sparse.csr_matrix:
        names = adata.var_names.astype(str)
        if names.equals(self.selected_genes_):
            return _counts_matrix(adata)
        positions = names.get_indexer(self.selected_genes_)
        if (positions < 0).any():
            missing = self.selected_genes_[positions < 0][:5].tolist()
            raise ValueError(f"query AnnData is missing reference-selected genes: {missing}")
        return _counts_matrix(adata)[:, positions].tocsr()

    def _fit_cluster_statistics(self, counts: sparse.csr_matrix, embedding: np.ndarray, clusters: np.ndarray) -> None:
        ncount = np.asarray(counts.sum(axis=1)).ravel().astype(float)
        nfeature = np.diff(counts.indptr).astype(float)
        log_ncount = np.log1p(ncount)
        log_nfeature = np.log1p(nfeature)
        rows: list[dict[str, object]] = []
        self.cluster_stats_: dict[str, dict[str, float]] = {}
        for cluster in self.cluster_labels_:
            mask = clusters == cluster
            centroid = self.cluster_model_.cluster_centers_[int(str(cluster).split("_")[-1])]
            distance = np.linalg.norm(embedding[mask] - centroid, axis=1)
            count_median, count_scale = _stable_scale(log_ncount[mask])
            feature_median, feature_scale = _stable_scale(log_nfeature[mask])
            distance_median, distance_scale = _stable_scale(distance)
            stats = {
                "n_cells": float(mask.sum()),
                "abundance": float(mask.mean()),
                "log_nCount_median": count_median,
                "log_nCount_scale": count_scale,
                "log_nFeature_median": feature_median,
                "log_nFeature_scale": feature_scale,
                "centroid_distance_median": distance_median,
                "centroid_distance_scale": distance_scale,
            }
            self.cluster_stats_[str(cluster)] = stats
            rows.append({"cluster": str(cluster), **stats})
        global_count_median, global_count_scale = _stable_scale(log_ncount)
        global_feature_median, global_feature_scale = _stable_scale(log_nfeature)
        global_distance_median, global_distance_scale = _stable_scale(
            np.linalg.norm(embedding - self.cluster_model_.cluster_centers_[self.cluster_model_.labels_], axis=1)
        )
        self.global_stats_ = {
            "n_cells": float(len(ncount)),
            "abundance": 1.0,
            "log_nCount_median": global_count_median,
            "log_nCount_scale": global_count_scale,
            "log_nFeature_median": global_feature_median,
            "log_nFeature_scale": global_feature_scale,
            "centroid_distance_median": global_distance_median,
            "centroid_distance_scale": global_distance_scale,
        }
        self.reference_statistics_ = pd.DataFrame(rows)

    def _fit_artificial_bank(self, counts: sparse.csr_matrix) -> None:
        rng = np.random.default_rng(self.reference_seed + 104729)
        requested = self.n_artificial_doublets
        n_artificial = int(requested if requested is not None else min(self.reference_n_cells_, 2000))
        n_artificial = max(1, n_artificial)
        clusters = np.asarray(self.reference_clusters_, dtype=object)
        pools = {cluster: np.flatnonzero(clusters == cluster) for cluster in self.cluster_labels_}
        valid_homo = [cluster for cluster, values in pools.items() if len(values) >= 2]
        valid_hetero = [cluster for cluster, values in pools.items() if len(values) >= 1]
        rows: list[sparse.csr_matrix] = []
        pair_rows: list[dict[str, object]] = []
        for number in range(n_artificial):
            if valid_homo and (not valid_hetero or number % 2 == 0):
                cluster = str(rng.choice(valid_homo))
                left, right = rng.choice(pools[cluster], size=2, replace=False)
                subtype = "homotypic"
            elif len(valid_hetero) >= 2:
                cluster_a, cluster_b = rng.choice(valid_hetero, size=2, replace=False)
                left = int(rng.choice(pools[str(cluster_a)]))
                right = int(rng.choice(pools[str(cluster_b)]))
                subtype = "heterotypic"
            else:
                left, right = rng.choice(np.arange(self.reference_n_cells_), size=2, replace=False)
                subtype = "mixed_fallback"
            rows.append((counts[int(left)] + counts[int(right)]).tocsr())
            pair_rows.append(
                {
                    "artificial_id": f"fixed_artificial_{number:06d}",
                    "parent_1_id": str(self.reference_cell_ids_[int(left)]),
                    "parent_2_id": str(self.reference_cell_ids_[int(right)]),
                    "synthetic_subtype": subtype,
                }
            )
        artificial_counts = sparse.vstack(rows, format="csr")
        artificial_normalized = _normalize_log_counts(artificial_counts, self.target_sum)
        self.artificial_embedding_ = self.embedding_scaler_.transform(self.pca_model_.transform(artificial_normalized))
        self.artificial_pair_map_ = pd.DataFrame(pair_rows)
        self.artificial_neighbor_index_ = NearestNeighbors(
            n_neighbors=min(max(1, self.n_neighbors), self.artificial_embedding_.shape[0]), metric="euclidean"
        ).fit(self.artificial_embedding_)
        self.combined_reference_embedding_ = np.vstack([self.reference_embedding_, self.artificial_embedding_])
        self.combined_reference_is_artificial_ = np.concatenate(
            [np.zeros(self.reference_n_cells_, dtype=bool), np.ones(self.artificial_embedding_.shape[0], dtype=bool)]
        )
        self.combined_neighbor_index_ = NearestNeighbors(
            n_neighbors=min(max(1, self.n_neighbors + 1), self.combined_reference_embedding_.shape[0]), metric="euclidean"
        ).fit(self.combined_reference_embedding_)

    def _fit_ecdf_calibrators(self, counts: sparse.csr_matrix) -> None:
        raw = self._raw_descriptors(
            counts,
            self.reference_embedding_,
            self.reference_clusters_,
            self.reference_cell_ids_.to_numpy(dtype=str),
        )
        self.ecdf_reference_values_: dict[str, np.ndarray] = {}
        self.tail_thresholds_: dict[str, float] = {}
        for name, values in raw.items():
            value = np.asarray(values, dtype=float)
            finite = np.sort(value[np.isfinite(value)])
            if finite.size == 0:
                finite = np.array([0.0], dtype=float)
            self.ecdf_reference_values_[name] = finite
            self.tail_thresholds_[name] = float(np.quantile(finite, 0.90))

    def _fit_categorical_mappings(self, reference_adata: AnnData) -> None:
        samples = reference_adata.obs.get("sample_id", pd.Series(self.reference_dataset_ or "reference", index=reference_adata.obs_names))
        sample_levels = sorted(pd.Series(samples, index=reference_adata.obs_names).astype(str).dropna().unique().tolist())
        if "__unknown__" not in sample_levels:
            sample_levels.append("__unknown__")
        self.categorical_levels_ = {
            "benchmark_cluster": list(self.cluster_labels_),
            "duodose_cluster": list(self.cluster_labels_),
            "sample_id": sample_levels,
        }

    def _cluster_stat(self, clusters: np.ndarray, name: str) -> np.ndarray:
        return np.asarray([self.cluster_stats_.get(str(cluster), self.global_stats_)[name] for cluster in clusters], dtype=float)

    def _raw_descriptors(
        self,
        counts: sparse.csr_matrix,
        embedding: np.ndarray,
        clusters: np.ndarray,
        query_ids: np.ndarray,
    ) -> dict[str, np.ndarray]:
        ncount = np.asarray(counts.sum(axis=1)).ravel().astype(float)
        nfeature = np.diff(counts.indptr).astype(float)
        log_ncount = np.log1p(ncount)
        log_nfeature = np.log1p(nfeature)
        count_z = (log_ncount - self._cluster_stat(clusters, "log_nCount_median")) / self._cluster_stat(clusters, "log_nCount_scale")
        feature_z = (log_nfeature - self._cluster_stat(clusters, "log_nFeature_median")) / self._cluster_stat(clusters, "log_nFeature_scale")
        centroid_numbers = np.asarray([int(str(cluster).split("_")[-1]) for cluster in clusters], dtype=int)
        centroid_distance = np.linalg.norm(embedding - self.cluster_model_.cluster_centers_[centroid_numbers], axis=1)
        distance_z = (centroid_distance - self._cluster_stat(clusters, "centroid_distance_median")) / self._cluster_stat(clusters, "centroid_distance_scale")
        neighbor_purity, neighbor_distance = self._reference_neighbor_features(embedding, clusters, query_ids)
        artificial_proximity, scrublet_compatible = self._artificial_features(embedding, query_ids)
        dosage_raw = _sigmoid(0.75 * count_z + 0.25 * feature_z)
        identity_raw = np.clip(0.55 * _sigmoid(distance_z) + 0.45 * (1.0 - neighbor_purity), 0.0, 1.0)
        sensitive_raw = np.maximum(dosage_raw, scrublet_compatible)
        duo_raw = 1.0 - (1.0 - dosage_raw) * (1.0 - scrublet_compatible)
        return {
            "ncount_z": np.asarray(count_z, dtype=float),
            "nfeature_z": np.asarray(feature_z, dtype=float),
            "centroid_distance_z": np.asarray(distance_z, dtype=float),
            "neighbor_purity": np.asarray(neighbor_purity, dtype=float),
            "neighbor_distance": np.asarray(neighbor_distance, dtype=float),
            "artificial_proximity": np.asarray(artificial_proximity, dtype=float),
            "scrublet_compatible": np.asarray(scrublet_compatible, dtype=float),
            "dosage_raw": np.asarray(dosage_raw, dtype=float),
            "identity_raw": np.asarray(identity_raw, dtype=float),
            "sensitive_raw": np.asarray(sensitive_raw, dtype=float),
            "duodose_raw": np.asarray(duo_raw, dtype=float),
        }

    def _reference_neighbor_features(
        self,
        embedding: np.ndarray,
        clusters: np.ndarray,
        query_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_neighbors = min(max(1, self.n_neighbors + 1), self.reference_n_cells_)
        distances, indices = self.reference_neighbor_index_.kneighbors(embedding, n_neighbors=n_neighbors, return_distance=True)
        purity = np.zeros(len(query_ids), dtype=float)
        mean_distance = np.zeros(len(query_ids), dtype=float)
        reference_ids = self.reference_cell_ids_.to_numpy(dtype=str)
        for row, (query_id, cluster) in enumerate(zip(query_ids, clusters, strict=True)):
            keep = reference_ids[indices[row]] != str(query_id)
            local_indices = indices[row][keep][: self.n_neighbors]
            local_distances = distances[row][keep][: self.n_neighbors]
            if local_indices.size == 0:
                purity[row] = 1.0
                mean_distance[row] = 0.0
            else:
                purity[row] = float(np.mean(self.reference_clusters_[local_indices] == cluster))
                mean_distance[row] = float(np.mean(local_distances))
        return purity, mean_distance

    def _artificial_features(self, embedding: np.ndarray, query_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distances, _ = self.artificial_neighbor_index_.kneighbors(embedding, return_distance=True)
        proximity = 1.0 / (1.0 + np.mean(distances, axis=1))
        n_neighbors = min(max(1, self.n_neighbors + 1), self.combined_reference_embedding_.shape[0])
        _, indices = self.combined_neighbor_index_.kneighbors(embedding, n_neighbors=n_neighbors, return_distance=True)
        reference_ids = self.reference_cell_ids_.to_numpy(dtype=str)
        scrublet = np.zeros(len(query_ids), dtype=float)
        for row, query_id in enumerate(query_ids):
            local = indices[row]
            is_reference = local < self.reference_n_cells_
            same_reference = np.zeros(local.size, dtype=bool)
            same_reference[is_reference] = reference_ids[local[is_reference]] == str(query_id)
            local = local[~same_reference][: self.n_neighbors]
            scrublet[row] = float(self.combined_reference_is_artificial_[local].mean()) if local.size else 0.0
        return proximity, scrublet

    def _ecdf(self, name: str, values: np.ndarray) -> np.ndarray:
        reference = self.ecdf_reference_values_[name]
        ranks = np.searchsorted(reference, np.asarray(values, dtype=float), side="right") / float(len(reference))
        return np.clip(ranks, 0.0, 1.0)

    def _tail(self, name: str, values: np.ndarray) -> np.ndarray:
        return np.clip((self._ecdf(name, values) - 0.5) / 0.5, 0.0, 1.0)

    def _feature_frame(
        self,
        adata: AnnData,
        raw: dict[str, np.ndarray],
        clusters: np.ndarray,
        *,
        dataset_id: str | None,
        random_state: int | None,
    ) -> pd.DataFrame:
        obs = adata.obs.copy()
        obs.index = adata.obs_names
        frame = pd.DataFrame(index=adata.obs_names)
        frame["cell_id"] = adata.obs_names.astype(str)
        frame["dataset"] = str(dataset_id or self.reference_dataset_ or "")
        frame["seed"] = int(self.reference_seed if random_state is None else random_state)
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
        frame["semireal_cluster"] = clusters
        frame["benchmark_cluster"] = clusters
        frame["duodose_cluster"] = clusters
        samples = obs.get("sample_id", pd.Series(self.reference_dataset_ or "reference", index=obs.index)).astype(str)
        frame["sample_id"] = samples.where(samples.isin(self.categorical_levels_["sample_id"]), "__unknown__")
        if "experimental_doublet" in obs:
            frame["experimental_doublet"] = obs["experimental_doublet"].to_numpy()

        counts = self._aligned_counts(adata)
        ncount = np.asarray(counts.sum(axis=1)).ravel().astype(float)
        nfeature = np.diff(counts.indptr).astype(float)
        scrub = self._ecdf("scrublet_compatible", raw["scrublet_compatible"])
        dosage = self._ecdf("dosage_raw", raw["dosage_raw"])
        identity = self._ecdf("identity_raw", raw["identity_raw"])
        identity_deviation = 1.0 - identity
        sensitive = self._ecdf("sensitive_raw", raw["sensitive_raw"])
        duo = self._ecdf("duodose_raw", raw["duodose_raw"])
        dosage_tail = self._tail("dosage_raw", raw["dosage_raw"])
        identity_tail = self._tail("identity_raw", raw["identity_raw"])
        scrub_tail = self._tail("scrublet_compatible", raw["scrublet_compatible"])
        duo_tail = self._tail("duodose_raw", raw["duodose_raw"])
        hybrid = 1.0 - (1.0 - scrub_tail) * (1.0 - dosage_tail)
        abundance = self._cluster_stat(clusters, "abundance")
        columns = {
            "scrublet_score": scrub,
            "handcrafted_homotypic_score": dosage,
            "handcrafted_identity_mixture_score": identity,
            "handcrafted_combined_score": duo,
            "handcrafted_sensitive_score": sensitive,
            "handcrafted_dosage_raw_score": raw["dosage_raw"],
            "handcrafted_dosage_reference_ecdf": dosage,
            "handcrafted_dosage_reference_tail": dosage_tail,
            "hybrid_overall_score": hybrid,
            "hybrid_homotypic_score": dosage_tail,
            "hybrid_heterotypic_score": scrub_tail,
            "nCount": ncount,
            "log_nCount": np.log1p(ncount),
            # Row-local library-complexity shape. Two independent parent cells
            # tend to contribute complementary detected genes relative to their
            # total UMI count, whereas a coherent high-RNA singlet mainly moves
            # along the library-size axis.
            "library_complexity_balance": np.log1p(nfeature) - 0.5 * np.log1p(ncount),
            "cluster_nCount_z": raw["ncount_z"],
            "handcrafted_homotypic_reference_score": dosage,
            "handcrafted_artificial_doublet_neighbor_score": scrub,
            "handcrafted_homotypic_reference_ecdf": dosage,
            "handcrafted_artificial_doublet_neighbor_ecdf": scrub,
            "handcrafted_homotypic_reference_tail": dosage_tail,
            "handcrafted_artificial_doublet_neighbor_tail": scrub_tail,
            "handcrafted_combined_raw_score": raw["duodose_raw"],
            "handcrafted_combined_reference_ecdf": duo,
            "handcrafted_combined_reference_tail": duo_tail,
            "handcrafted_artificial_doublet_compatible_score": scrub,
            "dosage_outlier_score": dosage,
            "identity_inlier_score": identity,
            "uniform_dosage_inflation_score": dosage,
            # Frozen compatibility alias. Audit-facing semantics call this
            # identity_deviation_score and never treat it as independent.
            "biological_program_coherence_score": identity_deviation,
            "handcrafted_homotypic_candidate_score": sensitive,
            "handcrafted_homotypic_dosage_score": dosage,
            "module_residual_rank_mean": dosage,
            "module_residual_rank_spread": np.abs(dosage - identity),
            "cluster_count_robust_z": raw["ncount_z"],
            "cluster_gene_robust_z": raw["nfeature_z"],
            "cluster_stable_dosage_robust_z": raw["ncount_z"],
            "cluster_marker_dosage_robust_z": raw["ncount_z"],
            "dosage_residual": raw["ncount_z"],
            "cluster_abundance": abundance,
            "cluster_level_expected_homotypic_burden": abundance**2,
            "benchmark_cluster_frequency": abundance,
            "handcrafted_combined_score_legacy_alias": duo,
            "handcrafted_sensitive_max_score": sensitive,
            "handcrafted_dosage_gated_inlier_score": dosage,
        }
        for name, values in columns.items():
            frame[name] = np.asarray(values, dtype=float)
        frame["nFeature"] = nfeature
        frame["log_nFeature"] = np.log1p(nfeature)
        frame["safe_feature_mode"] = SAFE_FEATURE_MODE_FITTED_REFERENCE
        frame["safe_feature_transformer_id"] = self.transformer_id_ if hasattr(self, "transformer_id_") else "pending"
        frame["safe_feature_reference_pool_id"] = self.reference_pool_id_
        frame["safe_feature_reference_n_cells"] = int(self.reference_n_cells_)
        frame["safe_feature_reference_seed"] = int(self.reference_seed)
        return frame

    def _model_matrix_from_frame(self, score_frame: pd.DataFrame, *, freeze_columns: bool) -> pd.DataFrame:
        columns = [column for column in FULL_FEATURE_COLUMNS if column in score_frame]
        matrix = score_frame.loc[:, columns].copy() if columns else pd.DataFrame(index=score_frame.index)
        for categorical, levels in self.categorical_levels_.items():
            values = score_frame.get(categorical, pd.Series("__unknown__", index=score_frame.index)).astype(str)
            values = values.where(values.isin(levels), "__unknown__")
            for level in levels:
                matrix[f"{categorical}_{level}"] = (values == level).astype(float)
        matrix = matrix.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        for column in matrix.columns:
            if matrix[column].isna().all():
                matrix[column] = 0.0
        included, _ = split_safe_feature_columns(matrix.columns)
        matrix = matrix.loc[:, included].astype(float)
        if self.model_feature_allowlist:
            selected: list[str] = []
            for pattern in self.model_feature_allowlist:
                matches = [column for column in matrix.columns if fnmatch(str(column), pattern)]
                if not matches:
                    raise ValueError(f"frozen SafeFeature allowlist entry matched no model feature: {pattern!r}")
                selected.extend(column for column in matches if column not in selected)
            matrix = matrix.loc[:, selected]
        if freeze_columns:
            matrix = matrix.reindex(columns=self.model_feature_columns_, fill_value=0.0)
        return matrix

    def _build_manifest(self) -> pd.DataFrame:
        return build_safe_feature_manifest(self.model_feature_columns_)
