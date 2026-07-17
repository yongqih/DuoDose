"""Shared fitted-reference training path for controlled benchmark bundles."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import pandas as pd
from anndata import AnnData

from .models.registry import BACKEND_SPECS
from .net import train_predict_diagnostic_model
from .rf_weighting import FORMAL_HIGH_RNA_NEGATIVE_WEIGHT, formal_rf_sample_weights
from .safe_feature_transformer import SafeFeatureTransformer
from .semireal_bundle import SemiRealSplitBundle


@dataclass
class BenchmarkCoreResult:
    """Outputs from the frozen parent-disjoint semi-real benchmark engine."""

    transformer: SafeFeatureTransformer
    fit_scores: pd.DataFrame
    validation_scores: pd.DataFrame
    test_scores: pd.DataFrame
    observed_scores: pd.DataFrame
    test_features: pd.DataFrame
    observed_features: pd.DataFrame
    method_probabilities_test: dict[str, pd.DataFrame]
    method_probabilities_observed: dict[str, pd.DataFrame]
    fitted_backends: dict[str, object]
    training_summaries: list[dict[str, Any]]
    timings: dict[str, float]


def _reference_fit_rows(
    bundle: SemiRealSplitBundle,
    explicit_reference_adata: AnnData | None = None,
) -> AnnData:
    if explicit_reference_adata is not None:
        reference = explicit_reference_adata.copy()
        if reference.n_obs < 3:
            raise ValueError("fitted-reference SafeFeatures require at least three explicit reference singlets")
        labels = reference.obs.get("true_label", pd.Series("clean", index=reference.obs_names)).astype(str)
        if labels.isin({"homotypic_doublet", "heterotypic_doublet"}).any():
            raise ValueError("explicit fitted-reference input must contain singlets only")
        return reference
    origin = bundle.fit_adata.obs.get(
        "semireal_origin", pd.Series("", index=bundle.fit_adata.obs_names)
    ).astype(str)
    reference = bundle.fit_adata[
        origin.isin({"observed_background", "real_labeled_singlet"}).to_numpy(), :
    ].copy()
    if reference.n_obs < 3:
        raise ValueError("fitted-reference SafeFeatures require fit-split singlet reference rows")
    return reference


def _backend_training_kwargs(backend: str, train_scores: pd.DataFrame) -> dict[str, object]:
    if backend != "rf":
        return {}
    return {
        "sample_weight": formal_rf_sample_weights(train_scores),
        "high_rna_negative_weight": FORMAL_HIGH_RNA_NEGATIVE_WEIGHT,
    }


def run_benchmark_core(
    bundle: SemiRealSplitBundle,
    observed_adata: AnnData,
    *,
    protocol: Mapping[str, Any],
    seed: int,
    backends: Iterable[str] = ("rf", "dl"),
    device: str = "auto",
    amp: bool = False,
    dl_max_epochs: int = 200,
    dl_patience: int = 20,
    dl_batch_size: int | None = None,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
    verbose_progress: bool = False,
    construction_seconds: float = 0.0,
    reference_adata: AnnData | None = None,
) -> BenchmarkCoreResult:
    """Fit and evaluate the frozen downstream path on a preconstructed bundle.

    The function does not construct cells or consume experimental labels. The
    controlled semi-real builder supplies fit, validation, and test matrices
    through ``SemiRealSplitBundle``.
    """

    started = time.perf_counter()
    clustering = protocol["clustering"]
    construction = protocol["construction"]
    feature_started = time.perf_counter()
    reference = _reference_fit_rows(bundle, reference_adata)
    reference_pool_id = (
        f"{bundle.dataset}|seed={int(seed)}|variant={construction['construction_variant']}|"
        f"{'explicit_reference_singlets' if reference_adata is not None else 'fit_split_clean_singlets'}"
    )
    if progress_callback is not None:
        progress_callback({"event": "milestone", "message": "fitting frozen-reference SafeFeature transformer"})
    transformer = SafeFeatureTransformer(
        random_state=int(seed),
        reference_seed=int(seed),
        n_components=int(clustering["n_pcs"]),
        n_clusters=int(clustering["n_clusters"]),
        n_neighbors=int(clustering["n_neighbors"]),
        model_feature_allowlist=protocol["features"]["allowlist"],
    ).fit(reference, reference_pool_id=reference_pool_id, dataset=bundle.dataset)
    fit_scores = transformer.transform(bundle.fit_adata, dataset_id=f"{bundle.dataset}_fit", random_state=int(seed))
    validation_scores = transformer.transform(
        bundle.val_adata, dataset_id=f"{bundle.dataset}_validation", random_state=int(seed)
    )
    test_scores = transformer.transform(bundle.test_adata, dataset_id=f"{bundle.dataset}_test", random_state=int(seed))
    observed_input = observed_adata.copy()
    observed_scores = transformer.transform(observed_input, dataset_id=bundle.dataset, random_state=int(seed))
    for column in ("true_label", "true_doublet_label", "doublet_subtype"):
        if column in observed_scores:
            observed_scores[column] = pd.NA
    test_features = transformer.build_model_matrix(test_scores)
    observed_features = transformer.build_model_matrix(observed_scores)
    safe_feature_seconds = time.perf_counter() - feature_started

    train_scores = pd.concat([fit_scores, validation_scores], axis=0)
    test_probabilities: dict[str, pd.DataFrame] = {}
    observed_probabilities: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, Any]] = []
    fitted_backends: dict[str, object] = {}
    model_training_seconds = 0.0
    prediction_seconds = 0.0
    for backend in backends:
        if backend not in BACKEND_SPECS:
            raise ValueError(f"unsupported manuscript backend {backend!r}; choose rf or dl")
        spec = BACKEND_SPECS[backend]
        if progress_callback is not None:
            progress_callback({"event": "method_start", "message": f"starting {spec.display_name}", "method": spec.display_name})
        kwargs: dict[str, Any] = {
            "train_cell_scores": train_scores,
            "test_cell_scores": test_scores,
            "method": spec.internal_name,
            "random_state": int(seed),
            "net_train_seed": int(seed),
            "train_index": fit_scores.index,
            "validation_index": validation_scores.index,
            "safe_feature_transformer": transformer,
            "progress_callback": progress_callback,
            "verbose_progress": bool(verbose_progress),
        }
        if backend == "dl":
            kwargs.update(
                device=device,
                use_amp=bool(amp),
                max_epochs=int(dl_max_epochs),
                patience=int(dl_patience),
                batch_size=dl_batch_size,
            )
        else:
            kwargs.update(_backend_training_kwargs(backend, train_scores))
        training_started = time.perf_counter()
        result = train_predict_diagnostic_model(**kwargs)
        model_training_seconds += time.perf_counter() - training_started
        summary = dict(result.get("summary", {}))
        summary.update(dataset=bundle.dataset, seed=int(seed), backend=backend, public_method_name=spec.display_name)
        summaries.append(summary)
        fitted = result.get("fitted_backend")
        if summary.get("status") != "success" or fitted is None:
            raise RuntimeError(f"{backend} training failed for {bundle.dataset}: {summary.get('message', 'unknown error')}")
        test_probabilities[spec.display_name] = result["test_probabilities"].reindex(test_scores.index)
        fitted_backends[spec.display_name] = fitted
        prediction_started = time.perf_counter()
        observed_probabilities[spec.display_name] = fitted.predict_probabilities(observed_scores).reindex(observed_input.obs_names)
        prediction_seconds += time.perf_counter() - prediction_started
        if progress_callback is not None:
            progress_callback({"event": "method_complete", "message": f"completed {spec.display_name}", "method": spec.display_name})

    return BenchmarkCoreResult(
        transformer=transformer,
        fit_scores=fit_scores,
        validation_scores=validation_scores,
        test_scores=test_scores,
        observed_scores=observed_scores,
        test_features=test_features,
        observed_features=observed_features,
        method_probabilities_test=test_probabilities,
        method_probabilities_observed=observed_probabilities,
        fitted_backends=fitted_backends,
        training_summaries=summaries,
        timings={
            "semi_real_construction_seconds": float(construction_seconds),
            "safe_feature_construction_seconds": float(safe_feature_seconds),
            "model_training_seconds": float(model_training_seconds),
            "prediction_seconds": float(prediction_seconds),
            "shared_benchmark_core_seconds": float(time.perf_counter() - started),
        },
    )
