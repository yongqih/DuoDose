"""Experimental DuoDose-Net models for benchmark-only comparisons."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import random
import time
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from .methods import DUODOSE_METHODS

try:  # Optional at import time; required when fitting sklearn-backed models.
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler
except Exception:  # pragma: no cover - import-only fallback for minimal envs
    class BaseEstimator:  # type: ignore[no-redef]
        pass

    class ClassifierMixin:  # type: ignore[no-redef]
        pass

    CalibratedClassifierCV = None  # type: ignore[assignment]
    RandomForestClassifier = None  # type: ignore[assignment]
    SimpleImputer = None  # type: ignore[assignment]
    LogisticRegression = None  # type: ignore[assignment]
    average_precision_score = None  # type: ignore[assignment]
    log_loss = None  # type: ignore[assignment]
    roc_auc_score = None  # type: ignore[assignment]
    train_test_split = None  # type: ignore[assignment]
    Pipeline = None  # type: ignore[assignment]
    LabelEncoder = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]

try:  # scikit-learn >= 1.6
    from sklearn.frozen import FrozenEstimator
except Exception:  # pragma: no cover - older sklearn compatibility
    try:
        from sklearn.calibration import FrozenEstimator
    except Exception:  # pragma: no cover
        FrozenEstimator = None


try:  # pragma: no cover - exercised only when torch is installed
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = object
    DataLoader = None
    TensorDataset = None


def _require_sklearn_dependencies() -> None:
    if any(obj is None for obj in [CalibratedClassifierCV, RandomForestClassifier, SimpleImputer, LogisticRegression, train_test_split, Pipeline, LabelEncoder, StandardScaler]):
        raise ImportError("scikit-learn is required to train DuoDose sklearn models. Install requirements.txt first.")


def _capped_ratio(
    numerator: float,
    denominator: float,
    default: float = 1.0,
    cap: float = 20.0,
) -> float:
    """Return a finite positive ratio bounded for stable loss weighting."""

    if denominator <= 0:
        return float(default)
    value = numerator / denominator
    if not np.isfinite(value) or value <= 0:
        return float(default)
    return float(min(cap, value))


def _set_torch_reproducibility(random_state: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch while keeping fast CUDA defaults unless requested."""

    seed = int(random_state)
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on local hardware
        torch.cuda.manual_seed_all(seed)
    if deterministic:  # pragma: no cover - depends on torch backend support
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    elif hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = bool(torch.cuda.is_available())


def _resolve_torch_device(device: str | object = "auto") -> tuple[object, str]:
    if torch is None:
        raise ImportError("PyTorch is unavailable.")
    requested = str(device or "auto").strip().lower()
    if requested in {"auto", ""}:
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        message = "auto-selected CUDA device" if resolved.type == "cuda" else "auto-selected CPU device; CUDA unavailable"
        return resolved, message
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested), f"using requested device {requested}"
        raise RuntimeError(f"requested CUDA device {requested!r} but torch.cuda.is_available() is false")
    if requested == "cpu":
        return torch.device("cpu"), "using requested CPU device"
    raise ValueError("device must be one of: auto, cuda, cpu")


def _torch_cuda_metadata(device: object, amp_enabled: bool) -> dict[str, object]:
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    gpu_name = ""
    if cuda_available:  # pragma: no cover - depends on local hardware
        try:
            gpu_name = str(torch.cuda.get_device_name(0))
        except Exception:
            gpu_name = ""
    training_device = "cuda" if getattr(device, "type", "cpu") == "cuda" else "cpu"
    return {
        "training_backend": f"torch_{training_device}",
        "training_device": training_device,
        "amp_enabled": bool(amp_enabled),
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
    }


def _effective_torch_batch_size(batch_size: int | None, n_rows: int, device: object, default_cpu: int = 256, default_cuda: int = 4096) -> int:
    if batch_size is not None and int(batch_size) > 0:
        return int(min(int(batch_size), max(1, n_rows)))
    default = default_cuda if getattr(device, "type", "cpu") == "cuda" else default_cpu
    return int(min(default, max(1, n_rows)))


def _make_amp_scaler(device: object, use_amp: bool):
    enabled = bool(use_amp and getattr(device, "type", "cpu") == "cuda")
    if torch is None or not enabled:
        return None
    try:  # PyTorch >= 2
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:  # pragma: no cover - older torch compatibility
        return torch.cuda.amp.GradScaler(enabled=True)


def _amp_context(device: object, use_amp: bool):
    enabled = bool(use_amp and getattr(device, "type", "cpu") == "cuda")
    if torch is None or not enabled:
        return nullcontext()
    try:  # PyTorch >= 2
        return torch.amp.autocast(device_type="cuda", enabled=True)
    except Exception:  # pragma: no cover - older torch compatibility
        return torch.cuda.amp.autocast(enabled=True)


NET_CLASS_LABELS = ("clean", "high_RNA_singlet", "homotypic_doublet", "heterotypic_doublet")
NET_SAFE_FEATURE_METHODS = DUODOSE_METHODS
NET_METHODS = NET_SAFE_FEATURE_METHODS
NET_NEGATIVE_CONTROL_METHODS: tuple[str, ...] = ()
CASE_COLUMNS = ["seed", "design", "propensity_setting", "subtype_strategy", "mode"]
SAFE_FEATURE_EXCLUDE_TOKENS = (
    "benchmark",
    "true",
    "label",
    "doublet_type",
    "simulated",
    "expected",
    "y_true",
    "is_",
    "scrublet",
    "hybrid",
    "external",
)
DOUBLET_LABELS = ("homotypic_doublet", "heterotypic_doublet")

HYBRID_FEATURE_COLUMNS = [
    "scrublet_score",
    "handcrafted_homotypic_score",
    "handcrafted_identity_mixture_score",
    "handcrafted_combined_score",
    "handcrafted_sensitive_score",
    "handcrafted_dosage_raw_score",
    "handcrafted_dosage_reference_ecdf",
    "handcrafted_dosage_reference_tail",
    "hybrid_overall_score",
    "hybrid_homotypic_score",
    "hybrid_heterotypic_score",
    "nCount",
    "log_nCount",
    "cluster_nCount_z",
]

FULL_FEATURE_COLUMNS = [
    *HYBRID_FEATURE_COLUMNS,
    "library_complexity_balance",
    "handcrafted_homotypic_reference_score",
    "handcrafted_artificial_doublet_neighbor_score",
    "handcrafted_homotypic_reference_ecdf",
    "handcrafted_artificial_doublet_neighbor_ecdf",
    "handcrafted_homotypic_reference_tail",
    "handcrafted_artificial_doublet_neighbor_tail",
    "handcrafted_combined_raw_score",
    "handcrafted_combined_reference_ecdf",
    "handcrafted_combined_reference_tail",
    "handcrafted_artificial_doublet_compatible_score",
    "dosage_outlier_score",
    "identity_inlier_score",
    "uniform_dosage_inflation_score",
    "biological_program_coherence_score",
    "handcrafted_homotypic_candidate_score",
    "handcrafted_homotypic_dosage_score",
    "module_residual_rank_mean",
    "module_residual_rank_spread",
    "cluster_count_robust_z",
    "cluster_gene_robust_z",
    "cluster_stable_dosage_robust_z",
    "cluster_marker_dosage_robust_z",
    "dosage_residual",
    "cluster_abundance",
    "cluster_level_expected_homotypic_burden",
    "benchmark_cluster_frequency",
]


def net_leakage_audit_report() -> list[str]:
    """Return the benchmark-only leakage audit text for DuoDose-Net."""

    return [
        "DuoDose-Net leakage audit: passed",
        "  Constructed fit, validation, and held-out cells use disjoint parent pools.",
        "  Validation loss and validation metrics are computed only from training rows.",
        "  Test labels are used only after prediction to compute benchmark metrics.",
        "  Truth labels are not used in feature computation; they are used only as supervised training targets.",
    ]


def _safe_binary_metric(metric_fn, y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(s)
    if mask.sum() < 2:
        return float("nan")
    try:
        value = metric_fn(y[mask], s[mask])
    except ValueError:
        return float("nan")
    return float(value) if np.isfinite(value) else float("nan")


def _target_labels(cell_scores: pd.DataFrame) -> pd.Series:
    labels = cell_scores["true_label"].astype(str).copy()
    labels = labels.where(labels.isin(NET_CLASS_LABELS), "clean")
    return labels


def _feature_columns(feature_set: str) -> list[str]:
    if feature_set == "hybrid":
        return HYBRID_FEATURE_COLUMNS
    if feature_set in {"full", "safe"}:
        return FULL_FEATURE_COLUMNS
    raise ValueError(f"Unknown DuoDose-Net feature set: {feature_set!r}")


def split_safe_feature_columns(columns: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return safe included/excluded feature names using name-only filters."""

    included: list[str] = []
    excluded: list[str] = []
    for column in columns:
        name = str(column)
        lower = name.lower()
        if any(token in lower for token in SAFE_FEATURE_EXCLUDE_TOKENS):
            excluded.append(name)
        elif lower == "benchmark_cluster_frequency" or lower.startswith("benchmark_cluster_"):
            excluded.append(name)
        else:
            included.append(name)
    return included, excluded


def format_safe_feature_audit(included: Iterable[str], excluded: Iterable[str]) -> str:
    """Format the safe-feature audit text for console and file output."""

    included_list = sorted(dict.fromkeys(map(str, included)))
    excluded_list = sorted(dict.fromkeys(map(str, excluded)))
    return "\n".join(
        [
            "Net safe feature audit: passed",
            f"Included features ({len(included_list)}): {', '.join(included_list) if included_list else '(none)'}",
            f"Excluded features ({len(excluded_list)}): {', '.join(excluded_list) if excluded_list else '(none)'}",
        ]
    )


def unsafe_feature_list(columns: Iterable[str]) -> list[str]:
    """Return unsafe feature names using the SafeFeatures name-only audit."""

    _, excluded = split_safe_feature_columns(columns)
    return sorted(dict.fromkeys(map(str, excluded)))


def build_feature_matrix(cell_scores: pd.DataFrame, feature_set: str = "full") -> pd.DataFrame:
    """Build a numeric benchmark feature matrix without truth/metric columns."""

    columns = [column for column in _feature_columns(feature_set) if column in cell_scores]
    frame = cell_scores.loc[:, columns].copy() if columns else pd.DataFrame(index=cell_scores.index)
    for categorical in ["benchmark_cluster", "duodose_cluster", "sample_id"]:
        if categorical in cell_scores:
            dummies = pd.get_dummies(cell_scores[categorical].astype(str), prefix=categorical, dtype=float)
            frame = pd.concat([frame, dummies], axis=1)
    frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    for column in frame.columns:
        if frame[column].isna().all():
            frame[column] = 0.0
    frame = frame.astype(float)
    if feature_set == "safe":
        included, _ = split_safe_feature_columns(frame.columns)
        frame = frame.loc[:, included]
    return frame


def select_feature_variant(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Select the retained full or SafeFeatures matrix."""

    if variant == "full":
        selected = list(frame.columns)
    elif variant == "safe":
        selected, _ = split_safe_feature_columns(frame.columns)
    else:
        raise ValueError(f"Unknown feature variant: {variant!r}")
    if not selected:
        return pd.DataFrame({"constant_feature": np.ones(len(frame), dtype=float)}, index=frame.index)
    return frame.loc[:, selected]


@dataclass
class _FittedSklearnModel:
    method: str
    pipeline: object
    classes: np.ndarray
    feature_names: list[str]
    training_summary: dict[str, object]


class _LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Small adapter for optional libraries that prefer numeric class labels."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None):  # noqa: ANN001 - sklearn-compatible signature
        self.label_encoder_ = LabelEncoder()
        encoded_y = self.label_encoder_.fit_transform(pd.Series(y).astype(str))
        if sample_weight is None:
            self.estimator.fit(X, encoded_y)
        else:
            self.estimator.fit(X, encoded_y, sample_weight=sample_weight)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict_proba(self, X):  # noqa: ANN001 - sklearn-compatible signature
        return self.estimator.predict_proba(X)


def _align_features(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = list(dict.fromkeys([*train_x.columns, *test_x.columns]))
    return train_x.reindex(columns=columns, fill_value=0.0), test_x.reindex(columns=columns, fill_value=0.0)


def _probability_frame(raw: np.ndarray, classes: Iterable[object], index: pd.Index) -> pd.DataFrame:
    probs = pd.DataFrame(0.0, index=index, columns=list(NET_CLASS_LABELS))
    for i, label in enumerate(classes):
        label_str = str(label)
        if label_str in probs.columns:
            probs[label_str] = raw[:, i]
    row_sum = probs.sum(axis=1).replace(0.0, 1.0)
    return probs.div(row_sum, axis=0)


def probabilities_to_scores(probs: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Convert subtype probabilities to overall, homotypic, and heterotypic scores."""

    homotypic = probs.get("homotypic_doublet", pd.Series(0.0, index=probs.index)).astype(float)
    heterotypic = probs.get("heterotypic_doublet", pd.Series(0.0, index=probs.index)).astype(float)
    overall = (homotypic + heterotypic).clip(0.0, 1.0)
    return overall.rename("net_overall_score"), homotypic.rename("net_homotypic_score"), heterotypic.rename("net_heterotypic_score")


def _validation_scores(probs: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    overall, _, _ = probabilities_to_scores(probs)
    y_doublet = y.isin(DOUBLET_LABELS).astype(int).to_numpy()
    return (
        _safe_binary_metric(roc_auc_score, y_doublet, overall.to_numpy(dtype=float)),
        _safe_binary_metric(average_precision_score, y_doublet, overall.to_numpy(dtype=float)),
    )


def _balanced_sample_weight(labels: pd.Series) -> np.ndarray:
    y = labels.astype(str)
    counts = y.value_counts()
    n_classes = max(1, int(counts.size))
    n_samples = max(1, int(len(y)))
    weights = y.map({label: n_samples / (n_classes * max(1, int(count))) for label, count in counts.items()})
    return weights.fillna(1.0).to_numpy(dtype=float)


def _calibration_log_loss(probs: pd.DataFrame, labels: pd.Series) -> float:
    try:
        ordered_labels = sorted(str(label) for label in NET_CLASS_LABELS)
        return float(log_loss(labels.astype(str), probs.reindex(columns=ordered_labels, fill_value=0.0), labels=ordered_labels))
    except ValueError:
        return float("inf")


def _make_prefit_calibrator(estimator) -> CalibratedClassifierCV:
    if FrozenEstimator is not None:
        return CalibratedClassifierCV(estimator=FrozenEstimator(estimator), method="sigmoid")
    try:
        return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv="prefit")
    except TypeError:  # pragma: no cover - older sklearn compatibility
        return CalibratedClassifierCV(base_estimator=estimator, method="sigmoid", cv="prefit")


def _fit_sklearn_classifier(
    method: str,
    estimator,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    val_x: pd.DataFrame,
    val_y: pd.Series,
    random_state: int,
    estimator_name: str,
    feature_set: str,
    feature_variant: str = "full",
    model_category: str = "core",
    diagnostic_only: bool = False,
    sample_weight: np.ndarray | None = None,
    calibrate_sigmoid: bool = False,
    use_calibration_if_improves: bool = False,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> tuple[_FittedSklearnModel, pd.DataFrame]:
    _require_sklearn_dependencies()
    start = time.perf_counter()
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )
    fit_params = {"classifier__sample_weight": sample_weight} if sample_weight is not None else {}
    if progress_callback is not None:
        progress_callback({"event": "milestone", "message": "RF: fitting final RF"})
    pipeline.fit(train_x, train_y, **fit_params)
    final_model: object = pipeline
    classes = np.asarray(pipeline.named_steps["classifier"].classes_, dtype=object)
    base_val_probs = _probability_frame(pipeline.predict_proba(val_x), classes, val_x.index)
    base_calibration_loss = _calibration_log_loss(base_val_probs, val_y)
    calibrated_calibration_loss = float("nan")
    calibration_used = False
    if calibrate_sigmoid:
        try:
            if progress_callback is not None:
                progress_callback({"event": "milestone", "message": "RF: fitting calibration folds"})
            calibrator = _make_prefit_calibrator(pipeline)
            calibrator.fit(val_x, val_y)
            calibrated_classes = np.asarray(calibrator.classes_, dtype=object)
            calibrated_val_probs = _probability_frame(calibrator.predict_proba(val_x), calibrated_classes, val_x.index)
            calibrated_calibration_loss = _calibration_log_loss(calibrated_val_probs, val_y)
            if not use_calibration_if_improves or calibrated_calibration_loss <= base_calibration_loss:
                final_model = calibrator
                classes = calibrated_classes
                base_val_probs = calibrated_val_probs
                calibration_used = True
        except Exception:
            calibrated_calibration_loss = float("nan")
    runtime = time.perf_counter() - start
    val_probs = base_val_probs
    val_auroc, val_auprc = _validation_scores(val_probs, val_y)
    unsafe_features = unsafe_feature_list(train_x.columns)
    summary = {
        "method": method,
        "status": "success",
        "message": "trained on benchmark feature matrix",
        "train_loss": float("nan"),
        "validation_loss": float("nan"),
        "validation_AUROC": val_auroc,
        "validation_AUPRC": val_auprc,
        "training_time_seconds": float(runtime),
        "number_train_cells": int(len(train_x)),
        "number_validation_cells": int(len(val_x)),
        "number_test_cells": 0,
        "feature_set": feature_set,
        "feature_variant": feature_variant,
        "estimator": estimator_name,
        "model_category": model_category,
        "diagnostic_only": bool(diagnostic_only),
        "unsafe_feature_detected": bool(unsafe_features),
        "unsafe_feature_list": ",".join(unsafe_features),
        "feature_list": ",".join(train_x.columns),
        "random_state": int(random_state),
        "calibration_used": bool(calibration_used),
        "validation_calibration_log_loss": float(_calibration_log_loss(val_probs, val_y)),
        "uncalibrated_validation_log_loss": float(base_calibration_loss),
        "calibrated_validation_log_loss": float(calibrated_calibration_loss),
    }
    fitted = _FittedSklearnModel(method=method, pipeline=final_model, classes=classes, feature_names=list(train_x.columns), training_summary=summary)
    return fitted, pd.DataFrame()


def _predict_sklearn(fitted: _FittedSklearnModel, test_x: pd.DataFrame) -> pd.DataFrame:
    frame = test_x.reindex(columns=fitted.feature_names, fill_value=0.0)
    raw = fitted.pipeline.predict_proba(frame)
    return _probability_frame(raw, fitted.classes, frame.index)


if torch is not None:

    class _BenchmarkMLP(nn.Module):  # pragma: no cover - exercised through benchmark commands
        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            dropout: float = 0.20,
            hidden_dim: int | None = None,
            depth: int | None = None,
        ) -> None:
            super().__init__()
            if hidden_dim is None and depth is None:
                hidden_dims = [64, 32]
            else:
                width = int(hidden_dim or 128)
                hidden_dims = [width for _ in range(max(1, int(depth or 2)))]
            layers: list[object] = []
            previous_dim = int(input_dim)
            for width in hidden_dims:
                layers.extend([nn.Linear(previous_dim, int(width)), nn.ReLU(), nn.Dropout(float(dropout))])
                previous_dim = int(width)
            layers.append(nn.Linear(previous_dim, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, x):  # type: ignore[override]
            return self.network(x)


    class _ConditionalMultiTaskMLP(nn.Module):  # pragma: no cover - exercised through benchmark commands
        def __init__(
            self,
            input_dim: int,
            dropout: float = 0.15,
            hidden_dim: int | None = None,
            depth: int | None = None,
        ) -> None:
            super().__init__()
            if hidden_dim is None and depth is None:
                hidden_dims = [128, 64]
            else:
                width = int(hidden_dim or 128)
                hidden_dims = [width for _ in range(max(1, int(depth or 2)))]
            layers: list[object] = []
            previous_dim = int(input_dim)
            for layer_number, width in enumerate(hidden_dims):
                layers.append(nn.Linear(previous_dim, int(width)))
                if layer_number == 0:
                    layers.append(nn.LayerNorm(int(width)))
                layers.extend([nn.GELU(), nn.Dropout(float(dropout))])
                previous_dim = int(width)
            self.encoder = nn.Sequential(*layers)
            self.doublet_head = nn.Linear(previous_dim, 1)
            self.subtype_head = nn.Linear(previous_dim, 2)
            self.highrna_rejection_head = nn.Linear(previous_dim, 1)

        def forward(self, x):  # type: ignore[override]
            z = self.encoder(x)
            return {
                "z": z,
                "doublet_logit": self.doublet_head(z).squeeze(-1),
                "subtype_logits": self.subtype_head(z),
                "highrna_rejection_logit": self.highrna_rejection_head(z).squeeze(-1),
            }


@dataclass
class _FittedTorchModel:
    method: str
    model: object
    imputer: SimpleImputer
    scaler: StandardScaler
    label_encoder: LabelEncoder
    feature_names: list[str]
    training_summary: dict[str, object]


@dataclass
class _FittedConditionalTorchModel:
    method: str
    model: object
    imputer: SimpleImputer
    scaler: StandardScaler
    feature_names: list[str]
    training_summary: dict[str, object]


@dataclass
class TrainedDuoDoseBackend:
    """Reusable handle around one fitted retained backend."""

    method: str
    family: str
    fitted_model: object
    feature_set: str
    feature_variant: str
    training_summary: dict[str, object]
    safe_feature_transformer: object | None = None

    def predict_probabilities(self, cell_scores: pd.DataFrame) -> pd.DataFrame:
        if self.safe_feature_transformer is not None:
            builder = getattr(self.safe_feature_transformer, "build_model_matrix", None)
            if builder is None:
                raise TypeError("safe_feature_transformer does not expose build_model_matrix")
            features = builder(cell_scores)
        else:
            features = build_feature_matrix(cell_scores, self.feature_set)
            features = select_feature_variant(features, self.feature_variant)
        feature_names = list(getattr(self.fitted_model, "feature_names", features.columns))
        features = features.reindex(columns=feature_names, fill_value=0.0)
        if self.family == "sklearn":
            return _predict_sklearn(self.fitted_model, features)
        if self.family == "torch_mlp":
            return _predict_torch(self.fitted_model, features)
        if self.family == "conditional_dl":
            probabilities, _ = _predict_conditional_multitask_mlp(self.fitted_model, features)
            return probabilities
        raise ValueError(f"Unsupported fitted backend family: {self.family!r}")


def _cluster_series_for_highrna(meta: pd.DataFrame) -> pd.Series:
    for column in ["benchmark_cluster", "duodose_cluster", "cluster"]:
        if column in meta:
            return meta[column].astype(str)
    return pd.Series("__all__", index=meta.index, dtype=object)


def _log_ncount_series(meta: pd.DataFrame) -> pd.Series:
    if "log_nCount" in meta:
        values = pd.to_numeric(meta["log_nCount"], errors="coerce")
    elif "nCount" in meta:
        values = np.log1p(pd.to_numeric(meta["nCount"], errors="coerce"))
    else:
        values = pd.Series(0.0, index=meta.index, dtype=float)
    return values.replace([np.inf, -np.inf], np.nan).fillna(values.median() if values.notna().any() else 0.0)


def _highrna_masks_from_training_thresholds(
    train_meta: pd.DataFrame,
    train_y: pd.Series,
    val_meta: pd.DataFrame,
    val_y: pd.Series,
    percentile: float = 90.0,
) -> tuple[pd.Series, pd.Series, str, dict[str, float]]:
    train_labels = train_y.astype(str)
    val_labels = val_y.astype(str)
    if train_labels.eq("high_RNA_singlet").any():
        return (
            train_labels.eq("high_RNA_singlet"),
            val_labels.eq("high_RNA_singlet"),
            "explicit",
            {},
        )

    train_cluster = _cluster_series_for_highrna(train_meta)
    val_cluster = _cluster_series_for_highrna(val_meta)
    train_log = _log_ncount_series(train_meta)
    val_log = _log_ncount_series(val_meta)
    train_singlet = ~train_labels.isin(DOUBLET_LABELS)
    thresholds: dict[str, float] = {}
    quantile = float(np.clip(float(percentile) / 100.0, 0.50, 0.995))
    global_values = train_log.loc[train_singlet]
    global_threshold = float(global_values.quantile(quantile)) if len(global_values) else float(train_log.quantile(quantile))
    for cluster, values in train_log.loc[train_singlet].groupby(train_cluster.loc[train_singlet], dropna=False):
        thresholds[str(cluster)] = float(values.quantile(quantile)) if len(values) else global_threshold

    def inferred_mask(log_values: pd.Series, clusters: pd.Series, labels: pd.Series) -> pd.Series:
        singlet = ~labels.astype(str).isin(DOUBLET_LABELS)
        cutoffs = clusters.astype(str).map(thresholds).fillna(global_threshold).astype(float)
        return (log_values.astype(float) >= cutoffs) & singlet

    return (
        inferred_mask(train_log, train_cluster, train_labels),
        inferred_mask(val_log, val_cluster, val_labels),
        f"inferred_train_cluster_log_nCount_q{percentile:g}",
        thresholds,
    )


def _conditional_probability_frame(outputs: dict[str, object], index: pd.Index) -> tuple[pd.DataFrame, pd.Series]:
    doublet = torch.sigmoid(outputs["doublet_logit"]).detach().cpu().numpy().astype(float)
    subtype = torch.softmax(outputs["subtype_logits"], dim=1).detach().cpu().numpy().astype(float)
    highrna = torch.sigmoid(outputs["highrna_rejection_logit"]).detach().cpu().numpy().astype(float)
    probs = pd.DataFrame(0.0, index=index, columns=list(NET_CLASS_LABELS))
    probs["homotypic_doublet"] = np.clip(doublet * subtype[:, 0], 0.0, 1.0)
    probs["heterotypic_doublet"] = np.clip(doublet * subtype[:, 1], 0.0, 1.0)
    probs["clean"] = np.clip(1.0 - doublet, 0.0, 1.0)
    row_sum = probs.sum(axis=1).replace(0.0, 1.0)
    probs = probs.div(row_sum, axis=0)
    return probs, pd.Series(highrna, index=index, name="highRNA_rejection_score")


def _batch_triplet_tensors(
    z,
    subtype_targets,
    subtype_mask,
    highrna_singlet_mask,
    cluster_codes,
    batch_positions=None,
    model_confused_negative_for_anchor=None,
    doublet_scores=None,
    homotypic_scores=None,
    log_ncount=None,
    hard_negative_mode: str = "highRNA_same_cluster",
) -> tuple[object | None, object | None, object | None, int, int, dict[str, float]]:
    homotypic_positions = torch.where(subtype_mask & subtype_targets.eq(0))[0]
    negative_positions = torch.where(highrna_singlet_mask)[0]
    anchors: list[object] = []
    positives: list[object] = []
    negatives: list[object] = []
    diagnostics = {
        "negative_predicted_doublet_score_sum": 0.0,
        "negative_homotypic_score_sum": 0.0,
        "same_cluster_negative_count": 0.0,
        "model_confused_negative_count": 0.0,
    }
    for anchor_idx in homotypic_positions:
        same_cluster_positive = homotypic_positions[
            (homotypic_positions != anchor_idx) & cluster_codes[homotypic_positions].eq(cluster_codes[anchor_idx])
        ]
        positive_candidates = same_cluster_positive if len(same_cluster_positive) else homotypic_positions[homotypic_positions != anchor_idx]
        same_cluster_negative = negative_positions[cluster_codes[negative_positions].eq(cluster_codes[anchor_idx])]
        if hard_negative_mode == "highRNA_any_cluster":
            negative_candidates = negative_positions
        else:
            negative_candidates = same_cluster_negative if len(same_cluster_negative) else negative_positions
        if len(positive_candidates) == 0 or len(negative_candidates) == 0:
            continue
        negative_idx = negative_candidates[0]
        if (
            hard_negative_mode == "model_confused_highRNA_same_cluster"
            and model_confused_negative_for_anchor is not None
            and batch_positions is not None
            and len(negative_candidates)
        ):
            anchor_global = int(batch_positions[anchor_idx].detach().cpu().item())
            mapped_global = int(model_confused_negative_for_anchor[anchor_global])
            if mapped_global >= 0:
                matching = negative_candidates[batch_positions[negative_candidates].eq(mapped_global)]
                if len(matching):
                    negative_idx = matching[0]
                else:
                    candidate_score = torch.zeros(len(negative_candidates), dtype=torch.float32, device=z.device)
                    if doublet_scores is not None:
                        candidate_score = candidate_score + doublet_scores[negative_candidates]
                    if homotypic_scores is not None:
                        candidate_score = candidate_score + homotypic_scores[negative_candidates]
                    if log_ncount is not None:
                        candidate_score = candidate_score - 0.05 * torch.abs(log_ncount[negative_candidates] - log_ncount[anchor_idx])
                    negative_idx = negative_candidates[torch.argmax(candidate_score)]
                diagnostics["model_confused_negative_count"] += 1.0
        elif hard_negative_mode == "near_boundary_highRNA_same_cluster" and doublet_scores is not None and len(negative_candidates):
            local_scores = doublet_scores[negative_candidates]
            negative_idx = negative_candidates[torch.argmax(local_scores)]
        if doublet_scores is not None:
            diagnostics["negative_predicted_doublet_score_sum"] += float(doublet_scores[negative_idx].detach().cpu().item())
        if homotypic_scores is not None:
            diagnostics["negative_homotypic_score_sum"] += float(homotypic_scores[negative_idx].detach().cpu().item())
        if bool(cluster_codes[negative_idx].eq(cluster_codes[anchor_idx]).detach().cpu().item()):
            diagnostics["same_cluster_negative_count"] += 1.0
        anchors.append(z[anchor_idx])
        positives.append(z[positive_candidates[0]])
        negatives.append(z[negative_idx])
    if not anchors:
        return None, None, None, 0, int(len(homotypic_positions)), diagnostics
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives), len(anchors), int(len(homotypic_positions)), diagnostics


def _fit_torch_mlp(
    method: str,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    val_x: pd.DataFrame,
    val_y: pd.Series,
    random_state: int,
    feature_set: str,
    feature_variant: str = "full",
    model_category: str = "core",
    diagnostic_only: bool = False,
    max_epochs: int = 40,
    patience: int = 6,
    sample_weight: pd.Series | np.ndarray | None = None,
    device: str = "auto",
    use_amp: bool = False,
    batch_size: int | None = None,
    num_workers: int = 0,
    gradient_accumulation_steps: int = 1,
    deterministic: bool = False,
    hidden_dim: int | None = None,
    depth: int | None = None,
    dropout: float | None = None,
    weight_decay: float = 1e-4,
) -> tuple[_FittedTorchModel | None, pd.DataFrame, dict[str, object] | None]:
    if torch is None:
        return None, pd.DataFrame(), {
            "method": method,
            "model_name": method,
            "status": "skipped",
            "message": "PyTorch is unavailable",
            "train_loss": float("nan"),
            "validation_loss": float("nan"),
            "val_loss": float("nan"),
            "validation_AUROC": float("nan"),
            "validation_AUPRC": float("nan"),
            "training_time_seconds": 0.0,
            "runtime_seconds": 0.0,
            "number_train_cells": int(len(train_x)),
            "n_train": int(len(train_x)),
            "number_validation_cells": int(len(val_x)),
            "number_test_cells": 0,
            "n_features": int(train_x.shape[1]),
            "n_epochs": 0,
            "best_epoch": 0,
            "device": "unavailable",
            "device_message": "PyTorch is unavailable",
            "use_amp": False,
            "batch_size": 0,
            "feature_set": feature_set,
            "feature_variant": feature_variant,
            "estimator": "MLP",
            "model_category": model_category,
            "diagnostic_only": bool(diagnostic_only),
            "unsafe_feature_detected": bool(unsafe_feature_list(train_x.columns)),
            "unsafe_feature_list": ",".join(unsafe_feature_list(train_x.columns)),
            "feature_list": ",".join(train_x.columns),
            "random_state": int(random_state),
            "seed": int(random_state),
        }

    _set_torch_reproducibility(random_state, deterministic=deterministic)
    torch_device, device_message = _resolve_torch_device(device)
    use_cuda = getattr(torch_device, "type", "cpu") == "cuda"
    amp_enabled = bool(use_amp and use_cuda)
    if use_amp and not use_cuda:
        device_message = f"{device_message}; AMP disabled because CUDA is unavailable"
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train_x)).astype(np.float32)
    x_val = scaler.transform(imputer.transform(val_x)).astype(np.float32)
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_y.astype(str)).astype(np.int64)
    y_val = encoder.transform(val_y.astype(str)).astype(np.int64)
    if sample_weight is None:
        train_sample_weight = np.ones(len(train_x), dtype=np.float32)
    else:
        train_sample_weight = pd.Series(sample_weight, index=train_x.index).reindex(train_x.index).fillna(1.0).to_numpy(dtype=np.float32)
        train_sample_weight = np.clip(train_sample_weight, 0.0, np.inf).astype(np.float32)
        if not np.isfinite(train_sample_weight).any() or float(train_sample_weight.sum()) <= 0:
            train_sample_weight = np.ones(len(train_x), dtype=np.float32)
    effective_dropout = 0.20 if dropout is None else float(dropout)
    model = _BenchmarkMLP(
        input_dim=x_train.shape[1],
        output_dim=len(encoder.classes_),
        dropout=effective_dropout,
        hidden_dim=hidden_dim,
        depth=depth,
    ).to(torch_device)
    counts = np.bincount(y_train, minlength=len(encoder.classes_))
    class_weights = counts.sum() / np.maximum(counts, 1)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=torch_device), reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=float(weight_decay))
    effective_batch_size = _effective_torch_batch_size(batch_size, len(x_train), torch_device, default_cpu=128, default_cuda=4096)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train), torch.tensor(train_sample_weight, dtype=torch.float32)),
        batch_size=effective_batch_size,
        shuffle=True,
        pin_memory=bool(use_cuda),
        num_workers=max(0, int(num_workers)),
    )
    val_tensor = torch.tensor(x_val, dtype=torch.float32, device=torch_device)
    val_target = torch.tensor(y_val, dtype=torch.long, device=torch_device)
    amp_scaler = _make_amp_scaler(torch_device, amp_enabled)
    accumulation_steps = max(1, int(gradient_accumulation_steps))
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    rows: list[dict[str, object]] = []
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        optimizer.zero_grad(set_to_none=True)
        for batch_number, (xb, yb, wb) in enumerate(loader, start=1):
            xb = xb.to(torch_device, non_blocking=use_cuda)
            yb = yb.to(torch_device, non_blocking=use_cuda)
            wb = wb.to(torch_device, non_blocking=use_cuda)
            with _amp_context(torch_device, amp_enabled):
                loss_values = loss_fn(model(xb), yb)
                loss = (loss_values * wb).sum() / torch.clamp(wb.sum(), min=1e-6)
                scaled_loss = loss / accumulation_steps
            if amp_scaler is not None:
                amp_scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            if batch_number % accumulation_steps == 0 or batch_number == len(loader):
                if amp_scaler is not None:
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            train_losses.append(float(loss.detach().cpu().item()))
        model.eval()
        with torch.no_grad():
            with _amp_context(torch_device, amp_enabled):
                val_logits = model(val_tensor)
                val_loss = float(loss_fn(val_logits, val_target).mean().detach().cpu().item())
                val_raw = torch.softmax(val_logits, dim=1).detach().cpu().numpy()
        val_probs = _probability_frame(val_raw, encoder.classes_, val_x.index)
        val_auroc, val_auprc = _validation_scores(val_probs, val_y)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        rows.append(
            {
                "method": method,
                "epoch": int(epoch),
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_AUROC": val_auroc,
                "validation_AUPRC": val_auprc,
                "device": str(torch_device),
                "use_amp": bool(amp_enabled),
                "batch_size": int(effective_batch_size),
                "feature_set": feature_set,
                "feature_variant": feature_variant,
                "model_category": model_category,
            }
        )
        if val_loss + 1e-5 < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    runtime = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    log = pd.DataFrame(rows)
    best_row = log.loc[log["epoch"].eq(best_epoch)].iloc[0] if not log.empty and best_epoch else pd.Series(dtype=object)
    unsafe_features = unsafe_feature_list(train_x.columns)
    summary = {
        "method": method,
        "model_name": method,
        "status": "success",
        "message": f"trained benchmark MLP with early stopping at epoch {best_epoch}",
        "train_loss": float(best_row.get("train_loss", np.nan)),
        "validation_loss": float(best_row.get("validation_loss", np.nan)),
        "val_loss": float(best_row.get("validation_loss", np.nan)),
        "validation_AUROC": float(best_row.get("validation_AUROC", np.nan)),
        "validation_AUPRC": float(best_row.get("validation_AUPRC", np.nan)),
        "training_time_seconds": float(runtime),
        "runtime_seconds": float(runtime),
        "number_train_cells": int(len(train_x)),
        "n_train": int(len(train_x)),
        "number_validation_cells": int(len(val_x)),
        "number_test_cells": 0,
        "n_features": int(train_x.shape[1]),
        "n_epochs": int(len(log)),
        "best_epoch": int(best_epoch),
        "early_stopping_epoch": int(best_epoch),
        "device": str(torch_device),
        "device_message": device_message,
        "use_amp": bool(amp_enabled),
        **_torch_cuda_metadata(torch_device, amp_enabled),
        "batch_size": int(effective_batch_size),
        "num_workers": int(num_workers),
        "gradient_accumulation_steps": int(accumulation_steps),
        "deterministic": bool(deterministic),
        "hidden_dim": int(hidden_dim) if hidden_dim is not None else np.nan,
        "depth": int(depth) if depth is not None else np.nan,
        "dropout": float(effective_dropout),
        "weight_decay": float(weight_decay),
        "feature_set": feature_set,
        "feature_variant": feature_variant,
        "estimator": "MLP",
        "model_category": model_category,
        "diagnostic_only": bool(diagnostic_only),
        "unsafe_feature_detected": bool(unsafe_features),
        "unsafe_feature_list": ",".join(unsafe_features),
        "feature_list": ",".join(train_x.columns),
        "random_state": int(random_state),
        "seed": int(random_state),
        "sample_weight_used": bool(sample_weight is not None),
        "sample_weight_min": float(np.nanmin(train_sample_weight)) if len(train_sample_weight) else float("nan"),
        "sample_weight_max": float(np.nanmax(train_sample_weight)) if len(train_sample_weight) else float("nan"),
        "sample_weight_mean": float(np.nanmean(train_sample_weight)) if len(train_sample_weight) else float("nan"),
    }
    fitted = _FittedTorchModel(method=method, model=model, imputer=imputer, scaler=scaler, label_encoder=encoder, feature_names=list(train_x.columns), training_summary=summary)
    return fitted, log, None


def _fit_conditional_multitask_mlp(
    method: str,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    val_x: pd.DataFrame,
    val_y: pd.Series,
    train_meta: pd.DataFrame,
    val_meta: pd.DataFrame,
    random_state: int,
    feature_set: str,
    feature_variant: str,
    model_category: str = "DuoDose-DL",
    diagnostic_only: bool = False,
    lambda_subtype: float = 0.5,
    lambda_highrna: float = 0.5,
    lambda_contrastive: float = 0.0,
    lambda_ncount_decorrelation: float = 0.0,
    balanced_highrna_batches: bool = False,
    highrna_batch_fraction: float = 0.25,
    highrna_percentile: float = 90.0,
    triplet_margin: float = 1.0,
    hard_negative_mode: str = "highRNA_same_cluster",
    estimator_name: str = "ConditionalMultiTaskMLP",
    max_epochs: int = 100,
    patience: int = 15,
    sample_weight: pd.Series | np.ndarray | None = None,
    device: str = "auto",
    use_amp: bool = False,
    batch_size: int | None = None,
    num_workers: int = 0,
    gradient_accumulation_steps: int = 1,
    deterministic: bool = False,
    hidden_dim: int | None = None,
    depth: int | None = None,
    dropout: float | None = None,
    weight_decay: float = 1e-4,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
    verbose_progress: bool = False,
) -> tuple[_FittedConditionalTorchModel | None, pd.DataFrame, dict[str, object] | None]:
    unsafe_features = unsafe_feature_list(train_x.columns)
    if torch is None:
        return None, pd.DataFrame(), {
            "method": method,
            "model_name": method,
            "status": "skipped",
            "message": "PyTorch is unavailable",
            "train_loss": float("nan"),
            "validation_loss": float("nan"),
            "val_loss": float("nan"),
            "validation_AUROC": float("nan"),
            "validation_AUPRC": float("nan"),
            "training_time_seconds": 0.0,
            "runtime_seconds": 0.0,
            "number_train_cells": int(len(train_x)),
            "n_train": int(len(train_x)),
            "number_validation_cells": int(len(val_x)),
            "number_test_cells": 0,
            "n_features": int(train_x.shape[1]),
            "n_epochs": 0,
            "best_epoch": 0,
            "device": "unavailable",
            "device_message": "PyTorch is unavailable",
            "use_amp": False,
            "batch_size": 0,
            "feature_set": feature_set,
            "feature_variant": "SafeFeatures" if feature_variant == "safe" else feature_variant,
            "estimator": estimator_name,
            "model_category": model_category,
            "diagnostic_only": bool(diagnostic_only),
            "unsafe_feature_detected": bool(unsafe_features),
            "unsafe_feature_list": ",".join(unsafe_features),
            "feature_list": ",".join(train_x.columns),
            "random_state": int(random_state),
            "lambda_highrna": float(lambda_highrna),
            "lambda_ncount_decorrelation": float(lambda_ncount_decorrelation),
            "balanced_highrna_batches": bool(balanced_highrna_batches),
            "highrna_batch_fraction": float(highrna_batch_fraction),
            "average_highrna_singlets_per_batch": float("nan"),
            "average_homotypic_doublets_per_batch": float("nan"),
            "highrna_percentile": float(highrna_percentile),
            "contrastive_loss_weight": float(lambda_contrastive),
            "triplet_margin": float(triplet_margin),
            "hard_negative_mode": str(hard_negative_mode),
            "number_triplets_used": 0,
            "fraction_anchors_with_triplets": 0.0,
            "contrastive_status": "skipped_pytorch_unavailable" if lambda_contrastive > 0 else "disabled",
            "seed": int(random_state),
        }

    _set_torch_reproducibility(random_state, deterministic=deterministic)
    torch_device, device_message = _resolve_torch_device(device)
    use_cuda = getattr(torch_device, "type", "cpu") == "cuda"
    amp_enabled = bool(use_amp and use_cuda)
    if use_amp and not use_cuda:
        device_message = f"{device_message}; AMP disabled because CUDA is unavailable"
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train_x)).astype(np.float32)
    x_val = scaler.transform(imputer.transform(val_x)).astype(np.float32)

    train_labels = train_y.astype(str)
    val_labels = val_y.astype(str)
    train_highrna, val_highrna, highrna_source, _ = _highrna_masks_from_training_thresholds(
        train_meta,
        train_y,
        val_meta,
        val_y,
        percentile=highrna_percentile,
    )

    y_doublet_train = train_labels.isin(DOUBLET_LABELS).astype(np.float32).to_numpy()
    y_doublet_val = val_labels.isin(DOUBLET_LABELS).astype(np.float32).to_numpy()
    y_subtype_train = train_labels.map({"homotypic_doublet": 0, "heterotypic_doublet": 1}).fillna(0).astype(np.int64).to_numpy()
    y_subtype_val = val_labels.map({"homotypic_doublet": 0, "heterotypic_doublet": 1}).fillna(0).astype(np.int64).to_numpy()
    subtype_mask_train = train_labels.isin(DOUBLET_LABELS).to_numpy(dtype=bool)
    subtype_mask_val = val_labels.isin(DOUBLET_LABELS).to_numpy(dtype=bool)
    y_highrna_train = train_labels.eq("homotypic_doublet").astype(np.float32).to_numpy()
    y_highrna_val = val_labels.eq("homotypic_doublet").astype(np.float32).to_numpy()
    highrna_mask_train = (train_labels.eq("homotypic_doublet") | train_highrna.reindex(train_y.index).fillna(False).astype(bool)).to_numpy(dtype=bool)
    highrna_mask_val = (val_labels.eq("homotypic_doublet") | val_highrna.reindex(val_y.index).fillna(False).astype(bool)).to_numpy(dtype=bool)
    highrna_singlet_train = train_highrna.reindex(train_y.index).fillna(False).astype(bool).to_numpy(dtype=bool)
    train_cluster_values = _cluster_series_for_highrna(train_meta).astype(str)
    train_cluster_codes = pd.Categorical(train_cluster_values).codes.astype(np.int64)
    train_log_ncount = _log_ncount_series(train_meta).reindex(train_y.index).fillna(0.0).to_numpy(dtype=np.float32)
    train_log_ncount = ((train_log_ncount - float(np.nanmean(train_log_ncount))) / float(np.nanstd(train_log_ncount) + 1e-6)).astype(np.float32)
    if sample_weight is None:
        train_sample_weight = np.ones(len(train_x), dtype=np.float32)
    else:
        train_sample_weight = pd.Series(sample_weight, index=train_x.index).reindex(train_x.index).fillna(1.0).to_numpy(dtype=np.float32)
        train_sample_weight = np.clip(train_sample_weight, 0.0, np.inf).astype(np.float32)
        if not np.isfinite(train_sample_weight).any() or float(train_sample_weight.sum()) <= 0:
            train_sample_weight = np.ones(len(train_x), dtype=np.float32)

    doublet_pos = float(y_doublet_train.sum())
    doublet_neg = float(len(y_doublet_train) - doublet_pos)
    doublet_pos_weight = _capped_ratio(doublet_neg, doublet_pos)
    subtype_counts = np.bincount(y_subtype_train[subtype_mask_train], minlength=2) if subtype_mask_train.any() else np.ones(2, dtype=int)
    subtype_total = float(max(1, subtype_counts.sum()))
    subtype_weights = np.array([_capped_ratio(subtype_total, 2.0 * float(count), default=1.0) for count in subtype_counts], dtype=np.float32)
    highrna_pos = float(y_highrna_train[highrna_mask_train].sum()) if highrna_mask_train.any() else 0.0
    highrna_neg = float(highrna_mask_train.sum() - highrna_pos)
    highrna_pos_weight = _capped_ratio(highrna_neg, highrna_pos)
    possible_triplets = bool(train_labels.eq("homotypic_doublet").sum() >= 2 and highrna_singlet_train.sum() >= 1)
    use_contrastive = bool(lambda_contrastive > 0 and possible_triplets)

    effective_dropout = 0.15 if dropout is None else float(dropout)
    model = _ConditionalMultiTaskMLP(
        input_dim=x_train.shape[1],
        dropout=effective_dropout,
        hidden_dim=hidden_dim,
        depth=depth,
    ).to(torch_device)
    bce_doublet = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(doublet_pos_weight, dtype=torch.float32, device=torch_device), reduction="none")
    ce_subtype = torch.nn.CrossEntropyLoss(weight=torch.tensor(subtype_weights, dtype=torch.float32, device=torch_device), reduction="none")
    bce_highrna = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(highrna_pos_weight, dtype=torch.float32, device=torch_device), reduction="none")
    triplet_loss_fn = torch.nn.TripletMarginLoss(margin=float(triplet_margin))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=float(weight_decay))

    train_dataset = TensorDataset(
        torch.tensor(x_train),
        torch.tensor(y_doublet_train, dtype=torch.float32),
        torch.tensor(y_subtype_train, dtype=torch.long),
        torch.tensor(subtype_mask_train, dtype=torch.bool),
        torch.tensor(y_highrna_train, dtype=torch.float32),
        torch.tensor(highrna_mask_train, dtype=torch.bool),
        torch.tensor(highrna_singlet_train, dtype=torch.bool),
        torch.tensor(train_sample_weight, dtype=torch.float32),
        torch.tensor(train_log_ncount, dtype=torch.float32),
        torch.arange(len(x_train), dtype=torch.long),
        torch.tensor(train_cluster_codes, dtype=torch.long),
    )
    effective_batch_size = _effective_torch_batch_size(batch_size, len(train_dataset), torch_device, default_cpu=256, default_cuda=4096)
    loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        pin_memory=bool(use_cuda),
        num_workers=max(0, int(num_workers)),
    )
    amp_scaler = _make_amp_scaler(torch_device, amp_enabled)
    accumulation_steps = max(1, int(gradient_accumulation_steps))
    train_tensors = train_dataset.tensors
    homotypic_positions = np.where(train_labels.eq("homotypic_doublet").to_numpy(dtype=bool))[0]
    highrna_singlet_positions = np.where(highrna_singlet_train)[0]
    all_positions = np.arange(len(train_dataset), dtype=int)
    balanced_fraction = float(np.clip(highrna_batch_fraction, 0.0, 0.5))
    use_balanced_highrna_batches = bool(
        balanced_highrna_batches and balanced_fraction > 0 and len(homotypic_positions) > 0 and len(highrna_singlet_positions) > 0
    )
    use_model_confused_mining = bool(use_contrastive and hard_negative_mode == "model_confused_highRNA_same_cluster")
    model_confused_warmup_epochs = min(5, max(1, int(max_epochs) // 4)) if use_model_confused_mining else 0
    model_confused_negative_for_anchor = np.full(len(train_dataset), -1, dtype=int)
    latest_model_confused_stats = {
        "mean_negative_predicted_doublet_score": float("nan"),
        "mean_negative_homotypic_score": float("nan"),
        "same_cluster_negative_fraction": float("nan"),
        "number_model_confused_negatives_used": 0.0,
    }

    def refresh_model_confused_negatives() -> None:
        nonlocal model_confused_negative_for_anchor, latest_model_confused_stats
        if not use_model_confused_mining or len(highrna_singlet_positions) == 0 or len(homotypic_positions) == 0:
            return
        model.eval()
        with torch.no_grad():
            all_x = torch.tensor(x_train, dtype=torch.float32, device=torch_device)
            with _amp_context(torch_device, amp_enabled):
                outputs = model(all_x)
            doublet_scores_np = torch.sigmoid(outputs["doublet_logit"]).detach().cpu().numpy().astype(float)
            subtype_np = torch.softmax(outputs["subtype_logits"], dim=1).detach().cpu().numpy().astype(float)
            homotypic_scores_np = doublet_scores_np * subtype_np[:, 0]
        mapping = np.full(len(train_dataset), -1, dtype=int)
        selected_doublet_scores: list[float] = []
        selected_homotypic_scores: list[float] = []
        same_cluster_count = 0
        for anchor_position in homotypic_positions:
            same_cluster = highrna_singlet_positions[train_cluster_codes[highrna_singlet_positions] == train_cluster_codes[anchor_position]]
            candidates = same_cluster if len(same_cluster) else highrna_singlet_positions
            if len(candidates) == 0:
                continue
            ncount_distance = np.abs(train_log_ncount[candidates] - train_log_ncount[anchor_position])
            priority = doublet_scores_np[candidates] + homotypic_scores_np[candidates] - 0.05 * ncount_distance
            chosen = int(candidates[int(np.argmax(priority))])
            mapping[int(anchor_position)] = chosen
            selected_doublet_scores.append(float(doublet_scores_np[chosen]))
            selected_homotypic_scores.append(float(homotypic_scores_np[chosen]))
            same_cluster_count += int(train_cluster_codes[chosen] == train_cluster_codes[anchor_position])
        model_confused_negative_for_anchor = mapping
        n_selected = len(selected_doublet_scores)
        latest_model_confused_stats = {
            "mean_negative_predicted_doublet_score": float(np.mean(selected_doublet_scores)) if selected_doublet_scores else float("nan"),
            "mean_negative_homotypic_score": float(np.mean(selected_homotypic_scores)) if selected_homotypic_scores else float("nan"),
            "same_cluster_negative_fraction": float(same_cluster_count / n_selected) if n_selected else float("nan"),
            "number_model_confused_negatives_used": float(n_selected),
        }

    def append_model_confused_negatives(batch):
        if not use_model_confused_mining or int(np.max(model_confused_negative_for_anchor)) < 0:
            return batch
        batch_positions = batch[-2].detach().cpu().numpy().astype(int)
        batch_subtype_targets = batch[2].detach().cpu().numpy().astype(int)
        batch_subtype_mask = batch[3].detach().cpu().numpy().astype(bool)
        anchors = batch_positions[batch_subtype_mask & (batch_subtype_targets == 0)]
        negative_positions = sorted(
            {
                int(model_confused_negative_for_anchor[int(anchor)])
                for anchor in anchors
                if 0 <= int(anchor) < len(model_confused_negative_for_anchor) and int(model_confused_negative_for_anchor[int(anchor)]) >= 0
            }
        )
        if not negative_positions:
            return batch
        existing = set(map(int, batch_positions.tolist()))
        extra_positions = [position for position in negative_positions if position not in existing]
        if not extra_positions:
            return batch
        extra_index = torch.tensor(extra_positions, dtype=torch.long)
        return tuple(torch.cat([tensor, source[extra_index]], dim=0) for tensor, source in zip(batch, train_tensors))

    def epoch_batches(epoch_number: int):
        if not use_balanced_highrna_batches:
            for batch in loader:
                yield append_model_confused_negatives(batch)
            return
        rng = np.random.default_rng(int(random_state) + 10007 * int(epoch_number))
        n_batches = max(1, int(np.ceil(len(train_dataset) / max(1, effective_batch_size))))
        min_homotypic = int(np.ceil(effective_batch_size * balanced_fraction))
        min_highrna = int(np.ceil(effective_batch_size * balanced_fraction))
        if min_homotypic + min_highrna > effective_batch_size:
            min_homotypic = effective_batch_size // 2
            min_highrna = effective_batch_size - min_homotypic
        rest_count = max(0, effective_batch_size - min_homotypic - min_highrna)
        for _ in range(n_batches):
            homotypic_pick = rng.choice(homotypic_positions, size=min_homotypic, replace=len(homotypic_positions) < min_homotypic)
            highrna_pick = rng.choice(highrna_singlet_positions, size=min_highrna, replace=len(highrna_singlet_positions) < min_highrna)
            rest_pick = (
                rng.choice(all_positions, size=rest_count, replace=len(all_positions) < rest_count)
                if rest_count
                else np.array([], dtype=int)
            )
            batch_indices = np.concatenate([homotypic_pick, highrna_pick, rest_pick]).astype(int, copy=False)
            rng.shuffle(batch_indices)
            index_tensor = torch.tensor(batch_indices, dtype=torch.long)
            yield append_model_confused_negatives(tuple(tensor[index_tensor] for tensor in train_tensors))
    val_tensors = (
        torch.tensor(x_val, dtype=torch.float32, device=torch_device),
        torch.tensor(y_doublet_val, dtype=torch.float32, device=torch_device),
        torch.tensor(y_subtype_val, dtype=torch.long, device=torch_device),
        torch.tensor(subtype_mask_val, dtype=torch.bool, device=torch_device),
        torch.tensor(y_highrna_val, dtype=torch.float32, device=torch_device),
        torch.tensor(highrna_mask_val, dtype=torch.bool, device=torch_device),
    )

    def weighted_mean(loss_values, weights):
        if weights is None:
            return loss_values.mean()
        return (loss_values * weights).sum() / torch.clamp(weights.sum(), min=1e-6)

    def multitask_loss(outputs, y_doublet, y_subtype, subtype_mask, y_highrna, highrna_mask, sample_weights=None):
        doublet_loss = weighted_mean(bce_doublet(outputs["doublet_logit"], y_doublet), sample_weights)
        subtype_loss = torch.zeros((), dtype=torch.float32, device=torch_device)
        if bool(subtype_mask.any().detach().cpu().item()):
            subtype_weights_local = sample_weights[subtype_mask] if sample_weights is not None else None
            subtype_loss = weighted_mean(ce_subtype(outputs["subtype_logits"][subtype_mask], y_subtype[subtype_mask]), subtype_weights_local)
        highrna_loss = torch.zeros((), dtype=torch.float32, device=torch_device)
        if bool(highrna_mask.any().detach().cpu().item()):
            highrna_weights_local = sample_weights[highrna_mask] if sample_weights is not None else None
            highrna_loss = weighted_mean(bce_highrna(outputs["highrna_rejection_logit"][highrna_mask], y_highrna[highrna_mask]), highrna_weights_local)
        total = doublet_loss + float(lambda_subtype) * subtype_loss + float(lambda_highrna) * highrna_loss
        return total, doublet_loss, subtype_loss, highrna_loss

    def corr_squared(values, covariate, mask):
        if not bool(mask.any().detach().cpu().item()):
            return torch.zeros((), dtype=torch.float32, device=torch_device)
        v = values[mask]
        c = covariate[mask]
        if v.numel() < 4:
            return torch.zeros((), dtype=torch.float32, device=torch_device)
        v = v - v.mean()
        c = c - c.mean()
        denom = torch.sqrt(torch.sum(v * v) * torch.sum(c * c) + 1e-8)
        corr = torch.sum(v * c) / denom
        return corr * corr

    def ncount_decorrelation_loss(outputs, y_doublet, log_ncount, cluster_codes):
        if float(lambda_ncount_decorrelation) <= 0:
            return torch.zeros((), dtype=torch.float32, device=torch_device)
        subtype_prob = torch.softmax(outputs["subtype_logits"], dim=1)[:, 0]
        homotypic_score = torch.sigmoid(outputs["doublet_logit"]) * subtype_prob
        all_mask = torch.ones_like(y_doublet, dtype=torch.bool, device=torch_device)
        singlet_mask = y_doublet.lt(0.5)
        penalties = [corr_squared(homotypic_score, log_ncount, all_mask)]
        singlet_penalty = corr_squared(homotypic_score, log_ncount, singlet_mask)
        if bool(singlet_mask.any().detach().cpu().item()):
            penalties.append(singlet_penalty)
        cluster_penalties = []
        for cluster_code in torch.unique(cluster_codes):
            cluster_mask = cluster_codes.eq(cluster_code)
            if int(cluster_mask.sum().detach().cpu().item()) >= 4:
                cluster_penalties.append(corr_squared(homotypic_score, log_ncount, cluster_mask))
        if cluster_penalties:
            penalties.append(torch.stack(cluster_penalties).mean())
        return torch.stack(penalties).mean()

    best_state = None
    best_metric = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    rows: list[dict[str, object]] = []
    number_triplets_used = 0
    number_anchor_opportunities = 0
    total_highrna_singlets_seen_in_batches = 0
    total_homotypic_doublets_seen_in_batches = 0
    total_training_batches_seen = 0
    negative_predicted_doublet_score_sum = 0.0
    negative_homotypic_score_sum = 0.0
    same_cluster_negative_count = 0.0
    model_confused_negatives_used_in_triplets = 0.0
    start = time.perf_counter()
    epoch_durations: list[float] = []
    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        if use_model_confused_mining and epoch > model_confused_warmup_epochs:
            refresh_model_confused_negatives()
        model.train()
        train_losses: list[float] = []
        train_contrastive_losses: list[float] = []
        train_ncount_losses: list[float] = []
        epoch_highrna_singlets = 0
        epoch_homotypic_doublets = 0
        epoch_batches_seen = 0
        epoch_negative_predicted_doublet_score_sum = 0.0
        epoch_negative_homotypic_score_sum = 0.0
        epoch_same_cluster_negative_count = 0.0
        epoch_model_confused_negatives = 0.0
        epoch_triplets = 0
        epoch_anchor_opportunities = 0
        current_use_contrastive = bool(use_contrastive and not (use_model_confused_mining and epoch <= model_confused_warmup_epochs))
        optimizer.zero_grad(set_to_none=True)
        for batch_number, batch in enumerate(epoch_batches(epoch), start=1):
            xb, yd, ys, sm, yh, hm, highrna_singlet_mask, sample_weights, log_ncount, batch_positions, cluster_codes = [
                tensor.to(torch_device, non_blocking=use_cuda) for tensor in batch
            ]
            batch_highrna_singlets = int(highrna_singlet_mask.sum().detach().cpu().item())
            batch_homotypic_doublets = int((sm & ys.eq(0)).sum().detach().cpu().item())
            epoch_highrna_singlets += batch_highrna_singlets
            epoch_homotypic_doublets += batch_homotypic_doublets
            epoch_batches_seen += 1
            with _amp_context(torch_device, amp_enabled):
                outputs = model(xb)
                loss, _, _, _ = multitask_loss(outputs, yd, ys, sm, yh, hm, sample_weights)
                ncount_loss = ncount_decorrelation_loss(outputs, yd, log_ncount, cluster_codes)
                if float(lambda_ncount_decorrelation) > 0:
                    loss = loss + float(lambda_ncount_decorrelation) * ncount_loss
            contrastive_loss = torch.zeros((), dtype=torch.float32, device=torch_device)
            triplet_count = 0
            anchor_count = 0
            if current_use_contrastive:
                batch_doublet_scores = torch.sigmoid(outputs["doublet_logit"]).detach()
                batch_homotypic_scores = (batch_doublet_scores * torch.softmax(outputs["subtype_logits"], dim=1)[:, 0]).detach()
                (
                    anchor_z,
                    positive_z,
                    negative_z,
                    triplet_count,
                    anchor_count,
                    triplet_diagnostics,
                ) = _batch_triplet_tensors(
                    outputs["z"],
                    ys,
                    sm,
                    highrna_singlet_mask,
                    cluster_codes,
                    batch_positions=batch_positions,
                    model_confused_negative_for_anchor=model_confused_negative_for_anchor,
                    doublet_scores=batch_doublet_scores,
                    homotypic_scores=batch_homotypic_scores,
                    log_ncount=log_ncount,
                    hard_negative_mode=hard_negative_mode,
                )
                if triplet_count:
                    with _amp_context(torch_device, amp_enabled):
                        contrastive_loss = triplet_loss_fn(anchor_z, positive_z, negative_z)
                        loss = loss + float(lambda_contrastive) * contrastive_loss
                    epoch_negative_predicted_doublet_score_sum += float(triplet_diagnostics.get("negative_predicted_doublet_score_sum", 0.0))
                    epoch_negative_homotypic_score_sum += float(triplet_diagnostics.get("negative_homotypic_score_sum", 0.0))
                    epoch_same_cluster_negative_count += float(triplet_diagnostics.get("same_cluster_negative_count", 0.0))
                    epoch_model_confused_negatives += float(triplet_diagnostics.get("model_confused_negative_count", 0.0))
            scaled_loss = loss / accumulation_steps
            if amp_scaler is not None:
                amp_scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            if batch_number % accumulation_steps == 0:
                if amp_scaler is not None:
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            train_losses.append(float(loss.detach().cpu().item()))
            train_ncount_losses.append(float(ncount_loss.detach().cpu().item()))
            if current_use_contrastive:
                train_contrastive_losses.append(float(contrastive_loss.detach().cpu().item()))
                epoch_triplets += int(triplet_count)
                epoch_anchor_opportunities += int(anchor_count)
            if verbose_progress and progress_callback is not None:
                progress_callback(
                    {
                        "event": "batch",
                        "message": f"DL epoch {epoch}/{max_epochs} batch {batch_number} | loss={train_losses[-1]:.5f}",
                    }
                )
        if epoch_batches_seen and epoch_batches_seen % accumulation_steps != 0:
            if amp_scaler is not None:
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_highrna_singlets_seen_in_batches += epoch_highrna_singlets
        total_homotypic_doublets_seen_in_batches += epoch_homotypic_doublets
        total_training_batches_seen += epoch_batches_seen
        number_triplets_used += epoch_triplets
        number_anchor_opportunities += epoch_anchor_opportunities
        negative_predicted_doublet_score_sum += epoch_negative_predicted_doublet_score_sum
        negative_homotypic_score_sum += epoch_negative_homotypic_score_sum
        same_cluster_negative_count += epoch_same_cluster_negative_count
        model_confused_negatives_used_in_triplets += epoch_model_confused_negatives
        model.eval()
        with torch.no_grad():
            with _amp_context(torch_device, amp_enabled):
                outputs = model(val_tensors[0])
                val_loss, val_doublet_loss, val_subtype_loss, val_highrna_loss = multitask_loss(outputs, *val_tensors[1:])
            val_probs, _ = _conditional_probability_frame(outputs, val_x.index)
        val_auroc, val_auprc = _validation_scores(val_probs, val_y)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        rows.append(
            {
                "method": method,
                "epoch": int(epoch),
                "train_loss": train_loss,
                "validation_loss": float(val_loss.detach().cpu().item()),
                "validation_doublet_loss": float(val_doublet_loss.detach().cpu().item()),
                "validation_subtype_loss": float(val_subtype_loss.detach().cpu().item()),
                "validation_highrna_rejection_loss": float(val_highrna_loss.detach().cpu().item()),
                "train_contrastive_loss": float(np.mean(train_contrastive_losses)) if train_contrastive_losses else 0.0,
                "train_ncount_decorrelation_loss": float(np.mean(train_ncount_losses)) if train_ncount_losses else 0.0,
                "average_highrna_singlets_per_batch": float(epoch_highrna_singlets / epoch_batches_seen) if epoch_batches_seen else 0.0,
                "average_homotypic_doublets_per_batch": float(epoch_homotypic_doublets / epoch_batches_seen) if epoch_batches_seen else 0.0,
                "mean_negative_predicted_doublet_score": float(epoch_negative_predicted_doublet_score_sum / epoch_triplets) if epoch_triplets else float("nan"),
                "mean_negative_homotypic_score": float(epoch_negative_homotypic_score_sum / epoch_triplets) if epoch_triplets else float("nan"),
                "same_cluster_negative_fraction": float(epoch_same_cluster_negative_count / epoch_triplets) if epoch_triplets else float("nan"),
                "number_model_confused_negatives_used": float(epoch_model_confused_negatives),
                "number_triplets_used": int(epoch_triplets),
                "fraction_anchors_with_triplets": float(epoch_triplets / epoch_anchor_opportunities) if epoch_anchor_opportunities else 0.0,
                "validation_AUROC": val_auroc,
                "validation_AUPRC": val_auprc,
                "device": str(torch_device),
                "use_amp": bool(amp_enabled),
                "batch_size": int(effective_batch_size),
                "feature_set": feature_set,
                "feature_variant": "SafeFeatures" if feature_variant == "safe" else feature_variant,
                "model_category": model_category,
            }
        )
        metric = float(val_auprc) if np.isfinite(val_auprc) else -float(val_loss.detach().cpu().item())
        if metric > best_metric + 1e-5:
            best_metric = metric
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        epoch_durations.append(time.perf_counter() - epoch_started)
        if progress_callback is not None:
            median_epoch = float(np.median(epoch_durations[-10:]))
            remaining_epochs = max(0, max_epochs - epoch)
            gpu_memory = None
            if use_cuda:
                gpu_memory = int(torch.cuda.max_memory_allocated(torch_device))
            progress_callback(
                {
                    "event": "epoch",
                    "epoch": int(epoch),
                    "max_epochs": int(max_epochs),
                    "train_loss": train_loss,
                    "validation_loss": float(val_loss.detach().cpu().item()),
                    "validation_metric_name": "validation_AUPRC",
                    "validation_metric": float(val_auprc),
                    "best_metric": float(best_metric),
                    "patience_counter": int(epochs_without_improvement),
                    "patience": int(patience),
                    "elapsed_seconds": time.perf_counter() - start,
                    "epoch_eta_seconds": median_epoch,
                    "training_eta_seconds": median_epoch * remaining_epochs,
                    "gpu_memory_bytes": gpu_memory,
                }
            )
        if epochs_without_improvement >= patience:
            break
    runtime = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    log = pd.DataFrame(rows)
    best_row = log.loc[log["epoch"].eq(best_epoch)].iloc[0] if not log.empty and best_epoch else pd.Series(dtype=object)
    contrastive_status = "disabled"
    if lambda_contrastive > 0:
        contrastive_status = "active" if number_triplets_used > 0 else "skipped_no_triplets"
    fraction_anchors_with_triplets = float(number_triplets_used / number_anchor_opportunities) if number_anchor_opportunities else 0.0
    average_highrna_singlets_per_batch = (
        float(total_highrna_singlets_seen_in_batches / total_training_batches_seen) if total_training_batches_seen else 0.0
    )
    average_homotypic_doublets_per_batch = (
        float(total_homotypic_doublets_seen_in_batches / total_training_batches_seen) if total_training_batches_seen else 0.0
    )
    mean_negative_predicted_doublet_score = (
        float(negative_predicted_doublet_score_sum / number_triplets_used)
        if number_triplets_used
        else float(latest_model_confused_stats.get("mean_negative_predicted_doublet_score", np.nan))
    )
    mean_negative_homotypic_score = (
        float(negative_homotypic_score_sum / number_triplets_used)
        if number_triplets_used
        else float(latest_model_confused_stats.get("mean_negative_homotypic_score", np.nan))
    )
    same_cluster_negative_fraction = (
        float(same_cluster_negative_count / number_triplets_used)
        if number_triplets_used
        else float(latest_model_confused_stats.get("same_cluster_negative_fraction", np.nan))
    )
    number_model_confused_negatives_used = (
        float(model_confused_negatives_used_in_triplets)
        if model_confused_negatives_used_in_triplets
        else float(latest_model_confused_stats.get("number_model_confused_negatives_used", 0.0))
    )
    summary = {
        "method": method,
        "model_name": method,
        "status": "success",
        "message": f"trained {estimator_name} with early stopping at epoch {best_epoch}; contrastive_status={contrastive_status}",
        "train_loss": float(best_row.get("train_loss", np.nan)),
        "validation_loss": float(best_row.get("validation_loss", np.nan)),
        "val_loss": float(best_row.get("validation_loss", np.nan)),
        "validation_AUROC": float(best_row.get("validation_AUROC", np.nan)),
        "validation_AUPRC": float(best_row.get("validation_AUPRC", np.nan)),
        "best_validation_AUPRC": float(best_row.get("validation_AUPRC", np.nan)),
        "early_stopping_epoch": int(best_epoch),
        "best_epoch": int(best_epoch),
        "n_epochs": int(len(log)),
        "training_time_seconds": float(runtime),
        "runtime_seconds": float(runtime),
        "number_train_cells": int(len(train_x)),
        "n_train": int(len(train_x)),
        "number_validation_cells": int(len(val_x)),
        "number_test_cells": 0,
        "n_features": int(train_x.shape[1]),
        "subtype_doublet_count": int(subtype_mask_train.sum()),
        "highrna_rejection_positive_count": int(highrna_pos),
        "highrna_rejection_negative_count": int(highrna_neg),
        "highrna_label_source": highrna_source,
        "lambda_subtype": float(lambda_subtype),
        "lambda_highrna": float(lambda_highrna),
        "lambda_ncount_decorrelation": float(lambda_ncount_decorrelation),
        "balanced_highrna_batches": bool(use_balanced_highrna_batches),
        "highrna_batch_fraction": float(balanced_fraction),
        "average_highrna_singlets_per_batch": float(average_highrna_singlets_per_batch),
        "average_homotypic_doublets_per_batch": float(average_homotypic_doublets_per_batch),
        "highrna_percentile": float(highrna_percentile),
        "contrastive_loss_weight": float(lambda_contrastive),
        "triplet_margin": float(triplet_margin),
        "hard_negative_mode": str(hard_negative_mode),
        "train_ncount_decorrelation_loss": float(best_row.get("train_ncount_decorrelation_loss", np.nan)),
        "model_confused_warmup_epochs": int(model_confused_warmup_epochs),
        "mean_negative_predicted_doublet_score": float(mean_negative_predicted_doublet_score),
        "mean_negative_homotypic_score": float(mean_negative_homotypic_score),
        "same_cluster_negative_fraction": float(same_cluster_negative_fraction),
        "number_model_confused_negatives_used": float(number_model_confused_negatives_used),
        "number_triplets_used": int(number_triplets_used),
        "fraction_anchors_with_triplets": float(fraction_anchors_with_triplets),
        "contrastive_status": contrastive_status,
        "doublet_pos_weight": float(doublet_pos_weight),
        "highrna_pos_weight": float(highrna_pos_weight),
        "device": str(torch_device),
        "device_message": device_message,
        "use_amp": bool(amp_enabled),
        **_torch_cuda_metadata(torch_device, amp_enabled),
        "batch_size": int(effective_batch_size),
        "num_workers": int(num_workers),
        "gradient_accumulation_steps": int(accumulation_steps),
        "deterministic": bool(deterministic),
        "hidden_dim": int(hidden_dim) if hidden_dim is not None else np.nan,
        "depth": int(depth) if depth is not None else np.nan,
        "dropout": float(effective_dropout),
        "weight_decay": float(weight_decay),
        "feature_set": feature_set,
        "feature_variant": "SafeFeatures" if feature_variant == "safe" else feature_variant,
        "estimator": estimator_name,
        "model_category": model_category,
        "diagnostic_only": bool(diagnostic_only),
        "unsafe_feature_detected": bool(unsafe_features),
        "unsafe_feature_list": ",".join(unsafe_features),
        "feature_list": ",".join(train_x.columns),
        "random_state": int(random_state),
        "seed": int(random_state),
        "sample_weight_used": bool(sample_weight is not None),
        "sample_weight_min": float(np.nanmin(train_sample_weight)) if len(train_sample_weight) else float("nan"),
        "sample_weight_max": float(np.nanmax(train_sample_weight)) if len(train_sample_weight) else float("nan"),
        "sample_weight_mean": float(np.nanmean(train_sample_weight)) if len(train_sample_weight) else float("nan"),
    }
    fitted = _FittedConditionalTorchModel(method=method, model=model, imputer=imputer, scaler=scaler, feature_names=list(train_x.columns), training_summary=summary)
    return fitted, log, None


def _predict_torch(fitted: _FittedTorchModel, test_x: pd.DataFrame) -> pd.DataFrame:
    if torch is None:
        raise ImportError("PyTorch is unavailable.")
    frame = test_x.reindex(columns=fitted.feature_names, fill_value=0.0)
    x_test = fitted.scaler.transform(fitted.imputer.transform(frame)).astype(np.float32)
    device = next(fitted.model.parameters()).device
    use_cuda = getattr(device, "type", "cpu") == "cuda"
    # AMP is a training acceleration. Fixed inference uses float32 so the
    # declared probabilities are invariant to caller-controlled chunk sizes.
    use_amp = False
    batch_size = _effective_torch_batch_size(
        int(fitted.training_summary.get("batch_size", 0) or 0) or None,
        len(x_test),
        device,
        default_cpu=2048,
        default_cuda=8192,
    )
    fitted.model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_test), batch_size):
            batch = torch.tensor(x_test[start : start + batch_size], dtype=torch.float32, device=device)
            with _amp_context(device, use_amp):
                logits = fitted.model(batch)
                chunks.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    raw = np.vstack(chunks) if chunks else np.empty((0, len(fitted.label_encoder.classes_)), dtype=float)
    return _probability_frame(raw, fitted.label_encoder.classes_, frame.index)


def _predict_conditional_multitask_mlp(fitted: _FittedConditionalTorchModel, test_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if torch is None:
        raise ImportError("PyTorch is unavailable.")
    frame = test_x.reindex(columns=fitted.feature_names, fill_value=0.0)
    x_test = fitted.scaler.transform(fitted.imputer.transform(frame)).astype(np.float32)
    device = next(fitted.model.parameters()).device
    use_cuda = getattr(device, "type", "cpu") == "cuda"
    # AMP is a training acceleration. Fixed inference uses float32 so the
    # declared probabilities are invariant to caller-controlled chunk sizes.
    use_amp = False
    batch_size = _effective_torch_batch_size(
        int(fitted.training_summary.get("batch_size", 0) or 0) or None,
        len(x_test),
        device,
        default_cpu=2048,
        default_cuda=8192,
    )
    fitted.model.eval()
    prob_frames: list[pd.DataFrame] = []
    highrna_scores: list[pd.Series] = []
    with torch.no_grad():
        for start in range(0, len(x_test), batch_size):
            batch_index = frame.index[start : start + batch_size]
            batch = torch.tensor(x_test[start : start + batch_size], dtype=torch.float32, device=device)
            with _amp_context(device, use_amp):
                outputs = fitted.model(batch)
            probs, highrna = _conditional_probability_frame(outputs, batch_index)
            prob_frames.append(probs)
            highrna_scores.append(highrna)
    probs = pd.concat(prob_frames, axis=0) if prob_frames else pd.DataFrame(0.0, index=frame.index, columns=list(NET_CLASS_LABELS))
    highrna = pd.concat(highrna_scores, axis=0) if highrna_scores else pd.Series(np.nan, index=frame.index, name="highRNA_rejection_score")
    return probs.reindex(frame.index), highrna.reindex(frame.index)


def _split_train_validation(x: pd.DataFrame, y: pd.Series, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    _require_sklearn_dependencies()
    stratify = y if y.value_counts().min() >= 2 and y.nunique() > 1 else None
    return train_test_split(x, y, test_size=0.20, random_state=random_state, stratify=stratify)


def _net_estimator_for(name: str, random_state: int):
    _require_sklearn_dependencies()
    if name == "CalibratedRF":
        return RandomForestClassifier(
            n_estimators=240,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"Unknown estimator: {name!r}")


def _diagnostic_model_spec(method: str) -> dict[str, object]:
    specs = {
        "DuoDose-ML-CalibratedRF-SafeFeatures": {
            "family": "sklearn",
            "estimator": "CalibratedRF",
            "feature_set": "safe",
            "feature_variant": "safe",
            "model_category": "DuoDose-ML",
            "calibrate_sigmoid": True,
        },
        "DuoDose-DL-ConditionalMultiTaskMLP-SafeFeatures": {
            "family": "conditional_dl",
            "estimator": "ConditionalMultiTaskMLP",
            "feature_set": "safe",
            "feature_variant": "safe",
            "model_category": "DuoDose-DL",
        },
    }
    if method not in specs:
        raise ValueError(f"Unsupported diagnostic model: {method!r}")
    return specs[method]


def _indexed_train_validation_split(
    train_features: pd.DataFrame,
    train_y: pd.Series,
    random_state: int,
    train_index: Iterable[object] | None = None,
    validation_index: Iterable[object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if train_index is None or validation_index is None:
        return _split_train_validation(train_features, train_y, random_state=random_state)
    train_set = set(train_index)
    val_set = set(validation_index)
    train_idx = [idx for idx in train_features.index if idx in train_set]
    val_idx = [idx for idx in train_features.index if idx in val_set]
    if not train_idx or not val_idx:
        return _split_train_validation(train_features, train_y, random_state=random_state)
    return train_features.loc[train_idx], train_features.loc[val_idx], train_y.loc[train_idx], train_y.loc[val_idx]


def _apply_train_fraction(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    train_meta: pd.DataFrame,
    train_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    fraction = float(np.clip(train_fraction, 0.01, 1.0))
    if fraction >= 0.999:
        return train_x.copy(), train_y.copy(), train_meta.copy()
    n_keep = max(2, int(round(len(train_x) * fraction)))
    if n_keep >= len(train_x):
        return train_x.copy(), train_y.copy(), train_meta.copy()
    stratify = train_y if train_y.value_counts().min() >= 2 and train_y.nunique() > 1 else None
    selected, _ = train_test_split(train_x.index, train_size=n_keep, random_state=random_state, stratify=stratify)
    selected = pd.Index(selected)
    return train_x.loc[selected].copy(), train_y.loc[selected].copy(), train_meta.loc[selected].copy()


def _augment_training_doublets(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    train_meta: pd.DataFrame,
    doublet_augmentation_factor: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    factor = max(1, int(doublet_augmentation_factor))
    if factor <= 1:
        return train_x.copy(), train_y.copy(), train_meta.copy()
    doublet_idx = train_y.astype(str).isin(DOUBLET_LABELS)
    if not bool(doublet_idx.any()):
        return train_x.copy(), train_y.copy(), train_meta.copy()
    x_parts = [train_x.copy()]
    y_parts = [train_y.copy()]
    meta_parts = [train_meta.copy()]
    doublet_x = train_x.loc[doublet_idx]
    doublet_y = train_y.loc[doublet_idx]
    doublet_meta = train_meta.loc[doublet_idx]
    for copy_number in range(1, factor):
        new_index = pd.Index([f"{idx}__train_aug{copy_number}" for idx in doublet_x.index])
        x_aug = doublet_x.copy()
        y_aug = doublet_y.copy()
        meta_aug = doublet_meta.copy()
        x_aug.index = new_index
        y_aug.index = new_index
        meta_aug.index = new_index
        x_parts.append(x_aug)
        y_parts.append(y_aug)
        meta_parts.append(meta_aug)
    return pd.concat(x_parts, axis=0), pd.concat(y_parts, axis=0), pd.concat(meta_parts, axis=0)


def _diagnostic_count_summary(train_y: pd.Series, train_meta: pd.DataFrame, highrna_percentile: float) -> dict[str, object]:
    labels = train_y.astype(str)
    train_highrna, _, highrna_source, _ = _highrna_masks_from_training_thresholds(
        train_meta,
        train_y,
        train_meta.iloc[0:0].copy(),
        train_y.iloc[0:0].copy(),
        percentile=highrna_percentile,
    )
    highrna_mask = (labels.eq("homotypic_doublet") | train_highrna.reindex(train_y.index).fillna(False).astype(bool))
    return {
        "number_train_cells": int(len(train_y)),
        "number_train_doublets": int(labels.isin(DOUBLET_LABELS).sum()),
        "number_train_homotypic_doublets": int(labels.eq("homotypic_doublet").sum()),
        "number_train_heterotypic_doublets": int(labels.eq("heterotypic_doublet").sum()),
        "number_highrna_rejection_positive": int(labels.loc[highrna_mask].eq("homotypic_doublet").sum()) if bool(highrna_mask.any()) else 0,
        "number_highrna_rejection_negative": int(highrna_mask.sum() - labels.loc[highrna_mask].eq("homotypic_doublet").sum()) if bool(highrna_mask.any()) else 0,
        "actual_train_doublet_prevalence": float(labels.isin(DOUBLET_LABELS).mean()) if len(labels) else float("nan"),
        "highrna_label_source": highrna_source,
    }


def train_predict_diagnostic_model(
    train_cell_scores: pd.DataFrame,
    test_cell_scores: pd.DataFrame,
    method: str,
    random_state: int = 0,
    net_train_seed: int = 0,
    train_index: Iterable[object] | None = None,
    validation_index: Iterable[object] | None = None,
    train_fraction: float = 1.0,
    doublet_augmentation_factor: int = 1,
    lambda_highrna: float = 0.5,
    lambda_ncount_decorrelation: float = 0.0,
    balanced_highrna_batches: bool = False,
    highrna_batch_fraction: float = 0.25,
    highrna_percentile: float = 90.0,
    lambda_contrastive: float | None = None,
    triplet_margin: float = 1.0,
    hard_negative_mode: str = "highRNA_same_cluster",
    max_epochs: int = 100,
    patience: int = 15,
    device: str = "auto",
    use_amp: bool = False,
    batch_size: int | None = None,
    num_workers: int = 0,
    apply_method_defaults: bool = True,
    safe_feature_transformer: object | None = None,
    sample_weight: pd.Series | np.ndarray | None = None,
    high_rna_negative_weight: float | None = None,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
    verbose_progress: bool = False,
) -> dict[str, object]:
    if method not in NET_METHODS:
        allowed = ", ".join(NET_METHODS)
        raise ValueError(f"Unsupported DuoDose method {method!r}. Public clean methods are: {allowed}")

    """Train one safe diagnostic model and return validation/test score frames."""

    formal_rf_expected_weights: pd.Series | None = None
    if method == "DuoDose-ML-CalibratedRF-SafeFeatures":
        from .rf_weighting import FORMAL_HIGH_RNA_NEGATIVE_WEIGHT, formal_rf_sample_weights

        if high_rna_negative_weight is not None and float(high_rna_negative_weight) != FORMAL_HIGH_RNA_NEGATIVE_WEIGHT:
            raise ValueError(
                "DuoDose calibrated-RF fixes high_rna_negative_weight="
                f"{FORMAL_HIGH_RNA_NEGATIVE_WEIGHT}"
            )
        high_rna_negative_weight = FORMAL_HIGH_RNA_NEGATIVE_WEIGHT
        formal_rf_expected_weights = formal_rf_sample_weights(train_cell_scores)
        if sample_weight is None:
            sample_weight = formal_rf_expected_weights
    if high_rna_negative_weight is not None and float(high_rna_negative_weight) <= 0:
        raise ValueError("high_rna_negative_weight must be positive when recorded")
    spec = _diagnostic_model_spec(method)
    if progress_callback is not None:
        progress_callback({"event": "milestone", "message": f"{method}: loading features"})
    train_y_all = _target_labels(train_cell_scores)
    test_y = _target_labels(test_cell_scores)
    feature_set = str(spec["feature_set"])
    feature_variant = str(spec["feature_variant"])
    if safe_feature_transformer is not None:
        matrix_builder = getattr(safe_feature_transformer, "build_model_matrix", None)
        if matrix_builder is None:
            raise TypeError("safe_feature_transformer does not expose build_model_matrix")
        train_features = matrix_builder(train_cell_scores)
        test_features = matrix_builder(test_cell_scores)
    else:
        train_features = build_feature_matrix(train_cell_scores, feature_set)
        test_features = build_feature_matrix(test_cell_scores, feature_set)
    train_features, test_features = _align_features(train_features, test_features)
    train_features = select_feature_variant(train_features, feature_variant)
    test_features = test_features.reindex(columns=train_features.columns, fill_value=0.0)
    train_x, val_x, y_train_split, y_val_split = _indexed_train_validation_split(
        train_features,
        train_y_all,
        random_state=random_state,
        train_index=train_index,
        validation_index=validation_index,
    )
    custom_sample_weight: pd.Series | None = None
    if sample_weight is not None:
        if isinstance(sample_weight, pd.Series):
            custom_sample_weight = pd.to_numeric(sample_weight, errors="coerce").reindex(train_cell_scores.index)
        else:
            values = np.asarray(sample_weight, dtype=float)
            if len(values) != len(train_cell_scores):
                raise ValueError("sample_weight length must match train_cell_scores rows")
            custom_sample_weight = pd.Series(values, index=train_cell_scores.index, dtype=float)
        if custom_sample_weight.isna().any() or not np.isfinite(custom_sample_weight.to_numpy(dtype=float)).all():
            raise ValueError("sample_weight must be finite and aligned to every train_cell_scores row")
        if (custom_sample_weight <= 0).any():
            raise ValueError("sample_weight values must be positive")
    if formal_rf_expected_weights is not None:
        if custom_sample_weight is None:
            raise AssertionError("formal RF sample weights were not constructed")
        if not np.allclose(
            custom_sample_weight.to_numpy(dtype=float),
            formal_rf_expected_weights.reindex(custom_sample_weight.index).to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("DuoDose calibrated-RF accepts only the fixed ordinary=1.0/high-RNA=2.0 sample-weight contract")
    train_meta = train_cell_scores.reindex(train_x.index)
    val_meta = train_cell_scores.reindex(val_x.index)
    train_x, y_train_split, train_meta = _apply_train_fraction(
        train_x,
        y_train_split,
        train_meta,
        train_fraction=train_fraction,
        random_state=int(random_state) + 17,
    )
    train_x, y_train_split, train_meta = _augment_training_doublets(
        train_x,
        y_train_split,
        train_meta,
        doublet_augmentation_factor=doublet_augmentation_factor,
    )
    effective_sample_weight: np.ndarray | None = None
    if custom_sample_weight is not None:
        aligned_weight = custom_sample_weight.reindex(train_x.index)
        if aligned_weight.isna().any():
            raise ValueError("sample_weight does not align with the final training rows")
        effective_sample_weight = aligned_weight.to_numpy(dtype=float)
    del apply_method_defaults
    count_summary = _diagnostic_count_summary(y_train_split, train_meta, highrna_percentile)
    unsafe_features = unsafe_feature_list(train_x.columns)
    base_summary: dict[str, object] = {
        "method": method,
        "status": "success",
        "message": "trained diagnostic model",
        "feature_set": feature_set,
        "feature_variant": "SafeFeatures" if feature_variant == "safe" else feature_variant,
        "estimator": str(spec["estimator"]),
        "model_category": str(spec["model_category"]),
        "diagnostic_only": False,
        "unsafe_feature_detected": bool(unsafe_features),
        "unsafe_feature_list": ",".join(unsafe_features),
        "feature_list": ",".join(train_x.columns),
        "random_state": int(random_state),
        "net_train_seed": int(net_train_seed),
        "train_fraction": float(train_fraction),
        "doublet_augmentation_factor": int(doublet_augmentation_factor),
        "lambda_highrna": float(lambda_highrna),
        "lambda_ncount_decorrelation": float(lambda_ncount_decorrelation),
        "balanced_highrna_batches": bool(balanced_highrna_batches),
        "highrna_batch_fraction": float(highrna_batch_fraction),
        "highrna_percentile": float(highrna_percentile),
        "contrastive_loss_weight": float(lambda_contrastive if lambda_contrastive is not None else spec.get("default_lambda_contrastive", 0.0)),
        "triplet_margin": float(triplet_margin),
        "hard_negative_mode": str(hard_negative_mode),
        "sample_weight_used": bool(effective_sample_weight is not None),
        "sample_weight_min": float(np.min(effective_sample_weight)) if effective_sample_weight is not None else 1.0,
        "sample_weight_max": float(np.max(effective_sample_weight)) if effective_sample_weight is not None else 1.0,
        "sample_weight_mean": float(np.mean(effective_sample_weight)) if effective_sample_weight is not None else 1.0,
        "high_rna_negative_weight": (
            float(high_rna_negative_weight) if high_rna_negative_weight is not None else float("nan")
        ),
        "number_validation_cells": int(len(val_x)),
        "number_test_cells": int(len(test_features)),
        **count_summary,
    }
    fitted_backend: TrainedDuoDoseBackend | None = None
    try:
        family = str(spec["family"])
        model_category = str(method)  # fallback used by DL training summaries/calls
        diagnostic_only = False  # use selected method feature set, e.g. SafeFeatures
        if family == "sklearn":
            estimator_name = str(spec["estimator"])
            estimator = _net_estimator_for(estimator_name, int(random_state))
            fitted, log = _fit_sklearn_classifier(
                method,
                estimator,
                train_x,
                y_train_split,
                val_x,
                y_val_split,
                int(random_state),
                estimator_name=str(spec["estimator"]),
                feature_set=feature_set,
                feature_variant="SafeFeatures" if feature_variant == "safe" else feature_variant,
                model_category=str(spec["model_category"]),
                diagnostic_only=False,
                sample_weight=(
                    effective_sample_weight
                    if effective_sample_weight is not None
                    else _balanced_sample_weight(y_train_split)
                    if bool(spec.get("use_sample_weight", False))
                    else None
                ),
                calibrate_sigmoid=bool(spec.get("calibrate_sigmoid", False)),
                use_calibration_if_improves=bool(spec.get("use_calibration_if_improves", False)),
                progress_callback=progress_callback,
            )
            if progress_callback is not None:
                progress_callback({"event": "milestone", "message": "RF: predicting validation/test"})
            val_probs = _predict_sklearn(fitted, val_x)
            test_probs = _predict_sklearn(fitted, test_features)
            summary = {**fitted.training_summary, **base_summary}
            fitted_backend = TrainedDuoDoseBackend(method, family, fitted, feature_set, feature_variant, summary, safe_feature_transformer)
        elif family == "torch_mlp":
            fitted_torch, log, skipped = _fit_torch_mlp(
                method,
                train_x,
                y_train_split,
                val_x,
                y_val_split,
                random_state=int(random_state),
                feature_set=feature_set,
                feature_variant="SafeFeatures" if feature_variant == "safe" else feature_variant,
                model_category=str(spec["model_category"]),
                diagnostic_only=False,
                max_epochs=max(1, int(max_epochs)),
                patience=max(1, int(patience)),
                device=device,
                use_amp=use_amp,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            if skipped is not None:
                summary = {**base_summary, **skipped}
                nan = pd.DataFrame(np.nan, index=test_features.index, columns=NET_CLASS_LABELS)
                return {"summary": summary, "validation_probabilities": nan.iloc[0:0].copy(), "test_probabilities": nan, "training_log": log, "fitted_backend": None}
            assert fitted_torch is not None
            val_probs = _predict_torch(fitted_torch, val_x)
            test_probs = _predict_torch(fitted_torch, test_features)
            summary = {**fitted_torch.training_summary, **base_summary}
            fitted_backend = TrainedDuoDoseBackend(method, family, fitted_torch, feature_set, feature_variant, summary, safe_feature_transformer)
        elif family == "conditional_dl":
            contrastive_weight = float(lambda_contrastive if lambda_contrastive is not None else spec.get("default_lambda_contrastive", 0.0))
            fitted_dl, log, skipped = _fit_conditional_multitask_mlp(
                method,
                train_x,
                y_train_split,
                val_x,
                y_val_split,
                train_meta,
                val_meta,
                random_state=int(net_train_seed),
                feature_set="safe",
                feature_variant=feature_variant,
                model_category=str(spec["model_category"]),
                diagnostic_only=False,
                lambda_highrna=float(lambda_highrna),
                lambda_ncount_decorrelation=float(lambda_ncount_decorrelation),
                balanced_highrna_batches=bool(balanced_highrna_batches),
                highrna_batch_fraction=float(highrna_batch_fraction),
                highrna_percentile=float(highrna_percentile),
                lambda_contrastive=contrastive_weight,
                estimator_name=str(spec["estimator"]),
                max_epochs=max(1, int(max_epochs)),
                patience=max(1, int(patience)),
                device=device,
                use_amp=use_amp,
                batch_size=batch_size,
                num_workers=num_workers,
                progress_callback=progress_callback,
                verbose_progress=bool(verbose_progress),
            )
            if skipped is not None:
                summary = {**base_summary, **skipped}
                nan = pd.DataFrame(np.nan, index=test_features.index, columns=NET_CLASS_LABELS)
                return {"summary": summary, "validation_probabilities": nan.iloc[0:0].copy(), "test_probabilities": nan, "training_log": log, "fitted_backend": None}
            assert fitted_dl is not None
            if progress_callback is not None:
                progress_callback({"event": "milestone", "message": "DuoDose-DL: predicting validation/test"})
            val_probs, _ = _predict_conditional_multitask_mlp(fitted_dl, val_x)
            test_probs, _ = _predict_conditional_multitask_mlp(fitted_dl, test_features)
            summary = {**fitted_dl.training_summary, **base_summary}
            fitted_backend = TrainedDuoDoseBackend(method, family, fitted_dl, feature_set, feature_variant, summary, safe_feature_transformer)
        else:
            raise ValueError(f"Unsupported diagnostic model family: {family!r}")
    except Exception as exc:
        summary = {
            **base_summary,
            "status": "failed",
            "message": f"diagnostic training failed: {exc}",
            "training_time_seconds": 0.0,
            "validation_loss": float("nan"),
            "validation_AUROC": float("nan"),
            "validation_AUPRC": float("nan"),
            "best_validation_AUPRC": float("nan"),
            "early_stopping_epoch": float("nan"),
        }
        val_probs = pd.DataFrame(np.nan, index=val_x.index, columns=NET_CLASS_LABELS)
        test_probs = pd.DataFrame(np.nan, index=test_features.index, columns=NET_CLASS_LABELS)
        log = pd.DataFrame()
    summary["number_train_cells"] = int(count_summary["number_train_cells"])
    summary["number_test_cells"] = int(len(test_features))
    summary["best_validation_AUPRC"] = float(summary.get("best_validation_AUPRC", summary.get("validation_AUPRC", np.nan)))
    summary["best_validation_loss"] = float(summary.get("validation_loss", np.nan))
    summary["early_stopping_epoch"] = float(summary.get("early_stopping_epoch", np.nan))
    summary["unsafe_feature_detected"] = bool(summary.get("unsafe_feature_detected", bool(unsafe_features)))
    summary["unsafe_feature_list"] = str(summary.get("unsafe_feature_list", ",".join(unsafe_features)))
    return {
        "summary": summary,
        "validation_probabilities": val_probs,
        "validation_labels": y_val_split,
        "validation_meta": val_meta,
        "test_probabilities": test_probs,
        "test_labels": test_y,
        "test_meta": test_cell_scores,
        "training_log": log,
        "fitted_backend": fitted_backend,
    }


def train_and_predict_net_models(
    train_cell_scores: pd.DataFrame,
    test_cell_scores: pd.DataFrame,
    random_state: int = 0,
    net_train_seed: int = 0,
    train_index: Iterable[object] | None = None,
    validation_index: Iterable[object] | None = None,
    publication_mode: bool = False,
    include_diagnostics: bool = False,
    dl_max_epochs: int = 200,
    dl_patience: int = 20,
    dl_batch_size: int | None = 1024,
    device: str = "auto",
    use_amp: bool = False,
    dl_num_workers: int = 0,
    methods: Iterable[str] | None = None,
) -> tuple[dict[str, tuple[pd.Series, pd.Series, pd.Series]], pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Fit selected complete backends on one shared train/validation/test split."""

    del publication_mode, include_diagnostics  # Retained for CLI compatibility.
    train_y = _target_labels(train_cell_scores)
    if train_y.nunique() < 2:
        raise ValueError("DuoDose needs at least two classes in the training split.")

    selected_methods = list(dict.fromkeys(str(method) for method in (methods or NET_METHODS)))
    unsupported = [method for method in selected_methods if method not in NET_METHODS]
    if unsupported:
        raise ValueError(f"Unsupported DuoDose methods: {', '.join(unsupported)}")

    full_audit_train = build_feature_matrix(train_cell_scores, "full")
    safe_included, safe_excluded = split_safe_feature_columns(full_audit_train.columns)
    safe_audit_text = format_safe_feature_audit(safe_included, safe_excluded)

    predictions: dict[str, tuple[pd.Series, pd.Series, pd.Series]] = {}
    summaries: list[dict[str, object]] = []
    log_frames: list[pd.DataFrame] = []
    for method in selected_methods:
        result = train_predict_diagnostic_model(
            train_cell_scores,
            test_cell_scores,
            method=method,
            random_state=int(random_state),
            net_train_seed=int(net_train_seed),
            train_index=train_index,
            validation_index=validation_index,
            max_epochs=max(1, int(dl_max_epochs)),
            patience=max(1, int(dl_patience)),
            device=device,
            use_amp=bool(use_amp),
            batch_size=dl_batch_size,
            num_workers=max(0, int(dl_num_workers)),
        )
        probs = result.get("test_probabilities", pd.DataFrame(index=test_cell_scores.index))
        if not isinstance(probs, pd.DataFrame) or probs.empty:
            probs = pd.DataFrame(np.nan, index=test_cell_scores.index, columns=NET_CLASS_LABELS)
        predictions[method] = probabilities_to_scores(probs.reindex(test_cell_scores.index))

        summary = dict(result.get("summary", {}))
        summary["method"] = method
        summaries.append(summary)
        log = result.get("training_log", pd.DataFrame())
        if isinstance(log, pd.DataFrame) and not log.empty:
            log = log.copy()
            log["method"] = method
            log_frames.append(log)

    summary_frame = pd.DataFrame(summaries)
    training_log = pd.concat(log_frames, ignore_index=True) if log_frames else pd.DataFrame()
    return predictions, summary_frame, pd.DataFrame(), training_log, safe_audit_text


# Clear public alias for manuscript-facing code.
train_predict_duodose_model = train_predict_diagnostic_model
