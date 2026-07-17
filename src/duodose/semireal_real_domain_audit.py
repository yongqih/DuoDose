"""Leakage-aware domain audit for cached semi-real and experimental doublets.

The audit intentionally compares experimentally annotated doublets with held-out
semi-real *heterotypic* doublets.  It is a diagnostic for transferability of
the formal semi-real construction, not a source of labels or tuning signals.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .plotting_style import apply_manuscript_style

from .domain_audit_contract import PRIMARY_ANALYSIS

try:  # Stratified groups are preferable, but retain compatibility with older sklearn.
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - depends on the installed sklearn version.
    StratifiedGroupKFold = None  # type: ignore[assignment,misc]


RANDOM_STATE = 0
N_FOLDS = 3
FOLD_PREVALENCE_TOLERANCE = (0.35, 0.65)
FOLD_SIZE_RATIO_TOLERANCE = (0.50, 1.50)
FORMAL_CONSTRUCTION_VARIANT = "raw_sum_parents_removed"
FORMAL_SAFE_FEATURE_MODE = "fitted_reference"
SCHEMA_VERSION = "semireal_real_domain_audit_v2"

# This is deliberately an allowlist.  Domain discrimination must not be driven
# by a population-composition proxy, fitted method score, rank calibration, or
# direct technical covariate.
RAW_MECHANISM_FEATURE_ALLOWLIST = frozenset(
    {
        "identity_inlier_score",
        "biological_program_coherence_score",
        "dosage_outlier_score",
        "uniform_dosage_inflation_score",
        "dosage_residual",
        "cluster_stable_dosage_robust_z",
        "cluster_marker_dosage_robust_z",
    }
)

DIRECT_TECHNICAL_FEATURES = frozenset(
    {
        "ncount",
        "log_ncount",
        "nfeature",
        "log_nfeature",
        "n_genes",
        "log_n_genes",
        "n_genes_by_counts",
        "total_counts",
        "log_total_counts",
        "pct_counts_mt",
        "percent_mito",
        "mito_fraction",
        "mitochondrial_fraction",
        "cluster_ncount_z",
        "cluster_count_robust_z",
        "cluster_gene_robust_z",
    }
)
SOURCE_METADATA_FEATURES = frozenset(
    {
        "dataset",
        "seed",
        "split",
        "sample",
        "sample_id",
        "batch",
        "batch_id",
        "cell_id",
        "barcode",
        "parent_1_id",
        "parent_2_id",
        "synthetic_cell_id",
        "duodose_cluster",
        "cluster",
        "cell_type",
        "celltype",
    }
)
FORBIDDEN_MECHANISM_TOKENS = (
    "cluster_abundance",
    "cluster_level_expected",
    "duodose_cluster_",
    "sample_id_",
    "batch_",
    "cell_type_",
    "duodose",
    "hybrid",
    "scrublet",
    "rank",
    "percentile",
    "quantile",
    "tail",
    "calibrat",
    "ecdf",
    "probability",
    "ensemble",
    "cell_id",
    "barcode",
    "parent",
    "dataset",
    "seed",
    "split",
    "source",
    "path",
)

SUMMARY_COLUMNS = [
    "dataset",
    "source_run_id",
    "analysis",
    "is_primary",
    "status",
    "message",
    "n_experimental_doublets",
    "n_semireal_heterotypic_doublets",
    "n_semireal_before_parent_unique_filter",
    "n_semireal_after_parent_unique_filter",
    "parent_unique_retention_fraction",
    "n_unique_parents_retained",
    "parent_overlap_across_folds",
    "n_features",
    "n_folds",
    "auroc_mean",
    "auroc_std",
    "auprc_mean",
    "auprc_std",
    "pooled_oof_auroc",
    "pooled_oof_auprc",
    "balanced_accuracy",
    "mcc",
    "split_strategy",
    "fold_experimental_counts",
    "fold_semireal_counts",
    "fold_domain_prevalences",
    "fold_balance_status",
    "fold_balance_message",
    "transformer_reference_provenance_status",
    "construction_variant",
    "safe_feature_mode",
    "safe_feature_transformer_id",
    "safe_feature_reference_pool_id",
    "semireal_split_used",
]
FOLD_COLUMNS = [
    "dataset",
    "source_run_id",
    "analysis",
    "fold",
    "n_train",
    "n_test",
    "n_experimental_test",
    "n_semireal_heterotypic_test",
    "experimental_domain_prevalence",
    "parent_overlap_count",
    "n_parent_components_test",
    "split_strategy",
    "fold_balance_status",
    "fold_balance_message",
    "auroc",
    "auprc",
]
PREDICTION_COLUMNS = [
    "dataset",
    "source_run_id",
    "analysis",
    "fold",
    "stable_cell_id",
    "domain",
    "domain_label",
    "probability_experimental_domain",
]
COEFFICIENT_COLUMNS = [
    "dataset",
    "source_run_id",
    "analysis",
    "feature",
    "mean_coefficient",
    "mean_absolute_coefficient",
    "n_folds",
]
CLUSTER_BALANCE_COLUMNS = [
    "dataset",
    "source_run_id",
    "canonical_cluster",
    "experimental_before_matching",
    "semireal_heterotypic_before_matching",
    "experimental_after_matching",
    "semireal_heterotypic_after_matching",
    "post_matching_counts_equal",
    "status",
    "message",
]
FEATURE_AUDIT_COLUMNS = [
    "dataset",
    "source_run_id",
    "feature",
    "included",
    "category",
    "exclusion_reason",
    "primary_group",
    "source_groups",
    "is_composite",
    "direct_dependencies",
]
PARENT_UNIQUE_SELECTION_COLUMNS = [
    "dataset",
    "source_run_id",
    "stable_cell_id",
    "parent_1_id",
    "parent_2_id",
    "parent_1_cluster",
    "parent_2_cluster",
    "cluster_pair",
    "canonical_cluster",
    "selected",
    "selection_round",
    "selection_reason",
]
MATCHING_EXCLUSION_COLUMNS = [
    "dataset",
    "source_run_id",
    "stable_cell_id",
    "domain",
    "canonical_cluster",
    "reason",
]


class DomainAuditInputError(ValueError):
    """Raised when a cached bundle cannot support the formal domain audit."""


@dataclass(frozen=True)
class DomainAuditBundle:
    dataset: str
    source_run_id: str
    audit_dir: Path
    real_metadata: pd.DataFrame
    semi_metadata: pd.DataFrame
    real_raw_features: pd.DataFrame
    semi_raw_features: pd.DataFrame
    parent_map: pd.DataFrame
    semireal_split_used: str
    construction_variant: str
    safe_feature_mode: str
    safe_feature_transformer_id: str
    safe_feature_reference_pool_id: str
    parent_removal_audit: Mapping[str, Any]


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, compression="infer")
    except Exception as error:  # pragma: no cover - depends on malformed user inputs.
        raise DomainAuditInputError(f"Could not read cached audit input '{path}': {error}") from error


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:  # pragma: no cover - depends on malformed user inputs.
        raise DomainAuditInputError(f"Could not read audit manifest '{path}': {error}") from error
    if not isinstance(payload, Mapping):
        raise DomainAuditInputError(f"Audit manifest '{path}' is not a JSON object.")
    return payload


def _assert_manifest_references(manifest: Mapping[str, Any], audit_dir: Path) -> None:
    artifacts = manifest.get("artifacts", {})
    if artifacts is not None and not isinstance(artifacts, Mapping):
        raise DomainAuditInputError("domain_audit_export_manifest.json has a malformed artifacts mapping.")
    if isinstance(artifacts, Mapping):
        missing = [
            f"{name}={value}"
            for name, value in artifacts.items()
            if not (audit_dir / str(value)).is_file()
        ]
        if missing:
            raise DomainAuditInputError(
                "domain_audit_export_manifest.json references missing exported artifacts: " + ", ".join(missing)
            )
    source_files = manifest.get("source_files", {})
    if source_files is not None and not isinstance(source_files, Mapping):
        raise DomainAuditInputError("domain_audit_export_manifest.json has a malformed source_files mapping.")
    for name in ("dataset_load_path", "score_cache_dir", "shared_parent_pair_plan"):
        value = source_files.get(name) if isinstance(source_files, Mapping) else None
        if value and not Path(str(value)).exists():
            raise DomainAuditInputError(
                f"domain_audit_export_manifest.json references missing {name}: '{value}'."
            )


def _first_existing(audit_dir: Path, names: Sequence[str], glob_pattern: str) -> Path:
    exact = [audit_dir / name for name in names if (audit_dir / name).is_file()]
    matches = exact or sorted(audit_dir.glob(glob_pattern))
    if not matches:
        expected = ", ".join(names)
        raise DomainAuditInputError(
            f"Missing required domain-audit input in '{audit_dir}'. Expected one of: {expected}. "
            "Regenerate this bundle with --export-domain-audit-only."
        )
    if len(matches) > 1 and not exact:
        raise DomainAuditInputError(
            f"Ambiguous cached input for '{glob_pattern}' in '{audit_dir}': "
            f"{', '.join(str(path.name) for path in matches)}. Keep one canonical export."
        )
    return matches[0]


def _optional_existing(audit_dir: Path, names: Sequence[str], glob_pattern: str) -> Path | None:
    exact = [audit_dir / name for name in names if (audit_dir / name).is_file()]
    matches = exact or sorted(audit_dir.glob(glob_pattern))
    if len(matches) > 1 and not exact:
        raise DomainAuditInputError(
            f"Ambiguous cached input for '{glob_pattern}' in '{audit_dir}': "
            f"{', '.join(str(path.name) for path in matches)}. Keep one canonical export."
        )
    return matches[0] if matches else None


def _stable_identifier(frame: pd.DataFrame, source: str) -> str:
    for column in ("stable_cell_id", "cell_id", "obs_name", "synthetic_cell_id", "barcode"):
        if column not in frame.columns:
            continue
        values = frame[column]
        if values.notna().all() and values.astype(str).is_unique:
            return column
    raise DomainAuditInputError(
        f"{source} needs a non-missing, unique stable identifier column. "
        "Expected one of stable_cell_id, cell_id, obs_name, synthetic_cell_id, or barcode."
    )


def _with_stable_identifier(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    identifier = _stable_identifier(frame, source)
    result = frame.copy()
    result["stable_cell_id"] = result[identifier].astype(str)
    if not result["stable_cell_id"].is_unique:
        raise DomainAuditInputError(f"{source} has duplicate stable cell identifiers.")
    return result


def _align_metadata_and_features(
    metadata: pd.DataFrame,
    raw_features: pd.DataFrame,
    *,
    metadata_name: str,
    feature_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = _with_stable_identifier(metadata, metadata_name)
    raw_features = _with_stable_identifier(raw_features, feature_name)
    metadata_ids = pd.Index(metadata["stable_cell_id"])
    feature_ids = pd.Index(raw_features["stable_cell_id"])
    if not metadata_ids.equals(feature_ids):
        if set(metadata_ids) != set(feature_ids):
            raise DomainAuditInputError(
                f"{metadata_name} and {feature_name} do not contain the same stable cell identifiers."
            )
        raw_features = raw_features.set_index("stable_cell_id").loc[metadata_ids].reset_index()
    return metadata.reset_index(drop=True), raw_features.reset_index(drop=True)


def _single_value(frame: pd.DataFrame, aliases: Sequence[str], source: str) -> str:
    column = next((name for name in aliases if name in frame.columns), None)
    if column is None:
        raise DomainAuditInputError(
            f"{source} is missing required metadata '{aliases[0]}'. "
            "Regenerate cached inputs with --export-domain-audit-only."
        )
    values = frame[column].dropna().astype(str).str.strip()
    values = values.loc[values.ne("")]
    unique = sorted(values.unique())
    if len(unique) != 1:
        rendered = ", ".join(unique) if unique else "none"
        raise DomainAuditInputError(
            f"{source} must contain exactly one non-empty '{column}' value; found {rendered}."
        )
    return unique[0]


def _assert_common_fitted_reference(
    real_metadata: pd.DataFrame,
    semi_metadata: pd.DataFrame,
) -> tuple[str, str, str]:
    real_mode = _single_value(real_metadata, ("safe_feature_mode",), "fully-real metadata")
    semi_mode = _single_value(semi_metadata, ("safe_feature_mode",), "semi-real metadata")
    if real_mode != FORMAL_SAFE_FEATURE_MODE or semi_mode != FORMAL_SAFE_FEATURE_MODE:
        raise DomainAuditInputError(
            "Cached domain-audit inputs must use safe_feature_mode='fitted_reference'. "
            "Context-local exports cannot be compared here; regenerate with --export-domain-audit-only."
        )
    real_transformer = _single_value(
        real_metadata,
        ("safe_feature_transformer_id", "transformer_id"),
        "fully-real metadata",
    )
    semi_transformer = _single_value(
        semi_metadata,
        ("safe_feature_transformer_id", "transformer_id"),
        "semi-real metadata",
    )
    if real_transformer != semi_transformer:
        raise DomainAuditInputError(
            "Fully-real and semi-real inputs use different SafeFeatureTransformer identifiers. "
            "Regenerate both inputs together with --export-domain-audit-only."
        )
    real_reference = _single_value(
        real_metadata,
        ("safe_feature_reference_pool_id", "reference_pool_id"),
        "fully-real metadata",
    )
    semi_reference = _single_value(
        semi_metadata,
        ("safe_feature_reference_pool_id", "reference_pool_id"),
        "semi-real metadata",
    )
    if real_reference != semi_reference:
        raise DomainAuditInputError(
            "Fully-real and semi-real inputs use different fitted reference pools. "
            "Regenerate both inputs together with --export-domain-audit-only."
        )
    return real_mode, real_transformer, real_reference


def _manifest_values(payload: Any, keys: Iterable[str]) -> list[str]:
    wanted = set(keys)
    values: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key in wanted and value is not None:
                if isinstance(value, (list, tuple, set)):
                    values.extend(str(item).strip() for item in value if str(item).strip())
                elif not isinstance(value, Mapping):
                    rendered = str(value).strip()
                    if rendered:
                        values.append(rendered)
            values.extend(_manifest_values(value, wanted))
    elif isinstance(payload, list):
        for value in payload:
            values.extend(_manifest_values(value, wanted))
    return values


def _nonempty_field_values(frame: pd.DataFrame, aliases: Sequence[str]) -> list[str]:
    values: list[str] = []
    for alias in aliases:
        if alias not in frame.columns:
            continue
        series = frame[alias].dropna().astype(str).str.strip()
        values.extend(series.loc[series.ne("")].tolist())
    return values


def _assert_formal_construction(
    semi_metadata: pd.DataFrame,
    parent_map: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> str:
    values = _nonempty_field_values(semi_metadata, ("construction_variant",))
    values.extend(_nonempty_field_values(parent_map, ("construction_variant",)))
    values.extend(_manifest_values(manifest, ("construction_variant",)))
    unique = sorted(set(values))
    if not unique:
        raise DomainAuditInputError(
            "Cached audit inputs do not record construction_variant. This audit requires "
            "construction_variant='raw_sum_parents_removed'; regenerate with --export-domain-audit-only."
        )
    if len(unique) != 1:
        raise DomainAuditInputError(
            "Cached audit inputs contain ambiguous construction variants: "
            f"{', '.join(unique)}. Regenerate one formal raw_sum_parents_removed bundle."
        )
    variant = unique[0]
    if variant != FORMAL_CONSTRUCTION_VARIANT:
        raise DomainAuditInputError(
            "This domain audit accepts only construction_variant='raw_sum_parents_removed'. "
            f"Found '{variant}'. Do not substitute downsampled_parents_removed, "
            "downsampled_parents_retained, or raw_sum_parents_retained."
        )
    return variant


def _as_bool(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    unknown = ~normalized.isin({"true", "false", "1", "0", "yes", "no"}) & series.notna()
    if unknown.any():
        raise DomainAuditInputError(f"Parent-removal field '{column}' is not boolean-like.")
    return normalized.isin({"true", "1", "yes"})


def _parent_removal_pair(
    parent_map: pd.DataFrame,
    families: Sequence[tuple[str, str]],
    purpose: str,
) -> tuple[str, str, int]:
    for first, second in families:
        if first not in parent_map.columns or second not in parent_map.columns:
            continue
        included = _as_bool(parent_map[first], first) | _as_bool(parent_map[second], second)
        count = int(included.sum())
        if count:
            raise DomainAuditInputError(
                f"Formal parents-removed audit failed: {count} synthetic rows have a parent in {purpose} "
                f"according to '{first}'/'{second}'."
            )
        return first, second, count
    raise DomainAuditInputError(
        f"Cached parent map lacks explicit parent-removal fields for {purpose}. "
        "Regenerate it with --export-domain-audit-only; this audit will not infer removal status."
    )


def _assert_parent_removal(parent_map: pd.DataFrame) -> Mapping[str, Any]:
    reference = _parent_removal_pair(
        parent_map,
        (
            ("parent_1_in_reference", "parent_2_in_reference"),
            ("parent_1_in_reference_pool", "parent_2_in_reference_pool"),
            ("parent_1_in_fit_reference", "parent_2_in_fit_reference"),
        ),
        "the fitted reference pool",
    )
    try:
        clean_training = _parent_removal_pair(
            parent_map,
            (
                ("parent_1_in_clean_training", "parent_2_in_clean_training"),
                ("parent_1_in_clean_singlet_training", "parent_2_in_clean_singlet_training"),
                ("parent_1_in_fit_clean_singlets", "parent_2_in_fit_clean_singlets"),
            ),
            "clean-singlet training rows",
        )
    except DomainAuditInputError:
        declared_mode = _single_value(
            parent_map,
            ("parent_reference_mode",),
            "semi-real parent map",
        )
        if declared_mode != "removed":
            raise
        clean_training = ("parent_reference_mode", "parent_reference_mode", 0)
    return {
        "reference_pool_fields": list(reference[:2]),
        "clean_training_fields": list(clean_training[:2]),
        "reference_parent_rows_present": reference[2],
        "clean_training_parent_rows_present": clean_training[2],
        "parents_removed_verified": True,
    }


def _experimental_doublet_mask(metadata: pd.DataFrame) -> pd.Series:
    candidates = (
        "experimental_singlet_doublet_label",
        "experimental_doublet_label",
        "experimental_label",
    )
    column = next((name for name in candidates if name in metadata.columns), None)
    if column is None:
        raise DomainAuditInputError(
            "Fully-real metadata lacks an experimental singlet/doublet label. Predicted labels are not accepted."
        )
    values = metadata[column]
    if pd.api.types.is_numeric_dtype(values):
        return values.eq(1)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "doublet", "doublets", "experimental_doublet", "experimentally_detectable_doublet"})


def _heterotypic_mask(metadata: pd.DataFrame) -> pd.Series:
    candidates = ("synthetic_subtype", "doublet_subtype", "model_class_label", "true_label")
    column = next((name for name in candidates if name in metadata.columns), None)
    if column is None:
        raise DomainAuditInputError(
            "Semi-real metadata does not record a synthetic subtype. This audit requires held-out "
            "heterotypic semi-real doublets and will not substitute all doublets."
        )
    values = metadata[column].astype(str).str.strip().str.lower()
    mask = values.isin({"heterotypic", "heterotypic_doublet"})
    if not bool(mask.any()):
        raise DomainAuditInputError(
            "Semi-real metadata contains no held-out heterotypic doublets. Regenerate the formal export."
        )
    return mask


def _split_mask(metadata: pd.DataFrame, split_name: str) -> pd.Series:
    if "split" not in metadata.columns:
        raise DomainAuditInputError("Semi-real metadata is missing split labels.")
    values = metadata["split"].astype(str).str.strip().str.lower()
    expected = {split_name.lower(), f"semi_real_{split_name.lower()}"}
    mask = values.isin(expected)
    if not bool(mask.any()):
        observed = ", ".join(sorted(values.dropna().unique()))
        raise DomainAuditInputError(
            f"Semi-real export labelled '{split_name}' has incompatible split values: {observed}."
        )
    return mask


def _normalize_parent_map(parent_map: pd.DataFrame) -> pd.DataFrame:
    parent_map = _with_stable_identifier(parent_map, "semi-real parent map")
    for column in ("parent_1_id", "parent_2_id"):
        if column not in parent_map.columns:
            raise DomainAuditInputError(f"Semi-real parent map is missing required '{column}'.")
        if parent_map[column].isna().any() or parent_map[column].astype(str).str.strip().eq("").any():
            raise DomainAuditInputError(f"Semi-real parent map contains missing '{column}' values.")
    return parent_map


def _load_feature_manifest(audit_dir: Path) -> pd.DataFrame:
    path = _optional_existing(
        audit_dir,
        ("domain_audit_feature_manifest.csv",),
        "*feature*manifest*.csv*",
    )
    if path is None:
        raise DomainAuditInputError(
            "Cached domain-audit inputs lack domain_audit_feature_manifest.csv. "
            "Regenerate with --export-domain-audit-only; the mechanism allowlist must be auditable."
        )
    manifest = _read_csv(path)
    if "feature" not in manifest.columns and "feature_name" in manifest.columns:
        manifest = manifest.rename(columns={"feature_name": "feature"})
    if "feature" not in manifest.columns:
        raise DomainAuditInputError(f"Feature manifest '{path}' is missing the 'feature' column.")
    return manifest


def _load_bundle(audit_dir: str | Path) -> DomainAuditBundle:
    directory = Path(audit_dir).expanduser().resolve()
    if not directory.is_dir():
        raise DomainAuditInputError(f"Domain-audit input directory does not exist: '{directory}'.")
    manifest_path = _first_existing(
        directory,
        ("domain_audit_export_manifest.json",),
        "*export*manifest*.json",
    )
    manifest = _read_json(manifest_path)
    _assert_manifest_references(manifest, directory)
    real_metadata_path = _first_existing(
        directory,
        ("domain_audit_fully_real_metadata.csv.gz", "domain_audit_fully_real_metadata.csv"),
        "*fully*real*metadata*.csv*",
    )
    real_feature_path = _first_existing(
        directory,
        (
            "domain_audit_fully_real_safe_features_raw.csv.gz",
            "domain_audit_fully_real_safe_features_raw.csv",
        ),
        "*fully*real*safe*features*raw*.csv*",
    )
    split_name = "test"
    semi_metadata_path = _optional_existing(
        directory,
        ("domain_audit_test_metadata.csv.gz", "domain_audit_test_metadata.csv"),
        "*test*metadata*.csv*",
    )
    semi_feature_path = _optional_existing(
        directory,
        (
            "domain_audit_test_safe_features_raw.csv.gz",
            "domain_audit_test_safe_features_raw.csv",
        ),
        "*test*safe*features*raw*.csv*",
    )
    if semi_metadata_path is None or semi_feature_path is None:
        split_name = "validation"
        semi_metadata_path = _first_existing(
            directory,
            ("domain_audit_validation_metadata.csv.gz", "domain_audit_validation_metadata.csv"),
            "*validation*metadata*.csv*",
        )
        semi_feature_path = _first_existing(
            directory,
            (
                "domain_audit_validation_safe_features_raw.csv.gz",
                "domain_audit_validation_safe_features_raw.csv",
            ),
            "*validation*safe*features*raw*.csv*",
        )
    parent_map_path = _first_existing(
        directory,
        ("domain_audit_semireal_parent_map.csv.gz", "domain_audit_semireal_parent_map.csv"),
        "*parent*map*.csv*",
    )
    real_metadata, real_features = _align_metadata_and_features(
        _read_csv(real_metadata_path),
        _read_csv(real_feature_path),
        metadata_name="fully-real metadata",
        feature_name="fully-real raw SafeFeatures",
    )
    semi_metadata, semi_features = _align_metadata_and_features(
        _read_csv(semi_metadata_path),
        _read_csv(semi_feature_path),
        metadata_name=f"semi-real {split_name} metadata",
        feature_name=f"semi-real {split_name} raw SafeFeatures",
    )
    if "split" in real_metadata.columns:
        real_splits = real_metadata["split"].dropna().astype(str).str.strip().str.lower()
        if not real_splits.empty and not real_splits.eq("fully_real").all():
            raise DomainAuditInputError(
                "A cached file labelled fully real contains a non-fully_real split. "
                "Do not rename semi-real test rows as fully real; regenerate with --export-domain-audit-only."
            )
    for synthetic_column in ("model_class_label", "synthetic_subtype"):
        if synthetic_column in real_metadata.columns and real_metadata[synthetic_column].notna().any():
            raise DomainAuditInputError(
                "A cached file labelled fully real contains synthetic model labels. "
                "Regenerate it from original experimental cells with --export-domain-audit-only."
            )
    mode, transformer_id, reference_pool_id = _assert_common_fitted_reference(real_metadata, semi_metadata)
    parent_map = _normalize_parent_map(_read_csv(parent_map_path))
    construction_variant = _assert_formal_construction(semi_metadata, parent_map, manifest)
    parent_removal_audit = _assert_parent_removal(parent_map)

    real_mask = _experimental_doublet_mask(real_metadata)
    if not bool(real_mask.any()):
        raise DomainAuditInputError(
            "Fully-real metadata has no experimentally annotated doublets. Do not use predicted doublets."
        )
    semi_mask = _split_mask(semi_metadata, split_name) & _heterotypic_mask(semi_metadata)
    if not bool(semi_mask.any()):
        raise DomainAuditInputError(
            f"No held-out heterotypic semi-real doublets are available in the {split_name} export."
        )
    real_metadata = real_metadata.loc[real_mask].reset_index(drop=True)
    real_features = real_features.loc[real_mask].reset_index(drop=True)
    semi_metadata = semi_metadata.loc[semi_mask].reset_index(drop=True)
    semi_features = semi_features.loc[semi_mask].reset_index(drop=True)
    parent_map = parent_map.set_index("stable_cell_id").loc[semi_metadata["stable_cell_id"]].reset_index()
    exported_pair_keys = parent_map.apply(
        lambda row: "|".join(sorted((str(row["parent_1_id"]), str(row["parent_2_id"])))), axis=1
    )
    parent_removal_audit = {
        **dict(parent_removal_audit),
        "n_duplicate_parent_pairs_in_exported_heterotypic_split": int(exported_pair_keys.duplicated().sum()),
        "n_repeated_parent_pairs_in_exported_heterotypic_split": int(exported_pair_keys.duplicated(keep=False).sum()),
    }

    dataset_values = _nonempty_field_values(real_metadata, ("dataset",))
    if not dataset_values:
        dataset_values = _manifest_values(manifest, ("dataset", "dataset_name"))
    dataset = sorted(set(dataset_values))[0] if len(set(dataset_values)) == 1 else directory.parent.name
    source_run_values = _manifest_values(manifest, ("source_run_id", "run_id"))
    if not source_run_values:
        source_run_values = _nonempty_field_values(semi_metadata, ("source_run_id", "run_id"))
    source_run_id = sorted(set(source_run_values))[0] if len(set(source_run_values)) == 1 else directory.name

    return DomainAuditBundle(
        dataset=str(dataset),
        source_run_id=str(source_run_id),
        audit_dir=directory,
        real_metadata=real_metadata,
        semi_metadata=semi_metadata,
        real_raw_features=real_features,
        semi_raw_features=semi_features,
        parent_map=parent_map,
        semireal_split_used=split_name,
        construction_variant=construction_variant,
        safe_feature_mode=mode,
        safe_feature_transformer_id=transformer_id,
        safe_feature_reference_pool_id=reference_pool_id,
        parent_removal_audit=parent_removal_audit,
    )


def _feature_manifest_lookup(manifest: pd.DataFrame) -> Mapping[str, Mapping[str, Any]]:
    if manifest.empty:
        return {}
    lookup: dict[str, Mapping[str, Any]] = {}
    for _, row in manifest.drop_duplicates("feature", keep="last").iterrows():
        lookup[str(row["feature"])] = row.to_dict()
    return lookup


def _audit_feature_list(bundle: DomainAuditBundle) -> tuple[list[str], pd.DataFrame, Mapping[str, Any]]:
    real_columns = set(bundle.real_raw_features.columns) - {"stable_cell_id"}
    semi_columns = set(bundle.semi_raw_features.columns) - {"stable_cell_id"}
    if real_columns != semi_columns:
        only_real = sorted(real_columns - semi_columns)
        only_semi = sorted(semi_columns - real_columns)
        raise DomainAuditInputError(
            "Fully-real and semi-real raw SafeFeature columns differ. "
            f"Only fully real: {only_real[:8]}; only semi real: {only_semi[:8]}."
        )
    manifest = _feature_manifest_lookup(_load_feature_manifest(bundle.audit_dir))
    rows: list[dict[str, Any]] = []
    included: list[str] = []
    for feature in sorted(real_columns):
        lower = feature.lower()
        manifest_row = manifest.get(feature, {})
        if feature in RAW_MECHANISM_FEATURE_ALLOWLIST:
            category = "audit_raw_mechanism_features"
            reason = ""
            selected = True
            included.append(feature)
        elif lower in DIRECT_TECHNICAL_FEATURES:
            category = "direct_technical_covariate"
            reason = "direct_technical_covariate"
            selected = False
        elif lower in SOURCE_METADATA_FEATURES:
            category = "source_or_identifier"
            reason = "source_or_identifier"
            selected = False
        elif any(token in lower for token in FORBIDDEN_MECHANISM_TOKENS):
            category = "contaminating_or_downstream"
            reason = "forbidden_name_or_category"
            selected = False
        else:
            category = "unclassified"
            reason = "not_in_explicit_raw_mechanism_allowlist"
            selected = False
        rows.append(
            {
                "dataset": bundle.dataset,
                "source_run_id": bundle.source_run_id,
                "feature": feature,
                "included": selected,
                "category": category,
                "exclusion_reason": reason,
                "primary_group": manifest_row.get("primary_group", ""),
                "source_groups": manifest_row.get("source_groups", ""),
                "is_composite": manifest_row.get("is_composite", ""),
                "direct_dependencies": manifest_row.get("direct_dependencies", ""),
            }
        )
    audit = pd.DataFrame(rows, columns=FEATURE_AUDIT_COLUMNS)
    _assert_raw_mechanism_features(included)
    if not included:
        raise DomainAuditInputError(
            "Cached raw SafeFeatures contain no explicit audit_raw_mechanism_features. "
            "Regenerate a common fitted-reference domain-audit bundle; no fallback feature set is allowed."
        )
    feature_list = {
        "feature_set": "audit_raw_mechanism_features",
        "candidate_features": sorted(real_columns),
        "included_features": included,
        "excluded_features": audit.loc[~audit["included"], ["feature", "exclusion_reason"]].to_dict("records"),
        "unclassified_features": audit.loc[audit["category"].eq("unclassified"), "feature"].tolist(),
        "counts": {
            "candidate": int(len(audit)),
            "included": int(audit["included"].sum()),
            "excluded": int((~audit["included"]).sum()),
            "unclassified": int(audit["category"].eq("unclassified").sum()),
        },
        "allowlist": sorted(RAW_MECHANISM_FEATURE_ALLOWLIST),
    }
    return included, audit, feature_list


def _assert_raw_mechanism_features(features: Sequence[str]) -> None:
    invalid = [feature for feature in features if feature not in RAW_MECHANISM_FEATURE_ALLOWLIST]
    if invalid:
        raise AssertionError(f"Raw mechanism feature allowlist violated: {invalid}")
    contaminating = []
    for feature in features:
        lower = feature.lower()
        if lower in DIRECT_TECHNICAL_FEATURES or lower in SOURCE_METADATA_FEATURES:
            contaminating.append(feature)
        elif any(token in lower for token in FORBIDDEN_MECHANISM_TOKENS):
            contaminating.append(feature)
    if contaminating:
        raise AssertionError(f"Contaminating feature retained by raw mechanism audit: {contaminating}")


def _parent_components(parent_map: pd.DataFrame) -> Mapping[str, str]:
    parents = pd.unique(parent_map[["parent_1_id", "parent_2_id"]].astype(str).to_numpy().ravel())
    parent = {value: value for value in parents}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for _, row in parent_map.iterrows():
        union(str(row["parent_1_id"]), str(row["parent_2_id"]))
    return {
        str(row["stable_cell_id"]): f"semi_parent_component:{find(str(row['parent_1_id']))}"
        for _, row in parent_map.iterrows()
    }


def _numeric_column(metadata: pd.DataFrame, raw_features: pd.DataFrame, names: Sequence[str]) -> pd.Series | None:
    for name in names:
        if name in metadata.columns:
            return pd.to_numeric(metadata[name], errors="coerce")
        if name in raw_features.columns:
            return pd.to_numeric(raw_features[name], errors="coerce")
    return None


def _append_domain(
    metadata: pd.DataFrame,
    raw_features: pd.DataFrame,
    *,
    domain: str,
    feature_names: Sequence[str],
    parent_components: Mapping[str, str] | None,
) -> pd.DataFrame:
    result = metadata.copy()
    result["domain"] = domain
    result["domain_label"] = int(domain == "experimental_doublet")
    result["_group"] = (
        result["stable_cell_id"].map(parent_components).fillna("")
        if parent_components is not None
        else ""
    )
    if parent_components is None:
        result["_group"] = "real:" + result["stable_cell_id"].astype(str)
    if result["_group"].astype(str).str.strip().eq("").any():
        raise DomainAuditInputError("A held-out semi-real doublet is missing its parent-component group.")
    feature_indexed = raw_features.set_index("stable_cell_id")
    for name in feature_names:
        result[f"raw::{name}"] = pd.to_numeric(
            feature_indexed.loc[result["stable_cell_id"], name].to_numpy(), errors="coerce"
        )
    log_count = _numeric_column(metadata, raw_features, ("log_nCount", "log_ncount", "log_total_counts"))
    if log_count is None:
        counts = _numeric_column(metadata, raw_features, ("nCount", "ncount", "total_counts"))
        log_count = np.log1p(counts) if counts is not None else None
    log_feature = _numeric_column(
        metadata,
        raw_features,
        ("log_nFeature", "log_nfeature", "log_n_genes"),
    )
    if log_feature is None:
        detected = _numeric_column(metadata, raw_features, ("nFeature", "nfeature", "n_genes", "n_genes_by_counts"))
        log_feature = np.log1p(detected) if detected is not None else None
    mito = _numeric_column(
        metadata,
        raw_features,
        ("percent_mito", "pct_counts_mt", "mito_fraction", "mitochondrial_fraction"),
    )
    if log_count is not None:
        result["technical::log_ncount"] = np.asarray(log_count)
    if log_feature is not None:
        result["technical::log_nfeature"] = np.asarray(log_feature)
    if mito is not None:
        result["technical::mito_fraction"] = np.asarray(mito)
    if "duodose_cluster" not in result.columns:
        raise DomainAuditInputError(
            "Domain audit metadata must provide canonical 'duodose_cluster' for cluster-constrained matching."
        )
    result["canonical_cluster"] = result["duodose_cluster"].astype(str).str.strip()
    if result["canonical_cluster"].eq("").any() or result["canonical_cluster"].eq("nan").any():
        raise DomainAuditInputError("Canonical duodose_cluster contains missing values; matching cannot proceed.")
    return result


def _cluster_pair(parent_1_cluster: object, parent_2_cluster: object) -> str:
    values = [str(parent_1_cluster).strip(), str(parent_2_cluster).strip()]
    normalized = [value if value and value.lower() not in {"nan", "none", "<na>"} else "unknown" for value in values]
    return "|".join(sorted(normalized))


def _select_parent_unique_heterotypic_edges(
    bundle: DomainAuditBundle,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    """Round-robin maximal matching over held-out semi-real parent-pair strata."""

    edges = bundle.parent_map.copy()
    required = {"stable_cell_id", "parent_1_id", "parent_2_id"}
    missing = sorted(required - set(edges.columns))
    if missing:
        raise DomainAuditInputError("Parent-unique selection requires: " + ", ".join(missing))
    edges["parent_1_id"] = edges["parent_1_id"].astype(str)
    edges["parent_2_id"] = edges["parent_2_id"].astype(str)
    if edges["parent_1_id"].eq(edges["parent_2_id"]).any():
        raise DomainAuditInputError("Parent-unique selection received a semi-real doublet with identical parent IDs.")
    if edges["stable_cell_id"].duplicated().any():
        raise DomainAuditInputError("Parent-unique selection received duplicate semi-real cell IDs.")
    for column in ("parent_1_cluster", "parent_2_cluster"):
        if column not in edges.columns:
            edges[column] = "unknown"
    if "duodose_cluster" not in bundle.semi_metadata.columns or "duodose_cluster" not in bundle.real_metadata.columns:
        raise DomainAuditInputError(
            "Parent-unique selection requires canonical duodose_cluster values for semi-real and experimental cells."
        )
    semi_clusters = bundle.semi_metadata.set_index("stable_cell_id")["duodose_cluster"].astype(str).str.strip()
    edges["canonical_cluster"] = edges["stable_cell_id"].map(semi_clusters)
    if edges["canonical_cluster"].isna().any() or edges["canonical_cluster"].eq("").any():
        raise DomainAuditInputError("Parent-unique selection could not map every semi-real edge to a canonical cluster.")
    experimental_cluster_capacity = (
        bundle.real_metadata["duodose_cluster"].astype(str).str.strip().value_counts(sort=False).to_dict()
    )
    edges["cluster_pair"] = [
        _cluster_pair(parent_1_cluster, parent_2_cluster)
        for parent_1_cluster, parent_2_cluster in zip(edges["parent_1_cluster"], edges["parent_2_cluster"], strict=True)
    ]
    strata = {
        stratum: group.sort_values(
            ["cluster_pair", "parent_1_id", "parent_2_id", "stable_cell_id"], kind="mergesort"
        ).reset_index(drop=True)
        for stratum, group in edges.groupby("cluster_pair", sort=True)
    }
    positions = {stratum: 0 for stratum in strata}
    used_parents: set[str] = set()
    selected_by_canonical_cluster: dict[str, int] = {}
    selected_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    selection_round = 0
    while any(positions[stratum] < len(strata[stratum]) for stratum in strata):
        selection_round += 1
        for stratum in sorted(strata):
            position = positions[stratum]
            if position >= len(strata[stratum]):
                continue
            row = strata[stratum].iloc[position]
            positions[stratum] += 1
            parent_1 = str(row["parent_1_id"])
            parent_2 = str(row["parent_2_id"])
            canonical_cluster = str(row["canonical_cluster"])
            has_cluster_capacity = selected_by_canonical_cluster.get(canonical_cluster, 0) < int(
                experimental_cluster_capacity.get(canonical_cluster, 0)
            )
            selected = parent_1 not in used_parents and parent_2 not in used_parents and has_cluster_capacity
            if selected:
                used_parents.update((parent_1, parent_2))
                selected_ids.add(str(row["stable_cell_id"]))
                selected_by_canonical_cluster[canonical_cluster] = (
                    selected_by_canonical_cluster.get(canonical_cluster, 0) + 1
                )
                reason = "selected_parent_unique"
            elif parent_1 in used_parents or parent_2 in used_parents:
                reason = "excluded_parent_reuse"
            else:
                reason = "excluded_no_experimental_cluster_capacity"
            records.append(
                {
                    "dataset": bundle.dataset,
                    "source_run_id": bundle.source_run_id,
                    "stable_cell_id": str(row["stable_cell_id"]),
                    "parent_1_id": parent_1,
                    "parent_2_id": parent_2,
                    "parent_1_cluster": row["parent_1_cluster"],
                    "parent_2_cluster": row["parent_2_cluster"],
                    "cluster_pair": stratum,
                    "canonical_cluster": canonical_cluster,
                    "selected": selected,
                    "selection_round": selection_round,
                    "selection_reason": reason,
                }
            )
    selection = pd.DataFrame(records, columns=PARENT_UNIQUE_SELECTION_COLUMNS)
    selected = selection.loc[selection["selected"]].copy()
    selected_parent_ids = pd.unique(selected[["parent_1_id", "parent_2_id"]].astype(str).to_numpy().ravel())
    if len(selected_parent_ids) != 2 * len(selected):
        raise AssertionError("Parent-unique selection retained a reused parent.")
    stats = {
        "n_semireal_before_parent_unique_filter": int(len(selection)),
        "n_semireal_after_parent_unique_filter": int(len(selected)),
        "parent_unique_retention_fraction": float(len(selected) / len(selection)) if len(selection) else np.nan,
        "n_unique_parents_retained": int(len(selected_parent_ids)),
        "n_excluded_due_to_parent_reuse": int(selection["selection_reason"].eq("excluded_parent_reuse").sum()),
        "n_excluded_no_experimental_cluster_capacity": int(
            selection["selection_reason"].eq("excluded_no_experimental_cluster_capacity").sum()
        ),
        "selected_ids": set(selected_ids),
        "cluster_pair_by_id": dict(zip(selected["stable_cell_id"], selected["cluster_pair"], strict=True)),
        "retained_counts_by_cluster_pair": {
            str(cluster_pair): int(count)
            for cluster_pair, count in selected["cluster_pair"].value_counts(sort=False).sort_index().items()
        },
    }
    return selection, stats


def _apply_parent_unique_selection(
    frame: pd.DataFrame,
    selection_stats: Mapping[str, Any],
) -> pd.DataFrame:
    selected_ids = set(selection_stats["selected_ids"])
    cluster_pair_by_id = dict(selection_stats["cluster_pair_by_id"])
    semi_mask = frame["domain"].eq("semireal_heterotypic_doublet")
    selected_mask = frame["stable_cell_id"].isin(selected_ids)
    result = frame.loc[~semi_mask | selected_mask].copy().reset_index(drop=True)
    selected_semi_mask = result["domain"].eq("semireal_heterotypic_doublet")
    result.loc[selected_semi_mask, "parent_unique_cluster_pair"] = result.loc[
        selected_semi_mask, "stable_cell_id"
    ].map(cluster_pair_by_id)
    # Each retained edge is parent-unique, so it is a standalone CV group.
    result.loc[selected_semi_mask, "_group"] = "semi_parent_unique:" + result.loc[
        selected_semi_mask, "stable_cell_id"
    ].astype(str)
    return result


def _combined_frame(bundle: DomainAuditBundle, feature_names: Sequence[str]) -> pd.DataFrame:
    parent_components = _parent_components(bundle.parent_map)
    real = _append_domain(
        bundle.real_metadata,
        bundle.real_raw_features,
        domain="experimental_doublet",
        feature_names=feature_names,
        parent_components=None,
    )
    semi = _append_domain(
        bundle.semi_metadata,
        bundle.semi_raw_features,
        domain="semireal_heterotypic_doublet",
        feature_names=feature_names,
        parent_components=parent_components,
    )
    return pd.concat([real, semi], ignore_index=True, sort=False)


def _deterministic_cap(frame: pd.DataFrame, max_cells_per_domain: int | None) -> pd.DataFrame:
    if max_cells_per_domain is None:
        return frame.copy()
    if max_cells_per_domain < 1:
        raise ValueError("max_cells_per_domain must be positive or None.")
    keep: list[pd.DataFrame] = []
    for _, group in frame.groupby("domain", sort=True):
        if len(group) <= max_cells_per_domain:
            keep.append(group)
            continue
        ranks = group["stable_cell_id"].map(
            lambda value: int(hashlib.sha256(f"{RANDOM_STATE}:{value}".encode("utf-8")).hexdigest()[:16], 16)
        )
        keep.append(group.assign(_cap_rank=ranks).nsmallest(max_cells_per_domain, "_cap_rank").drop(columns="_cap_rank"))
    return pd.concat(keep, ignore_index=True)


def _coarse_bin(series: pd.Series, n_bins: int = 5) -> pd.Series:
    if series.notna().sum() == 0:
        return pd.Series(-1, index=series.index, dtype=int)
    ranks = series.rank(method="first", pct=True)
    return np.minimum((ranks * n_bins).fillna(-1).astype(int), n_bins - 1)


def _cluster_balance(
    before: pd.DataFrame,
    after: pd.DataFrame,
    bundle: DomainAuditBundle,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    clusters = sorted(set(before["canonical_cluster"]).union(set(after["canonical_cluster"])))
    for cluster in clusters:
        before_cluster = before.loc[before["canonical_cluster"].eq(cluster)]
        after_cluster = after.loc[after["canonical_cluster"].eq(cluster)]
        exp_before = int(before_cluster["domain"].eq("experimental_doublet").sum())
        semi_before = int(before_cluster["domain"].eq("semireal_heterotypic_doublet").sum())
        exp_after = int(after_cluster["domain"].eq("experimental_doublet").sum())
        semi_after = int(after_cluster["domain"].eq("semireal_heterotypic_doublet").sum())
        equal = exp_after == semi_after
        status = "PASS" if equal and exp_after > 0 else "WARNING"
        message = "matched 1:1 within canonical cluster and coarse technical strata"
        if exp_after == 0 and exp_before and semi_before:
            message = "no shared coarse log-count/log-feature stratum"
        elif not equal:
            message = "post-matching domain counts are unequal"
        rows.append(
            {
                "dataset": bundle.dataset,
                "source_run_id": bundle.source_run_id,
                "canonical_cluster": cluster,
                "experimental_before_matching": exp_before,
                "semireal_heterotypic_before_matching": semi_before,
                "experimental_after_matching": exp_after,
                "semireal_heterotypic_after_matching": semi_after,
                "post_matching_counts_equal": bool(equal),
                "status": status,
                "message": message,
            }
        )
    return pd.DataFrame(rows, columns=CLUSTER_BALANCE_COLUMNS)


def _matched_subsample(
    frame: pd.DataFrame,
    bundle: DomainAuditBundle,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = ("technical::log_ncount", "technical::log_nfeature")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DomainAuditInputError(
            "Cluster-constrained matching requires coarse log-count and log-feature covariates; "
            f"missing {missing}. Regenerate the cached domain-audit input bundle."
        )
    matched_rows: list[pd.DataFrame] = []
    matching_exclusions: list[dict[str, Any]] = []

    def append_pair(experimental_row: pd.DataFrame, semi_row: pd.DataFrame) -> None:
        """Keep each experimental cell with its selected semi-real edge for CV."""

        component = str(semi_row["_group"].iloc[0])
        if not component or component == "nan":
            raise DomainAuditInputError("A matched semi-real doublet is missing its parent component.")
        experimental_row = experimental_row.copy()
        semi_row = semi_row.copy()
        experimental_row["_matched_parent_component"] = component
        semi_row["_matched_parent_component"] = component
        experimental_row["parent_unique_cluster_pair"] = semi_row["parent_unique_cluster_pair"].iloc[0]
        matched_rows.extend((experimental_row, semi_row))

    for cluster, cluster_frame in frame.groupby("canonical_cluster", sort=True):
        working = cluster_frame.copy()
        working["_count_bin"] = _coarse_bin(working["technical::log_ncount"])
        working["_feature_bin"] = _coarse_bin(working["technical::log_nfeature"])
        used_experimental_ids: set[str] = set()
        matched_semi_ids: set[str] = set()
        for _, stratum in working.groupby(["_count_bin", "_feature_bin"], sort=True, dropna=False):
            experimental = stratum.loc[stratum["domain"].eq("experimental_doublet")].sort_values(
                ["technical::log_ncount", "technical::log_nfeature", "stable_cell_id"], kind="mergesort"
            )
            semi = stratum.loc[stratum["domain"].eq("semireal_heterotypic_doublet")].sort_values(
                ["technical::log_ncount", "technical::log_nfeature", "stable_cell_id"], kind="mergesort"
            )
            n_pairs = min(len(experimental), len(semi))
            if n_pairs:
                for pair_index in range(n_pairs):
                    experimental_row = experimental.iloc[[pair_index]].copy()
                    semi_row = semi.iloc[[pair_index]].copy()
                    append_pair(experimental_row, semi_row)
                    used_experimental_ids.add(str(experimental_row["stable_cell_id"].iloc[0]))
                    matched_semi_ids.add(str(semi_row["stable_cell_id"].iloc[0]))

        # Coarse bins are the primary match.  A deterministic nearest-covariate
        # fallback within the same canonical cluster retains selected edges when
        # one exact bin has no experimental counterpart.
        remaining_experimental = working.loc[
            working["domain"].eq("experimental_doublet")
            & ~working["stable_cell_id"].astype(str).isin(used_experimental_ids)
        ].copy()
        remaining_semi = working.loc[
            working["domain"].eq("semireal_heterotypic_doublet")
            & ~working["stable_cell_id"].astype(str).isin(matched_semi_ids)
        ].copy()
        if len(remaining_experimental) < len(remaining_semi):
            if remaining_experimental.empty:
                for _, excluded in remaining_semi.iterrows():
                    matching_exclusions.append(
                        {
                            "dataset": bundle.dataset,
                            "source_run_id": bundle.source_run_id,
                            "stable_cell_id": str(excluded["stable_cell_id"]),
                            "domain": str(excluded["domain"]),
                            "canonical_cluster": str(cluster),
                            "reason": "dropped_unmatched_stratum_no_experimental_cells",
                        }
                    )
                continue
            raise DomainAuditInputError(
                f"Canonical cluster '{cluster}' has {len(remaining_semi)} selected semi-real doublets but only "
                f"{len(remaining_experimental)} remaining experimental doublets for 1:1 matching."
            )
        distance_columns = ["technical::log_ncount", "technical::log_nfeature"]
        if "technical::mito_fraction" in working.columns and working["technical::mito_fraction"].notna().any():
            distance_columns.append("technical::mito_fraction")
        scales: dict[str, float] = {}
        for column in distance_columns:
            values = pd.to_numeric(working[column], errors="coerce").to_numpy(dtype=float)
            scale = float(np.nanstd(values))
            scales[column] = scale if np.isfinite(scale) and scale > 0 else 1.0
        remaining_semi = remaining_semi.sort_values(
            ["technical::log_ncount", "technical::log_nfeature", "stable_cell_id"], kind="mergesort"
        )
        for _, semi_row in remaining_semi.iterrows():
            distances = np.zeros(len(remaining_experimental), dtype=float)
            for column in distance_columns:
                candidate_values = pd.to_numeric(remaining_experimental[column], errors="coerce").to_numpy(dtype=float)
                target_value = float(pd.to_numeric(pd.Series([semi_row[column]]), errors="coerce").iloc[0])
                difference = np.abs(candidate_values - target_value) / scales[column]
                distances += np.where(np.isfinite(difference), difference, 1.0)
            tie_break = remaining_experimental["stable_cell_id"].astype(str).to_numpy()
            selected_position = int(np.lexsort((tie_break, distances))[0])
            experimental_row = remaining_experimental.iloc[[selected_position]]
            append_pair(experimental_row, semi_row.to_frame().T)
            remaining_experimental = remaining_experimental.drop(remaining_experimental.index[selected_position])
    matched = pd.concat(matched_rows, ignore_index=True) if matched_rows else frame.iloc[0:0].copy()
    balance = _cluster_balance(frame, matched, bundle)
    exclusions = pd.DataFrame(matching_exclusions, columns=MATCHING_EXCLUSION_COLUMNS)
    if not exclusions.empty:
        dropped_clusters = set(exclusions["canonical_cluster"].astype(str))
        mask = balance["canonical_cluster"].astype(str).isin(dropped_clusters)
        balance.loc[mask, "status"] = "PASS"
        balance.loc[mask, "message"] = "unmatched semi-real-only stratum dropped before strict 1:1 matching"
    if matched.empty or matched["domain"].nunique() != 2:
        raise DomainAuditInputError(
            "No 1:1 matches remained after canonical-cluster and coarse log-count/log-feature matching."
        )
    expected_semi = int(frame["domain"].eq("semireal_heterotypic_doublet").sum()) - len(exclusions)
    matched_semi = int(matched["domain"].eq("semireal_heterotypic_doublet").sum())
    if matched_semi != expected_semi:
        raise DomainAuditInputError(
            f"Only {matched_semi} of {expected_semi} selected parent-unique semi-real doublets were matched."
        )
    return matched, balance, exclusions


def _matched_parent_block_splits(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, Mapping[str, Any]]]:
    """Create balanced parent-disjoint folds for matched experimental/semi-real pairs."""

    if "_matched_parent_component" not in frame.columns:
        raise DomainAuditInputError("Matched audit rows are missing parent-component block assignments.")
    component_column = frame["_matched_parent_component"].astype(str)
    if component_column.eq("").any() or component_column.eq("nan").any():
        raise DomainAuditInputError("Matched audit rows contain an empty parent-component block assignment.")
    blocks: list[dict[str, Any]] = []
    for component, block in frame.groupby("_matched_parent_component", sort=True):
        experimental = block.loc[block["domain"].eq("experimental_doublet")]
        semi_real = block.loc[block["domain"].eq("semireal_heterotypic_doublet")]
        if experimental.empty or semi_real.empty:
            raise DomainAuditInputError(f"Matched block '{component}' does not contain both domains.")
        if len(experimental) != len(semi_real):
            raise DomainAuditInputError(
                f"Matched block '{component}' is not one-to-one: "
                f"{len(experimental)} experimental versus {len(semi_real)} semi-real rows."
            )
        parent_ids: set[str] = set()
        for column in ("parent_1_id", "parent_2_id"):
            if column not in semi_real.columns:
                raise DomainAuditInputError(f"Matched semi-real rows are missing '{column}' for parent-fold validation.")
            parent_ids.update(
                value
                for value in semi_real[column].dropna().astype(str).str.strip()
                if value and value.lower() not in {"nan", "none", "<na>"}
            )
        if not parent_ids:
            raise DomainAuditInputError(f"Matched block '{component}' has no usable parent IDs.")
        cluster_pairs = semi_real["parent_unique_cluster_pair"].dropna().astype(str).unique()
        if len(cluster_pairs) != 1:
            raise DomainAuditInputError(
                f"Matched block '{component}' must contain exactly one selected semi-real cluster-pair stratum."
            )
        blocks.append(
            {
                "component": str(component),
                "indices": np.asarray(block.index, dtype=int),
                "n_experimental": int(len(experimental)),
                "n_semireal": int(len(semi_real)),
                "parent_ids": frozenset(parent_ids),
                "cluster_pair": str(cluster_pairs[0]),
            }
        )
    if len(blocks) < 2:
        raise DomainAuditInputError("Matched audit has fewer than two parent-connected blocks.")
    parent_block_owner: dict[str, str] = {}
    for block in blocks:
        for parent_id in block["parent_ids"]:
            existing_block = parent_block_owner.setdefault(parent_id, str(block["component"]))
            if existing_block != block["component"]:
                raise DomainAuditInputError(
                    "Parent-unique semi-real selection was violated: "
                    f"parent '{parent_id}' occurs in both '{existing_block}' and '{block['component']}'."
                )
    n_parent_unique_pairs = sum(int(block["n_semireal"]) for block in blocks)
    if n_parent_unique_pairs < 30:
        raise DomainAuditInputError(
            "Insufficient parent-unique semi-real heterotypic doublets for matched audit: "
            f"{n_parent_unique_pairs} matched pairs remain; at least 30 are required."
        )

    def assign_blocks(n_folds: int, fallback_reason: str = "") -> list[tuple[np.ndarray, np.ndarray, Mapping[str, Any]]]:
        if len(blocks) < n_folds:
            raise DomainAuditInputError(f"Only {len(blocks)} parent-connected blocks are available for {n_folds} folds.")
        ordered = sorted(
            blocks,
            key=lambda block: (-(block["n_experimental"] + block["n_semireal"]), -block["n_semireal"], block["component"]),
        )
        fold_blocks: list[list[dict[str, Any]]] = [[] for _ in range(n_folds)]
        fold_experimental = np.zeros(n_folds, dtype=float)
        fold_semireal = np.zeros(n_folds, dtype=float)
        total_experimental = sum(block["n_experimental"] for block in ordered)
        total_semireal = sum(block["n_semireal"] for block in ordered)
        target_experimental = total_experimental / n_folds
        target_semireal = total_semireal / n_folds
        target_total = (total_experimental + total_semireal) / n_folds
        stratum_total = pd.Series([block["cluster_pair"] for block in ordered]).value_counts().to_dict()
        fold_stratum_counts: list[dict[str, int]] = [dict() for _ in range(n_folds)]

        def objective(candidate_fold: int, block: Mapping[str, Any]) -> float:
            projected_experimental = fold_experimental.copy()
            projected_semireal = fold_semireal.copy()
            projected_experimental[candidate_fold] += int(block["n_experimental"])
            projected_semireal[candidate_fold] += int(block["n_semireal"])
            projected_total = projected_experimental + projected_semireal
            prevalence = np.divide(
                projected_experimental,
                projected_total,
                out=np.full(n_folds, 0.5, dtype=float),
                where=projected_total > 0,
            )
            stratum = str(block["cluster_pair"])
            projected_stratum = fold_stratum_counts[candidate_fold].get(stratum, 0) + int(block["n_semireal"])
            target_stratum = float(stratum_total[stratum]) / n_folds
            stratum_penalty = ((projected_stratum - target_stratum) / max(target_stratum, 1.0)) ** 2
            return float(
                np.square((projected_total - target_total) / max(target_total, 1.0)).sum()
                + 2.0 * np.square((projected_experimental - target_experimental) / max(target_experimental, 1.0)).sum()
                + 2.0 * np.square((projected_semireal - target_semireal) / max(target_semireal, 1.0)).sum()
                + 4.0 * np.square(prevalence - 0.5).sum()
                + 0.25 * stratum_penalty
            )

        # Seed every fold with a balanced block before greedily filling deficits.
        for fold_index, block in enumerate(ordered[:n_folds]):
            fold_blocks[fold_index].append(block)
            fold_experimental[fold_index] += int(block["n_experimental"])
            fold_semireal[fold_index] += int(block["n_semireal"])
            stratum = str(block["cluster_pair"])
            fold_stratum_counts[fold_index][stratum] = fold_stratum_counts[fold_index].get(stratum, 0) + int(
                block["n_semireal"]
            )
        for block in ordered[n_folds:]:
            best_fold = min(range(n_folds), key=lambda fold_index: (objective(fold_index, block), fold_index))
            fold_blocks[best_fold].append(block)
            fold_experimental[best_fold] += int(block["n_experimental"])
            fold_semireal[best_fold] += int(block["n_semireal"])
            stratum = str(block["cluster_pair"])
            fold_stratum_counts[best_fold][stratum] = fold_stratum_counts[best_fold].get(stratum, 0) + int(
                block["n_semireal"]
            )

        all_indices = np.arange(len(frame), dtype=int)
        seen_parent_folds: dict[str, int] = {}
        splits: list[tuple[np.ndarray, np.ndarray, Mapping[str, Any]]] = []
        for fold_index, assigned_blocks in enumerate(fold_blocks, start=1):
            test_indices = np.sort(np.concatenate([block["indices"] for block in assigned_blocks]))
            train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
            n_experimental = int(fold_experimental[fold_index - 1])
            n_semireal = int(fold_semireal[fold_index - 1])
            if n_experimental == 0 or n_semireal == 0:
                raise DomainAuditInputError(f"Fold {fold_index} lacks one domain after balanced block assignment.")
            prevalence = n_experimental / (n_experimental + n_semireal)
            if not (FOLD_PREVALENCE_TOLERANCE[0] <= prevalence <= FOLD_PREVALENCE_TOLERANCE[1]):
                raise DomainAuditInputError(
                    f"Fold {fold_index} experimental prevalence {prevalence:.3f} is outside "
                    f"{FOLD_PREVALENCE_TOLERANCE[0]:.2f}-{FOLD_PREVALENCE_TOLERANCE[1]:.2f}."
                )
            total_size_ratio = (n_experimental + n_semireal) / target_total
            if not (FOLD_SIZE_RATIO_TOLERANCE[0] <= total_size_ratio <= FOLD_SIZE_RATIO_TOLERANCE[1]):
                raise DomainAuditInputError(
                    f"Fold {fold_index} has {n_experimental + n_semireal} rows versus target {target_total:.1f} "
                    f"(size ratio {total_size_ratio:.3f}), outside "
                    f"{FOLD_SIZE_RATIO_TOLERANCE[0]:.2f}-{FOLD_SIZE_RATIO_TOLERANCE[1]:.2f}."
                )
            parent_ids = set().union(*(set(block["parent_ids"]) for block in assigned_blocks))
            for parent_id in parent_ids:
                existing_fold = seen_parent_folds.setdefault(parent_id, fold_index)
                if existing_fold != fold_index:
                    raise DomainAuditInputError(
                        f"Parent '{parent_id}' appears in validation folds {existing_fold} and {fold_index}."
                    )
            strategy = "matched_parent_unique_round_robin_stratified_greedy"
            if n_folds == 2:
                strategy += "_two_fold_limited_sample"
            splits.append(
                (
                    train_indices,
                    test_indices,
                    {
                        "split_strategy": strategy,
                        "n_experimental_test": n_experimental,
                        "n_semireal_heterotypic_test": n_semireal,
                        "experimental_domain_prevalence": prevalence,
                        "parent_overlap_count": 0,
                        "n_parent_components_test": len(assigned_blocks),
                        "fold_balance_status": "PASS",
                        "fold_balance_message": fallback_reason,
                    },
                )
            )
        return splits

    n_folds = N_FOLDS if n_parent_unique_pairs >= 60 else 2
    return assign_blocks(n_folds)


def _cv_splits(
    frame: pd.DataFrame,
    *,
    analysis: str,
) -> Iterable[tuple[np.ndarray, np.ndarray, Mapping[str, Any]]]:
    if analysis == PRIMARY_ANALYSIS:
        return _matched_parent_block_splits(frame)
    y = frame["domain_label"].to_numpy(dtype=int)
    groups = frame["_group"].astype(str).to_numpy()
    if len(np.unique(y)) != 2 or min(np.bincount(y)) < N_FOLDS:
        raise DomainAuditInputError("Each domain needs at least three rows for three-fold domain-audit CV.")
    if len(np.unique(groups)) < N_FOLDS:
        raise DomainAuditInputError("At least three parent-disjoint groups are required for domain-audit CV.")
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        raw_splits = splitter.split(np.zeros(len(frame)), y, groups)
        strategy = "stratified_group_kfold_parent_components"
    else:
        splitter = GroupKFold(n_splits=N_FOLDS)
        raw_splits = splitter.split(np.zeros(len(frame)), y, groups)
        strategy = "group_kfold_parent_components"
    result: list[tuple[np.ndarray, np.ndarray, Mapping[str, Any]]] = []
    for train_index, test_index in raw_splits:
        test_y = y[test_index]
        n_experimental = int(test_y.sum())
        n_semireal = int((test_y == 0).sum())
        prevalence = n_experimental / len(test_y)
        result.append(
            (
                train_index,
                test_index,
                {
                    "split_strategy": strategy,
                    "n_experimental_test": n_experimental,
                    "n_semireal_heterotypic_test": n_semireal,
                    "experimental_domain_prevalence": prevalence,
                    "parent_overlap_count": np.nan,
                    "n_parent_components_test": int(len(np.unique(groups[test_index]))),
                    "fold_balance_status": (
                        "PASS" if FOLD_PREVALENCE_TOLERANCE[0] <= prevalence <= FOLD_PREVALENCE_TOLERANCE[1] else "WARNING"
                    ),
                    "fold_balance_message": "",
                },
            )
        )
    return result


def _evaluate_analysis(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    analysis: str,
    bundle: DomainAuditBundle,
    parent_unique_stats: Mapping[str, Any],
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not feature_columns:
        raise DomainAuditInputError(f"{analysis} has no permitted feature columns.")
    x = frame.loc[:, feature_columns]
    y = frame["domain_label"].to_numpy(dtype=int)
    predictions: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    coefficients: list[np.ndarray] = []
    pooled_labels: list[int] = []
    pooled_probabilities: list[float] = []
    splits = list(_cv_splits(frame, analysis=analysis))
    for fold, (train_index, test_index, fold_details) in enumerate(splits, start=1):
        if progress_callback is not None:
            progress_callback({"event": "fold", "message": f"{bundle.dataset}: {analysis} fold {fold}/{len(splits)}", "dataset": bundle.dataset, "fold": fold, "n_folds": len(splits)})
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]
        )
        pipeline.fit(x.iloc[train_index], y[train_index])
        probabilities = pipeline.predict_proba(x.iloc[test_index])[:, 1]
        test_y = y[test_index]
        if int(fold_details["n_experimental_test"]) != int(test_y.sum()) or int(
            fold_details["n_semireal_heterotypic_test"]
        ) != int((test_y == 0).sum()):
            raise DomainAuditInputError(f"Fold {fold} domain counts disagree with the assigned validation rows.")
        folds.append(
            {
                "dataset": bundle.dataset,
                "source_run_id": bundle.source_run_id,
                "analysis": analysis,
                "fold": fold,
                "n_train": int(len(train_index)),
                "n_test": int(len(test_index)),
                "n_experimental_test": int(test_y.sum()),
                "n_semireal_heterotypic_test": int((test_y == 0).sum()),
                "experimental_domain_prevalence": float(fold_details["experimental_domain_prevalence"]),
                "parent_overlap_count": fold_details["parent_overlap_count"],
                "n_parent_components_test": int(fold_details["n_parent_components_test"]),
                "split_strategy": str(fold_details["split_strategy"]),
                "fold_balance_status": str(fold_details["fold_balance_status"]),
                "fold_balance_message": str(fold_details["fold_balance_message"]),
                "auroc": float(roc_auc_score(test_y, probabilities)),
                "auprc": float(average_precision_score(test_y, probabilities)),
            }
        )
        coefficients.append(pipeline.named_steps["classifier"].coef_.ravel())
        pooled_labels.extend(test_y.tolist())
        pooled_probabilities.extend(probabilities.tolist())
        for row_index, probability in zip(test_index, probabilities, strict=True):
            row = frame.iloc[row_index]
            predictions.append(
                {
                    "dataset": bundle.dataset,
                    "source_run_id": bundle.source_run_id,
                    "analysis": analysis,
                    "fold": fold,
                    "stable_cell_id": row["stable_cell_id"],
                    "domain": row["domain"],
                    "domain_label": int(row["domain_label"]),
                    "probability_experimental_domain": float(probability),
                }
            )
    fold_frame = pd.DataFrame(folds, columns=FOLD_COLUMNS)
    pooled_label_array = np.asarray(pooled_labels, dtype=int)
    pooled_probability_array = np.asarray(pooled_probabilities, dtype=float)
    fold_prevalences = fold_frame["experimental_domain_prevalence"].to_numpy(dtype=float)
    similar_prevalence = bool(fold_prevalences.max() - fold_prevalences.min() <= 0.10)
    if similar_prevalence:
        auprc_mean = float(fold_frame["auprc"].mean())
        auprc_std = float(fold_frame["auprc"].std(ddof=0))
        summary_message = ""
    else:
        auprc_mean = np.nan
        auprc_std = np.nan
        summary_message = "Fold AUPRC is not averaged because validation prevalences differ; use pooled_oof_auprc."
    coefficient_array = np.vstack(coefficients)
    coefficient_frame = pd.DataFrame(
        {
            "dataset": bundle.dataset,
            "source_run_id": bundle.source_run_id,
            "analysis": analysis,
            "feature": list(feature_columns),
            "mean_coefficient": coefficient_array.mean(axis=0),
            "mean_absolute_coefficient": np.abs(coefficient_array).mean(axis=0),
            "n_folds": coefficient_array.shape[0],
        }
    )
    summary = {
        "dataset": bundle.dataset,
        "source_run_id": bundle.source_run_id,
        "analysis": analysis,
        "is_primary": analysis == PRIMARY_ANALYSIS,
        "status": "PASS",
        "message": summary_message,
        "n_experimental_doublets": int(frame["domain"].eq("experimental_doublet").sum()),
        "n_semireal_heterotypic_doublets": int(frame["domain"].eq("semireal_heterotypic_doublet").sum()),
        "n_semireal_before_parent_unique_filter": int(
            parent_unique_stats["n_semireal_before_parent_unique_filter"]
        ),
        "n_semireal_after_parent_unique_filter": int(
            parent_unique_stats["n_semireal_after_parent_unique_filter"]
        ),
        "parent_unique_retention_fraction": float(parent_unique_stats["parent_unique_retention_fraction"]),
        "n_unique_parents_retained": int(parent_unique_stats["n_unique_parents_retained"]),
        "parent_overlap_across_folds": int(
            pd.to_numeric(fold_frame["parent_overlap_count"], errors="coerce").fillna(0).max()
        )
        if analysis == PRIMARY_ANALYSIS
        else np.nan,
        "n_features": int(len(feature_columns)),
        "n_folds": int(len(fold_frame)),
        "auroc_mean": float(fold_frame["auroc"].mean()),
        "auroc_std": float(fold_frame["auroc"].std(ddof=0)),
        "auprc_mean": auprc_mean,
        "auprc_std": auprc_std,
        "pooled_oof_auroc": float(roc_auc_score(pooled_label_array, pooled_probability_array)),
        "pooled_oof_auprc": float(average_precision_score(pooled_label_array, pooled_probability_array)),
        "balanced_accuracy": float(balanced_accuracy_score(pooled_label_array, pooled_probability_array >= 0.5)),
        "mcc": float(matthews_corrcoef(pooled_label_array, pooled_probability_array >= 0.5)),
        "split_strategy": ";".join(sorted(fold_frame["split_strategy"].dropna().unique())),
        "fold_experimental_counts": json.dumps(fold_frame["n_experimental_test"].astype(int).tolist()),
        "fold_semireal_counts": json.dumps(fold_frame["n_semireal_heterotypic_test"].astype(int).tolist()),
        "fold_domain_prevalences": json.dumps([round(value, 6) for value in fold_prevalences.tolist()]),
        "fold_balance_status": "PASS" if fold_frame["fold_balance_status"].eq("PASS").all() else "WARNING",
        "fold_balance_message": "; ".join(
            sorted({message for message in fold_frame["fold_balance_message"].astype(str) if message})
        ),
        "transformer_reference_provenance_status": "PASS",
        "construction_variant": bundle.construction_variant,
        "safe_feature_mode": bundle.safe_feature_mode,
        "safe_feature_transformer_id": bundle.safe_feature_transformer_id,
        "safe_feature_reference_pool_id": bundle.safe_feature_reference_pool_id,
        "semireal_split_used": bundle.semireal_split_used,
    }
    return summary, fold_frame, pd.DataFrame(predictions, columns=PREDICTION_COLUMNS), coefficient_frame


def _not_available_summary(
    bundle: DomainAuditBundle,
    analysis: str,
    message: str,
    *,
    parent_unique_stats: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": bundle.dataset,
        "source_run_id": bundle.source_run_id,
        "analysis": analysis,
        "is_primary": analysis == PRIMARY_ANALYSIS,
        "status": "NOT_AVAILABLE",
        "message": message,
        "n_experimental_doublets": np.nan,
        "n_semireal_heterotypic_doublets": np.nan,
        "n_semireal_before_parent_unique_filter": int(parent_unique_stats["n_semireal_before_parent_unique_filter"]),
        "n_semireal_after_parent_unique_filter": int(parent_unique_stats["n_semireal_after_parent_unique_filter"]),
        "parent_unique_retention_fraction": float(parent_unique_stats["parent_unique_retention_fraction"]),
        "n_unique_parents_retained": int(parent_unique_stats["n_unique_parents_retained"]),
        "parent_overlap_across_folds": np.nan,
        "n_features": np.nan,
        "n_folds": 0,
        "auroc_mean": np.nan,
        "auroc_std": np.nan,
        "auprc_mean": np.nan,
        "auprc_std": np.nan,
        "construction_variant": bundle.construction_variant,
        "safe_feature_mode": bundle.safe_feature_mode,
        "safe_feature_transformer_id": bundle.safe_feature_transformer_id,
        "safe_feature_reference_pool_id": bundle.safe_feature_reference_pool_id,
        "semireal_split_used": bundle.semireal_split_used,
    }


def _write_csv(frame: pd.DataFrame, output_path: Path, columns: Sequence[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.reindex(columns=list(columns)).to_csv(output_path, index=False)


def _write_json(payload: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _make_plots(summary: pd.DataFrame, coefficients: pd.DataFrame, output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plotting dependency.
        return []
    apply_manuscript_style()
    successful = summary.loc[summary["status"].eq("PASS")].copy()
    if successful.empty:
        return []
    paths: list[str] = []

    def save_figure(figure: Any, stem: str) -> None:
        png_path = output_dir / f"{stem}.png"
        pdf_path = output_dir / f"{stem}.pdf"
        figure.savefig(png_path, dpi=180)
        figure.savefig(pdf_path)
        paths.extend((png_path.name, pdf_path.name))

    labels = {
        "unmatched_heterotypic_safe_features": "Unmatched raw mechanisms",
        "matched_heterotypic_safe_features": "Matched raw mechanisms",
        "technical_covariates_only": "Technical covariates only",
    }
    fig, axis = plt.subplots(figsize=(8, 4.5))
    ordered = [name for name in labels if name in set(successful["analysis"])]
    positions = np.arange(len(ordered))
    values = [successful.loc[successful["analysis"].eq(name), "auroc_mean"].mean() for name in ordered]
    errors = [successful.loc[successful["analysis"].eq(name), "auroc_std"].mean() for name in ordered]
    axis.bar(positions, values, yerr=errors, color=["#90a4ae", "#1565c0", "#ef6c00"][: len(ordered)])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Cross-validated AUROC")
    axis.set_xticks(positions, [labels[name] for name in ordered], rotation=12, ha="right")
    axis.set_title("Experimental versus held-out semi-real heterotypic domains")
    fig.tight_layout()
    save_figure(fig, "domain_audit_auroc_comparison")
    plt.close(fig)

    primary = coefficients.loc[coefficients["analysis"].eq(PRIMARY_ANALYSIS)].nlargest(12, "mean_absolute_coefficient")
    if not primary.empty:
        fig, axis = plt.subplots(figsize=(8, max(3.5, 0.36 * len(primary))))
        axis.barh(primary["feature"], primary["mean_coefficient"], color="#1565c0")
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.invert_yaxis()
        axis.set_xlabel("Mean standardized logistic coefficient")
        axis.set_title("Top matched-domain audit coefficients")
        fig.tight_layout()
        save_figure(fig, "domain_audit_primary_coefficients")
        plt.close(fig)
    return paths


def _report(
    summary: pd.DataFrame,
    feature_list: Sequence[Mapping[str, Any]],
    coefficients: pd.DataFrame,
    bundles: Sequence[DomainAuditBundle],
    plot_paths: Sequence[str],
    output_path: Path,
) -> None:
    lines = [
        "# Semi-real versus experimental doublet domain audit",
        "",
        "Experimental doublet labels predominantly capture experimentally detectable cross-cell-state doublets, whereas reliable experimental homotypic labels are generally unavailable. Therefore, the primary domain audit compares experimental doublets with held-out semi-real heterotypic doublets.",
        "",
        "## Formal input contract",
        "",
        f"- Construction variant: `{FORMAL_CONSTRUCTION_VARIANT}` only.",
        f"- SafeFeature mode: `{FORMAL_SAFE_FEATURE_MODE}` with one shared fitted transformer and reference pool per run.",
        "- Synthetic parents must be absent from the fitted reference pool and clean-singlet training rows.",
        "- Experimental inputs are experimentally labelled doublets only; predicted doublets and real singlets are excluded.",
        "",
        "## Analyses",
        "",
        "1. `unmatched_heterotypic_safe_features`: raw mechanism features before matching.",
        "2. `matched_heterotypic_safe_features`: primary analysis after deterministic round-robin parent-unique semi-real selection, then matched 1:1 within canonical `duodose_cluster` and technical covariates.",
        "3. `technical_covariates_only`: log nCount, log nFeature, and mitochondrial fraction only.",
        "",
        "## Results",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- `{row['dataset']}` / `{row['analysis']}`: {row['status']}; "
            f"AUROC={row['auroc_mean']!s}, AUPRC={row['auprc_mean']!s}. {row['message']}"
        )
        if row["analysis"] == PRIMARY_ANALYSIS:
            lines.append(
                "  - Parent-unique held-out semi-real edges: "
                f"{row['n_semireal_after_parent_unique_filter']!s} retained from "
                f"{row['n_semireal_before_parent_unique_filter']!s}; "
                f"unique parents={row['n_unique_parents_retained']!s}; "
                f"validation-fold parent overlap={row['parent_overlap_across_folds']!s}."
            )
    lines.extend(["", "## Feature audit", ""])
    for entry in feature_list:
        counts = entry["counts"]
        lines.append(
            f"- `{entry['dataset']}` / `{entry['source_run_id']}`: {counts['candidate']} candidates; "
            f"{counts['included']} included, {counts['excluded']} excluded, {counts['unclassified']} unclassified."
        )
    primary_summary = summary.loc[(summary["analysis"].eq(PRIMARY_ANALYSIS)) & (summary["status"].eq("PASS"))]
    if not primary_summary.empty:
        lines.extend(["", "## Interpretation", ""])
        if (primary_summary["auroc_mean"] >= 0.75).any():
            lines.append(
                "- High primary AUROC indicates remaining feature-distribution separation; inspect the leading coefficients rather than treating this as a performance result."
            )
        top = coefficients.loc[coefficients["analysis"].eq(PRIMARY_ANALYSIS)].nlargest(8, "mean_absolute_coefficient")
        if not top.empty:
            lines.append("- Leading matched-analysis coefficients: " + ", ".join(top["feature"].astype(str).tolist()) + ".")
        technical = summary.loc[(summary["analysis"].eq("technical_covariates_only")) & (summary["status"].eq("PASS"))]
        if not technical.empty and technical["auroc_mean"].mean() > primary_summary["auroc_mean"].mean():
            lines.append(
                "- Technical covariates discriminate domains more strongly than the matched raw mechanism set; this suggests residual technical distribution shift rather than an interpretable biological mechanism difference.")
    lines.extend(["", "## Cached input provenance", ""])
    for bundle in bundles:
        lines.append(
            f"- `{bundle.dataset}` / `{bundle.source_run_id}`: split={bundle.semireal_split_used}, "
            f"construction={bundle.construction_variant}, safe_feature_mode={bundle.safe_feature_mode}, "
            f"transformer={bundle.safe_feature_transformer_id}, "
            f"reference_pool={bundle.safe_feature_reference_pool_id}."
        )
    if plot_paths:
        lines.extend(["", "## Figures", "", *[f"- `{path}`" for path in plot_paths]])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_audit_dirs(input_dirs: Sequence[str | Path]) -> list[Path]:
    resolved: list[Path] = []
    for raw in input_dirs:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise DomainAuditInputError(f"Domain-audit input path does not exist: '{path}'.")
        candidates = []
        if path.is_dir() and (path / "domain_audit_export_manifest.json").is_file():
            candidates.append(path)
        if path.is_dir() and (path / "domain_audit").is_dir():
            candidates.append(path / "domain_audit")
        if path.is_dir():
            candidates.extend(sorted(candidate.parent for candidate in path.rglob("domain_audit_export_manifest.json")))
        if path.is_file() and path.name == "domain_audit_export_manifest.json":
            candidates.append(path.parent)
        if not candidates:
            raise DomainAuditInputError(
                f"No domain_audit_export_manifest.json was found below '{path}'. "
                "Generate cached inputs with --export-domain-audit-only first."
            )
        resolved.extend(candidates)
    unique = sorted({candidate.resolve() for candidate in resolved}, key=lambda value: str(value).lower())
    return unique


def validate_domain_audit_bundle(input_dir: str | Path) -> DomainAuditBundle:
    """Validate one cached formal bundle without fitting an audit classifier."""

    return _load_bundle(input_dir)


def run_domain_audit(
    input_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    max_cells_per_domain: int | None = 2000,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> Mapping[str, Path]:
    """Run the constrained cached-input domain audit and write stable outputs."""

    audit_dirs = _resolve_audit_dirs(input_dirs)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundles = [_load_bundle(directory) for directory in audit_dirs]
    summary_rows: list[dict[str, Any]] = []
    fold_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    feature_audits: list[pd.DataFrame] = []
    feature_lists: list[Mapping[str, Any]] = []
    cluster_balances: list[pd.DataFrame] = []
    parent_unique_selections: list[pd.DataFrame] = []
    matching_exclusions: list[pd.DataFrame] = []

    for bundle in bundles:
        if progress_callback is not None:
            progress_callback({"event": "milestone", "message": f"{bundle.dataset}: provenance validation", "dataset": bundle.dataset})
        feature_names, feature_audit, feature_list = _audit_feature_list(bundle)
        feature_audits.append(feature_audit)
        feature_lists.append({"dataset": bundle.dataset, "source_run_id": bundle.source_run_id, **feature_list})
        if progress_callback is not None:
            progress_callback({"event": "milestone", "message": f"{bundle.dataset}: parent-unique filtering", "dataset": bundle.dataset})
        parent_unique_selection, parent_unique_stats = _select_parent_unique_heterotypic_edges(bundle)
        parent_unique_selections.append(parent_unique_selection)
        full = _combined_frame(bundle, feature_names)
        full = _apply_parent_unique_selection(full, parent_unique_stats)
        full = _deterministic_cap(full, max_cells_per_domain)
        raw_columns = [f"raw::{name}" for name in feature_names]
        technical_columns = [
            column
            for column in ("technical::log_ncount", "technical::log_nfeature", "technical::mito_fraction")
            if column in full.columns and full[column].notna().any()
        ]
        analyses: list[tuple[str, pd.DataFrame, Sequence[str]]] = [
            ("unmatched_heterotypic_safe_features", full, raw_columns),
        ]
        try:
            if progress_callback is not None:
                progress_callback({"event": "milestone", "message": f"{bundle.dataset}: matching experimental and semi-real cells", "dataset": bundle.dataset})
            matched, balance, exclusions = _matched_subsample(full, bundle)
            cluster_balances.append(balance)
            matching_exclusions.append(exclusions)
            analyses.append((PRIMARY_ANALYSIS, matched, raw_columns))
        except DomainAuditInputError as error:
            cluster_balances.append(_cluster_balance(full, full.iloc[0:0], bundle))
            summary_rows.append(
                _not_available_summary(
                    bundle,
                    PRIMARY_ANALYSIS,
                    str(error),
                    parent_unique_stats=parent_unique_stats,
                )
            )
        analyses.append(("technical_covariates_only", full, technical_columns))
        for analysis, frame, columns in analyses:
            if analysis == PRIMARY_ANALYSIS and any(
                row["dataset"] == bundle.dataset
                and row["source_run_id"] == bundle.source_run_id
                and row["analysis"] == PRIMARY_ANALYSIS
                for row in summary_rows
            ):
                continue
            required_technical_columns = {"technical::log_ncount", "technical::log_nfeature"}
            if analysis == "technical_covariates_only" and not required_technical_columns.issubset(columns):
                summary_rows.append(
                    _not_available_summary(
                        bundle,
                        analysis,
                        "technical_covariates_only requires log nCount and log nFeature; mitochondrial fraction is included when available.",
                        parent_unique_stats=parent_unique_stats,
                    )
                )
                continue
            try:
                summary, folds, predictions, coefficients = _evaluate_analysis(
                    frame,
                    columns,
                    analysis=analysis,
                    bundle=bundle,
                    parent_unique_stats=parent_unique_stats,
                    progress_callback=progress_callback,
                )
            except DomainAuditInputError as error:
                summary_rows.append(
                    _not_available_summary(
                        bundle,
                        analysis,
                        str(error),
                        parent_unique_stats=parent_unique_stats,
                    )
                )
                continue
            summary_rows.append(summary)
            fold_frames.append(folds)
            prediction_frames.append(predictions)
            coefficient_frames.append(coefficients)

    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    folds = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame(columns=FOLD_COLUMNS)
    predictions = (
        pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(columns=PREDICTION_COLUMNS)
    )
    coefficients = (
        pd.concat(coefficient_frames, ignore_index=True) if coefficient_frames else pd.DataFrame(columns=COEFFICIENT_COLUMNS)
    )
    feature_audit = (
        pd.concat(feature_audits, ignore_index=True) if feature_audits else pd.DataFrame(columns=FEATURE_AUDIT_COLUMNS)
    )
    cluster_balance = (
        pd.concat(cluster_balances, ignore_index=True)
        if cluster_balances
        else pd.DataFrame(columns=CLUSTER_BALANCE_COLUMNS)
    )
    parent_unique_selection = (
        pd.concat(parent_unique_selections, ignore_index=True)
        if parent_unique_selections
        else pd.DataFrame(columns=PARENT_UNIQUE_SELECTION_COLUMNS)
    )
    matching_exclusion = (
        pd.concat(matching_exclusions, ignore_index=True)
        if matching_exclusions
        else pd.DataFrame(columns=MATCHING_EXCLUSION_COLUMNS)
    )
    parent_unique_by_run: dict[tuple[str, str], Mapping[str, Any]] = {}
    for bundle in bundles:
        selected = parent_unique_selection.loc[
            parent_unique_selection["dataset"].eq(bundle.dataset)
            & parent_unique_selection["source_run_id"].eq(bundle.source_run_id)
            & parent_unique_selection["selected"].eq(True)
        ]
        total = parent_unique_selection.loc[
            parent_unique_selection["dataset"].eq(bundle.dataset)
            & parent_unique_selection["source_run_id"].eq(bundle.source_run_id)
        ]
        parent_unique_by_run[(bundle.dataset, bundle.source_run_id)] = {
            "n_before": int(len(total)),
            "n_after": int(len(selected)),
            "n_excluded_parent_reuse": int(total["selection_reason"].eq("excluded_parent_reuse").sum()),
            "n_excluded_no_experimental_cluster_capacity": int(
                total["selection_reason"].eq("excluded_no_experimental_cluster_capacity").sum()
            ),
            "n_unique_parents_retained": int(
                len(pd.unique(selected[["parent_1_id", "parent_2_id"]].astype(str).to_numpy().ravel()))
            ),
            "retained_counts_by_cluster_pair": {
                str(cluster_pair): int(count)
                for cluster_pair, count in selected["cluster_pair"].value_counts(sort=False).sort_index().items()
            },
        }
    if progress_callback is not None:
        progress_callback({"event": "milestone", "message": "writing domain-audit dataset outputs"})
    _write_csv(summary, output / "domain_audit_summary.csv", SUMMARY_COLUMNS)
    _write_csv(folds, output / "domain_audit_fold_metrics.csv", FOLD_COLUMNS)
    _write_csv(predictions, output / "domain_audit_predictions.csv", PREDICTION_COLUMNS)
    _write_csv(coefficients, output / "domain_audit_coefficients.csv", COEFFICIENT_COLUMNS)
    _write_csv(feature_audit, output / "domain_audit_feature_audit.csv", FEATURE_AUDIT_COLUMNS)
    _write_csv(cluster_balance, output / "domain_audit_cluster_balance.csv", CLUSTER_BALANCE_COLUMNS)
    _write_csv(
        parent_unique_selection,
        output / "domain_audit_parent_unique_selection.csv",
        PARENT_UNIQUE_SELECTION_COLUMNS,
    )
    _write_csv(
        matching_exclusion,
        output / "domain_audit_matching_exclusions.csv",
        MATCHING_EXCLUSION_COLUMNS,
    )
    _write_json(
        {
            "schema_version": SCHEMA_VERSION,
            "formal_construction_variant": FORMAL_CONSTRUCTION_VARIANT,
            "formal_safe_feature_mode": FORMAL_SAFE_FEATURE_MODE,
            "primary_analysis": PRIMARY_ANALYSIS,
            "feature_set": "audit_raw_mechanism_features",
            "feature_lists": list(feature_lists),
            "bundles": [
                {
                    "dataset": bundle.dataset,
                    "source_run_id": bundle.source_run_id,
                    "audit_dir": str(bundle.audit_dir),
                    "semireal_split_used": bundle.semireal_split_used,
                    "construction_variant": bundle.construction_variant,
                    "safe_feature_mode": bundle.safe_feature_mode,
                    "safe_feature_transformer_id": bundle.safe_feature_transformer_id,
                    "safe_feature_reference_pool_id": bundle.safe_feature_reference_pool_id,
                    "parent_removal_audit": dict(bundle.parent_removal_audit),
                    "parent_unique_selection": parent_unique_by_run[(bundle.dataset, bundle.source_run_id)],
                }
                for bundle in bundles
            ],
        },
        output / "domain_audit_feature_list.json",
    )
    _write_json(
        {
            "schema_version": SCHEMA_VERSION,
            "random_state": RANDOM_STATE,
            "n_folds": N_FOLDS,
            "max_cells_per_domain": max_cells_per_domain,
            "primary_analysis": PRIMARY_ANALYSIS,
            "formal_construction_variant": FORMAL_CONSTRUCTION_VARIANT,
            "formal_safe_feature_mode": FORMAL_SAFE_FEATURE_MODE,
            "rationale": (
                "Experimental doublet labels predominantly capture experimentally detectable cross-cell-state "
                "doublets, whereas reliable experimental homotypic labels are generally unavailable. Therefore, "
                "the primary domain audit compares experimental doublets with held-out semi-real heterotypic doublets."
            ),
            "analyses": [
                "unmatched_heterotypic_safe_features",
                PRIMARY_ANALYSIS,
                "technical_covariates_only",
            ],
            "bundles": [
                {
                    "dataset": bundle.dataset,
                    "source_run_id": bundle.source_run_id,
                    "construction_variant": bundle.construction_variant,
                    "safe_feature_mode": bundle.safe_feature_mode,
                    "safe_feature_transformer_id": bundle.safe_feature_transformer_id,
                    "safe_feature_reference_pool_id": bundle.safe_feature_reference_pool_id,
                    "parent_removal_audit": dict(bundle.parent_removal_audit),
                    "parent_unique_selection": parent_unique_by_run[(bundle.dataset, bundle.source_run_id)],
                }
                for bundle in bundles
            ],
            "parent_unique_selection_rule": (
                "Deterministic round-robin greedy maximal matching across heterotypic cluster-pair strata; "
                "no parent may occur in more than one retained held-out semi-real doublet."
            ),
        },
        output / "domain_audit_config.json",
    )
    plot_paths = _make_plots(summary, coefficients, output)
    _report(summary, feature_lists, coefficients, bundles, plot_paths, output / "domain_audit_report.md")
    return {
        "summary": output / "domain_audit_summary.csv",
        "feature_audit": output / "domain_audit_feature_audit.csv",
        "feature_list": output / "domain_audit_feature_list.json",
        "cluster_balance": output / "domain_audit_cluster_balance.csv",
        "parent_unique_selection": output / "domain_audit_parent_unique_selection.csv",
        "matching_exclusions": output / "domain_audit_matching_exclusions.csv",
        "report": output / "domain_audit_report.md",
    }
