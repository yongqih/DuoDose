"""Opt-in export of exact supervised SafeFeature inputs for domain audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from duodose.net import split_safe_feature_columns
from duodose.safe_feature_manifest import build_safe_feature_manifest


DOMAIN_AUDIT_INPUT_SCHEMA_VERSION = "domain_audit_inputs_v1"


def _write_table(frame: pd.DataFrame, outdir: Path, stem: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    parquet = outdir / f"{stem}.parquet"
    try:
        frame.to_parquet(parquet, index=False)
        return parquet
    except (ImportError, ModuleNotFoundError, ValueError):
        csv = outdir / f"{stem}.csv.gz"
        frame.to_csv(csv, index=False, compression="gzip")
        return csv


def _metadata(
    frame: pd.DataFrame,
    *,
    split: str,
    dataset: str | None,
    safe_feature_metadata: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["row_id"] = frame.index.astype(str)
    out["cell_id"] = frame.get("cell_id", pd.Series(frame.index.astype(str), index=frame.index)).astype(str)
    out["split"] = split
    fields = {
        "model_class_label": "true_label",
        "synthetic_subtype": "doublet_subtype",
        "experimental_singlet_doublet_label": "experimental_doublet",
        "dataset": "dataset",
        "semireal_cluster": "semireal_cluster",
        "benchmark_cluster": "benchmark_cluster",
        "duodose_cluster": "duodose_cluster",
        "nCount": "nCount",
        "nFeature": "nFeature",
        "parent_1_id": "parent_1_id",
        "parent_2_id": "parent_2_id",
        "count_construction_mode": "count_construction_mode",
        "parent_reference_mode": "parent_reference_mode",
        # Optional technical and biological covariates are preserved verbatim
        # when the active loader/export frame provides them.  The audit never
        # fabricates a missing covariate.
        "sample_id": "sample_id",
        "sample": "sample",
        "batch": "batch",
        "batch_id": "batch_id",
        "donor": "donor",
        "cell_type": "cell_type",
        "celltype": "celltype",
        "cluster": "cluster",
        "mitochondrial_fraction": "mitochondrial_fraction",
        "mito_fraction": "mito_fraction",
        "pct_counts_mt": "pct_counts_mt",
        "experimental_doublet_subtype": "experimental_doublet_subtype",
        "experimental_doublet_type": "experimental_doublet_type",
        "doublet_type": "doublet_type",
    }
    legacy_parent_fields = {"parent_1_id": "parent1_id", "parent_2_id": "parent2_id"}
    for output_name, source_name in fields.items():
        if source_name in frame:
            out[output_name] = frame[source_name].to_numpy()
        elif output_name in legacy_parent_fields and legacy_parent_fields[output_name] in frame:
            out[output_name] = frame[legacy_parent_fields[output_name]].to_numpy()
        elif output_name == "dataset" and dataset is not None:
            out[output_name] = dataset
        else:
            out[output_name] = np.nan
    metadata = dict(safe_feature_metadata or {})
    for name in (
        "safe_feature_mode",
        "safe_feature_transformer_id",
        "safe_feature_reference_pool_id",
        "safe_feature_reference_n_cells",
        "safe_feature_reference_seed",
    ):
        if name in frame:
            out[name] = frame[name].to_numpy()
        else:
            out[name] = metadata.get(name, np.nan)
    return out.reset_index(drop=True)


def _score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "row_id" not in out:
        out.insert(0, "row_id", out.index.astype(str))
    if "cell_id" not in out:
        out.insert(1, "cell_id", out.index.astype(str))
    return out.reset_index(drop=True)


def _assert_fully_real_export(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    diagnostics: Mapping[str, object],
) -> None:
    if set(metadata["split"].astype(str)) != {"fully_real"}:
        raise AssertionError("fully real audit metadata must contain only split='fully_real'")
    if "experimental_doublet" in frame:
        source_labels = pd.to_numeric(frame["experimental_doublet"], errors="coerce")
        if source_labels.notna().any() and metadata["experimental_singlet_doublet_label"].isna().any():
            raise AssertionError("fully real experimental labels were available but were not exported")
    if metadata["cell_id"].astype(str).str.contains("_semireal_doublet_", regex=False).any():
        raise AssertionError("fully real audit metadata contains synthetic doublet IDs")
    expected_rows = len(frame)
    if len(diagnostics["test_features"]) != expected_rows or len(diagnostics["transformed_test_features"]) != expected_rows:
        raise AssertionError("fully real feature matrices do not match the original real-cell row count")
    if diagnostics.get("imputer_fit_split") != "fit_only":
        raise AssertionError("fully real export is not using a fit-split imputer")
    if diagnostics.get("scaler_fit_split") not in {"fit_only", "not_used_for_random_forest"}:
        raise AssertionError("fully real export is not using the fit-split scaler")


def _raw_features_aligned_to_frame(frame: pd.DataFrame, features: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Require a stable one-to-one raw SafeFeature row mapping before export."""

    if frame.index.duplicated().any():
        raise ValueError(f"{name} score frame has duplicate row IDs")
    if "cell_id" not in frame:
        raise ValueError(f"{name} score frame is missing cell_id")
    cell_ids = frame["cell_id"].astype(str)
    if cell_ids.duplicated().any():
        raise ValueError(f"{name} score frame has duplicate cell IDs")
    if features.index.duplicated().any():
        raise ValueError(f"{name} raw SafeFeatures has duplicate row IDs")
    if not features.index.astype(str).equals(frame.index.astype(str)):
        raise ValueError(f"{name} raw SafeFeatures are not aligned to the score-frame row IDs")
    if features.columns.astype(str).duplicated().any():
        raise ValueError(f"{name} raw SafeFeatures has duplicate feature columns")
    if any(not pd.api.types.is_numeric_dtype(features[column]) for column in features.columns):
        raise ValueError(f"{name} raw SafeFeatures contains nonnumeric model columns")
    return features.copy().reindex(frame.index)


def _raw_features_with_cell_id(frame: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Serialize the already-aligned raw feature rows with their stable cell ID."""

    out = features.copy()
    out.insert(0, "cell_id", frame.loc[out.index, "cell_id"].astype(str).to_numpy())
    return out.reset_index(drop=True)


def _experimental_doublet_mask(frame: pd.DataFrame) -> np.ndarray:
    if "experimental_doublet" not in frame:
        raise ValueError("fully real score frame is missing experimental_doublet; predicted labels are not accepted")
    labels = pd.to_numeric(frame["experimental_doublet"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("fully real experimental_doublet must contain only observed binary 0/1 labels")
    mask = labels.eq(1).to_numpy()
    if not mask.any():
        raise ValueError("fully real score frame contains no experimentally annotated doublets")
    return mask


def _semireal_doublet_mask(frame: pd.DataFrame, *, split_name: str) -> np.ndarray:
    observed_split = frame.get("semireal_split", frame.get("split", pd.Series("", index=frame.index))).astype(str)
    expected = "test" if split_name == "test" else "validation"
    if not observed_split.eq(expected).all():
        raise ValueError(f"semi-real {split_name} score frame does not preserve semireal_split={expected!r}")
    labels = frame.get("true_label", frame.get("model_class_label", pd.Series("", index=frame.index))).astype(str)
    mask = labels.isin(["homotypic_doublet", "heterotypic_doublet"]).to_numpy()
    if not mask.any():
        raise ValueError(f"semi-real {split_name} score frame contains no constructed homotypic/heterotypic doublets")
    return mask


def _validate_parent_map(parent_map: pd.DataFrame, *, semireal_ids: pd.Index, split_name: str) -> dict[str, object]:
    required = {"synthetic_cell_id", "split", "parent_1_id", "parent_2_id"}
    missing = sorted(required - set(parent_map.columns))
    if missing:
        raise ValueError(f"semi-real parent map is missing required columns: {', '.join(missing)}")
    parent_map = parent_map.copy()
    parent_map["synthetic_cell_id"] = parent_map["synthetic_cell_id"].astype(str)
    if parent_map["synthetic_cell_id"].duplicated().any():
        raise ValueError("semi-real parent map contains duplicate synthetic_cell_id values")
    selected = parent_map.loc[parent_map["split"].astype(str).eq(split_name)].copy()
    selected = selected.set_index("synthetic_cell_id").reindex(semireal_ids.astype(str))
    if selected["parent_1_id"].isna().any() or selected["parent_2_id"].isna().any():
        raise ValueError(f"semi-real parent map is missing {split_name} parent rows for exported doublets")
    parent_1 = selected["parent_1_id"].astype(str).str.strip()
    parent_2 = selected["parent_2_id"].astype(str).str.strip()
    if parent_1.isin(["", "nan", "None", "<NA>"]).any() or parent_2.isin(["", "nan", "None", "<NA>"]).any():
        raise ValueError(f"semi-real parent map contains missing parent IDs in split {split_name}")
    pairs = pd.Series(["|".join(sorted((left, right))) for left, right in zip(parent_1, parent_2)], index=selected.index)
    return {
        "n_parent_map_rows": int(len(parent_map)),
        "n_exported_parent_rows": int(len(selected)),
        "n_duplicate_parent_pairs_in_exported_split": int(pairs.duplicated().sum()),
        "n_repeated_semireal_parent_pairs_in_exported_split": int(pairs.duplicated(keep=False).sum()),
    }


def export_domain_audit_inputs(
    *,
    output_dir: Path,
    dataset: str,
    seed: int,
    fully_real_score_frame: pd.DataFrame,
    fully_real_raw_features: pd.DataFrame,
    semireal_score_frame: pd.DataFrame,
    semireal_raw_features: pd.DataFrame,
    semireal_split_name: str,
    parent_map: pd.DataFrame,
    safe_feature_metadata: Mapping[str, object] | None = None,
    source_files: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Write a self-contained, raw-feature-only domain-audit input bundle.

    ``output_dir`` is the exact final ``domain_audit`` directory.  Dataset,
    seed, or construction components are intentionally never appended here;
    callers own that path construction and cannot trigger the historical
    duplicated-path bug.
    """

    if semireal_split_name not in {"test", "validation"}:
        raise ValueError("semireal_split_name must be 'test' or 'validation'")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fully_real_raw_features = _raw_features_aligned_to_frame(fully_real_score_frame, fully_real_raw_features, name="fully real")
    semireal_raw_features = _raw_features_aligned_to_frame(semireal_score_frame, semireal_raw_features, name=f"semi-real {semireal_split_name}")
    if list(fully_real_raw_features.columns.astype(str)) != list(semireal_raw_features.columns.astype(str)):
        raise ValueError("fully real and semi-real raw SafeFeature columns are inconsistent")
    if fully_real_score_frame["cell_id"].astype(str).str.contains("_semireal_doublet_", regex=False).any():
        raise ValueError("fully real score frame contains synthetic doublet IDs")

    real_mask = _experimental_doublet_mask(fully_real_score_frame)
    semi_mask = _semireal_doublet_mask(semireal_score_frame, split_name=semireal_split_name)
    real_frame = fully_real_score_frame.loc[real_mask].copy()
    real_features = fully_real_raw_features.loc[real_mask].copy()
    semi_frame = semireal_score_frame.loc[semi_mask].copy()
    semi_features = semireal_raw_features.loc[semi_mask].copy()
    if real_frame["cell_id"].astype(str).duplicated().any() or semi_frame["cell_id"].astype(str).duplicated().any():
        raise ValueError("selected domain-audit cells contain duplicate stable IDs")
    parent_audit = _validate_parent_map(
        parent_map,
        semireal_ids=pd.Index(semi_frame["cell_id"].astype(str)),
        split_name=semireal_split_name,
    )
    safe_feature_metadata = dict(safe_feature_metadata or {})
    safe_feature_metadata.setdefault("safe_feature_mode", "context_local")

    outputs = {
        "fully_real_score_frame": _write_table(_score_frame(real_frame), output_dir, "domain_audit_fully_real_score_frame"),
        "fully_real_safe_features_raw": _write_table(
            _raw_features_with_cell_id(real_frame, real_features),
            output_dir,
            "domain_audit_fully_real_safe_features_raw",
        ),
        "fully_real_metadata": _write_table(
            _metadata(real_frame, split="fully_real", dataset=dataset, safe_feature_metadata=safe_feature_metadata),
            output_dir,
            "domain_audit_fully_real_metadata",
        ),
        "semireal_score_frame": _write_table(_score_frame(semi_frame), output_dir, f"domain_audit_{semireal_split_name}_score_frame"),
        "semireal_safe_features_raw": _write_table(
            _raw_features_with_cell_id(semi_frame, semi_features),
            output_dir,
            f"domain_audit_{semireal_split_name}_safe_features_raw",
        ),
        "semireal_metadata": _write_table(
            _metadata(semi_frame, split="semi_real_test" if semireal_split_name == "test" else "validation", dataset=dataset, safe_feature_metadata=safe_feature_metadata),
            output_dir,
            f"domain_audit_{semireal_split_name}_metadata",
        ),
    }
    parent_map_path = output_dir / "semireal_parent_map.csv.gz"
    parent_map.reset_index(drop=True).to_csv(parent_map_path, index=False, compression="gzip")
    outputs["parent_map"] = parent_map_path
    manifest = build_safe_feature_manifest(fully_real_raw_features.columns.astype(str))
    manifest.to_csv(output_dir / "domain_audit_feature_manifest.csv", index=False)
    feature_names = list(map(str, fully_real_raw_features.columns))
    (output_dir / "safe_feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    included, excluded = split_safe_feature_columns(feature_names)
    (output_dir / "safe_feature_exclusion_audit.json").write_text(
        json.dumps({"included": included, "excluded": excluded}, indent=2), encoding="utf-8"
    )
    export_manifest = {
        "schema_version": DOMAIN_AUDIT_INPUT_SCHEMA_VERSION,
        "dataset": str(dataset),
        "seed": int(seed),
        "fully_real_domain_rows": int(len(real_frame)),
        "fully_real_total_rows_before_experimental_filter": int(len(fully_real_score_frame)),
        "semireal_doublet_rows": int(len(semi_frame)),
        "semireal_total_rows_before_doublet_filter": int(len(semireal_score_frame)),
        "semireal_export_split": semireal_split_name,
        "feature_count": int(len(feature_names)),
        "feature_columns_identical": True,
        "fully_real_selection": "experimental_doublet == 1 only; no predicted-doublet column accepted",
        "semireal_selection": "true_label in {homotypic_doublet, heterotypic_doublet} only",
        "source_files": {str(key): str(value) for key, value in dict(source_files or {}).items()},
        "artifacts": {name: Path(path).name for name, path in outputs.items()},
        "safe_feature_metadata": safe_feature_metadata,
        **parent_audit,
    }
    manifest_path = output_dir / "domain_audit_export_manifest.json"
    manifest_path.write_text(json.dumps(export_manifest, indent=2), encoding="utf-8")
    outputs["feature_manifest"] = output_dir / "domain_audit_feature_manifest.csv"
    outputs["export_manifest"] = manifest_path
    return {name: str(path) for name, path in outputs.items()}


def export_domain_audit(
    result: Mapping[str, object],
    *,
    output_dir: Path,
    fit_score_frame: pd.DataFrame,
    validation_score_frame: pd.DataFrame,
    fully_real_score_frame: pd.DataFrame,
    semi_real_test_score_frame: pd.DataFrame | None = None,
    dataset: str | None = None,
    fully_real_split_name: str = "fully_real",
) -> dict[str, str]:
    """Export raw/transformed matrices and fit-only preprocessing metadata."""

    diagnostics = result.get("feature_diagnostics", {})
    if not isinstance(diagnostics, Mapping) or not diagnostics:
        raise ValueError("domain audit requires successful model feature diagnostics")
    required = {
        "train_features",
        "validation_features",
        "test_features",
        "transformed_train_features",
        "transformed_validation_features",
        "transformed_test_features",
        "feature_names",
        "imputer_statistics",
        "scaler_mean",
        "scaler_scale",
        "categorical_feature_mapping",
    }
    missing = sorted(required - set(diagnostics))
    if missing:
        raise ValueError(f"domain audit diagnostics are incomplete: {', '.join(missing)}")

    safe_feature_metadata = diagnostics.get("safe_feature_transformer_metadata", {})
    if not isinstance(safe_feature_metadata, Mapping):
        raise TypeError("safe_feature_transformer_metadata must be a mapping when present")
    fully_real_metadata = _metadata(
        fully_real_score_frame,
        split=fully_real_split_name,
        dataset=dataset,
        safe_feature_metadata=safe_feature_metadata,
    )
    if safe_feature_metadata.get("safe_feature_mode") == "fitted_reference":
        expected_id = str(safe_feature_metadata.get("safe_feature_transformer_id", ""))
        all_metadata = [
            _metadata(fit_score_frame, split="fit", dataset=dataset, safe_feature_metadata=safe_feature_metadata),
            _metadata(validation_score_frame, split="validation", dataset=dataset, safe_feature_metadata=safe_feature_metadata),
            fully_real_metadata,
        ]
        if not expected_id or any(
            not metadata["safe_feature_transformer_id"].astype(str).eq(expected_id).all()
            for metadata in all_metadata
        ):
            raise AssertionError("all fitted-reference audit splits must use the same SafeFeature transformer ID")
    if fully_real_split_name == "fully_real":
        _assert_fully_real_export(fully_real_score_frame, fully_real_metadata, diagnostics)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "fit_score_frame": _write_table(_score_frame(fit_score_frame), output_dir, "domain_audit_fit_score_frame"),
        "validation_score_frame": _write_table(_score_frame(validation_score_frame), output_dir, "domain_audit_validation_score_frame"),
        "fully_real_score_frame": _write_table(_score_frame(fully_real_score_frame), output_dir, "domain_audit_fully_real_score_frame"),
        "fit_safe_features_raw": _write_table(diagnostics["train_features"].reset_index(drop=True), output_dir, "domain_audit_fit_safe_features_raw"),
        "validation_safe_features_raw": _write_table(diagnostics["validation_features"].reset_index(drop=True), output_dir, "domain_audit_validation_safe_features_raw"),
        "fully_real_safe_features_raw": _write_table(diagnostics["test_features"].reset_index(drop=True), output_dir, "domain_audit_fully_real_safe_features_raw"),
        "fit_safe_features_transformed": _write_table(diagnostics["transformed_train_features"].reset_index(drop=True), output_dir, "domain_audit_fit_safe_features_transformed"),
        "validation_safe_features_transformed": _write_table(diagnostics["transformed_validation_features"].reset_index(drop=True), output_dir, "domain_audit_validation_safe_features_transformed"),
        "fully_real_safe_features_transformed": _write_table(diagnostics["transformed_test_features"].reset_index(drop=True), output_dir, "domain_audit_fully_real_safe_features_transformed"),
        "fit_metadata": _write_table(
            _metadata(fit_score_frame, split="fit", dataset=dataset, safe_feature_metadata=safe_feature_metadata),
            output_dir,
            "domain_audit_fit_metadata",
        ),
        "validation_metadata": _write_table(
            _metadata(validation_score_frame, split="validation", dataset=dataset, safe_feature_metadata=safe_feature_metadata),
            output_dir,
            "domain_audit_validation_metadata",
        ),
        "fully_real_metadata": _write_table(fully_real_metadata, output_dir, "domain_audit_fully_real_metadata"),
    }
    semi_real_raw = diagnostics.get("semi_real_test_features")
    semi_real_transformed = diagnostics.get("transformed_semi_real_test_features")
    if semi_real_test_score_frame is not None and isinstance(semi_real_raw, pd.DataFrame) and isinstance(semi_real_transformed, pd.DataFrame):
        outputs.update(
            {
                "semi_real_test_score_frame": _write_table(
                    _score_frame(semi_real_test_score_frame), output_dir, "domain_audit_test_score_frame"
                ),
                "semi_real_test_safe_features_raw": _write_table(
                    semi_real_raw.reset_index(drop=True), output_dir, "domain_audit_test_safe_features_raw"
                ),
                "semi_real_test_safe_features_transformed": _write_table(
                    semi_real_transformed.reset_index(drop=True), output_dir, "domain_audit_test_safe_features_transformed"
                ),
                "semi_real_test_metadata": _write_table(
                    _metadata(
                        semi_real_test_score_frame,
                        split="semi_real_test",
                        dataset=dataset,
                        safe_feature_metadata=safe_feature_metadata,
                    ),
                    output_dir,
                    "domain_audit_test_metadata",
                ),
            }
        )
    feature_names = list(map(str, diagnostics["feature_names"]))
    included, excluded = split_safe_feature_columns(feature_names)
    (output_dir / "safe_feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    (output_dir / "safe_feature_exclusion_audit.json").write_text(
        json.dumps({"included": included, "excluded": excluded}, indent=2), encoding="utf-8"
    )
    alignment = {
        "feature_order_identical": all(
            list(diagnostics[key].columns) == feature_names
            for key in ("train_features", "validation_features", "test_features")
        ),
        "train_only_columns": list(diagnostics.get("train_only_columns", [])),
        "fully_real_only_columns": list(diagnostics.get("test_only_columns", [])),
        "imputer_fit_split": diagnostics.get("imputer_fit_split", "fit_only"),
        "scaler_fit_split": diagnostics.get("scaler_fit_split", "fit_only"),
    }
    (output_dir / "feature_alignment_audit.json").write_text(json.dumps(alignment, indent=2), encoding="utf-8")
    imputer = pd.Series(diagnostics["imputer_statistics"], name="imputer_statistic")
    imputer.rename_axis("feature").reset_index().to_csv(output_dir / "imputer_statistics.csv", index=False)
    scaler = pd.DataFrame({"mean": diagnostics["scaler_mean"], "scale": diagnostics["scaler_scale"]})
    scaler.index.name = "feature"
    scaler.reset_index().to_csv(output_dir / "scaler_mean_scale.csv", index=False)
    (output_dir / "categorical_feature_mapping.json").write_text(
        json.dumps(diagnostics["categorical_feature_mapping"], indent=2), encoding="utf-8"
    )
    return {name: str(path) for name, path in outputs.items()}
