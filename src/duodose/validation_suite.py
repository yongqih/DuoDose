"""Reusable contracts and audit primitives for the unified validation suite."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from .domain_audit_contract import PRIMARY_ANALYSIS, normalize_primary_analysis
import yaml
from anndata import AnnData, concat as concat_anndata
from scipy import sparse

from duodose.models.registry import BACKEND_SPECS, DEFAULT_DUODOSE_BACKEND
from duodose.net import NET_CLASS_LABELS, probabilities_to_scores, train_predict_diagnostic_model
from duodose.safe_feature_manifest import safe_feature_provenance
from duodose.semireal_bundle import canonical_parent_pair
from duodose.semireal_real_domain_audit import RAW_MECHANISM_FEATURE_ALLOWLIST


VALID_STATUSES = ("PASS", "FAIL", "NOT_APPLICABLE", "NOT_RUN", "INCOMPLETE")
HARD_AUDITS = {
    "parent_disjoint",
    "parent_removal",
    "frozen_reference",
    "same_cell_feature_invariance",
    "cell_order_invariance",
    "chunking_invariance",
    "transformer_serialization",
    "rf_serialization",
    "dl_serialization",
    "deterministic_rf_rerun",
    "feature_leakage",
    "probability_contract",
    "schema_contract",
}
PROHIBITED_FEATURE_TOKENS = (
    "true",
    "label",
    "split",
    "cell_id",
    "barcode",
    "parent",
    "dataset",
    "source",
    "path",
    "run_id",
    "seed",
    "probability",
    "scrublet",
    "hybrid",
    "rank",
    "percentile",
    "tail",
    "calibrat",
    "cluster_abundance",
    "sample_abundance",
)
REQUIRED_OUTPUTS = (
    "validation_suite_summary.csv",
    "validation_suite_checks.csv",
    "parent_disjoint_audit.csv",
    "parent_split_membership.csv.gz",
    "parent_maps.csv.gz",
    "same_cell_feature_invariance.csv",
    "cell_order_invariance.csv",
    "chunking_invariance.csv",
    "transformer_save_load_invariance.csv",
    "model_save_load_invariance.csv",
    "frozen_reference_audit.csv",
    "deterministic_rerun_audit.csv",
    "domain_feature_audit.csv",
    "subtype_permutation_results.csv",
    "subtype_permutation_summary.csv",
    "full_label_permutation_results.csv",
    "full_label_permutation_summary.csv",
    "probability_contract_audit.csv",
    "run_status_audit.csv",
    "metric_contract_audit.csv",
    "schema_audit.csv",
    "domain_audit_contract_check.csv",
    "validation_suite_config.json",
    "validation_suite_environment.json",
    "validation_suite_manifest.json",
    "validation_suite_report.md",
    "figures/maximum_invariance_error.png",
    "figures/maximum_invariance_error.pdf",
    "figures/parent_overlap_summary.png",
    "figures/parent_overlap_summary.pdf",
    "figures/subtype_permutation_observed_vs_null.png",
    "figures/subtype_permutation_observed_vs_null.pdf",
    "figures/full_label_permutation_auroc.png",
    "figures/full_label_permutation_auroc.pdf",
    "figures/full_label_permutation_auprc.png",
    "figures/full_label_permutation_auprc.pdf",
    "figures/probability_contract_status.png",
    "figures/probability_contract_status.pdf",
)


class ValidationSuiteError(RuntimeError):
    """Raised when a hard validation contract is violated."""


@dataclass(frozen=True)
class SuiteCheck:
    audit: str
    check: str
    status: str
    required: bool
    message: str = ""
    value: object = ""
    maximum_absolute_error: float = float("nan")

    def as_dict(self) -> dict[str, object]:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"invalid validation status {self.status!r}")
        return {
            "audit": self.audit,
            "check": self.check,
            "status": self.status,
            "required": bool(self.required),
            "message": self.message,
            "value": self.value,
            "maximum_absolute_error": self.maximum_absolute_error,
        }


def canonical_json_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_ids_hash(values: Iterable[object]) -> str:
    return canonical_json_hash([str(value) for value in values])


def load_validation_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"validation-suite config does not exist: {config_path}")
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("validation-suite config must contain a YAML mapping")
    contract = value.get("contract", {})
    expected = {
        "main_public_name": "DuoDose",
        "main_internal_name": "DuoDose-ML-CalibratedRF-SafeFeatures",
        "main_backend": "rf",
        "high_rna_negative_weight": 2.0,
        "default_backend": "rf",
        "ablation_public_name": "DuoDose-DL",
        "ablation_internal_name": "DuoDose-DL-ConditionalMultiTaskMLP-SafeFeatures",
        "ablation_backend": "dl",
        "construction_variant": "raw_sum_parents_removed",
        "safe_feature_mode": "fitted_reference",
        "parent_disjoint": True,
    }
    mismatches = [f"{key}={contract.get(key)!r}, expected {wanted!r}" for key, wanted in expected.items() if contract.get(key) != wanted]
    if mismatches:
        raise ValueError("invalid frozen validation contract: " + "; ".join(mismatches))
    if DEFAULT_DUODOSE_BACKEND != "rf" or set(BACKEND_SPECS) != {"rf", "dl"}:
        raise ValueError("public backend registry does not match the frozen RF/DL contract")
    tolerance = float(contract.get("numerical_tolerance", 0.0))
    if tolerance <= 0:
        raise ValueError("numerical_tolerance must be positive")
    for mode in ("smoke", "quick", "full"):
        if mode not in value.get("modes", {}):
            raise ValueError(f"validation-suite config is missing mode {mode!r}")
    value["_config_path"] = str(config_path)
    value["_config_hash"] = canonical_json_hash({key: item for key, item in value.items() if not key.startswith("_")})
    return value


def _atomic_target(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def atomic_write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_target(target)
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, target)
    return target


def atomic_write_json(path: str | Path, value: object) -> Path:
    return atomic_write_text(path, json.dumps(value, indent=2, default=str) + "\n")


def atomic_write_csv(path: str | Path, frame: pd.DataFrame) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_target(target)
    frame.to_csv(temporary, index=False, compression="gzip" if target.suffix == ".gz" else None)
    os.replace(temporary, target)
    return target


def stale_temporary_outputs(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    return sorted(path for path in root.rglob("*.tmp") if path.is_file()) if root.exists() else []


def verify_required_outputs(output_dir: str | Path) -> pd.DataFrame:
    root = Path(output_dir)
    rows = []
    for relative in REQUIRED_OUTPUTS:
        path = root / relative
        exists = path.is_file()
        size = path.stat().st_size if exists else 0
        rows.append(
            {
                "output": relative,
                "status": "PASS" if exists and size > 0 else "INCOMPLETE",
                "size_bytes": int(size),
                "reason": "" if exists and size > 0 else "required output is missing or empty",
            }
        )
    return pd.DataFrame(rows)


def completed_run_is_reusable(output_dir: str | Path, run_hash: str) -> bool:
    root = Path(output_dir)
    completion = root / ".validation_suite_complete.json"
    if stale_temporary_outputs(root) or not completion.is_file():
        return False
    try:
        payload = json.loads(completion.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("run_hash") == str(run_hash) and verify_required_outputs(root)["status"].eq("PASS").all()


def make_fixture_adata(n_cells: int, n_genes: int, *, seed: int = 0) -> AnnData:
    """Create deterministic count data with multiple expression programs."""

    if n_cells < 240 or n_genes < 20:
        raise ValueError("fixture must contain at least 240 cells and 20 genes")
    rng = np.random.default_rng(int(seed))
    n_clusters = 4
    clusters = np.arange(n_cells) % n_clusters
    baseline = rng.gamma(1.5, 1.0, size=(n_clusters, n_genes))
    width = max(5, n_genes // 12)
    for cluster in range(n_clusters):
        start = cluster * width
        baseline[cluster, start : start + width] += 4.0
    libraries = rng.lognormal(7.0, 0.28, size=n_cells)
    rates = baseline[clusters]
    rates = rates / rates.sum(axis=1, keepdims=True) * libraries[:, None]
    counts = sparse.csr_matrix(rng.poisson(rates).astype(np.int32))
    experimental = np.zeros(n_cells, dtype=int)
    experimental[np.arange(0, n_cells, 17)] = 1
    obs = pd.DataFrame(
        {"experimental_doublet": experimental, "fixture_cluster": clusters.astype(str)},
        index=[f"fixture_cell_{index:05d}" for index in range(n_cells)],
    )
    adata = AnnData(X=counts, obs=obs, var=pd.DataFrame(index=[f"gene_{index:04d}" for index in range(n_genes)]))
    adata.layers["counts"] = counts.copy()
    return adata


def protocol_override_for_mode(protocol: Mapping[str, Any], mode_config: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(protocol))
    if mode_config.get("use_frozen_protocol_sizes"):
        return value
    sizes = value["semi_real"]
    clustering = value["clustering"]
    for key in (
        "n_reference_singlets",
        "n_train_homotypic_doublets",
        "n_train_heterotypic_doublets",
        "n_test_homotypic_doublets",
        "n_test_heterotypic_doublets",
    ):
        sizes[key] = int(mode_config[key])
    sizes["minimum_eligible_singlets"] = min(int(sizes["minimum_eligible_singlets"]), int(mode_config["max_cells"]) // 3)
    for key in ("n_clusters", "min_cluster_size", "n_pcs", "n_neighbors"):
        clustering[key] = int(mode_config[key])
    return value


def deterministic_subset(adata: AnnData, max_cells: int | None, *, seed: int = 0) -> AnnData:
    if max_cells is None or adata.n_obs <= int(max_cells):
        return adata.copy()
    rng = np.random.default_rng(int(seed))
    selected = np.sort(rng.choice(adata.n_obs, size=int(max_cells), replace=False))
    return adata[selected, :].copy()


def _numeric_frame(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    selected = list(columns) if columns is not None else list(frame.columns)
    return frame.reindex(columns=selected).apply(pd.to_numeric, errors="coerce")


def compare_frames(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    audit: str,
    context: str,
    tolerance: float,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    common_index = baseline.index.intersection(candidate.index, sort=False)
    if len(common_index) != len(baseline.index) or len(common_index) != len(candidate.index):
        raise ValidationSuiteError(f"{audit}/{context}: row identifiers differ between compared frames")
    selected = list(columns) if columns is not None else list(baseline.columns)
    if selected != [column for column in candidate.columns if column in selected] or any(column not in candidate for column in selected):
        raise ValidationSuiteError(f"{audit}/{context}: feature or probability ordering differs")
    left = _numeric_frame(baseline.reindex(common_index), selected)
    right = _numeric_frame(candidate.reindex(common_index), selected)
    difference = (left - right).abs()
    rows = []
    for column in selected:
        values = difference[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        maximum = float(np.max(values[finite])) if finite.any() else float("inf")
        mean = float(np.mean(values[finite])) if finite.any() else float("inf")
        rows.append(
            {
                "audit": audit,
                "context": context,
                "feature_or_output": column,
                "n_compared": int(len(common_index)),
                "maximum_absolute_difference": maximum,
                "mean_absolute_difference": mean,
                "n_exceeding_tolerance": int(np.sum(values[finite] >= tolerance)) if finite.any() else int(len(values)),
                "tolerance": float(tolerance),
                "status": "PASS" if maximum < tolerance else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def _generated_expression_duplicate_count(run: Any, parent_map: pd.DataFrame) -> int:
    fingerprints: list[str] = []
    for split, adata in (("train", run.bundle.fit_adata), ("validation", run.bundle.val_adata), ("test", run.bundle.test_adata)):
        wanted = parent_map.loc[parent_map["split"].astype(str).eq(split), "synthetic_cell_id"].astype(str)
        positions = adata.obs_names.astype(str).get_indexer(wanted)
        if np.any(positions < 0):
            raise ValidationSuiteError(f"{split} parent map references generated cell IDs absent from its AnnData")
        matrix = adata.layers["counts"] if "counts" in adata.layers else adata.X
        matrix = sparse.csr_matrix(matrix)[positions, :]
        for row_index in range(matrix.shape[0]):
            row = matrix.getrow(row_index)
            digest = hashlib.sha256()
            digest.update(np.asarray(row.indices, dtype=np.int64).tobytes())
            digest.update(np.asarray(row.data, dtype=np.float64).tobytes())
            fingerprints.append(digest.hexdigest())
    return int(pd.Series(fingerprints, dtype=str).duplicated().sum())


def parent_pair_diagnostics(run: Any, parent_map: pd.DataFrame | None = None) -> tuple[dict[str, int], pd.DataFrame]:
    """Audit ordered, canonical, identifier, expression, and split duplicate semantics."""

    frame = (run.bundle.parent_map if parent_map is None else parent_map).copy()
    required = {"split", "synthetic_cell_id", "synthetic_subtype", "parent_1_id", "parent_2_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValidationSuiteError("parent map is missing columns: " + ", ".join(missing))
    canonical = frame.apply(
        lambda row: canonical_parent_pair(row["parent_1_id"], row["parent_2_id"]),
        axis=1,
        result_type="expand",
    )
    canonical.columns = ["canonical_parent_1", "canonical_parent_2"]
    frame["canonical_parent_1"] = canonical["canonical_parent_1"].to_numpy()
    frame["canonical_parent_2"] = canonical["canonical_parent_2"].to_numpy()
    ordered_columns = ["parent_1_id", "parent_2_id"]
    canonical_columns = ["canonical_parent_1", "canonical_parent_2"]
    canonical_groups = frame.groupby(canonical_columns, sort=False, dropna=False)
    reversed_equivalent = 0
    for _, group in canonical_groups:
        if len(group) > 1 and group[ordered_columns].drop_duplicates().shape[0] > 1:
            reversed_equivalent += len(group) - 1
    pair_split_counts = frame.groupby(canonical_columns, sort=False)["split"].nunique()
    parent_membership = pd.concat(
        [
            frame[["split", "parent_1_id"]].rename(columns={"parent_1_id": "parent_id"}),
            frame[["split", "parent_2_id"]].rename(columns={"parent_2_id": "parent_id"}),
        ],
        ignore_index=True,
    ).drop_duplicates()
    diagnostics = {
        "n_raw_ordered_duplicate_pairs": int(frame.duplicated(ordered_columns).sum()),
        "n_reversed_order_equivalent_pairs": int(reversed_equivalent),
        "n_canonical_duplicate_pairs": int(frame.duplicated(canonical_columns).sum()),
        "n_duplicate_parent_map_rows": int(frame.drop(columns=canonical_columns).duplicated().sum()),
        "n_duplicate_generated_cell_ids": int(frame["synthetic_cell_id"].astype(str).duplicated().sum()),
        "n_duplicate_generated_expression_profiles": _generated_expression_duplicate_count(run, frame),
        "n_cross_split_canonical_pair_overlaps": int(pair_split_counts.gt(1).sum()),
        "n_cross_split_parent_overlaps": int(parent_membership.groupby("parent_id")["split"].nunique().gt(1).sum()),
    }
    return diagnostics, frame


def audit_parent_disjoint(run: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    duplicate_diagnostics, parent_map = parent_pair_diagnostics(run)
    split_parents: dict[str, set[str]] = {}
    membership_rows = []
    for split in ("train", "validation", "test"):
        rows = parent_map.loc[parent_map["split"].astype(str).eq(split)]
        parents = set(rows["parent_1_id"].astype(str)) | set(rows["parent_2_id"].astype(str))
        split_parents[split] = parents
        membership_rows.extend({"parent_id": parent, "split": split} for parent in sorted(parents))
    reference_ids = set(map(str, run.transformer.reference_cell_ids_))
    retained_singlets: set[str] = set()
    split_counts: dict[str, dict[str, int]] = {}
    for split, adata in (("train", run.bundle.fit_adata), ("validation", run.bundle.val_adata), ("test", run.bundle.test_adata)):
        labels = adata.obs["true_label"].astype(str)
        origin = adata.obs.get("semireal_origin", pd.Series("", index=adata.obs_names)).astype(str)
        retained_singlets |= set(adata.obs_names[origin.isin({"observed_background", "real_labeled_singlet"})].astype(str))
        split_counts[split] = {
            "singlet": int(labels.isin(["clean", "singlet", "high_RNA_singlet"]).sum()),
            "homotypic_doublet": int(labels.eq("homotypic_doublet").sum()),
            "heterotypic_doublet": int(labels.eq("heterotypic_doublet").sum()),
        }
    all_parents = set().union(*split_parents.values())
    overlaps = {
        "train_validation_parent_overlap": len(split_parents["train"] & split_parents["validation"]),
        "train_test_parent_overlap": len(split_parents["train"] & split_parents["test"]),
        "validation_test_parent_overlap": len(split_parents["validation"] & split_parents["test"]),
        "generated_parent_retained_singlet_overlap": len(all_parents & retained_singlets),
        "reference_parent_overlap": len(all_parents & reference_ids),
        "reference_validation_cell_overlap": len(reference_ids & set(map(str, run.bundle.val_adata.obs_names))),
        "reference_test_cell_overlap": len(reference_ids & set(map(str, run.bundle.test_adata.obs_names))),
        "parents_marked_in_reference": int(parent_map.get("parent_1_in_reference", False).astype(bool).sum() + parent_map.get("parent_2_in_reference", False).astype(bool).sum()),
        **duplicate_diagnostics,
    }
    rows = []
    diagnostic_only = {
        "n_raw_ordered_duplicate_pairs",
        "n_reversed_order_equivalent_pairs",
        "n_duplicate_generated_expression_profiles",
    }
    for check, value in overlaps.items():
        is_diagnostic = check in diagnostic_only
        rows.append(
            {
                "check": check,
                "value": int(value),
                "required_value": "reported; not a hard parent-disjoint criterion" if is_diagnostic else 0,
                "status": "PASS" if is_diagnostic or int(value) == 0 else "FAIL",
                "message": "diagnostic duplicate-pair count" if is_diagnostic else "",
            }
        )
    for split, counts in split_counts.items():
        for label, value in counts.items():
            rows.append({"check": f"{split}_{label}_count", "value": int(value), "required_value": ">0", "status": "PASS" if value > 0 else "FAIL", "message": ""})
    for split in ("train", "validation", "test"):
        rows.append({"check": f"{split}_parent_count", "value": len(split_parents[split]), "required_value": ">0", "status": "PASS" if split_parents[split] else "FAIL", "message": ""})
    cluster_pair = parent_map.get("cluster_pair", parent_map.get("parent_cluster_pair", pd.Series("", index=parent_map.index))).astype(str)
    rows.append({"check": "cluster_pair_coverage", "value": int(cluster_pair.nunique()), "required_value": ">0", "status": "PASS" if cluster_pair.nunique() > 0 else "FAIL", "message": ""})
    return pd.DataFrame(rows), pd.DataFrame(membership_rows), parent_map


def transform_and_predict(run: Any, adata: AnnData, *, dataset_id: str = "validation_context") -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    scores = run.transformer.transform(adata, dataset_id=dataset_id, random_state=int(run.seed))
    matrix = run.transformer.build_model_matrix(scores)
    probabilities = {method: backend.predict_probabilities(scores) for method, backend in run.fitted_backends.items()}
    return matrix, probabilities


def transform_predict_chunks(run: Any, adata: AnnData, chunk_size: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    matrices = []
    probability_parts: dict[str, list[pd.DataFrame]] = {method: [] for method in run.fitted_backends}
    for start in range(0, adata.n_obs, int(chunk_size)):
        matrix, probabilities = transform_and_predict(run, adata[start : start + int(chunk_size), :].copy())
        matrices.append(matrix)
        for method, frame in probabilities.items():
            probability_parts[method].append(frame)
    return pd.concat(matrices).reindex(adata.obs_names), {method: pd.concat(parts).reindex(adata.obs_names) for method, parts in probability_parts.items()}


def audit_same_cell_features(run: Any, tolerance: float) -> pd.DataFrame:
    query = run.bundle.test_adata.copy()
    baseline = run.transformer.transform(query, dataset_id="same_cell")
    context = concat_anndata([run.bundle.val_adata[: min(25, run.bundle.val_adata.n_obs), :].copy(), query], join="inner", index_unique=None)
    contextual = run.transformer.transform(context, dataset_id="same_cell").reindex(query.obs_names)
    columns = [column for column in baseline.columns if column in RAW_MECHANISM_FEATURE_ALLOWLIST and column in contextual]
    if not columns:
        raise ValidationSuiteError("same-cell audit found no canonical raw mechanism features")
    return compare_frames(baseline, contextual, audit="same_cell_feature_invariance", context="standalone_vs_mixed_query", tolerance=tolerance, columns=columns)


def audit_order_invariance(run: Any, tolerance: float) -> pd.DataFrame:
    query = run.bundle.test_adata.copy()
    baseline_matrix, baseline_probabilities = transform_and_predict(run, query)
    rng = np.random.default_rng(int(run.seed))
    orders = {"reversed": np.arange(query.n_obs)[::-1], "seeded_random": rng.permutation(query.n_obs)}
    frames = []
    for name, order in orders.items():
        matrix, probabilities = transform_and_predict(run, query[order, :].copy())
        frames.append(compare_frames(baseline_matrix, matrix.reindex(query.obs_names), audit="cell_order_invariance", context=f"raw_safe_features_{name}", tolerance=tolerance))
        for method, baseline in baseline_probabilities.items():
            frames.append(compare_frames(baseline, probabilities[method].reindex(query.obs_names), audit="cell_order_invariance", context=f"{method}_{name}", tolerance=tolerance))
    return pd.concat(frames, ignore_index=True)


def audit_chunking_invariance(run: Any, chunk_sizes: Sequence[int], tolerance: float) -> pd.DataFrame:
    query = run.bundle.test_adata.copy()
    baseline_matrix, baseline_probabilities = transform_and_predict(run, query)
    frames = []
    for chunk_size in chunk_sizes:
        matrix, probabilities = transform_predict_chunks(run, query, int(chunk_size))
        frames.append(compare_frames(baseline_matrix, matrix, audit="chunking_invariance", context=f"raw_safe_features_chunk_{chunk_size}", tolerance=tolerance))
        for method, baseline in baseline_probabilities.items():
            frames.append(compare_frames(baseline, probabilities[method], audit="chunking_invariance", context=f"{method}_chunk_{chunk_size}", tolerance=tolerance))
    return pd.concat(frames, ignore_index=True)


def audit_transformer_serialization(run: Any, output_dir: str | Path, tolerance: float) -> pd.DataFrame:
    query = run.bundle.test_adata.copy()
    baseline_scores = run.transformer.transform(query, dataset_id="serialization")
    baseline = run.transformer.build_model_matrix(baseline_scores)
    directory = Path(output_dir) / "transformer"
    run.transformer.save(directory)
    loaded = type(run.transformer).load(directory)
    candidate = loaded.build_model_matrix(loaded.transform(query, dataset_id="serialization"))
    comparison = compare_frames(baseline, candidate, audit="transformer_serialization", context="before_vs_after", tolerance=tolerance)
    state_rows = [
        ("feature_names_identical", list(run.transformer.model_feature_columns_) == list(loaded.model_feature_columns_)),
        ("transformer_fingerprint_identical", run.transformer.transformer_id_ == loaded.transformer_id_),
        ("reference_pool_fingerprint_identical", run.transformer.reference_pool_id_ == loaded.reference_pool_id_),
        ("input_schema_identical", list(run.transformer.selected_genes_) == list(loaded.selected_genes_)),
        ("centroids_preserved", np.allclose(run.transformer.cluster_model_.cluster_centers_, loaded.cluster_model_.cluster_centers_, atol=0, rtol=0)),
        ("scaling_parameters_preserved", np.allclose(run.transformer.embedding_scaler_.scale_, loaded.embedding_scaler_.scale_, atol=0, rtol=0)),
    ]
    state = pd.DataFrame(
        {
            "audit": "transformer_serialization",
            "context": [name for name, _ in state_rows],
            "feature_or_output": "state",
            "n_compared": 1,
            "maximum_absolute_difference": [0.0 if passed else float("inf") for _, passed in state_rows],
            "mean_absolute_difference": [0.0 if passed else float("inf") for _, passed in state_rows],
            "n_exceeding_tolerance": [0 if passed else 1 for _, passed in state_rows],
            "tolerance": tolerance,
            "status": ["PASS" if passed else "FAIL" for _, passed in state_rows],
        }
    )
    return pd.concat([comparison, state], ignore_index=True)


def audit_model_serialization(run: Any, output_dir: str | Path, tolerance: float) -> pd.DataFrame:
    query_scores = run.transformer.transform(run.bundle.test_adata, dataset_id="model_serialization")
    frames = []
    directory = Path(output_dir) / "models"
    directory.mkdir(parents=True, exist_ok=True)
    for method, backend in run.fitted_backends.items():
        baseline = backend.predict_probabilities(query_scores)
        path = directory / f"{method.replace('-', '_')}.joblib"
        joblib.dump(backend, path)
        loaded = joblib.load(path)
        candidate = loaded.predict_probabilities(query_scores)
        frames.append(compare_frames(baseline, candidate, audit="model_serialization", context=method, tolerance=tolerance))
    return pd.concat(frames, ignore_index=True)


def audit_frozen_reference(run: Any) -> pd.DataFrame:
    reference = set(map(str, run.transformer.reference_cell_ids_))
    experimental = set(run.original_adata.obs_names[run.original_adata.obs["experimental_doublet"].astype(int).eq(1)].astype(str))
    parent_map = run.bundle.parent_map
    heldout_doublets = set(parent_map.loc[parent_map["split"].astype(str).isin(["validation", "test"]), "synthetic_cell_id"].astype(str))
    checks = {
        "validation_cells_in_reference": len(reference & set(map(str, run.bundle.val_adata.obs_names))),
        "test_cells_in_reference": len(reference & set(map(str, run.bundle.test_adata.obs_names))),
        "heldout_semireal_doublets_in_reference": len(reference & heldout_doublets),
        "experimental_doublets_in_reference": len(reference & experimental),
    }
    rows = []
    for key, value in checks.items():
        if key == "experimental_doublets_in_reference":
            rows.append(
                {
                    "check": key,
                    "value": int(value),
                    "status": "NOT_APPLICABLE",
                    "message": "experimental labels are evaluation-only and cannot be used to select the fitted reference pool",
                }
            )
        else:
            rows.append({"check": key, "value": int(value), "status": "PASS" if value == 0 else "FAIL", "message": ""})
    metadata = run.transformer.metadata()
    for split in ("fit", "validation", "test", "fully_real"):
        frame = getattr(run, f"{split}_scores" if split != "fit" else "fit_scores")
        transformer_ids = set(frame["safe_feature_transformer_id"].astype(str))
        reference_ids = set(frame["safe_feature_reference_pool_id"].astype(str))
        passed = transformer_ids == {metadata["safe_feature_transformer_id"]} and reference_ids == {metadata["safe_feature_reference_pool_id"]}
        rows.append({"check": f"{split}_provenance_identical", "value": f"{transformer_ids}|{reference_ids}", "status": "PASS" if passed else "FAIL", "message": "" if passed else "transformer/reference fingerprint mismatch"})
    rows.extend(
        [
            {"check": "reference_cell_ids_fingerprint", "value": stable_ids_hash(run.transformer.reference_cell_ids_), "status": "PASS", "message": ""},
            {"check": "reference_schema_fingerprint", "value": stable_ids_hash(run.transformer.selected_genes_), "status": "PASS", "message": ""},
            {"check": "transformer_state_fingerprint", "value": run.transformer.transformer_id_, "status": "PASS", "message": ""},
            {"check": "feature_manifest_fingerprint", "value": canonical_json_hash(run.transformer.manifest_.to_dict("records")), "status": "PASS", "message": ""},
        ]
    )
    return pd.DataFrame(rows)


def validate_feature_names(feature_names: Sequence[str], allowlist: Sequence[str]) -> pd.DataFrame:
    allowed = set(map(str, allowlist))
    rows = []
    for feature in feature_names:
        feature = str(feature)
        lower = feature.lower()
        reasons = [token for token in PROHIBITED_FEATURE_TOKENS if token in lower]
        if lower.startswith("duodose_cluster_") or lower.startswith("sample_id_"):
            reasons.append("prohibited categorical one-hot")
        if feature not in allowed:
            reasons.append("not in frozen exact allowlist")
        provenance = safe_feature_provenance(feature)
        if not provenance["provenance_complete"]:
            reasons.append("missing semantic provenance metadata")
        if provenance["uses_truth_labels"] is not False:
            reasons.append("truth-label provenance is not explicitly false")
        if provenance["uses_model_output"] is not False:
            reasons.append("model-output provenance is not explicitly false")
        if provenance["uses_outcome_calibration"] is not False:
            reasons.append("outcome-calibration provenance is not explicitly false")
        if provenance["uses_dataset_rank"] is not False:
            reasons.append("dataset-rank provenance is not explicitly false")
        if not provenance["allowed_in_rf"] or not provenance["allowed_in_dl"]:
            reasons.append("semantic provenance prohibits model input")
        reasons = list(dict.fromkeys(reasons))
        rows.append(
            {
                **provenance,
                "feature": feature,
                "included": True,
                "exclusion_reason": "; ".join(reasons),
                "status": "PASS" if not reasons else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def validate_probability_frame(frame: pd.DataFrame, method: str, tolerance: float) -> pd.DataFrame:
    rows = []
    expected = list(NET_CLASS_LABELS)
    order_ok = list(frame.columns) == expected
    rows.append({"method": method, "check": "stable_output_order", "value": ",".join(frame.columns), "maximum_error": 0.0, "status": "PASS" if order_ok else "FAIL", "message": "" if order_ok else f"expected {expected}"})
    numeric = frame.reindex(columns=expected).apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    finite_ok = bool(np.isfinite(values).all())
    bounds_error = float(max(np.nanmax(np.maximum(-values, 0.0)), np.nanmax(np.maximum(values - 1.0, 0.0)))) if values.size else float("inf")
    sum_error = float(np.nanmax(np.abs(np.nansum(values, axis=1) - 1.0))) if len(values) else float("inf")
    rows.extend(
        [
            {"method": method, "check": "finite_values", "value": int(np.isfinite(values).sum()), "maximum_error": 0.0 if finite_ok else float("inf"), "status": "PASS" if finite_ok else "FAIL", "message": ""},
            {"method": method, "check": "probability_bounds", "value": "[0,1]", "maximum_error": bounds_error, "status": "PASS" if bounds_error <= tolerance else "FAIL", "message": ""},
            {"method": method, "check": "declared_class_normalization", "value": "sum(clean,high_RNA,homotypic,heterotypic)=1", "maximum_error": sum_error, "status": "PASS" if sum_error <= tolerance else "FAIL", "message": ""},
        ]
    )
    overall, hom, hetero = probabilities_to_scores(numeric)
    denominator = hom + hetero
    valid = denominator > tolerance
    q_sum_error = float(np.max(np.abs((hom[valid] / denominator[valid] + hetero[valid] / denominator[valid]) - 1.0))) if valid.any() else 0.0
    rows.append({"method": method, "check": "conditional_subtype_normalization", "value": int(valid.sum()), "maximum_error": q_sum_error, "status": "PASS" if q_sum_error <= tolerance else "FAIL", "message": "condition applies only where P(doublet)>0"})
    alignment_ok = frame.index.is_unique and overall.index.equals(frame.index)
    rows.append({"method": method, "check": "row_cell_alignment", "value": int(len(frame)), "maximum_error": 0.0, "status": "PASS" if alignment_ok else "FAIL", "message": ""})
    return pd.DataFrame(rows)


def metric_contract_table() -> pd.DataFrame:
    rows = [
        ("AUROC", "homotypic_doublet or heterotypic_doublet", "clean/singlet/high_RNA_singlet", "all controlled test rows", "none", "not applicable"),
        ("overall_AUPRC", "homotypic_doublet or heterotypic_doublet", "clean/singlet/high_RNA_singlet", "all controlled test rows", "none", "not applicable"),
        ("homotypic_AUPRC", "homotypic_doublet", "clean/singlet/high_RNA_singlet", "homotypic plus controlled negatives", "heterotypic_doublet", "not applicable"),
        ("heterotypic_AUPRC", "heterotypic_doublet", "clean/singlet/high_RNA_singlet", "heterotypic plus controlled negatives", "homotypic_doublet", "not applicable"),
        ("macro_subtype_AUPRC", "mean of subtype AUPRCs", "as above", "controlled subtype evaluations", "opposite subtype per component", "not applicable"),
        ("homotypic_vs_high_RNA_singlet_AUPRC", "homotypic_doublet", "high_RNA_singlet", "those two controlled classes", "all other classes", "not applicable"),
        ("high_RNA_singlet_FPR", "selected high_RNA_singlet", "unselected high_RNA_singlet", "smallest deterministic overall-score prefix reaching at least 50% homotypic recall", "non-high-RNA denominator rows", "method-specific minimum K at matched 50% homotypic recall"),
        ("high_RNA_singlet_FPR_at_fixed_20pct_candidate_budget", "selected high_RNA_singlet", "unselected high_RNA_singlet", "top 20% of each test set", "non-high-RNA denominator rows", "K=round(0.20*N_test)"),
        ("high_RNA_singlet_FPR_at_true_doublet_budget", "selected high_RNA_singlet", "unselected high_RNA_singlet", "historical top overall-score budget equal to true doublet count", "non-high-RNA denominator rows", "K=true controlled doublet count"),
        ("precision_at_K", "selected true doublet", "selected non-doublet", "all controlled test rows", "none", "K=true controlled doublet count; deterministic cell-ID tie break"),
        ("recall_at_K", "selected true doublet", "unselected true doublet", "all controlled test rows", "none", "K=true controlled doublet count; deterministic cell-ID tie break"),
        ("real_doublet_enriched_AUROC_AUPRC", "experimentally annotated doublet", "annotated non-doublet", "doublet-enriched dataset", "none; n_excluded=0", "not applicable"),
    ]
    return pd.DataFrame(rows, columns=["metric", "positive_class", "negative_class", "evaluated_subset", "exclusions", "K_definition"]).assign(undefined_behavior="NOT_AVAILABLE/NaN with explicit reason when a class is absent", implementation="reproducibility.lib.common.controlled_metric_row")


def schema_audit(adata: AnnData, run: Any) -> pd.DataFrame:
    matrix = adata.layers["counts"] if "counts" in adata.layers else adata.X
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
    checks = [
        ("count_matrix_orientation", matrix.shape == (adata.n_obs, adata.n_vars), f"{matrix.shape}"),
        ("unique_cell_ids", not adata.obs_names.duplicated().any(), int(adata.obs_names.duplicated().sum())),
        ("unique_gene_ids", not adata.var_names.duplicated().any(), int(adata.var_names.duplicated().sum())),
        ("metadata_alignment", adata.obs.index.equals(adata.obs_names), int(len(adata.obs))),
        ("label_alignment", "experimental_doublet" in adata.obs and len(adata.obs["experimental_doublet"]) == adata.n_obs, "experimental_doublet"),
        ("nonnegative_finite_counts", bool(np.isfinite(values).all() and np.min(values, initial=0) >= 0), float(np.min(values, initial=0))),
        ("stable_feature_order", list(run.test_features.columns) == list(run.transformer.model_feature_columns_), len(run.test_features.columns)),
        ("stable_probability_order", all(list(frame.columns) == list(NET_CLASS_LABELS) for frame in run.method_probabilities_test.values()), ",".join(NET_CLASS_LABELS)),
        ("no_accidental_index_column", all(not str(column).lower().startswith("unnamed:") for column in run.test_features.columns), ""),
    ]
    return pd.DataFrame({"check": [item[0] for item in checks], "value": [item[2] for item in checks], "status": ["PASS" if item[1] else "FAIL" for item in checks], "message": ["" if item[1] else "schema contract violated" for item in checks]})


def audit_run_status(final_results_dir: str | Path, protocol: Mapping[str, Any]) -> pd.DataFrame:
    root = Path(final_results_dir)
    rows = []
    datasets = list(protocol["datasets"]["real_doublet_enriched"])
    seeds = list(protocol["seeds"]["controlled_benchmark"])
    for dataset in datasets:
        for seed in seeds:
            internal_path = root / "controlled" / dataset / f"seed_{seed}"
            internal_metrics_path = internal_path / "controlled_metrics.csv"
            internal_metrics = pd.read_csv(internal_metrics_path) if internal_metrics_path.is_file() else pd.DataFrame()
            manifest_path = internal_path / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
            for method in ("DuoDose", "DuoDose-DL"):
                match = internal_metrics.loc[internal_metrics.get("method", pd.Series(dtype=str)).astype(str).eq(method)] if not internal_metrics.empty else pd.DataFrame()
                reason = ""
                status = "PASS"
                if len(match) != 1:
                    status, reason = "INCOMPLETE", "missing or duplicate internal method row"
                elif str(match.iloc[0].get("status", "")).lower() != "success":
                    status, reason = "INCOMPLETE", str(match.iloc[0].get("message", "non-success internal run"))
                elif not manifest:
                    status, reason = "INCOMPLETE", "run manifest missing"
                elif not manifest.get("git_commit"):
                    status, reason = "INCOMPLETE", "code commit missing from run manifest"
                elif not manifest.get("protocol_config_sha256"):
                    status, reason = "INCOMPLETE", "explicit configuration hash missing from run manifest"
                raw_status = str(match.iloc[0].get("status", "missing")) if len(match) == 1 else "missing"
                execution_status = "COMPLETED" if raw_status.lower() == "success" else "FAILED" if raw_status.lower() in {"failed", "error"} else "SKIPPED" if raw_status.lower() == "skipped" else "UNAVAILABLE" if raw_status.lower() in {"unavailable", "not_available"} else "INCOMPLETE"
                formal_status = execution_status if execution_status != "COMPLETED" else "COMPLETED" if status == "PASS" else "INCOMPLETE"
                rows.append({"workflow": "controlled_benchmark", "dataset": dataset, "seed": seed, "method": method, "method_family": "internal", "execution_status": execution_status, "raw_execution_status": raw_status, "status": formal_status, "reason": reason, "config_hash_present": bool(manifest.get("protocol_config_sha256")), "seed_present": manifest.get("seed") is not None, "input_manifest_present": bool(manifest.get("input")), "code_version_present": bool(manifest.get("git_commit")), "result_path": str(internal_path.relative_to(root))})

            external_path = root / "external" / dataset / f"seed_{seed}"
            status_path = external_path / "external_method_status.csv"
            statuses = pd.read_csv(status_path) if status_path.is_file() else pd.DataFrame()
            external_manifest_path = external_path / "run_manifest.json"
            external_manifest = json.loads(external_manifest_path.read_text(encoding="utf-8")) if external_manifest_path.is_file() else {}
            for method in ("Scrublet", "scDblFinder", "DoubletFinder", "scds"):
                match = statuses.loc[statuses.get("method", pd.Series(dtype=str)).astype(str).eq(method)] if not statuses.empty else pd.DataFrame()
                reason = ""
                status = "PASS"
                if len(match) != 1:
                    status, reason = "INCOMPLETE", "external benchmark was interrupted before this method produced one status row"
                elif str(match.iloc[0].get("status", "")).lower() != "success":
                    status = "INCOMPLETE"
                    reason = str(match.iloc[0].get("message", "external method unavailable or failed")) or "non-success external status lacks reason"
                elif not external_manifest:
                    status, reason = "INCOMPLETE", "run manifest missing"
                elif not external_manifest.get("protocol_config_sha256"):
                    status, reason = "INCOMPLETE", "explicit configuration hash missing from run manifest"
                raw_status = str(match.iloc[0].get("status", "missing")) if len(match) == 1 else "missing"
                execution_status = "COMPLETED" if raw_status.lower() == "success" else "FAILED" if raw_status.lower() in {"failed", "error"} else "SKIPPED" if raw_status.lower() == "skipped" else "UNAVAILABLE" if raw_status.lower() in {"unavailable", "not_available"} else "INCOMPLETE"
                formal_status = execution_status if execution_status != "COMPLETED" else "COMPLETED" if status == "PASS" else "INCOMPLETE"
                rows.append({"workflow": "controlled_benchmark", "dataset": dataset, "seed": seed, "method": method, "method_family": "external", "execution_status": execution_status, "raw_execution_status": raw_status, "status": formal_status, "reason": reason, "config_hash_present": bool(external_manifest.get("protocol_config_sha256")), "seed_present": external_manifest.get("seed") is not None, "input_manifest_present": bool(external_manifest.get("input")), "code_version_present": bool(external_manifest.get("git_commit")), "result_path": str(external_path.relative_to(root))})

    application_methods = ("DuoDose", "Scrublet", "scDblFinder", "DoubletFinder", "scds")
    for dataset in protocol["datasets"]["real_application"]:
        for seed in protocol["seeds"]["real_application"]:
            application_path = root / "real_application" / str(dataset) / f"seed_{seed}"
            status_path = application_path / "real_application_method_status.csv"
            statuses = pd.read_csv(status_path) if status_path.is_file() else pd.DataFrame()
            manifest_path = application_path / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
            for method in application_methods:
                match = statuses.loc[statuses.get("method", pd.Series(dtype=str)).astype(str).eq(method)] if not statuses.empty else pd.DataFrame()
                reason = ""
                contract_status = "PASS"
                if len(match) != 1:
                    contract_status, reason = "INCOMPLETE", "missing or duplicate real-application method row"
                elif str(match.iloc[0].get("status", "")).lower() != "success":
                    contract_status = "INCOMPLETE"
                    reason = str(match.iloc[0].get("message", "real-application method unavailable or failed"))
                elif not manifest:
                    contract_status, reason = "INCOMPLETE", "run manifest missing"
                elif manifest.get("workflow") != "real_data_application":
                    contract_status, reason = "INCOMPLETE", "run manifest does not declare real_data_application"
                elif not manifest.get("protocol_config_sha256"):
                    contract_status, reason = "INCOMPLETE", "explicit configuration hash missing from run manifest"
                elif not manifest.get("git_commit"):
                    contract_status, reason = "INCOMPLETE", "code commit missing from run manifest"
                raw_status = str(match.iloc[0].get("status", "missing")) if len(match) == 1 else "missing"
                execution_status = "COMPLETED" if raw_status.lower() == "success" else "FAILED" if raw_status.lower() in {"failed", "error"} else "SKIPPED" if raw_status.lower() == "skipped" else "UNAVAILABLE" if raw_status.lower() in {"unavailable", "not_available"} else "INCOMPLETE"
                formal_status = execution_status if execution_status != "COMPLETED" else "COMPLETED" if contract_status == "PASS" else "INCOMPLETE"
                rows.append({"workflow": "real_data_application", "dataset": dataset, "seed": seed, "method": method, "method_family": "internal" if method == "DuoDose" else "external", "execution_status": execution_status, "raw_execution_status": raw_status, "status": formal_status, "reason": reason, "config_hash_present": bool(manifest.get("protocol_config_sha256")), "seed_present": manifest.get("seed") is not None, "input_manifest_present": bool(manifest.get("input")), "code_version_present": bool(manifest.get("git_commit")), "result_path": str(application_path.relative_to(root))})
    frame = pd.DataFrame(rows)
    if frame.duplicated(["workflow", "dataset", "seed", "method"]).any():
        raise ValidationSuiteError("run-status audit generated duplicate requested rows")
    return frame


def domain_audit_contract_check(existing_domain_audit_dir: str | Path | None) -> pd.DataFrame:
    """Validate a completed strict domain audit without rerunning or copying it."""

    if existing_domain_audit_dir is None:
        return pd.DataFrame([{"check": "existing_final_domain_audit", "status": "NOT_RUN", "value": "", "reason": "existing_domain_audit_dir is not configured"}])
    root = Path(existing_domain_audit_dir)
    summary_path = root / "domain_audit_all_datasets_summary.csv"
    run_status_path = root / "domain_audit_all_datasets_run_status.csv"
    if not root.is_dir() or not summary_path.is_file() or not run_status_path.is_file():
        return pd.DataFrame([{"check": "existing_final_domain_audit", "status": "INCOMPLETE", "value": str(root), "reason": "configured domain-audit directory lacks all-dataset summary or run-status ledger"}])
    summary = pd.read_csv(summary_path)
    run_status = pd.read_csv(run_status_path)
    completed = run_status.loc[run_status.get("audit_status", pd.Series(dtype=str)).astype(str).eq("COMPLETED"), "dataset"].astype(str).tolist()
    primary_rows = []
    manifest_ok = True
    required_files_ok = True
    feature_exclusion_ok = True
    for dataset in completed:
        dataset_dir = root / dataset
        config_path = dataset_dir / "domain_audit_config.json"
        dataset_summary_path = dataset_dir / "domain_audit_summary.csv"
        feature_path = dataset_dir / "domain_audit_feature_audit.csv"
        required = [
            config_path,
            dataset_summary_path,
            feature_path,
            dataset_dir / "domain_audit_parent_unique_selection.csv",
            dataset_dir / "domain_audit_predictions.csv",
            dataset_dir / "domain_audit_report.md",
        ]
        required_files_ok &= all(path.is_file() and path.stat().st_size > 0 for path in required)
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            manifest_ok &= config.get("schema_version") == "semireal_real_domain_audit_v2"
            manifest_ok &= config.get("formal_construction_variant") == "raw_sum_parents_removed"
            manifest_ok &= config.get("formal_safe_feature_mode") == "fitted_reference"
            manifest_ok &= normalize_primary_analysis(config.get("primary_analysis")) == PRIMARY_ANALYSIS
            manifest_ok &= all(bundle.get("parent_removal_audit", {}).get("parents_removed_verified") is True for bundle in config.get("bundles", []))
        except (OSError, ValueError, TypeError):
            manifest_ok = False
        if dataset_summary_path.is_file():
            dataset_summary = pd.read_csv(dataset_summary_path)
            primary_rows.append(dataset_summary.loc[dataset_summary.get("is_primary", pd.Series(False, index=dataset_summary.index)).astype(bool)])
        if feature_path.is_file():
            features = pd.read_csv(feature_path)
            included = features.loc[features.get("included", pd.Series(False, index=features.index)).astype(bool)]
            forbidden_categories = included.get("category", pd.Series("", index=included.index)).astype(str).isin(["source_or_identifier", "contaminating_or_downstream", "model_output", "calibrated_model_output"])
            feature_exclusion_ok &= not forbidden_categories.any()
    primary = pd.concat(primary_rows, ignore_index=True) if primary_rows else pd.DataFrame()
    supported_statuses = {"COMPLETED", "INSUFFICIENT_PARENT_DISJOINT_DATA", "SKIPPED", "NOT_AVAILABLE"}
    checks = {
        "completed_audit_manifest": manifest_ok and bool(completed),
        "required_dataset_outputs": required_files_ok and bool(completed),
        "construction_variant": not primary.empty and primary.get("construction_variant", pd.Series(dtype=str)).astype(str).eq("raw_sum_parents_removed").all(),
        "safe_feature_mode": not primary.empty and primary.get("safe_feature_mode", pd.Series(dtype=str)).astype(str).eq("fitted_reference").all(),
        "heldout_semireal_heterotypic_only": not primary.empty and primary.get("semireal_split_used", pd.Series(dtype=str)).astype(str).isin(["test", "validation"]).all() and primary.get("analysis", pd.Series(dtype=str)).map(normalize_primary_analysis).eq(PRIMARY_ANALYSIS).all(),
        "parent_unique_filtering": not primary.empty and pd.to_numeric(primary.get("n_semireal_after_parent_unique_filter"), errors="coerce").gt(0).all(),
        "parent_overlap_across_folds_zero": not primary.empty and pd.to_numeric(primary.get("parent_overlap_across_folds"), errors="coerce").eq(0).all(),
        "source_and_downstream_features_excluded": feature_exclusion_ok,
        "no_domain_label_permutation": not any("permutation" in path.name.lower() for path in root.rglob("*")) and not any("permutation" in str(column).lower() for column in summary.columns),
        "completed_and_skipped_dataset_accounting": not run_status.empty and run_status.get("audit_status", pd.Series(dtype=str)).astype(str).isin(supported_statuses).all() and set(run_status["dataset"].astype(str)).issubset(set(summary["dataset"].astype(str))),
    }
    return pd.DataFrame(
        [
            {
                "check": key,
                "status": "PASS" if passed else "INCOMPLETE",
                "value": bool(passed),
                "reason": "" if passed else "configured existing domain-audit contract is incomplete or invalid",
            }
            for key, passed in checks.items()
        ]
    )


def train_rf_with_fit_labels(run: Any, labels: pd.Series) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    fit = run.fit_scores.copy()
    aligned = labels.reindex(fit.index)
    if aligned.isna().any():
        raise ValidationSuiteError("permutation labels do not align to every fit row")
    fit["true_label"] = aligned.astype(str)
    combined = pd.concat([fit, run.validation_scores], axis=0)
    result = train_predict_diagnostic_model(
        train_cell_scores=combined,
        test_cell_scores=run.test_scores,
        method=BACKEND_SPECS["rf"].internal_name,
        random_state=int(run.seed),
        net_train_seed=int(run.seed),
        train_index=fit.index,
        validation_index=run.validation_scores.index,
        safe_feature_transformer=run.transformer,
    )
    if result.get("summary", {}).get("status") != "success" or result.get("fitted_backend") is None:
        raise ValidationSuiteError("permutation RF fit failed: " + str(result.get("summary", {}).get("message", "unknown error")))
    return result["fitted_backend"], result["test_probabilities"].reindex(run.test_scores.index), dict(result["summary"])


def assert_no_hard_failures(checks: pd.DataFrame) -> None:
    failures = checks.loc[checks["status"].eq("FAIL") & checks["required"].astype(bool)]
    if not failures.empty:
        details = "; ".join(f"{row.audit}/{row.check}: {row.message}" for row in failures.itertuples())
        raise ValidationSuiteError("required validation checks failed: " + details)
