"""Shared protocol-driven data, fitting, evaluation, and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import copy
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.metrics import average_precision_score, roc_auc_score

from duodose import realdata
from duodose.benchmark_core import run_benchmark_core
from duodose.domain_audit import export_domain_audit_inputs
from duodose.models.registry import BACKEND_SPECS
from duodose.net import probabilities_to_scores, train_predict_diagnostic_model
from duodose.protocol import load_final_protocol
from duodose.rf_weighting import FORMAL_HIGH_RNA_NEGATIVE_WEIGHT, formal_rf_sample_weights
from duodose.safe_feature_transformer import SafeFeatureTransformer
from duodose.semireal_bundle import SemiRealSplitBundle, make_parent_disjoint_semireal_bundle
from duodose.semireal_metrics import high_rna_metric_bundle


DOUBLET_LABELS = frozenset({"homotypic_doublet", "heterotypic_doublet"})
EXTERNAL_METHODS = ("Scrublet", "scDblFinder", "DoubletFinder", "scds")


@dataclass
class LoadedDataset:
    dataset: str
    adata: AnnData
    source_path: Path
    source_format: str
    label_source: str
    conversion_status: str


@dataclass
class ProtocolRun:
    dataset: str
    seed: int
    protocol: dict[str, Any]
    source_path: Path
    source_format: str
    label_source: str
    original_adata: AnnData
    bundle: SemiRealSplitBundle
    transformer: SafeFeatureTransformer
    fit_scores: pd.DataFrame
    validation_scores: pd.DataFrame
    test_scores: pd.DataFrame
    fully_real_scores: pd.DataFrame
    test_features: pd.DataFrame
    fully_real_features: pd.DataFrame
    method_probabilities_test: dict[str, pd.DataFrame]
    method_probabilities_real: dict[str, pd.DataFrame]
    fitted_backends: dict[str, object]
    training_summaries: list[dict[str, Any]]
    timings: dict[str, float]
    runtime_seconds: float


def split_csv(value: str | None) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def discover_dataset_manifest(data_dir: str | Path) -> pd.DataFrame:
    """Discover complete datasets by bundle/file identity, never sidecar stems."""

    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {root}")
    candidates = realdata.discover_input_candidates(root, selected=None, max_datasets=None)
    rows: list[dict[str, Any]] = []
    seen: dict[str, Path] = {}
    for candidate in candidates:
        if candidate.input_format in {"csv", "tsv"} and candidate.input_path.stem.lower() in {
            "barcodes",
            "genes",
            "features",
            "labels",
            "label",
            "metadata",
            "matrix",
            "counts",
        }:
            continue
        name = str(candidate.dataset)
        resolved = candidate.input_path.resolve()
        matrix_source = ""
        label_source = ""
        status = "valid"
        reason = ""
        if candidate.input_format == "directory":
            matrix = realdata._matrix_file(resolved)
            genes = realdata._genes_file(resolved)
            barcodes = realdata._barcodes_file(resolved)
            labels = realdata._label_file(resolved)
            matrix_source = str(matrix or "")
            label_source = str(labels or "")
            missing = [name for name, value in (("matrix", matrix), ("genes/features", genes), ("barcodes", barcodes), ("labels/metadata", labels)) if value is None]
            if missing:
                status = "invalid"
                reason = f"incomplete converted dataset bundle; missing {', '.join(missing)}"
        elif candidate.input_format == "rds":
            matrix_source = "embedded in RDS"
            label_source = "embedded in RDS"
        else:
            matrix_source = str(resolved)
            label_source = "embedded; validated by the active loader"
        if name in seen and seen[name] != resolved:
            raise ValueError(f"ambiguous duplicate dataset name {name!r}: {seen[name]} and {resolved}")
        seen[name] = resolved
        rows.append(
            {
                "dataset": name,
                "resolved_path": str(resolved),
                "dataset_format": candidate.input_format,
                "matrix_source": matrix_source,
                "label_source": label_source,
                "discovery_status": status,
                "skip_reason": reason,
            }
        )
    return pd.DataFrame(rows).drop_duplicates("dataset").sort_values("dataset", kind="stable").reset_index(drop=True)


def load_dataset_exact(
    data_dir: str | Path,
    dataset: str,
    *,
    conversion_dir: str | Path,
    convert_rds: bool,
    refresh_conversion: bool = False,
) -> LoadedDataset:
    manifest = discover_dataset_manifest(data_dir)
    match = manifest.loc[manifest["dataset"].eq(str(dataset))]
    if len(match) != 1:
        available = ", ".join(manifest["dataset"].astype(str))
        raise ValueError(f"dataset {dataset!r} was not found by exact name; available datasets: {available}")
    row = match.iloc[0]
    path = Path(row["resolved_path"])
    candidate = realdata.DatasetCandidate(str(dataset), path, str(row["dataset_format"]), path)
    if candidate.input_format == "rds":
        if not convert_rds:
            raise ValueError(f"dataset {dataset!r} is RDS; rerun with --convert-rds")
        candidate = realdata.convert_rds_candidate(
            candidate,
            Path(conversion_dir),
            refresh_cache=bool(refresh_conversion),
            quiet_external=False,
        )
        if candidate.converted_status not in {"success", "cached"}:
            raise RuntimeError(f"RDS conversion failed for {dataset}: {candidate.converted_message}")
    loaded = realdata.load_dataset(candidate.load_path, dataset_name=str(dataset), input_format=candidate.input_format)
    if loaded.status != "success" or loaded.adata is None:
        raise RuntimeError(f"dataset loading failed for {dataset}: {loaded.message}")
    adata = loaded.adata.copy()
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    if "experimental_doublet" not in adata.obs:
        raise ValueError(f"dataset {dataset!r} has no experimental doublet labels")
    return LoadedDataset(
        dataset=str(dataset),
        adata=adata,
        source_path=Path(candidate.load_path).resolve(),
        source_format=str(candidate.input_format),
        label_source=str(loaded.label_source_column),
        conversion_status=str(candidate.converted_status),
    )


def _reference_fit_rows(bundle: SemiRealSplitBundle) -> AnnData:
    origin = bundle.fit_adata.obs.get("semireal_origin", pd.Series("", index=bundle.fit_adata.obs_names)).astype(str)
    reference = bundle.fit_adata[origin.isin({"observed_background", "real_labeled_singlet"}).to_numpy(), :].copy()
    if reference.n_obs < 3:
        raise ValueError("fitted-reference SafeFeatures require fit-split observed background cells")
    return reference


def formal_backend_training_kwargs(backend: str, train_scores: pd.DataFrame) -> dict[str, object]:
    """Return immutable production-only training extras for one backend."""

    if backend != "rf":
        return {}
    return {
        "sample_weight": formal_rf_sample_weights(train_scores),
        "high_rna_negative_weight": FORMAL_HIGH_RNA_NEGATIVE_WEIGHT,
    }


def run_protocol_models(
    loaded: LoadedDataset,
    *,
    protocol_path: str | Path | None,
    seed: int,
    backends: Iterable[str] = ("rf", "dl"),
    device: str = "auto",
    amp: bool = False,
    dl_max_epochs: int = 200,
    dl_patience: int = 20,
    dl_batch_size: int | None = None,
    protocol_override: dict[str, Any] | None = None,
    use_explicit_validation_sizes: bool = False,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
    verbose_progress: bool = False,
) -> ProtocolRun:
    """Fit the frozen protocol without consuming experimental labels."""

    started = time.perf_counter()
    protocol = copy.deepcopy(protocol_override) if protocol_override is not None else load_final_protocol(protocol_path)
    if "_protocol_path" not in protocol:
        protocol["_protocol_path"] = str(Path(protocol_path).resolve()) if protocol_path else str(Path(load_final_protocol()["_protocol_path"]))
    construction = protocol["construction"]
    sizes = protocol["semi_real"]
    clustering = protocol["clustering"]
    original = loaded.adata.copy()
    original_labels = original.obs["experimental_doublet"].astype(int).copy()
    blind = original.copy()
    blind.obs["experimental_doublet"] = 0
    n_reference = min(int(sizes["n_reference_singlets"]), int(blind.n_obs // 2))
    if blind.n_obs < int(sizes["minimum_eligible_singlets"]):
        raise ValueError(
            f"dataset {loaded.dataset!r} has {blind.n_obs} cells; frozen protocol requires at least "
            f"{int(sizes['minimum_eligible_singlets'])}"
        )
    construction_started = time.perf_counter()
    if progress_callback is not None:
        progress_callback({"event": "milestone", "message": "constructing parent-disjoint semi-real splits"})
    validation_sizes = (
        {
            "n_validation_homotypic_doublets": int(sizes["n_validation_homotypic_doublets"]),
            "n_validation_heterotypic_doublets": int(sizes["n_validation_heterotypic_doublets"]),
        }
        if use_explicit_validation_sizes
        else {}
    )
    bundle = make_parent_disjoint_semireal_bundle(
        blind,
        dataset=loaded.dataset,
        seed=int(seed),
        n_singlets=n_reference,
        n_train_homotypic_doublets=int(sizes["n_train_homotypic_doublets"]),
        n_train_heterotypic_doublets=int(sizes["n_train_heterotypic_doublets"]),
        n_test_homotypic_doublets=int(sizes["n_test_homotypic_doublets"]),
        n_test_heterotypic_doublets=int(sizes["n_test_heterotypic_doublets"]),
        n_clusters=int(clustering["n_clusters"]),
        test_parent_fraction=0.40,
        validation_parent_fraction=0.25,
        high_rna_quantile=float(sizes["high_rna_quantile"]),
        min_cluster_size=int(clustering["min_cluster_size"]),
        construction_variant=str(construction["construction_variant"]),
        **validation_sizes,
    )
    construction_seconds = time.perf_counter() - construction_started
    fully_real_input = original.copy()
    fully_real_input.obs["experimental_doublet"] = original_labels
    core = run_benchmark_core(
        bundle,
        fully_real_input,
        protocol=protocol,
        seed=int(seed),
        backends=backends,
        device=device,
        amp=bool(amp),
        dl_max_epochs=int(dl_max_epochs),
        dl_patience=int(dl_patience),
        dl_batch_size=dl_batch_size,
        progress_callback=progress_callback,
        verbose_progress=bool(verbose_progress),
        construction_seconds=float(construction_seconds),
    )

    return ProtocolRun(
        dataset=loaded.dataset,
        seed=int(seed),
        protocol=protocol,
        source_path=loaded.source_path,
        source_format=loaded.source_format,
        label_source=loaded.label_source,
        original_adata=original,
        bundle=bundle,
        transformer=core.transformer,
        fit_scores=core.fit_scores,
        validation_scores=core.validation_scores,
        test_scores=core.test_scores,
        fully_real_scores=core.observed_scores,
        test_features=core.test_features,
        fully_real_features=core.observed_features,
        method_probabilities_test=core.method_probabilities_test,
        method_probabilities_real=core.method_probabilities_observed,
        fitted_backends=core.fitted_backends,
        training_summaries=core.training_summaries,
        timings=core.timings,
        runtime_seconds=float(time.perf_counter() - started),
    )


def _safe_metric(metric, y: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(score)
    if int(mask.sum()) < 2 or np.unique(y[mask]).size < 2:
        return float("nan")
    return float(metric(y[mask], score[mask]))


def _top_k_metrics(labels: pd.Series, score: pd.Series) -> tuple[float, float, int]:
    positive = labels.isin(DOUBLET_LABELS).to_numpy()
    values = pd.to_numeric(score, errors="coerce").to_numpy(dtype=float)
    ranking = pd.DataFrame(
        {
            "score": np.where(np.isfinite(values), values, -np.inf),
            "cell_id": labels.index.astype(str).to_numpy(),
            "positive": positive,
        }
    ).sort_values(["score", "cell_id"], ascending=[False, True], kind="mergesort")
    k = int(positive.sum())
    selected = ranking.iloc[:k]
    true_selected = int(selected["positive"].sum())
    return (
        float(true_selected / k) if k else float("nan"),
        float(true_selected / positive.sum()) if positive.sum() else float("nan"),
        k,
    )


def controlled_metric_row(
    *,
    dataset: str,
    seed: int,
    method: str,
    labels: pd.Series,
    obs: pd.DataFrame,
    overall_score: pd.Series,
    homotypic_score: pd.Series,
    heterotypic_score: pd.Series,
    status: str = "success",
    message: str = "",
    runtime_seconds: float = float("nan"),
) -> dict[str, Any]:
    labels = labels.astype(str)
    negative = labels.isin({"clean", "singlet", "high_RNA_singlet"})
    overall_y = labels.isin(DOUBLET_LABELS).astype(int).to_numpy()
    overall_values = overall_score.reindex(labels.index).to_numpy(dtype=float)
    hom_mask = (labels.eq("homotypic_doublet") | negative).to_numpy()
    het_mask = (labels.eq("heterotypic_doublet") | negative).to_numpy()
    high_mask = (labels.eq("homotypic_doublet") | labels.eq("high_RNA_singlet")).to_numpy()
    precision, recall, k = _top_k_metrics(labels, overall_score.reindex(labels.index))
    fpr = high_rna_metric_bundle(
        obs.reindex(labels.index),
        overall_score.reindex(labels.index),
        dataset=dataset,
        source_dataset=dataset,
        seed=int(seed),
        method=method,
    )
    hom_ap = _safe_metric(
        average_precision_score,
        labels.loc[hom_mask].eq("homotypic_doublet").astype(int).to_numpy(),
        homotypic_score.reindex(labels.index).to_numpy(dtype=float)[hom_mask],
    )
    het_ap = _safe_metric(
        average_precision_score,
        labels.loc[het_mask].eq("heterotypic_doublet").astype(int).to_numpy(),
        heterotypic_score.reindex(labels.index).to_numpy(dtype=float)[het_mask],
    )
    return {
        "dataset": dataset,
        "seed": int(seed),
        "method": method,
        "status": status,
        "message": message,
        "AUROC": _safe_metric(roc_auc_score, overall_y, overall_values),
        "overall_AUPRC": _safe_metric(average_precision_score, overall_y, overall_values),
        "homotypic_AUPRC": hom_ap,
        "heterotypic_AUPRC": het_ap,
        "macro_subtype_AUPRC": float(np.nanmean([hom_ap, het_ap])),
        "homotypic_vs_high_RNA_singlet_AUPRC": _safe_metric(
            average_precision_score,
            labels.loc[high_mask].eq("homotypic_doublet").astype(int).to_numpy(),
            homotypic_score.reindex(labels.index).to_numpy(dtype=float)[high_mask],
        ),
        "high_RNA_singlet_FPR": fpr["high_RNA_singlet_FPR"],
        "high_RNA_singlet_FPR_status": fpr["high_RNA_singlet_FPR_status"],
        "high_RNA_singlet_FPR_reason": fpr["high_RNA_singlet_FPR_reason"],
        "high_RNA_singlet_FPR_metric_version": fpr["high_RNA_singlet_FPR_metric_version"],
        "high_RNA_singlet_FPR_at_matched_50pct_homotypic_recall": fpr["high_RNA_singlet_FPR_at_matched_50pct_homotypic_recall"],
        "high_RNA_singlet_FPR_at_matched_70pct_homotypic_recall": fpr["high_RNA_singlet_FPR_at_matched_70pct_homotypic_recall"],
        "high_RNA_singlet_FPR_at_matched_80pct_homotypic_recall": fpr["high_RNA_singlet_FPR_at_matched_80pct_homotypic_recall"],
        "high_RNA_singlet_FPR_at_fixed_20pct_candidate_budget": fpr["high_RNA_singlet_FPR_at_fixed_20pct_candidate_budget"],
        "high_RNA_singlet_FPR_at_true_doublet_budget": fpr["high_RNA_singlet_FPR_at_true_doublet_budget"],
        "primary_FPR_target_homotypic_recall": fpr["primary_FPR_target_homotypic_recall"],
        "primary_FPR_actual_homotypic_recall": fpr["primary_FPR_actual_homotypic_recall"],
        "primary_FPR_actual_candidate_fraction": fpr["primary_FPR_actual_candidate_fraction"],
        "primary_FPR_number_selected_cells": fpr["primary_FPR_number_selected_cells"],
        "precision_at_K": precision,
        "recall_at_K": recall,
        "K": k,
        "runtime_seconds": runtime_seconds,
    }


def evaluate_internal_controlled(run: ProtocolRun) -> pd.DataFrame:
    labels = run.test_scores["true_label"].astype(str)
    rows = []
    for method, probabilities in run.method_probabilities_test.items():
        overall, homotypic, heterotypic = probabilities_to_scores(probabilities)
        rows.append(
            controlled_metric_row(
                dataset=run.dataset,
                seed=run.seed,
                method=method,
                labels=labels,
                obs=run.bundle.test_adata.obs,
                overall_score=overall,
                homotypic_score=homotypic,
                heterotypic_score=heterotypic,
            )
        )
    return pd.DataFrame(rows)


def evaluate_real_probabilities(run: ProtocolRun) -> pd.DataFrame:
    labels = run.original_adata.obs["experimental_doublet"].astype(int)
    rows = []
    for method, probabilities in run.method_probabilities_real.items():
        overall, _, _ = probabilities_to_scores(probabilities)
        values = overall.reindex(labels.index).to_numpy(dtype=float)
        rows.append(
            {
                "dataset": run.dataset,
                "seed": run.seed,
                "method": method,
                "AUROC": _safe_metric(roc_auc_score, labels.to_numpy(), values),
                "AUPRC": _safe_metric(average_precision_score, labels.to_numpy(), values),
                "n_positive": int(labels.eq(1).sum()),
                "n_negative": int(labels.eq(0).sum()),
                "n_excluded": 0,
                "status": "success",
                "message": "",
                "paper_metric_scope": "experimental doublet-enriched detection",
                "paper_metric_definition": "positive=experimentally annotated doublet; negative=all annotated non-doublets; experimental labels evaluation-only",
            }
        )
    return pd.DataFrame(rows)


def run_external_scores(
    adata: AnnData,
    *,
    dataset: str,
    seed: int,
    methods: Iterable[str],
    cache_dir: Path,
    expected_doublet_rate: float,
    refresh_cache: bool = False,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
    log_dir: Path | None = None,
    heartbeat_seconds: float = 45.0,
    audit_dirs: Mapping[str, Path] | None = None,
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    scores: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    for method in methods:
        if method not in EXTERNAL_METHODS:
            raise ValueError(f"unsupported external method {method!r}")
        method_log = None if log_dir is None else log_dir / f"{method}.log"
        if progress_callback is not None:
            progress_callback({"event": "method_start", "message": f"starting external method {method}", "method": method, "log_path": str(method_log or "")})
        score, status, message, runtime = realdata.run_external_score(
            adata,
            dataset,
            method,
            int(seed),
            bool(refresh_cache),
            cache_dir,
            False,
            float(expected_doublet_rate),
            log_path=method_log,
            heartbeat_seconds=float(heartbeat_seconds),
            progress_callback=progress_callback,
            audit_dir=(audit_dirs or {}).get(method),
        )
        scores[method] = score
        rows.append({"dataset": dataset, "seed": int(seed), "method": method, "status": status, "message": message, "runtime_seconds": runtime})
        if progress_callback is not None:
            progress_callback({"event": "method_complete", "message": f"external method {method}: {status} in {runtime:.1f}s", "method": method, "status": status})
    return scores, pd.DataFrame(rows)


def export_domain_bundle(run: ProtocolRun, output_dir: str | Path) -> dict[str, str]:
    metadata = run.transformer.metadata()
    metadata.update(
        construction_variant="raw_sum_parents_removed",
        count_construction_mode="raw_sum",
        parent_reference_mode="removed",
        parents_removed=True,
        parent_disjoint=True,
    )
    return export_domain_audit_inputs(
        output_dir=Path(output_dir),
        dataset=run.dataset,
        seed=run.seed,
        fully_real_score_frame=run.fully_real_scores,
        fully_real_raw_features=run.fully_real_features,
        semireal_score_frame=run.test_scores,
        semireal_raw_features=run.test_features,
        semireal_split_name="test",
        parent_map=run.bundle.parent_map,
        safe_feature_metadata=metadata,
        source_files={"dataset": str(run.source_path), "protocol": str(run.protocol["_protocol_path"])},
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }
    for package in ("numpy", "pandas", "scipy", "sklearn", "anndata", "torch"):
        try:
            module = __import__(package)
            record[f"{package}_version"] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            record[f"{package}_version"] = "NOT_INSTALLED"
    try:
        import torch

        record["cuda_available"] = bool(torch.cuda.is_available())
        record["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        record["cuda_version"] = str(torch.version.cuda or "")
    except ImportError:
        record.update(cuda_available=False, gpu_name="", cuda_version="")
    return record


def git_revision(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=False, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNAVAILABLE"


def write_run_manifest(
    output_dir: str | Path,
    *,
    workflow: str,
    protocol: dict[str, Any],
    dataset: str,
    seed: int,
    runtime_seconds: float,
    source_path: Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    source_manifest: dict[str, Any] = {"path": str(source_path)}
    if source_path.is_file():
        source_manifest.update(size_bytes=source_path.stat().st_size, sha256=sha256_file(source_path))
    manifest = {
        "schema_version": 1,
        "workflow": workflow,
        "command": subprocess.list2cmdline(sys.argv),
        "protocol_path": protocol["_protocol_path"],
        "protocol_config_sha256": sha256_file(Path(protocol["_protocol_path"])),
        "protocol_name": protocol["protocol_name"],
        "dataset": dataset,
        "seed": int(seed),
        "runtime_seconds": float(runtime_seconds),
        "git_commit": git_revision(repo_root),
        "git_worktree_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, check=False, capture_output=True, text=True).stdout.strip()),
        "input": source_manifest,
        "environment": environment_record(),
        "high_rna_negative_weight": protocol.get("models", {}).get("high_rna_negative_weight"),
        **dict(extra or {}),
    }
    path = output / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def write_output_manifest(output_dir: str | Path) -> Path:
    output = Path(output_dir)
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "output_manifest.json":
            rows.append(
                {
                    "path": str(path.relative_to(output)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    destination = output / "output_manifest.json"
    destination.write_text(json.dumps({"schema_version": 1, "files": rows}, indent=2), encoding="utf-8")
    return destination
