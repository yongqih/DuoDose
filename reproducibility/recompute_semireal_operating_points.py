"""Recompute revised semi-real high-RNA operating points from saved cell scores.

This migration never fits a model or reruns an external method. It is intended
for compatible cached runs after the controlled internal benchmark has been
rerun with the final feature contract. External score caches are reused only
when their cell IDs exactly match the corresponding controlled test rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from duodose.semireal_metrics import high_rna_metric_bundle, high_rna_operating_point_metrics

EXTERNAL_METHODS = ("Scrublet", "scDblFinder", "DoubletFinder", "scds")
INTERNAL_METHODS = ("DuoDose", "DuoDose-DL")
FPR_COLUMNS = (
    "high_RNA_singlet_FPR",
    "high_RNA_singlet_FPR_status",
    "high_RNA_singlet_FPR_reason",
    "high_RNA_singlet_FPR_metric_version",
    "high_RNA_singlet_FPR_at_matched_50pct_homotypic_recall",
    "high_RNA_singlet_FPR_at_matched_70pct_homotypic_recall",
    "high_RNA_singlet_FPR_at_matched_80pct_homotypic_recall",
    "high_RNA_singlet_FPR_at_fixed_20pct_candidate_budget",
    "high_RNA_singlet_FPR_at_true_doublet_budget",
    "primary_FPR_target_homotypic_recall",
    "primary_FPR_actual_homotypic_recall",
    "primary_FPR_actual_candidate_fraction",
    "primary_FPR_number_selected_cells",
    "fixed_20pct_FPR_actual_candidate_fraction",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if compression == "gzip" else ".csv"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=suffix, dir=path.parent)
    os.close(fd)
    tmp = Path(temporary)
    try:
        frame.to_csv(tmp, index=False, compression=compression)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_output_manifest(run_dir: Path) -> None:
    rows = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "output_manifest.json":
            rows.append({
                "path": str(path.relative_to(run_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    (run_dir / "output_manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": rows}, indent=2), encoding="utf-8"
    )


def _obs_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"cell_id", "true_label", "is_high_rna_singlet"}
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"controlled predictions are missing columns: {sorted(missing)}")
    if predictions["cell_id"].astype(str).duplicated().any():
        raise ValueError("controlled predictions contain duplicate cell_id values")
    obs = predictions.set_index(predictions["cell_id"].astype(str), drop=False).copy()
    obs.index.name = None
    obs["true_label"] = obs["true_label"].astype(str)
    high = obs["is_high_rna_singlet"]
    if high.dtype == bool:
        obs["is_high_rna_singlet"] = high
    else:
        normalized = high.astype(str).str.strip().str.lower()
        unknown = ~normalized.isin({"true", "false", "1", "0"})
        if bool(unknown.any()):
            raise ValueError("is_high_rna_singlet contains non-boolean values")
        obs["is_high_rna_singlet"] = normalized.isin({"true", "1"})
    return obs


def _update_metric_rows(metrics: pd.DataFrame, obs: pd.DataFrame, scores: dict[str, pd.Series], dataset: str, seed: int) -> pd.DataFrame:
    result = metrics.copy()
    for method, score in scores.items():
        mask = result["method"].astype(str).eq(method)
        if int(mask.sum()) != 1:
            raise ValueError(f"expected exactly one metric row for {dataset}/seed_{seed}/{method}")
        bundle = high_rna_metric_bundle(
            obs,
            score,
            dataset=dataset,
            source_dataset=dataset,
            seed=seed,
            method=method,
        )
        for column in FPR_COLUMNS:
            if column in bundle:
                result.loc[mask, column] = bundle[column]
    return result


def _load_external_score(run_dir: Path, dataset: str, method: str, cell_ids: pd.Index) -> pd.Series | None:
    candidates = sorted((run_dir / "score_cache").glob(f"{dataset}__{method}__*.csv"))
    valid: list[tuple[Path, np.ndarray]] = []
    expected = cell_ids.astype(str).tolist()
    for path in candidates:
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        if not {"cell_id", "score"}.issubset(frame):
            continue
        ids = frame["cell_id"].astype(str).tolist()
        values = pd.to_numeric(frame["score"], errors="coerce").to_numpy(dtype=float)
        if ids == expected and len(values) == len(expected) and np.isfinite(values).all():
            valid.append((path, values))
    if not valid:
        return None
    reference = valid[0][1]
    if any(not np.array_equal(reference, values) for _, values in valid[1:]):
        names = ", ".join(path.name for path, _ in valid)
        raise ValueError(f"ambiguous external caches with different scores for {dataset}/{method}: {names}")
    return pd.Series(reference, index=cell_ids, dtype=float)


def _migrate_manifest(run_dir: Path, protocol_path: Path, *, source_scores_reused: bool) -> None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_hash = str(manifest.get("protocol_config_sha256", ""))
    manifest["score_generation_protocol_sha256"] = old_hash
    manifest["protocol_path"] = str(protocol_path.resolve())
    manifest["protocol_config_sha256"] = _sha256(protocol_path)
    manifest["protocol_name"] = str(yaml.safe_load(protocol_path.read_text(encoding="utf-8"))["protocol_name"])
    manifest["metric_migration"] = {
        "name": "matched_homotypic_recall_v1",
        "external_scores_reused": bool(source_scores_reused),
        "cell_id_alignment": "exact_order_match",
        "models_rerun": False,
        "external_methods_rerun": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controlled-dir", required=True)
    parser.add_argument("--external-dir", required=True)
    parser.add_argument("--protocol", default=str(ROOT / "reproducibility/configs/final_protocol.yaml"))
    parser.add_argument("--external-only", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--update-manifests", action="store_true")
    args = parser.parse_args()

    controlled_root = Path(args.controlled_dir).resolve()
    external_root = Path(args.external_dir).resolve()
    protocol_path = Path(args.protocol).resolve()
    summary_rows: list[dict[str, object]] = []

    prediction_paths = sorted(controlled_root.glob("*/seed_*/controlled_test_predictions.csv.gz"))
    if not prediction_paths:
        raise FileNotFoundError(f"no controlled prediction files found under {controlled_root}")

    for prediction_path in prediction_paths:
        dataset = prediction_path.parents[1].name
        seed = int(prediction_path.parent.name.removeprefix("seed_"))
        predictions = pd.read_csv(prediction_path)
        obs = _obs_from_predictions(predictions)
        cell_ids = obs.index

        if not args.external_only:
            metric_path = prediction_path.parent / "controlled_metrics.csv"
            if metric_path.is_file():
                metrics = pd.read_csv(metric_path)
                scores: dict[str, pd.Series] = {}
                for method in INTERNAL_METHODS:
                    column = f"{method.replace('-', '_')}_overall"
                    if column in predictions:
                        scores[method] = pd.Series(
                            pd.to_numeric(predictions[column], errors="coerce").to_numpy(dtype=float),
                            index=cell_ids,
                        )
                if scores:
                    operating = high_rna_operating_point_metrics(obs, scores, dataset=dataset, source_dataset=dataset, seed=seed)
                    _atomic_csv(operating, prediction_path.parent / "controlled_high_RNA_operating_points.csv")
                    _atomic_csv(_update_metric_rows(metrics, obs, scores, dataset, seed), metric_path)
                    if args.update_manifests:
                        _migrate_manifest(prediction_path.parent, protocol_path, source_scores_reused=True)
                        _write_output_manifest(prediction_path.parent)

        external_run = external_root / dataset / f"seed_{seed}"
        external_metric_path = external_run / "external_controlled_metrics.csv"
        if not external_metric_path.is_file():
            if args.skip_missing:
                summary_rows.append({"dataset": dataset, "seed": seed, "status": "SKIPPED", "message": "external metrics missing"})
                continue
            raise FileNotFoundError(external_metric_path)
        external_metrics = pd.read_csv(external_metric_path)
        external_scores: dict[str, pd.Series] = {}
        missing_methods: list[str] = []
        for method in EXTERNAL_METHODS:
            score = _load_external_score(external_run, dataset, method, cell_ids)
            if score is None:
                missing_methods.append(method)
            else:
                external_scores[method] = score
        if missing_methods:
            message = f"missing exactly aligned score cache: {', '.join(missing_methods)}"
            if args.skip_missing:
                summary_rows.append({"dataset": dataset, "seed": seed, "status": "SKIPPED", "message": message})
                continue
            raise FileNotFoundError(f"{dataset}/seed_{seed}: {message}")
        operating = high_rna_operating_point_metrics(obs, external_scores, dataset=dataset, source_dataset=dataset, seed=seed)
        _atomic_csv(operating, external_run / "external_high_RNA_operating_points.csv")
        _atomic_csv(_update_metric_rows(external_metrics, obs, external_scores, dataset, seed), external_metric_path)
        if args.update_manifests:
            _migrate_manifest(external_run, protocol_path, source_scores_reused=True)
            _write_output_manifest(external_run)
        summary_rows.append({"dataset": dataset, "seed": seed, "status": "SUCCESS", "message": "external scores reused after exact cell-ID alignment"})

    summary = pd.DataFrame(summary_rows)
    output = external_root / "operating_point_migration_status.csv"
    _atomic_csv(summary, output)
    failures = summary.loc[~summary["status"].eq("SUCCESS")] if not summary.empty else summary
    print(f"Wrote {output}; success={int(summary['status'].eq('SUCCESS').sum()) if not summary.empty else 0}; non-success={len(failures)}")


if __name__ == "__main__":
    main()
