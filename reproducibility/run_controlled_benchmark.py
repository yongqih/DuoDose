"""Run the frozen internal-method controlled semi-real benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from duodose.net import probabilities_to_scores  # noqa: E402
from duodose.models.registry import BACKEND_SPECS  # noqa: E402
from duodose.progress import ProgressReporter, ProgressSettings, add_progress_arguments, progress_paths  # noqa: E402
from duodose.semireal_metrics import high_rna_operating_point_metrics  # noqa: E402
from duodose.protocol import load_final_protocol  # noqa: E402
from reproducibility.lib.common import (  # noqa: E402
    evaluate_internal_controlled,
    load_dataset_exact,
    run_protocol_models,
    sha256_file,
    split_csv,
    write_output_manifest,
    write_run_manifest,
)


def _integers(value: str) -> list[int]:
    return [int(item) for item in split_csv(value)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conversion-dir", required=True)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--backends", default="rf,dl")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dl-max-epochs", type=int, default=200)
    parser.add_argument("--dl-patience", type=int, default=20)
    parser.add_argument("--dl-batch-size", type=int, default=None)
    parser.add_argument("--convert-rds", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only dataset/seed outputs that pass the frozen per-run output contract.",
    )
    add_progress_arguments(parser)
    return parser


def _completed_run(run_dir: Path, dataset: str, seed: int, backends: list[str], protocol: dict) -> pd.DataFrame | None:
    required = (
        "controlled_metrics.csv",
        "training_summaries.csv",
        "semireal_parent_map.csv.gz",
        "construction_report.json",
        "controlled_test_predictions.csv.gz",
        "controlled_high_RNA_operating_points.csv",
        "run_manifest.json",
        "output_manifest.json",
    )
    if any(not (run_dir / name).is_file() or (run_dir / name).stat().st_size == 0 for name in required):
        return None
    try:
        metrics = pd.read_csv(run_dir / "controlled_metrics.csv")
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    except (ValueError, TypeError):
        return None
    if manifest.get("protocol_config_sha256") != sha256_file(Path(protocol["_protocol_path"])):
        return None
    expected = {BACKEND_SPECS[backend].display_name for backend in backends}
    if set(metrics.get("method", pd.Series(dtype=str)).astype(str)) != expected:
        return None
    if not metrics.get("status", pd.Series(dtype=str)).astype(str).str.lower().eq("success").all():
        return None
    if not metrics.get("dataset", pd.Series(dtype=str)).astype(str).eq(dataset).all():
        return None
    if not pd.to_numeric(metrics.get("seed", pd.Series(dtype=float)), errors="coerce").eq(seed).all():
        return None
    if metrics.duplicated(["dataset", "seed", "method"]).any():
        return None
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    protocol = load_final_protocol(args.protocol)
    datasets = split_csv(args.datasets)
    if datasets == ["all"]:
        datasets = list(protocol["datasets"]["real_doublet_enriched"])
    seeds = _integers(args.seeds) if args.seeds else list(protocol["seeds"]["controlled_benchmark"])
    backends = split_csv(args.backends)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ledger_path, snapshot_path = progress_paths(output, args)
    config_hash = str(args.progress_config_hash or sha256_file(Path(protocol["_protocol_path"])))
    reporter = ProgressReporter(
        stage="controlled_benchmark",
        total_units=len(datasets) * len(seeds) * len(backends),
        settings=ProgressSettings.from_args(args),
        ledger_path=ledger_path,
        snapshot_path=snapshot_path,
        config_hash=config_hash,
        output_path=output,
    )
    all_metrics: list[pd.DataFrame] = []
    statuses: list[dict[str, object]] = []
    cached_runs = {
        (dataset, seed): cached
        for dataset in datasets
        for seed in seeds
        if args.resume
        and (cached := _completed_run(output / dataset / f"seed_{seed}", dataset, seed, backends, protocol)) is not None
    }
    if args.resume:
        reporter.event(
            f"Resume scan complete: valid completed runs={len(cached_runs)}, "
            f"remaining={len(datasets) * len(seeds) - len(cached_runs)}"
        )
    try:
        for dataset_index, dataset in enumerate(datasets, start=1):
            reporter.event(f"[Dataset {dataset_index}/{len(datasets)}] {dataset}", dataset=dataset)
            for seed_index, seed in enumerate(seeds, start=1):
                reporter.event(f"[Seed {seed_index}/{len(seeds)}] {seed}", dataset=dataset, seed=seed)
                run_dir = output / dataset / f"seed_{seed}"
                run_started = time.perf_counter()
                cached = cached_runs.get((dataset, seed))
                if cached is not None:
                    all_metrics.append(cached)
                    statuses.append({"dataset": dataset, "seed": seed, "status": "SUCCESS", "message": "reused validated output", "runtime_seconds": 0.0})
                    for backend in backends:
                        reporter.cached_unit(dataset=dataset, seed=seed, method=BACKEND_SPECS[backend].display_name, output_path=run_dir)
                    pd.DataFrame(statuses).to_csv(output / "controlled_run_status.csv", index=False)
                    continue

                active_units: dict[str, float] = {}
                resolved_methods: set[str] = set()
                cell_count = 0
                gene_count = 0

                def progress_callback(event):
                    method = str(event.get("method", ""))
                    if event.get("event") == "method_start" and method:
                        method_order = [BACKEND_SPECS[name].display_name for name in backends]
                        active_units[method] = reporter.start_unit(
                            dataset=dataset,
                            seed=seed,
                            method=method,
                            cell_count=cell_count,
                            gene_count=gene_count,
                            output_path=run_dir,
                            prefix=f"[Method {method_order.index(method) + 1}/{len(method_order)}] {method}",
                        )
                    elif event.get("event") == "method_complete" and method in active_units:
                        reporter.complete_unit(active_units.pop(method), message=f"completed {method}")
                        resolved_methods.add(method)
                    else:
                        reporter.callback(event)

                try:
                    loaded = load_dataset_exact(
                        args.data_dir,
                        dataset,
                        conversion_dir=args.conversion_dir,
                        convert_rds=bool(args.convert_rds),
                    )
                    cell_count, gene_count = int(loaded.adata.n_obs), int(loaded.adata.n_vars)
                    run = run_protocol_models(
                        loaded,
                        protocol_path=args.protocol,
                        seed=seed,
                        backends=backends,
                        device=args.device,
                        amp=bool(args.amp),
                        dl_max_epochs=args.dl_max_epochs,
                        dl_patience=args.dl_patience,
                        dl_batch_size=args.dl_batch_size,
                        progress_callback=progress_callback,
                        verbose_progress=bool(args.verbose_progress),
                    )
                    metrics = evaluate_internal_controlled(run)
                    reporter.event("writing controlled benchmark outputs", dataset=dataset, seed=seed)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    metrics.to_csv(run_dir / "controlled_metrics.csv", index=False)
                    pd.DataFrame(run.training_summaries).to_csv(run_dir / "training_summaries.csv", index=False)
                    run.bundle.parent_map.to_csv(run_dir / "semireal_parent_map.csv.gz", index=False, compression="gzip")
                    (run_dir / "construction_report.json").write_text(json.dumps(run.bundle.construction_report, indent=2), encoding="utf-8")
                    predictions = run.test_scores[["cell_id", "true_label", "is_high_rna_singlet"]].copy()
                    overall_scores: dict[str, pd.Series] = {}
                    for method, probabilities in run.method_probabilities_test.items():
                        overall, homotypic, heterotypic = probabilities_to_scores(probabilities)
                        overall_scores[method] = overall.reindex(predictions.index)
                        prefix = method.replace("-", "_")
                        predictions[f"{prefix}_overall"] = overall
                        predictions[f"{prefix}_homotypic"] = homotypic
                        predictions[f"{prefix}_heterotypic"] = heterotypic
                    predictions.to_csv(run_dir / "controlled_test_predictions.csv.gz", index=False, compression="gzip")
                    operating_obs = run.bundle.test_adata.obs.reindex(predictions.index).copy()
                    operating_obs["true_label"] = predictions["true_label"].astype(str)
                    operating_obs["is_high_rna_singlet"] = predictions["is_high_rna_singlet"].astype(bool)
                    high_rna_operating_point_metrics(
                        operating_obs,
                        overall_scores,
                        dataset=dataset,
                        source_dataset=dataset,
                        seed=seed,
                    ).to_csv(run_dir / "controlled_high_RNA_operating_points.csv", index=False)
                    write_run_manifest(
                        run_dir,
                        workflow="controlled_benchmark",
                        protocol=run.protocol,
                        dataset=dataset,
                        seed=seed,
                        runtime_seconds=run.runtime_seconds,
                        source_path=run.source_path,
                        extra={
                            "backends": backends,
                            "high_rna_negative_weight": run.protocol["models"]["high_rna_negative_weight"],
                        },
                    )
                    write_output_manifest(run_dir)
                    failure_path = run_dir / "failure.json"
                    if failure_path.exists():
                        failure_path.unlink()
                    all_metrics.append(metrics)
                    statuses.append({"dataset": dataset, "seed": seed, "status": "SUCCESS", "message": "", "runtime_seconds": time.perf_counter() - run_started})
                except Exception as exc:
                    for method, unit_started in list(active_units.items()):
                        reporter.fail_unit(unit_started, exc)
                        resolved_methods.add(method)
                        active_units.pop(method, None)
                    for method in [BACKEND_SPECS[name].display_name for name in backends]:
                        if method in resolved_methods:
                            continue
                        not_started = reporter.start_unit(dataset=dataset, seed=seed, method=method, cell_count=cell_count, gene_count=gene_count, output_path=run_dir, prefix=f"{method} (prerequisite failed)")
                        reporter.fail_unit(not_started, f"prerequisite failed before {method}: {exc}")
                        resolved_methods.add(method)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "failure.json").write_text(json.dumps({"error": str(exc)}, indent=2), encoding="utf-8")
                    statuses.append({"dataset": dataset, "seed": seed, "status": "FAILED", "message": str(exc), "runtime_seconds": time.perf_counter() - run_started})
                    if not args.continue_on_error:
                        raise
                pd.DataFrame(statuses).to_csv(output / "controlled_run_status.csv", index=False)
        combined = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
        combined.to_csv(output / "controlled_metrics_all_runs.csv", index=False)
    finally:
        reporter.close()


if __name__ == "__main__":
    main()
