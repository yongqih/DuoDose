"""Prepare canonical inputs and run the corrected strict real/semi-real audit."""

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

from duodose.domain_audit_aggregate import regenerate_domain_audit_outputs  # noqa: E402
from duodose.domain_audit_contract import PRIMARY_ANALYSIS, validate_primary_audit_files  # noqa: E402
from duodose.protocol import load_final_protocol  # noqa: E402
from duodose.progress import ProgressReporter, ProgressSettings, add_progress_arguments, progress_paths  # noqa: E402
from duodose.semireal_real_domain_audit import run_domain_audit, validate_domain_audit_bundle  # noqa: E402
from reproducibility.lib.common import (  # noqa: E402
    export_domain_bundle,
    load_dataset_exact,
    run_protocol_models,
    sha256_file,
    split_csv,
    write_output_manifest,
    write_run_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--conversion-dir", required=True)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--max-cells-per-domain", type=int, default=2000)
    parser.add_argument("--convert-rds", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse dataset audits with a complete strict output contract.")
    add_progress_arguments(parser)
    return parser


def _completed_dataset_audit(dataset_output: Path) -> bool:
    required = (
        "domain_audit_config.json",
        "domain_audit_summary.csv",
        "domain_audit_feature_audit.csv",
        "domain_audit_parent_unique_selection.csv",
        "domain_audit_predictions.csv",
        "domain_audit_report.md",
    )
    if any(not (dataset_output / name).is_file() or (dataset_output / name).stat().st_size == 0 for name in required):
        return False
    try:
        summary = pd.read_csv(dataset_output / "domain_audit_summary.csv")
        config = json.loads((dataset_output / "domain_audit_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return False
    primary = summary.loc[summary.get("is_primary", pd.Series(False, index=summary.index)).astype(bool)]
    return bool(
        not primary.empty
        and config.get("schema_version") == "semireal_real_domain_audit_v2"
        and config.get("formal_construction_variant") == "raw_sum_parents_removed"
        and config.get("formal_safe_feature_mode") == "fitted_reference"
        and config.get("primary_analysis") == PRIMARY_ANALYSIS
        and validate_primary_audit_files(dataset_output).completed
        and primary.get("construction_variant", pd.Series(dtype=str)).astype(str).eq("raw_sum_parents_removed").all()
        and primary.get("safe_feature_mode", pd.Series(dtype=str)).astype(str).eq("fitted_reference").all()
        and pd.to_numeric(primary.get("parent_overlap_across_folds"), errors="coerce").eq(0).all()
    )


def main() -> None:
    args = build_parser().parse_args()
    protocol = load_final_protocol(args.protocol)
    datasets = split_csv(args.datasets)
    if datasets == ["all"]:
        datasets = list(protocol["datasets"]["real_doublet_enriched"])
    output = Path(args.output_dir)
    cache = Path(args.cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    ledger_path, snapshot_path = progress_paths(output, args)
    reporter = ProgressReporter(
        stage="domain_audit",
        total_units=len(datasets),
        settings=ProgressSettings.from_args(args),
        ledger_path=ledger_path,
        snapshot_path=snapshot_path,
        config_hash=str(args.progress_config_hash or sha256_file(Path(protocol["_protocol_path"]))),
        output_path=output,
    )
    statuses: list[dict[str, object]] = []
    completed_cache = {dataset for dataset in datasets if args.resume and _completed_dataset_audit(output / dataset)}
    if args.resume:
        reporter.event(f"Resume scan complete: valid completed runs={len(completed_cache)}, remaining={len(datasets) - len(completed_cache)}")
    for dataset_index, dataset in enumerate(datasets, start=1):
        started = time.perf_counter()
        dataset_output = output / dataset
        bundle_dir = cache / dataset / "seed_0" / "domain_audit"
        reporter.event(f"[Dataset {dataset_index}/{len(datasets)}] {dataset}", dataset=dataset, seed=0)
        if dataset in completed_cache:
            reporter.cached_unit(dataset=dataset, seed=0, method="strict_domain_audit", output_path=dataset_output)
            statuses.append({"dataset": dataset, "status": "SUCCESS", "message": "reused validated output", "runtime_seconds": 0.0, "cache_path": str(bundle_dir), "output_path": str(dataset_output)})
            pd.DataFrame(statuses).to_csv(output / "domain_audit_batch_run_status.csv", index=False)
            continue
        unit_started = reporter.start_unit(dataset=dataset, seed=0, method="strict_domain_audit", output_path=dataset_output)
        try:
            valid_cached = False
            reporter.event(f"{dataset}: checking cached audit bundle", dataset=dataset, seed=0)
            if (bundle_dir / "domain_audit_export_manifest.json").is_file():
                try:
                    validate_domain_audit_bundle(bundle_dir)
                    valid_cached = True
                    print(f"[{dataset}] reusing validated cached audit bundle")
                except Exception:
                    valid_cached = False
            if not valid_cached:
                reporter.event(f"{dataset}: rebuilding canonical audit bundle", dataset=dataset, seed=0)
                loaded = load_dataset_exact(args.data_dir, dataset, conversion_dir=args.conversion_dir, convert_rds=bool(args.convert_rds))
                run = run_protocol_models(loaded, protocol_path=args.protocol, seed=0, backends=(), progress_callback=reporter.callback)
                export_domain_bundle(run, bundle_dir)
                validate_domain_audit_bundle(bundle_dir)
                write_run_manifest(bundle_dir, workflow="domain_audit_input_export", protocol=run.protocol, dataset=dataset, seed=0, runtime_seconds=run.runtime_seconds, source_path=run.source_path)
                write_output_manifest(bundle_dir)
            reporter.event(f"{dataset}: provenance validation", dataset=dataset, seed=0)
            run_domain_audit([bundle_dir], dataset_output, max_cells_per_domain=args.max_cells_per_domain, progress_callback=reporter.callback)
            statuses.append({"dataset": dataset, "status": "SUCCESS", "message": "", "runtime_seconds": time.perf_counter() - started, "cache_path": str(bundle_dir), "output_path": str(dataset_output)})
            reporter.complete_unit(unit_started, message=f"completed domain audit for {dataset}")
        except Exception as exc:
            reporter.fail_unit(unit_started, exc)
            statuses.append({"dataset": dataset, "status": "FAILED", "message": str(exc), "runtime_seconds": time.perf_counter() - started, "cache_path": str(bundle_dir), "output_path": str(dataset_output)})
            if not args.continue_on_error:
                pd.DataFrame(statuses).to_csv(output / "domain_audit_batch_run_status.csv", index=False)
                reporter.close(status="FAILED")
                raise
        pd.DataFrame(statuses).to_csv(output / "domain_audit_batch_run_status.csv", index=False)
    try:
        reporter.event("updating combined domain-audit summary")
        regenerate_domain_audit_outputs(output, cache)
        write_output_manifest(output)
    finally:
        reporter.close()


if __name__ == "__main__":
    main()
