"""Run external methods on the frozen held-out controlled semi-real split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from duodose.protocol import load_final_protocol  # noqa: E402
from duodose.progress import ProgressReporter, ProgressSettings, add_progress_arguments, progress_paths  # noqa: E402
from duodose.semireal_metrics import high_rna_operating_point_metrics  # noqa: E402
from reproducibility.lib.common import (  # noqa: E402
    controlled_metric_row,
    load_dataset_exact,
    run_external_scores,
    run_protocol_models,
    sha256_file,
    split_csv,
    write_output_manifest,
    write_run_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conversion-dir", required=True)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--methods", default="Scrublet,scDblFinder,DoubletFinder,scds")
    parser.add_argument("--convert-rds", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    add_progress_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    protocol = load_final_protocol(args.protocol)
    methods = split_csv(args.methods)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ledger_path, snapshot_path = progress_paths(output, args)
    reporter = ProgressReporter(
        stage="external_baselines",
        total_units=len(methods),
        settings=ProgressSettings.from_args(args),
        ledger_path=ledger_path,
        snapshot_path=snapshot_path,
        config_hash=str(args.progress_config_hash or sha256_file(Path(protocol["_protocol_path"]))),
        output_path=output,
    )
    loaded = load_dataset_exact(
        args.data_dir,
        args.dataset,
        conversion_dir=args.conversion_dir,
        convert_rds=bool(args.convert_rds),
    )
    active_units: dict[str, float] = {}
    resolved_methods: set[str] = set()

    def progress_callback(event):
        method = str(event.get("method", ""))
        if event.get("event") == "method_start" and method:
            active_units[method] = reporter.start_unit(
                dataset=args.dataset,
                seed=args.seed,
                method=method,
                cell_count=int(loaded.adata.n_obs),
                gene_count=int(loaded.adata.n_vars),
                output_path=output,
                log_path=event.get("log_path", ""),
                prefix=f"[Method {methods.index(method) + 1}/{len(methods)}] {method}",
            )
        elif event.get("event") == "method_complete" and method in active_units:
            started = active_units.pop(method)
            if str(event.get("status", "success")).lower() == "success":
                reporter.complete_unit(started, message=f"completed {method}")
            else:
                reporter.fail_unit(started, str(event.get("message", event.get("status", "failed"))))
            resolved_methods.add(method)
        else:
            reporter.callback(event)

    try:
        reporter.event(f"[Dataset 1/1] {args.dataset}", dataset=args.dataset, seed=args.seed)
        reporter.event(f"[Seed 1/1] {args.seed}", dataset=args.dataset, seed=args.seed)
        run = run_protocol_models(loaded, protocol_path=args.protocol, seed=args.seed, backends=(), progress_callback=progress_callback)
        scores, statuses = run_external_scores(
            run.bundle.test_adata,
            dataset=f"{args.dataset}_controlled_test",
            seed=args.seed,
            methods=methods,
            cache_dir=output / "score_cache",
            expected_doublet_rate=float(protocol["prediction"]["expected_doublet_rate"]),
            refresh_cache=bool(args.refresh_cache),
            progress_callback=progress_callback,
            log_dir=output / "logs",
        )
        labels = run.test_scores["true_label"].astype(str)
        rows = []
        for method, score in scores.items():
            status = statuses.loc[statuses["method"].eq(method)].iloc[0]
            rows.append(
                controlled_metric_row(
                    dataset=args.dataset,
                    seed=args.seed,
                    method=method,
                    labels=labels,
                    obs=run.bundle.test_adata.obs,
                    overall_score=score,
                    homotypic_score=score,
                    heterotypic_score=score,
                    status=str(status["status"]),
                    message=str(status["message"]),
                    runtime_seconds=float(status["runtime_seconds"]),
                )
            )
        reporter.event("writing external benchmark outputs", dataset=args.dataset, seed=args.seed)
        pd.DataFrame(rows).to_csv(output / "external_controlled_metrics.csv", index=False)
        operating_obs = run.bundle.test_adata.obs.reindex(labels.index).copy()
        operating_obs["true_label"] = labels
        if "is_high_rna_singlet" not in operating_obs:
            operating_obs["is_high_rna_singlet"] = labels.eq("high_RNA_singlet")
        high_rna_operating_point_metrics(
            operating_obs,
            {method: score.reindex(labels.index) for method, score in scores.items()},
            dataset=args.dataset,
            source_dataset=args.dataset,
            seed=args.seed,
        ).to_csv(output / "external_high_RNA_operating_points.csv", index=False)
        statuses.to_csv(output / "external_method_status.csv", index=False)
        write_run_manifest(output, workflow="external_benchmark", protocol=run.protocol, dataset=args.dataset, seed=args.seed, runtime_seconds=run.runtime_seconds + float(statuses["runtime_seconds"].sum()), source_path=run.source_path, extra={"methods": methods})
        write_output_manifest(output)
    except Exception as exc:
        for method, started in list(active_units.items()):
            reporter.fail_unit(started, exc)
            resolved_methods.add(method)
            active_units.pop(method, None)
        for method in methods:
            if method in resolved_methods:
                continue
            not_started = reporter.start_unit(dataset=args.dataset, seed=args.seed, method=method, cell_count=int(loaded.adata.n_obs), gene_count=int(loaded.adata.n_vars), output_path=output, prefix=f"{method} (prerequisite failed)")
            reporter.fail_unit(not_started, f"prerequisite failed before {method}: {exc}")
            resolved_methods.add(method)
        raise
    finally:
        reporter.close()


if __name__ == "__main__":
    main()
