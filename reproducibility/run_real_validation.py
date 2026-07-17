"""Evaluate frozen methods on experimental doublet-enriched labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from duodose.protocol import load_final_protocol  # noqa: E402
from duodose.models.registry import BACKEND_SPECS  # noqa: E402
from duodose.progress import ProgressReporter, ProgressSettings, add_progress_arguments, progress_paths  # noqa: E402
from reproducibility.lib.common import (  # noqa: E402
    _safe_metric,
    evaluate_real_probabilities,
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
    parser.add_argument("--backends", default="rf,dl")
    parser.add_argument("--external-methods", default="Scrublet,scDblFinder,DoubletFinder,scds")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dl-max-epochs", type=int, default=200)
    parser.add_argument("--dl-patience", type=int, default=20)
    parser.add_argument("--convert-rds", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    add_progress_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    protocol = load_final_protocol(args.protocol)
    backends = split_csv(args.backends)
    methods = split_csv(args.external_methods)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ledger_path, snapshot_path = progress_paths(output, args)
    reporter = ProgressReporter(
        stage="real_data_validation",
        total_units=len(backends) + len(methods),
        settings=ProgressSettings.from_args(args),
        ledger_path=ledger_path,
        snapshot_path=snapshot_path,
        config_hash=str(args.progress_config_hash or sha256_file(Path(protocol["_protocol_path"]))),
        output_path=output,
    )
    loaded = load_dataset_exact(args.data_dir, args.dataset, conversion_dir=args.conversion_dir, convert_rds=bool(args.convert_rds))
    method_order = [BACKEND_SPECS[name].display_name for name in backends] + methods
    active_units: dict[str, float] = {}
    resolved_methods: set[str] = set()

    def progress_callback(event):
        method = str(event.get("method", ""))
        if event.get("event") == "method_start" and method and method not in active_units:
            active_units[method] = reporter.start_unit(
                dataset=args.dataset,
                seed=args.seed,
                method=method,
                cell_count=int(loaded.adata.n_obs),
                gene_count=int(loaded.adata.n_vars),
                output_path=output,
                log_path=event.get("log_path", ""),
                prefix=f"[Method {method_order.index(method) + 1}/{len(method_order)}] {method}",
            )
        elif event.get("event") == "method_complete" and method in active_units:
            status = str(event.get("status", "success")).lower()
            started = active_units.pop(method)
            if status in {"success", "completed"}:
                reporter.complete_unit(started, message=f"completed {method}")
            else:
                reporter.fail_unit(started, str(event.get("message", status)))
            resolved_methods.add(method)
        else:
            reporter.callback(event)

    try:
        reporter.event(f"[Dataset 1/1] {args.dataset}", dataset=args.dataset, seed=args.seed)
        reporter.event("[Seed 1/1] " + str(args.seed), dataset=args.dataset, seed=args.seed)
        run = run_protocol_models(
            loaded,
            protocol_path=args.protocol,
            seed=args.seed,
            backends=backends,
            device=args.device,
            amp=bool(args.amp),
            dl_max_epochs=args.dl_max_epochs,
            dl_patience=args.dl_patience,
            progress_callback=progress_callback,
            verbose_progress=bool(args.verbose_progress),
        )
        metrics = evaluate_real_probabilities(run)
        external_scores, statuses = run_external_scores(
            run.original_adata,
            dataset=args.dataset,
            seed=args.seed,
            methods=methods,
            cache_dir=output / "score_cache",
            expected_doublet_rate=float(protocol["prediction"]["expected_doublet_rate"]),
            refresh_cache=bool(args.refresh_cache),
            progress_callback=progress_callback,
            log_dir=output / "logs",
        )
        labels = run.original_adata.obs["experimental_doublet"].astype(int)
        external_rows = []
        for method, score in external_scores.items():
            status = statuses.loc[statuses["method"].eq(method)].iloc[0]
            values = score.reindex(labels.index).to_numpy(dtype=float)
            external_rows.append(
                {
                    "dataset": args.dataset,
                    "seed": args.seed,
                    "method": method,
                    "AUROC": _safe_metric(roc_auc_score, labels.to_numpy(), values),
                    "AUPRC": _safe_metric(average_precision_score, labels.to_numpy(), values),
                    "n_positive": int(labels.eq(1).sum()),
                    "n_negative": int(labels.eq(0).sum()),
                    "n_excluded": 0,
                    "status": status["status"],
                    "message": status["message"],
                    "paper_metric_scope": "experimental doublet-enriched detection",
                    "paper_metric_definition": "positive=experimentally annotated doublet; negative=all annotated non-doublets; experimental labels evaluation-only",
                }
            )
        if external_rows:
            metrics = pd.concat([metrics, pd.DataFrame(external_rows)], ignore_index=True)
        reporter.event("writing real-validation outputs", dataset=args.dataset, seed=args.seed)
        metrics.to_csv(output / "real_doublet_enriched_metrics.csv", index=False)
        statuses.to_csv(output / "external_method_status.csv", index=False)
        pd.DataFrame(run.training_summaries).to_csv(output / "training_summaries.csv", index=False)
        write_run_manifest(output, workflow="real_validation", protocol=run.protocol, dataset=args.dataset, seed=args.seed, runtime_seconds=run.runtime_seconds + float(statuses["runtime_seconds"].sum()), source_path=run.source_path, extra={"backends": backends, "external_methods": methods})
        write_output_manifest(output)
    except Exception as exc:
        for method, started in list(active_units.items()):
            reporter.fail_unit(started, exc)
            resolved_methods.add(method)
            active_units.pop(method, None)
        for method in method_order:
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
