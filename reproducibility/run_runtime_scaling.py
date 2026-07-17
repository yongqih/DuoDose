"""Measure runtime and memory scaling under the frozen protocol."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from reproducibility.lib.common import (  # noqa: E402
    LoadedDataset,
    load_dataset_exact,
    run_external_scores,
    run_protocol_models,
    sha256_file,
    split_csv,
    write_output_manifest,
    write_run_manifest,
)
from duodose.progress import ProgressReporter, ProgressSettings, add_progress_arguments, progress_paths  # noqa: E402


class MemoryMonitor:
    def __init__(self) -> None:
        self.peak = float("nan")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        try:
            import psutil

            process = psutil.Process()
            self.peak = float(process.memory_info().rss)

            def sample() -> None:
                while not self._stop.wait(0.05):
                    self.peak = max(self.peak, float(process.memory_info().rss))

            self._thread = threading.Thread(target=sample, daemon=True)
            self._thread.start()
        except ImportError:
            pass
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conversion-dir", required=True)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--cell-counts", default="5000,10000,20000,largest_feasible")
    parser.add_argument("--methods", default="DuoDose,DuoDose-DL,Scrublet,scDblFinder")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1, help="Requested orchestration/BLAS thread count to record.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--convert-rds", action="store_true")
    add_progress_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be at least 1")
    load_started = time.perf_counter()
    loaded = load_dataset_exact(args.data_dir, args.dataset, conversion_dir=args.conversion_dir, convert_rds=bool(args.convert_rds))
    load_seconds = time.perf_counter() - load_started
    requested = split_csv(args.cell_counts)
    counts = [loaded.adata.n_obs if item == "largest_feasible" else min(int(item), loaded.adata.n_obs) for item in requested]
    counts = list(dict.fromkeys(counts))
    methods = split_csv(args.methods)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    from duodose.protocol import load_final_protocol

    protocol = load_final_protocol(args.protocol)
    ledger_path, snapshot_path = progress_paths(output, args)
    reporter = ProgressReporter(
        stage="runtime_benchmark",
        total_units=len(counts) * args.repetitions * len(methods),
        settings=ProgressSettings.from_args(args),
        ledger_path=ledger_path,
        snapshot_path=snapshot_path,
        config_hash=str(args.progress_config_hash or sha256_file(Path(protocol["_protocol_path"]))),
        output_path=output,
    )
    rows: list[dict[str, object]] = []
    try:
        for scale_index, n_cells in enumerate(counts, start=1):
            for repetition in range(args.repetitions):
                rng = np.random.default_rng(10000 * n_cells + repetition)
                positions = np.sort(rng.choice(loaded.adata.n_obs, size=n_cells, replace=False))
                subset = loaded.adata[positions, :].copy()
                local = LoadedDataset(loaded.dataset, subset, loaded.source_path, loaded.source_format, loaded.label_source, loaded.conversion_status)
                for method_index, method in enumerate(methods, start=1):
                    unit_started = reporter.start_unit(
                        dataset=args.dataset,
                        seed=repetition,
                        method=method,
                        cell_count=n_cells,
                        gene_count=int(subset.n_vars),
                        output_path=output,
                        prefix=f"[Scale {scale_index}/{len(counts)} | repetition {repetition + 1}/{args.repetitions} | method {method_index}/{len(methods)}] {method}",
                    )
                    gpu_peak = float("nan")
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()
                    except ImportError:
                        torch = None
                    try:
                        with MemoryMonitor() as memory:
                            started = time.perf_counter()
                            if method in {"DuoDose", "DuoDose-DL"}:
                                backend = "rf" if method == "DuoDose" else "dl"
                                run = run_protocol_models(
                                    local,
                                    protocol_path=args.protocol,
                                    seed=repetition,
                                    backends=(backend,),
                                    device=args.device,
                                    amp=bool(args.amp),
                                    progress_callback=reporter.callback,
                                    verbose_progress=bool(args.verbose_progress),
                                )
                                timings = run.timings
                                method_status = "success"
                                method_message = ""
                            else:
                                _, status = run_external_scores(
                                    subset,
                                    dataset=f"{args.dataset}_{n_cells}",
                                    seed=repetition,
                                    methods=(method,),
                                    cache_dir=output / "score_cache",
                                    expected_doublet_rate=0.08,
                                    refresh_cache=True,
                                    progress_callback=reporter.callback,
                                    log_dir=output / "logs" / f"n_{n_cells}" / f"repetition_{repetition}",
                                )
                                timings = {"model_training_seconds": 0.0, "prediction_seconds": float(status.iloc[0]["runtime_seconds"]), "safe_feature_construction_seconds": 0.0, "semi_real_construction_seconds": 0.0}
                                method_status = str(status.iloc[0]["status"])
                                method_message = str(status.iloc[0]["message"])
                            total = time.perf_counter() - started
                        if torch is not None and torch.cuda.is_available():
                            gpu_peak = float(torch.cuda.max_memory_allocated())
                        rows.append({"dataset": args.dataset, "n_cells": n_cells, "n_genes": subset.n_vars, "method": method, "repetition": repetition, "status": method_status, "message": method_message, "data_loading_seconds": load_seconds, "loading_preprocessing_seconds": load_seconds + float(timings.get("semi_real_construction_seconds", 0.0)), **timings, "total_wall_clock_seconds": total, "peak_ram_bytes": memory.peak, "peak_gpu_memory_bytes": gpu_peak})
                        pd.DataFrame(rows).to_csv(output / "runtime_scaling_by_run.csv", index=False)
                        if method_status.lower() == "success":
                            reporter.complete_unit(unit_started, message=f"completed {method} at {n_cells} cells")
                        else:
                            reporter.fail_unit(unit_started, method_message)
                    except Exception as exc:
                        reporter.fail_unit(unit_started, exc)
                        raise
    finally:
        reporter.close()
    frame = pd.DataFrame(rows)
    numeric = [column for column in frame.columns if column.endswith("seconds") or column.endswith("bytes")]
    summary = frame.groupby(["dataset", "n_cells", "n_genes", "method"], as_index=False)[numeric].agg(["mean", "std"])
    summary.columns = ["_".join(str(part) for part in column if part) for column in summary.columns.to_flat_index()]
    summary.to_csv(output / "runtime_scaling_summary.csv", index=False)
    thread_configuration = {
        "logical_cpu_count": os.cpu_count(),
        "n_jobs_requested": int(args.n_jobs),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
        "sklearn_rf_n_jobs": -1,
        "scrublet_wrapper_n_jobs": 1,
    }
    write_run_manifest(output, workflow="runtime_scaling", protocol=protocol, dataset=args.dataset, seed=0, runtime_seconds=float(frame["total_wall_clock_seconds"].sum()), source_path=loaded.source_path, extra={"methods": methods, "cell_counts": counts, "repetitions": args.repetitions, "n_jobs_requested": int(args.n_jobs), "thread_configuration": thread_configuration, "device_requested": args.device, "amp_requested": bool(args.amp)})
    write_output_manifest(output)
    if not frame["status"].astype(str).str.lower().eq("success").all():
        failures = frame.loc[~frame["status"].astype(str).str.lower().eq("success"), ["method", "message"]]
        raise RuntimeError("required runtime method failed: " + "; ".join(f"{row.method}: {row.message}" for row in failures.itertuples()))


if __name__ == "__main__":
    main()
