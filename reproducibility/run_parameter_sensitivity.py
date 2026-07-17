"""Run the compact predeclared RF parameter-sensitivity analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from duodose.net import probabilities_to_scores  # noqa: E402
from duodose.parameter_sensitivity_audit import (  # noqa: E402
    aggregate_sensitivity,
    deterministic_top_fraction,
    sensitivity_run_status,
    training_size_protocol,
)
from duodose.protocol import load_final_protocol  # noqa: E402
from duodose.progress import ProgressReporter, ProgressSettings, add_progress_arguments, progress_paths  # noqa: E402
from reproducibility.lib.common import (  # noqa: E402
    evaluate_internal_controlled,
    load_dataset_exact,
    run_protocol_models,
    sha256_file,
    write_output_manifest,
    write_run_manifest,
)


def _candidate_fpr(labels: pd.Series, score: pd.Series, fraction: float) -> float:
    return float(deterministic_top_fraction(labels, score, fraction)["high_RNA_singlet_FPR"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conversion-dir", required=True)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--convert-rds", action="store_true")
    add_progress_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    protocol = load_final_protocol(args.protocol)
    loaded = load_dataset_exact(args.data_dir, args.dataset, conversion_dir=args.conversion_dir, convert_rds=bool(args.convert_rds))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ledger_path, snapshot_path = progress_paths(output, args)
    factors = list(protocol["parameter_sensitivity"]["semi_real_size_factors"])
    seeds = list(protocol["seeds"]["parameter_sensitivity"])
    reporter = ProgressReporter(
        stage="parameter_sensitivity",
        total_units=len(seeds) * len(factors),
        settings=ProgressSettings.from_args(args),
        ledger_path=ledger_path,
        snapshot_path=snapshot_path,
        config_hash=str(args.progress_config_hash or sha256_file(Path(protocol["_protocol_path"]))),
        output_path=output,
    )
    rows = []
    try:
        for seed_index, seed in enumerate(seeds, start=1):
            for factor_index, factor in enumerate(factors, start=1):
                unit_started = reporter.start_unit(
                    dataset=args.dataset,
                    seed=int(seed),
                    method="DuoDose",
                    cell_count=int(loaded.adata.n_obs),
                    gene_count=int(loaded.adata.n_vars),
                    output_path=output,
                    prefix=f"[semi_real_size_factor {factor_index}/{len(factors)}={factor} | seed {seed_index}/{len(seeds)}={seed}] {args.dataset}",
                )
                try:
                    varied = training_size_protocol(protocol, float(factor))
                    run = run_protocol_models(
                        loaded,
                        protocol_path=args.protocol,
                        seed=int(seed),
                        backends=("rf",),
                        device="cpu",
                        protocol_override=varied,
                        use_explicit_validation_sizes=True,
                        progress_callback=reporter.callback,
                        verbose_progress=bool(args.verbose_progress),
                    )
                    base = evaluate_internal_controlled(run).iloc[0].to_dict()
                    probabilities = run.method_probabilities_test["DuoDose"]
                    overall, _, _ = probabilities_to_scores(probabilities)
                    labels = run.test_scores["true_label"].astype(str)
                    for rate in protocol["parameter_sensitivity"]["expected_doublet_rates"]:
                        reporter.event(f"expected_doublet_rate={rate} | dataset={args.dataset} | seed={seed} | factor={factor}", force=bool(args.verbose_progress))
                        rows.append({**base, "semi_real_size_factor": factor, "expected_doublet_rate": rate, "high_RNA_singlet_FPR_at_expected_rate": _candidate_fpr(labels, overall, float(rate)), "fit_runtime_seconds": run.runtime_seconds})
                    reporter.complete_unit(unit_started, message=f"completed semi_real_size_factor={factor}, seed={seed}")
                except Exception as exc:
                    reporter.fail_unit(unit_started, exc)
                    raise
    finally:
        reporter.close()
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "parameter_sensitivity_by_run.csv", index=False)
    summary = aggregate_sensitivity(frame)
    summary.to_csv(output / "parameter_sensitivity_summary.csv", index=False)
    sensitivity_run_status(
        frame,
        dataset=args.dataset,
        seeds=seeds,
        factors=factors,
        expected_rates=list(protocol["parameter_sensitivity"]["expected_doublet_rates"]),
    ).to_csv(output / "parameter_sensitivity_run_status.csv", index=False)
    write_run_manifest(output, workflow="parameter_sensitivity", protocol=protocol, dataset=args.dataset, seed=0, runtime_seconds=float(frame.drop_duplicates(["seed", "semi_real_size_factor"])["fit_runtime_seconds"].sum()), source_path=loaded.source_path)
    write_output_manifest(output)


if __name__ == "__main__":
    main()
