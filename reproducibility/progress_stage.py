"""Small bridge used by the PowerShell master runner for stage snapshots."""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from duodose.progress import ETAEstimator, RuntimeLedger, atomic_write_json, code_commit, format_duration, iso_now, portable_result_path, utc_now  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["start", "finish"], required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--stage-number", type=int, required=True)
    parser.add_argument("--stage-total", type=int, required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--output-path", default="")
    parser.add_argument("--log-path", nargs="?", const="", default="")
    parser.add_argument("--status", default="RUNNING")
    parser.add_argument("--start-time", default="")
    parser.add_argument("--elapsed-seconds", type=float, default=0.0)
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--failure-reason", nargs="?", const="", default="")
    parser.add_argument("--remaining-stages", nargs="?", const="", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results_dir = Path(args.ledger).expanduser().resolve().parent
    ledger = RuntimeLedger(args.ledger, results_dir=results_dir)
    if args.action == "start":
        ledger.mark_running_incomplete(config_hash=args.config_hash, analysis_stage=args.stage)
        ledger.append(
            {
                "analysis_stage": args.stage,
                "start_time": args.start_time,
                "status": "RUNNING",
                "config_hash": args.config_hash,
                "code_commit": code_commit(),
                "output_path": args.output_path,
                "log_path": args.log_path,
                "reused_from_cache": False,
            }
        )
    estimator = ETAEstimator(ledger.read(), config_hash=args.config_hash)
    remaining = [value for value in args.remaining_stages.split(",") if value]
    estimate = estimator.estimate_tasks({"analysis_stage": stage} for stage in remaining)
    completion = utc_now() + timedelta(seconds=estimate.seconds) if estimate.seconds is not None else None
    status = str(args.status).upper()
    if args.action == "finish":
        ledger.finish(
            {
                "analysis_stage": args.stage,
                "start_time": args.start_time,
                "end_time": iso_now(),
                "elapsed_seconds": round(max(0.0, args.elapsed_seconds), 6),
                "status": status,
                "exit_code": args.exit_code,
                "config_hash": args.config_hash,
                "code_commit": code_commit(),
                "output_path": args.output_path,
                "log_path": args.log_path,
                "reused_from_cache": status == "SKIPPED_VALID_CACHE",
                "failure_reason": args.failure_reason,
            }
        )
    atomic_write_json(
        args.snapshot,
        {
            "current_stage": args.stage,
            "current_stage_number": args.stage_number,
            "total_stages": args.stage_total,
            "current_dataset": "",
            "current_seed": "",
            "current_method": "",
            "completed_units": args.stage_number - (0 if status in {"COMPLETED", "SKIPPED_VALID_CACHE"} else 1),
            "total_units": args.stage_total,
            "elapsed_seconds": round(max(0.0, args.elapsed_seconds), 3),
            "estimated_remaining_seconds": estimate.seconds,
            "eta_estimation_method": estimate.method,
            "estimated_completion_time": completion.isoformat(timespec="seconds") if completion else None,
            "current_status": status,
            "latest_log_path": portable_result_path(args.log_path, results_dir),
            "output_path": portable_result_path(args.output_path, results_dir),
            "last_update_time": iso_now(),
        },
    )
    print(
        f"[Stage {args.stage_number}/{args.stage_total}] {args.stage} | {status} | start {args.start_time or 'unknown'} | "
        f"elapsed {format_duration(args.elapsed_seconds)} | overall ETA {format_duration(estimate.seconds)} "
        f"({estimate.method}) | output {args.output_path} | log {args.log_path or 'pending'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
