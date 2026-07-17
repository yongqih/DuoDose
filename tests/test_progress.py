from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import time
from pathlib import Path

from duodose.progress import LEDGER_COLUMNS, ETAEstimator, ProgressReporter, ProgressSettings, RuntimeLedger, atomic_write_json, migrate_progress_artifacts


def _row(*, config_hash: str = "frozen", status: str = "COMPLETED", method: str = "DuoDose", cells: int = 1000, elapsed: float = 10.0):
    return {
        "analysis_stage": "controlled_benchmark",
        "method": method,
        "cell_count": cells,
        "elapsed_seconds": elapsed,
        "status": status,
        "config_hash": config_hash,
        "reused_from_cache": False,
    }


def test_eta_is_unavailable_before_two_compatible_observations() -> None:
    estimator = ETAEstimator([_row()], config_hash="frozen")
    estimate = estimator.estimate(stage="controlled_benchmark", method="DuoDose", cell_count=1000)
    assert estimate.seconds is None
    assert estimate.method == "insufficient_compatible_history"


def test_eta_prefers_same_method_and_similar_size_rolling_median() -> None:
    rows = [
        _row(cells=900, elapsed=8),
        _row(cells=1100, elapsed=12),
        _row(method="DuoDose-DL", cells=1000, elapsed=500),
        _row(config_hash="other", cells=1000, elapsed=900),
        _row(status="FAILED", cells=1000, elapsed=1000),
    ]
    estimate = ETAEstimator(rows, config_hash="frozen").estimate(stage="controlled_benchmark", method="DuoDose", cell_count=1000)
    assert estimate.seconds == 10
    assert estimate.method == "same_method_similar_size_rolling_median"
    assert estimate.observations == 2


def test_interrupted_running_ledger_row_becomes_incomplete(tmp_path: Path) -> None:
    ledger = RuntimeLedger(tmp_path / "runtime.csv")
    ledger.append(
        {
            "analysis_stage": "controlled_benchmark",
            "dataset": "fixture",
            "seed": 0,
            "method": "DuoDose",
            "start_time": "2026-01-01T00:00:00+00:00",
            "status": "RUNNING",
            "config_hash": "frozen",
        }
    )
    assert ledger.mark_running_incomplete(config_hash="frozen") == 1
    row = ledger.read()[0]
    assert row["status"] == "INCOMPLETE"
    assert "interrupted" in row["failure_reason"]


def test_reporter_counts_cached_completed_and_failed_separately(tmp_path: Path) -> None:
    stream = io.StringIO()
    reporter = ProgressReporter(
        stage="controlled_benchmark",
        total_units=3,
        settings=ProgressSettings(enabled=True, interactive=False, refresh_seconds=0.01),
        ledger_path=tmp_path / "runtime.csv",
        snapshot_path=tmp_path / "progress.json",
        config_hash="frozen",
        output_path=tmp_path,
        stream=stream,
    )
    reporter.cached_unit(dataset="fixture", seed=0, method="Scrublet")
    started = reporter.start_unit(dataset="fixture", seed=0, method="DuoDose")
    reporter.complete_unit(started)
    failed = reporter.start_unit(dataset="fixture", seed=0, method="DuoDose-DL")
    reporter.fail_unit(failed, "fixture failure")
    reporter.close()
    snapshot = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert snapshot["completed_units"] == 1
    assert snapshot["cached_units"] == 1
    assert snapshot["failed_units"] == 1
    assert snapshot["resolved_units"] == 3
    assert snapshot["progress_fraction"] == 1.0
    assert snapshot["current_status"] == "INCOMPLETE"
    assert "\x1b" not in stream.getvalue()
    statuses = [row["status"] for row in RuntimeLedger(tmp_path / "runtime.csv").read()]
    assert statuses == ["SKIPPED_VALID_CACHE", "COMPLETED", "FAILED"]


def test_progress_is_not_complete_with_unresolved_units(tmp_path: Path) -> None:
    reporter = ProgressReporter(
        stage="fixture",
        total_units=2,
        settings=ProgressSettings(enabled=False),
        ledger_path=tmp_path / "runtime.csv",
        snapshot_path=tmp_path / "progress.json",
        config_hash="frozen",
    )
    started = reporter.start_unit(method="one")
    reporter.complete_unit(started)
    reporter.close()
    snapshot = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert snapshot["progress_fraction"] == 0.5
    assert snapshot["current_status"] == "INCOMPLETE"


def test_atomic_json_replacement_leaves_no_temporary_file(tmp_path: Path) -> None:
    target = tmp_path / "formal_progress.json"
    for value in range(5):
        atomic_write_json(target, {"value": value})
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 4}
    assert list(tmp_path.glob(".*.tmp")) == []


def test_reporter_serializes_paths_relative_to_results_dir(tmp_path: Path) -> None:
    results_dir = tmp_path / "results" / "final_v1"
    output = results_dir / "validation_suite"
    log = results_dir / "logs" / "validation.log"
    reporter = ProgressReporter(
        stage="validation",
        total_units=1,
        settings=ProgressSettings(enabled=False),
        ledger_path=output / "runtime_ledger.csv",
        snapshot_path=output / "formal_progress.json",
        config_hash="frozen",
        output_path=output,
    )
    started = reporter.start_unit(method="schema", log_path=log)
    reporter.complete_unit(started)
    reporter.close()
    row = RuntimeLedger(output / "runtime_ledger.csv").read()[0]
    snapshot = json.loads((output / "formal_progress.json").read_text(encoding="utf-8"))
    assert row["output_path"] == "validation_suite"
    assert row["log_path"] == "logs/validation.log"
    assert snapshot["output_path"] == "validation_suite"
    assert snapshot["latest_log_path"] == "logs/validation.log"


def test_migrate_progress_artifacts_rewrites_paths_atomically(tmp_path: Path) -> None:
    results_dir = tmp_path / "results" / "final_v1"
    output = results_dir / "validation_suite"
    output.mkdir(parents=True)
    ledger_path = output / "runtime_ledger.csv"
    with ledger_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS)
        writer.writeheader()
        writer.writerow({"analysis_stage": "validation", "output_path": str(output), "log_path": str(results_dir / "logs" / "run.log")})
    snapshot_path = output / "formal_progress.json"
    atomic_write_json(snapshot_path, {"output_path": str(output), "latest_log_path": str(results_dir / "logs" / "run.log")})
    migrate_progress_artifacts(ledger_path=ledger_path, snapshot_path=snapshot_path, results_dir=results_dir)
    row = RuntimeLedger(ledger_path).read()[0]
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert row["output_path"] == "validation_suite"
    assert row["log_path"] == "logs/run.log"
    assert snapshot == {"output_path": "validation_suite", "latest_log_path": "logs/run.log"}
    assert list(output.glob(".*.tmp")) == []


def test_heartbeat_wrapper_streams_output_and_propagates_failure(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "reproducibility" / "run_with_heartbeat.py"
    log = tmp_path / "child.log"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--log-file",
            str(log),
            "--label",
            "fixture",
            "--heartbeat-seconds",
            "1",
            "--",
            sys.executable,
            "-c",
            "import time; print('child started', flush=True); time.sleep(0.05); raise SystemExit(7)",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 7
    assert "launched PID" in completed.stdout
    assert "exited 7" in completed.stdout
    assert "child started" in log.read_text(encoding="utf-8")


def test_atomic_json_retries_transient_windows_lock(tmp_path: Path, monkeypatch) -> None:
    import duodose.progress as progress_module

    target = tmp_path / "formal_progress.json"
    real_replace = progress_module.os.replace
    calls = {"count": 0}

    def flaky_replace(source, destination):
        calls["count"] += 1
        if calls["count"] < 3:
            error = PermissionError(13, "Access is denied", str(destination))
            error.winerror = 5
            raise error
        return real_replace(source, destination)

    monkeypatch.setattr(progress_module.os, "replace", flaky_replace)
    atomic_write_json(target, {"status": "ok"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "ok"}
    assert calls["count"] == 3


def test_progress_snapshot_lock_does_not_fail_scientific_unit(tmp_path: Path, monkeypatch) -> None:
    import duodose.progress as progress_module

    reporter = ProgressReporter(
        stage="controlled_benchmark",
        total_units=1,
        settings=ProgressSettings(enabled=False),
        ledger_path=tmp_path / "runtime.csv",
        snapshot_path=tmp_path / "formal_progress.json",
        config_hash="frozen",
    )

    def permanently_locked(*args, **kwargs):
        error = PermissionError(13, "Access is denied", "formal_progress.json")
        error.winerror = 5
        raise error

    monkeypatch.setattr(progress_module, "atomic_write_json", permanently_locked)
    # Snapshot telemetry failure must not raise or invalidate the model unit.
    reporter.snapshot(status="RUNNING")
