"""Progress, runtime-history, and live-status helpers for formal workflows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, TextIO


LEDGER_COLUMNS = (
    "analysis_stage",
    "dataset",
    "seed",
    "method",
    "cell_count",
    "gene_count",
    "start_time",
    "end_time",
    "elapsed_seconds",
    "status",
    "exit_code",
    "config_hash",
    "code_commit",
    "output_path",
    "log_path",
    "reused_from_cache",
    "failure_reason",
)

TERMINAL_SUCCESS_STATUSES = {"COMPLETED", "SUCCESS", "SKIPPED_VALID_CACHE", "CACHED"}
FAILED_STATUSES = {"FAILED", "INCOMPLETE", "INTERRUPTED"}
_CODE_COMMIT: str | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat(timespec="seconds")


def code_commit() -> str:
    global _CODE_COMMIT
    if _CODE_COMMIT is not None:
        return _CODE_COMMIT
    configured = os.environ.get("DUODOSE_CODE_COMMIT", "").strip()
    if configured:
        _CODE_COMMIT = configured
        return configured
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        _CODE_COMMIT = completed.stdout.strip() if completed.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        _CODE_COMMIT = "unknown"
    return _CODE_COMMIT


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "estimating..."
    try:
        value = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        return "estimating..."
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _replace_with_retry(temporary: Path, target: Path, *, attempts: int = 40) -> None:
    """Replace *target* robustly when Windows briefly locks the destination.

    Antivirus, file indexing, PowerShell preview, or another progress reader can
    transiently hold ``formal_progress.json`` without delete sharing.  A live
    progress snapshot must never abort model training, so retry the atomic
    replace for several seconds before surfacing a genuine filesystem error.
    """

    delay = 0.01
    last_error: OSError | None = None
    for attempt in range(max(1, int(attempts))):
        try:
            os.replace(temporary, target)
            return
        except OSError as exc:
            retryable = isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in {5, 32, 33}
            if not retryable or attempt + 1 >= attempts:
                raise
            last_error = exc
            time.sleep(delay)
            delay = min(0.25, delay * 1.6)
    if last_error is not None:  # pragma: no cover - loop either returns or raises
        raise last_error


def _atomic_replace_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        _replace_with_retry(temporary, path)
    finally:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            # A leftover uniquely named temp file is harmless and can be
            # removed on the next cleanup pass; do not fail scientific work.
            pass


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    _atomic_replace_text(Path(path), json.dumps(dict(payload), indent=2, allow_nan=False) + "\n")


def portable_result_path(value: str | Path | None, results_dir: str | Path) -> str:
    """Serialize a runtime path relative to the public results directory."""

    if value is None or str(value).strip() == "":
        return ""
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    base = Path(results_dir).expanduser().resolve()
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(base)
    except ValueError:
        try:
            relative = Path(os.path.relpath(resolved, base))
        except ValueError as exc:
            raise ValueError(f"cannot serialize path on a different drive relative to results_dir: {resolved}") from exc
    return relative.as_posix()


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _normalize_ledger_row(row: Mapping[str, Any], results_dir: str | Path) -> dict[str, Any]:
    normalized = {column: row.get(column, "") for column in LEDGER_COLUMNS}
    normalized["output_path"] = portable_result_path(normalized["output_path"], results_dir)
    normalized["log_path"] = portable_result_path(normalized["log_path"], results_dir)
    normalized["reused_from_cache"] = _bool_value(normalized["reused_from_cache"])
    return normalized


class RuntimeLedger:
    """Atomically persisted runtime observations used by resume and ETA logic."""

    def __init__(self, path: str | Path, *, results_dir: str | Path | None = None) -> None:
        self.path = Path(path)
        self.results_dir = Path(results_dir) if results_dir is not None else self.path.parent
        self._lock = threading.Lock()

    def read(self) -> list[dict[str, str]]:
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return []
        with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def append(self, row: Mapping[str, Any]) -> None:
        with self._lock:
            rows: list[dict[str, Any]] = self.read()
            rows.append(_normalize_ledger_row(row, self.results_dir))
            self._write(rows)

    def finish(self, row: Mapping[str, Any]) -> None:
        """Atomically replace the matching RUNNING observation with a terminal row."""

        terminal = _normalize_ledger_row(row, self.results_dir)
        identity = tuple(str(terminal.get(column, "")) for column in ("analysis_stage", "dataset", "seed", "method", "start_time"))
        with self._lock:
            rows: list[dict[str, Any]] = self.read()
            replaced = False
            for index in range(len(rows) - 1, -1, -1):
                candidate = rows[index]
                candidate_identity = tuple(str(candidate.get(column, "")) for column in ("analysis_stage", "dataset", "seed", "method", "start_time"))
                if candidate_identity == identity and str(candidate.get("status", "")).upper() == "RUNNING":
                    rows[index] = terminal
                    replaced = True
                    break
            if not replaced:
                rows.append(terminal)
            self._write(rows)

    def _write(self, rows: Iterable[Mapping[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS, extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow(_normalize_ledger_row(row, self.results_dir))
            _replace_with_retry(temporary, self.path)
        finally:
            try:
                if temporary.exists():
                    temporary.unlink()
            except OSError:
                pass

    def normalize_paths(self) -> None:
        """Atomically migrate existing path columns to portable values."""

        with self._lock:
            self._write(self.read())

    def mark_running_incomplete(self, *, config_hash: str | None = None, analysis_stage: str | None = None) -> int:
        """Resolve stale RUNNING rows left by an interrupted process."""

        with self._lock:
            rows: list[dict[str, Any]] = self.read()
            changed = 0
            for row in rows:
                if str(row.get("status", "")).upper() != "RUNNING":
                    continue
                if config_hash and str(row.get("config_hash", "")) != config_hash:
                    continue
                if analysis_stage and str(row.get("analysis_stage", "")) != analysis_stage:
                    continue
                row["status"] = "INCOMPLETE"
                row["end_time"] = iso_now()
                row["failure_reason"] = "process interrupted before terminal status was recorded"
                changed += 1
            if changed:
                self._write(rows)
            return changed


@dataclass(frozen=True)
class ETAEstimate:
    seconds: float | None
    method: str
    observations: int


class ETAEstimator:
    """Robust hierarchy-based ETA estimator over compatible successful runs."""

    def __init__(self, rows: Iterable[Mapping[str, Any]], *, config_hash: str, rolling_window: int = 20) -> None:
        self.config_hash = str(config_hash)
        self.rolling_window = max(2, int(rolling_window))
        self.rows = [dict(row) for row in rows]

    def _compatible(self, *, stage: str, method: str | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in self.rows:
            if str(row.get("config_hash", "")) != self.config_hash:
                continue
            if str(row.get("status", "")).upper() not in TERMINAL_SUCCESS_STATUSES:
                continue
            if bool(str(row.get("reused_from_cache", "")).lower() in {"true", "1", "yes"}):
                # Cached rows normally have zero elapsed time and are evidence of validity,
                # not runtime observations. Their original completed rows remain usable.
                continue
            if str(row.get("analysis_stage", "")) != str(stage):
                continue
            if method is not None and str(row.get("method", "")) != str(method):
                continue
            try:
                elapsed = float(row.get("elapsed_seconds", ""))
            except (TypeError, ValueError):
                continue
            if elapsed <= 0:
                continue
            row["_elapsed"] = elapsed
            rows.append(row)
        return rows[-self.rolling_window :]

    @staticmethod
    def _median(rows: list[dict[str, Any]]) -> float:
        return float(statistics.median(float(row["_elapsed"]) for row in rows))

    def estimate(self, *, stage: str, method: str | None = None, cell_count: int | None = None) -> ETAEstimate:
        same_method = self._compatible(stage=stage, method=method) if method else []
        if method and cell_count and cell_count > 0:
            similar: list[dict[str, Any]] = []
            for row in same_method:
                try:
                    historical_cells = int(float(row.get("cell_count", "")))
                except (TypeError, ValueError):
                    continue
                if historical_cells > 0 and 0.5 <= historical_cells / cell_count <= 2.0:
                    similar.append(row)
            if len(similar) >= 2:
                return ETAEstimate(self._median(similar), "same_method_similar_size_rolling_median", len(similar))
        if len(same_method) >= 2:
            return ETAEstimate(self._median(same_method), "same_method_rolling_median", len(same_method))
        stage_rows = self._compatible(stage=stage)
        if len(stage_rows) >= 2:
            return ETAEstimate(self._median(stage_rows), "current_stage_rolling_median", len(stage_rows))
        return ETAEstimate(None, "insufficient_compatible_history", len(stage_rows))

    def estimate_tasks(self, tasks: Iterable[Mapping[str, Any]]) -> ETAEstimate:
        seconds = 0.0
        methods: list[str] = []
        observations = 0
        count = 0
        for task in tasks:
            estimate = self.estimate(
                stage=str(task.get("analysis_stage", "")),
                method=str(task.get("method", "")) or None,
                cell_count=int(task["cell_count"]) if task.get("cell_count") not in (None, "") else None,
            )
            if estimate.seconds is None:
                return ETAEstimate(None, "insufficient_history_for_remaining_tasks", observations)
            seconds += estimate.seconds
            methods.append(estimate.method)
            observations += estimate.observations
            count += 1
        if count == 0:
            return ETAEstimate(0.0, "no_remaining_tasks", 0)
        return ETAEstimate(seconds, "+".join(sorted(set(methods))), observations)


@dataclass
class ProgressSettings:
    enabled: bool = True
    refresh_seconds: float = 1.0
    verbose: bool = False
    interactive: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace, *, stream: TextIO | None = None) -> "ProgressSettings":
        target = stream or sys.stderr
        value = getattr(args, "progress", None)
        enabled = True if value is None else bool(value)
        return cls(
            enabled=enabled,
            refresh_seconds=max(0.1, float(getattr(args, "progress_refresh_seconds", 1.0))),
            verbose=bool(getattr(args, "verbose_progress", False)),
            interactive=bool(enabled and hasattr(target, "isatty") and target.isatty()),
        )


def add_progress_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true", default=None, help="Enable progress output (default).")
    group.add_argument("--no-progress", dest="progress", action="store_false", help="Disable progress output.")
    parser.add_argument("--progress-refresh-seconds", type=float, default=1.0, help="Interactive refresh interval; subprocess heartbeats remain less frequent.")
    parser.add_argument("--verbose-progress", action="store_true", help="Include optional fine-grained training progress.")
    parser.add_argument("--runtime-ledger", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--progress-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--progress-config-hash", default="", help=argparse.SUPPRESS)


def progress_paths(output_dir: str | Path, args: argparse.Namespace) -> tuple[Path, Path]:
    output = Path(output_dir)
    ledger = Path(args.runtime_ledger) if getattr(args, "runtime_ledger", None) else output / "runtime_ledger.csv"
    snapshot = Path(args.progress_file) if getattr(args, "progress_file", None) else output / "formal_progress.json"
    return ledger, snapshot


def _progress_results_dir(ledger_path: str | Path, output_path: str | Path) -> Path:
    ledger_parent = Path(ledger_path).expanduser().resolve().parent
    if str(output_path).strip():
        output = Path(output_path).expanduser().resolve()
        if output == ledger_parent:
            return output.parent
    return ledger_parent


def migrate_progress_artifacts(
    *,
    ledger_path: str | Path,
    snapshot_path: str | Path,
    results_dir: str | Path,
) -> None:
    """Atomically migrate existing public progress paths without rerunning work."""

    ledger = RuntimeLedger(ledger_path, results_dir=results_dir)
    if ledger.path.is_file():
        ledger.normalize_paths()
    snapshot = Path(snapshot_path)
    if snapshot.is_file():
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
        for field in ("output_path", "log_path", "latest_log_path"):
            if field in payload:
                payload[field] = portable_result_path(payload[field], results_dir)
        atomic_write_json(snapshot, payload)


class ProgressReporter:
    """One task-level progress reporter with atomic live snapshots."""

    def __init__(
        self,
        *,
        stage: str,
        total_units: int,
        settings: ProgressSettings,
        ledger_path: str | Path,
        snapshot_path: str | Path,
        config_hash: str,
        output_path: str | Path = "",
        results_dir: str | Path | None = None,
        initial_cached: int = 0,
        stream: TextIO | None = None,
    ) -> None:
        self.stage = str(stage)
        self.total_units = max(0, int(total_units))
        self.settings = settings
        self.results_dir = Path(results_dir) if results_dir is not None else _progress_results_dir(ledger_path, output_path)
        self.ledger = RuntimeLedger(ledger_path, results_dir=self.results_dir)
        self.snapshot_path = Path(snapshot_path)
        self.config_hash = str(config_hash)
        self.output_path = str(output_path)
        self.stream = stream or sys.stderr
        self.started = time.perf_counter()
        self.completed_units = 0
        self.cached_units = max(0, int(initial_cached))
        self.failed_units = 0
        self.resolved_units = self.cached_units
        self.current: dict[str, Any] = {}
        self.last_message_time = 0.0
        self.durations: list[tuple[str, float]] = []
        self._bar = None
        interrupted = self.ledger.mark_running_incomplete(config_hash=self.config_hash, analysis_stage=self.stage)
        self._estimator = ETAEstimator(self.ledger.read(), config_hash=self.config_hash)
        if self.settings.enabled and self.settings.interactive:
            try:
                from tqdm.auto import tqdm

                self._bar = tqdm(
                    total=self.total_units,
                    initial=min(self.cached_units, self.total_units),
                    desc=self.stage,
                    unit="run",
                    mininterval=self.settings.refresh_seconds,
                    dynamic_ncols=True,
                    file=self.stream,
                )
            except ImportError:
                self.settings.interactive = False
        self.snapshot(status="RUNNING")
        if interrupted:
            self._print(f"Resume scan found {interrupted} interrupted {self.stage} unit(s); they remain incomplete and will be rebuilt.", force=True)

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started

    def _print(self, message: str, *, force: bool = False) -> None:
        if not self.settings.enabled:
            return
        now = time.monotonic()
        if not force and now - self.last_message_time < self.settings.refresh_seconds:
            return
        self.last_message_time = now
        line = str(message).replace("\x1b", "")
        if self._bar is not None:
            self._bar.write(line)
        else:
            print(line, file=self.stream, flush=True)

    def estimate(self, *, method: str | None = None, cell_count: int | None = None) -> ETAEstimate:
        return self._estimator.estimate(stage=self.stage, method=method, cell_count=cell_count)

    def event(self, message: str, *, force: bool = True, **context: Any) -> None:
        self.current.update({key: value for key, value in context.items() if value is not None})
        self._print(message, force=force)
        self.snapshot(status=str(context.get("status", "RUNNING")))

    def start_unit(
        self,
        *,
        dataset: str = "",
        seed: int | str = "",
        method: str = "",
        cell_count: int | None = None,
        gene_count: int | None = None,
        output_path: str | Path = "",
        log_path: str | Path = "",
        prefix: str = "",
    ) -> float:
        estimate = self.estimate(method=method or None, cell_count=cell_count)
        self.current = {
            "dataset": dataset,
            "seed": seed,
            "method": method,
            "cell_count": cell_count if cell_count is not None else "",
            "gene_count": gene_count if gene_count is not None else "",
            "output_path": str(output_path or self.output_path),
            "log_path": str(log_path),
            "unit_start_iso": iso_now(),
        }
        self.ledger.append(self._ledger_row(started=time.perf_counter(), status="RUNNING", exit_code=0, reused=False))
        label = prefix or " | ".join(str(value) for value in (dataset, f"seed {seed}" if seed != "" else "", method) if value != "")
        self._print(
            f"[{self.resolved_units + 1}/{self.total_units}] {label} | elapsed {format_duration(self.elapsed_seconds)} | "
            f"ETA {format_duration(estimate.seconds)} ({estimate.method})",
            force=True,
        )
        self.snapshot(status="RUNNING", eta=estimate)
        return time.perf_counter()

    def _ledger_row(self, *, started: float, status: str, exit_code: int, reused: bool, failure_reason: str = "") -> dict[str, Any]:
        elapsed = 0.0 if reused else max(0.0, time.perf_counter() - started)
        start_iso = str(self.current.get("unit_start_iso", iso_now()))
        return {
            "analysis_stage": self.stage,
            "dataset": self.current.get("dataset", ""),
            "seed": self.current.get("seed", ""),
            "method": self.current.get("method", ""),
            "cell_count": self.current.get("cell_count", ""),
            "gene_count": self.current.get("gene_count", ""),
            "start_time": start_iso,
            "end_time": iso_now(),
            "elapsed_seconds": round(elapsed, 6),
            "status": status,
            "exit_code": int(exit_code),
            "config_hash": self.config_hash,
            "code_commit": code_commit(),
            "output_path": self.current.get("output_path", self.output_path),
            "log_path": self.current.get("log_path", ""),
            "reused_from_cache": bool(reused),
            "failure_reason": failure_reason,
        }

    def complete_unit(self, started: float, *, message: str = "completed") -> None:
        row = self._ledger_row(started=started, status="COMPLETED", exit_code=0, reused=False)
        self.ledger.finish(row)
        duration = float(row["elapsed_seconds"])
        self.durations.append((str(self.current.get("method") or self.current.get("dataset") or "unit"), duration))
        self.completed_units += 1
        self.resolved_units += 1
        if self._bar is not None:
            self._bar.update(1)
        self._estimator = ETAEstimator(self.ledger.read(), config_hash=self.config_hash)
        self._print(f"{message} | unit {format_duration(duration)} | resolved {self.resolved_units}/{self.total_units}", force=True)
        self.snapshot(status="RUNNING")

    def cached_unit(self, *, dataset: str = "", seed: int | str = "", method: str = "", output_path: str | Path = "") -> None:
        self.current = {"dataset": dataset, "seed": seed, "method": method, "output_path": str(output_path or self.output_path), "unit_start_iso": iso_now()}
        self.ledger.append(self._ledger_row(started=time.perf_counter(), status="SKIPPED_VALID_CACHE", exit_code=0, reused=True))
        self.cached_units += 1
        self.resolved_units += 1
        if self._bar is not None:
            self._bar.update(1)
        self._print(f"cached valid: {dataset} seed={seed} {method} | resolved {self.resolved_units}/{self.total_units}", force=True)
        self.snapshot(status="RUNNING")

    def fail_unit(self, started: float, error: BaseException | str, *, exit_code: int = 1) -> None:
        reason = str(error)
        self.ledger.finish(self._ledger_row(started=started, status="FAILED", exit_code=exit_code, reused=False, failure_reason=reason))
        self.failed_units += 1
        self.resolved_units += 1
        if self._bar is not None:
            self._bar.update(1)
        self._print(f"FAILED: {reason} | resolved {self.resolved_units}/{self.total_units}", force=True)
        self.snapshot(status="FAILED")

    def snapshot(self, *, status: str, eta: ETAEstimate | None = None) -> None:
        if eta is None:
            eta = self.estimate(method=str(self.current.get("method", "")) or None, cell_count=self.current.get("cell_count") or None)
            if eta.seconds is not None:
                eta = ETAEstimate(eta.seconds * max(0, self.total_units - self.resolved_units), eta.method, eta.observations)
        completion = utc_now() + timedelta(seconds=eta.seconds) if eta.seconds is not None else None
        payload = {
                "current_stage": self.stage,
                "current_dataset": self.current.get("dataset", ""),
                "current_seed": self.current.get("seed", ""),
                "current_method": self.current.get("method", ""),
                "completed_units": self.completed_units,
                "cached_units": self.cached_units,
                "failed_units": self.failed_units,
                "resolved_units": self.resolved_units,
                "total_units": self.total_units,
                "progress_fraction": (self.resolved_units / self.total_units) if self.total_units else 1.0,
                "elapsed_seconds": round(self.elapsed_seconds, 3),
                "estimated_remaining_seconds": None if eta.seconds is None else round(float(eta.seconds), 3),
                "eta_estimation_method": eta.method,
                "eta_observations": eta.observations,
                "estimated_completion_time": completion.isoformat(timespec="seconds") if completion else None,
                "current_status": status,
                "latest_log_path": portable_result_path(self.current.get("log_path", ""), self.results_dir),
                "output_path": portable_result_path(self.current.get("output_path", self.output_path), self.results_dir),
                "last_update_time": iso_now(),
            }
        try:
            atomic_write_json(self.snapshot_path, payload)
        except OSError as exc:
            # Progress snapshots are operational telemetry, not scientific
            # outputs.  A transient Windows file lock must not convert a valid
            # RF/DL run into a failed benchmark unit.
            self._print(f"WARNING: progress snapshot update skipped after retries: {exc}", force=True)

    def close(self, *, status: str | None = None) -> None:
        if status is None:
            status = "COMPLETED" if self.resolved_units == self.total_units and self.failed_units == 0 else "INCOMPLETE"
        self.snapshot(status=status)
        if self._bar is not None:
            self._bar.close()
        values = [value for _, value in self.durations]
        average = statistics.mean(values) if values else 0.0
        median = statistics.median(values) if values else 0.0
        slowest = sorted(self.durations, key=lambda item: item[1], reverse=True)[:3]
        slowest_text = ", ".join(f"{name}={format_duration(value)}" for name, value in slowest) or "none"
        self._print(
            f"Stage {self.stage}: completed={self.completed_units}, cached={self.cached_units}, failed={self.failed_units}, "
            f"runtime={format_duration(self.elapsed_seconds)}, average/unit={format_duration(average)}, "
            f"median/unit={format_duration(median)}, slowest={slowest_text}, output={self.output_path}",
            force=True,
        )

    def callback(self, event: Mapping[str, Any]) -> None:
        """Render model/domain callbacks without changing their implementation."""

        kind = str(event.get("event", "milestone"))
        if kind == "epoch":
            memory = event.get("gpu_memory_bytes")
            memory_text = ""
            if memory not in (None, ""):
                memory_text = f" | GPU {float(memory) / (1024 ** 2):.0f} MiB"
            self.event(
                f"DL epoch {event.get('epoch')}/{event.get('max_epochs')} | train={float(event.get('train_loss', float('nan'))):.5f} "
                f"| val={float(event.get('validation_loss', float('nan'))):.5f} | {event.get('validation_metric_name', 'validation metric')}="
                f"{float(event.get('validation_metric', float('nan'))):.5f} | best={float(event.get('best_metric', float('nan'))):.5f} "
                f"| patience={event.get('patience_counter')}/{event.get('patience')} | elapsed {format_duration(event.get('elapsed_seconds'))} "
                f"| epoch ETA {format_duration(event.get('epoch_eta_seconds'))} | training ETA {format_duration(event.get('training_eta_seconds'))}{memory_text}",
                force=True,
            )
        elif kind == "batch" and self.settings.verbose:
            self.event(str(event.get("message", "training batch")), force=False)
        else:
            self.event(str(event.get("message", kind)), force=True)


__all__ = [
    "ETAEstimate",
    "ETAEstimator",
    "LEDGER_COLUMNS",
    "ProgressReporter",
    "ProgressSettings",
    "RuntimeLedger",
    "add_progress_arguments",
    "atomic_write_json",
    "format_duration",
    "progress_paths",
]
