"""Run a child command while streaming logs and emitting quiet-process heartbeats."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from duodose.progress import format_duration  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--heartbeat-seconds", type=float, default=45.0)
    parser.add_argument("--estimated-seconds", type=float, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a child command is required after --")
    heartbeat_seconds = max(1.0, float(args.heartbeat_seconds))
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    last_activity = started
    lock = threading.Lock()
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=os.environ.copy(),
        )
        print(f"[{args.label}] launched PID {process.pid}; log: {log_path}", flush=True)

        def consume() -> None:
            nonlocal last_activity
            assert process.stdout is not None
            for line in process.stdout:
                with lock:
                    last_activity = time.perf_counter()
                    log.write(line)
                print(line, end="", flush=True)

        reader = threading.Thread(target=consume, name="duodose-log-reader", daemon=True)
        reader.start()
        next_heartbeat = started + heartbeat_seconds
        while process.poll() is None:
            now = time.perf_counter()
            if now >= next_heartbeat:
                with lock:
                    inactive = now - last_activity
                eta_text = ""
                if args.estimated_seconds is not None:
                    eta_text = f" | estimated method runtime {format_duration(args.estimated_seconds)}"
                print(
                    f"[{args.label}] still running | PID {process.pid} | elapsed {format_duration(now - started)} "
                    f"| last log activity {format_duration(inactive)} ago{eta_text}",
                    flush=True,
                )
                next_heartbeat = now + heartbeat_seconds
            time.sleep(min(0.5, heartbeat_seconds / 4.0))
        reader.join(timeout=10.0)
        return_code = int(process.wait())
        elapsed = time.perf_counter() - started
        print(f"[{args.label}] exited {return_code} after {format_duration(elapsed)}; log: {log_path}", flush=True)
        return return_code


if __name__ == "__main__":
    raise SystemExit(main())
