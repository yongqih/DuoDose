"""Optional wrappers for external Python doublet detection methods."""

from __future__ import annotations

from contextlib import nullcontext, redirect_stderr, redirect_stdout
import io
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

from .r_runtime import find_rscript


def _coerce_counts(counts: Any) -> Any:
    if sparse.issparse(counts):
        return counts.tocsr().astype(np.float32)
    return np.asarray(counts, dtype=np.float32)


def _result(method: str, score: Any, status: str, message: str) -> dict[str, Any]:
    if score is None:
        score_array = np.asarray([], dtype=float)
    else:
        score_array = np.asarray(score, dtype=float).reshape(-1)
    return {
        "method": method,
        "score": score_array,
        "status": status,
        "message": message,
    }


def _extract_score_from_classifier(classifier: Any, n_cells: int) -> tuple[np.ndarray | None, str]:
    for name in ("doublet_score", "doublet_scores", "scores"):
        value = getattr(classifier, name, None)
        if callable(value):
            try:
                score = np.asarray(value(), dtype=float)
            except TypeError:
                continue
        elif value is not None:
            score = np.asarray(value, dtype=float)
        else:
            continue
        if score.ndim == 2:
            if n_cells in score.shape:
                axis = 1 if score.shape[0] == n_cells else 0
                score = np.nanmean(score, axis=axis)
            else:
                score = score.reshape(-1)
        score = score.reshape(-1)
        if score.shape[0] == n_cells:
            return score, f"used {name}"

    for name in ("doublet_scores_obs_", "doublet_scores_", "score_", "scores_"):
        if not hasattr(classifier, name):
            continue
        score = np.asarray(getattr(classifier, name), dtype=float)
        if score.ndim == 2:
            if n_cells in score.shape:
                axis = 1 if score.shape[0] == n_cells else 0
                score = np.nanmean(score, axis=axis)
            else:
                score = score.reshape(-1)
        score = score.reshape(-1)
        if score.shape[0] == n_cells:
            return score, f"used {name}"

    return None, "continuous score unavailable"


def run_scrublet(
    counts: Any,
    expected_doublet_rate: float | None = None,
    random_state: int = 0,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run Scrublet on raw counts and return observed-cell doublet scores."""

    try:
        import scrublet as scr
    except ImportError:
        return _result("Scrublet", None, "skipped", "scrublet not installed")

    try:
        matrix = _coerce_counts(counts)
        kwargs: dict[str, Any] = {"random_state": random_state}
        if expected_doublet_rate is not None:
            kwargs["expected_doublet_rate"] = float(expected_doublet_rate)
        scrub = scr.Scrublet(matrix, **kwargs)
        if quiet:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                scores, _ = scrub.scrub_doublets()
        else:
            scores, _ = scrub.scrub_doublets()
        score = getattr(scrub, "doublet_scores_obs_", scores)
        return _result("Scrublet", score, "success", "continuous Scrublet observed-cell score")
    except Exception as exc:  # pragma: no cover - depends on optional package behavior.
        return _result("Scrublet", None, "failed", str(exc))


def _make_doubletdetection_classifier(dd_module: Any, random_state: int, n_iters: int = 25) -> Any:
    candidates = [
        {
            "n_iters": n_iters,
            "standard_scaling": True,
            "pseudocount": 0.1,
            "n_jobs": 1,
            "random_state": random_state,
        },
        {"n_iters": n_iters, "standard_scaling": True, "pseudocount": 0.1, "n_jobs": 1},
        {"n_iters": n_iters, "clustering_algorithm": "louvain", "n_jobs": 1, "random_state": random_state},
        {"n_iters": n_iters, "n_jobs": 1, "random_state": random_state},
        {"n_iters": n_iters, "random_state": random_state},
        {"n_iters": n_iters},
        {},
    ]
    last_error: Exception | None = None
    for kwargs in candidates:
        try:
            return dd_module.BoostClassifier(**kwargs)
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return dd_module.BoostClassifier()


def run_doubletdetection(
    counts: Any,
    expected_doublet_rate: float | None = None,
    random_state: int = 0,
    n_iters: int = 25,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run DoubletDetection's BoostClassifier on raw counts."""

    del expected_doublet_rate
    try:
        import doubletdetection as dd
    except ImportError:
        return _result("DoubletDetection", None, "skipped", "doubletdetection not installed")

    try:
        matrix = _coerce_counts(counts)
        n_cells = int(matrix.shape[0])
        classifier = _make_doubletdetection_classifier(dd, random_state=random_state, n_iters=n_iters)
        if quiet:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                classifier.fit(matrix)
        else:
            classifier.fit(matrix)
        score, message = _extract_score_from_classifier(classifier, n_cells)
        if score is not None:
            return _result("DoubletDetection", score, "success", f"continuous DoubletDetection score ({message})")

        labels = np.asarray(classifier.predict(), dtype=float).reshape(-1)
        if labels.shape[0] != n_cells:
            return _result("DoubletDetection", None, "failed", "DoubletDetection returned an unexpected score shape")
        return _result("DoubletDetection", labels, "success", "binary DoubletDetection labels used as score")
    except Exception as exc:  # pragma: no cover - depends on optional package behavior.
        return _result("DoubletDetection", None, "failed", str(exc))


def _nan_score(n_cells: int) -> np.ndarray:
    return np.full(int(n_cells), np.nan, dtype=float)


def _short_process_message(prefix: str, completed: subprocess.CompletedProcess[str] | None = None, extra: str = "") -> str:
    parts = [prefix]
    if extra:
        parts.append(extra)
    if completed is not None:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        if stderr:
            parts.append(stderr[-1000:])
        elif stdout:
            parts.append(stdout[-1000:])
    return " | ".join(part for part in parts if part)


def run_r_external_method(
    counts: Any,
    cell_ids: Any,
    gene_ids: Any,
    method_name: str,
    script_path: str | Path,
    expected_doublet_rate: float | None = 0.1,
    random_state: int = 0,
    quiet: bool = False,
    log_path: str | Path | None = None,
    heartbeat_seconds: float = 45.0,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
    audit_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run an R-based external doublet detector and align continuous scores."""

    cell_ids = np.asarray(cell_ids, dtype=str)
    gene_ids = np.asarray(gene_ids, dtype=str)
    n_cells = int(len(cell_ids))
    rscript = find_rscript()
    if rscript is None:
        return _result(method_name, _nan_score(n_cells), "skipped", "Rscript not found")
    script = Path(script_path)
    if not script.exists():
        return _result(method_name, _nan_score(n_cells), "skipped", f"R script not found: {script}")

    try:
        matrix = _coerce_counts(counts)
        if sparse.issparse(matrix):
            r_matrix = matrix.T.tocoo()
        else:
            r_matrix = sparse.coo_matrix(np.asarray(matrix).T)
        with tempfile.TemporaryDirectory(prefix=f"duodose_{method_name}_") as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_csv = tmp_path / "scores.csv"
            mmwrite(input_dir / "matrix.mtx", r_matrix)
            pd.Series(cell_ids).to_csv(input_dir / "barcodes.tsv", index=False, header=False)
            pd.Series(gene_ids).to_csv(input_dir / "genes.tsv", index=False, header=False)
            cmd = [
                rscript,
                str(script),
                str(input_dir),
                str(output_csv),
                str(0.1 if expected_doublet_rate is None else float(expected_doublet_rate)),
                str(int(random_state)),
            ]
            if audit_dir is not None:
                cmd.append(str(Path(audit_dir).resolve()))
            process_started = time.perf_counter()
            last_activity = process_started
            output_lines: list[str] = []
            activity_lock = threading.Lock()
            persistent_log = Path(log_path) if log_path is not None else None
            if persistent_log is not None:
                persistent_log.parent.mkdir(parents=True, exist_ok=True)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "subprocess_start",
                        "message": f"{method_name}: launched R subprocess PID {process.pid}; log={persistent_log or 'captured output'}",
                        "method": method_name,
                        "pid": process.pid,
                        "log_path": str(persistent_log or ""),
                    }
                )

            def consume_output() -> None:
                nonlocal last_activity
                assert process.stdout is not None
                context = persistent_log.open("a", encoding="utf-8", buffering=1) if persistent_log is not None else nullcontext(None)
                with context as handle:
                    for line in process.stdout:
                        with activity_lock:
                            last_activity = time.perf_counter()
                            output_lines.append(line)
                            if handle is not None:
                                handle.write(line)

            reader = threading.Thread(target=consume_output, name=f"duodose-{method_name}-log", daemon=True)
            reader.start()
            heartbeat = max(30.0, min(60.0, float(heartbeat_seconds)))
            next_heartbeat = process_started + heartbeat
            while process.poll() is None:
                now = time.perf_counter()
                if now >= next_heartbeat:
                    with activity_lock:
                        inactive_seconds = now - last_activity
                    message = (
                        f"{method_name}: still running | PID {process.pid} | elapsed {now - process_started:.1f}s "
                        f"| last log activity {inactive_seconds:.1f}s ago"
                    )
                    if progress_callback is not None:
                        progress_callback({"event": "subprocess_heartbeat", "message": message, "method": method_name, "pid": process.pid})
                    elif not quiet:
                        print(message, flush=True)
                    next_heartbeat = now + heartbeat
                time.sleep(min(0.5, heartbeat / 4.0))
            reader.join(timeout=10.0)
            completed = subprocess.CompletedProcess(cmd, int(process.wait()), stdout="".join(output_lines), stderr="")
            if not output_csv.exists():
                status = "failed" if completed.returncode != 0 else "failed"
                return _result(method_name, _nan_score(n_cells), status, _short_process_message("R method produced no output CSV", completed))

            out = pd.read_csv(output_csv)
            if "score" not in out:
                return _result(method_name, _nan_score(n_cells), "failed", "R output CSV missing required score column")
            if "cell_id" in out:
                aligned = out.drop_duplicates("cell_id").set_index("cell_id").reindex(cell_ids)
                score = pd.to_numeric(aligned["score"], errors="coerce").to_numpy(dtype=float)
                missing = int(aligned["score"].isna().sum())
            else:
                score = pd.to_numeric(out["score"], errors="coerce").to_numpy(dtype=float)
                missing = 0
            if score.shape[0] != n_cells:
                return _result(method_name, _nan_score(n_cells), "failed", f"R output score length {score.shape[0]} did not match {n_cells} cells")
            status_values = out.get("status", pd.Series(["success"])).fillna("success").astype(str)
            status = "success" if status_values.str.lower().eq("success").all() else str(status_values.iloc[0])
            messages = [str(value) for value in out.get("message", pd.Series(dtype=object)).dropna().astype(str).unique().tolist() if str(value)]
            message = "; ".join(messages[:3])
            if completed.returncode != 0 and status == "success":
                status = "failed"
                message = _short_process_message("R method exited with non-zero status", completed, message)
            if missing and status == "success":
                status = "failed"
                message = f"{missing} cells missing aligned R scores"
            if not message:
                message = f"continuous {method_name} score"
            if quiet and completed.returncode == 0:
                message = message
            return _result(method_name, score, status, message)
    except Exception as exc:  # pragma: no cover - depends on local R/runtime state.
        return _result(method_name, _nan_score(n_cells), "failed", str(exc))


def run_solo(
    adata: Any,
    expected_doublet_rate: float | None = None,
    random_state: int = 0,
    quiet: bool = False,
    max_epochs_scvi: int = 50,
    max_epochs_solo: int = 50,
) -> dict[str, Any]:
    """Run scvi-tools Solo and return observed-cell doublet probabilities."""

    del expected_doublet_rate
    try:
        import scvi
        from scvi.external import SOLO
        from scvi.model import SCVI
    except Exception as exc:
        return _result("Solo", _nan_score(int(adata.n_obs)), "skipped", f"scvi-tools import failed: {exc}")

    try:
        ad = adata.copy()
        counts = ad.layers["counts"].copy() if "counts" in ad.layers else ad.X.copy()
        ad.layers["counts"] = counts
        gene_sums = np.asarray(counts.sum(axis=0)).ravel()
        keep_genes = np.isfinite(gene_sums) & (gene_sums > 0)
        if keep_genes.any() and keep_genes.sum() < ad.n_vars:
            ad = ad[:, keep_genes].copy()
        if ad.n_vars == 0:
            return _result("Solo", _nan_score(int(adata.n_obs)), "failed", "Solo input has no nonzero genes")
        ad.var_names_make_unique()
        if hasattr(scvi, "settings"):
            scvi.settings.seed = int(random_state)

        def _fit() -> Any:
            SCVI.setup_anndata(ad, layer="counts")
            vae = SCVI(ad, n_latent=10)
            vae.train(max_epochs=int(max_epochs_scvi))
            solo_model = SOLO.from_scvi_model(vae)
            solo_model.train(max_epochs=int(max_epochs_solo))
            return solo_model.predict(soft=True)

        if quiet:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                pred = _fit()
        else:
            pred = _fit()

        if isinstance(pred, pd.DataFrame):
            lower = {str(col).lower(): col for col in pred.columns}
            column = lower.get("doublet") or lower.get("doublet_probability")
            if column is None:
                singlet_col = lower.get("singlet") or lower.get("singlet_probability")
                if singlet_col is not None and pred.shape[1] == 2:
                    column = [col for col in pred.columns if col != singlet_col][0]
                elif pred.shape[1] == 2:
                    # scvi-tools has historically returned two soft columns;
                    # when no names are recognizable, use the second column as
                    # the doublet-like probability for compatibility.
                    column = pred.columns[1]
                else:
                    return _result("Solo", _nan_score(int(adata.n_obs)), "failed", f"Solo soft predictions lacked a recognizable doublet column: {list(pred.columns)}")
            score = pd.to_numeric(pred[column], errors="coerce").to_numpy(dtype=float)
        else:
            arr = np.asarray(pred, dtype=float)
            score = arr[:, -1] if arr.ndim == 2 else arr.reshape(-1)
        if score.shape[0] != int(adata.n_obs):
            return _result("Solo", _nan_score(int(adata.n_obs)), "failed", f"Solo returned score length {score.shape[0]} for {adata.n_obs} cells")
        return _result("Solo", score, "success", "continuous Solo doublet probability")
    except Exception as exc:  # pragma: no cover - depends on optional package behavior.
        return _result("Solo", _nan_score(int(adata.n_obs)), "failed", str(exc))
