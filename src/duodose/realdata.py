"""Focused real-data helpers shared by public examples and the API."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from scipy.io import mmread
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

from .data import ensure_counts_layer
from .detect import detect as _detect
from .external_methods import run_r_external_method, run_scrublet
from .r_runtime import find_rscript


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_METHODS = ["Scrublet", "DoubletFinder", "scDblFinder", "scds"]
METHOD_ORDER = list(EXTERNAL_METHODS)
SAME_LIKE = "same-cell-type-like doublet"
DIFFERENT_LIKE = "different-cell-type-like doublet"
SINGLET_LIKE = "singlet"
LABEL_COLUMNS = [
    "experimental_doublet",
    "doublet",
    "is_doublet",
    "doublet_label",
    "classification",
    "label",
    "class",
]


@dataclass
class DatasetLoadResult:
    dataset: str
    adata: AnnData | None
    input_format: str
    label_source_column: str
    status: str
    message: str


@dataclass
class DatasetCandidate:
    dataset: str
    input_path: Path
    input_format: str
    load_path: Path
    converted_path: Path | None = None
    converted_status: str = "not_applicable"
    converted_message: str = ""


def _split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _split_int_csv(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    tokens = _split_csv(value)
    if not tokens:
        return tuple(default)
    values: list[int] = []
    for token in tokens:
        try:
            values.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid integer in comma-separated list: {token!r}") from exc
    return tuple(dict.fromkeys(values)) or tuple(default)


def resolve_external_methods(value: str | None) -> list[str]:
    requested = value or "all"
    tokens = [token.lower() for token in _split_csv(requested)] or ["all"]
    unsupported = [token for token in tokens if token in {"solo", "doubletdetection", "doublet_detection"}]
    if unsupported:
        raise ValueError(
            "Unsupported external method: "
            f"{unsupported[0]}. Supported methods are: scrublet, doubletfinder, scdblfinder, scds, all, none."
        )
    if "none" in tokens:
        return []
    if "all" in tokens:
        return list(EXTERNAL_METHODS)
    mapping = {
        "scrublet": "Scrublet",
        "doubletfinder": "DoubletFinder",
        "doublet_finder": "DoubletFinder",
        "scdblfinder": "scDblFinder",
        "sc_dbl_finder": "scDblFinder",
        "scds": "scds",
    }
    invalid = [token for token in tokens if token not in mapping]
    if invalid:
        raise ValueError(
            f"Unknown external method(s): {', '.join(invalid)}. "
            "Supported methods are: scrublet, doubletfinder, scdblfinder, scds, all, none."
        )
    selected: list[str] = []
    for token in tokens:
        method = mapping[token]
        if method not in selected:
            selected.append(method)
    return selected


def resolve_rerun_methods(value: str | None) -> set[str]:
    tokens = _split_csv(value)
    if not tokens:
        return set()
    aliases = {method.lower(): method for method in METHOD_ORDER}
    aliases.update({method.replace(" ", "_").lower(): method for method in METHOD_ORDER})
    aliases.update({method.replace(" ", "-").lower(): method for method in METHOD_ORDER})
    resolved: set[str] = set()
    invalid: list[str] = []
    for token in tokens:
        key = token.strip().lower()
        method = aliases.get(key)
        if method is None:
            invalid.append(token)
        else:
            resolved.add(method)
    if invalid:
        raise ValueError(f"Unknown --rerun-methods value(s): {', '.join(invalid)}")
    return resolved


def _read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def _parse_binary_label_series(series: pd.Series) -> pd.Series | None:
    normalized = series.astype(str).str.strip().str.lower()
    positive = {"1", "true", "t", "yes", "y", "doublet", "doublets", "multiplet", "multiplets"}
    negative = {"0", "false", "f", "no", "n", "singlet", "singlets", "cell", "cells"}
    parsed = pd.Series(np.nan, index=series.index, dtype=float)
    parsed.loc[normalized.isin(positive)] = 1.0
    parsed.loc[normalized.isin(negative)] = 0.0
    numeric = pd.to_numeric(series, errors="coerce")
    parsed.loc[numeric.eq(1)] = 1.0
    parsed.loc[numeric.eq(0)] = 0.0
    if parsed.isna().any() or parsed.nunique(dropna=True) < 2:
        return None
    return parsed.astype(int)


def _find_label_column(frame: pd.DataFrame) -> tuple[str, pd.Series] | tuple[None, None]:
    for column in LABEL_COLUMNS:
        if column in frame.columns:
            parsed = _parse_binary_label_series(frame[column])
            if parsed is not None:
                return column, parsed
    for column in frame.columns:
        lower = str(column).lower()
        if any(token in lower for token in ["doublet", "label", "class", "classification"]):
            parsed = _parse_binary_label_series(frame[column])
            if parsed is not None:
                return str(column), parsed
    return None, None


def _metadata_index_column(frame: pd.DataFrame, barcodes: pd.Index):
    barcode_set = set(barcodes.astype(str))
    for column in frame.columns:
        values = frame[column].astype(str)
        overlap = values.isin(barcode_set).sum()
        if overlap >= max(1, int(0.5 * len(barcodes))):
            return column
    return None


def _load_labels(path: Path, barcodes: pd.Index | None = None) -> tuple[pd.Series | None, str, str]:
    if not path.exists():
        return None, "", "label file missing"
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    frames = [_read_table(path), pd.read_csv(path, sep=sep, header=None)]
    column = None
    parsed = None
    frame = frames[0]
    for candidate in frames:
        frame = candidate
        column, parsed = _find_label_column(frame)
        if column is None or parsed is None:
            if frame.shape[1] == 1:
                parsed = _parse_binary_label_series(frame.iloc[:, 0])
                column = str(frame.columns[0]) if parsed is not None else None
            elif frame.shape[1] >= 2:
                parsed = _parse_binary_label_series(frame.iloc[:, 1])
                column = str(frame.columns[1]) if parsed is not None else None
        if column is not None and parsed is not None:
            break
    if column is None or parsed is None:
        return None, "", f"no parseable label column in {path.name}"
    labels = parsed.copy()
    if barcodes is not None:
        idx_column = _metadata_index_column(frame, barcodes)
        if idx_column is not None:
            labels.index = frame[idx_column].astype(str)
            labels = labels.reindex(barcodes.astype(str))
            if labels.isna().any():
                return None, str(column), f"labels in {path.name} did not align to barcodes"
        elif len(labels) == len(barcodes):
            labels.index = barcodes.astype(str)
        else:
            return None, str(column), f"label length {len(labels)} did not match {len(barcodes)} cells"
    return labels.astype(int), str(column), ""


def _matrix_file(directory: Path) -> Path | None:
    for name in ["counts.mtx", "matrix.mtx", "counts.mtx.gz", "matrix.mtx.gz"]:
        path = directory / name
        if path.exists():
            return path
    return None


def _genes_file(directory: Path) -> Path | None:
    for name in ["genes.tsv", "features.tsv", "genes.txt", "features.txt"]:
        path = directory / name
        if path.exists():
            return path
    return None


def _barcodes_file(directory: Path) -> Path | None:
    for name in ["barcodes.tsv", "barcodes.txt", "cells.tsv", "cells.txt"]:
        path = directory / name
        if path.exists():
            return path
    return None


def _label_file(directory: Path) -> Path | None:
    for name in ["labels.tsv", "label.tsv", "doublet_labels.tsv", "metadata.tsv", "labels.csv", "metadata.csv"]:
        path = directory / name
        if path.exists():
            return path
    return None


def _read_name_file(path: Path | None, fallback_prefix: str, n: int) -> list[str]:
    if path is None or not path.exists():
        return [f"{fallback_prefix}_{i}" for i in range(n)]
    frame = pd.read_csv(path, sep="\t", header=None)
    if frame.empty:
        return [f"{fallback_prefix}_{i}" for i in range(n)]
    values = frame.iloc[:, 1] if frame.shape[1] > 1 and fallback_prefix == "gene" else frame.iloc[:, 0]
    names = values.astype(str).tolist()
    if len(names) != n:
        return [f"{fallback_prefix}_{i}" for i in range(n)]
    return names


def _make_unique(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        base = str(name)
        count = counts.get(base, 0)
        counts[base] = count + 1
        out.append(base if count == 0 else f"{base}_{count}")
    return out


def _load_directory_dataset(path: Path) -> DatasetLoadResult:
    matrix_path = _matrix_file(path)
    barcode_path = _barcodes_file(path)
    label_path = _label_file(path)
    if matrix_path is None or barcode_path is None or label_path is None:
        return DatasetLoadResult(path.name, None, "directory", "", "skipped", "missing matrix, barcodes, or labels file")
    matrix = mmread(matrix_path).tocsr()
    barcodes = pd.Index(pd.read_csv(barcode_path, sep="\t", header=None).iloc[:, 0].astype(str))
    if matrix.shape[0] == len(barcodes):
        X = matrix.tocsr()
        n_genes = matrix.shape[1]
    elif matrix.shape[1] == len(barcodes):
        X = matrix.T.tocsr()
        n_genes = matrix.shape[0]
    else:
        return DatasetLoadResult(path.name, None, "directory", "", "skipped", "matrix dimensions did not match barcodes")
    genes = _make_unique(_read_name_file(_genes_file(path), "gene", n_genes))
    labels, source, message = _load_labels(label_path, barcodes)
    if labels is None:
        return DatasetLoadResult(path.name, None, "directory", source, "skipped", message)
    adata = AnnData(X=X, obs=pd.DataFrame(index=barcodes.astype(str)), var=pd.DataFrame(index=genes))
    adata.obs["experimental_doublet"] = labels.reindex(adata.obs_names).astype(int).to_numpy()
    adata.layers["counts"] = adata.X.copy()
    return DatasetLoadResult(path.name, adata, "directory_mtx", source, "success", "")


def check_converted_label_alignment(dataset: str, path: Path, adata: AnnData | None, outdir: Path, write_debug: bool) -> tuple[bool, str]:
    barcode_path = path / "barcodes.tsv"
    label_path = path / "labels.tsv"
    if adata is None or not barcode_path.exists() or not label_path.exists():
        return True, "label alignment check skipped: converted files unavailable"
    barcodes = pd.read_csv(barcode_path, sep="\t", header=None).iloc[:, 0].astype(str).tolist()
    labels_frame = pd.read_csv(label_path, sep="\t")
    if "cell_id" not in labels_frame.columns:
        return False, "labels.tsv is missing cell_id column"
    label_cells = labels_frame["cell_id"].astype(str).tolist()
    obs_names = adata.obs_names.astype(str).tolist()
    n = max(len(barcodes), len(label_cells), len(obs_names), 10)
    rows = []
    for i in range(min(10, n)):
        rows.append(
            {
                "dataset": dataset,
                "row": i,
                "barcodes_tsv_cell_id": barcodes[i] if i < len(barcodes) else "",
                "labels_tsv_cell_id": label_cells[i] if i < len(label_cells) else "",
                "adata_obs_name": obs_names[i] if i < len(obs_names) else "",
                "barcode_equals_label": i < len(barcodes) and i < len(label_cells) and barcodes[i] == label_cells[i],
                "label_equals_obs": i < len(label_cells) and i < len(obs_names) and label_cells[i] == obs_names[i],
            }
        )
    summary = {
        "dataset": dataset,
        "row": "summary",
        "barcodes_tsv_cell_id": f"n_barcodes={len(barcodes)}",
        "labels_tsv_cell_id": f"n_labels={len(label_cells)}",
        "adata_obs_name": f"n_obs={len(obs_names)}",
        "barcode_equals_label": len(barcodes) == len(label_cells) and barcodes == label_cells,
        "label_equals_obs": len(label_cells) == len(obs_names) and label_cells == obs_names,
    }
    rows.append(summary)
    if write_debug:
        debug_dir = outdir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(debug_dir / f"{dataset}_label_alignment_check.csv", index=False)
    ok = bool(summary["barcode_equals_label"] and summary["label_equals_obs"])
    message = "label alignment check passed" if ok else "converted RDS label alignment mismatch"
    return ok, message


def _load_h5ad_dataset(path: Path) -> DatasetLoadResult:
    from anndata import read_h5ad

    adata = read_h5ad(path)
    column, parsed = _find_label_column(adata.obs.reset_index(drop=True))
    if column is None or parsed is None:
        return DatasetLoadResult(path.stem, None, "h5ad", "", "skipped", "no parseable label column in adata.obs")
    adata = adata.copy()
    parsed.index = adata.obs_names
    adata.obs["experimental_doublet"] = parsed.astype(int).to_numpy()
    return DatasetLoadResult(path.stem, adata, "h5ad", column, "success", "")


def _load_csv_dataset(path: Path) -> DatasetLoadResult:
    frame = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",", index_col=0)
    column, parsed = _find_label_column(frame.reset_index(drop=False))
    if column is not None and parsed is not None:
        if column in frame.columns:
            labels = parsed
            counts = frame.drop(columns=[column])
        else:
            labels = parsed
            counts = frame
    else:
        return DatasetLoadResult(path.stem, None, path.suffix.lstrip("."), "", "skipped", "csv/tsv dataset needs an embedded parseable label column")
    numeric = counts.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.loc[:, numeric.notna().any(axis=0)].fillna(0.0)
    adata = AnnData(X=sparse.csr_matrix(numeric.to_numpy(dtype=np.float32)), obs=pd.DataFrame(index=numeric.index.astype(str)), var=pd.DataFrame(index=_make_unique(numeric.columns.astype(str).tolist())))
    if len(labels) != adata.n_obs:
        return DatasetLoadResult(path.stem, None, path.suffix.lstrip("."), str(column), "skipped", "label length did not match matrix rows")
    adata.obs["experimental_doublet"] = labels.to_numpy(dtype=int)
    adata.layers["counts"] = adata.X.copy()
    return DatasetLoadResult(path.stem, adata, path.suffix.lstrip("."), str(column), "success", "")


def load_dataset(path: Path, dataset_name: str | None = None, input_format: str | None = None) -> DatasetLoadResult:
    try:
        if path.is_dir():
            result = _load_directory_dataset(path)
            if dataset_name is not None:
                result.dataset = dataset_name
            if input_format is not None:
                result.input_format = input_format
            return result
        suffix = path.suffix.lower()
        if suffix == ".h5ad":
            result = _load_h5ad_dataset(path)
        elif suffix in {".csv", ".tsv"}:
            result = _load_csv_dataset(path)
        elif suffix == ".rds":
            result = DatasetLoadResult(path.stem, None, "rds", "", "skipped", ".rds inputs require --convert-rds or a converted directory")
        else:
            result = DatasetLoadResult(path.stem, None, suffix.lstrip("."), "", "skipped", f"unsupported input format: {suffix}")
        if dataset_name is not None:
            result.dataset = dataset_name
        if input_format is not None:
            result.input_format = input_format
        return result
    except Exception as exc:
        return DatasetLoadResult(dataset_name or (path.stem if path.is_file() else path.name), None, input_format or "unknown", "", "failed", str(exc))


def _has_converted_files(path: Path) -> bool:
    return all((path / name).exists() for name in ["matrix.mtx", "genes.tsv", "barcodes.tsv", "labels.tsv"])


def _safe_dataset_name(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("/", "_").replace("\\", "_")


def discover_input_candidates(data_dir: Path, selected: list[str] | None, max_datasets: int | None) -> list[DatasetCandidate]:
    candidates: list[DatasetCandidate] = []
    converted_parts = {part.lower() for part in ["converted", "cache"]}
    for path in sorted(data_dir.rglob("*"), key=lambda p: str(p).lower()):
        if path.name.startswith("."):
            continue
        if any(part.lower() in converted_parts for part in path.relative_to(data_dir).parts[:-1]):
            continue
        suffix = path.suffix.lower()
        if path.is_dir():
            if _matrix_file(path) is not None:
                candidates.append(DatasetCandidate(path.name, path, "directory", path))
        elif suffix in {".h5ad", ".csv", ".tsv"}:
            candidates.append(DatasetCandidate(path.stem, path, suffix.lstrip("."), path))
        elif suffix == ".rds":
            candidates.append(DatasetCandidate(_safe_dataset_name(path), path, "rds", path))
    if selected:
        wanted = set(selected)
        candidates = [candidate for candidate in candidates if candidate.dataset in wanted or candidate.input_path.stem in wanted or candidate.input_path.name in wanted]
    if max_datasets is not None:
        candidates = candidates[: int(max_datasets)]
    return candidates


def _conversion_report(path: Path) -> dict[str, object]:
    report = path / "conversion_report.json"
    if not report.exists():
        return {}
    try:
        return json.loads(report.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_rds_conversion_log(log_path: Path, completed: subprocess.CompletedProcess[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"command: {' '.join(str(part) for part in completed.args)}",
        f"returncode: {completed.returncode}",
        "",
        "stdout:",
        completed.stdout or "",
        "",
        "stderr:",
        completed.stderr or "",
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")


def convert_rds_candidate(candidate: DatasetCandidate, outdir: Path, refresh_cache: bool, quiet_external: bool = False) -> DatasetCandidate:
    if candidate.input_format != "rds":
        return candidate
    converted = outdir / "converted" / candidate.dataset
    log_path = outdir / "logs" / f"{candidate.dataset}_rds_conversion.log"
    candidate.converted_path = converted
    if _has_converted_files(converted) and not refresh_cache:
        candidate.load_path = converted
        candidate.converted_status = "cached"
        candidate.converted_message = "using existing converted directory"
        return candidate
    rscript = find_rscript()
    if rscript is None:
        candidate.converted_status = "skipped"
        candidate.converted_message = "Rscript not found"
        return candidate
    script = REPO_ROOT / "scripts" / "convert_xili_rds.R"
    converted.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [rscript, str(script), str(candidate.input_path), str(converted)],
        check=False,
        capture_output=True,
        text=True,
    )
    _write_rds_conversion_log(log_path, completed)
    if not quiet_external:
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.stderr:
            print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr)
    report = _conversion_report(converted)
    if completed.returncode == 0 and _has_converted_files(converted):
        candidate.load_path = converted
        candidate.converted_status = str(report.get("status", "success"))
        candidate.converted_message = str(report.get("message", "converted"))
        return candidate
    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    candidate.converted_status = str(report.get("status", "failed"))
    candidate.converted_message = str(report.get("message", stderr or stdout or "RDS conversion failed"))
    return candidate


def prepare_candidates(
    candidates: list[DatasetCandidate],
    outdir: Path,
    convert_rds: bool,
    refresh_cache: bool,
    quiet_external: bool = False,
) -> list[DatasetCandidate]:
    prepared: list[DatasetCandidate] = []
    for candidate in candidates:
        if candidate.input_format == "rds" and convert_rds:
            prepared.append(convert_rds_candidate(candidate, outdir, refresh_cache=refresh_cache, quiet_external=quiet_external))
        else:
            prepared.append(candidate)
    return prepared


def _ensure_counts_layer(adata: AnnData) -> AnnData:
    work = adata.copy()
    if "counts" not in work.layers:
        work.layers["counts"] = work.X.copy()
    work.var_names = _make_unique(work.var_names.astype(str).tolist())
    counts = work.layers["counts"]
    gene_sums = np.asarray(counts.sum(axis=0)).ravel()
    cell_sums = np.asarray(counts.sum(axis=1)).ravel()
    keep_genes = np.isfinite(gene_sums) & (gene_sums > 0)
    keep_cells = np.isfinite(cell_sums) & (cell_sums > 0)
    work = work[keep_cells, keep_genes].copy()
    if "counts" not in work.layers:
        work.layers["counts"] = work.X.copy()
    return work


def _counts_matrix(adata: AnnData):
    return adata.layers["counts"] if "counts" in adata.layers else adata.X


def _row_sums(X) -> np.ndarray:
    return np.asarray(X.sum(axis=1)).ravel().astype(float)


def _row_nnz(X) -> np.ndarray:
    if sparse.issparse(X):
        return np.diff(X.tocsr().indptr).astype(float)
    return (np.asarray(X) > 0).sum(axis=1).astype(float)


def _normalized_log_matrix(counts, target_sum: float = 1e4):
    if sparse.issparse(counts):
        X = counts.tocsr().astype(np.float32).copy()
        totals = np.asarray(X.sum(axis=1)).ravel()
        scale = np.divide(target_sum, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)
        X = sparse.diags(scale).dot(X).tocsr()
        X.data = np.log1p(X.data)
        return X
    X = np.asarray(counts, dtype=np.float32).copy()
    totals = X.sum(axis=1)
    scale = np.divide(target_sum, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)
    X *= scale[:, None]
    return np.log1p(X)


def _pca_representation(counts, n_components: int = 30, random_state: int = 0, fit_mask: np.ndarray | None = None) -> np.ndarray:
    X = _normalized_log_matrix(counts)
    n_cells, n_genes = X.shape
    n_components = int(max(2, min(n_components, n_cells - 1, n_genes - 1))) if n_cells > 2 and n_genes > 2 else 1
    if n_components <= 1:
        dense = X.toarray() if sparse.issparse(X) else np.asarray(X)
        return StandardScaler(with_mean=not sparse.issparse(X)).fit_transform(dense)[:, :1]
    if fit_mask is None:
        fit_mask = np.ones(n_cells, dtype=bool)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    X_fit = X[fit_mask]
    if sparse.issparse(X):
        model = TruncatedSVD(n_components=n_components, random_state=random_state)
        model.fit(X_fit)
        rep = model.transform(X)
    else:
        model = PCA(n_components=n_components, random_state=random_state)
        model.fit(np.asarray(X_fit))
        rep = model.transform(np.asarray(X))
    return StandardScaler().fit_transform(rep)

# semi-real dataset construction function
def add_semireal_clusters(
    adata: AnnData,
    *,
    n_clusters: int = 12,
    random_state: int = 0,
    cluster_key: str = "semireal_cluster",
) -> AnnData:
    """Cluster real singlets for controlled semi-real doublet construction."""

    work = adata.copy()
    counts = _counts_matrix(work)
    rep = _pca_representation(counts, n_components=30, random_state=random_state)

    n_cells = work.n_obs
    k = int(min(max(2, n_clusters), max(2, n_cells // 50)))
    labels = KMeans(n_clusters=k, n_init=20, random_state=random_state).fit_predict(rep)

    work.obs[cluster_key] = pd.Series([f"cluster_{i}" for i in labels], index=work.obs_names, dtype=str)
    return work


def _downsample_semireal_row(row, fraction: float, rng: np.random.Generator):
    row = row.tocsr(copy=True)
    if row.nnz:
        row.data = rng.binomial(
            np.rint(row.data).clip(min=0).astype(np.int64),
            float(np.clip(fraction, 0.0, 1.0)),
        ).astype(np.float32)
        row.eliminate_zeros()
    return row


def make_semireal_from_real_singlets(
    adata: AnnData,
    *,
    dataset: str,
    n_singlets: int = 5000,
    n_homotypic_doublets: int = 500,
    n_heterotypic_doublets: int = 500,
    n_clusters: int = 12,
    high_rna_quantile: float = 0.90,
    saturation_range: tuple[float, float] = (0.6, 1.0),
    min_cluster_size: int = 10,
    random_state: int = 0,
) -> AnnData:
    """
    Build a controlled semi-real benchmark from real labeled singlets.

    Negative:
      real labeled singlets, including high-RNA singlet hard negatives.

    Positive:
      constructed homotypic doublets from same-cluster real singlet parents.
      constructed heterotypic doublets from different-cluster real singlet parents.

    Important:
      Real experimental doublets are NOT used as positives here.
      They are excluded before semi-real construction.
    """

    rng = np.random.default_rng(int(random_state))
    work = _ensure_counts_layer(adata)
    counts = _counts_matrix(work).tocsr() if sparse.issparse(_counts_matrix(work)) else sparse.csr_matrix(_counts_matrix(work))

    if "experimental_doublet" not in work.obs:
        raise ValueError("semi-real construction requires experimental_doublet labels")

    real_singlet_mask = work.obs["experimental_doublet"].astype(int).eq(0).to_numpy()
    real_singlet_indices = np.flatnonzero(real_singlet_mask)

    if len(real_singlet_indices) < max(200, min_cluster_size * 2):
        raise ValueError(f"Too few real labeled singlets for semi-real benchmark: {len(real_singlet_indices)}")

    if len(real_singlet_indices) > int(n_singlets):
        real_singlet_indices = rng.choice(real_singlet_indices, size=int(n_singlets), replace=False)

    background = work[real_singlet_indices, :].copy()
    background = add_semireal_clusters(background, n_clusters=n_clusters, random_state=random_state)
    background_counts = _counts_matrix(background).tocsr() if sparse.issparse(_counts_matrix(background)) else sparse.csr_matrix(_counts_matrix(background))

    clusters = background.obs["semireal_cluster"].astype(str).to_numpy()
    cluster_values = np.array(sorted(pd.unique(clusters).tolist()))
    pools = {cluster: np.flatnonzero(clusters == cluster) for cluster in cluster_values}
    valid_clusters = np.array([cluster for cluster in cluster_values if len(pools[cluster]) >= int(min_cluster_size)])

    if len(valid_clusters) < 2:
        raise ValueError(
            f"Need at least two clusters with >= {min_cluster_size} singlets; found {len(valid_clusters)}"
        )

    # Mark high-RNA real singlets as hard negatives using within-cluster nCount rank.
    n_counts = np.asarray(background_counts.sum(axis=1)).ravel().astype(float)
    log_counts = np.log1p(n_counts)
    high_rna = np.zeros(background.n_obs, dtype=bool)

    for cluster in valid_clusters:
        local = np.flatnonzero(clusters == cluster)
        cutoff = float(np.quantile(log_counts[local], float(high_rna_quantile)))
        high_rna[local] = log_counts[local] >= cutoff

    singlet_obs = background.obs.copy()
    singlet_obs["experimental_doublet"] = 0
    singlet_obs["benchmark_cell_type"] = np.where(high_rna, "high_RNA_singlet", "singlet")
    singlet_obs["is_high_rna_singlet"] = high_rna
    singlet_obs["semireal_origin"] = "real_labeled_singlet"
    singlet_obs["true_doublet_label"] = "clean"
    singlet_obs["doublet_subtype"] = ""
    singlet_obs["doublet_like_subtype"] = SINGLET_LIKE
    singlet_obs["parent1_id"] = ""
    singlet_obs["parent2_id"] = ""
    singlet_obs["parent_cluster1"] = ""
    singlet_obs["parent_cluster2"] = ""
    singlet_obs["doublet_saturation"] = np.nan

    if "sample_id" not in singlet_obs:
        singlet_obs["sample_id"] = "lib1"
    if "library" not in singlet_obs:
        singlet_obs["library"] = singlet_obs["sample_id"].astype(str)

    doublet_rows = []
    doublet_obs_rows = []

    low, high = sorted(saturation_range)

    def add_doublet(parent_i: int, parent_j: int, subtype: str) -> None:
        saturation = float(rng.uniform(low, high))
        row = background_counts.getrow(parent_i) + background_counts.getrow(parent_j)
        row = _downsample_semireal_row(row, saturation, rng)
        doublet_rows.append(row)

        c1 = str(clusters[parent_i])
        c2 = str(clusters[parent_j])
        p1 = str(background.obs_names[parent_i])
        p2 = str(background.obs_names[parent_j])

        doublet_obs_rows.append(
            {
                "experimental_doublet": 1,
                "benchmark_cell_type": "semireal_doublet",
                "is_high_rna_singlet": False,
                "semireal_origin": "constructed_doublet",
                "semireal_cluster": c1,
                "true_doublet_label": f"{subtype}_doublet",
                "doublet_subtype": subtype,
                "doublet_like_subtype": SAME_LIKE if subtype == "homotypic" else DIFFERENT_LIKE,
                "parent1_id": p1,
                "parent2_id": p2,
                "parent_cluster1": c1,
                "parent_cluster2": c2,
                "doublet_parent_cluster_1": c1,
                "doublet_parent_cluster_2": c2,
                "doublet_saturation": saturation,
                "sample_id": str(singlet_obs.iloc[parent_i].get("sample_id", "lib1")),
                "library": str(singlet_obs.iloc[parent_i].get("library", "lib1")),
            }
        )

    for _ in range(int(n_homotypic_doublets)):
        cluster = str(rng.choice(valid_clusters))
        pool = pools[cluster]
        p1, p2 = rng.choice(pool, size=2, replace=False)
        add_doublet(int(p1), int(p2), "homotypic")

    for _ in range(int(n_heterotypic_doublets)):
        c1, c2 = rng.choice(valid_clusters, size=2, replace=False)
        p1 = int(rng.choice(pools[str(c1)]))
        p2 = int(rng.choice(pools[str(c2)]))
        add_doublet(p1, p2, "heterotypic")

    doublet_obs = pd.DataFrame(
        doublet_obs_rows,
        index=[f"{dataset}_semireal_doublet_{i:05d}" for i in range(len(doublet_obs_rows))],
    )

    combined_obs = pd.concat([singlet_obs, doublet_obs], axis=0, sort=False)
    combined_obs["experimental_doublet"] = combined_obs["experimental_doublet"].astype(int)
    combined_obs["is_high_rna_singlet"] = combined_obs["is_high_rna_singlet"].fillna(False).astype(bool)

    combined_X = sparse.vstack([background_counts, sparse.vstack(doublet_rows, format="csr")], format="csr")
    out = AnnData(X=combined_X, obs=combined_obs, var=background.var.copy())
    out.layers["counts"] = out.X.copy()

    out.uns["semireal_construction"] = {
        "source_dataset": dataset,
        "random_state": int(random_state),
        "n_real_labeled_singlets_available": int(real_singlet_mask.sum()),
        "n_real_singlets_used": int(background.n_obs),
        "n_high_rna_singlets": int(high_rna.sum()),
        "n_homotypic_doublets": int(n_homotypic_doublets),
        "n_heterotypic_doublets": int(n_heterotypic_doublets),
        "n_total_cells": int(out.n_obs),
        "n_genes": int(out.n_vars),
        "n_clusters_requested": int(n_clusters),
        "n_clusters_used": int(len(valid_clusters)),
        "high_rna_quantile": float(high_rna_quantile),
        "saturation_low": float(low),
        "saturation_high": float(high),
        "min_cluster_size": int(min_cluster_size),
        "label_definition": "positive=constructed homotypic/heterotypic doublets; negative=real labeled singlets",
    }

    return out


def _infer_clusters_for_ncount(adata: AnnData, random_state: int) -> pd.Series:
    counts = _counts_matrix(adata)
    rep = _pca_representation(counts, n_components=20, random_state=random_state)
    n_cells = adata.n_obs
    n_clusters = int(np.clip(np.sqrt(max(n_cells, 2) / 2.0), 2, 20))
    if n_cells < n_clusters:
        n_clusters = max(1, n_cells)
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit_predict(rep)
    return pd.Series(labels.astype(str), index=adata.obs_names)


def cluster_specific_ncount_score(adata: AnnData, random_state: int) -> tuple[pd.Series, str, str]:
    try:
        counts = _counts_matrix(adata)
        ncount = pd.Series(_row_sums(counts), index=adata.obs_names, dtype=float)
        clusters = _infer_clusters_for_ncount(adata, random_state=random_state)
        log_count = np.log1p(ncount)
        score = pd.Series(0.0, index=adata.obs_names, dtype=float)
        for cluster, idx in clusters.groupby(clusters).groups.items():
            values = log_count.loc[idx]
            median = float(values.median())
            mad = float(np.median(np.abs(values.to_numpy(dtype=float) - median)))
            scale = 1.4826 * mad if mad > 0 else float(values.std(ddof=0) or 1.0)
            score.loc[idx] = (values - median) / scale
        return score, "success", "within-cluster robust log nCount z-score"
    except Exception as exc:
        return pd.Series(np.nan, index=adata.obs_names, dtype=float), "failed", str(exc)


def _cache_key(dataset: str, method: str, adata: AnnData, random_state: int, config: dict[str, object] | None = None) -> str:
    counts = _counts_matrix(adata)
    nnz = int(counts.nnz) if sparse.issparse(counts) else int(np.count_nonzero(counts))
    payload = {
        "dataset": dataset,
        "method": method,
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "nnz": nnz,
        "random_state": int(random_state),
        "config": config or {},
        "version": "real_xili_v1",
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _load_cached_score(
    cache_dir: Path,
    dataset: str,
    method: str,
    adata: AnnData,
    random_state: int,
    refresh_cache: bool,
    config: dict[str, object] | None = None,
) -> tuple[pd.Series, dict[str, object]] | None:
    if refresh_cache:
        return None
    path = cache_dir / f"{dataset}__{method}__{_cache_key(dataset, method, adata, random_state, config)}.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "cell_id" not in frame or "score" not in frame:
        return None
    score = pd.to_numeric(frame.drop_duplicates("cell_id").set_index("cell_id")["score"].reindex(adata.obs_names), errors="coerce")
    metadata = {
        "status": str(frame.get("status", pd.Series(["success"])).iloc[0]),
        "message": str(frame.get("message", pd.Series(["cached score"])).iloc[0]),
        "runtime_sec": 0.0,
    }
    if metadata["status"].lower() != "success":
        return None
    if score.isna().any() or not np.isfinite(score.to_numpy(dtype=float)).all():
        return None
    return score, metadata


def _write_cached_score(
    cache_dir: Path,
    dataset: str,
    method: str,
    adata: AnnData,
    random_state: int,
    score: pd.Series,
    status: str,
    message: str,
    config: dict[str, object] | None = None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{dataset}__{method}__{_cache_key(dataset, method, adata, random_state, config)}.csv"
    pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            "score": pd.Series(score, index=adata.obs_names).to_numpy(dtype=float),
            "status": status,
            "message": message,
        }
    ).to_csv(path, index=False)


def _score_cache_filename(dataset: str, method: str) -> str:
    return f"{dataset}__{method}__cell_scores.csv"


def _score_cache_paths(outdir: Path, score_cache_dir: Path | None, dataset: str, method: str) -> list[Path]:
    paths: list[Path] = []
    filename = _score_cache_filename(dataset, method)
    if score_cache_dir is not None:
        paths.append(score_cache_dir / filename)
    default_path = outdir / "cell_scores" / filename
    if default_path not in paths:
        paths.append(default_path)
    return paths


def _log_cache_validation(outdir: Path, message: str) -> None:
    log_dir = outdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with (log_dir / "cache_validation.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def _should_rerun_method(method: str, rerun_methods: set[str]) -> bool:
    return method in rerun_methods


def load_valid_cell_score_cache(
    *,
    outdir: Path,
    score_cache_dir: Path | None,
    dataset: str,
    method: str,
    adata: AnnData,
    refresh_cache: bool,
    rerun_methods: set[str],
) -> pd.Series | None:
    if refresh_cache:
        _log_cache_validation(outdir, f"Skipping cache for {method} on {dataset}: --refresh-cache set")
        return None
    if _should_rerun_method(method, rerun_methods):
        message = f"Rerunning {method} on {dataset}"
        print(message)
        _log_cache_validation(outdir, message)
        return None

    expected_cells = adata.obs_names.astype(str).tolist()
    expected_labels = adata.obs["experimental_doublet"].astype(int).to_numpy()
    for path in _score_cache_paths(outdir, score_cache_dir, dataset, method):
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            message = f"WARNING: invalid score cache for {method} on {dataset}: could not read {path}: {exc}"
            print(message)
            _log_cache_validation(outdir, message)
            continue
        required = {"cell_id", "experimental_doublet", "method", "score"}
        missing = required.difference(frame.columns)
        if missing:
            message = f"WARNING: invalid score cache for {method} on {dataset}: missing columns {sorted(missing)} in {path}"
            print(message)
            _log_cache_validation(outdir, message)
            continue
        cache_cells = frame["cell_id"].astype(str).tolist()
        if cache_cells != expected_cells:
            message = f"WARNING: invalid score cache for {method} on {dataset}: cell_id order mismatch in {path}"
            print(message)
            _log_cache_validation(outdir, message)
            continue
        cache_labels = pd.to_numeric(frame["experimental_doublet"], errors="coerce")
        if (
            cache_labels.shape[0] != expected_labels.shape[0]
            or cache_labels.isna().any()
            or not np.array_equal(cache_labels.astype(int).to_numpy(), expected_labels)
        ):
            message = f"WARNING: invalid score cache for {method} on {dataset}: experimental_doublet labels mismatch in {path}"
            print(message)
            _log_cache_validation(outdir, message)
            continue
        cache_methods = frame["method"].astype(str)
        if not cache_methods.eq(method).all():
            message = f"WARNING: invalid score cache for {method} on {dataset}: method column mismatch in {path}"
            print(message)
            _log_cache_validation(outdir, message)
            continue
        score = pd.to_numeric(frame["score"], errors="coerce")
        if score.shape[0] != adata.n_obs or not np.isfinite(score.to_numpy(dtype=float)).any():
            message = f"WARNING: invalid score cache for {method} on {dataset}: no finite scores in {path}"
            print(message)
            _log_cache_validation(outdir, message)
            continue
        message = f"Using cached scores for {method} on {dataset}"
        print(message)
        _log_cache_validation(outdir, f"{message}: {path}")
        return pd.Series(score.to_numpy(dtype=float), index=adata.obs_names, dtype=float)
    return None


def write_cell_score_cache(outdir: Path, dataset: str, method: str, adata: AnnData, score: pd.Series) -> None:
    score_dir = outdir / "cell_scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    path = score_dir / _score_cache_filename(dataset, method)
    pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            "experimental_doublet": adata.obs["experimental_doublet"].astype(int).to_numpy(),
            "method": method,
            "score": pd.Series(score, index=adata.obs_names).reindex(adata.obs_names).to_numpy(dtype=float),
        }
    ).to_csv(path, index=False)


def run_duodose_score(
    adata: AnnData,
    dataset: str,
    random_state: int,
    refresh_cache: bool,
    cache_dir: Path,
    *,
    expected_doublet_rate: float,
    n_simulated_doublets: int | None,
    n_hvgs: int,
    n_pcs: int,
) -> tuple[dict[str, pd.Series], str, str, float, AnnData | None]:
    start = time.perf_counter()
    scores = {
        method: pd.Series(np.nan, index=adata.obs_names, dtype=float)
        for method in [
            "DuoDose",
            "DuoDose-identity",
            "DuoDose-dosage",
            "DuoDose-combined",
            "DuoDose-max",
            "DuoDose-gated-025",
            "DuoDose-gated-050",
            "DuoDose-gated-max",
            "DuoDose-gated-inlier",
        ]
    }
    result: AnnData | None = None
    try:
        work = AnnData(X=adata.X.copy(), obs=pd.DataFrame(index=adata.obs_names.copy()), var=adata.var.copy())
        work.layers["counts"] = _counts_matrix(adata).copy()
        work.obs["xili_unsupervised_cluster"] = _infer_clusters_for_ncount(adata, random_state=random_state).reindex(adata.obs_names).astype(str).to_numpy()
        n_sim = n_simulated_doublets
        if n_sim is None:
            n_sim = int(np.clip(max(200, 2 * adata.n_obs), 200, 5000))
        result = _detect(
            work,
            cluster_key="xili_unsupervised_cluster",
            expected_doublet_rate=float(expected_doublet_rate),
            model="logistic",
            random_state=random_state,
            n_simulated_doublets=int(n_sim),
            n_hvgs=int(min(max(1, n_hvgs), adata.n_vars)),
            n_pcs=int(max(2, min(n_pcs, adata.n_obs - 1, adata.n_vars - 1))),
            return_debug=True,
        )
        column_map = {
            "DuoDose": "duodose_score",
            "DuoDose-identity": "duodose_identity_score",
            "DuoDose-dosage": "duodose_dosage_score",
            "DuoDose-combined": "duodose_score_combined",
            "DuoDose-max": "duodose_score_max",
            "DuoDose-gated-025": "duodose_gated_025_score",
            "DuoDose-gated-050": "duodose_gated_050_score",
            "DuoDose-gated-max": "duodose_gated_max_score",
            "DuoDose-gated-inlier": "duodose_gated_inlier_score",
        }
        for method, column in column_map.items():
            scores[method] = pd.Series(result.obs[column].to_numpy(dtype=float), index=result.obs_names).reindex(adata.obs_names)
        status = "success"
        message = "DuoDose combined identity-neighbor and dosage SafeFeatures score trained without experimental labels"
    except Exception as exc:
        status = "failed"
        message = str(exc)
    runtime = time.perf_counter() - start
    if status == "success":
        for method, score in scores.items():
            _write_cached_score(cache_dir, dataset, method, adata, random_state, score, status, message, config={"real_duodose_combined": True})
    return scores, status, message, runtime, result


def run_external_score(
    adata: AnnData,
    dataset: str,
    method: str,
    random_state: int,
    refresh_cache: bool,
    cache_dir: Path,
    quiet_external: bool,
    expected_doublet_rate: float,
    log_path: str | Path | None = None,
    heartbeat_seconds: float = 45.0,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
    audit_dir: str | Path | None = None,
) -> tuple[pd.Series, str, str, float]:
    config = {"expected_doublet_rate": float(expected_doublet_rate)}
    cached = _load_cached_score(cache_dir, dataset, method, adata, random_state, refresh_cache, config=config)
    if cached is not None:
        score, metadata = cached
        return score, str(metadata["status"]), str(metadata["message"]), float(metadata["runtime_sec"])
    counts = _counts_matrix(adata)
    start = time.perf_counter()
    if method == "Scrublet":
        result = run_scrublet(counts, expected_doublet_rate=float(expected_doublet_rate), random_state=random_state, quiet=quiet_external)
    else:
        script_name = {
            "DoubletFinder": "run_doubletfinder.R",
            "scDblFinder": "run_scdblfinder.R",
            "scds": "run_scds.R",
        }[method]
        result = run_r_external_method(
            counts,
            cell_ids=adata.obs_names.astype(str),
            gene_ids=adata.var_names.astype(str),
            method_name=method,
            script_path=REPO_ROOT / "benchmarks" / "external" / script_name,
            expected_doublet_rate=float(expected_doublet_rate),
            random_state=random_state,
            quiet=quiet_external,
            log_path=log_path,
            heartbeat_seconds=heartbeat_seconds,
            progress_callback=progress_callback,
            audit_dir=audit_dir,
        )
    runtime = time.perf_counter() - start
    values = np.asarray(result.get("score", []), dtype=float).reshape(-1)
    status = str(result.get("status", "failed"))
    message = str(result.get("message", ""))
    if values.shape[0] != adata.n_obs:
        values = np.full(adata.n_obs, np.nan, dtype=float)
        if status == "success":
            status = "failed"
            message = "method returned wrong score length"
    score = pd.Series(values, index=adata.obs_names, dtype=float)
    _write_cached_score(cache_dir, dataset, method, adata, random_state, score, status, message, config=config)
    return score, status, message, runtime


