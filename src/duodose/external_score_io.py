"""Import externally computed doublet scores for benchmark comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"cell_id", "method", "score"}
OPTIONAL_COLUMNS = {"homotypic_score", "heterotypic_score", "label", "runtime_seconds", "status", "message"}


def load_external_score_file(path: str | Path) -> pd.DataFrame:
    """Load and validate an external score CSV.

    Required columns are ``cell_id,method,score``. Optional subtype score and
    status columns are preserved when present.
    """

    score_path = Path(path)
    frame = pd.read_csv(score_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"External score file is missing required column(s): {', '.join(sorted(missing))}")
    keep = [column for column in [*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS] if column in frame.columns]
    frame = frame.loc[:, keep].copy()
    frame["cell_id"] = frame["cell_id"].astype(str)
    frame["method"] = frame["method"].astype(str)
    for column in ["score", "homotypic_score", "heterotypic_score", "runtime_seconds"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "status" not in frame:
        frame["status"] = "success"
    if "message" not in frame:
        frame["message"] = ""
    frame.attrs["source_path"] = str(score_path)
    return frame


def external_scores_for_index(
    score_table: pd.DataFrame,
    index: pd.Index,
) -> tuple[dict[str, tuple[pd.Series, pd.Series, pd.Series]], dict[str, dict[str, float | str]]]:
    """Align imported external scores to an AnnData index."""

    if score_table.empty:
        return {}, {}
    obs_index = pd.Index(index.astype(str), name="cell_id")
    scores: dict[str, tuple[pd.Series, pd.Series, pd.Series]] = {}
    metadata: dict[str, dict[str, float | str]] = {}
    source = str(score_table.attrs.get("source_path", "external score file"))
    for method, group in score_table.groupby("method", sort=False):
        method_name = str(method)
        dedup = group.drop_duplicates(subset=["cell_id"], keep="first").set_index("cell_id")
        matched = obs_index.intersection(pd.Index(dedup.index.astype(str)))
        score = pd.Series(np.nan, index=index, dtype=float)
        homotypic = pd.Series(np.nan, index=index, dtype=float)
        heterotypic = pd.Series(np.nan, index=index, dtype=float)
        if len(matched):
            loc = pd.Index(index.astype(str)).get_indexer(matched)
            target_index = index[loc]
            score.loc[target_index] = dedup.loc[matched, "score"].to_numpy(dtype=float)
            hom_col = "homotypic_score" if "homotypic_score" in dedup else "score"
            het_col = "heterotypic_score" if "heterotypic_score" in dedup else "score"
            homotypic.loc[target_index] = dedup.loc[matched, hom_col].to_numpy(dtype=float)
            heterotypic.loc[target_index] = dedup.loc[matched, het_col].to_numpy(dtype=float)
        missing_count = int(len(index) - len(matched))
        status_values = group.get("status", pd.Series("success", index=group.index)).fillna("success").astype(str)
        status = "success" if len(matched) else "skipped"
        if status_values.ne("success").any() and status != "skipped":
            status = str(status_values.mode().iloc[0])
        message_values = [str(value) for value in group.get("message", pd.Series("", index=group.index)).fillna("").unique() if str(value)]
        message = f"imported external score file: {source}"
        if missing_count:
            message += f"; warning: {missing_count} missing cell(s) skipped"
        if message_values:
            message += "; " + " | ".join(message_values[:3])
        runtime = float(group["runtime_seconds"].dropna().max()) if "runtime_seconds" in group and group["runtime_seconds"].notna().any() else 0.0
        scores[method_name] = (score, homotypic, heterotypic)
        metadata[method_name] = {
            "runtime_seconds": runtime,
            "status": status,
            "message": message,
            "cache_hit": "false",
        }
    return scores, metadata

