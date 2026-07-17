"""Runtime method completeness contract for manuscript aggregation."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


CANONICAL_RUNTIME_METHODS = (
    "DuoDose",
    "DuoDose-DL",
    "Scrublet",
    "scDblFinder",
    "DoubletFinder",
    "scds",
)
METHOD_ALIASES = {
    "DuoDose-ML-CalibratedRF-SafeFeatures": "DuoDose",
    "DuoDose-CalibratedRF": "DuoDose",
    "DuoDose-DL-ConditionalMultiTaskMLP-SafeFeatures": "DuoDose-DL",
    "doubletfinder": "DoubletFinder",
    "scdblfinder": "scDblFinder",
    "scrublet": "Scrublet",
    "scds": "scds",
}
VALID_STATUSES = {"COMPLETED", "FAILED", "UNAVAILABLE", "INCOMPLETE", "NOT_RUN"}


def normalize_runtime_method(value: object) -> str:
    text = str(value).strip()
    return METHOD_ALIASES.get(text, METHOD_ALIASES.get(text.lower(), text))


def build_runtime_method_completeness_audit(
    raw: pd.DataFrame,
    *,
    expected_methods: Sequence[str],
    requested_methods: Sequence[str],
    expected_successful_rows: int | None = None,
) -> pd.DataFrame:
    frame = raw.copy()
    if "method" in frame:
        frame["canonical_method"] = frame["method"].map(normalize_runtime_method)
    else:
        frame["canonical_method"] = pd.Series(dtype=str)
    expected = {normalize_runtime_method(value) for value in expected_methods}
    requested = {normalize_runtime_method(value) for value in requested_methods}
    rows: list[dict[str, object]] = []
    for method in CANONICAL_RUNTIME_METHODS:
        subset = frame.loc[frame["canonical_method"].eq(method)]
        statuses = subset.get("status", pd.Series("success", index=subset.index)).astype(str).str.lower()
        successful = int(statuses.isin({"success", "completed", "pass"}).sum())
        failed = int(statuses.eq("failed").sum())
        unavailable = int(statuses.eq("unavailable").sum())
        incomplete = int(statuses.eq("incomplete").sum())
        requested_flag = method in requested
        expected_flag = method in expected
        if successful:
            status = "COMPLETED" if expected_successful_rows is None or successful >= expected_successful_rows else "INCOMPLETE"
        elif failed:
            status = "FAILED"
        elif unavailable:
            status = "UNAVAILABLE"
        elif incomplete:
            status = "INCOMPLETE"
        else:
            status = "NOT_RUN"
        plotted = successful > 0
        if plotted:
            reason = ""
        elif not requested_flag and not len(subset):
            reason = "method was in the formal scope but was not requested by the completed runtime stage"
        elif status == "FAILED":
            reason = "runtime stage recorded failed rows"
        elif status == "UNAVAILABLE":
            reason = "runtime stage recorded the method as unavailable"
        elif status == "INCOMPLETE":
            reason = "runtime measurements do not cover the expected completed grid"
        else:
            reason = "no runtime measurement or explicit stage result was found"
        rows.append(
            {
                "method": method,
                "expected_by_protocol": expected_flag,
                "requested_by_runtime_stage": requested_flag,
                "raw_rows_found": int(len(subset)),
                "successful_rows": successful,
                "failed_rows": failed,
                "unavailable_rows": unavailable,
                "incomplete_rows": incomplete,
                "plotted": plotted,
                "omission_reason": reason,
                "status": status,
            }
        )
    result = pd.DataFrame(rows)
    if set(result["status"]) - VALID_STATUSES:
        raise AssertionError("runtime audit emitted a non-canonical status")
    silent = result["expected_by_protocol"] & ~result["plotted"] & result["omission_reason"].astype(str).str.strip().eq("")
    if silent.any():
        raise AssertionError("a configured runtime method was silently omitted")
    return result
