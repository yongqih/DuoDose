"""Canonical controlled semi-real metrics shared by benchmark-adjacent paths.

The manuscript-facing high-RNA false-positive rate is evaluated at a matched
homotypic-recall operating point.  This prevents a method from appearing safe
merely because it retrieves very few homotypic doublets.  Historical equal-K
and fixed-candidate-budget values are retained under explicit names for
supplementary sensitivity analyses.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


DOUBLET_LABELS = frozenset({"homotypic_doublet", "heterotypic_doublet"})
PRIMARY_HOMOTYPIC_RECALL = 0.50
SUPPLEMENTARY_FIXED_CANDIDATE_FRACTION = 0.20
OVERALL_RECALL_OPERATING_POINTS = (0.50, 0.70, 0.80, 0.90)
HOMOTYPIC_RECALL_OPERATING_POINTS = (0.50, 0.70, 0.80)
PRIMARY_FPR_COLUMN = "high_RNA_singlet_FPR_at_matched_50pct_homotypic_recall"
FIXED_20_FPR_COLUMN = "high_RNA_singlet_FPR_at_fixed_20pct_candidate_budget"
TRUE_DOUBLET_BUDGET_FPR_COLUMN = "high_RNA_singlet_FPR_at_true_doublet_budget"
HIGH_RNA_METRIC_VERSION = "matched_homotypic_recall_v1"

HIGH_RNA_OPERATING_POINT_COLUMNS = [
    "dataset",
    "source_dataset",
    "seed",
    "method",
    "operating_point_type",
    "operating_point_name",
    "target_candidate_fraction",
    "candidate_fraction_source",
    "target_recall",
    "actual_candidate_fraction",
    "threshold",
    "number_selected_cells",
    "precision",
    "overall_recall",
    "homotypic_recall",
    "heterotypic_recall",
    "high_RNA_singlet_FPR",
    "high_RNA_singlet_FPR_at_fixed_candidate_fraction",
    "high_RNA_singlet_FPR_at_matched_overall_recall",
    "high_RNA_singlet_FPR_at_matched_homotypic_recall",
    "n_cells",
    "n_true_doublets",
    "n_homotypic_doublets",
    "n_heterotypic_doublets",
    "n_high_RNA_singlets",
    "n_selected_high_RNA_singlets",
    "status",
    "message",
]


def high_rna_singlet_top_budget_fpr(
    obs: pd.DataFrame,
    score: pd.Series | np.ndarray,
    *,
    label_column: str = "true_label",
    high_rna_column: str = "is_high_rna_singlet",
) -> dict[str, object]:
    """Historical FPR at K equal to the number of true constructed doublets.

    This value is retained for backward-compatible supplementary reporting. It
    is not the manuscript-facing primary FPR because K/N differs across
    semi-real datasets.
    """

    if label_column not in obs:
        return {
            "high_RNA_singlet_FPR": float("nan"),
            "high_RNA_singlet_FPR_status": "NOT_AVAILABLE",
            "high_RNA_singlet_FPR_reason": f"missing required label column {label_column!r}",
            "n_high_RNA_singlets": 0,
            "n_true_doublets_for_FPR_budget": 0,
        }
    labels = obs[label_column].astype(str)
    high_rna = obs.get(high_rna_column, pd.Series(False, index=obs.index)).fillna(False).astype(bool)
    n_high_rna = int(high_rna.sum())
    n_doublets = int(labels.isin(DOUBLET_LABELS).sum())
    if n_high_rna == 0:
        return {
            "high_RNA_singlet_FPR": float("nan"),
            "high_RNA_singlet_FPR_status": "NOT_AVAILABLE",
            "high_RNA_singlet_FPR_reason": "no high-RNA singlet rows are available in this semi-real split",
            "n_high_RNA_singlets": n_high_rna,
            "n_true_doublets_for_FPR_budget": n_doublets,
        }
    if n_doublets == 0:
        return {
            "high_RNA_singlet_FPR": float("nan"),
            "high_RNA_singlet_FPR_status": "NOT_AVAILABLE",
            "high_RNA_singlet_FPR_reason": "no true semi-real doublets are available for the historical top-score budget",
            "n_high_RNA_singlets": n_high_rna,
            "n_true_doublets_for_FPR_budget": n_doublets,
        }
    score_array = pd.Series(score, index=obs.index, dtype=float).reindex(obs.index).to_numpy(dtype=float)
    order = _deterministic_descending_order(score_array, obs.index)[: min(n_doublets, len(score_array))]
    return {
        "high_RNA_singlet_FPR": float(high_rna.iloc[order].sum() / max(1, n_high_rna)),
        "high_RNA_singlet_FPR_status": "AVAILABLE",
        "high_RNA_singlet_FPR_reason": "",
        "n_high_RNA_singlets": n_high_rna,
        "n_true_doublets_for_FPR_budget": n_doublets,
    }


def _deterministic_descending_order(scores: np.ndarray, cell_ids: pd.Index) -> np.ndarray:
    """Rank finite scores descending, then cell IDs ascending to resolve ties."""

    values = np.asarray(scores, dtype=float)
    sortable = np.where(np.isfinite(values), values, -np.inf)
    ranking = pd.DataFrame(
        {
            "score": sortable,
            "cell_id": pd.Index(cell_ids).astype(str),
            "position": np.arange(len(values), dtype=int),
        }
    )
    return ranking.sort_values(
        ["score", "cell_id", "position"],
        ascending=[False, True, True],
        kind="mergesort",
    )["position"].to_numpy(dtype=int)


def _operating_point_row(
    *,
    dataset: str,
    source_dataset: str,
    seed: int,
    method: str,
    operating_point_type: str,
    operating_point_name: str,
    target_candidate_fraction: float,
    candidate_fraction_source: str,
    target_recall: float,
    scores: np.ndarray,
    order: np.ndarray,
    n_selected: int,
    overall_mask: np.ndarray,
    homotypic_mask: np.ndarray,
    heterotypic_mask: np.ndarray,
    high_rna_mask: np.ndarray,
    status: str = "SUCCESS",
    message: str = "",
) -> dict[str, object]:
    n_cells = len(scores)
    selected = np.zeros(n_cells, dtype=bool)
    selected[order[: max(0, min(int(n_selected), n_cells))]] = True
    selected_count = int(selected.sum())
    n_doublets = int(overall_mask.sum())
    n_homotypic = int(homotypic_mask.sum())
    n_heterotypic = int(heterotypic_mask.sum())
    n_high_rna = int(high_rna_mask.sum())
    selected_high_rna = int((selected & high_rna_mask).sum())
    threshold = float("nan")
    if selected_count:
        boundary = float(scores[order[selected_count - 1]])
        threshold = boundary if np.isfinite(boundary) else float("-inf")
    precision = float((selected & overall_mask).sum() / selected_count) if selected_count else float("nan")
    overall_recall = float((selected & overall_mask).sum() / n_doublets) if n_doublets else float("nan")
    homotypic_recall = float((selected & homotypic_mask).sum() / n_homotypic) if n_homotypic else float("nan")
    heterotypic_recall = float((selected & heterotypic_mask).sum() / n_heterotypic) if n_heterotypic else float("nan")
    high_rna_fpr = float(selected_high_rna / n_high_rna) if n_high_rna else float("nan")
    if not n_high_rna and status == "SUCCESS":
        status = "NOT_AVAILABLE"
        message = "no high-RNA singlet rows are available in this semi-real test split"
    row = {
        "dataset": dataset,
        "source_dataset": source_dataset,
        "seed": int(seed),
        "method": method,
        "operating_point_type": operating_point_type,
        "operating_point_name": operating_point_name,
        "target_candidate_fraction": float(target_candidate_fraction),
        "candidate_fraction_source": candidate_fraction_source,
        "target_recall": float(target_recall),
        "actual_candidate_fraction": float(selected_count / n_cells) if n_cells else float("nan"),
        "threshold": threshold,
        "number_selected_cells": selected_count,
        "precision": precision,
        "overall_recall": overall_recall,
        "homotypic_recall": homotypic_recall,
        "heterotypic_recall": heterotypic_recall,
        "high_RNA_singlet_FPR": high_rna_fpr,
        "high_RNA_singlet_FPR_at_fixed_candidate_fraction": float("nan"),
        "high_RNA_singlet_FPR_at_matched_overall_recall": float("nan"),
        "high_RNA_singlet_FPR_at_matched_homotypic_recall": float("nan"),
        "n_cells": int(n_cells),
        "n_true_doublets": n_doublets,
        "n_homotypic_doublets": n_homotypic,
        "n_heterotypic_doublets": n_heterotypic,
        "n_high_RNA_singlets": n_high_rna,
        "n_selected_high_RNA_singlets": selected_high_rna,
        "status": status,
        "message": message,
    }
    if operating_point_type == "fixed_candidate_fraction":
        row["high_RNA_singlet_FPR_at_fixed_candidate_fraction"] = high_rna_fpr
    elif operating_point_type == "matched_overall_recall":
        row["high_RNA_singlet_FPR_at_matched_overall_recall"] = high_rna_fpr
    elif operating_point_type == "matched_homotypic_recall":
        row["high_RNA_singlet_FPR_at_matched_homotypic_recall"] = high_rna_fpr
    return row


def _minimum_selected_for_recall(order: np.ndarray, positive_mask: np.ndarray, target_recall: float) -> int | None:
    n_positive = int(positive_mask.sum())
    if n_positive == 0:
        return None
    target_count = int(np.ceil(float(target_recall) * n_positive - 1e-12))
    cumulative = np.cumsum(positive_mask[order].astype(int))
    reached = np.flatnonzero(cumulative >= target_count)
    return int(reached[0] + 1) if len(reached) else None


def high_rna_operating_point_metrics(
    obs: pd.DataFrame,
    method_scores: Mapping[str, pd.Series | np.ndarray],
    *,
    dataset: str,
    source_dataset: str,
    seed: int,
    fixed_candidate_fraction: float = SUPPLEMENTARY_FIXED_CANDIDATE_FRACTION,
    label_column: str = "true_label",
    high_rna_column: str = "is_high_rna_singlet",
) -> pd.DataFrame:
    """Evaluate high-RNA FPR at frozen matched and fixed-budget operating points."""

    if label_column not in obs:
        return pd.DataFrame(columns=HIGH_RNA_OPERATING_POINT_COLUMNS)
    labels = obs[label_column].astype(str)
    overall_mask = labels.isin(DOUBLET_LABELS).to_numpy(dtype=bool)
    homotypic_mask = labels.eq("homotypic_doublet").to_numpy(dtype=bool)
    heterotypic_mask = labels.eq("heterotypic_doublet").to_numpy(dtype=bool)
    high_rna_mask = obs.get(high_rna_column, pd.Series(False, index=obs.index)).fillna(False).astype(bool).to_numpy()
    n_cells = len(obs)
    candidate_fraction = float(fixed_candidate_fraction)
    candidate_fraction_source = "frozen_20pct_candidate_budget" if np.isclose(candidate_fraction, 0.20) else "explicit_fixed_candidate_fraction"
    if not np.isfinite(candidate_fraction) or not 0.0 <= candidate_fraction <= 1.0:
        raise ValueError("fixed_candidate_fraction must be within [0, 1]")
    n_fixed = int(np.rint(candidate_fraction * n_cells))
    rows: list[dict[str, object]] = []
    for method, score in method_scores.items():
        score_array = pd.Series(score, index=obs.index, dtype=float).reindex(obs.index).to_numpy(dtype=float)
        finite_available = bool(np.isfinite(score_array).any())
        order = _deterministic_descending_order(score_array, obs.index) if finite_available else np.arange(n_cells, dtype=int)
        unavailable = "" if finite_available else "method has no finite continuous scores"
        rows.append(
            _operating_point_row(
                dataset=dataset,
                source_dataset=source_dataset,
                seed=seed,
                method=str(method),
                operating_point_type="fixed_candidate_fraction",
                operating_point_name="high_RNA_singlet_FPR_at_fixed_20pct_candidate_budget",
                target_candidate_fraction=candidate_fraction,
                candidate_fraction_source=candidate_fraction_source,
                target_recall=float("nan"),
                scores=score_array,
                order=order,
                n_selected=n_fixed if finite_available else 0,
                overall_mask=overall_mask,
                homotypic_mask=homotypic_mask,
                heterotypic_mask=heterotypic_mask,
                high_rna_mask=high_rna_mask,
                status="SUCCESS" if finite_available else "NOT_AVAILABLE",
                message=unavailable,
            )
        )
        for target in OVERALL_RECALL_OPERATING_POINTS:
            selected = _minimum_selected_for_recall(order, overall_mask, target) if finite_available else None
            rows.append(
                _operating_point_row(
                    dataset=dataset,
                    source_dataset=source_dataset,
                    seed=seed,
                    method=str(method),
                    operating_point_type="matched_overall_recall",
                    operating_point_name=f"high_RNA_singlet_FPR_at_matched_{int(round(target * 100))}pct_overall_recall",
                    target_candidate_fraction=candidate_fraction,
                    candidate_fraction_source=candidate_fraction_source,
                    target_recall=target,
                    scores=score_array,
                    order=order,
                    n_selected=selected or 0,
                    overall_mask=overall_mask,
                    homotypic_mask=homotypic_mask,
                    heterotypic_mask=heterotypic_mask,
                    high_rna_mask=high_rna_mask,
                    status="SUCCESS" if selected is not None else "NOT_AVAILABLE",
                    message="" if selected is not None else (unavailable or "no true doublets are available in this semi-real test split"),
                )
            )
        for target in HOMOTYPIC_RECALL_OPERATING_POINTS:
            selected = _minimum_selected_for_recall(order, homotypic_mask, target) if finite_available else None
            rows.append(
                _operating_point_row(
                    dataset=dataset,
                    source_dataset=source_dataset,
                    seed=seed,
                    method=str(method),
                    operating_point_type="matched_homotypic_recall",
                    operating_point_name=f"high_RNA_singlet_FPR_at_matched_{int(round(target * 100))}pct_homotypic_recall",
                    target_candidate_fraction=candidate_fraction,
                    candidate_fraction_source=candidate_fraction_source,
                    target_recall=target,
                    scores=score_array,
                    order=order,
                    n_selected=selected or 0,
                    overall_mask=overall_mask,
                    homotypic_mask=homotypic_mask,
                    heterotypic_mask=heterotypic_mask,
                    high_rna_mask=high_rna_mask,
                    status="SUCCESS" if selected is not None else "NOT_AVAILABLE",
                    message="" if selected is not None else (unavailable or "no homotypic doublets are available in this semi-real test split"),
                )
            )
    return pd.DataFrame(rows, columns=HIGH_RNA_OPERATING_POINT_COLUMNS)


def high_rna_metric_bundle(
    obs: pd.DataFrame,
    score: pd.Series | np.ndarray,
    *,
    dataset: str,
    source_dataset: str,
    seed: int,
    method: str,
    label_column: str = "true_label",
    high_rna_column: str = "is_high_rna_singlet",
) -> dict[str, object]:
    """Return primary and supplementary FPRs for one method/run.

    ``high_RNA_singlet_FPR`` is intentionally the primary matched-50%
    homotypic-recall value.  The historical equal-K value is preserved under
    an explicit name and must not be used for cross-dataset interpretation.
    """

    aligned_obs = obs.copy()
    if label_column not in aligned_obs:
        return {
            "high_RNA_singlet_FPR": float("nan"),
            "high_RNA_singlet_FPR_status": "NOT_AVAILABLE",
            "high_RNA_singlet_FPR_reason": f"missing required label column {label_column!r}",
            "high_RNA_singlet_FPR_metric_version": HIGH_RNA_METRIC_VERSION,
        }
    table = high_rna_operating_point_metrics(
        aligned_obs,
        {method: score},
        dataset=dataset,
        source_dataset=source_dataset,
        seed=seed,
        fixed_candidate_fraction=SUPPLEMENTARY_FIXED_CANDIDATE_FRACTION,
        label_column=label_column,
        high_rna_column=high_rna_column,
    )
    historical = high_rna_singlet_top_budget_fpr(
        aligned_obs,
        score,
        label_column=label_column,
        high_rna_column=high_rna_column,
    )

    def selected(name: str) -> pd.Series | None:
        rows = table.loc[table["operating_point_name"].eq(name)]
        return None if rows.empty else rows.iloc[0]

    primary = selected(PRIMARY_FPR_COLUMN)
    fixed20 = selected(FIXED_20_FPR_COLUMN)
    matched70 = selected("high_RNA_singlet_FPR_at_matched_70pct_homotypic_recall")
    matched80 = selected("high_RNA_singlet_FPR_at_matched_80pct_homotypic_recall")
    status = str(primary["status"]) if primary is not None else "NOT_AVAILABLE"
    reason = str(primary["message"]) if primary is not None else "primary operating point was not generated"
    value = float(primary["high_RNA_singlet_FPR"]) if primary is not None else float("nan")
    return {
        "high_RNA_singlet_FPR": value,
        "high_RNA_singlet_FPR_status": status,
        "high_RNA_singlet_FPR_reason": reason,
        "high_RNA_singlet_FPR_metric_version": HIGH_RNA_METRIC_VERSION,
        PRIMARY_FPR_COLUMN: value,
        "high_RNA_singlet_FPR_at_matched_70pct_homotypic_recall": float(matched70["high_RNA_singlet_FPR"]) if matched70 is not None else float("nan"),
        "high_RNA_singlet_FPR_at_matched_80pct_homotypic_recall": float(matched80["high_RNA_singlet_FPR"]) if matched80 is not None else float("nan"),
        FIXED_20_FPR_COLUMN: float(fixed20["high_RNA_singlet_FPR"]) if fixed20 is not None else float("nan"),
        TRUE_DOUBLET_BUDGET_FPR_COLUMN: float(historical["high_RNA_singlet_FPR"]),
        "primary_FPR_target_homotypic_recall": PRIMARY_HOMOTYPIC_RECALL,
        "primary_FPR_actual_homotypic_recall": float(primary["homotypic_recall"]) if primary is not None else float("nan"),
        "primary_FPR_actual_candidate_fraction": float(primary["actual_candidate_fraction"]) if primary is not None else float("nan"),
        "primary_FPR_number_selected_cells": int(primary["number_selected_cells"]) if primary is not None else 0,
        "fixed_20pct_FPR_actual_candidate_fraction": float(fixed20["actual_candidate_fraction"]) if fixed20 is not None else float("nan"),
    }
