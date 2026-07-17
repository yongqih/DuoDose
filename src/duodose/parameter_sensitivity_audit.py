"""Contracts and diagnostics for the formal parameter-sensitivity analysis."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


SENSITIVITY_METRICS = (
    "overall_AUPRC",
    "homotypic_AUPRC",
    "heterotypic_AUPRC",
    "homotypic_vs_high_RNA_singlet_AUPRC",
    "high_RNA_singlet_FPR",
    "high_RNA_singlet_FPR_at_expected_rate",
)


def training_size_protocol(protocol: Mapping[str, Any], factor: float) -> dict[str, Any]:
    """Vary fit-split doublets while freezing validation and test sizes."""

    varied = copy.deepcopy(dict(protocol))
    if float(factor) <= 0:
        raise ValueError("semi_real_size_factor must be positive")
    for name in ("n_train_homotypic_doublets", "n_train_heterotypic_doublets"):
        varied["semi_real"][name] = max(1, int(round(float(protocol["semi_real"][name]) * float(factor))))
    return varied


def historical_size_protocol(protocol: Mapping[str, Any], factor: float) -> dict[str, Any]:
    """Reproduce the completed grid's former all-split scaling for auditing."""

    varied = copy.deepcopy(dict(protocol))
    for name in (
        "n_train_homotypic_doublets",
        "n_train_heterotypic_doublets",
        "n_validation_homotypic_doublets",
        "n_validation_heterotypic_doublets",
        "n_test_homotypic_doublets",
        "n_test_heterotypic_doublets",
    ):
        varied["semi_real"][name] = max(1, int(round(float(protocol["semi_real"][name]) * float(factor))))
    return varied


def deterministic_top_fraction(
    labels: pd.Series,
    score: pd.Series,
    fraction: float,
) -> dict[str, Any]:
    """Select exactly round(fraction * N) rows with deterministic score ties."""

    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("expected doublet rate must be between zero and one")
    labels = labels.astype(str)
    values = pd.to_numeric(score.reindex(labels.index), errors="coerce")
    high = labels.eq("high_RNA_singlet")
    ranking = pd.DataFrame(
        {
            "score": values.where(np.isfinite(values), -np.inf),
            "cell_id": labels.index.astype(str),
            "high": high,
        },
        index=labels.index,
    ).sort_values(["score", "cell_id"], ascending=[False, True], kind="mergesort")
    k = min(len(ranking), max(0, int(round(float(fraction) * len(ranking)))))
    selected = ranking.iloc[:k]
    n_high = int(high.sum())
    n_selected_high = int(selected["high"].sum())
    threshold = float(selected["score"].iloc[-1]) if k else float("nan")
    return {
        "threshold": threshold,
        "number_selected_cells": int(k),
        "actual_candidate_fraction": float(k / len(ranking)) if len(ranking) else float("nan"),
        "n_high_RNA_singlets": n_high,
        "n_selected_high_RNA_singlets": n_selected_high,
        "high_RNA_singlet_FPR": float(n_selected_high / n_high) if n_high else float("nan"),
        "status": "AVAILABLE" if n_high else "NOT_AVAILABLE",
        "reason": "" if n_high else "no high-RNA singlet rows",
    }


def canonical_top_true_doublet_budget(labels: pd.Series, score: pd.Series) -> dict[str, Any]:
    """Reproduce the canonical historical top-true-doublet selection rule."""

    labels = labels.astype(str)
    values = pd.to_numeric(score.reindex(labels.index), errors="coerce").to_numpy(dtype=float)
    high = labels.eq("high_RNA_singlet").to_numpy(dtype=bool)
    n_doublets = int(labels.isin({"homotypic_doublet", "heterotypic_doublet"}).sum())
    order = np.argsort(-np.where(np.isfinite(values), values, -np.inf))[: min(n_doublets, len(values))]
    n_high = int(high.sum())
    selected_high = int(high[order].sum())
    return {
        "threshold": float(values[order[-1]]) if len(order) else float("nan"),
        "number_selected_cells": int(len(order)),
        "actual_candidate_fraction": float(len(order) / len(values)) if len(values) else float("nan"),
        "n_high_RNA_singlets": n_high,
        "n_selected_high_RNA_singlets": selected_high,
        "high_RNA_singlet_FPR": float(selected_high / n_high) if n_high else float("nan"),
        "status": "AVAILABLE" if n_high and n_doublets else "NOT_AVAILABLE",
        "reason": "" if n_high and n_doublets else "high-RNA singlets or true doublets unavailable",
        "n_true_doublets": n_doublets,
    }


def _quantiles(values: pd.Series, prefix: str) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return {
        f"{prefix}_score_q05": float(numeric.quantile(0.05)) if len(numeric) else float("nan"),
        f"{prefix}_score_q25": float(numeric.quantile(0.25)) if len(numeric) else float("nan"),
        f"{prefix}_score_median": float(numeric.median()) if len(numeric) else float("nan"),
        f"{prefix}_score_q75": float(numeric.quantile(0.75)) if len(numeric) else float("nan"),
        f"{prefix}_score_q95": float(numeric.quantile(0.95)) if len(numeric) else float("nan"),
    }


def score_threshold_record(
    *,
    dataset: str,
    seed: int,
    factor: float,
    expected_rate: float,
    labels: pd.Series,
    overall_score: pd.Series,
) -> dict[str, Any]:
    labels = labels.astype(str)
    score = pd.to_numeric(overall_score.reindex(labels.index), errors="coerce")
    negative_singlet = labels.isin({"clean", "singlet"})
    canonical = canonical_top_true_doublet_budget(labels, score)
    expected = deterministic_top_fraction(labels, score, expected_rate)
    row: dict[str, Any] = {
        "dataset": dataset,
        "seed": int(seed),
        "semi_real_size_factor": float(factor),
        "expected_doublet_rate": float(expected_rate),
        "n_test_cells": int(len(labels)),
        "n_test_singlets": int(negative_singlet.sum()),
        "n_test_high_RNA_singlets": int(labels.eq("high_RNA_singlet").sum()),
        "n_test_homotypic_doublets": int(labels.eq("homotypic_doublet").sum()),
        "n_test_heterotypic_doublets": int(labels.eq("heterotypic_doublet").sum()),
        "fixed_probability_threshold": float("nan"),
        "fixed_probability_threshold_status": "NOT_APPLICABLE",
        "canonical_top_true_doublet_threshold": canonical["threshold"],
        "canonical_number_selected_cells": canonical["number_selected_cells"],
        "canonical_candidate_fraction": canonical["actual_candidate_fraction"],
        "canonical_high_RNA_singlet_denominator": canonical["n_high_RNA_singlets"],
        "canonical_high_RNA_singlet_false_positive_count": canonical["n_selected_high_RNA_singlets"],
        "high_RNA_singlet_FPR": canonical["high_RNA_singlet_FPR"],
        "expected_rate_threshold": expected["threshold"],
        "expected_rate_number_selected_cells": expected["number_selected_cells"],
        "expected_rate_actual_candidate_fraction": expected["actual_candidate_fraction"],
        "expected_rate_high_RNA_singlet_denominator": expected["n_high_RNA_singlets"],
        "expected_rate_high_RNA_singlet_false_positive_count": expected["n_selected_high_RNA_singlets"],
        "high_RNA_singlet_FPR_at_expected_rate": expected["high_RNA_singlet_FPR"],
    }
    row.update(_quantiles(score.loc[negative_singlet], "singlet"))
    row.update(_quantiles(score.loc[labels.eq("high_RNA_singlet")], "high_RNA_singlet"))
    row.update(_quantiles(score.loc[labels.eq("homotypic_doublet")], "homotypic_doublet"))
    row.update(_quantiles(score.loc[labels.eq("heterotypic_doublet")], "heterotypic_doublet"))
    return row


def stable_values_hash(values: Iterable[Any]) -> str:
    payload = "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_fingerprints(run: Any) -> dict[str, Any]:
    parent_map = run.bundle.parent_map

    def parent_hash(split: str) -> str:
        rows = parent_map.loc[parent_map["split"].astype(str).eq(split)]
        return stable_values_hash([*rows["parent_1_id"].astype(str), *rows["parent_2_id"].astype(str)])

    high = run.bundle.test_adata.obs.get("is_high_rna_singlet", pd.Series(False, index=run.bundle.test_adata.obs_names)).fillna(False).astype(bool)
    reference_cluster_pairs = [
        f"{cell_id}={cluster}"
        for cell_id, cluster in zip(run.transformer.reference_cell_ids_, run.transformer.reference_clusters_, strict=True)
    ]
    return {
        "train_parent_ids_hash": parent_hash("train"),
        "validation_parent_ids_hash": parent_hash("validation"),
        "test_parent_ids_hash": parent_hash("test"),
        "reference_pool_ids_hash": stable_values_hash(run.transformer.reference_cell_ids_),
        "validation_cell_ids_hash": stable_values_hash(run.bundle.val_adata.obs_names),
        "test_cell_ids_hash": stable_values_hash(run.bundle.test_adata.obs_names),
        "high_RNA_singlet_ids_hash": stable_values_hash(run.bundle.test_adata.obs_names[high]),
        "safe_feature_transformer_id": str(run.transformer.transformer_id_),
        "safe_feature_reference_pool_id": str(run.transformer.reference_pool_id_),
        "feature_schema_hash": stable_values_hash(run.transformer.model_feature_columns_),
        "cluster_definition_hash": stable_values_hash(reference_cluster_pairs),
        "n_genes": int(run.bundle.test_adata.n_vars),
    }


def aggregate_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    summary = frame.groupby(["semi_real_size_factor", "expected_doublet_rate"], as_index=False)[list(SENSITIVITY_METRICS)].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(str(part) for part in column if part) for column in summary.columns.to_flat_index()]
    return summary.reset_index(drop=True)


def sensitivity_run_status(
    frame: pd.DataFrame,
    *,
    dataset: str,
    seeds: Sequence[int],
    factors: Sequence[float],
    expected_rates: Sequence[float],
) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for factor in factors:
            for rate in expected_rates:
                match = frame.loc[
                    frame["seed"].astype(int).eq(int(seed))
                    & np.isclose(pd.to_numeric(frame["semi_real_size_factor"]), float(factor))
                    & np.isclose(pd.to_numeric(frame["expected_doublet_rate"]), float(rate))
                ]
                if len(match) == 1:
                    source_status = str(match.iloc[0].get("status", "")).lower()
                    status = "SUCCESS" if source_status == "success" else "FAILED"
                    reason = str(match.iloc[0].get("message", ""))
                elif len(match) == 0:
                    status, reason = "NOT_RUN", "expected parameter combination is absent"
                else:
                    status, reason = "INCOMPLETE", "duplicate rows exist for one parameter combination"
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "semi_real_size_factor": float(factor),
                        "expected_doublet_rate": float(rate),
                        "model_fit_key": f"{dataset}|seed={int(seed)}|factor={float(factor):g}",
                        "expected_rate_rows_share_one_model_fit": True,
                        "source_row_count": int(len(match)),
                        "status": status,
                        "reason": reason,
                    }
                )
    return pd.DataFrame(rows)


def metric_contract_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": "high_RNA_singlet_FPR",
                "population": "held-out semi-real test cells",
                "numerator_definition": "high-RNA singlets among the top n_true_test_doublets overall scores",
                "denominator_definition": "all held-out test high-RNA singlets",
                "score_used": "P(homotypic_doublet)+P(heterotypic_doublet)",
                "threshold_rule": "rank top K where K is the true test homotypic+heterotypic count",
                "threshold_source_split": "test scores and test truth prevalence",
                "expected_rate_used": "no",
                "higher_or_lower_is_better": "lower",
                "implementation_file": "src/duodose/semireal_metrics.py",
                "implementation_function": "high_rna_singlet_top_budget_fpr",
                "contract_status": "WARNING",
                "notes": "no fixed probability threshold or argmax; historical NumPy argsort has no explicit cell-ID tie break",
            },
            {
                "metric": "high_RNA_singlet_FPR_at_expected_rate",
                "population": "held-out semi-real test cells",
                "numerator_definition": "high-RNA singlets among exactly round(expected_rate*n_test_cells) top overall scores",
                "denominator_definition": "all held-out test high-RNA singlets",
                "score_used": "P(homotypic_doublet)+P(heterotypic_doublet)",
                "threshold_rule": "exact rank budget K=round(expected_rate*N); score descending then cell ID ascending",
                "threshold_source_split": "test score distribution",
                "expected_rate_used": "yes",
                "higher_or_lower_is_better": "lower",
                "implementation_file": "reproducibility/run_parameter_sensitivity.py",
                "implementation_function": "_candidate_fpr / deterministic_top_fraction",
                "contract_status": "PASS",
                "notes": "no fixed probability threshold or argmax; K=round(rate*N)",
            },
        ]
    )
