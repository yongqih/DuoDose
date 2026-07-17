"""Audit experimental-label blind spots without changing frozen DuoDose W2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from duodose.net import probabilities_to_scores  # noqa: E402
from duodose.protocol import load_final_protocol  # noqa: E402
from reproducibility.lib.common import load_dataset_exact, run_protocol_models, split_csv  # noqa: E402


DIAGNOSTIC_CANDIDATES = (
    ("dosage_residual", "residualized dosage", True),
    ("cluster_gene_robust_z", "nFeature residual given cluster-relative RNA complexity", True),
    ("handcrafted_dosage_raw_score", "absolute dosage", True),
    ("identity_inlier_score", "identity-inlier evidence", False),
    ("handcrafted_identity_mixture_score", "identity-mixture evidence", True),
    ("handcrafted_artificial_doublet_neighbor_score", "artificial-neighbor proximity", True),
    ("cluster_marker_dosage_robust_z", "cluster-relative marker dosage", True),
)

UNAVAILABLE_DIAGNOSTICS = (
    ("expression_entropy_residual", "expression entropy residual", "not exported by frozen W2 SafeFeatures"),
    ("sparsity_dropout_residual", "sparsity/dropout residual", "not exported by frozen W2 SafeFeatures"),
    ("RNA_matched_centroid_distance", "RNA-matched centroid distance", "not exported by frozen W2 SafeFeatures"),
    ("within_cluster_neighbor_density", "within-cluster neighbor density", "not exported as a primitive frozen W2 feature"),
    ("broad_cluster_purity", "broad-cluster purity", "not exported as a primitive frozen W2 feature"),
)

def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


GROUP_NAMES = {
    "A_constructed_homotypic": "constructed homotypic doublets",
    "B_constructed_heterotypic": "constructed heterotypic doublets",
    "C_nominal_highRNA_FP": "nominal high-RNA false positives",
    "D_matched_highRNA_excluded": "matched experimentally labeled high-RNA singlets excluded from top K",
    "E_matched_ordinary_singlet": "matched ordinary experimentally labeled singlets",
    "F_experimental_doublet": "experimentally labeled doublets",
}


def _portable_source_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return f"external_input/{resolved.name}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--conversion-dir", required=True)
    parser.add_argument("--output-dir", default="results/final_v1/label_uncertainty_audit")
    parser.add_argument("--real-application-dir", default="results/final_v1/real_application")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--convert-rds", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _exact_top_k(score: pd.Series, k: int) -> pd.Series:
    table = pd.DataFrame(
        {"score": pd.to_numeric(score, errors="coerce").fillna(-np.inf), "cell_id": score.index.astype(str)},
        index=score.index,
    ).sort_values(["score", "cell_id"], ascending=[False, True], kind="mergesort")
    selected = pd.Series(False, index=score.index)
    selected.loc[table.index[: max(0, min(int(k), len(table)))]] = True
    return selected


def _high_rna_mask(scores: pd.DataFrame, quantile: float) -> pd.Series:
    counts = pd.to_numeric(scores["nCount"], errors="coerce")
    cluster = _cluster_series(scores)
    thresholds = counts.groupby(cluster).transform(lambda values: values.quantile(float(quantile)))
    return counts.ge(thresholds) & counts.notna()


def _cluster_series(frame: pd.DataFrame) -> pd.Series:
    for column in ("benchmark_cluster", "duodose_cluster", "cluster", "cluster_label"):
        if column in frame:
            return frame[column].astype(str)
    raise ValueError("fitted-reference score frame has no projected cluster column")


def _detected_gene_count(adata: Any) -> pd.Series:
    matrix = adata.layers["counts"] if "counts" in adata.layers else adata.X
    values = np.asarray((matrix > 0).sum(axis=1)).ravel()
    return pd.Series(values, index=adata.obs_names, dtype=float)


def _match_controls(
    cases: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    secondary_nfeature: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    available = set(controls.index.astype(str))
    pairs = []
    for case_id, case in cases.sort_index(kind="stable").iterrows():
        local = controls.loc[controls.index.astype(str).isin(available)]
        same = local.loc[local["cluster"].astype(str).eq(str(case["cluster"]))]
        if same.empty:
            continue
        distance = (same["log_nCount"] - float(case["log_nCount"])).abs()
        if secondary_nfeature:
            distance = distance + (same["log_nFeature"] - float(case["log_nFeature"])).abs()
        ranked = pd.DataFrame({"distance": distance, "control_id": same.index.astype(str)}, index=same.index).sort_values(
            ["distance", "control_id"], kind="mergesort"
        )
        control_id = ranked.index[0]
        available.remove(str(control_id))
        pairs.append(
            {
                "case_cell_id": str(case_id),
                "control_cell_id": str(control_id),
                "cluster": str(case["cluster"]),
                "absolute_log_nCount_difference": float(abs(case["log_nCount"] - same.loc[control_id, "log_nCount"])),
                "absolute_log_nFeature_difference": float(abs(case["log_nFeature"] - same.loc[control_id, "log_nFeature"])),
                "matching_contract": "cluster+log_nCount+log_nFeature" if secondary_nfeature else "cluster+log_nCount",
            }
        )
    pair_frame = pd.DataFrame(pairs)
    matched = controls.loc[pair_frame["control_cell_id"]].copy() if not pair_frame.empty else controls.iloc[0:0].copy()
    return matched, pair_frame


def _deduplicate_features(frame: pd.DataFrame, candidates: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
    retained: list[str] = []
    audit: list[dict[str, Any]] = []
    for feature in candidates:
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
        duplicate_of = ""
        maximum = np.nan
        for representative in retained:
            other = pd.to_numeric(frame[representative], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values) & np.isfinite(other)
            current = float(np.max(np.abs(values[finite] - other[finite]))) if finite.any() else np.inf
            if current <= 1e-12 and np.array_equal(np.isfinite(values), np.isfinite(other)):
                duplicate_of = representative
                maximum = current
                break
        if not duplicate_of:
            retained.append(feature)
        audit.append(
            {
                "feature": feature,
                "availability": "available",
                "exact_duplicate_of": duplicate_of,
                "maximum_absolute_difference": maximum,
                "retained_for_diagnostic_model": not bool(duplicate_of),
            }
        )
    return retained, audit


def _cliffs_delta(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if not len(positive) or not len(negative):
        return np.nan
    combined = np.r_[positive, negative]
    ranks = rankdata(combined)
    rank_sum = ranks[: len(positive)].sum()
    u = rank_sum - len(positive) * (len(positive) + 1) / 2
    return float((2 * u) / (len(positive) * len(negative)) - 1)


def _feature_statistics(dataset: str, grouped: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    case = grouped.loc[grouped["true_cell_group"].eq("C_nominal_highRNA_FP")]
    control = grouped.loc[grouped["true_cell_group"].eq("D_matched_highRNA_excluded")]
    for feature in features:
        case_values = pd.to_numeric(case[feature], errors="coerce").to_numpy(dtype=float)
        control_values = pd.to_numeric(control[feature], errors="coerce").to_numpy(dtype=float)
        pooled = np.r_[case_values[np.isfinite(case_values)], control_values[np.isfinite(control_values)]]
        scale = float(np.std(pooled, ddof=1)) if len(pooled) > 1 else np.nan
        auroc = np.nan
        if np.isfinite(case_values).sum() and np.isfinite(control_values).sum():
            y = np.r_[np.ones(np.isfinite(case_values).sum()), np.zeros(np.isfinite(control_values).sum())]
            x = np.r_[case_values[np.isfinite(case_values)], control_values[np.isfinite(control_values)]]
            auroc = float(roc_auc_score(y, x))
        for group_id, group in grouped.groupby("true_cell_group", sort=True):
            values = pd.to_numeric(group[feature], errors="coerce")
            rows.append(
                {
                    "dataset": dataset,
                    "feature": feature,
                    "group": group_id,
                    "group_description": GROUP_NAMES[group_id],
                    "n": int(values.notna().sum()),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "matched_standardized_effect_C_minus_D": float((np.nanmedian(case_values) - np.nanmedian(control_values)) / scale)
                    if np.isfinite(scale) and scale > 0
                    else np.nan,
                    "Cliffs_delta_C_vs_D": _cliffs_delta(case_values, control_values),
                    "AUROC_C_vs_D": auroc,
                    "correlation_with_nCount": float(values.corr(pd.to_numeric(group["nCount"], errors="coerce"), method="spearman")),
                    "correlation_with_nFeature": float(values.corr(pd.to_numeric(group["nFeature"], errors="coerce"), method="spearman")),
                }
            )
    return pd.DataFrame(rows)


def _identity_column(obs: pd.DataFrame) -> str | None:
    preferred = ("HTO", "hashtag", "donor", "genotype", "sample", "sample_id", "orig.ident")
    lower = {str(column).lower(): str(column) for column in obs.columns}
    for token in preferred:
        matches = [column for key, column in lower.items() if token.lower() in key]
        for column in matches:
            values = obs[column].dropna().astype(str)
            if values.nunique() > 1 and values.nunique() < max(2, len(values) // 2):
                return column
    return None


def _blind_spot_rows(dataset: str, obs: pd.DataFrame, cluster: pd.Series, evidence: str) -> tuple[dict[str, Any], pd.DataFrame]:
    identity_column = _identity_column(obs)
    if identity_column is None:
        return (
            {
                "dataset": dataset,
                "identity_column": "",
                "P_same_identity": np.nan,
                "P_observable": np.nan,
                "P_same_identity_given_homotypic": np.nan,
                "cross_identity_homotypic_fraction": np.nan,
                "theoretical_invisible_fraction": np.nan,
                "sample_size": int(len(obs)),
                "reliability": "NOT_AVAILABLE",
                "assumptions": "sample/donor/hashtag assignments are unavailable; no estimate fabricated",
                "evidence_source": evidence,
            },
            pd.DataFrame(columns=["dataset", "identity", "n_cells", "proportion", "identity_column"]),
        )
    identity = obs[identity_column].astype(str)
    proportions = identity.value_counts(normalize=True)
    p_same = float(np.square(proportions).sum())
    joint = pd.crosstab(cluster.reindex(obs.index).astype(str), identity, normalize="all")
    cluster_proportions = cluster.reindex(obs.index).astype(str).value_counts(normalize=True)
    denominator = float(np.square(cluster_proportions).sum())
    p_same_hom = float(np.square(joint.to_numpy()).sum() / denominator) if denominator > 0 else np.nan
    rows = pd.DataFrame(
        {
            "dataset": dataset,
            "identity": proportions.index,
            "n_cells": identity.value_counts().reindex(proportions.index).to_numpy(),
            "proportion": proportions.to_numpy(),
            "identity_column": identity_column,
        }
    )
    return (
        {
            "dataset": dataset,
            "identity_column": identity_column,
            "P_same_identity": p_same,
            "P_observable": 1 - p_same,
            "P_same_identity_given_homotypic": p_same_hom,
            "cross_identity_homotypic_fraction": 1 - p_same_hom,
            "theoretical_invisible_fraction": p_same,
            "sample_size": int(len(obs)),
            "reliability": "AVAILABLE_FROM_LOCAL_ASSIGNMENTS",
            "assumptions": "random pairing within the retained observed-cell identity distribution",
            "evidence_source": evidence,
        },
        rows,
    )


def _build_groups(run: Any, loaded: Any, real_application_dir: Path, quantile: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[dict[str, Any]]]:
    fitted = run.fitted_backends["DuoDose"]
    real_probabilities = run.method_probabilities_real["DuoDose"]
    real_overall, real_hom, real_het = probabilities_to_scores(real_probabilities)
    k = int(loaded.adata.obs["experimental_doublet"].astype(int).sum())
    top_k = _exact_top_k(real_overall, k)
    real = run.fully_real_scores.copy()
    real["nCount"] = pd.to_numeric(real["nCount"], errors="coerce")
    if "nFeature" not in real:
        matrix = loaded.adata.layers["counts"] if "counts" in loaded.adata.layers else loaded.adata.X
        real["nFeature"] = np.asarray((matrix > 0).sum(axis=1)).ravel()
    real["nFeature"] = pd.to_numeric(real["nFeature"], errors="coerce")
    real["log_nCount"] = np.log1p(real["nCount"])
    real["log_nFeature"] = np.log1p(real["nFeature"])
    real["cluster"] = _cluster_series(real)
    real["experimental_doublet"] = loaded.adata.obs["experimental_doublet"].astype(int).reindex(real.index)
    real["high_RNA"] = _high_rna_mask(real, quantile)
    real["top_K"] = top_k.reindex(real.index)
    real["overall_doublet_probability"] = real_overall
    real["homotypic_probability"] = real_hom
    real["heterotypic_probability"] = real_het

    cases = real.loc[real["experimental_doublet"].eq(0) & real["high_RNA"] & real["top_K"]].copy()
    excluded_high = real.loc[real["experimental_doublet"].eq(0) & real["high_RNA"] & ~real["top_K"]].copy()
    ordinary = real.loc[real["experimental_doublet"].eq(0) & ~real["high_RNA"]].copy()
    matched_d_primary, primary_pairs = _match_controls(cases, excluded_high, secondary_nfeature=False)
    matched_d_secondary, secondary_pairs = _match_controls(cases, excluded_high, secondary_nfeature=True)
    matched_cases = cases.loc[secondary_pairs["case_cell_id"]].copy() if not secondary_pairs.empty else cases.iloc[0:0].copy()
    matched_e, ordinary_pairs = _match_controls(matched_cases, ordinary, secondary_nfeature=True)
    final_case_ids = pd.Index(ordinary_pairs["case_cell_id"]) if not ordinary_pairs.empty else pd.Index([])
    secondary_pairs = secondary_pairs.loc[secondary_pairs["case_cell_id"].isin(final_case_ids)].copy()
    ordinary_pairs = ordinary_pairs.loc[ordinary_pairs["case_cell_id"].isin(final_case_ids)].copy()
    matched_cases = cases.loc[final_case_ids].copy()
    matched_d_secondary = excluded_high.loc[secondary_pairs["control_cell_id"]].copy()
    matched_e = ordinary.loc[ordinary_pairs["control_cell_id"]].copy()
    for pair_frame in (primary_pairs, secondary_pairs, ordinary_pairs):
        pair_frame["n_cases_available"] = int(len(cases))
        pair_frame["n_cases_retained_for_secondary_contract"] = int(len(matched_cases))
        pair_frame["secondary_contract_retention_fraction"] = (
            float(len(matched_cases) / len(cases)) if len(cases) else np.nan
        )
    matching = pd.concat(
        [
            primary_pairs.assign(control_group="D_highRNA_excluded_primary"),
            secondary_pairs.assign(control_group="D_highRNA_excluded_secondary"),
            ordinary_pairs.assign(control_group="E_ordinary_singlet"),
        ],
        ignore_index=True,
    )
    # The secondary matched contract is the conservative main analysis.
    groups = []
    for frame, group in (
        (matched_cases, "C_nominal_highRNA_FP"),
        (matched_d_secondary, "D_matched_highRNA_excluded"),
        (matched_e, "E_matched_ordinary_singlet"),
        (real.loc[real["experimental_doublet"].eq(1)], "F_experimental_doublet"),
    ):
        groups.append(frame.assign(true_cell_group=group, source_domain="experimental_background"))
    test = run.test_scores.copy()
    test_probabilities = run.method_probabilities_test["DuoDose"]
    test_overall, test_hom, test_het = probabilities_to_scores(test_probabilities)
    test["overall_doublet_probability"] = test_overall
    test["homotypic_probability"] = test_hom
    test["heterotypic_probability"] = test_het
    test["nCount"] = pd.to_numeric(test["nCount"], errors="coerce")
    if "nFeature" not in test:
        test["nFeature"] = _detected_gene_count(run.bundle.test_adata).reindex(test.index)
    test["top_K"] = _exact_top_k(test_overall, int(test["true_label"].astype(str).isin({"homotypic_doublet", "heterotypic_doublet"}).sum()))
    for label, group in (("homotypic_doublet", "A_constructed_homotypic"), ("heterotypic_doublet", "B_constructed_heterotypic")):
        groups.append(test.loc[test["true_label"].astype(str).eq(label)].assign(true_cell_group=group, source_domain="semi_real_test"))
    grouped = pd.concat(groups, axis=0, sort=False)

    available = [feature for feature, _, _ in DIAGNOSTIC_CANDIDATES if feature in grouped]
    retained, duplicate_audit = _deduplicate_features(pd.concat([run.fit_scores, run.validation_scores]), available)
    dictionary = []
    semantics = {feature: (description, expected) for feature, description, expected in DIAGNOSTIC_CANDIDATES}
    for row in duplicate_audit:
        description, expected = semantics[row["feature"]]
        dictionary.append({**row, "description": description, "higher_is_homotypic_like_hypothesis": expected, "source": "frozen W2 raw SafeFeature score frame"})
    for feature, description, reason in UNAVAILABLE_DIAGNOSTICS:
        dictionary.append(
            {
                "feature": feature,
                "availability": "NOT_AVAILABLE",
                "exact_duplicate_of": "",
                "maximum_absolute_difference": np.nan,
                "retained_for_diagnostic_model": False,
                "description": description,
                "higher_is_homotypic_like_hypothesis": "",
                "source": reason,
            }
        )

    # Fit the diagnostic-only model using semi-real fit/validation rows only.
    split_models = []
    for split_name, scores, split_adata in (
        ("fit", run.fit_scores, run.bundle.fit_adata),
        ("validation", run.validation_scores, run.bundle.val_adata),
    ):
        scores = scores.copy()
        if "nFeature" not in scores:
            scores["nFeature"] = _detected_gene_count(split_adata).reindex(scores.index)
        probabilities = fitted.predict_probabilities(scores)
        score, _, _ = probabilities_to_scores(probabilities)
        top = _exact_top_k(score, int(scores["true_label"].astype(str).isin({"homotypic_doublet", "heterotypic_doublet"}).sum()))
        positives = scores.loc[scores["true_label"].astype(str).eq("homotypic_doublet")]
        negatives = scores.loc[scores["true_label"].astype(str).eq("high_RNA_singlet") & ~top]
        if "nFeature" not in scores:
            scores = scores.copy()
            scores["nFeature"] = np.nan
        positives = positives.copy()
        negatives = negatives.copy()
        for frame in (positives, negatives):
            frame["cluster"] = _cluster_series(frame)
            frame["log_nCount"] = np.log1p(pd.to_numeric(frame["nCount"], errors="coerce"))
            frame["log_nFeature"] = np.log1p(pd.to_numeric(frame.get("nFeature", np.nan), errors="coerce"))
        matched_negative, pairs = _match_controls(positives, negatives, secondary_nfeature=True)
        n = min(len(positives), len(matched_negative))
        selected_positive = positives.loc[pairs["case_cell_id"]].iloc[:n] if n else positives.iloc[0:0]
        selected_negative = matched_negative.iloc[:n]
        train_frame = pd.concat([selected_positive, selected_negative])
        target = np.r_[np.ones(len(selected_positive), dtype=int), np.zeros(len(selected_negative), dtype=int)]
        split_models.append((split_name, train_frame, target))
    fit_frame, fit_target = split_models[0][1], split_models[0][2]
    validation_frame, validation_target = split_models[1][1], split_models[1][2]
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=run.seed)),
        ]
    )
    model.fit(fit_frame[retained], fit_target)
    validation_score = model.predict_proba(validation_frame[retained])[:, 1]
    validation_auroc = float(roc_auc_score(validation_target, validation_score)) if len(np.unique(validation_target)) == 2 else np.nan
    validation_auprc = float(average_precision_score(validation_target, validation_score)) if len(np.unique(validation_target)) == 2 else np.nan
    combined_training = pd.concat([fit_frame, validation_frame])
    combined_target = np.r_[fit_target, validation_target]
    model.fit(combined_training[retained], combined_target)
    grouped["homotypic_like_diagnostic_score"] = model.predict_proba(grouped[retained])[:, 1]
    manifest = {
        "model": "regularized logistic regression",
        "purpose": "diagnostic-only homotypic-likeness score; not a doublet probability",
        "features": retained,
        "prohibited_outputs_used": False,
        "raw_nCount_or_nFeature_used_as_model_inputs": False,
        "training_domains": ["semi_real_fit", "semi_real_validation"],
        "n_fit_rows": int(len(fit_target)),
        "n_validation_rows": int(len(validation_target)),
        "validation_AUROC": validation_auroc,
        "validation_AUPRC": validation_auprc,
    }
    return grouped, matching, pd.DataFrame(dictionary), retained, [manifest]


def _external_consensus(dataset: str, group: pd.DataFrame, real_dir: Path) -> pd.DataFrame:
    score_path = real_dir / dataset / "seed_0" / "real_application_method_scores.csv.gz"
    if not score_path.is_file():
        return pd.DataFrame([{"dataset": dataset, "status": "NOT_AVAILABLE", "message": f"missing {score_path}"}])
    scores = pd.read_csv(score_path).set_index("cell_id")
    cases = group.loc[group["true_cell_group"].eq("C_nominal_highRNA_FP")]
    methods = [method for method in ("Scrublet", "scDblFinder", "DoubletFinder", "scds") if f"{method}_common_display_top_k" in scores]
    aligned = scores.reindex(cases.index.astype(str))
    support = pd.DataFrame({method: aligned[f"{method}_common_display_top_k"].fillna(False).astype(bool) for method in methods})
    counts = support.sum(axis=1) if methods else pd.Series(0, index=aligned.index)
    cluster_counts = cases["cluster"].astype(str).value_counts().sort_values(ascending=False)
    cluster_summary = ";".join(f"{cluster}:{count}" for cluster, count in cluster_counts.items())
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "status": "AVAILABLE" if methods else "NOT_AVAILABLE",
                "n_nominal_false_positives": int(len(cases)),
                "external_methods_available": ",".join(methods),
                "proportion_supported_by_at_least_one_external_method": float((counts >= 1).mean()) if len(counts) else np.nan,
                "proportion_supported_by_at_least_two_external_methods": float((counts >= 2).mean()) if len(counts) else np.nan,
                "proportion_supported_only_by_DuoDose": float((counts == 0).mean()) if len(counts) else np.nan,
                "median_homotypic_probability": float(pd.to_numeric(cases["homotypic_probability"], errors="coerce").median()),
                "median_heterotypic_probability": float(pd.to_numeric(cases["heterotypic_probability"], errors="coerce").median()),
                "fraction_homotypic_evidence_exceeds_heterotypic": float(
                    (pd.to_numeric(cases["homotypic_probability"], errors="coerce") > pd.to_numeric(cases["heterotypic_probability"], errors="coerce")).mean()
                )
                if len(cases)
                else np.nan,
                "broad_cluster_counts": cluster_summary,
                "truth_interpretation": "descriptive consensus only; method agreement does not define truth",
            }
        ]
    )


def _write_figures(output: Path, statistics: pd.DataFrame, scores: pd.DataFrame, consensus: pd.DataFrame, theoretical: pd.DataFrame, real_dir: Path) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "axes.titleweight": "bold"})
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    ax.axis("off")
    labels = ["Observed cells", "Cross-identity multiplets\nmay be detectable", "Same-identity multiplets\nmay remain unresolved", "Label-relative audit"]
    for index, label in enumerate(labels):
        x = 0.12 + index * 0.25
        ax.text(x, 0.5, label, ha="center", va="center", bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#4C78A8"})
        if index < len(labels) - 1:
            ax.annotate("", xy=(x + 0.10, 0.5), xytext=(x + 0.15, 0.5), arrowprops={"arrowstyle": "<-", "color": "#555555"})
    fig.tight_layout()
    fig.savefig(figures / "experimental_label_blind_spot_schematic.png", dpi=300)
    fig.savefig(figures / "experimental_label_blind_spot_schematic.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    available = theoretical.loc[theoretical["reliability"].ne("NOT_AVAILABLE")]
    if available.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "Sample/donor/hashtag assignments were not retained;\nno theoretical invisible fraction was fabricated.", ha="center", va="center")
    else:
        ax.bar(np.arange(len(available)), available["theoretical_invisible_fraction"], color="#4C78A8")
        ax.set_xticks(np.arange(len(available)), available["dataset"], rotation=45, ha="right")
        ax.set_ylabel("Theoretical same-identity invisible fraction")
    fig.tight_layout()
    fig.savefig(figures / "theoretical_same_identity_invisible_fraction.png", dpi=300)
    fig.savefig(figures / "theoretical_same_identity_invisible_fraction.pdf")
    plt.close(fig)

    effects = statistics.loc[statistics["group"].eq("C_nominal_highRNA_FP")].groupby("feature")["matched_standardized_effect_C_minus_D"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7.1, 4.2))
    ax.barh(np.arange(len(effects)), effects, color=np.where(effects >= 0, "#C43C39", "#4C78A8"))
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(np.arange(len(effects)), effects.index)
    ax.set_xlabel("Matched standardized effect: nominal FP minus excluded high-RNA")
    fig.tight_layout()
    fig.savefig(figures / "nominal_FP_matched_diagnostic_effects.png", dpi=300)
    fig.savefig(figures / "nominal_FP_matched_diagnostic_effects.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.1, 4.2))
    ordered_groups = list(GROUP_NAMES)
    values = [scores.loc[scores["true_cell_group"].eq(group), "homotypic_like_diagnostic_score"].dropna().to_numpy() for group in ordered_groups]
    ax.boxplot(values, tick_labels=[GROUP_NAMES[group] for group in ordered_groups], showfliers=False)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("Homotypic-like diagnostic score")
    fig.tight_layout()
    fig.savefig(figures / "homotypic_like_score_distributions.png", dpi=300)
    fig.savefig(figures / "homotypic_like_score_distributions.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.1, 4.0))
    if not consensus.empty:
        values = consensus.groupby("dataset")["proportion_supported_by_at_least_one_external_method"].first().dropna().sort_values()
        ax.bar(np.arange(len(values)), values, color="#59A14F")
        ax.set_xticks(np.arange(len(values)), values.index, rotation=45, ha="right")
        ax.set_ylabel("Nominal FPs supported by >=1 external method")
    fig.tight_layout()
    fig.savefig(figures / "external_method_consensus.png", dpi=300)
    fig.savefig(figures / "external_method_consensus.pdf")
    plt.close(fig)

    coordinates_path = real_dir / "hm-12k" / "seed_0" / "real_application_umap_coordinates.csv.gz"
    if coordinates_path.is_file():
        coordinates = pd.read_csv(coordinates_path).set_index("cell_id")
        hm = scores.loc[(scores["dataset"].eq("hm-12k")) & scores["source_domain"].eq("experimental_background")]
        coordinates = coordinates.join(hm[["true_cell_group"]], how="left")
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        ax.scatter(coordinates["umap_1"], coordinates["umap_2"], s=2, color="#D7D9DC", rasterized=True)
        palette = {"C_nominal_highRNA_FP": "#C43C39", "D_matched_highRNA_excluded": "#4C78A8", "F_experimental_doublet": "#59A14F"}
        for group, color in palette.items():
            subset = coordinates.loc[coordinates["true_cell_group"].eq(group)]
            ax.scatter(subset["umap_1"], subset["umap_2"], s=7, color=color, label=GROUP_NAMES[group], rasterized=True)
        ax.legend(frameon=False, fontsize=7, loc="best")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        fig.tight_layout()
        fig.savefig(figures / "hm12k_nominal_FP_UMAP.png", dpi=300)
        fig.savefig(figures / "hm12k_nominal_FP_UMAP.pdf")
        plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    protocol = load_final_protocol(args.protocol)
    datasets = split_csv(args.datasets) or list(protocol["datasets"]["real_doublet_enriched"])
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    real_dir = Path(args.real_application_dir).resolve()
    method_rows = []
    theoretical_rows = []
    identity_frames = []
    group_counts = []
    matching_frames = []
    dictionary_frames = []
    statistics_frames = []
    score_frames = []
    consensus_frames = []
    model_manifests = []
    failures = []
    for position, dataset in enumerate(datasets, 1):
        print(f"[{position}/{len(datasets)}] {dataset}: loading and reconstructing frozen W2 diagnostics", flush=True)
        try:
            loaded = load_dataset_exact(
                args.data_dir,
                dataset,
                conversion_dir=args.conversion_dir,
                convert_rds=bool(args.convert_rds),
            )
            run = run_protocol_models(loaded, protocol_path=args.protocol, seed=int(args.seed), backends=("rf",))
            conversion_report = loaded.source_path / "conversion_report.json" if loaded.source_path.is_dir() else loaded.source_path
            evidence = f"{_portable_source_path(conversion_report)}; conversion report retains binary labels only"
            identity_column = _identity_column(loaded.adata.obs)
            method_rows.append(
                {
                    "dataset": dataset,
                    "experimental_doublet_label_method": "UNKNOWN",
                    "HTO_cell_hashing": "UNKNOWN",
                    "MULTI_seq_or_lipid_barcode": "UNKNOWN",
                    "genotype_or_donor_demultiplexing": "UNKNOWN",
                    "known_cell_line_mixture": "UNKNOWN",
                    "same_sample_doublets_theoretically_observable": "UNKNOWN",
                    "same_sample_homotypic_doublets_theoretically_observable": "UNKNOWN",
                    "retained_identity_assignment_column": identity_column or "",
                    "evidence_source_inside_repository": evidence,
                    "status": "UNKNOWN" if identity_column is None else "IDENTITY_ASSIGNMENTS_AVAILABLE_MECHANISM_UNKNOWN",
                }
            )
            grouped, matching, dictionary, retained, manifests = _build_groups(
                run, loaded, real_dir, float(protocol["semi_real"]["high_rna_quantile"])
            )
            grouped["dataset"] = dataset
            grouped["seed"] = int(args.seed)
            matching["dataset"] = dataset
            matching["seed"] = int(args.seed)
            dictionary["dataset"] = dataset
            statistics = _feature_statistics(dataset, grouped, retained)
            score_export_columns = [
                "dataset", "seed", "source_domain", "true_cell_group", "nCount", "nFeature", "cluster", "top_K",
                "overall_doublet_probability", "homotypic_probability", "heterotypic_probability",
                "homotypic_like_diagnostic_score", *retained,
            ]
            score_export = grouped.reindex(columns=score_export_columns).rename_axis("cell_id").reset_index()
            score_frames.append(score_export)
            matching_frames.append(matching)
            dictionary_frames.append(dictionary)
            statistics_frames.append(statistics)
            counts = grouped.groupby("true_cell_group").size().rename("n_cells").reset_index()
            counts.insert(0, "dataset", dataset)
            group_counts.append(counts)
            consensus_frames.append(_external_consensus(dataset, grouped, real_dir))
            cluster = _cluster_series(run.fully_real_scores)
            theoretical, proportions = _blind_spot_rows(dataset, loaded.adata.obs, cluster, evidence)
            theoretical_rows.append(theoretical)
            identity_frames.append(proportions)
            for manifest in manifests:
                manifest["dataset"] = dataset
                model_manifests.append(manifest)
        except Exception as exc:
            failures.append({"dataset": dataset, "status": "FAILED", "message": str(exc)})
            print(f"[{dataset}] FAILED: {exc}", flush=True)
            if not args.continue_on_error:
                break

    method = pd.DataFrame(method_rows)
    theoretical = pd.DataFrame(theoretical_rows)
    identity = pd.concat(identity_frames, ignore_index=True) if identity_frames else pd.DataFrame(columns=["dataset", "identity", "n_cells", "proportion", "identity_column"])
    counts = pd.concat(group_counts, ignore_index=True) if group_counts else pd.DataFrame()
    matching = pd.concat(matching_frames, ignore_index=True) if matching_frames else pd.DataFrame()
    dictionary = pd.concat(dictionary_frames, ignore_index=True).drop_duplicates(["feature", "availability", "exact_duplicate_of"]) if dictionary_frames else pd.DataFrame()
    statistics = pd.concat(statistics_frames, ignore_index=True) if statistics_frames else pd.DataFrame()
    scores = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    consensus = pd.concat(consensus_frames, ignore_index=True, sort=False) if consensus_frames else pd.DataFrame()
    consistency_rows = []
    if not statistics.empty:
        effects = statistics.loc[statistics["group"].eq("C_nominal_highRNA_FP")]
        for feature, frame in effects.groupby("feature"):
            values = pd.to_numeric(frame["matched_standardized_effect_C_minus_D"], errors="coerce").dropna()
            consistency_rows.append(
                {
                    "feature": feature,
                    "n_datasets": int(len(values)),
                    "median_standardized_effect": float(values.median()),
                    "fraction_positive_direction": float((values > 0).mean()) if len(values) else np.nan,
                    "fraction_consistent_with_median_direction": float((np.sign(values) == np.sign(values.median())).mean()) if len(values) else np.nan,
                }
            )
    consistency = pd.DataFrame(consistency_rows)
    theoretical_available = int(theoretical.get("reliability", pd.Series(dtype=str)).ne("NOT_AVAILABLE").sum())
    diagnostic_differences = []
    if not scores.empty:
        for dataset, frame in scores.groupby("dataset"):
            case = frame.loc[frame["true_cell_group"].eq("C_nominal_highRNA_FP"), "homotypic_like_diagnostic_score"]
            control = frame.loc[frame["true_cell_group"].eq("D_matched_highRNA_excluded"), "homotypic_like_diagnostic_score"]
            if len(case) and len(control):
                diagnostic_differences.append(float(case.median() - control.median()))
    n_positive_datasets = int((np.asarray(diagnostic_differences) > 0).sum()) if diagnostic_differences else 0
    conclusion = (
        "LABEL_UNCERTAINTY_SIGNAL_PARTIAL"
        if len(diagnostic_differences) >= 2 and n_positive_datasets >= max(2, int(np.ceil(len(diagnostic_differences) * 0.60)))
        else "NO_CLEAR_LABEL_UNCERTAINTY_SIGNAL"
    )
    # Strong is deliberately unavailable without a quantified theoretical blind spot.
    if theoretical_available == 0 and conclusion == "LABEL_UNCERTAINTY_SIGNAL_PARTIAL":
        conclusion_reason = "matched diagnostic enrichment was present, but retained identity assignments were unavailable, so the theoretical blind spot could not be quantified"
    else:
        conclusion_reason = "the prespecified matched diagnostic consistency criterion was not met"

    atomic_write_csv(output / "dataset_label_method_audit.csv", method)
    atomic_write_csv(output / "experimental_identity_proportions.csv", identity)
    atomic_write_csv(output / "theoretical_blind_spot_estimates.csv", theoretical)
    atomic_write_csv(output / "nominal_fp_group_counts.csv", counts)
    atomic_write_csv(output / "nominal_fp_matching_audit.csv", matching)
    atomic_write_csv(output / "diagnostic_feature_dictionary.csv", dictionary)
    atomic_write_csv(output / "nominal_fp_feature_statistics.csv", statistics)
    atomic_write_csv(output / "nominal_fp_cross_dataset_consistency.csv", consistency)
    atomic_write_csv(output / "external_method_consensus.csv", consensus)
    scores.to_csv(output / "homotypic_like_scores.csv.gz", index=False)
    atomic_write_json(output / "homotypic_like_model_manifest.json", {"models": model_manifests, "failures": failures})
    _write_figures(output, statistics, scores, consensus, theoretical, real_dir)
    report = [
        "# Experimental-label uncertainty audit",
        "",
        f"Conclusion: **{conclusion}**",
        "",
        conclusion_reason + ".",
        "",
        "This audit uses the terms experimentally labeled singlet, nominal false positive, and homotypic-like profile. It does not establish that nominal false positives are true doublets.",
        "",
        f"Retained sample/donor/hashtag assignments were available for theoretical calculation in {theoretical_available} dataset(s). Missing assignments were recorded as NOT_AVAILABLE rather than inferred from dataset names.",
        "",
        f"Nominal false positives had a higher median homotypic-like diagnostic score than secondary-contract matched excluded high-RNA singlets in {n_positive_datasets} of {len(diagnostic_differences)} evaluable datasets.",
        "",
        "The diagnostic model excludes DuoDose probabilities, combined scores, raw nCount, and raw nFeature. It is descriptive and is not part of production DuoDose.",
        "",
        "External-method consensus is descriptive only and does not define truth.",
        "",
        "Semi-real high-RNA FPR remains a strict label-relative stress test, not an unbiased estimate of absolute biological FPR.",
    ]
    (output / "label_uncertainty_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if failures:
        atomic_write_csv(output / "audit_failures.csv", pd.DataFrame(failures))
    print(f"Label uncertainty audit: {conclusion}; outputs={output}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
