"""Audit manuscript-facing background and high-RNA metric contracts from frozen results."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from duodose.protocol import load_final_protocol  # noqa: E402
from duodose.semireal_bundle import make_parent_disjoint_semireal_bundle  # noqa: E402
from reproducibility.lib.common import load_dataset_exact  # noqa: E402


def _stable_id_hash(values: pd.Index | list[str]) -> str:
    ordered = sorted(map(str, values))
    return hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()


def _pool_row(name: str, ids: pd.Index | list[str], labels: pd.Series, *, role: str) -> dict[str, object]:
    cell_ids = pd.Index(map(str, ids))
    joined = labels.reindex(cell_ids)
    n_singlet = int(joined.eq(0).sum())
    n_doublet = int(joined.eq(1).sum())
    n_unknown = int(joined.isna().sum())
    return {
        "dataset": "cline-ch",
        "pool": name,
        "scientific_role": role,
        "total_cells": int(len(cell_ids)),
        "n_experimentally_labeled_singlet": n_singlet,
        "n_experimentally_labeled_doublet": n_doublet,
        "n_unknown_or_constructed": n_unknown,
        "experimentally_labeled_doublets_present": bool(n_doublet > 0),
        "cell_id_sha256": _stable_id_hash(cell_ids),
        "hash_definition": "SHA256 of unique cell IDs sorted lexicographically and joined by newline",
    }


def _background_audit(results: Path, protocol_path: Path) -> pd.DataFrame:
    loaded = load_dataset_exact(
        results / "data" / "converted",
        "cline-ch",
        conversion_dir=results / "data",
        convert_rds=False,
    )
    protocol = load_final_protocol(protocol_path)
    original = loaded.adata.copy()
    experimental_labels = original.obs["experimental_doublet"].astype(int).copy()
    blind = original.copy()
    blind.obs["experimental_doublet"] = 0
    sizes = protocol["semi_real"]
    clustering = protocol["clustering"]
    construction = protocol["construction"]
    bundle = make_parent_disjoint_semireal_bundle(
        blind,
        dataset="cline-ch",
        seed=0,
        n_singlets=min(int(sizes["n_reference_singlets"]), int(blind.n_obs // 2)),
        n_train_homotypic_doublets=int(sizes["n_train_homotypic_doublets"]),
        n_train_heterotypic_doublets=int(sizes["n_train_heterotypic_doublets"]),
        n_test_homotypic_doublets=int(sizes["n_test_homotypic_doublets"]),
        n_test_heterotypic_doublets=int(sizes["n_test_heterotypic_doublets"]),
        n_clusters=int(clustering["n_clusters"]),
        test_parent_fraction=0.40,
        validation_parent_fraction=0.25,
        high_rna_quantile=float(sizes["high_rna_quantile"]),
        min_cluster_size=int(clustering["min_cluster_size"]),
        construction_variant=str(construction["construction_variant"]),
    )
    frozen_parent_map = pd.read_csv(results / "controlled" / "cline-ch" / "seed_0" / "semireal_parent_map.csv.gz")
    compare_columns = list(frozen_parent_map.columns)
    reconstructed = bundle.parent_map[compare_columns].astype(str).reset_index(drop=True)
    frozen = frozen_parent_map[compare_columns].astype(str).reset_index(drop=True)
    if not reconstructed.equals(frozen):
        raise RuntimeError("deterministic audit reconstruction does not reproduce the frozen cline-ch parent map")

    labels = experimental_labels.copy()
    labels.index = labels.index.astype(str)
    rows: list[dict[str, object]] = []
    rows.append(_pool_row("all_observed_background_cells", original.obs_names, labels, role="label-blinded construction input"))
    rows.append(_pool_row("construction_reference_background", bundle.reference_cell_ids, labels, role="reference-side background before parent-disjoint split"))
    rows.append(_pool_row("semi_real_parent_pool", bundle.synthetic_parent_cell_ids, labels, role="eligible parent-side background"))
    actual_parent_ids = pd.unique(bundle.parent_map[["parent_1_id", "parent_2_id"]].to_numpy().ravel()).tolist()
    rows.append(_pool_row("semi_real_parents_actually_used", actual_parent_ids, labels, role="distinct parents used by constructed doublets"))

    for split_name, adata in (("fit", bundle.fit_adata), ("validation", bundle.val_adata), ("test", bundle.test_adata)):
        rows.append(_pool_row(f"{split_name}_split", adata.obs_names, labels, role="observed-background and constructed rows"))
        origin = adata.obs["semireal_origin"].astype(str)
        background_ids = adata.obs_names[origin.isin({"observed_background", "real_labeled_singlet"})]
        rows.append(_pool_row(f"{split_name}_background_negative_population", background_ids, labels, role="label-blinded observed cells treated as controlled negatives"))
        if split_name == "fit":
            rows.append(_pool_row("fitted_reference_pool", background_ids, labels, role="SafeFeatureTransformer fit reference"))

    audit = pd.DataFrame(rows)
    audit["construction_variant"] = str(construction["construction_variant"])
    audit["experimental_labels_used_for_selection"] = False
    audit["frozen_parent_map_reproduced_exactly"] = True
    return audit


def _high_rna_contract(results: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        {
            "metric": "high_RNA_singlet_FPR",
            "usage": "primary manuscript comparison",
            "numerator": "held-out high-RNA singlets selected before reaching the matched homotypic-recall target",
            "denominator": "all held-out high-RNA singlets",
            "high_RNA_subset_definition": "observed test-background cells at or above the fit-split 90th-percentile library-size rule",
            "score_used": "method-specific overall doublet score",
            "threshold_rule": "smallest deterministic rank prefix recovering at least 50% of held-out homotypic doublets",
            "threshold_source": "held-out homotypic truth and score ranking; identical recall target for every method",
            "K_definition": "method-specific minimum candidate count required to reach 50% homotypic recall",
            "tie_handling": "score descending, then stable cell ID ascending",
            "expected_rate_dependency": False,
            "lower_is_better": True,
            "is_label_relative_fpr": True,
            "cross_dataset_suitable": True,
            "manuscript_recommendation": "MAIN_TEXT",
        },
        {
            "metric": "high_RNA_singlet_FPR_at_fixed_20pct_candidate_budget",
            "usage": "supplementary standardized candidate-budget sensitivity",
            "numerator": "selected held-out high-RNA singlets",
            "denominator": "all held-out high-RNA singlets",
            "high_RNA_subset_definition": "same held-out high-RNA singlet subset as the primary metric",
            "score_used": "method-specific overall doublet score",
            "threshold_rule": "select exactly round(0.20 * N_test) cells",
            "threshold_source": "frozen common 20% candidate fraction",
            "K_definition": "round(0.20 * number of test cells)",
            "tie_handling": "score descending, then stable cell ID ascending",
            "expected_rate_dependency": False,
            "lower_is_better": True,
            "is_label_relative_fpr": True,
            "cross_dataset_suitable": True,
            "manuscript_recommendation": "SUPPLEMENTARY",
        },
        {
            "metric": "high_RNA_singlet_FPR_at_matched_70pct_homotypic_recall",
            "usage": "supplementary matched-sensitivity robustness",
            "numerator": "held-out high-RNA singlets selected before reaching the matched homotypic-recall target",
            "denominator": "all held-out high-RNA singlets",
            "high_RNA_subset_definition": "same held-out high-RNA singlet subset as the primary metric",
            "score_used": "method-specific overall doublet score",
            "threshold_rule": "smallest deterministic rank prefix recovering at least 70% of held-out homotypic doublets",
            "threshold_source": "held-out homotypic truth and score ranking",
            "K_definition": "method-specific minimum candidate count required to reach 70% homotypic recall",
            "tie_handling": "score descending, then stable cell ID ascending",
            "expected_rate_dependency": False,
            "lower_is_better": True,
            "is_label_relative_fpr": True,
            "cross_dataset_suitable": True,
            "manuscript_recommendation": "SUPPLEMENTARY",
        },
        {
            "metric": "high_RNA_singlet_FPR_at_matched_80pct_homotypic_recall",
            "usage": "supplementary matched-sensitivity robustness",
            "numerator": "held-out high-RNA singlets selected before reaching the matched homotypic-recall target",
            "denominator": "all held-out high-RNA singlets",
            "high_RNA_subset_definition": "same held-out high-RNA singlet subset as the primary metric",
            "score_used": "method-specific overall doublet score",
            "threshold_rule": "smallest deterministic rank prefix recovering at least 80% of held-out homotypic doublets",
            "threshold_source": "held-out homotypic truth and score ranking",
            "K_definition": "method-specific minimum candidate count required to reach 80% homotypic recall",
            "tie_handling": "score descending, then stable cell ID ascending",
            "expected_rate_dependency": False,
            "lower_is_better": True,
            "is_label_relative_fpr": True,
            "cross_dataset_suitable": True,
            "manuscript_recommendation": "SUPPLEMENTARY",
        },
        {
            "metric": "high_RNA_singlet_FPR_at_true_doublet_budget",
            "usage": "historical equal-K sensitivity retained for continuity",
            "numerator": "selected held-out high-RNA singlets",
            "denominator": "all held-out high-RNA singlets",
            "high_RNA_subset_definition": "same held-out high-RNA singlet subset as the primary metric",
            "score_used": "method-specific overall doublet score",
            "threshold_rule": "select the top n_true_test_doublets cells",
            "threshold_source": "held-out score ranking",
            "K_definition": "number of constructed homotypic plus heterotypic doublets in the test split",
            "tie_handling": "score descending, then stable cell ID ascending",
            "expected_rate_dependency": False,
            "lower_is_better": True,
            "is_label_relative_fpr": True,
            "cross_dataset_suitable": False,
            "manuscript_recommendation": "SUPPLEMENTARY_ONLY",
        },
    ]
    contract = pd.DataFrame(rows)
    controlled = pd.read_csv(results / "tables" / "final_controlled_comparison_by_run.csv")
    pivot = controlled.pivot_table(index=["dataset", "seed"], columns="method", values="high_RNA_singlet_FPR", aggfunc="first")
    rows_out = []
    for method in ["Scrublet", "scDblFinder", "DoubletFinder", "scds"]:
        if "DuoDose" not in pivot or method not in pivot:
            continue
        delta = pivot["DuoDose"] - pivot[method]
        rows_out.append(
            {
                "comparison": f"DuoDose versus {method} at matched 50% homotypic recall",
                "n_paired_dataset_seed_runs": int(delta.notna().sum()),
                "duodose_mean": float(pivot["DuoDose"].mean()),
                "external_mean": float(pivot[method].mean()),
                "duodose_minus_external_mean": float(delta.mean()),
                "n_runs_duodose_lower": int((delta < 0).sum()),
                "conclusion": "lower matched-recall high-RNA FPR favors DuoDose" if float(delta.mean()) < 0 else "no mean DuoDose advantage",
            }
        )
    return contract, pd.DataFrame(rows_out)

def _write_markdown(background: pd.DataFrame, contract: pd.DataFrame, comparison: pd.DataFrame, output: Path) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        clean = frame.fillna("").astype(str).map(lambda value: value.replace("|", "\\|").replace("\n", " "))
        header = "| " + " | ".join(map(str, clean.columns)) + " |"
        separator = "| " + " | ".join("---" for _ in clean.columns) + " |"
        body = ["| " + " | ".join(row) + " |" for row in clean.itertuples(index=False, name=None)]
        return "\n".join([header, separator, *body])

    display = background[[
        "pool", "total_cells", "n_experimentally_labeled_singlet",
        "n_experimentally_labeled_doublet", "n_unknown_or_constructed", "cell_id_sha256",
    ]].copy()
    lines = [
        "# Background Population Contract Audit",
        "",
        "The frozen formal runner intentionally set `experimental_doublet = 0` before semi-real construction. "
        "The construction was therefore label-blind and used all observed cells as the background universe. "
        "The historical phrase `real labeled singlets` was inaccurate.",
        "",
        "The deterministic reconstruction reproduced the frozen parent map exactly. Experimental annotations were joined only after reconstruction for this audit.",
        "",
        markdown_table(display),
        "",
        "## Conclusion",
        "",
        "Experimentally labeled doublets entered the label-blinded construction reference, fitted SafeFeature reference, parent pool, and controlled negative populations. "
        "This is an intentional label-free background design with possible doublet contamination, not evidence that labels were used during fitting. No formal rerun is required, but manuscript wording must say `observed background cells` and disclose the contamination possibility.",
        "",
        "## High-RNA Metric Contract",
        "",
        markdown_table(contract),
        "",
        markdown_table(comparison),
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/final_v1")
    parser.add_argument("--protocol", default="reproducibility/configs/final_protocol.yaml")
    args = parser.parse_args()
    results = Path(args.results_dir).resolve()
    audit_dir = results / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    background = _background_audit(results, Path(args.protocol).resolve())
    contract, comparison = _high_rna_contract(results)
    background.to_csv(audit_dir / "background_population_contract_audit.csv", index=False)
    contract.to_csv(audit_dir / "high_rna_metric_contract_audit.csv", index=False)
    comparison.to_csv(audit_dir / "high_rna_formal_comparison_audit.csv", index=False)
    _write_markdown(background, contract, comparison, audit_dir / "background_population_contract_audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
