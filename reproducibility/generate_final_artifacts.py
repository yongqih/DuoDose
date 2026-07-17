"""Generate manuscript-facing tables, figures, and a conservative results report."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from duodose.protocol import load_final_protocol
from duodose.plotting_style import apply_manuscript_style, audit_figure_style_contract
from duodose.runtime_completeness import CANONICAL_RUNTIME_METHODS, build_runtime_method_completeness_audit


ALLOWED_METHODS = ["DuoDose", "DuoDose-DL", "Scrublet", "scDblFinder", "DoubletFinder", "scds"]
CONTROLLED_METRICS = [
    "AUROC",
    "overall_AUPRC",
    "homotypic_AUPRC",
    "heterotypic_AUPRC",
    "macro_subtype_AUPRC",
    "homotypic_vs_high_RNA_singlet_AUPRC",
    "high_RNA_singlet_FPR",
    "high_RNA_singlet_FPR_at_matched_50pct_homotypic_recall",
    "high_RNA_singlet_FPR_at_matched_70pct_homotypic_recall",
    "high_RNA_singlet_FPR_at_matched_80pct_homotypic_recall",
    "high_RNA_singlet_FPR_at_fixed_20pct_candidate_budget",
    "high_RNA_singlet_FPR_at_true_doublet_budget",
    "precision_at_K",
    "recall_at_K",
]


def _read_all(root: Path, name: str) -> pd.DataFrame:
    frames = []
    for path in sorted(root.rglob(name)):
        if "tables" in path.parts or "figures" in path.parts:
            continue
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError):
            continue
        frame["source_file"] = str(path.relative_to(root))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _assert_methods(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "method" not in frame:
        return frame
    historical = sorted(set(frame["method"].dropna().astype(str)) - set(ALLOWED_METHODS))
    if historical:
        raise ValueError(f"manuscript input contains non-frozen methods: {', '.join(historical)}")
    return frame.loc[frame["method"].isin(ALLOWED_METHODS)].copy()


def _summary(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["method"])
    available = [metric for metric in metrics if metric in frame]
    grouped = frame.groupby("method", sort=False)[available].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(column) for column in grouped.columns]
    return grouped.reset_index()


def _plot_controlled(summary: pd.DataFrame, output: Path) -> None:
    if summary.empty or "overall_AUPRC_mean" not in summary:
        return
    import matplotlib.pyplot as plt
    apply_manuscript_style()

    ordered = summary.sort_values("overall_AUPRC_mean", ascending=False)
    colors = ["#2B6F6D" if method == "DuoDose" else "#5B5F97" if method == "DuoDose-DL" else "#7A7A7A" for method in ordered["method"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(ordered["method"], ordered["overall_AUPRC_mean"], color=colors)
    ax.set_ylabel("Mean overall AUPRC (threshold-independent)")
    ax.set_title("Historical sensitivity: held-out size co-varied")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "final_controlled_overall_auprc.png", dpi=220)
    plt.close(fig)


def _plot_method_bars(summary: pd.DataFrame, metric: str, ylabel: str, stem: str, output: Path) -> None:
    if summary.empty or metric not in summary:
        return
    import matplotlib.pyplot as plt
    apply_manuscript_style()

    ordered = summary.sort_values(metric, ascending=False)
    colors = ["#2B6F6D" if method == "DuoDose" else "#5B5F97" if method == "DuoDose-DL" else "#7A7A7A" for method in ordered["method"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(ordered["method"], ordered[metric], color=colors)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / f"{stem}.png", dpi=220)
    plt.close(fig)


def _plot_runtime(frame: pd.DataFrame, output: Path, audit: pd.DataFrame | None = None) -> None:
    if frame.empty or not {"n_cells", "method", "total_wall_clock_seconds"}.issubset(frame):
        return
    import matplotlib.pyplot as plt
    apply_manuscript_style()

    summary = frame.groupby(["n_cells", "method"], as_index=False)["total_wall_clock_seconds"].mean()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    plotted = set(audit.loc[audit["plotted"].astype(bool), "method"].astype(str)) if audit is not None else set(summary["method"].astype(str))
    for method in CANONICAL_RUNTIME_METHODS:
        rows = summary.loc[summary["method"].astype(str).eq(method)]
        if method not in plotted or rows.empty:
            continue
        rows = rows.sort_values("n_cells")
        ax.plot(rows["n_cells"], rows["total_wall_clock_seconds"], marker="o", label=method)
    ax.set_xlabel("Cells")
    ax.set_ylabel("Mean wall-clock time (s)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "final_runtime_scaling.png", dpi=220)
    plt.close(fig)


def _plot_sensitivity(frame: pd.DataFrame, output: Path) -> None:
    required = {"semi_real_size_factor", "expected_doublet_rate", "overall_AUPRC"}
    if frame.empty or not required.issubset(frame):
        return
    import matplotlib.pyplot as plt
    apply_manuscript_style()

    summary = frame.groupby(["semi_real_size_factor", "expected_doublet_rate"], as_index=False)["overall_AUPRC"].mean()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for factor, rows in summary.groupby("semi_real_size_factor", sort=True):
        rows = rows.sort_values("expected_doublet_rate")
        ax.plot(rows["expected_doublet_rate"], rows["overall_AUPRC"], marker="o", label=f"size x{factor:g}")
    ax.set_xlabel("Expected doublet rate")
    ax.set_ylabel("Mean overall AUPRC")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "final_parameter_sensitivity.png", dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--protocol", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    protocol = load_final_protocol(args.protocol)
    root = Path(args.results_dir).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else root
    tables = output / "tables"
    figures = output / "figures"
    reports = output / "reports"
    for directory in (tables, figures, reports):
        directory.mkdir(parents=True, exist_ok=True)

    internal = _assert_methods(_read_all(root, "controlled_metrics.csv"))
    external = _assert_methods(_read_all(root, "external_controlled_metrics.csv"))
    controlled = pd.concat([internal, external], ignore_index=True) if not internal.empty or not external.empty else pd.DataFrame()
    controlled.to_csv(tables / "final_controlled_comparison_by_run.csv", index=False)
    controlled_summary = _summary(controlled.loc[controlled.get("status", "success").astype(str).str.lower().eq("success")] if not controlled.empty else controlled, CONTROLLED_METRICS)
    controlled_summary.to_csv(tables / "final_controlled_comparison_summary.csv", index=False)

    # Preserve the complete operating-point analysis as a first-class table.
    # The manuscript-facing primary row is matched 50% homotypic recall;
    # fixed 20%, matched 70/80%, and the historical true-doublet budget are
    # retained transparently as supplementary sensitivity analyses.
    internal_operating = _assert_methods(_read_all(root, "controlled_high_RNA_operating_points.csv"))
    external_operating = _assert_methods(_read_all(root, "external_high_RNA_operating_points.csv"))
    operating = (
        pd.concat([internal_operating, external_operating], ignore_index=True, sort=False)
        if not internal_operating.empty or not external_operating.empty
        else pd.DataFrame()
    )
    operating.to_csv(tables / "final_high_RNA_operating_points_by_run.csv", index=False)
    if not operating.empty and {"method", "operating_point_name", "high_RNA_singlet_FPR"}.issubset(operating):
        valid_operating = operating.loc[operating.get("status", "SUCCESS").astype(str).str.upper().eq("SUCCESS")].copy()
        operating_summary = (
            valid_operating.groupby(["operating_point_name", "operating_point_type", "method"], as_index=False)
            .agg(
                high_RNA_singlet_FPR_mean=("high_RNA_singlet_FPR", "mean"),
                high_RNA_singlet_FPR_std=("high_RNA_singlet_FPR", "std"),
                actual_candidate_fraction_mean=("actual_candidate_fraction", "mean"),
                homotypic_recall_mean=("homotypic_recall", "mean"),
                precision_mean=("precision", "mean"),
                n_runs=("high_RNA_singlet_FPR", "count"),
            )
        )
    else:
        operating_summary = pd.DataFrame(
            columns=[
                "operating_point_name", "operating_point_type", "method",
                "high_RNA_singlet_FPR_mean", "high_RNA_singlet_FPR_std",
                "actual_candidate_fraction_mean", "homotypic_recall_mean",
                "precision_mean", "n_runs",
            ]
        )
    operating_summary.to_csv(tables / "final_high_RNA_operating_points_summary.csv", index=False)

    application_status = _read_all(root / "real_application", "real_application_method_status.csv")
    application_candidates = _read_all(root / "real_application", "real_application_candidate_summary.csv")
    application_status.to_csv(tables / "final_real_application_method_status.csv", index=False)
    application_candidates.to_csv(tables / "final_real_application_candidate_summary.csv", index=False)

    for name in ("runtime_scaling_summary.csv", "parameter_sensitivity_summary.csv", "domain_audit_all_datasets_summary.csv", "domain_audit_all_datasets_failures.csv"):
        matches = sorted(root.rglob(name))
        if matches:
            shutil.copy2(matches[0], tables / name)

    runtime_raw = _read_all(root, "runtime_scaling_by_run.csv")
    runtime_manifest_matches = sorted((root / "runtime").glob("run_manifest.json"))
    runtime_manifest = json.loads(runtime_manifest_matches[0].read_text(encoding="utf-8")) if runtime_manifest_matches else {}
    requested_runtime_methods = runtime_manifest.get("methods", [])
    if not requested_runtime_methods:
        command = str(runtime_manifest.get("command", ""))
        marker = "--methods "
        requested_runtime_methods = command.split(marker, 1)[1].split()[0].split(",") if marker in command else []
    counts = {int(value) for value in runtime_manifest.get("cell_counts", []) if str(value).isdigit()}
    expected_per_method = len(counts) * int(protocol["runtime"]["repetitions"]) if counts else None
    runtime_audit = build_runtime_method_completeness_audit(
        runtime_raw,
        expected_methods=protocol["runtime"]["methods"],
        requested_methods=requested_runtime_methods,
        expected_successful_rows=expected_per_method,
    )
    runtime_audit.to_csv(root / "runtime" / "runtime_method_completeness_audit.csv", index=False)
    runtime_audit.to_csv(tables / "runtime_method_completeness_audit.csv", index=False)
    method_status = []
    for workflow, frame in (("controlled_internal", internal), ("controlled_external", external), ("real_data_application", application_status)):
        if not frame.empty:
            columns = [column for column in ("dataset", "seed", "method", "status", "message", "source_file") if column in frame]
            part = frame[columns].copy()
            part.insert(0, "workflow", workflow)
            method_status.append(part)
    (pd.concat(method_status, ignore_index=True) if method_status else pd.DataFrame(columns=["workflow", "dataset", "seed", "method", "status", "message", "source_file"])).to_csv(tables / "final_method_status_ledger.csv", index=False)

    analysis_status = []
    for workflow, filename in (
        ("controlled", "controlled_run_status.csv"),
        ("domain_audit", "domain_audit_all_datasets_run_status.csv"),
        ("domain_batch", "domain_audit_batch_run_status.csv"),
        ("runtime", "runtime_scaling_by_run.csv"),
        ("validation_suite", "validation_suite_checks.csv"),
    ):
        frame = _read_all(root, filename)
        if not frame.empty:
            frame.insert(0, "workflow", workflow)
            analysis_status.append(frame)
    (pd.concat(analysis_status, ignore_index=True, sort=False) if analysis_status else pd.DataFrame(columns=["workflow", "status", "message"])).to_csv(tables / "final_analysis_status_ledger.csv", index=False)

    _plot_controlled(controlled_summary, figures)
    representative = str(protocol["datasets"]["representative_dataset"])
    application_run = root / "real_application" / representative / f"seed_{int(protocol['seeds']['real_application'][0])}"
    source = application_run / "real_application_cross_method_umap.png"
    if source.is_file():
        shutil.copy2(source, figures / "final_real_application_cross_method_umap.png")
    _plot_runtime(runtime_raw, figures, runtime_audit)
    _plot_sensitivity(_read_all(root, "parameter_sensitivity_by_run.csv"), figures)
    for stem in ("domain_audit_all_datasets_auroc_comparison", "domain_audit_all_datasets_matched_direction_adjusted"):
        matches = sorted((root / "domain_audit").glob(f"{stem}.png"))
        if matches:
            shutil.copy2(matches[0], figures / matches[0].name)

    report_lines = [
        "# Final DuoDose clean-rerun report",
        "",
        "This report is generated only from machine-readable outputs under the supplied clean results directory.",
        "",
        "## Method scope",
        "",
        "The internal comparison is limited to DuoDose (calibrated RF) and DuoDose-DL. External methods are Scrublet, scDblFinder, DoubletFinder, and scds. Failed methods remain in status tables and are not silently omitted.",
        "The main real-data application exposes only calibrated-RF DuoDose; DuoDose-DL remains confined to controlled ablation analyses.",
        "The calibrated-RF SafeFeature contract includes the row-local library_complexity_balance feature.",
        "The primary high-RNA false-positive rate is matched at 50% homotypic recall; fixed top-20%, matched 70%/80%, and historical true-doublet-budget values are supplementary.",
        "",
        "## Claim boundaries",
        "",
        "The real-data application is qualitative and descriptive. Experimental labels are an overlay only and are not used for fitting, reference selection, thresholds, clustering, PCA, UMAP, or external methods. Model-inferred -like classes are not experimental subtype ground truth.",
        "",
        f"Controlled successful rows: {int(controlled.get('status', pd.Series(dtype=str)).astype(str).str.lower().eq('success').sum()) if not controlled.empty else 0}",
        f"Real-application successful method rows: {int(application_status.get('status', pd.Series(dtype=str)).astype(str).str.lower().eq('success').sum()) if not application_status.empty else 0}",
        f"Runtime methods completed and plotted: {', '.join(runtime_audit.loc[runtime_audit['plotted'].astype(bool), 'method'].astype(str))}",
        f"Runtime methods explicitly not plotted: {', '.join(runtime_audit.loc[~runtime_audit['plotted'].astype(bool), 'method'].astype(str))}",
    ]
    (reports / "final_validation_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    pdf_outputs = (
        sorted(figures.glob("*.pdf"))
        + sorted((root / "validation_suite" / "figures").glob("*.pdf"))
        + sorted((root / "domain_audit").rglob("*.pdf"))
        + sorted((root / "real_application").rglob("*.pdf"))
    )
    png_outputs = (
        sorted(figures.glob("*.png"))
        + sorted((root / "validation_suite" / "figures").glob("*.png"))
        + sorted((root / "domain_audit").rglob("*.png"))
        + sorted((root / "real_application").rglob("*.png"))
    )
    style_audit = audit_figure_style_contract(ROOT, pdf_outputs=pdf_outputs, png_outputs=png_outputs)
    style_audit.to_csv(root / "figure_style_contract_audit.csv", index=False)
    table_files = sorted(path.name for path in tables.iterdir() if path.is_file() and path.name != "final_table_manifest.json")
    figure_files = sorted(path.name for path in figures.iterdir() if path.is_file() and path.name != "final_figure_manifest.json")
    (tables / "final_table_manifest.json").write_text(
        json.dumps({"schema_version": 2, "real_application_role": "qualitative_descriptive_application", "primary_high_RNA_FPR": "matched 50% homotypic recall", "supplementary_high_RNA_FPR": ["fixed top 20% candidate budget", "matched 70% homotypic recall", "matched 80% homotypic recall", "historical true-doublet budget"], "contract_audits": ["runtime_method_completeness_audit.csv", "../figure_style_contract_audit.csv"], "files": table_files}, indent=2),
        encoding="utf-8",
    )
    (figures / "final_figure_manifest.json").write_text(
        json.dumps({"schema_version": 2, "real_application_primary_figure": "final_real_application_cross_method_umap", "layout": "3x3", "internal_methods": ["DuoDose"], "font_contract": "Arial-first; final manuscript aggregation is PNG-only", "primary_high_RNA_FPR": "matched 50% homotypic recall", "runtime_method_order": list(CANONICAL_RUNTIME_METHODS), "runtime_plotted_methods": runtime_audit.loc[runtime_audit["plotted"].astype(bool), "method"].astype(str).tolist(), "files": figure_files}, indent=2),
        encoding="utf-8",
    )
    report_files = sorted(path.name for path in reports.iterdir() if path.is_file() and path.name != "final_report_manifest.json")
    (reports / "final_report_manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": report_files}, indent=2),
        encoding="utf-8",
    )
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "final_artifact_manifest.json":
            files.append(str(path.relative_to(output)))
    (output / "final_artifact_manifest.json").write_text(json.dumps({"schema_version": 1, "allowed_methods": ALLOWED_METHODS, "files": files}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
