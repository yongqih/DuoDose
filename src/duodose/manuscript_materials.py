"""Aggregate frozen clean results into manuscript-facing figures and tables."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .plotting_style import (
    MANUSCRIPT_COLORS,
    apply_manuscript_style,
    finish_manuscript_axes,
    label_panels,
    save_manuscript_png,
)
from .safe_feature_manifest import build_safe_feature_manifest


MAIN_METHODS = ["DuoDose", "Scrublet", "scDblFinder", "DoubletFinder", "scds"]
COMPLETE_METHODS = ["DuoDose", "DuoDose-DL", "Scrublet", "scDblFinder", "DoubletFinder", "scds"]
CATEGORICAL_TICK_ROTATION = 45
CORE_METRICS = [
    "overall_AUPRC",
    "homotypic_AUPRC",
    "heterotypic_AUPRC",
    "macro_subtype_AUPRC",
]
TABLE1_METRICS = CORE_METRICS + [
    "homotypic_vs_high_RNA_singlet_AUPRC",
    "high_RNA_singlet_FPR",
]
METRIC_LABELS = {
    "overall_AUPRC": "Overall AUPRC",
    "homotypic_AUPRC": "Homotypic AUPRC",
    "heterotypic_AUPRC": "Heterotypic AUPRC",
    "macro_subtype_AUPRC": "Macro subtype AUPRC",
    "homotypic_vs_high_RNA_singlet_AUPRC": "Homotypic vs high-RNA AUPRC",
    "high_RNA_singlet_FPR": "High-RNA FPR at 50% homotypic recall",
    "precision_at_K": "Precision at K",
    "recall_at_K": "Recall at K",
}
HIGH_RNA_FPR_DEFINITION = (
    "Fraction of held-out high-RNA singlets selected by the smallest deterministic "
    "score-ranked candidate set that recovers at least 50% of held-out homotypic "
    "doublets; lower is better. All methods are compared at the same homotypic recall."
)
INTERNAL_ONLY_COLUMN_TOKENS = ("cache", "output_path", "source_file", "summary_path")
MEMORY_COLUMN_TOKENS = ("ram", "memory")


@dataclass
class BuildState:
    results_dir: Path
    output_dir: Path
    repository_root: Path
    generated: list[Path] = field(default_factory=list)
    sources: set[Path] = field(default_factory=set)
    panel_rows: list[dict[str, Any]] = field(default_factory=list)
    table_rows: list[dict[str, Any]] = field(default_factory=list)
    omitted: list[dict[str, str]] = field(default_factory=list)
    resolved_font: str = ""

    def source(self, relative: str, *, required: bool = True) -> Path | None:
        path = (self.results_dir / relative).resolve()
        try:
            path.relative_to(self.results_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"scientific source escapes clean results directory: {path}") from exc
        if not path.is_file() or path.stat().st_size == 0:
            if required:
                raise FileNotFoundError(f"required clean source is missing or empty: {path}")
            return None
        self.sources.add(path)
        return path

    def register(self, path: Path) -> Path:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"generated artifact is missing or empty: {path}")
        self.generated.append(path.resolve())
        return path

    def relative_output(self, path: Path) -> str:
        return path.resolve().relative_to(self.output_dir.resolve()).as_posix()

    def relative_source(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.results_dir.resolve()).as_posix()
        except ValueError:
            return resolved.relative_to(self.repository_root.resolve()).as_posix()


def _read_csv(state: BuildState, relative: str, *, required: bool = True) -> pd.DataFrame:
    path = state.source(relative, required=required)
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path)


def _write_csv(state: BuildState, frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return state.register(path)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _method_colors(methods: Iterable[str]) -> list[str]:
    return [MANUSCRIPT_COLORS.get(str(method), "#7F8C8D") for method in methods]


def _set_categorical_xticks(ax: Any, positions: Iterable[float], labels: Iterable[Any]) -> None:
    """Apply the manuscript contract for every rotated categorical x tick."""

    ax.set_xticks(list(positions), [str(label) for label in labels])
    for tick in ax.get_xticklabels():
        tick.set_rotation(CATEGORICAL_TICK_ROTATION)
        tick.set_ha("right")
        tick.set_rotation_mode("anchor")


def _assert_row_axes_aligned(rows: Iterable[Iterable[Any]], *, tolerance: float = 1e-9) -> None:
    """Fail generation when panel plotting areas in a row do not align."""

    for row_index, row in enumerate(rows, start=1):
        axes = list(row)
        if len(axes) < 2:
            continue
        axes[0].figure.canvas.draw()
        positions = [ax.get_position() for ax in axes]
        bottoms = [position.y0 for position in positions]
        tops = [position.y1 for position in positions]
        if max(bottoms) - min(bottoms) > tolerance or max(tops) - min(tops) > tolerance:
            bounds = [(round(position.y0, 12), round(position.y1, 12)) for position in positions]
            raise RuntimeError(f"plotting areas are not vertically aligned in row {row_index}: {bounds}")


def _boxplot_metric(ax: Any, frame: pd.DataFrame, metric: str, methods: list[str], title: str) -> None:
    groups = [pd.to_numeric(frame.loc[frame["method"] == method, metric], errors="coerce").dropna().to_numpy() for method in methods]
    artists = ax.boxplot(groups, positions=np.arange(len(methods)), widths=0.58, patch_artist=True, showfliers=False)
    for patch, color in zip(artists["boxes"], _method_colors(methods)):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor("#333333")
    for median in artists["medians"]:
        median.set_color("white")
        median.set_linewidth(1.4)
    for index, (values, color) in enumerate(zip(groups, _method_colors(methods))):
        if len(values):
            offsets = np.linspace(-0.16, 0.16, len(values))
            ax.scatter(index + offsets, values, s=6, color=color, alpha=0.32, linewidths=0, zorder=3)
    _set_categorical_xticks(ax, np.arange(len(methods)), methods)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    finish_manuscript_axes(ax)


def _bar_values(ax: Any, labels: list[str], values: np.ndarray, *, title: str, ylabel: str, colors: list[str] | None = None) -> None:
    colors = colors or ["#0B6E75"] * len(labels)
    ax.bar(np.arange(len(labels)), values, color=colors, width=0.72, zorder=2)
    _set_categorical_xticks(ax, np.arange(len(labels)), labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    finish_manuscript_axes(ax)


def _drop_columns_with_tokens(frame: pd.DataFrame, tokens: Iterable[str]) -> pd.DataFrame:
    lowered = tuple(str(token).lower() for token in tokens)
    return frame.loc[:, [column for column in frame.columns if not any(token in str(column).lower() for token in lowered)]].copy()


def _runtime_manuscript_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove incomplete memory measurements from manuscript-facing runtime data."""

    return _drop_columns_with_tokens(frame, MEMORY_COLUMN_TOKENS)


def _annotate_heatmap(ax: Any, image: Any, values: np.ndarray, *, decimals: int = 3) -> None:
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            if not np.isfinite(value):
                continue
            red, green, blue, _ = image.cmap(image.norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(column, row, f"{value:.{decimals}f}", ha="center", va="center", fontsize=7.5, color="white" if luminance < 0.52 else "black")


def _real_application_overlap(state: BuildState) -> tuple[pd.DataFrame, list[Path]]:
    """Summarize clean exact-top-K overlap between DuoDose and external methods."""

    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    external_methods = MAIN_METHODS[1:]
    for path in sorted((state.results_dir / "real_application").glob("*/seed_0/real_application_method_scores.csv.gz")):
        state.sources.add(path.resolve())
        sources.append(path.resolve())
        dataset = path.parts[-3]
        scores = pd.read_csv(path)
        duodose_column = "DuoDose_common_display_top_k"
        if duodose_column not in scores:
            raise ValueError(f"clean real-application scores lack {duodose_column}: {path}")
        duodose = scores[duodose_column].astype(bool).to_numpy()
        external_flags: list[np.ndarray] = []
        for method in external_methods:
            column = f"{method}_common_display_top_k"
            if column not in scores:
                raise ValueError(f"clean real-application scores lack {column}: {path}")
            external = scores[column].astype(bool).to_numpy()
            external_flags.append(external)
            intersection = int(np.sum(duodose & external))
            union = int(np.sum(duodose | external))
            rows.append(
                {
                    "dataset": dataset,
                    "external_method": method,
                    "common_display_top_k": int(np.sum(duodose)),
                    "overlap_count": intersection,
                    "union_count": union,
                    "jaccard": float(intersection / union) if union else np.nan,
                    "duodose_only_count": int(np.sum(duodose & ~external)),
                    "external_only_count": int(np.sum(~duodose & external)),
                }
            )
        consensus = np.sum(np.column_stack(external_flags), axis=1) >= 2
        intersection = int(np.sum(duodose & consensus))
        union = int(np.sum(duodose | consensus))
        rows.append(
            {
                "dataset": dataset,
                "external_method": "External consensus (>=2 methods)",
                "common_display_top_k": int(np.sum(duodose)),
                "overlap_count": intersection,
                "union_count": union,
                "jaccard": float(intersection / union) if union else np.nan,
                "duodose_only_count": int(np.sum(duodose & ~consensus)),
                "external_only_count": int(np.sum(~duodose & consensus)),
            }
        )
    if not rows:
        raise FileNotFoundError("no clean real-application method-score files were found")
    return pd.DataFrame(rows), sources


def _panel_manifest(
    state: BuildState,
    *,
    figure: str,
    panel: str,
    sources: Iterable[Path],
    metrics: str,
    action: str,
    output: Path,
) -> None:
    state.panel_rows.append(
        {
            "figure": figure,
            "panel": panel,
            "source_files": ";".join(state.relative_source(path) for path in sources),
            "source_tables_or_metrics": metrics,
            "panel_action": action,
            "final_output": state.relative_output(output),
        }
    )


def _table_manifest(state: BuildState, *, table: str, sources: Iterable[Path], output: Path, notes: str) -> None:
    state.table_rows.append(
        {
            "table": table,
            "source_files": ";".join(state.relative_source(path) for path in sources),
            "final_output": state.relative_output(output),
            "notes": notes,
        }
    )


def _successful_controlled(state: BuildState) -> tuple[pd.DataFrame, Path]:
    source = state.source("tables/final_controlled_comparison_by_run.csv")
    assert source is not None
    frame = pd.read_csv(source)
    frame = frame.loc[frame["status"].astype(str).str.lower().eq("success")].copy()
    unknown = sorted(set(frame["method"].dropna().astype(str)) - set(COMPLETE_METHODS))
    if unknown:
        raise ValueError(f"controlled source contains methods outside the frozen manuscript contract: {unknown}")
    required_metrics = [
        "AUROC",
        *CORE_METRICS,
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
    missing = [metric for metric in required_metrics if metric not in frame]
    if missing:
        raise ValueError(
            "controlled source predates the revised metric contract; missing columns: "
            + ", ".join(missing)
        )
    for metric in required_metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    primary = frame["high_RNA_singlet_FPR"]
    explicit = frame["high_RNA_singlet_FPR_at_matched_50pct_homotypic_recall"]
    mismatch = ~(np.isclose(primary, explicit, rtol=0.0, atol=1e-12, equal_nan=True))
    if bool(np.asarray(mismatch).any()):
        raise ValueError(
            "high_RNA_singlet_FPR must be the matched-50%-homotypic-recall value "
            "before manuscript figures are generated"
        )
    return frame, source


def _create_table1(state: BuildState, controlled: pd.DataFrame, source: Path) -> dict[str, Any]:
    grouped = controlled.loc[controlled["method"].isin(MAIN_METHODS)].groupby("method", sort=False)[TABLE1_METRICS].agg(["mean", "std", "count"])
    numeric_rows: list[dict[str, Any]] = []
    display_rows: list[dict[str, Any]] = []
    for method in MAIN_METHODS:
        if method not in grouped.index:
            raise ValueError(f"Table 1 is missing required method {method}")
        numeric: dict[str, Any] = {"method": method}
        display: dict[str, Any] = {"method": method}
        for metric in TABLE1_METRICS:
            mean = float(grouped.loc[method, (metric, "mean")])
            std = float(grouped.loc[method, (metric, "std")])
            count = int(grouped.loc[method, (metric, "count")])
            numeric[f"{metric}_mean"] = mean
            numeric[f"{metric}_sd"] = std
            numeric[f"{metric}_n"] = count
            display[METRIC_LABELS[metric]] = f"{mean:.3f} ± {std:.3f}"
        numeric_rows.append(numeric)
        display_rows.append(display)
    numeric_frame = pd.DataFrame(numeric_rows)
    display_frame = pd.DataFrame(display_rows)
    main_dir = state.output_dir / "main_tables"
    display_path = _write_csv(state, display_frame, main_dir / "Table1_core_benchmark_summary.csv")
    numeric_path = _write_csv(state, numeric_frame, state.output_dir / "table_components" / "Table1_core_benchmark_summary_numeric.csv")
    _table_manifest(state, table="Table 1", sources=[source], output=display_path, notes="Mean ± SD; DuoDose-DL excluded from main-text table.")
    return {
        "Summary": display_frame,
        "Numeric": numeric_frame,
        "Notes": pd.DataFrame(
            {
                "note": [
                    "Values are mean ± SD across completed dataset-seed runs.",
                    "Higher is better for AUPRC metrics.",
                    f"High-RNA singlet FPR: {HIGH_RNA_FPR_DEFINITION}",
                    "DuoDose-DL is a supplementary ablation and is intentionally excluded.",
                ]
            }
        ),
        "output": main_dir / "Table1_core_benchmark_summary.xlsx",
        "numeric_path": numeric_path,
    }


def _dataset_metadata(state: BuildState) -> tuple[pd.DataFrame, list[Path]]:
    real_summary_source = state.source("tables/final_real_application_candidate_summary.csv")
    assert real_summary_source is not None
    real = pd.read_csv(real_summary_source).set_index("dataset")
    rows: list[dict[str, Any]] = []
    sources = [real_summary_source]
    controlled_root = state.results_dir / "controlled"
    for report_path in sorted(controlled_root.glob("*/seed_0/construction_report.json"), key=lambda path: path.parts[-3].lower()):
        state.sources.add(report_path.resolve())
        sources.append(report_path.resolve())
        report = json.loads(report_path.read_text(encoding="utf-8"))
        dataset = str(report["dataset"])
        parent_map_path = report_path.parent / "semireal_parent_map.csv.gz"
        parent_overlap = np.nan
        reference_overlap = np.nan
        if parent_map_path.is_file():
            state.sources.add(parent_map_path.resolve())
            sources.append(parent_map_path.resolve())
            parent = pd.read_csv(parent_map_path)
            split_parents: dict[str, set[str]] = {}
            for split, group in parent.groupby("split"):
                split_parents[str(split)] = set(group["parent_1_id"].astype(str)) | set(group["parent_2_id"].astype(str))
            split_names = sorted(split_parents)
            parent_overlap = sum(len(split_parents[a] & split_parents[b]) for i, a in enumerate(split_names) for b in split_names[i + 1 :])
            reference_overlap = int(parent.get("parent_1_in_reference", False).astype(bool).sum() + parent.get("parent_2_in_reference", False).astype(bool).sum())
        real_row = real.loc[dataset] if dataset in real.index else pd.Series(dtype=object)
        rows.append(
            {
                "dataset": dataset,
                "n_cells": int(real_row.get("n_cells", report.get("n_observed_background_cells_available", report.get("n_real_labeled_singlets_available", 0)))),
                "n_genes": int(report.get("n_genes", 0)),
                "labeled_doublet_count": real_row.get("labeled_doublet_count", np.nan),
                "labeled_doublet_fraction": real_row.get("labeled_doublet_fraction", np.nan),
                "n_fit_cells": report.get("n_fit_cells", np.nan),
                "n_validation_cells": report.get("n_validation_cells", np.nan),
                "n_test_cells": report.get("n_test_cells", np.nan),
                "n_fit_homotypic_doublets": report.get("n_fit_homotypic_doublets", np.nan),
                "n_fit_heterotypic_doublets": report.get("n_fit_heterotypic_doublets", np.nan),
                "n_validation_homotypic_doublets": report.get("n_validation_homotypic_doublets", np.nan),
                "n_validation_heterotypic_doublets": report.get("n_validation_heterotypic_doublets", np.nan),
                "n_test_homotypic_doublets": report.get("n_test_homotypic_doublets", np.nan),
                "n_test_heterotypic_doublets": report.get("n_test_heterotypic_doublets", np.nan),
                "parent_overlap_across_splits": parent_overlap,
                "reference_parent_overlap_count": reference_overlap,
                "construction_variant": report.get("construction_variant", ""),
                "safe_feature_mode": "fitted_reference",
                "construction_background_population": "label-blinded observed cells; may include experimentally labeled doublets",
            }
        )
    if not rows:
        raise FileNotFoundError("no clean seed-0 construction reports were found")
    return pd.DataFrame(rows), sources


def _figure2(state: BuildState, controlled: pd.DataFrame, source: Path) -> Path:
    import matplotlib.pyplot as plt

    main = controlled.loc[controlled["method"].isin(MAIN_METHODS)].copy()
    component = _write_csv(state, main, state.output_dir / "figure_components" / "Figure2_controlled_benchmark_values.csv")
    fig = plt.figure(figsize=(12.4, 7.1))
    grid = fig.add_gridspec(2, 12, hspace=0.58, wspace=0.9)
    axes = [fig.add_subplot(grid[0, start : start + 4]) for start in (0, 4, 8)]
    axes.append(fig.add_subplot(grid[1, 0:4]))
    axes.append(fig.add_subplot(grid[1, 5:12]))
    for ax, metric, title in zip(axes[:4], CORE_METRICS, ["Overall detection", "Homotypic detection", "Heterotypic detection", "Subtype-balanced performance"]):
        _boxplot_metric(ax, main, metric, MAIN_METHODS, title)
    pivot = main.pivot_table(index=["dataset", "seed"], columns="method", values="overall_AUPRC", aggfunc="first")
    complete_external = [method for method in MAIN_METHODS[1:] if method in pivot]
    delta = pivot["DuoDose"] - pivot[complete_external].max(axis=1)
    dataset_delta = delta.groupby(level="dataset").mean().sort_values()
    colors = ["#0B6E75" if value >= 0 else "#B65C5C" for value in dataset_delta]
    axes[4].barh(dataset_delta.index, dataset_delta.to_numpy(), color=colors, zorder=2)
    axes[4].axvline(0, color="#444444", linewidth=0.9)
    delta_min = float(np.nanmin(dataset_delta.to_numpy()))
    delta_max = float(np.nanmax(dataset_delta.to_numpy()))
    axes[4].set_xlim(delta_min - 0.01, delta_max + 0.01)
    axes[4].set_xlabel("ΔAUPRC vs best external")
    axes[4].set_title("Mean per-dataset advantage")
    axes[4].tick_params(axis="y", pad=1)
    finish_manuscript_axes(axes[4], grid_axis="x")
    label_panels(axes)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.12, top=0.965)
    _assert_row_axes_aligned([axes[:3], axes[3:]])
    output = state.output_dir / "main_figures" / "Figure2_controlled_benchmark.png"
    save_manuscript_png(fig, output)
    state.register(output)
    for panel, metric in zip("ABCD", CORE_METRICS):
        _panel_manifest(state, figure="Figure 2", panel=panel, sources=[source], metrics=metric, action="regenerated from clean numeric data", output=output)
    _panel_manifest(state, figure="Figure 2", panel="E", sources=[source], metrics="overall_AUPRC; DuoDose minus best external baseline by dataset/seed", action="regenerated from clean numeric data", output=output)
    return output


def _figure3(state: BuildState, controlled: pd.DataFrame, source: Path) -> Path:
    import matplotlib.pyplot as plt

    main = controlled.loc[controlled["method"].isin(MAIN_METHODS)].copy()
    _write_csv(state, main, state.output_dir / "figure_components" / "Figure3_homotypic_highRNA_values.csv")
    metrics = ["homotypic_vs_high_RNA_singlet_AUPRC", "high_RNA_singlet_FPR", "precision_at_K"]
    titles = [
        "Homotypic vs high-RNA separation",
        "High-RNA FPR at 50% homotypic recall",
        "Precision at common K",
    ]
    fig, axes_array = plt.subplots(2, 2, figsize=(10.2, 7.1))
    axes = list(axes_array.flat)
    for ax, metric, title in zip(axes[:3], metrics, titles):
        _boxplot_metric(ax, main, metric, MAIN_METHODS, title)
        if metric == "high_RNA_singlet_FPR":
            ax.text(0.98, 0.96, "Lower is better", transform=ax.transAxes, ha="right", va="top", fontsize=8)
    pivot = main.pivot_table(index=["dataset", "seed"], columns="method", values="homotypic_vs_high_RNA_singlet_AUPRC", aggfunc="first")
    external = [method for method in MAIN_METHODS[1:] if method in pivot]
    paired_delta = pivot["DuoDose"] - pivot[external].max(axis=1)
    dataset_delta = paired_delta.groupby(level="dataset").mean().sort_values()
    axes[3].barh(dataset_delta.index, dataset_delta.to_numpy(), color=["#0B6E75" if value >= 0 else "#B65C5C" for value in dataset_delta], zorder=2)
    axes[3].axvline(0, color="#444444", linewidth=0.9)
    axes[3].set_xlabel("Mean paired delta AUPRC")
    axes[3].set_title("DuoDose advantage vs best external")
    finish_manuscript_axes(axes[3], grid_axis="x")
    delta_component = pd.DataFrame({"dataset": dataset_delta.index, "duodose_minus_best_external_homotypic_vs_high_RNA_AUPRC": dataset_delta.to_numpy()})
    _write_csv(state, delta_component, state.output_dir / "figure_components" / "Figure3_dataset_paired_advantage.csv")
    label_panels(axes)
    fig.subplots_adjust(hspace=0.58, wspace=0.32, top=0.965)
    _assert_row_axes_aligned([axes[:2], axes[2:]])
    output = state.output_dir / "main_figures" / "Figure3_homotypic_vs_highRNA.png"
    save_manuscript_png(fig, output)
    state.register(output)
    for panel, metric in zip("ABC", metrics):
        _panel_manifest(state, figure="Figure 3", panel=panel, sources=[source], metrics=metric, action="regenerated from clean numeric data", output=output)
    _panel_manifest(state, figure="Figure 3", panel="D", sources=[source], metrics="paired per-dataset DuoDose minus best external homotypic_vs_high_RNA_singlet_AUPRC", action="regenerated from clean numeric data", output=output)
    return output


def _figure4(state: BuildState) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    source = state.source("real_application/cline-ch/seed_0/real_application_cross_method_umap.png")
    assert source is not None
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    font_path = Path("C:/Windows/Fonts/arialbd.ttf")
    font = ImageFont.truetype(str(font_path), 82) if font_path.is_file() else ImageFont.load_default()
    width, height = image.size
    for index, label in enumerate("ABCDEFGHI"):
        row, column = divmod(index, 3)
        x = int(column * width / 3 + 20)
        y = int(row * height / 3 + 16)
        draw.text((x, y), label, font=font, fill="#202124", stroke_width=5, stroke_fill="white")
    output = state.output_dir / "main_figures" / "Figure4_real_application_cline_ch.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True, dpi=(300, 300))
    state.register(output)
    metrics = ["clusters", "experimental singlet/doublet labels", "Scrublet score", "DoubletFinder score", "scDblFinder score", "scds score", "DuoDose overall doublet probability", "DuoDose subtype evidence", "DuoDose common top-K candidate classes"]
    for panel, metric in zip("ABCDEFGHI", metrics):
        _panel_manifest(state, figure="Figure 4", panel=panel, sources=[source], metrics=metric, action="reused finalized clean figure; standardized panel labels added", output=output)
    return output


def _permutation_panel(ax: Any, subtype: pd.DataFrame, full: pd.DataFrame, *, show_legend: bool = True) -> None:
    rows = []
    for control, frame, preferred in [("Macro subtype AUPRC", subtype, "macro_subtype_AUPRC"), ("Overall AUPRC", full, "overall_AUPRC")]:
        if frame.empty or "metric" not in frame:
            continue
        selected = frame.loc[frame["metric"].eq(preferred)]
        if selected.empty:
            selected = frame.head(1)
        if selected.empty:
            continue
        row = selected.iloc[0]
        rows.append((control, _safe_float(row["observed_value"]), _safe_float(row["null_mean"]), _safe_float(row["null_025_quantile"]), _safe_float(row["null_975_quantile"]), int(row.get("n_permutations", 0)), _safe_float(row.get("empirical_p_value"))))
    x = np.arange(len(rows))
    observed = np.asarray([row[1] for row in rows])
    null = np.asarray([row[2] for row in rows])
    ax.bar(x - 0.18, observed, width=0.36, color="#0B6E75", label="Observed", zorder=2)
    ax.bar(x + 0.18, null, width=0.36, color="#A9B4BC", label="Permutation null mean", zorder=2)
    lower = null - np.asarray([row[3] for row in rows])
    upper = np.asarray([row[4] for row in rows]) - null
    ax.errorbar(x + 0.18, null, yerr=np.vstack([lower, upper]), fmt="none", ecolor="#333333", capsize=3, linewidth=1)
    ax.set_xticks(x, [row[0] for row in rows])
    upper_limit = max(float(np.nanmax(observed)), float(np.nanmax(np.asarray([row[4] for row in rows])))) if rows else 1.0
    for index, row in enumerate(rows):
        p_text = "NA" if not np.isfinite(row[6]) else f"{row[6]:.3g}"
        ax.text(index, upper_limit + 0.035, f"n = {row[5]}\nempirical P = {p_text}", ha="center", va="bottom", fontsize=7.2)
    ax.set_ylim(0, min(1.0, upper_limit + 0.16))
    ax.set_ylabel("AUPRC")
    ax.set_title("Permutation controls")
    if show_legend:
        ax.legend(loc="upper left")
    finish_manuscript_axes(ax)


def _figure5(state: BuildState) -> Path:
    import matplotlib.pyplot as plt

    domain_source = state.source("domain_audit/domain_audit_all_datasets_summary.csv")
    subtype_source = state.source("validation_suite/subtype_permutation_summary.csv")
    full_source = state.source("validation_suite/full_label_permutation_summary.csv")
    runtime_source = state.source("runtime/runtime_scaling_summary.csv")
    sensitivity_source = state.source("sensitivity/parameter_sensitivity_summary.csv")
    assert all(path is not None for path in [domain_source, subtype_source, full_source, runtime_source, sensitivity_source])
    domain = pd.read_csv(domain_source)
    subtype = pd.read_csv(subtype_source)
    full = pd.read_csv(full_source)
    runtime = pd.read_csv(runtime_source)
    sensitivity = pd.read_csv(sensitivity_source)
    fig = plt.figure(figsize=(11.2, 8.1))
    grid = fig.add_gridspec(4, 2, height_ratios=[0.14, 1, 0.18, 1], hspace=0.34, wspace=0.38)
    top_legend_axis = fig.add_subplot(grid[0, 1])
    runtime_legend_axis = fig.add_subplot(grid[2, 0])
    sensitivity_legend_axis = fig.add_subplot(grid[2, 1])
    for legend_axis in (top_legend_axis, runtime_legend_axis, sensitivity_legend_axis):
        legend_axis.axis("off")
    axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[3, 0]), fig.add_subplot(grid[3, 1])]
    matched = domain.loc[domain["analysis"].astype(str).str.contains("matched", case=False, na=False) & ~domain["analysis"].astype(str).str.contains("unmatched", case=False, na=False)].copy()
    matched = matched.drop_duplicates("dataset").sort_values("pooled_oof_auroc")
    axes[0].barh(matched["dataset"], matched["pooled_oof_auroc"], color="#52796F", zorder=2)
    axes[0].axvline(0.5, color="#8B0000", linestyle="--", linewidth=1)
    axes[0].set_xlim(0.45, 1.0)
    axes[0].set_xlabel("Matched domain-classifier AUROC")
    axes[0].set_title("Strict domain audit")
    finish_manuscript_axes(axes[0], grid_axis="x")
    _permutation_panel(axes[1], subtype, full, show_legend=False)
    permutation_handles, permutation_labels = axes[1].get_legend_handles_labels()
    top_legend_axis.legend(
        permutation_handles,
        permutation_labels,
        ncol=2,
        loc="center",
        frameon=False,
        columnspacing=1.2,
    )
    for method in MAIN_METHODS:
        subset = runtime.loc[runtime["method"].eq(method)].sort_values("n_cells")
        if subset.empty:
            continue
        axes[2].errorbar(subset["n_cells"], subset["total_wall_clock_seconds_mean"], yerr=subset["total_wall_clock_seconds_std"], marker="o", capsize=3, label=method, color=MANUSCRIPT_COLORS[method])
    axes[2].set_xlabel("Cells")
    axes[2].set_ylabel("Wall-clock time (s)")
    axes[2].set_title("Runtime scaling (HMEC-orig-MULTI; n = 3)")
    runtime_legend_axis.legend(
        *axes[2].get_legend_handles_labels(),
        ncol=3,
        loc="center",
        frameon=False,
        columnspacing=1.1,
        handlelength=1.8,
    )
    finish_manuscript_axes(axes[2], grid_axis="both")
    ranking_metrics = [
        ("overall_AUPRC", "Overall AUPRC", "#0B6E75"),
        ("homotypic_AUPRC", "Homotypic AUPRC", "#D66900"),
        ("homotypic_vs_high_RNA_singlet_AUPRC", "Homotypic vs high-RNA AUPRC", "#7A5195"),
    ]
    by_size = sensitivity.sort_values(["semi_real_size_factor", "expected_doublet_rate"]).drop_duplicates("semi_real_size_factor")
    for metric, label, color in ranking_metrics:
        axes[3].errorbar(by_size["semi_real_size_factor"], by_size[f"{metric}_mean"], yerr=by_size[f"{metric}_std"], marker="o", capsize=3, label=label, color=color)
    axes[3].set_xlabel("Semi-real size factor")
    axes[3].set_ylabel("AUPRC")
    axes[3].set_title("Ranking sensitivity to semi-real size")
    axes[3].text(0.02, 0.03, "Expected rate changes thresholds, not continuous rankings", transform=axes[3].transAxes, fontsize=7.2)
    finish_manuscript_axes(axes[3], grid_axis="both")
    sensitivity_legend_axis.legend(*axes[3].get_legend_handles_labels(), ncol=1, loc="center", frameon=False, handlelength=1.8)
    label_panels(axes)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.98)
    _assert_row_axes_aligned([axes[:2], axes[2:]])
    output = state.output_dir / "main_figures" / "Figure5_robustness_practicality.png"
    save_manuscript_png(fig, output)
    state.register(output)
    for panel, source, metric in [
        ("A", domain_source, "matched raw-mechanism pooled_oof_auroc"),
        ("B", subtype_source, "Macro subtype AUPRC and Overall AUPRC; observed versus n=100 permutation null mean with 2.5th-97.5th percentile error bars and empirical P values"),
        ("C", runtime_source, "total_wall_clock_seconds mean +/- SD across 3 repetitions on HMEC-orig-MULTI; main-text methods only"),
        ("D", sensitivity_source, "overall, homotypic, and homotypic-vs-high-RNA AUPRC mean +/- SD across 3 seeds by semi-real size factor; expected rate affects thresholds only"),
    ]:
        sources = [source, full_source] if panel == "B" else [source]
        _panel_manifest(state, figure="Figure 5", panel=panel, sources=sources, metrics=metric, action="regenerated from clean numeric data", output=output)
    return output


def _heatmap(ax: Any, pivot: pd.DataFrame, *, title: str, colorbar: bool = False) -> Any:
    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    _set_categorical_xticks(ax, np.arange(len(pivot.columns)), pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_title(title)
    if colorbar:
        return image
    return image


def _supplementary_figures(state: BuildState, controlled: pd.DataFrame, controlled_source: Path, metadata: pd.DataFrame, metadata_sources: list[Path]) -> list[Path]:
    import matplotlib.pyplot as plt

    outputs: list[Path] = []
    supplement_dir = state.output_dir / "supplementary_figures"

    # Figure S1
    _write_csv(state, metadata, state.output_dir / "figure_components" / "FigureS1_dataset_overview_values.csv")
    ordered = metadata.sort_values("n_cells")
    fig = plt.figure(figsize=(13.2, 8.1))
    grid = fig.add_gridspec(4, 3, height_ratios=[0.14, 1, 0.14, 1], hspace=0.34, wspace=0.38)
    top_legend_axis = fig.add_subplot(grid[0, 2])
    bottom_legend_axis = fig.add_subplot(grid[2, 0])
    top_legend_axis.axis("off")
    bottom_legend_axis.axis("off")
    axes = [fig.add_subplot(grid[1, index]) for index in range(3)]
    axes.append(fig.add_subplot(grid[3, 0]))
    axes.append(fig.add_subplot(grid[3, 1:3]))
    axes[0].barh(ordered["dataset"], ordered["n_cells"], color="#52796F")
    axes[0].set_xlabel("Cells"); axes[0].set_title("Dataset size"); finish_manuscript_axes(axes[0], grid_axis="x")
    axes[1].barh(ordered["dataset"], ordered["labeled_doublet_fraction"] * 100, color="#D97706")
    axes[1].set_xlabel("Experimentally labeled doublets (%)"); axes[1].set_title("Real-label prevalence"); finish_manuscript_axes(axes[1], grid_axis="x")
    split_columns = ["n_fit_cells", "n_validation_cells", "n_test_cells"]
    bottom = np.zeros(len(ordered))
    for column, color, label in zip(split_columns, ["#0B6E75", "#77A6B6", "#CBD5E1"], ["Fit", "Validation", "Test"]):
        axes[2].barh(ordered["dataset"], ordered[column], left=bottom, color=color, label=label)
        bottom += pd.to_numeric(ordered[column], errors="coerce").fillna(0).to_numpy()
    axes[2].set_title("Parent-disjoint split sizes"); axes[2].set_xlabel("Cells"); finish_manuscript_axes(axes[2], grid_axis="x")
    top_legend_axis.legend(*axes[2].get_legend_handles_labels(), loc="center", ncol=3, frameon=False, columnspacing=1.0)
    fit_h = ordered["n_fit_homotypic_doublets"].to_numpy(dtype=float)
    fit_e = ordered["n_fit_heterotypic_doublets"].to_numpy(dtype=float)
    axes[3].barh(ordered["dataset"], fit_h, color="#B85C38", label="Homotypic")
    axes[3].barh(ordered["dataset"], fit_e, left=fit_h, color="#2D6A8A", label="Heterotypic")
    axes[3].set_title("Fit doublet composition")
    axes[3].set_xlabel("Constructed doublets")
    bottom_legend_axis.legend(*axes[3].get_legend_handles_labels(), loc="center", ncol=2, frameon=False, columnspacing=1.2)
    finish_manuscript_axes(axes[3], grid_axis="x")
    cross_split_observed = int(pd.to_numeric(ordered["parent_overlap_across_splits"], errors="coerce").fillna(0).sum())
    reference_observed = int(pd.to_numeric(ordered["reference_parent_overlap_count"], errors="coerce").fillna(0).sum())
    audit_rows = [
        ["Cross-split parent overlap", cross_split_observed, 0, "PASS" if cross_split_observed == 0 else "FAIL"],
        ["Parent cells present in fitted reference", reference_observed, 0, "PASS" if reference_observed == 0 else "FAIL"],
    ]
    axes[4].axis("off")
    axes[4].set_title("Parent-disjoint construction audit")
    audit_table = axes[4].table(
        cellText=audit_rows,
        colLabels=["Check", "Observed", "Required", "Status"],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.55, 0.15, 0.15, 0.15],
        loc="center",
    )
    audit_table.auto_set_font_size(False)
    audit_table.set_fontsize(8.5)
    audit_table.scale(1.0, 2.0)
    for column in range(4):
        audit_table[(0, column)].set_facecolor("#0B6E75")
        audit_table[(0, column)].get_text().set_color("white")
        audit_table[(0, column)].get_text().set_fontweight("bold")
    for row in (1, 2):
        audit_table[(row, 3)].set_facecolor("#D9F0E3" if audit_rows[row - 1][3] == "PASS" else "#F8D7DA")
        audit_table[(row, 3)].get_text().set_fontweight("bold")
    axes[4].text(0.5, 0.20, f"Audited across {len(ordered)} completed datasets", transform=axes[4].transAxes, ha="center", va="center", fontsize=8)
    label_panels(axes)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.07, top=0.98)
    _assert_row_axes_aligned([axes[:3], axes[3:]])
    output = supplement_dir / "FigureS1_dataset_overview.png"; save_manuscript_png(fig, output); state.register(output); outputs.append(output)
    for panel, metric in zip("ABCDE", ["n_cells", "labeled_doublet_fraction", "fit/validation/test cell counts", "fit homotypic/heterotypic counts", "cross-split parent overlap and parent-in-reference audit"]):
        _panel_manifest(state, figure="Figure S1", panel=panel, sources=metadata_sources, metrics=metric, action="regenerated from clean numeric data", output=output)

    # Figure S2
    complete = controlled.loc[controlled["method"].isin(COMPLETE_METHODS)].copy()
    fig = plt.figure(figsize=(12.7, 9.0))
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.045], hspace=0.44, wspace=0.34)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    colorbar_axis = fig.add_subplot(grid[:, 2])
    images = []
    for ax, metric in zip(axes, CORE_METRICS):
        pivot = complete.pivot_table(index="dataset", columns="method", values=metric, aggfunc="mean").reindex(columns=COMPLETE_METHODS)
        images.append(_heatmap(ax, pivot, title=METRIC_LABELS[metric]))
    colorbar = fig.colorbar(images[-1], cax=colorbar_axis, label="Mean AUPRC")
    colorbar.ax.tick_params(labelsize=8)
    label_panels(axes)
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.10, top=0.97)
    _assert_row_axes_aligned([axes[:2], axes[2:]])
    output = supplement_dir / "FigureS2_complete_controlled_benchmark.png"; save_manuscript_png(fig, output); state.register(output); outputs.append(output)
    for panel, metric in zip("ABCD", CORE_METRICS):
        _panel_manifest(state, figure="Figure S2", panel=panel, sources=[controlled_source], metrics=metric, action="regenerated from clean numeric data", output=output)

    # Figure S3
    fig, axes_array = plt.subplots(2, 2, figsize=(9.3, 8.4)); axes = list(axes_array.flat)
    for ax, metric in zip(axes, CORE_METRICS):
        pivot = complete.loc[complete["method"].isin(["DuoDose", "DuoDose-DL"])].pivot_table(index="dataset", columns="method", values=metric, aggfunc="mean").dropna()
        ax.scatter(pivot["DuoDose"], pivot["DuoDose-DL"], color="#7A5195", s=32, alpha=0.85)
        limits = [min(pivot.min()) - 0.03, max(pivot.max()) + 0.03]
        ax.plot(limits, limits, linestyle="--", color="#666666", linewidth=1)
        ax.set_xlim(limits); ax.set_ylim(limits)
        ax.set_xlabel("DuoDose RF"); ax.set_ylabel("DuoDose-DL"); ax.set_title(METRIC_LABELS[metric]); finish_manuscript_axes(ax, grid_axis="both")
    label_panels(axes); fig.subplots_adjust(hspace=0.48, wspace=0.32, top=0.97)
    _assert_row_axes_aligned([axes[:2], axes[2:]])
    output = supplement_dir / "FigureS3_RF_vs_DL_ablation.png"; save_manuscript_png(fig, output); state.register(output); outputs.append(output)
    for panel, metric in zip("ABCD", CORE_METRICS): _panel_manifest(state, figure="Figure S3", panel=panel, sources=[controlled_source], metrics=metric, action="regenerated from clean numeric data", output=output)

    # Figure S4
    invariance_names = ["same_cell_feature_invariance.csv", "cell_order_invariance.csv", "chunking_invariance.csv", "transformer_save_load_invariance.csv", "model_save_load_invariance.csv"]
    invariance_frames = []
    invariance_sources = []
    for name in invariance_names:
        path = state.source(f"validation_suite/{name}")
        assert path is not None
        frame = pd.read_csv(path); frame["source"] = name.replace("_invariance.csv", "").replace("_", " "); invariance_frames.append(frame); invariance_sources.append(path)
    invariance = pd.concat(invariance_frames, ignore_index=True)
    parent_source = state.source("validation_suite/parent_disjoint_audit.csv"); subtype_source = state.source("validation_suite/subtype_permutation_summary.csv"); full_source = state.source("validation_suite/full_label_permutation_summary.csv")
    assert parent_source and subtype_source and full_source
    parent = pd.read_csv(parent_source); subtype = pd.read_csv(subtype_source); full = pd.read_csv(full_source)
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(10.5, 8.1))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 0.14, 1], hspace=0.38, wspace=0.34)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])]
    legend_axis = fig.add_subplot(grid[1, :])
    legend_axis.axis("off")
    inv_summary = invariance.groupby("source")["maximum_absolute_difference"].max().fillna(0).sort_values()
    inv_values = inv_summary.to_numpy(dtype=float)
    axes[0].barh(inv_summary.index, inv_values, color="#0B6E75")
    axes[0].axvline(1e-6, color="#8B0000", linestyle="--", linewidth=1, label="Tolerance (1e-6)")
    axes[0].set_xlim(0, max(1.1e-6, float(np.nanmax(inv_values)) * 1.15 if len(inv_values) else 1.1e-6))
    for index, value in enumerate(inv_values):
        axes[0].text(max(value, 0) + 0.02e-6, index, f"{value:.2g}", va="center", fontsize=7.2)
    axes[0].set_xlabel("Maximum absolute difference")
    axes[0].set_title("Invariance and serialization")
    finish_manuscript_axes(axes[0], grid_axis="x")
    grouped_checks = [
        (
            "Cross-split parent overlap",
            [
                "train_validation_parent_overlap",
                "train_test_parent_overlap",
                "validation_test_parent_overlap",
                "n_cross_split_canonical_pair_overlaps",
                "n_cross_split_parent_overlaps",
            ],
        ),
        (
            "Reference leakage",
            [
                "generated_parent_retained_singlet_overlap",
                "reference_parent_overlap",
                "reference_validation_cell_overlap",
                "reference_test_cell_overlap",
                "parents_marked_in_reference",
            ],
        ),
        (
            "Duplicate/pair integrity",
            [
                "n_raw_ordered_duplicate_pairs",
                "n_reversed_order_equivalent_pairs",
                "n_canonical_duplicate_pairs",
                "n_duplicate_parent_map_rows",
                "n_duplicate_generated_cell_ids",
                "n_duplicate_generated_expression_profiles",
            ],
        ),
    ]
    audit_rows = []
    for category, checks in grouped_checks:
        subset = parent.loc[parent["check"].isin(checks)].copy()
        if len(subset) != len(checks):
            missing = sorted(set(checks) - set(subset["check"].astype(str)))
            raise ValueError(f"parent-disjoint audit is missing checks for {category}: {missing}")
        values = pd.to_numeric(subset["value"], errors="coerce").fillna(0)
        passed = int(subset["status"].astype(str).eq("PASS").sum())
        status = "PASS" if passed == len(checks) else "FAIL"
        audit_rows.append([category, f"{passed}/{len(checks)}", int(values.abs().max()), status])
    axes[1].axis("off")
    axes[1].set_title("Parent/reference and pair-integrity audits")
    audit_table = axes[1].table(
        cellText=audit_rows,
        colLabels=["Audit group", "Passed", "Maximum count", "Status"],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.48, 0.16, 0.22, 0.14],
        loc="center",
    )
    audit_table.auto_set_font_size(False)
    audit_table.set_fontsize(7.8)
    audit_table.scale(1.0, 1.75)
    for column in range(4):
        audit_table[(0, column)].set_facecolor("#0B6E75")
        audit_table[(0, column)].get_text().set_color("white")
        audit_table[(0, column)].get_text().set_fontweight("bold")
    for row in range(1, len(audit_rows) + 1):
        audit_table[(row, 3)].set_facecolor("#D9F0E3" if audit_rows[row - 1][3] == "PASS" else "#F8D7DA")
        audit_table[(row, 3)].get_text().set_fontweight("bold")
    _permutation_panel(axes[2], subtype, pd.DataFrame(), show_legend=False)
    axes[2].set_title("Subtype-label permutation")
    axes[2].set_xticks([])
    _permutation_panel(axes[3], pd.DataFrame(), full, show_legend=False)
    axes[3].set_title("Full-label permutation")
    axes[3].set_xticks([])
    legend_axis.legend(
        handles=[Patch(facecolor="#0B6E75", label="Observed"), Patch(facecolor="#A9B4BC", label="Permutation null mean")],
        loc="center",
        ncol=2,
        frameon=False,
        columnspacing=2.0,
    )
    label_panels(axes)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.96)
    _assert_row_axes_aligned([axes[:2], axes[2:]])
    output = supplement_dir / "FigureS4_validation_and_negative_controls.png"; save_manuscript_png(fig, output); state.register(output); outputs.append(output)
    for panel, sources, metric in [("A", invariance_sources, "true maximum_absolute_difference on a linear zero-preserving scale; tolerance=1e-6"), ("B", [parent_source], "cross-split overlap, reference leakage, and duplicate/pair-integrity grouped audit"), ("C", [subtype_source], "n=100 subtype permutation; observed, null mean, 2.5th-97.5th percentiles, empirical P"), ("D", [full_source], "n=100 full-label permutation; observed, null mean, 2.5th-97.5th percentiles, empirical P")]: _panel_manifest(state, figure="Figure S4", panel=panel, sources=sources, metrics=metric, action="regenerated from clean numeric data", output=output)

    # Figure S5
    domain_source = state.source("domain_audit/domain_audit_all_datasets_summary.csv"); assert domain_source
    domain = pd.read_csv(domain_source)
    fig, axes_array = plt.subplots(1, 3, figsize=(13.4, 5.5)); axes = list(axes_array.flat)
    analyses = [("pooled_oof_auroc", "Matched mechanism features"), ("unmatched_mechanism_auroc", "Unmatched mechanism features"), ("technical_covariates_only_auroc", "Technical covariates only")]
    domain_unique = domain.drop_duplicates("dataset").copy()
    shared_order = domain_unique.sort_values("pooled_oof_auroc")["dataset"].astype(str).tolist()
    for ax, (metric_column, title) in zip(axes, analyses):
        subset = domain_unique.set_index("dataset").reindex(shared_order).reset_index()
        ax.barh(subset["dataset"], subset[metric_column], color="#52796F" if metric_column == "pooled_oof_auroc" else "#7A8791")
        ax.axvline(0.5, color="#8B0000", linestyle="--", linewidth=1); ax.set_xlim(0.45, 1); ax.set_title(title); ax.set_xlabel("Pooled out-of-fold AUROC"); finish_manuscript_axes(ax, grid_axis="x")
    label_panels(axes); fig.subplots_adjust(wspace=0.5, top=0.96)
    _assert_row_axes_aligned([axes])
    output = supplement_dir / "FigureS5_full_domain_audit.png"; save_manuscript_png(fig, output); state.register(output); outputs.append(output)
    for panel, analysis in zip("ABC", ["matched raw mechanism", "unmatched raw mechanism", "technical covariates only"]): _panel_manifest(state, figure="Figure S5", panel=panel, sources=[domain_source], metrics=f"{analysis}: pooled_oof_auroc", action="regenerated from clean numeric data", output=output)

    # Figure S6
    real_source = state.source("tables/final_real_application_candidate_summary.csv"); runtime_source = state.source("runtime/runtime_scaling_summary.csv"); sensitivity_source = state.source("sensitivity/parameter_sensitivity_summary.csv")
    assert real_source and runtime_source and sensitivity_source
    real = pd.read_csv(real_source).sort_values("common_display_fraction"); runtime = pd.read_csv(runtime_source); sensitivity = pd.read_csv(sensitivity_source)
    overlap, overlap_sources = _real_application_overlap(state)
    _write_csv(state, overlap, state.output_dir / "figure_components" / "FigureS6_real_application_topK_overlap.csv")
    fig = plt.figure(figsize=(12.2, 8.7))
    grid = fig.add_gridspec(4, 2, height_ratios=[0.14, 1, 0.18, 1], hspace=0.34, wspace=0.34)
    top_legend_axis = fig.add_subplot(grid[0, 0])
    bottom_legend_axis = fig.add_subplot(grid[2, 0])
    top_legend_axis.axis("off")
    bottom_legend_axis.axis("off")
    bottom_right = grid[3, 1].subgridspec(1, 2, width_ratios=[1, 0.055], wspace=0.16)
    axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[3, 0]), fig.add_subplot(bottom_right[0, 0])]
    colorbar_axis = fig.add_subplot(bottom_right[0, 1])
    left = np.zeros(len(real))
    for column, color, label in [("n_subtype_ambiguous", "#8E72B5", "Ambiguous"), ("n_heterotypic_like", "#1479B8", "Heterotypic-like"), ("n_homotypic_like", "#D66900", "Homotypic-like")]:
        values = 100.0 * real[column].to_numpy(dtype=float) / real["common_display_top_k"].to_numpy(dtype=float)
        axes[0].barh(real["dataset"], values, left=left, color=color, label=label); left += values
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Subtype composition within common top-K (%)"); axes[0].set_title("Real-data candidate composition"); finish_manuscript_axes(axes[0], grid_axis="x")
    top_legend_axis.legend(*axes[0].get_legend_handles_labels(), loc="center", ncol=3, frameon=False, columnspacing=1.0)
    overlap_plot = overlap.loc[overlap["external_method"].isin(MAIN_METHODS[1:])]
    for method in MAIN_METHODS[1:]:
        values = overlap_plot.loc[overlap_plot["external_method"].eq(method), "jaccard"].to_numpy(dtype=float)
        positions = np.full(len(values), MAIN_METHODS[1:].index(method), dtype=float) + np.linspace(-0.12, 0.12, len(values))
        axes[1].scatter(positions, values, color=MANUSCRIPT_COLORS[method], s=18, alpha=0.7)
        axes[1].plot(MAIN_METHODS[1:].index(method), np.nanmean(values), marker="D", color="#202124", markersize=5)
    _set_categorical_xticks(axes[1], np.arange(len(MAIN_METHODS[1:])), MAIN_METHODS[1:])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Top-K Jaccard with DuoDose")
    axes[1].set_title("Cross-method candidate overlap")
    finish_manuscript_axes(axes[1])
    for method in COMPLETE_METHODS:
        subset = runtime.loc[runtime["method"].eq(method)].sort_values("n_cells")
        if not subset.empty: axes[2].errorbar(subset["n_cells"], subset["total_wall_clock_seconds_mean"], yerr=subset["total_wall_clock_seconds_std"], marker="o", capsize=3, label=method, color=MANUSCRIPT_COLORS[method])
    axes[2].set_xlabel("Cells"); axes[2].set_ylabel("Wall-clock time (s)"); axes[2].set_title("Runtime scaling (mean +/- SD; n = 3)"); finish_manuscript_axes(axes[2], grid_axis="both")
    bottom_legend_axis.legend(*axes[2].get_legend_handles_labels(), ncol=3, loc="center", frameon=False, columnspacing=1.0, handlelength=1.7)
    pivot = sensitivity.pivot(index="semi_real_size_factor", columns="expected_doublet_rate", values="high_RNA_singlet_FPR_at_expected_rate_mean").sort_index().sort_index(axis=1)
    heat_values = pivot.to_numpy(dtype=float)
    image = axes[3].imshow(heat_values, cmap="magma_r", aspect="auto", vmin=0, vmax=max(0.35, np.nanmax(heat_values))); axes[3].set_xticks(np.arange(len(pivot.columns)), [f"{v:.0%}" for v in pivot.columns]); axes[3].set_yticks(np.arange(len(pivot.index)), [f"{v:g}×" for v in pivot.index]); axes[3].set_xlabel("Expected doublet rate"); axes[3].set_ylabel("Semi-real size factor"); axes[3].set_title("Expected-rate high-RNA FPR sensitivity"); fig.colorbar(image, cax=colorbar_axis)
    _annotate_heatmap(axes[3], image, heat_values)
    label_panels(axes); fig.subplots_adjust(left=0.07, right=0.97, bottom=0.08, top=0.98)
    _assert_row_axes_aligned([axes[:2], axes[2:]])
    output = supplement_dir / "FigureS6_real_data_and_practicality.png"; save_manuscript_png(fig, output); state.register(output); outputs.append(output)
    for panel, sources, metric in [("A", [real_source], "candidate subtype counts normalized to 100% within each dataset's common top-K pool"), ("B", overlap_sources, "exact common-top-K Jaccard overlap with DuoDose; points are datasets and diamonds are means"), ("C", [runtime_source], "total_wall_clock_seconds mean +/- SD across 3 repetitions including DuoDose-DL"), ("D", [sensitivity_source], "high_RNA_singlet_FPR_at_expected_rate_mean with numeric annotations")]: _panel_manifest(state, figure="Figure S6", panel=panel, sources=sources, metrics=metric, action="regenerated from clean numeric data", output=output)
    return outputs


def _flatten_mapping(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_mapping(child, path))
    elif isinstance(value, list):
        rows.append({"parameter": prefix, "value": ", ".join(map(str, value))})
    else:
        rows.append({"parameter": prefix, "value": value})
    return rows


RF_FROZEN_HYPERPARAMETERS: dict[str, Any] = {
    "n_estimators": 240,
    "criterion": "gini",
    "max_features": "sqrt",
    "min_samples_leaf": 2,
    "bootstrap": True,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
    "calibration_method": "sigmoid",
    "calibration_scheme": "held_out_validation_prefit",
    "calibration_folds": "NOT_APPLICABLE (prefit calibration on the full held-out validation split)",
}

RF_NOT_APPLICABLE_HYPERPARAMETERS = [
    "hidden_dim", "depth", "dropout", "weight_decay", "batch_size", "best_epoch",
    "n_epochs", "early_stopping_epoch", "lambda_subtype", "lambda_highrna",
    "lambda_ncount_decorrelation", "training_backend", "training_device",
    "amp_enabled", "cuda_available", "deterministic", "best_validation_loss",
]


def _document_model_hyperparameters(training: pd.DataFrame) -> pd.DataFrame:
    """Add frozen estimator settings and explicit NOT_APPLICABLE values for Table S2."""
    frame = training.copy()
    for column in [*RF_FROZEN_HYPERPARAMETERS, *RF_NOT_APPLICABLE_HYPERPARAMETERS]:
        if column not in frame.columns:
            frame[column] = pd.NA
    backend = frame.get("backend", pd.Series("", index=frame.index)).astype(str).str.lower()
    estimator = frame.get("estimator", pd.Series("", index=frame.index)).astype(str)
    method = frame.get("method", pd.Series("", index=frame.index)).astype(str)
    rf_mask = backend.eq("rf") | estimator.eq("CalibratedRF") | method.eq("DuoDose-ML-CalibratedRF-SafeFeatures")
    for column, value in RF_FROZEN_HYPERPARAMETERS.items():
        frame.loc[rf_mask, column] = value
    for column in RF_NOT_APPLICABLE_HYPERPARAMETERS:
        frame[column] = frame[column].astype(object)
        frame.loc[rf_mask, column] = "NOT_APPLICABLE"
    return frame


def _supplementary_tables(state: BuildState, metadata: pd.DataFrame, metadata_sources: list[Path], controlled: pd.DataFrame, controlled_source: Path) -> dict[str, dict[str, Any]]:
    supplement = state.output_dir / "supplementary_tables"
    workbooks: dict[str, dict[str, Any]] = {}
    s1 = _write_csv(state, metadata, supplement / "TableS1_dataset_metadata.csv")
    _table_manifest(state, table="Table S1", sources=metadata_sources, output=s1, notes="Dataset size, split composition, and parent/reference QC.")

    config_path = state.source("validation_suite/validation_suite_config.json")
    construction_path = state.source("controlled/cline-ch/seed_0/construction_report.json")
    training_path = state.source("controlled/cline-ch/seed_0/training_summaries.csv")
    environment_path = state.source("validation_suite/validation_suite_environment.json")
    assert config_path and construction_path and training_path and environment_path
    frozen_config = json.loads(config_path.read_text(encoding="utf-8"))
    construction = json.loads(construction_path.read_text(encoding="utf-8"))
    construction["n_observed_background_cells_available"] = construction.pop("n_real_labeled_singlets_available", construction.get("n_observed_background_cells_available"))
    construction["n_reference_background_cells_used"] = construction.pop("n_reference_singlets_used", construction.get("n_reference_background_cells_used"))
    construction["background_population_definition"] = "label-blinded observed cells; may include experimentally labeled doublets"
    construction["downsampled_library_lower_quantile"] = "NOT_APPLICABLE for raw_sum_parents_removed"
    construction["downsampled_library_upper_quantile"] = "NOT_APPLICABLE for raw_sum_parents_removed"
    construction["legacy_saturation_range_ignored"] = "NOT_APPLICABLE"
    construction["label_definition"] = "positive=constructed homotypic/heterotypic doublets; negative=label-blinded observed background including high-RNA cells"
    formal_controlled_seeds = sorted({int(value) for value in controlled["seed"].dropna().tolist()})
    formal_benchmark_rows = [
        {"parameter": "formal_benchmark.controlled_seeds", "value": ", ".join(map(str, formal_controlled_seeds))},
        {"parameter": "formal_benchmark.controlled_seed_count", "value": len(formal_controlled_seeds)},
        {"parameter": "formal_benchmark.seed_role", "value": "independent benchmark repetitions; seed 0 remains the default application seed"},
    ]
    protocol_frame = pd.DataFrame(
        _flatten_mapping(frozen_config.get("frozen_contract", {}), "frozen_contract")
        + formal_benchmark_rows
        + _flatten_mapping(construction, "representative_construction")
    )
    training = _document_model_hyperparameters(pd.read_csv(training_path))
    seen_features: set[str] = set()
    for _, training_row in training.iterrows():
        method = str(training_row.get("public_method_name") or training_row.get("method") or "")
        for feature in str(training_row.get("feature_list") or "").split(","):
            feature = feature.strip()
            if feature and feature.lower() != "nan" and feature not in seen_features:
                seen_features.add(feature)
    feature_frame = build_safe_feature_manifest(sorted(seen_features))
    feature_frame.insert(0, "safe_feature_mode", str(frozen_config.get("frozen_contract", {}).get("safe_feature_mode", "")))
    feature_frame = feature_frame.loc[:, [column for column in [
        "safe_feature_mode", "feature_name", "public_display_name", "category", "primary_group", "source_groups", "is_composite", "direct_dependencies", "source_function", "source_file", "reference_state_used", "computed_before_model_fit", "uses_truth_labels", "uses_model_output", "uses_outcome_calibration", "uses_dataset_rank", "allowed_in_rf", "allowed_in_dl", "feature_version", "dependency_graph_version"
    ] if column in feature_frame]]
    hyper_columns = [column for column in [
        "public_method_name", "method", "backend", "estimator", "random_state",
        "n_estimators", "criterion", "max_features", "min_samples_leaf", "bootstrap", "class_weight", "n_jobs",
        "calibration_method", "calibration_scheme", "calibration_folds", "calibration_used",
        "validation_calibration_log_loss", "uncalibrated_validation_log_loss", "calibrated_validation_log_loss",
        "number_train_cells", "number_validation_cells", "number_test_cells", "number_train_doublets",
        "number_train_homotypic_doublets", "number_train_heterotypic_doublets", "hidden_dim", "depth", "dropout",
        "weight_decay", "batch_size", "best_epoch", "n_epochs", "early_stopping_epoch", "lambda_subtype",
        "lambda_highrna", "lambda_ncount_decorrelation", "highrna_percentile", "high_rna_negative_weight",
        "training_backend", "training_device", "amp_enabled", "cuda_available", "deterministic",
        "best_validation_AUPRC", "best_validation_loss", "training_time_seconds"
    ] if column in training]
    hyper = training[hyper_columns].copy()
    environment = pd.DataFrame(_flatten_mapping(json.loads(environment_path.read_text(encoding="utf-8"))))
    r_versions_path = state.repository_root / "reproducibility" / "environment" / "r_package_versions.csv"
    if not r_versions_path.is_file():
        raise FileNotFoundError(f"required R package version table is missing: {r_versions_path}")
    state.sources.add(r_versions_path.resolve())
    external_software = pd.concat(
        [
            pd.DataFrame([{"package": "Scrublet", "version": "0.2.3", "ecosystem": "Python", "role": "external baseline"}]),
            pd.read_csv(r_versions_path).assign(ecosystem="R/Bioconductor", role="external baseline runtime"),
        ],
        ignore_index=True,
    )
    workbooks["TableS2_protocol_features_hyperparameters_software.xlsx"] = {"Protocol": protocol_frame, "Feature provenance": feature_frame, "Model hyperparameters": hyper, "Python environment": environment, "External software": external_software}

    s3 = _write_csv(state, controlled, supplement / "TableS3_full_benchmark_results.csv")
    _table_manifest(state, table="Table S3", sources=[controlled_source], output=s3, notes="Full dataset × seed × method controlled results, including the DL ablation and explicit primary/supplementary high-RNA FPR columns.")

    operating_source = state.source("tables/final_high_RNA_operating_points_by_run.csv")
    assert operating_source
    operating = pd.read_csv(operating_source)
    s3_operating = _write_csv(
        state,
        operating,
        supplement / "TableS3_high_RNA_operating_points.csv",
    )
    _table_manifest(
        state,
        table="Table S3 operating points",
        sources=[operating_source],
        output=s3_operating,
        notes=(
            "Complete per-run standardized operating-point audit with eight rows per method-run: fixed top-20%; "
            "matched overall recall 50%/70%/80%/90%; and matched homotypic recall 50%/70%/80%. "
            "The historical true-doublet-budget FPR is retained as a column in Table S3 full benchmark results "
            "and is not a separate operating-point row."
        ),
    )

    domain_source = state.source("domain_audit/domain_audit_all_datasets_summary.csv"); checks_source = state.source("validation_suite/validation_suite_checks.csv"); subtype_source = state.source("validation_suite/subtype_permutation_summary.csv"); full_source = state.source("validation_suite/full_label_permutation_summary.csv")
    assert domain_source and checks_source and subtype_source and full_source
    manuscript_domain = _drop_columns_with_tokens(pd.read_csv(domain_source), INTERNAL_ONLY_COLUMN_TOKENS)
    workbooks["TableS4_domain_validation_summaries.xlsx"] = {"Domain audit": manuscript_domain, "Validation checks": _drop_columns_with_tokens(pd.read_csv(checks_source), INTERNAL_ONLY_COLUMN_TOKENS), "Subtype permutation": pd.read_csv(subtype_source), "Full-label permutation": pd.read_csv(full_source)}

    runtime_source = state.source("runtime/runtime_scaling_summary.csv"); runtime_runs_source = state.source("runtime/runtime_scaling_by_run.csv"); sensitivity_source = state.source("sensitivity/parameter_sensitivity_summary.csv"); sensitivity_runs_source = state.source("sensitivity/parameter_sensitivity_by_run.csv")
    assert runtime_source and runtime_runs_source and sensitivity_source and sensitivity_runs_source
    workbooks["TableS5_runtime_and_scalability.xlsx"] = {"Runtime scaling summary": _runtime_manuscript_frame(pd.read_csv(runtime_source)), "Runtime scaling by run": _runtime_manuscript_frame(pd.read_csv(runtime_runs_source)), "Sensitivity summary": pd.read_csv(sensitivity_source), "Sensitivity by run": pd.read_csv(sensitivity_runs_source)}

    candidate_source = state.source("tables/final_real_application_candidate_summary.csv")
    assert candidate_source
    overlap, _ = _real_application_overlap(state)
    overlap_summary = overlap.groupby("external_method", sort=False)["jaccard"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    workbooks["TableS6_real_application_summaries.xlsx"] = {"Candidate summary": pd.read_csv(candidate_source), "Top-K overlap by dataset": overlap, "Top-K overlap summary": overlap_summary}

    for filename, sheets in workbooks.items():
        for sheet_name, frame in sheets.items():
            component_name = f"{Path(filename).stem}__{sheet_name.replace(' ', '_').replace('-', '_')}.csv"
            _write_csv(state, frame, state.output_dir / "table_components" / component_name)
    return workbooks


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA: return None
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, float) and not np.isfinite(value): return None
    if pd.isna(value): return None
    return value


def _frame_spec(frame: pd.DataFrame) -> dict[str, Any]:
    return {"columns": list(map(str, frame.columns)), "rows": [[_json_value(value) for value in row] for row in frame.itertuples(index=False, name=None)]}


def _build_workbooks(
    state: BuildState,
    workbooks: dict[str, dict[str, Any]],
    table1: dict[str, Any] | None,
    *,
    require_xlsx: bool = False,
) -> list[Path]:
    workbook_specs = []
    all_workbooks = dict(workbooks)
    if table1 is not None:
        all_workbooks = {"Table1_core_benchmark_summary.xlsx": {key: value for key, value in table1.items() if isinstance(value, pd.DataFrame)}, **all_workbooks}
    for filename, sheets in all_workbooks.items():
        target_dir = "main_tables" if filename.startswith("Table1") else "supplementary_tables"
        workbook_specs.append({"output": f"{target_dir}/{filename}", "sheets": [{"name": name[:31], **_frame_spec(frame)} for name, frame in sheets.items()]})
    spec_path = state.output_dir / "table_components" / "workbook_spec.json"
    spec_path.write_text(json.dumps({"workbooks": workbook_specs}, indent=2, ensure_ascii=False), encoding="utf-8")
    state.register(spec_path)
    builder_source = state.repository_root / "reproducibility/build_manuscript_workbooks.mjs"
    node = os.environ.get("DUODOSE_NODE") or shutil.which("node")

    def omit_workbooks(reason: str, *, details: str = "") -> list[Path]:
        if require_xlsx:
            raise RuntimeError(reason + (f"\n{details}" if details else ""))
        for workbook in workbook_specs:
            state.omitted.append({
                "artifact": workbook["output"],
                "reason": reason,
            })
        status = state.output_dir / "reports/xlsx_generation_status.md"
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(
            "# XLSX generation status\n\n"
            "XLSX workbooks were not generated in this run. This does not affect the manuscript figures, "
            "CSV tables, table components, or scientific results.\n\n"
            f"Reason: {reason}\n\n"
            "The complete workbook specification is available at "
            "`table_components/workbook_spec.json`, and every worksheet is also exported as a CSV under "
            "`table_components/`. To make XLSX mandatory, rerun with `--require-xlsx` after configuring "
            "Node.js and `@oai/artifact-tool`.\n"
            + (f"\n## Tool output\n\n```text\n{details.strip()}\n```\n" if details.strip() else ""),
            encoding="utf-8",
        )
        state.register(status)
        return []

    if not builder_source.is_file():
        return omit_workbooks(f"workbook builder is missing: {builder_source}")
    if not node:
        return omit_workbooks(
            "Node.js was not found; optional XLSX authoring was skipped."
        )
    try:
        with tempfile.TemporaryDirectory(prefix="duodose_manuscript_xlsx_") as temporary:
            workdir = Path(temporary)
            builder = workdir / builder_source.name
            shutil.copy2(builder_source, builder)
            dependency_root = os.environ.get("DUODOSE_NODE_MODULES")
            if dependency_root:
                junction = workdir / "node_modules"
                linked = subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(junction), str(Path(dependency_root).resolve())],
                    text=True,
                    capture_output=True,
                )
                if linked.returncode != 0:
                    return omit_workbooks(
                        "Could not prepare the optional Artifact Tool dependency junction.",
                        details=linked.stderr or linked.stdout,
                    )
            completed = subprocess.run(
                [node, str(builder), str(spec_path), str(state.output_dir)],
                cwd=workdir,
                text=True,
                capture_output=True,
            )
    except OSError as exc:
        return omit_workbooks(
            "Optional XLSX authoring could not be launched.",
            details=str(exc),
        )
    if completed.returncode != 0:
        return omit_workbooks(
            "Optional Artifact Tool workbook generation failed.",
            details=f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}",
        )
    outputs: list[Path] = []
    for workbook in workbook_specs:
        output = state.output_dir / workbook["output"]
        state.register(output); outputs.append(output)
        inspection = Path(f"{output}.inspect.ndjson")
        if inspection.is_file():
            inspection_dir = state.output_dir / "manifests/workbook_inspections"
            inspection_dir.mkdir(parents=True, exist_ok=True)
            inspection_target = inspection_dir / inspection.name
            shutil.move(str(inspection), inspection_target)
            state.register(inspection_target)
    if table1 is not None:
        table1_output = state.output_dir / "main_tables/Table1_core_benchmark_summary.xlsx"
        table1_source = state.source("tables/final_controlled_comparison_by_run.csv")
        assert table1_source is not None
        _table_manifest(state, table="Table 1", sources=[table1_source], output=table1_output, notes="Artifact Tool workbook with display, numeric, and notes sheets.")
    supplementary_sources = {
        "TableS2_protocol_features_hyperparameters_software.xlsx": [state.source("validation_suite/validation_suite_config.json"), state.source("controlled/cline-ch/seed_0/construction_report.json"), state.source("controlled/cline-ch/seed_0/training_summaries.csv"), state.source("validation_suite/validation_suite_environment.json")],
        "TableS4_domain_validation_summaries.xlsx": [state.source("domain_audit/domain_audit_all_datasets_summary.csv"), state.source("validation_suite/validation_suite_checks.csv")],
        "TableS5_runtime_and_scalability.xlsx": [state.source("runtime/runtime_scaling_summary.csv"), state.source("sensitivity/parameter_sensitivity_summary.csv")],
        "TableS6_real_application_summaries.xlsx": [state.source("tables/final_real_application_candidate_summary.csv"), *[path for path in sorted((state.results_dir / "real_application").glob("*/seed_0/real_application_method_scores.csv.gz"))]],
    }
    for filename, sources in supplementary_sources.items():
        notes = "Runtime and scalability; elapsed-time measurements and variability only." if filename == "TableS5_runtime_and_scalability.xlsx" else "Multi-sheet Artifact Tool workbook."
        _table_manifest(state, table=filename.split("_")[0].replace("TableS", "Table S"), sources=[path for path in sources if path], output=state.output_dir / "supplementary_tables" / filename, notes=notes)
    return outputs


def _write_reports(state: BuildState, *, main_outputs: list[Path], supplement_outputs: list[Path], table_outputs: list[Path]) -> tuple[Path, Path]:
    report = state.output_dir / "reports/manuscript_materials_report.md"
    writing = state.output_dir / "reports/manuscript_writing_index.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    source_dirs = sorted({state.relative_source(path.parent) for path in state.sources})
    report.write_text(
        "# DuoDose Manuscript Materials Report\n\n"
        "## Scope\n\n"
        "This lightweight aggregation used only completed clean artifacts under `results/final_v1`. It did not rerun model fitting, benchmarks, domain analyses, or real-data inference. Figure 1 is intentionally absent because it is a manually drawn conceptual schematic.\n\n"
        "## Scientific decisions\n\n"
        "- Representative real-data application: `cline-ch`.\n"
        "- Formal controlled benchmark seeds: `0, 1, 2, 3, 4`; seed 0 remains the default application seed.\n"
        "- Frozen RF: 240 trees, `max_features=sqrt`, `min_samples_leaf=2`, balanced-subsample class weights, and held-out sigmoid prefit calibration.\n"
        f"- Main-text high-RNA FPR definition: {HIGH_RNA_FPR_DEFINITION}\n"
        "- Controlled construction is label-blind: observed background cells may include experimentally labeled doublets; experimental labels are joined only for post-hoc audit/evaluation.\n"
        "- Precision at common K equals Recall at common K when K equals the number of controlled positives; Figure 3 therefore displays Precision at K only.\n"
        "- Runtime and scalability materials report elapsed-time measurements and variability only.\n"
        "- `DuoDose-DL` appears only in supplementary Figure S2/S3/S6 and Table S3/S5; it is absent from Figures 2-5 and Table 1.\n"
        "- Figure 4 reuses the finalized clean 3 × 3 UMAP and adds only standardized panel labels.\n"
        "- Domain-audit interpretation is conservative: matching reduced but did not eliminate domain separability; pooled out-of-fold AUROC values indicate weak-to-moderate residual separation, not indistinguishability.\n"
        f"- Unified Arial-first manuscript style was applied. Resolved local font: `{state.resolved_font}`.\n"
        "- Figures are PNG only; no PDF files are produced.\n\n"
        "## Clean source directories\n\n" + "\n".join(f"- `{directory}`" if directory.startswith("reproducibility/") else f"- `results/final_v1/{directory}`" for directory in source_dirs) + "\n\n"
        "## Main outputs\n\n" + "\n".join(f"- `{state.relative_output(path)}`" for path in [*main_outputs, *table_outputs] if "supplementary" not in path.parts) + "\n\n"
        "## Supplementary outputs\n\n" + "\n".join(f"- `{state.relative_output(path)}`" for path in supplement_outputs + [path for path in table_outputs if "supplementary" in path.parts]) + "\n\n"
        "## Missing or omitted inputs\n\n" + ("\n".join(f"- {item['artifact']}: {item['reason']}" for item in state.omitted) if state.omitted else "None.") + "\n",
        encoding="utf-8",
    )
    state.register(report)
    entries = [
        ("Figure 2", "main_figures/Figure2_controlled_benchmark.png"), ("Figure 3", "main_figures/Figure3_homotypic_vs_highRNA.png"), ("Figure 4", "main_figures/Figure4_real_application_cline_ch.png"), ("Figure 5", "main_figures/Figure5_robustness_practicality.png"), ("Table 1 (CSV)", "main_tables/Table1_core_benchmark_summary.csv"), ("Table 1 (XLSX)", "main_tables/Table1_core_benchmark_summary.xlsx"),
        *[(f"Figure S{index}", path) for index, path in enumerate(["supplementary_figures/FigureS1_dataset_overview.png", "supplementary_figures/FigureS2_complete_controlled_benchmark.png", "supplementary_figures/FigureS3_RF_vs_DL_ablation.png", "supplementary_figures/FigureS4_validation_and_negative_controls.png", "supplementary_figures/FigureS5_full_domain_audit.png", "supplementary_figures/FigureS6_real_data_and_practicality.png"], start=1)],
        ("Table S1", "supplementary_tables/TableS1_dataset_metadata.csv"), ("Table S2", "supplementary_tables/TableS2_protocol_features_hyperparameters_software.xlsx"), ("Table S3", "supplementary_tables/TableS3_full_benchmark_results.csv"), ("Table S3 operating points", "supplementary_tables/TableS3_high_RNA_operating_points.csv"), ("Table S4", "supplementary_tables/TableS4_domain_validation_summaries.xlsx"), ("Table S5: Runtime and scalability", "supplementary_tables/TableS5_runtime_and_scalability.xlsx"), ("Table S6", "supplementary_tables/TableS6_real_application_summaries.xlsx"),
    ]
    available_entries = [f"- **{label}:** `{path}`" for label, path in entries if (state.output_dir / path).is_file()]
    component_tables = {
        "Table S2 CSV components": [
            "table_components/TableS2_protocol_features_hyperparameters_software__Protocol.csv",
            "table_components/TableS2_protocol_features_hyperparameters_software__Feature_provenance.csv",
            "table_components/TableS2_protocol_features_hyperparameters_software__Model_hyperparameters.csv",
            "table_components/TableS2_protocol_features_hyperparameters_software__Python_environment.csv",
            "table_components/TableS2_protocol_features_hyperparameters_software__External_software.csv",
        ],
        "Table S4 CSV components": [
            "table_components/TableS4_domain_validation_summaries__Domain_audit.csv",
            "table_components/TableS4_domain_validation_summaries__Validation_checks.csv",
            "table_components/TableS4_domain_validation_summaries__Subtype_permutation.csv",
            "table_components/TableS4_domain_validation_summaries__Full_label_permutation.csv",
        ],
        "Table S5 CSV components": [
            "table_components/TableS5_runtime_and_scalability__Runtime_scaling_summary.csv",
            "table_components/TableS5_runtime_and_scalability__Runtime_scaling_by_run.csv",
            "table_components/TableS5_runtime_and_scalability__Sensitivity_summary.csv",
            "table_components/TableS5_runtime_and_scalability__Sensitivity_by_run.csv",
        ],
        "Table S6 CSV components": [
            "table_components/TableS6_real_application_summaries__Candidate_summary.csv",
            "table_components/TableS6_real_application_summaries__Top_K_overlap_by_dataset.csv",
            "table_components/TableS6_real_application_summaries__Top_K_overlap_summary.csv",
        ],
    }
    component_lines: list[str] = []
    for label, paths in component_tables.items():
        existing = [path for path in paths if (state.output_dir / path).is_file()]
        if not existing:
            continue
        component_lines.append(f"- **{label}:**")
        component_lines.extend(f"  - `{path}`" for path in existing)
    writing.write_text(
        "# Manuscript Writing Index\n\n"
        "Figure 1 is intentionally manual and is not generated here.\n\n"
        + "\n".join(available_entries)
        + "\n\n## Supplementary tables when XLSX is unavailable\n\n"
        "The CSV components below are the complete worksheet-level sources for Tables S2, S4, S5, and S6. "
        "They are valid submission sources even when optional XLSX authoring is skipped.\n\n"
        + "\n".join(component_lines)
        + "\n\n## Claim boundaries\n\n"
        "- Describe domain matching as reducing domain separability; do not claim that semi-real and experimental domains are indistinguishable.\n"
        "- Describe heterotypic detection as competitive, not leading.\n"
        "- Describe the primary high-RNA FPR as label-relative and evaluated at matched 50% homotypic recall.\n",
        encoding="utf-8",
    )
    state.register(writing)
    return report, writing


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _verify(state: BuildState, *, mode: str) -> None:
    from PIL import Image

    pdfs = list(state.output_dir.rglob("*.pdf"))
    if pdfs: raise RuntimeError(f"PNG-only contract violated; PDF files found: {pdfs}")
    for path in state.output_dir.rglob("*.png"):
        with Image.open(path) as image:
            image.verify()
        if path.stat().st_size == 0: raise RuntimeError(f"empty PNG: {path}")
    if mode in {"all", "main"}:
        for relative in ["main_figures/Figure2_controlled_benchmark.png", "main_figures/Figure3_homotypic_vs_highRNA.png", "main_figures/Figure4_real_application_cline_ch.png", "main_figures/Figure5_robustness_practicality.png", "main_tables/Table1_core_benchmark_summary.csv"]:
            path = state.output_dir / relative
            if not path.is_file() or path.stat().st_size == 0: raise RuntimeError(f"required main artifact missing: {path}")
        table1 = pd.read_csv(state.output_dir / "main_tables/Table1_core_benchmark_summary.csv")
        if "DuoDose-DL" in set(table1["method"]): raise RuntimeError("DuoDose-DL leaked into Table 1")
        for figure in ["Figure2_controlled_benchmark", "Figure3_homotypic_vs_highRNA", "Figure5_robustness_practicality"]:
            rows = [row for row in state.panel_rows if figure.split("_")[0].replace("Figure", "Figure ") in row["figure"]]
            if any("DuoDose-DL" in row["source_tables_or_metrics"] for row in rows): raise RuntimeError(f"DuoDose-DL leaked into main figure contract: {figure}")


def generate_manuscript_materials(
    *,
    repository_root: Path,
    results_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    mode: str = "all",
    skip_missing_noncritical: bool = False,
    require_xlsx: bool = False,
) -> dict[str, Any]:
    """Generate the requested cached manuscript package and verify its contracts."""

    repository_root = Path(repository_root).resolve()
    results_dir = Path(results_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not results_dir.is_dir(): raise FileNotFoundError(f"clean results directory does not exist: {results_dir}")
    if output_dir.exists():
        if not overwrite: raise FileExistsError(f"output directory already exists; pass --overwrite: {output_dir}")
        if output_dir == repository_root or output_dir == output_dir.anchor: raise ValueError("refusing to remove an unsafe output directory")
        shutil.rmtree(output_dir)
    for name in ["main_figures", "supplementary_figures", "main_tables", "supplementary_tables", "figure_components", "table_components", "manifests", "reports"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    state = BuildState(results_dir=results_dir, output_dir=output_dir, repository_root=repository_root)
    state.resolved_font = apply_manuscript_style()
    controlled, controlled_source = _successful_controlled(state)
    metadata, metadata_sources = _dataset_metadata(state)
    main_outputs: list[Path] = []
    supplement_outputs: list[Path] = []
    table_outputs: list[Path] = []
    table1: dict[str, Any] | None = None
    workbook_specs: dict[str, dict[str, Any]] = {}
    if mode in {"all", "main"}:
        main_outputs.extend([_figure2(state, controlled, controlled_source), _figure3(state, controlled, controlled_source), _figure4(state), _figure5(state)])
        table1 = _create_table1(state, controlled, controlled_source)
        table_outputs.extend([state.output_dir / "main_tables/Table1_core_benchmark_summary.csv", table1["numeric_path"]])
    if mode in {"all", "supplement"}:
        try:
            supplement_outputs = _supplementary_figures(state, controlled, controlled_source, metadata, metadata_sources)
            workbook_specs = _supplementary_tables(state, metadata, metadata_sources, controlled, controlled_source)
            table_outputs.extend([state.output_dir / "supplementary_tables/TableS1_dataset_metadata.csv", state.output_dir / "supplementary_tables/TableS3_full_benchmark_results.csv", state.output_dir / "supplementary_tables/TableS3_high_RNA_operating_points.csv"])
        except (FileNotFoundError, ValueError) as exc:
            if not skip_missing_noncritical: raise
            state.omitted.append({"artifact": "supplementary materials", "reason": str(exc)})
    workbook_outputs = _build_workbooks(state, workbook_specs, table1, require_xlsx=require_xlsx)
    table_outputs.extend(workbook_outputs)
    panel_manifest = _write_csv(state, pd.DataFrame(state.panel_rows), output_dir / "manifests/figure_panel_manifest.csv")
    table_manifest = _write_csv(state, pd.DataFrame(state.table_rows), output_dir / "manifests/table_manifest.csv")
    _write_reports(state, main_outputs=main_outputs, supplement_outputs=supplement_outputs, table_outputs=table_outputs)
    _verify(state, mode=mode)
    artifacts = []
    for path in sorted({path for path in output_dir.rglob("*") if path.is_file()}):
        artifacts.append({"path": state.relative_output(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    manifest = {
        "schema_version": 1,
        "workflow": "manuscript_material_aggregation",
        "scientific_source_root": "results/final_v1",
        "output_root": "manuscript_materials",
        "mode": mode,
        "png_only": True,
        "figure_1_generated": False,
        "main_methods": MAIN_METHODS,
        "supplementary_ablation_method": "DuoDose-DL",
        "representative_dataset": "cline-ch",
        "high_RNA_singlet_FPR_definition": HIGH_RNA_FPR_DEFINITION,
        "requested_font": "Arial",
        "resolved_font": state.resolved_font,
        "source_files": sorted(state.relative_source(path) for path in state.sources),
        "omitted_noncritical": state.omitted,
        "xlsx_required": bool(require_xlsx),
        "xlsx_generated": any(path.suffix.lower() == ".xlsx" for path in output_dir.rglob("*.xlsx")),
        "artifacts": artifacts,
        "verification": {"required_outputs_nonempty": True, "all_pngs_opened": True, "pdf_count": 0, "panel_manifest": state.relative_output(panel_manifest), "table_manifest": state.relative_output(table_manifest)},
    }
    manifest_path = output_dir / "manuscript_materials_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"manifest": manifest_path, "main_outputs": main_outputs, "supplement_outputs": supplement_outputs, "table_outputs": table_outputs, "font": state.resolved_font, "omitted": state.omitted}
