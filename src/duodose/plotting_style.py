"""Shared manuscript plotting style and static figure-contract auditing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


REQUESTED_FONT = "Arial"
FONT_FALLBACKS = ("Arial", "Liberation Sans", "DejaVu Sans")
MANUSCRIPT_DPI = 300
MANUSCRIPT_COLORS = {
    "DuoDose": "#0B6E75",
    "DuoDose-DL": "#7A5195",
    "Scrublet": "#E07A1F",
    "scDblFinder": "#3B82B8",
    "DoubletFinder": "#6A994E",
    "scds": "#9C6644",
}
MANUSCRIPT_TEXT = "#202124"
MANUSCRIPT_GRID = "#D9DEE3"
FORMAL_PLOTTING_ENTRY_POINTS = (
    ("duodose.real_application", "src/duodose/real_application.py"),
    ("duodose.domain_audit_aggregate", "src/duodose/domain_audit_aggregate.py"),
    ("duodose.semireal_real_domain_audit", "src/duodose/semireal_real_domain_audit.py"),
    ("duodose.plots", "src/duodose/plots.py"),
    ("generate_final_artifacts", "reproducibility/generate_final_artifacts.py"),
    ("run_validation_suite", "reproducibility/run_validation_suite.py"),
    ("run_real_application_figure", "examples/run_real_application_figure.py"),
)


def apply_manuscript_style() -> str:
    """Apply the single Arial-first style used by manuscript-facing figures."""

    import matplotlib
    from matplotlib import font_manager

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(FONT_FALLBACKS),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.transparent": False,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": MANUSCRIPT_TEXT,
            "axes.labelcolor": MANUSCRIPT_TEXT,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.color": MANUSCRIPT_TEXT,
            "ytick.color": MANUSCRIPT_TEXT,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "font.size": 9,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.5,
            "grid.color": MANUSCRIPT_GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
        }
    )
    resolved = Path(font_manager.findfont(REQUESTED_FONT, fallback_to_default=True)).stem
    return REQUESTED_FONT if "arial" in resolved.lower() else resolved


def label_panels(axes: Sequence[object], labels: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ") -> list[object]:
    """Apply consistent bold panel labels to a sequence of Matplotlib axes."""

    artists: list[object] = []
    for label, ax in zip(labels, axes):
        artists.append(
            ax.text(
                -0.12,
                1.06,
                label,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color=MANUSCRIPT_TEXT,
            )
        )
    return artists


def finish_manuscript_axes(ax: object, *, grid_axis: str | None = "y") -> None:
    """Apply the shared minimal axis treatment used by result panels."""

    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


def save_manuscript_png(fig: object, path: Path, *, dpi: int = MANUSCRIPT_DPI) -> None:
    """Save one publication-facing PNG and close its Matplotlib figure."""

    import matplotlib.pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _pdf_uses_type3(path: Path) -> bool | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        return bool(re.search(rb"/Subtype\s*/Type3\b", path.read_bytes()))
    except OSError:
        return None


def audit_figure_style_contract(
    repository_root: Path,
    *,
    pdf_outputs: Iterable[Path] = (),
    png_outputs: Iterable[Path] = (),
) -> pd.DataFrame:
    """Audit shared-style adoption, conflicting overrides, and PDF font type."""

    root = Path(repository_root).resolve()
    pdfs = [Path(path).resolve() for path in pdf_outputs]
    pngs = [Path(path).resolve() for path in png_outputs]
    rows: list[dict[str, object]] = []
    conflict_pattern = re.compile(
        r"(?:font\.family|font\.sans-serif|fontfamily|fontname)[^\n]*(?:Times New Roman|DejaVu Sans)",
        flags=re.IGNORECASE,
    )
    for module, relative in FORMAL_PLOTTING_ENTRY_POINTS:
        path = root / relative
        source = path.read_text(encoding="utf-8") if path.is_file() else ""
        shared = "apply_manuscript_style" in source
        conflicts = bool(conflict_pattern.search(source))
        module_pdfs = [pdf for pdf in pdfs if pdf.is_file()]
        type3 = [_pdf_uses_type3(pdf) for pdf in module_pdfs]
        inspected = [value for value in type3 if value is not None]
        status = "PASS" if path.is_file() and shared and not conflicts and not any(inspected) else "FAIL"
        notes: list[str] = []
        if not path.is_file():
            notes.append("entry point is missing")
        if not shared:
            notes.append("shared style is not invoked")
        if conflicts:
            notes.append("conflicting local font override found")
        if any(inspected):
            notes.append("Type 3 glyphs found in an inspected PDF")
        rows.append(
            {
                "figure_module": module,
                "plotting_entry_point": relative,
                "shared_style_applied": shared,
                "requested_font": REQUESTED_FONT,
                "conflicting_override_found": conflicts,
                "png_output_checked": any(path.is_file() and path.stat().st_size > 0 for path in pngs),
                "pdf_output_checked": bool(inspected),
                "contract_status": status,
                "notes": "; ".join(notes),
            }
        )
    return pd.DataFrame(rows)
