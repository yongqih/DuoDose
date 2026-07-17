"""Generate publication-facing manuscript materials from frozen clean results."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from duodose.manuscript_materials import generate_manuscript_materials


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate results/final_v1 into PNG figures, CSV tables, optional XLSX workbooks, manifests, and writing aids."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results/final_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("manuscript_materials"))
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing manuscript-material output directory.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--main-only", action="store_true", help="Generate Figures 2-5 and Table 1 only.")
    mode.add_argument("--supplement-only", action="store_true", help="Generate Figures S1-S6 and Tables S1-S6 only.")
    parser.add_argument(
        "--skip-missing-noncritical",
        action="store_true",
        help="Record and omit optional supplementary material whose clean source is unavailable.",
    )
    parser.add_argument(
        "--require-xlsx",
        action="store_true",
        help=(
            "Require XLSX workbook generation and fail if Node.js/@oai/artifact-tool is unavailable. "
            "By default, figures and CSV tables still complete when XLSX tooling is absent."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    mode = "main" if args.main_only else "supplement" if args.supplement_only else "all"
    started = time.perf_counter()
    result = generate_manuscript_materials(
        repository_root=ROOT,
        results_dir=(ROOT / args.results_dir).resolve() if not args.results_dir.is_absolute() else args.results_dir,
        output_dir=(ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir,
        overwrite=args.overwrite,
        mode=mode,
        skip_missing_noncritical=args.skip_missing_noncritical,
        require_xlsx=args.require_xlsx,
    )
    elapsed = time.perf_counter() - started
    print("Manuscript-material aggregation complete")
    print(f"  output: {result['manifest'].parent}")
    print(f"  mode: {mode}")
    print(f"  main figures: {len(result['main_outputs'])}")
    print(f"  supplementary figures: {len(result['supplement_outputs'])}")
    print(f"  table artifacts: {len(result['table_outputs'])}")
    print(f"  resolved font: {result['font']}")
    print(f"  omitted noncritical: {len(result['omitted'])}")
    print(f"  runtime: {elapsed:.1f} s")
    print(f"  manifest: {result['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
