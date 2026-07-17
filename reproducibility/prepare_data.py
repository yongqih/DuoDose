"""Discover, optionally convert, and checksum manuscript input datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from duodose import realdata  # noqa: E402
from duodose.protocol import load_final_protocol  # noqa: E402
from reproducibility.lib.common import discover_dataset_manifest, sha256_file, split_csv  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--convert-rds", action="store_true")
    parser.add_argument("--refresh-conversion", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    protocol = load_final_protocol(args.protocol)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = discover_dataset_manifest(args.data_dir)
    selected = split_csv(args.datasets)
    if selected and selected != ["all"]:
        missing = sorted(set(selected) - set(manifest["dataset"]))
        if missing:
            raise ValueError(f"unknown exact dataset names: {', '.join(missing)}")
        manifest = manifest.loc[manifest["dataset"].isin(selected)].copy()
    elif selected == ["all"]:
        configured = set(protocol["datasets"]["real_doublet_enriched"])
        missing = sorted(configured - set(manifest["dataset"]))
        if missing:
            raise FileNotFoundError("configured formal datasets were not discovered: " + ", ".join(missing))
        manifest = manifest.loc[manifest["dataset"].isin(configured)].copy()

    conversion_rows = []
    for row in manifest.itertuples(index=False):
        path = Path(row.resolved_path)
        candidate = realdata.DatasetCandidate(row.dataset, path, row.dataset_format, path)
        if row.dataset_format == "rds" and args.convert_rds:
            candidate = realdata.convert_rds_candidate(
                candidate,
                output,
                refresh_cache=bool(args.refresh_conversion),
                quiet_external=False,
            )
            if candidate.converted_status not in {"success", "cached"}:
                raise RuntimeError(f"RDS conversion failed for {row.dataset}: {candidate.converted_message}")
        conversion_rows.append(
            {
                "dataset": row.dataset,
                "source_path": str(path),
                "source_format": row.dataset_format,
                "source_size_bytes": path.stat().st_size if path.is_file() else None,
                "source_sha256": sha256_file(path) if path.is_file() else None,
                "resolved_input_path": str(candidate.load_path),
                "conversion_status": candidate.converted_status,
                "conversion_message": candidate.converted_message,
            }
        )
        if not Path(candidate.load_path).exists():
            raise FileNotFoundError(f"prepared input for {row.dataset} does not exist: {candidate.load_path}")
    manifest.to_csv(output / "dataset_discovery_manifest.csv", index=False)
    import pandas as pd

    pd.DataFrame(conversion_rows).to_csv(output / "dataset_input_manifest.csv", index=False)
    (output / "data_preparation_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": protocol["_protocol_path"],
                "data_dir": str(Path(args.data_dir).resolve()),
                "datasets": manifest["dataset"].tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Prepared dataset manifest:", ", ".join(manifest["dataset"]))


if __name__ == "__main__":
    main()
