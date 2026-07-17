"""Run the manuscript-facing, label-free real-data application figure stage."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from duodose.progress import ProgressReporter, ProgressSettings, add_progress_arguments, progress_paths  # noqa: E402
from duodose.protocol import load_final_protocol  # noqa: E402
from duodose.real_application import (  # noqa: E402
    EXTERNAL_FIGURE_METHODS,
    RealApplicationResult,
    apply_common_budget_candidate_classes,
    candidate_calls,
    candidate_display_audit,
    candidate_summary,
    choose_panel_annotation,
    common_top_k_masks,
    compute_shared_embedding,
    experimental_display_budget,
    figure_manifest_payload,
    fit_public_rf_label_free,
    label_blinded_adata,
    label_usage_audit,
    local_diagnostics,
    plot_cross_method_umap,
    plot_diagnostics,
    reference_audit,
    shared_embedding_audit,
    summarize_diagnostics,
    validate_figure_contract,
    write_json,
)
from reproducibility.lib.common import (  # noqa: E402
    discover_dataset_manifest,
    load_dataset_exact,
    run_external_scores,
    sha256_file,
    split_csv,
    write_output_manifest,
    write_run_manifest,
)


REQUIRED_METHODS = ("DuoDose", *EXTERNAL_FIGURE_METHODS)


def validate_output_root(path: str | Path) -> Path:
    """Reject writes outside the clean repository, including the legacy tree."""

    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"real-application outputs must remain inside the clean repository: {resolved}") from exc
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--dataset", help="One exact dataset name.")
    selection.add_argument("--datasets", default=None, help="Comma-separated exact names or 'all'.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conversion-dir", default=None)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--training-preset", choices=("fast", "default", "robust"), default="default")
    parser.add_argument("--expected-doublet-rate", type=float, default=None)
    parser.add_argument("--external-methods", default=",".join(EXTERNAL_FIGURE_METHODS))
    parser.add_argument("--convert-rds", action="store_true")
    parser.add_argument("--refresh-conversion", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--n-hvgs", type=int, default=None)
    parser.add_argument("--n-pcs", type=int, default=None)
    parser.add_argument("--n-neighbors", type=int, default=None)
    parser.add_argument("--umap-min-dist", type=float, default=None)
    parser.add_argument("--n-clusters", type=int, default=None)
    add_progress_arguments(parser)
    return parser


def _selected_datasets(args: argparse.Namespace, protocol: dict, manifest: pd.DataFrame) -> list[str]:
    configured = list(protocol["datasets"]["real_application"])
    available = set(manifest.loc[manifest["discovery_status"].eq("valid"), "dataset"].astype(str))
    if args.dataset:
        selected = [str(args.dataset)]
    elif args.datasets and str(args.datasets).strip().lower() != "all":
        selected = split_csv(args.datasets)
    else:
        selected = configured
    unknown = sorted(set(selected) - available)
    if unknown:
        raise ValueError(f"real-application dataset(s) unavailable by exact name: {', '.join(unknown)}")
    outside_protocol = sorted(set(selected) - set(configured))
    if outside_protocol:
        raise ValueError(f"dataset(s) are not configured for the formal real-data application: {', '.join(outside_protocol)}")
    return selected


def _write_frame(frame: pd.DataFrame, path: Path, *, index_name: str = "cell_id") -> None:
    output = frame.copy()
    output.index = output.index.astype(str)
    output.index.name = index_name
    output.to_csv(path, index=True)


def _method_status(dataset: str, seed: int, result, runtime_seconds: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "seed": int(seed),
                "method": "DuoDose",
                "internal_method_name": result.model_metadata.get("internal_method_name", ""),
                "backend": result.backend,
                "status": "success",
                "message": "",
                "runtime_seconds": float(runtime_seconds),
                "n_finite_scores": int(pd.to_numeric(result.scores["duodose_score"], errors="coerce").notna().sum()),
            }
        ]
    )


def _run_one(
    *,
    args: argparse.Namespace,
    protocol: dict,
    dataset: str,
    seed: int,
    output: Path,
    conversion_dir: Path,
    reporter: ProgressReporter,
) -> pd.DataFrame:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    (output / "failure.json").unlink(missing_ok=True)
    (output / ".scores_before_label_join.csv").unlink(missing_ok=True)
    loaded = load_dataset_exact(
        args.data_dir,
        dataset,
        conversion_dir=conversion_dir,
        convert_rds=bool(args.convert_rds),
        refresh_conversion=bool(args.refresh_conversion),
    )
    unit = reporter.start_unit(
        dataset=dataset,
        seed=seed,
        method="DuoDose RF + cross-method UMAP",
        cell_count=int(loaded.adata.n_obs),
        gene_count=int(loaded.adata.n_vars),
        output_path=output,
        prefix=f"real-data application: {dataset} seed {seed}",
    )
    try:
        app = protocol["real_application"]
        prediction = protocol["prediction"]
        expected_rate = float(args.expected_doublet_rate if args.expected_doublet_rate is not None else prediction["expected_doublet_rate"])
        methods = tuple(split_csv(args.external_methods))
        if set(methods) != set(EXTERNAL_FIGURE_METHODS):
            raise ValueError(f"formal real-data application requires exactly: {', '.join(EXTERNAL_FIGURE_METHODS)}")

        # Preserve labels separately; every fitted/scored/embedded input is label-blinded.
        blind, experimental_labels = label_blinded_adata(loaded.adata)
        reporter.event("fitting canonical public DuoDose RF on label-blinded data", dataset=dataset, seed=seed)
        fit_started = time.perf_counter()
        detector, duodose_result, safe_scores = fit_public_rf_label_free(
            blind,
            expected_doublet_rate=expected_rate,
            random_state=seed,
            training_preset=args.training_preset,
        )
        rf_runtime = time.perf_counter() - fit_started

        reporter.event("computing one shared PCA/UMAP embedding", dataset=dataset, seed=seed)
        coordinates, representation, clusters = compute_shared_embedding(
            blind,
            random_state=seed,
            n_hvgs=int(args.n_hvgs or app["n_hvgs"]),
            n_pcs=int(args.n_pcs or app["n_pcs"]),
            n_neighbors=int(args.n_neighbors or app["n_neighbors"]),
            min_dist=float(args.umap_min_dist if args.umap_min_dist is not None else app["umap_min_dist"]),
            n_clusters=int(args.n_clusters or app["n_clusters"]),
        )
        annotation, annotation_name = choose_panel_annotation(blind.obs, clusters)

        reporter.event("running external methods on the same label-blinded cells", dataset=dataset, seed=seed)
        external_scores, external_status = run_external_scores(
            blind,
            dataset=dataset,
            seed=seed,
            methods=methods,
            cache_dir=output / "score_cache",
            expected_doublet_rate=expected_rate,
            refresh_cache=bool(args.refresh_cache),
            progress_callback=reporter.callback,
            log_dir=output / "logs",
        )

        method_scores = pd.DataFrame(index=blind.obs_names)
        for method in EXTERNAL_FIGURE_METHODS:
            method_scores[method] = pd.to_numeric(external_scores.get(method, pd.Series(np.nan, index=blind.obs_names)), errors="coerce").reindex(blind.obs_names)
        method_scores["DuoDose"] = pd.to_numeric(duodose_result.scores["duodose_score"], errors="coerce").reindex(blind.obs_names)
        # Freeze all model outputs before experimental labels are joined for display.
        method_scores = method_scores.copy(deep=True)
        calls = candidate_calls(
            duodose_result.scores.reindex(blind.obs_names),
            overall_threshold=float(duodose_result.threshold),
            homotypic_threshold=float(prediction["subtype_homotypic_threshold"]),
            heterotypic_threshold=float(prediction["subtype_heterotypic_threshold"]),
        )
        calls = calls.copy(deep=True)
        score_hash_before_label_join = sha256_file(_temporary_score_snapshot(method_scores, output))

        # Labels enter only here, after score, candidate, cluster, PCA, and UMAP state is frozen.
        experimental_labels = pd.to_numeric(experimental_labels.reindex(blind.obs_names), errors="coerce")
        display_budget = experimental_display_budget(experimental_labels, n_cells=len(method_scores))
        top_k_masks = common_top_k_masks(method_scores, top_k=int(display_budget["common_display_top_k"]))
        calls = apply_common_budget_candidate_classes(
            calls,
            method_scores["DuoDose"],
            top_k=int(display_budget["common_display_top_k"]),
        )
        display_audit = candidate_display_audit(dataset, calls, display_budget)
        status = pd.concat([_method_status(dataset, seed, duodose_result, rf_runtime), external_status], ignore_index=True, sort=False)
        status["n_finite_scores"] = status["method"].map({method: int(method_scores[method].notna().sum()) for method in method_scores}).fillna(0).astype(int)
        status["labeled_singlet_count"] = int(display_budget["labeled_singlet_count"])
        status["labeled_doublet_count"] = int(display_budget["labeled_doublet_count"])
        status["labeled_doublet_fraction"] = float(display_budget["labeled_doublet_fraction"])
        status["common_display_top_k"] = int(display_budget["common_display_top_k"])
        status["common_display_fraction"] = float(display_budget["common_display_fraction"])
        status["n_common_display_candidates"] = status["method"].map({method: int(top_k_masks[method].sum()) for method in top_k_masks}).fillna(0).astype(int)
        status["backend"] = status.get("backend", pd.Series(index=status.index, dtype=object)).fillna("external")
        status["internal_method_name"] = status.get("internal_method_name", pd.Series(index=status.index, dtype=object)).fillna("")
        status = status[["dataset", "seed", "method", "internal_method_name", "backend", "status", "message", "runtime_seconds", "n_finite_scores", "labeled_singlet_count", "labeled_doublet_count", "labeled_doublet_fraction", "common_display_top_k", "common_display_fraction", "n_common_display_candidates"]]

        diagnostics = local_diagnostics(blind, representation, clusters, safe_scores, k=int(app["diagnostic_neighbors"]))
        embedding_audit = shared_embedding_audit(coordinates, method_scores)
        labels_audit = label_usage_audit()
        labels_audit["score_table_sha256_before_label_join"] = score_hash_before_label_join
        ref_audit = reference_audit(detector, duodose_result)
        summary = candidate_summary(dataset, calls, display_budget)
        group_summary = summarize_diagnostics(dataset, diagnostics, calls)
        result = RealApplicationResult(
            coordinates=coordinates,
            method_scores=method_scores,
            candidate_calls=calls,
            method_status=status,
            label_usage_audit=labels_audit,
            reference_audit=ref_audit,
            shared_embedding_audit=embedding_audit,
            candidate_display_audit=display_audit,
            candidate_summary=summary,
            group_diagnostics=group_summary,
            diagnostics=diagnostics,
            panel_annotation=annotation,
            panel_annotation_name=annotation_name,
            duodose_result=duodose_result,
            display_budget=display_budget,
        )
        validate_figure_contract(result)

        coordinate_export = coordinates.join(clusters.rename("cluster")).join(annotation.rename("panel_annotation"))
        coordinate_export["panel_annotation_name"] = annotation_name
        _write_frame(coordinate_export, output / "real_application_umap_coordinates.csv.gz")
        score_export = method_scores.join(top_k_masks.add_suffix("_common_display_top_k"))
        _write_frame(score_export, output / "real_application_method_scores.csv.gz")
        _write_frame(calls, output / "real_application_candidate_calls.csv.gz")
        status.to_csv(output / "real_application_method_status.csv", index=False)
        labels_audit.to_csv(output / "real_application_label_usage_audit.csv", index=False)
        ref_audit.to_csv(output / "real_application_reference_audit.csv", index=False)
        embedding_audit.to_csv(output / "real_application_shared_embedding_audit.csv", index=False)
        display_audit.to_csv(output / "real_application_candidate_display_audit.csv", index=False)
        summary.to_csv(output / "real_application_candidate_summary.csv", index=False)
        group_summary.to_csv(output / "real_application_group_diagnostics.csv", index=False)

        reporter.event("rendering the exact 3 x 3 cross-method figure", dataset=dataset, seed=seed)
        plot_cross_method_umap(
            output / "real_application_cross_method_umap.png",
            output / "real_application_cross_method_umap.pdf",
            dataset=dataset,
            coordinates=coordinates,
            annotation=annotation,
            annotation_name=annotation_name,
            experimental_labels=experimental_labels,
            method_scores=method_scores,
            calls=calls,
            status=status,
            display_budget=display_budget,
            top_k_masks=top_k_masks,
        )
        plot_diagnostics(
            output / "real_application_duodose_diagnostics.png",
            output / "real_application_duodose_diagnostics.pdf",
            dataset=dataset,
            diagnostics=diagnostics,
            calls=calls,
        )
        runtime = time.perf_counter() - started
        write_run_manifest(
            output,
            workflow="real_data_application",
            protocol=protocol,
            dataset=dataset,
            seed=seed,
            runtime_seconds=runtime,
            source_path=loaded.source_path,
            extra={
                "figure_role": "qualitative descriptive real-data application",
                "backend": "rf",
                "internal_methods": ["DuoDose"],
                "external_methods": list(EXTERNAL_FIGURE_METHODS),
                "experimental_labels_joined_after_scores_frozen": True,
                "experimental_labels_evaluation_only": True,
                "construction_variant": "raw_sum_parents_removed",
                "safe_feature_mode": "fitted_reference",
                "parent_disjoint": True,
                "overall_candidate_rule": "canonical public expected-rate threshold",
                "overall_score_threshold": float(duodose_result.threshold),
                "subtype_homotypic_threshold": float(prediction["subtype_homotypic_threshold"]),
                "subtype_heterotypic_threshold": float(prediction["subtype_heterotypic_threshold"]),
                "candidate_class_column": "duodose_common_budget_candidate_class",
                "raw_candidate_class_column": "duodose_raw_model_candidate_class",
                **display_budget,
            },
        )
        _temporary_score_snapshot(method_scores, output).unlink(missing_ok=True)
        write_json(output / "figure_manifest.json", figure_manifest_payload(dataset, output, result))
        write_output_manifest(output)
        reporter.complete_unit(unit, message=f"completed real-data application for {dataset}")
        return status
    except Exception as exc:
        reporter.fail_unit(unit, exc)
        write_json(
            output / "failure.json",
            {"dataset": dataset, "seed": int(seed), "status": "failed", "message": str(exc), "traceback": traceback.format_exc()},
        )
        raise


def _temporary_score_snapshot(frame: pd.DataFrame, output: Path) -> Path:
    path = output / ".scores_before_label_join.csv"
    frame.to_csv(path, index=True)
    return path


def main() -> None:
    args = build_parser().parse_args()
    protocol = load_final_protocol(args.protocol)
    protocol_seed = int(protocol["seeds"]["real_application"][0])
    seed = protocol_seed if args.seed is None else int(args.seed)
    if seed not in set(map(int, protocol["seeds"]["real_application"])):
        raise ValueError(f"seed {seed} is not configured for real_data_application")
    output_root = validate_output_root(args.output_dir)
    conversion_dir = Path(args.conversion_dir).resolve() if args.conversion_dir else output_root / "converted_data"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = discover_dataset_manifest(args.data_dir)
    datasets = _selected_datasets(args, protocol, manifest)
    ledger_path, snapshot_path = progress_paths(output_root, args)
    reporter = ProgressReporter(
        stage="real_data_application",
        total_units=len(datasets),
        settings=ProgressSettings.from_args(args),
        ledger_path=ledger_path,
        snapshot_path=snapshot_path,
        config_hash=str(args.progress_config_hash or sha256_file(Path(protocol["_protocol_path"]))),
        output_path=output_root,
        results_dir=output_root.parent,
    )
    all_status: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    try:
        for dataset in datasets:
            run_dir = output_root / dataset / f"seed_{seed}"
            try:
                all_status.append(
                    _run_one(
                        args=args,
                        protocol=protocol,
                        dataset=dataset,
                        seed=seed,
                        output=run_dir,
                        conversion_dir=conversion_dir,
                        reporter=reporter,
                    )
                )
            except Exception as exc:
                failures.append({"dataset": dataset, "seed": seed, "status": "failed", "message": str(exc)})
                if not args.continue_on_error:
                    raise
            finally:
                (pd.concat(all_status, ignore_index=True) if all_status else pd.DataFrame(columns=["dataset", "seed", "method", "status", "message"])).to_csv(
                    output_root / "real_application_method_status_all.csv", index=False
                )
                pd.DataFrame(failures, columns=["dataset", "seed", "status", "message"]).to_csv(output_root / "real_application_failures.csv", index=False)
    finally:
        reporter.close()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
