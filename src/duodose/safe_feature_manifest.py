"""Central SafeFeature dependency manifest and group-ablation resolver."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


DEPENDENCY_GRAPH_VERSION = "safe_feature_dependency_graph_v2"
FEATURE_PROVENANCE_VERSION = "safe_feature_provenance_v1"
LEGACY_FEATURE_NAME_MAP = {
    "duodose_homotypic_score": "handcrafted_homotypic_score",
    "duodose_heterotypic_score": "handcrafted_identity_mixture_score",
    "duodose_score": "handcrafted_combined_score",
    "duodose_sensitive_score": "handcrafted_sensitive_score",
    "duodose_conservative_raw_score": "handcrafted_dosage_raw_score",
    "duodose_conservative_rank_score": "handcrafted_dosage_reference_ecdf",
    "duodose_conservative_tail_score": "handcrafted_dosage_reference_tail",
    "homotypic_score": "handcrafted_homotypic_reference_score",
    "heterotypic_score": "handcrafted_artificial_doublet_neighbor_score",
    "homotypic_rank_score": "handcrafted_homotypic_reference_ecdf",
    "heterotypic_rank_score": "handcrafted_artificial_doublet_neighbor_ecdf",
    "homotypic_tail_score": "handcrafted_homotypic_reference_tail",
    "heterotypic_tail_score": "handcrafted_artificial_doublet_neighbor_tail",
    "duodose_score_raw": "handcrafted_combined_raw_score",
    "duodose_score_rank_calibrated": "handcrafted_combined_reference_ecdf",
    "duodose_score_tail_calibrated": "handcrafted_combined_reference_tail",
    "legacy_heterotypic_score": "handcrafted_artificial_doublet_compatible_score",
    "homotypic_candidate_score": "handcrafted_homotypic_candidate_score",
    "homotypic_final_score": "handcrafted_homotypic_dosage_score",
    "duodose_score_combined": "handcrafted_combined_score_legacy_alias",
    "duodose_score_max": "handcrafted_sensitive_max_score",
    "duodose_gated_inlier_score": "handcrafted_dosage_gated_inlier_score",
}
FEATURE_ABLATION_MODES = (
    "none",
    "remove_dosage",
    "remove_identity_mixture",
    "remove_cluster_relative",
    "remove_scrublet_derived",
)

_CLUSTER_REFERENCE_FEATURES = {
    "cluster_nCount_z",
    "cluster_count_robust_z",
    "cluster_gene_robust_z",
    "cluster_stable_dosage_robust_z",
    "cluster_marker_dosage_robust_z",
    "cluster_abundance",
    "cluster_level_expected_homotypic_burden",
}

# Raw dependency nodes name the source calculation without exposing benchmark
# labels.  Composite SafeFeatures inherit these source groups recursively.
_RAW_SOURCE_GROUPS: dict[str, frozenset[str]] = {
    # Count and detected-gene complexity are direct library-complexity dosage
    # sources even when they also remain row-local measurements.
    "raw:ncount": frozenset({"row_local", "dosage"}),
    "raw:nfeature": frozenset({"row_local", "dosage"}),
    "raw:benchmark_cluster": frozenset({"cluster_relative"}),
    "raw:categorical": frozenset({"categorical", "cluster_relative"}),
    "raw:dosage_module": frozenset({"dosage", "cluster_relative"}),
    "raw:identity_module": frozenset({"identity_mixture", "cluster_relative"}),
    "raw:scrublet": frozenset({"scrublet_derived", "identity_mixture"}),
}

_DOSAGE_PRIMITIVE_FEATURES = frozenset(
    {
        "nCount",
        "log_nCount",
        "library_complexity_balance",
        "cluster_nCount_z",
        "cluster_count_robust_z",
        "cluster_gene_robust_z",
        "cluster_stable_dosage_robust_z",
        "cluster_marker_dosage_robust_z",
        "dosage_outlier_score",
        "uniform_dosage_inflation_score",
        "dosage_residual",
    }
)


def _node(
    primary_group: str,
    direct_dependencies: tuple[str, ...],
    *,
    is_composite: bool,
) -> dict[str, object]:
    return {
        "primary_group": primary_group,
        "direct_dependencies": direct_dependencies,
        "is_composite": bool(is_composite),
    }


# Every fixed SafeFeature has an explicit dependency declaration.  This is the
# only source used for scientific group-ablation membership; benchmark scripts
# never infer it from feature-name substrings.
_FEATURE_NODES: dict[str, dict[str, object]] = {
    "nCount": _node("row_local", ("raw:ncount",), is_composite=False),
    "log_nCount": _node("row_local", ("nCount",), is_composite=True),
    "library_complexity_balance": _node(
        "dosage",
        ("raw:nfeature", "raw:ncount"),
        is_composite=True,
    ),
    "scrublet_score": _node("scrublet_derived", ("raw:scrublet",), is_composite=False),
    "handcrafted_homotypic_score": _node(
        "dosage",
        ("raw:dosage_module",),
        is_composite=False,
    ),
    "handcrafted_identity_mixture_score": _node(
        "identity_mixture",
        ("raw:identity_module",),
        is_composite=False,
    ),
    "handcrafted_combined_score": _node(
        "composite",
        ("raw:dosage_module", "raw:scrublet"),
        is_composite=True,
    ),
    "handcrafted_sensitive_score": _node(
        "composite",
        ("raw:dosage_module", "raw:scrublet"),
        is_composite=True,
    ),
    "handcrafted_dosage_raw_score": _node(
        "dosage",
        ("raw:dosage_module",),
        is_composite=False,
    ),
    "handcrafted_dosage_reference_ecdf": _node(
        "composite",
        ("handcrafted_dosage_raw_score",),
        is_composite=True,
    ),
    "handcrafted_dosage_reference_tail": _node(
        "composite",
        ("handcrafted_dosage_raw_score",),
        is_composite=True,
    ),
    "hybrid_overall_score": _node(
        "composite",
        ("scrublet_score", "handcrafted_homotypic_score"),
        is_composite=True,
    ),
    "hybrid_homotypic_score": _node(
        "composite",
        ("handcrafted_homotypic_score",),
        is_composite=True,
    ),
    "hybrid_heterotypic_score": _node(
        "composite",
        ("scrublet_score", "handcrafted_identity_mixture_score"),
        is_composite=True,
    ),
    "cluster_nCount_z": _node(
        "cluster_relative",
        ("log_nCount", "raw:benchmark_cluster"),
        is_composite=True,
    ),
    "handcrafted_homotypic_reference_score": _node("composite", ("handcrafted_homotypic_score",), is_composite=True),
    "handcrafted_artificial_doublet_neighbor_score": _node(
        "scrublet_derived",
        ("raw:scrublet",),
        is_composite=False,
    ),
    "handcrafted_homotypic_reference_ecdf": _node("composite", ("handcrafted_homotypic_reference_score",), is_composite=True),
    "handcrafted_artificial_doublet_neighbor_ecdf": _node("composite", ("handcrafted_artificial_doublet_neighbor_score",), is_composite=True),
    "handcrafted_homotypic_reference_tail": _node("composite", ("handcrafted_homotypic_reference_score",), is_composite=True),
    "handcrafted_artificial_doublet_neighbor_tail": _node("composite", ("handcrafted_artificial_doublet_neighbor_score",), is_composite=True),
    "handcrafted_combined_raw_score": _node("composite", ("handcrafted_combined_score",), is_composite=True),
    "handcrafted_combined_reference_ecdf": _node("composite", ("handcrafted_combined_score",), is_composite=True),
    "handcrafted_combined_reference_tail": _node("composite", ("handcrafted_combined_score",), is_composite=True),
    "handcrafted_artificial_doublet_compatible_score": _node("composite", ("handcrafted_artificial_doublet_neighbor_score",), is_composite=True),
    "dosage_outlier_score": _node("dosage", ("raw:dosage_module",), is_composite=False),
    "identity_inlier_score": _node("identity_mixture", ("raw:identity_module",), is_composite=False),
    "uniform_dosage_inflation_score": _node("dosage", ("raw:dosage_module",), is_composite=False),
    "biological_program_coherence_score": _node(
        "composite",
        ("identity_inlier_score",),
        is_composite=True,
    ),
    "handcrafted_homotypic_candidate_score": _node("composite", ("handcrafted_sensitive_score",), is_composite=True),
    "handcrafted_homotypic_dosage_score": _node("composite", ("handcrafted_homotypic_score",), is_composite=True),
    "module_residual_rank_mean": _node("composite", ("dosage_outlier_score",), is_composite=True),
    "module_residual_rank_spread": _node(
        "composite",
        ("handcrafted_homotypic_reference_ecdf", "handcrafted_artificial_doublet_neighbor_ecdf"),
        is_composite=True,
    ),
    "cluster_count_robust_z": _node("composite", ("cluster_nCount_z",), is_composite=True),
    "cluster_gene_robust_z": _node(
        "cluster_relative",
        ("raw:nfeature", "raw:benchmark_cluster"),
        is_composite=True,
    ),
    "cluster_stable_dosage_robust_z": _node("composite", ("cluster_nCount_z",), is_composite=True),
    "cluster_marker_dosage_robust_z": _node("composite", ("cluster_nCount_z",), is_composite=True),
    "dosage_residual": _node("dosage", ("raw:dosage_module",), is_composite=False),
    "cluster_abundance": _node("cluster_relative", ("raw:benchmark_cluster",), is_composite=True),
    "cluster_level_expected_homotypic_burden": _node(
        "composite",
        ("cluster_abundance",),
        is_composite=True,
    ),
}


def _reference_feature_group(feature: str) -> tuple[str, str]:
    """Preserve the existing SafeFeature provenance grouping."""

    lower = feature.lower()
    if feature in {"nCount", "log_nCount", "library_complexity_balance"}:
        return "row_local", "none"
    if feature in _CLUSTER_REFERENCE_FEATURES:
        return "cluster_reference", "reference_pca_cluster_statistics"
    if feature.startswith(("duodose_cluster_", "sample_id_")):
        return "categorical_reference", "fit_only_categorical_mapping"
    if "scrublet" in lower or "heterotypic" in lower:
        return "artificial_doublet_reference", "fixed_reference_and_artificial_doublet_knn"
    if "identity" in lower or "biological_program" in lower:
        return "neighbor_reference", "fixed_reference_knn"
    if "rank" in lower or "tail" in lower:
        return "reference_ecdf", "reference_ecdf_and_tail_threshold"
    return "frozen_score_formula", "reference_cluster_statistics_and_ecdf"


def _node_spec(feature: str) -> dict[str, object]:
    if feature in _FEATURE_NODES:
        return _FEATURE_NODES[feature]
    if feature.startswith(("duodose_cluster_", "sample_id_")):
        return _node("categorical", ("raw:categorical",), is_composite=False)
    reference_group, _ = _reference_feature_group(feature)
    if reference_group == "row_local":
        return _node("row_local", ("raw:ncount",), is_composite=False)
    if reference_group == "cluster_reference":
        return _node("cluster_relative", ("raw:benchmark_cluster",), is_composite=False)
    if reference_group == "artificial_doublet_reference":
        return _node("scrublet_derived", ("raw:scrublet",), is_composite=False)
    if reference_group == "neighbor_reference":
        return _node("identity_mixture", ("raw:identity_module",), is_composite=False)
    return _node("unclassified", (), is_composite=False)


def _source_groups(feature: str, *, visiting: set[str] | None = None) -> frozenset[str]:
    if feature in _RAW_SOURCE_GROUPS:
        return _RAW_SOURCE_GROUPS[feature]
    active = set() if visiting is None else set(visiting)
    if feature in active:
        raise ValueError(f"cycle in SafeFeature dependency graph at {feature!r}")
    active.add(feature)
    spec = _node_spec(feature)
    groups: set[str] = set()
    for dependency in spec["direct_dependencies"]:
        groups.update(_source_groups(str(dependency), visiting=active))
    primary_group = str(spec["primary_group"])
    if primary_group in {"dosage", "identity_mixture", "cluster_relative", "scrublet_derived", "row_local", "categorical"}:
        groups.add(primary_group)
    return frozenset(groups)


def _split_manifest_values(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item) for item in value if str(item)]
    return [item for item in str(value or "").split(";") if item]


def migrate_legacy_feature_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with historical pre-model names migrated to canonical names."""

    output = frame.copy()
    for legacy, canonical in LEGACY_FEATURE_NAME_MAP.items():
        if legacy not in output:
            continue
        if canonical in output:
            left = pd.to_numeric(output[legacy], errors="coerce")
            right = pd.to_numeric(output[canonical], errors="coerce")
            if not left.equals(right):
                raise ValueError(f"historical feature {legacy!r} conflicts with canonical feature {canonical!r}")
            output = output.drop(columns=[legacy])
        else:
            output = output.rename(columns={legacy: canonical})
    return output


def safe_feature_provenance(feature: str) -> dict[str, object]:
    """Return explicit semantic provenance used by leakage audits."""

    name = str(feature)
    spec = _node_spec(name)
    source_groups = set(_source_groups(name))
    known = name in _FEATURE_NODES or name.startswith(("duodose_cluster_", "sample_id_"))
    legacy_names = [legacy for legacy, canonical in LEGACY_FEATURE_NAME_MAP.items() if canonical == name]
    if name.startswith("handcrafted_"):
        category = "pre_model_handcrafted_score"
    elif "identity_mixture" in source_groups:
        category = "identity_feature"
    elif "dosage" in source_groups:
        category = "dosage_feature"
    elif name in {"nCount", "log_nCount"} or "cluster_relative" in source_groups:
        category = "technical_covariate"
    elif name.startswith(("duodose_cluster_", "sample_id_")):
        category = "metadata"
    else:
        category = "raw_mechanism_feature" if known else "prohibited_feature"
    prohibited_source = bool(name == "scrublet_score" or name.startswith("hybrid_"))
    allowed = bool(known and not prohibited_source and category != "prohibited_feature")
    return {
        "canonical_feature_name": name,
        "legacy_feature_name": ";".join(legacy_names),
        "public_display_name": name.replace("_", " ").replace("handcrafted ", "Pre-model handcrafted ").strip(),
        "category": category,
        "source_function": "SafeFeatureTransformer._feature_frame",
        "source_file": "src/duodose/safe_feature_transformer.py",
        "computed_before_model_fit": bool(known),
        "uses_truth_labels": False if known else None,
        "uses_model_output": False if known else None,
        "uses_outcome_calibration": False if known else None,
        "uses_dataset_rank": False if known else None,
        "deterministic_from_counts_and_frozen_reference": bool(known),
        "allowed_in_rf": allowed,
        "allowed_in_dl": allowed,
        "feature_version": FEATURE_PROVENANCE_VERSION,
        "provenance_complete": bool(known),
    }


def build_safe_feature_manifest(feature_columns: Iterable[object]) -> pd.DataFrame:
    """Build the canonical source/dependency manifest for a model matrix."""

    rows: list[dict[str, object]] = []
    for column in feature_columns:
        feature = str(column)
        reference_group, reference_state = _reference_feature_group(feature)
        spec = _node_spec(feature)
        source_groups = sorted(_source_groups(feature))
        direct_dependencies = [str(item) for item in spec["direct_dependencies"]]
        dosage_annotation = (
            "primitive"
            if feature in _DOSAGE_PRIMITIVE_FEATURES
            else "composite"
            if "dosage" in source_groups and bool(spec["is_composite"])
            else ""
        )
        provenance = safe_feature_provenance(feature)
        rows.append(
            {
                **provenance,
                "feature_name": feature,
                "feature_group": reference_group,
                "primary_group": str(spec["primary_group"]),
                "source_groups": ";".join(source_groups),
                "is_composite": bool(spec["is_composite"]),
                "direct_dependencies": ";".join(direct_dependencies),
                "dosage_annotation": dosage_annotation,
                # Retained for compatibility with the first manifest-driven
                # ablation implementation; source_groups is now authoritative.
                "ablation_groups": ";".join(source_groups),
                "dependency_graph_version": DEPENDENCY_GRAPH_VERSION,
                "row_local": bool(reference_group == "row_local"),
                "reference_fitted": bool(reference_group != "row_local"),
                "deterministic": True,
                "transform_supported": True,
                "reference_state_used": reference_state,
                "legacy_context_dependent": bool(feature not in {"nCount", "log_nCount", "library_complexity_balance"}),
                "included_in_fitted_mode": True,
                "exclusion_reason": "",
            }
        )
    return pd.DataFrame(rows)


def resolve_feature_ablation(
    mode: str,
    feature_columns: Iterable[object],
    *,
    manifest: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Resolve one named ablation from transitive manifest source groups."""

    if mode not in FEATURE_ABLATION_MODES:
        raise ValueError(f"feature ablation must be one of: {', '.join(FEATURE_ABLATION_MODES)}")
    original_features = [str(column) for column in feature_columns]
    canonical_manifest = build_safe_feature_manifest(original_features)
    if manifest is not None and not manifest.empty and "feature_name" in manifest:
        existing = manifest.copy()
        existing["feature_name"] = existing["feature_name"].astype(str)
        existing = existing.drop_duplicates("feature_name", keep="last").set_index("feature_name")
        canonical_manifest = canonical_manifest.set_index("feature_name")
        for column in (
            "feature_group",
            "row_local",
            "reference_fitted",
            "deterministic",
            "transform_supported",
            "reference_state_used",
            "legacy_context_dependent",
            "included_in_fitted_mode",
            "exclusion_reason",
        ):
            if column in existing:
                common = canonical_manifest.index.intersection(existing.index)
                canonical_manifest.loc[common, column] = existing.loc[common, column].to_numpy()
        canonical_manifest = canonical_manifest.reset_index()
    target_group = "" if mode == "none" else mode.removeprefix("remove_")
    source_memberships = canonical_manifest["source_groups"].map(_split_manifest_values)
    primary_groups = canonical_manifest["primary_group"].astype(str)
    removed_mask = source_memberships.map(lambda groups: bool(target_group and target_group in groups))
    directly_removed_features = canonical_manifest.loc[
        removed_mask & primary_groups.eq(target_group), "feature_name"
    ].astype(str).tolist()
    dependency_removed_features = canonical_manifest.loc[
        removed_mask & ~primary_groups.eq(target_group), "feature_name"
    ].astype(str).tolist()
    removed_features = [*directly_removed_features, *dependency_removed_features]
    if mode != "none" and not removed_features:
        raise ValueError(f"SafeFeature dependency manifest contains no features for group {target_group!r}")
    removed_set = set(removed_features)
    retained_features = [feature for feature in original_features if feature not in removed_set]
    if not retained_features:
        raise ValueError("feature ablation would remove every SafeFeature")
    retained_dependencies = canonical_manifest.loc[
        canonical_manifest["feature_name"].astype(str).isin(retained_features),
        ["feature_name", "source_groups"],
    ]
    leaked_retained_features = retained_dependencies.loc[
        retained_dependencies["source_groups"].map(lambda groups: target_group in _split_manifest_values(groups)),
        "feature_name",
    ].astype(str).tolist()
    if target_group and leaked_retained_features:
        raise AssertionError(
            f"retained SafeFeatures still depend on ablated group {target_group!r}: "
            + ", ".join(leaked_retained_features)
        )
    dosage_memberships = canonical_manifest["source_groups"].map(_split_manifest_values)
    dosage_primitive_features = canonical_manifest.loc[
        canonical_manifest["dosage_annotation"].astype(str).eq("primitive"), "feature_name"
    ].astype(str).tolist()
    dosage_derived_composite_features = canonical_manifest.loc[
        dosage_memberships.map(lambda groups: "dosage" in groups)
        & canonical_manifest["is_composite"].astype(bool)
        & ~canonical_manifest["dosage_annotation"].astype(str).eq("primitive"),
        "feature_name",
    ].astype(str).tolist()
    retained_dosage_features = canonical_manifest.loc[
        canonical_manifest["feature_name"].astype(str).isin(retained_features)
        & dosage_memberships.map(lambda groups: "dosage" in groups),
        "feature_name",
    ].astype(str).tolist()
    if target_group == "dosage" and retained_dosage_features:
        raise AssertionError(
            "remove_dosage retained direct or inherited dosage information: "
            + ", ".join(retained_dosage_features)
        )
    return {
        "mode": mode,
        "group": target_group or "none",
        "dependency_graph_version": DEPENDENCY_GRAPH_VERSION,
        "manifest": canonical_manifest,
        "original_features": original_features,
        "directly_removed_features": directly_removed_features,
        "dependency_removed_features": dependency_removed_features,
        "removed_features": removed_features,
        "retained_features": retained_features,
        "no_retained_feature_depends_on_ablated_group": True,
        "retained_dependency_violations": leaked_retained_features,
        "dosage_primitive_features": dosage_primitive_features,
        "dosage_derived_composite_features": dosage_derived_composite_features,
        "retained_dosage_features": retained_dosage_features,
        "no_retained_feature_contains_dosage_information": not retained_dosage_features,
    }
