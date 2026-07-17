from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from anndata import AnnData
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from duodose.protocol import load_final_protocol
from duodose.validation_suite import (
    ValidationSuiteError,
    atomic_write_csv,
    audit_chunking_invariance,
    audit_frozen_reference,
    audit_model_serialization,
    audit_order_invariance,
    audit_parent_disjoint,
    audit_run_status,
    audit_same_cell_features,
    audit_transformer_serialization,
    compare_frames,
    completed_run_is_reusable,
    domain_audit_contract_check,
    load_validation_config,
    make_fixture_adata,
    metric_contract_table,
    parent_pair_diagnostics,
    protocol_override_for_mode,
    schema_audit,
    stale_temporary_outputs,
    REQUIRED_OUTPUTS,
    validate_feature_names,
    validate_probability_frame,
)
from reproducibility.lib.common import LoadedDataset, run_protocol_models
from reproducibility.run_validation_suite import _permutation_summary, _validate_cli, build_parser


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "reproducibility" / "configs" / "final_validation_suite.yaml"
PROTOCOL = ROOT / "reproducibility" / "configs" / "final_protocol.yaml"


def _parent_map_run(rows: list[dict[str, str]]):
    frame = pd.DataFrame(rows)
    split_adatas = {}
    for split in ("train", "validation", "test"):
        ids = frame.loc[frame["split"].eq(split), "synthetic_cell_id"].drop_duplicates().tolist()
        matrix = sparse.csr_matrix(np.arange(max(1, len(ids)) * 3, dtype=float).reshape(max(1, len(ids)), 3))
        split_adatas[split] = AnnData(matrix[: len(ids)], obs=pd.DataFrame(index=ids), var=pd.DataFrame(index=["g0", "g1", "g2"]))
        split_adatas[split].layers["counts"] = split_adatas[split].X.copy()
    bundle = SimpleNamespace(
        parent_map=frame,
        fit_adata=split_adatas["train"],
        val_adata=split_adatas["validation"],
        test_adata=split_adatas["test"],
    )
    return SimpleNamespace(bundle=bundle)


def _pair_row(cell_id: str, split: str, parent_1: str, parent_2: str) -> dict[str, str]:
    return {
        "synthetic_cell_id": cell_id,
        "split": split,
        "synthetic_subtype": "heterotypic",
        "parent_1_id": parent_1,
        "parent_2_id": parent_2,
    }


@pytest.fixture(scope="module")
def validation_run():
    config = load_validation_config(CONFIG)
    mode = dict(config["modes"]["smoke"])
    protocol = protocol_override_for_mode(load_final_protocol(PROTOCOL), mode)
    adata = make_fixture_adata(mode["fixture_cells"], mode["fixture_genes"], seed=0)
    loaded = LoadedDataset("fixture", adata, Path("fixture"), "generated", "fixture", "not_applicable")
    return run_protocol_models(
        loaded,
        protocol_path=PROTOCOL,
        protocol_override=protocol,
        seed=0,
        backends=("rf", "dl"),
        device="cpu",
        amp=False,
        dl_max_epochs=3,
        dl_patience=2,
    )


def test_configuration_and_cli_contract():
    config = load_validation_config(CONFIG)
    assert config["contract"]["construction_variant"] == "raw_sum_parents_removed"
    assert config["contract"]["safe_feature_mode"] == "fitted_reference"
    args = build_parser().parse_args(["--mode", "smoke"])
    mode, subtype, full = _validate_cli(args, config)
    assert mode["fixture_cells"] >= 240
    assert (subtype, full) == (2, 2)
    bad = build_parser().parse_args(["--mode", "smoke", "--n-subtype-permutations", "3"])
    with pytest.raises(ValueError, match="at most 2"):
        _validate_cli(bad, config)
    full_args = build_parser().parse_args(["--mode", "full"])
    _, full_subtype, full_labels = _validate_cli(full_args, config)
    assert (full_subtype, full_labels) == (100, 100)


def test_permutation_metric_direction_is_explicit() -> None:
    results = pd.DataFrame(
        {
            "AUROC": [0.45, 0.50],
            "high_RNA_singlet_FPR": [0.30, 0.40],
            "contract_status": ["PASS", "PASS"],
        }
    )
    summary = _permutation_summary(results, {"AUROC": 0.80, "high_RNA_singlet_FPR": 0.10}, ["AUROC", "high_RNA_singlet_FPR"]).set_index("metric")
    assert summary.loc["AUROC", "metric_direction"] == "higher_is_better"
    assert summary.loc["high_RNA_singlet_FPR", "metric_direction"] == "lower_is_better"
    assert summary.loc["AUROC", "empirical_p_value"] == pytest.approx(1 / 3)
    assert summary.loc["high_RNA_singlet_FPR", "empirical_p_value"] == pytest.approx(1 / 3)
    with pytest.raises(ValidationSuiteError, match="direction is not declared"):
        _permutation_summary(results, {"unknown": 0.5}, ["unknown"])


def test_parent_disjoint_parent_removal_and_frozen_reference(validation_run):
    audit, membership, parent_map = audit_parent_disjoint(validation_run)
    assert audit["status"].eq("PASS").all()
    assert not membership.empty and not parent_map.empty
    assert validation_run.bundle.construction_report["construction_variant"] == "raw_sum_parents_removed"
    reference = audit_frozen_reference(validation_run)
    assert not reference["status"].eq("FAIL").any()
    assert reference.loc[reference["check"].eq("experimental_doublets_in_reference"), "status"].iloc[0] == "NOT_APPLICABLE"


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (
            [_pair_row("d0", "train", "A", "B"), _pair_row("d1", "train", "B", "A")],
            {"n_reversed_order_equivalent_pairs": 1, "n_canonical_duplicate_pairs": 1},
        ),
        (
            [_pair_row("d0", "train", "A", "B"), _pair_row("d0", "train", "A", "B")],
            {"n_duplicate_parent_map_rows": 1, "n_duplicate_generated_cell_ids": 1},
        ),
        (
            [_pair_row("d0", "train", "A", "B"), _pair_row("d0", "train", "C", "D")],
            {"n_duplicate_generated_cell_ids": 1},
        ),
        (
            [_pair_row("d0", "train", "A", "B"), _pair_row("d1", "train", "A", "B")],
            {"n_raw_ordered_duplicate_pairs": 1, "n_canonical_duplicate_pairs": 1},
        ),
        (
            [_pair_row("d0", "train", "A", "B"), _pair_row("d1", "test", "B", "A")],
            {"n_cross_split_canonical_pair_overlaps": 1, "n_cross_split_parent_overlaps": 2},
        ),
    ],
)
def test_parent_pair_duplicate_semantics(rows, expected):
    diagnostics, canonical = parent_pair_diagnostics(_parent_map_run(rows))
    assert {"canonical_parent_1", "canonical_parent_2"}.issubset(canonical.columns)
    for key, value in expected.items():
        assert diagnostics[key] == value


def test_all_fixed_inference_invariance_audits(validation_run, tmp_path):
    tolerance = 1e-6
    same = audit_same_cell_features(validation_run, tolerance)
    order = audit_order_invariance(validation_run, tolerance)
    chunks = audit_chunking_invariance(validation_run, [64, 113, 257], tolerance)
    transformer = audit_transformer_serialization(validation_run, tmp_path, tolerance)
    models = audit_model_serialization(validation_run, tmp_path, tolerance)
    for frame in (same, order, chunks, transformer, models):
        assert frame["status"].eq("PASS").all()
        assert frame["maximum_absolute_difference"].max() < tolerance
    assert set(models["context"]) == {"DuoDose", "DuoDose-DL"}


def test_leakage_probability_metric_and_schema_contracts(validation_run):
    allowlist = validation_run.protocol["features"]["allowlist"]
    leakage = validate_feature_names(validation_run.transformer.model_feature_columns_, allowlist)
    assert leakage["status"].eq("PASS").all()
    for method, frame in validation_run.method_probabilities_test.items():
        assert validate_probability_frame(frame, method, 1e-6)["status"].eq("PASS").all()
    metrics = metric_contract_table()
    assert set(["AUROC", "overall_AUPRC", "homotypic_AUPRC", "heterotypic_AUPRC"]).issubset(metrics["metric"])
    schema = schema_audit(validation_run.original_adata, validation_run)
    assert schema["status"].eq("PASS").all()


def test_atomic_output_and_interrupted_temp_detection(tmp_path):
    target = tmp_path / "audit.csv"
    atomic_write_csv(target, pd.DataFrame({"value": [1]}))
    assert target.is_file() and not (tmp_path / "audit.csv.tmp").exists()
    interrupted = tmp_path / "broken.csv.tmp"
    interrupted.write_text("partial", encoding="utf-8")
    assert stale_temporary_outputs(tmp_path) == [interrupted]


def test_resume_requires_matching_hash_and_complete_atomic_outputs(tmp_path):
    for relative in REQUIRED_OUTPUTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("nonempty", encoding="utf-8")
    (tmp_path / ".validation_suite_complete.json").write_text('{"run_hash": "expected"}', encoding="utf-8")
    assert completed_run_is_reusable(tmp_path, "expected")
    assert not completed_run_is_reusable(tmp_path, "different")
    (tmp_path / "validation_suite_checks.csv.tmp").write_text("partial", encoding="utf-8")
    assert not completed_run_is_reusable(tmp_path, "expected")


def test_run_status_explicitly_marks_interruption(tmp_path):
    protocol = load_final_protocol(PROTOCOL)
    frame = audit_run_status(tmp_path, protocol)
    expected = (
        len(protocol["datasets"]["real_doublet_enriched"]) * len(protocol["seeds"]["controlled_benchmark"]) * 6
        + len(protocol["datasets"]["real_application"]) * len(protocol["seeds"]["real_application"]) * 5
    )
    assert len(frame) == expected
    assert frame["status"].eq("INCOMPLETE").all()
    assert frame["reason"].str.len().gt(0).all()
    assert not frame.duplicated(["workflow", "dataset", "seed", "method"]).any()
    assert set(frame["workflow"]) == {"controlled_benchmark", "real_data_application"}


def test_missing_domain_audit_manifest_is_incomplete(tmp_path):
    (tmp_path / "domain_audit_all_datasets_summary.csv").write_text("dataset,status\ncline-ch,PASS\n", encoding="utf-8")
    audit = domain_audit_contract_check(tmp_path)
    assert audit.iloc[0]["status"] == "INCOMPLETE"
    assert "run-status ledger" in audit.iloc[0]["reason"]


def test_deliberate_parent_overlap_cannot_pass(validation_run):
    original = validation_run.bundle.parent_map.copy()
    try:
        train_parent = original.loc[original["split"].eq("train"), "parent_1_id"].iloc[0]
        index = original.index[original["split"].eq("validation")][0]
        validation_run.bundle.parent_map.loc[index, "parent_1_id"] = train_parent
        audit, _, _ = audit_parent_disjoint(validation_run)
        assert audit.loc[audit["check"].eq("train_validation_parent_overlap"), "status"].iloc[0] == "FAIL"
    finally:
        validation_run.bundle.parent_map = original


def test_deliberate_reference_leakage_and_fingerprint_mismatch_cannot_pass(validation_run):
    original_ids = validation_run.transformer.reference_cell_ids_.copy()
    original_transformer = validation_run.validation_scores["safe_feature_transformer_id"].copy()
    original_reference = validation_run.validation_scores["safe_feature_reference_pool_id"].copy()
    try:
        validation_run.transformer.reference_cell_ids_ = original_ids.append(pd.Index([validation_run.bundle.val_adata.obs_names[0]]))
        assert audit_frozen_reference(validation_run).loc[lambda frame: frame["check"].eq("validation_cells_in_reference"), "status"].iloc[0] == "FAIL"
        validation_run.transformer.reference_cell_ids_ = original_ids
        validation_run.validation_scores.loc[:, "safe_feature_transformer_id"] = "mismatched_transformer"
        validation_run.validation_scores.loc[:, "safe_feature_reference_pool_id"] = "mismatched_reference"
        audit = audit_frozen_reference(validation_run)
        assert audit.loc[audit["check"].eq("validation_provenance_identical"), "status"].iloc[0] == "FAIL"
    finally:
        validation_run.transformer.reference_cell_ids_ = original_ids
        validation_run.validation_scores.loc[:, "safe_feature_transformer_id"] = original_transformer
        validation_run.validation_scores.loc[:, "safe_feature_reference_pool_id"] = original_reference


def test_deliberate_prohibited_feature_and_malformed_probability_cannot_pass(validation_run):
    prohibited = validate_feature_names([*validation_run.transformer.model_feature_columns_, "scrublet_score"], validation_run.protocol["features"]["allowlist"])
    assert prohibited.loc[prohibited["feature"].eq("scrublet_score"), "status"].iloc[0] == "FAIL"
    malformed = validation_run.method_probabilities_test["DuoDose"].copy()
    malformed.iloc[0, 0] = np.nan
    malformed.iloc[1, 1] = 1.5
    audit = validate_probability_frame(malformed, "DuoDose", 1e-6)
    assert audit["status"].eq("FAIL").any()


def test_ambiguous_pre_model_score_without_provenance_cannot_pass(validation_run):
    feature = "handcrafted_unregistered_mystery_score"
    audit = validate_feature_names([feature], [feature])
    assert audit.iloc[0]["status"] == "FAIL"
    assert "missing semantic provenance metadata" in audit.iloc[0]["exclusion_reason"]


def test_deliberate_duplicate_cell_id_and_frame_mismatch_raise(validation_run):
    duplicate = validation_run.bundle.test_adata[:3, :].copy()
    duplicate.obs_names = ["duplicate", "duplicate", "third"]
    with pytest.raises(ValueError, match="unique query cell IDs"):
        validation_run.transformer.transform(duplicate)
    left = pd.DataFrame({"a": [1.0]}, index=["cell"])
    right = pd.DataFrame({"b": [1.0]}, index=["cell"])
    with pytest.raises(ValidationSuiteError, match="ordering differs"):
        compare_frames(left, right, audit="fingerprint", context="mismatch", tolerance=1e-6)
