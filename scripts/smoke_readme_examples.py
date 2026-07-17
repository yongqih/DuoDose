"""Compile and contract-smoke every Python example in README.md."""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from duodose import DuoDose, DuoDoseResult
from duodose.cli import build_parser


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PYTHON_BLOCK = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\((docs/[^)#]+\.md)(?:#[^)]+)?\)")


class FakeAnnData:
    """Small AnnData-shaped fixture for documentation contract tests."""

    def __init__(self) -> None:
        self.X = np.arange(80, dtype=float).reshape(8, 10)
        self.layers = {"counts": self.X.copy()}
        self.obs_names = pd.Index([f"cell_{i}" for i in range(8)])
        self.obs = pd.DataFrame(index=self.obs_names)
        self.uns: dict[str, object] = {}
        self.written_path: str | None = None

    @property
    def n_obs(self) -> int:
        return self.X.shape[0]

    @property
    def n_vars(self) -> int:
        return self.X.shape[1]

    def write_h5ad(self, path: str) -> None:
        self.written_path = str(path)


def _fake_result(adata: FakeAnnData, backend: str) -> DuoDoseResult:
    homotypic = np.linspace(0.05, 0.40, adata.n_obs)
    heterotypic = np.linspace(0.02, 0.30, adata.n_obs)[::-1]
    overall = homotypic + heterotypic
    predicted = overall >= np.quantile(overall, 0.75)
    subtype = np.where(
        predicted,
        np.where(homotypic >= heterotypic, "homotypic_doublet", "heterotypic_doublet"),
        pd.NA,
    )
    confidence = np.where(predicted, np.maximum(homotypic, heterotypic) / overall, np.nan)
    scores = pd.DataFrame(
        {
            "duodose_score": overall,
            "duodose_homotypic_score": homotypic,
            "duodose_heterotypic_score": heterotypic,
            "predicted_doublet": predicted,
            "predicted_subtype": subtype,
            "subtype_confidence": confidence,
        },
        index=adata.obs_names,
    )
    return DuoDoseResult(
        scores=scores,
        threshold=float(np.quantile(overall, 0.75)),
        backend=backend,
        config={"backend": backend},
    )


def _read_h5ad(_path: str) -> FakeAnnData:
    return FakeAnnData()


def _smoke_python_blocks(readme: str) -> int:
    blocks = PYTHON_BLOCK.findall(readme)
    if not blocks:
        raise AssertionError("README.md contains no Python examples")

    fake_anndata = ModuleType("anndata")
    fake_anndata.read_h5ad = _read_h5ad  # type: ignore[attr-defined]
    previous_anndata = sys.modules.get("anndata")
    original_fit_predict = DuoDose.fit_predict

    def fake_fit_predict(self: DuoDose, adata: FakeAnnData) -> DuoDoseResult:
        return _fake_result(adata, self.config.backend)

    sys.modules["anndata"] = fake_anndata
    DuoDose.fit_predict = fake_fit_predict  # type: ignore[method-assign]
    try:
        for number, source in enumerate(blocks, start=1):
            code = compile(source, f"README.md:python-block-{number}", "exec")
            namespace = {
                "__name__": f"readme_example_{number}",
                "adata": FakeAnnData(),
                "DuoDose": DuoDose,
            }
            exec(code, namespace, namespace)
    finally:
        DuoDose.fit_predict = original_fit_predict  # type: ignore[method-assign]
        if previous_anndata is None:
            sys.modules.pop("anndata", None)
        else:
            sys.modules["anndata"] = previous_anndata
    return len(blocks)


def _check_public_constructor() -> None:
    documented = {
        "backend",
        "expected_doublet_rate",
        "random_state",
        "device",
        "layer",
        "threshold_strategy",
        "threshold",
        "training_preset",
        "amp",
        "dl_batch_size",
        "dl_max_epochs",
        "dl_patience",
        "config",
    }
    actual = set(inspect.signature(DuoDose).parameters)
    if actual != documented:
        raise AssertionError(
            "README constructor contract is stale: "
            f"missing from docs={sorted(actual - documented)}, "
            f"removed from API={sorted(documented - actual)}"
        )


def _check_cli_contract() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "input.h5ad",
            "--output",
            "input_duodose.h5ad",
            "--layer",
            "counts",
            "--backend",
            "dl",
            "--expected-doublet-rate",
            "0.08",
            "--training-preset",
            "default",
            "--device",
            "auto",
            "--seed",
            "0",
        ]
    )
    assert args.command == "run"
    assert args.layer == "counts"
    assert args.backend == "dl"


def _check_document_links(readme: str) -> None:
    links = set(MARKDOWN_LINK.findall(readme))
    expected = {
        "docs/quickstart.md",
        "docs/method.md",
        "docs/parameters.md",
        "docs/outputs.md",
        "docs/troubleshooting.md",
        "docs/reproducibility.md",
    }
    if links != expected:
        raise AssertionError(f"README documentation links differ: {sorted(links)}")
    missing = [path for path in links if not (ROOT / path).is_file()]
    if missing:
        raise AssertionError(f"README links to missing documentation: {missing}")


def main() -> None:
    readme = README.read_text(encoding="utf-8")
    _check_public_constructor()
    _check_cli_contract()
    _check_document_links(readme)
    count = _smoke_python_blocks(readme)
    print(f"README smoke test passed: {count} Python examples executed")


if __name__ == "__main__":
    main()
