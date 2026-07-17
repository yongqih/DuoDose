"""Keep the README's public examples synchronized with the installed API."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = ROOT / "scripts" / "smoke_readme_examples.py"


def test_readme_examples() -> None:
    spec = importlib.util.spec_from_file_location("smoke_readme_examples", SMOKE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
