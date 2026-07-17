"""Portable discovery of the optional Rscript executable."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def find_rscript() -> str | None:
    """Return an explicit, PATH, or standard Windows Rscript installation."""

    explicit = os.environ.get("R_SCRIPT", "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return str(path.resolve())
        raise FileNotFoundError(f"R_SCRIPT does not point to an existing file: {path}")
    discovered = shutil.which("Rscript")
    if discovered:
        return discovered
    program_files = os.environ.get("ProgramFiles", "")
    if program_files:
        candidates = sorted(
            Path(program_files).glob("R/R-*/bin/Rscript.exe"),
            key=lambda path: path.parent.parent.name,
            reverse=True,
        )
        if candidates:
            return str(candidates[0].resolve())
    return None
