"""Frozen public registry for the two manuscript DuoDose backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendSpec:
    alias: str
    internal_name: str
    display_name: str
    family: str
    role: str


BACKEND_SPECS = {
    "rf": BackendSpec(
        alias="rf",
        internal_name="DuoDose-ML-CalibratedRF-SafeFeatures",
        display_name="DuoDose",
        family="sklearn",
        role="default manuscript method",
    ),
    "dl": BackendSpec(
        alias="dl",
        internal_name="DuoDose-DL-ConditionalMultiTaskMLP-SafeFeatures",
        display_name="DuoDose-DL",
        family="conditional_dl",
        role="conditional multitask neural-network ablation",
    ),
}

DUODOSE_BACKENDS = {alias: spec.internal_name for alias, spec in BACKEND_SPECS.items()}
DEFAULT_DUODOSE_BACKEND = "rf"
PUBLIC_METHOD_NAMES = {spec.internal_name: spec.display_name for spec in BACKEND_SPECS.values()}


def get_backend_spec(backend: str) -> BackendSpec:
    try:
        return BACKEND_SPECS[str(backend).strip().lower()]
    except KeyError as exc:
        available = ", ".join(DUODOSE_BACKENDS)
        raise ValueError(f"Unknown DuoDose backend {backend!r}. Available backends: {available}") from exc


def public_method_name(internal_name: str) -> str:
    return PUBLIC_METHOD_NAMES.get(str(internal_name), str(internal_name))
