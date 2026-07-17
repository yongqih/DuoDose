"""Compatibility helpers for selecting retained DuoDose backends."""

from __future__ import annotations

from collections.abc import Iterable

from .models.registry import DEFAULT_DUODOSE_BACKEND, DUODOSE_BACKENDS, PUBLIC_METHOD_NAMES, public_method_name

DUODOSE_METHOD_ALIASES = DUODOSE_BACKENDS
DUODOSE_METHODS = tuple(DUODOSE_BACKENDS.values())
DUODOSE_PUBLIC_METHOD_NAMES = PUBLIC_METHOD_NAMES


def _tokens(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    values = [value] if isinstance(value, str) else list(value)
    return [token.strip().lower() for item in values for token in str(item).split(",") if token.strip()]


def resolve_duodose_methods(value: str | Iterable[str] | None = None, *, default: str = "all") -> list[str]:
    tokens = _tokens(value) or _tokens(default)
    if "all" in tokens:
        if len(tokens) != 1:
            raise ValueError("'all' cannot be combined with individual DuoDose backends")
        return list(DUODOSE_METHODS)
    methods: list[str] = []
    for token in tokens:
        if token not in DUODOSE_BACKENDS:
            allowed = ", ".join(["all", *DUODOSE_BACKENDS])
            raise ValueError(f"Unsupported DuoDose backend {token!r}. Supported values: {allowed}")
        method = DUODOSE_BACKENDS[token]
        if method not in methods:
            methods.append(method)
    return methods


def resolve_duodose_cli_methods(*, method: str | None, methods: str | None, default: str = "all") -> list[str]:
    if method is not None and methods is not None:
        raise ValueError("Use either --duodose-method or --duodose-methods, not both")
    return resolve_duodose_methods(method if method is not None else methods, default=default)
