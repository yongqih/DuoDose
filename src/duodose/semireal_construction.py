"""Raw-count construction utilities for controlled semi-real doublets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import sparse


CONSTRUCTION_VARIANTS = {
    "raw_sum_parents_retained": ("raw_sum", "retained"),
    "raw_sum_parents_removed": ("raw_sum", "removed"),
    "downsampled_parents_retained": ("downsampled", "retained"),
    "downsampled_parents_removed": ("downsampled", "removed"),
}
DEFAULT_CONSTRUCTION_VARIANT = "raw_sum_parents_removed"


@dataclass(frozen=True)
class SemiRealConstructionConfig:
    variant: str
    count_construction_mode: str
    parent_reference_mode: str


def resolve_construction_config(
    construction_variant: str | None = None,
    count_construction_mode: str | None = None,
    parent_reference_mode: str | None = None,
) -> SemiRealConstructionConfig:
    """Resolve a named construction variant and optional independent overrides."""

    variant = str(construction_variant or DEFAULT_CONSTRUCTION_VARIANT)
    if variant not in CONSTRUCTION_VARIANTS:
        raise ValueError(f"Unknown construction variant {variant!r}; choose one of {', '.join(CONSTRUCTION_VARIANTS)}")
    default_count_mode, default_parent_mode = CONSTRUCTION_VARIANTS[variant]
    count_mode = str(count_construction_mode or default_count_mode)
    parent_mode = str(parent_reference_mode or default_parent_mode)
    if count_mode not in {"raw_sum", "downsampled"}:
        raise ValueError("count_construction_mode must be 'raw_sum' or 'downsampled'")
    if parent_mode not in {"retained", "removed"}:
        raise ValueError("parent_reference_mode must be 'retained' or 'removed'")
    resolved_variant = f"{count_mode}_parents_{parent_mode}"
    return SemiRealConstructionConfig(resolved_variant, count_mode, parent_mode)


def _as_raw_count_row(values) -> sparse.csr_matrix:
    row = values.tocsr(copy=True) if sparse.issparse(values) else sparse.csr_matrix(np.asarray(values))
    if row.shape[0] != 1:
        row = sparse.csr_matrix(row.reshape(1, -1))
    raw = row.data.astype(float, copy=False)
    if raw.size and (np.any(~np.isfinite(raw)) or np.any(raw < 0) or not np.allclose(raw, np.rint(raw), rtol=0.0, atol=1e-8)):
        raise ValueError("semi-real doublets require raw non-negative integer count vectors")
    row.data = np.rint(raw).astype(np.int64, copy=False)
    return row


@dataclass(frozen=True)
class EmpiricalUpperTailLibrarySampler:
    """Sample target UMI totals from an upper-tail reference-singlet distribution."""

    candidate_library_sizes: np.ndarray
    lower_quantile: float = 0.70
    upper_quantile: float = 0.995

    @classmethod
    def from_reference_counts(
        cls,
        reference_counts,
        *,
        lower_quantile: float = 0.70,
        upper_quantile: float = 0.995,
    ) -> "EmpiricalUpperTailLibrarySampler":
        if not 0.0 <= float(lower_quantile) <= float(upper_quantile) <= 1.0:
            raise ValueError("downsampled library quantiles must satisfy 0 <= lower <= upper <= 1")
        totals = np.asarray(reference_counts.sum(axis=1)).ravel().astype(float)
        totals = totals[np.isfinite(totals) & (totals > 0)]
        if totals.size == 0:
            raise ValueError("downsampled construction needs positive reference-singlet library sizes")
        low = float(np.quantile(totals, float(lower_quantile)))
        high = float(np.quantile(totals, float(upper_quantile)))
        candidates = np.rint(totals[(totals >= low) & (totals <= high)]).astype(np.int64)
        if candidates.size == 0:
            candidates = np.rint(totals).astype(np.int64)
        return cls(np.maximum(candidates, 1), float(lower_quantile), float(upper_quantile))

    def __call__(self, rng: np.random.Generator, raw_parent_sum_library_size: int) -> int:
        raw_total = int(raw_parent_sum_library_size)
        if raw_total < 1:
            raise ValueError("cannot construct a downsampled doublet with zero total parent counts")
        sampled = int(rng.choice(self.candidate_library_sizes))
        return int(min(raw_total, max(1, sampled)))


def construct_semireal_doublet(
    parent_1_counts,
    parent_2_counts,
    mode: str,
    target_library_sampler: Callable[[np.random.Generator, int], int] | None,
    rng: np.random.Generator,
    *,
    random_seed: int | None = None,
) -> tuple[sparse.csr_matrix, dict[str, object]]:
    """Construct one raw-count doublet and record its construction metadata."""

    if mode not in {"raw_sum", "downsampled"}:
        raise ValueError("mode must be 'raw_sum' or 'downsampled'")
    parent_1 = _as_raw_count_row(parent_1_counts)
    parent_2 = _as_raw_count_row(parent_2_counts)
    if parent_1.shape[1] != parent_2.shape[1]:
        raise ValueError("parent count vectors must have the same number of genes")
    combined = (parent_1 + parent_2).tocsr()
    raw_total = int(np.asarray(combined.sum()).item())
    if raw_total < 1:
        raise ValueError("cannot construct a semi-real doublet from zero-count parents")

    if mode == "raw_sum":
        target_total = raw_total
        synthetic = combined
    else:
        if target_library_sampler is None:
            raise ValueError("downsampled construction requires a target-library sampler")
        target_total = int(target_library_sampler(rng, raw_total))
        target_total = min(raw_total, max(1, target_total))
        probabilities = combined.data.astype(float) / float(raw_total)
        sampled = rng.multinomial(target_total, probabilities)
        synthetic = sparse.csr_matrix(
            (sampled.astype(np.int64), combined.indices.copy(), np.array([0, len(sampled)], dtype=np.int32)),
            shape=combined.shape,
        )
        synthetic.eliminate_zeros()

    metadata = {
        "raw_parent_sum_library_size": raw_total,
        "target_library_size": int(target_total),
        "retention_fraction": float(target_total / raw_total),
        "count_construction_mode": mode,
        "random_seed": int(random_seed) if random_seed is not None else np.nan,
    }
    return synthetic, metadata
