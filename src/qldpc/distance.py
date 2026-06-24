"""Code distance estimation for CSS codes.

The exact minimum distance of a qLDPC code is NP-hard in general. This module
provides an exact enumerator for small ``k`` and a randomized search that finds
low-weight logical operators (yielding an upper bound on the true distance).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DistanceReport:
    """Distance information available for a CSS code."""

    d_literature: int | None
    d_x_exact: int | None
    d_z_exact: int | None
    d_exact: int | None
    d_x_upper: int | None
    d_z_upper: int | None
    d_upper: int | None

    def summary(self) -> str:
        parts = []
        if self.d_literature is not None:
            parts.append(f"d_literature={self.d_literature}")
        if self.d_exact is not None:
            parts.append(f"d_exact={self.d_exact}")
        if self.d_upper is not None and self.d_exact is None:
            parts.append(f"d_upper={self.d_upper}")
        return ", ".join(parts) if parts else "distance unknown"


def _min_logical_weight_exact(logical_basis: np.ndarray, *, max_k: int) -> int | None:
    """Exact minimum weight over all nontrivial combinations of logical rows."""
    if logical_basis.size == 0:
        return None
    num_rows, num_cols = logical_basis.shape
    if num_rows == 0:
        return None
    if num_rows > max_k:
        return None

    best = num_cols + 1
    for mask in range(1, 1 << num_rows):
        combo = np.zeros(num_cols, dtype=np.uint8)
        for row in range(num_rows):
            if mask & (1 << row):
                combo ^= logical_basis[row]
        weight = int(combo.sum())
        if weight < best:
            best = weight
    return best if best <= num_cols else None


def _min_logical_weight_search(
    logical_basis: np.ndarray,
    *,
    trials: int,
    seed: int,
) -> int | None:
    """Randomized search for a low-weight logical operator (upper bound on distance)."""
    if logical_basis.size == 0 or logical_basis.shape[0] == 0:
        return None

    num_rows, num_cols = logical_basis.shape
    rng = np.random.default_rng(seed)
    best = num_cols + 1
    for _ in range(trials):
        mask = int(rng.integers(1, 1 << num_rows))
        combo = np.zeros(num_cols, dtype=np.uint8)
        for row in range(num_rows):
            if mask & (1 << row):
                combo ^= logical_basis[row]
        weight = int(combo.sum())
        if weight < best:
            best = weight
    return best if best <= num_cols else None


def distance_report(
    logical_x: np.ndarray,
    logical_z: np.ndarray,
    *,
    d_literature: int | None = None,
    max_k_exact: int = 12,
    search_trials: int = 4000,
    seed: int = 2026,
) -> DistanceReport:
    """Collect literature, exact, and search-based distance information."""
    d_x_exact = _min_logical_weight_exact(logical_x, max_k=max_k_exact)
    d_z_exact = _min_logical_weight_exact(logical_z, max_k=max_k_exact)
    d_exact = None
    if d_x_exact is not None and d_z_exact is not None:
        d_exact = min(d_x_exact, d_z_exact)

    d_x_upper = None if d_x_exact is not None else _min_logical_weight_search(
        logical_x, trials=search_trials, seed=seed
    )
    d_z_upper = None if d_z_exact is not None else _min_logical_weight_search(
        logical_z, trials=search_trials, seed=seed + 1
    )
    d_upper = None
    if d_x_upper is not None or d_z_upper is not None:
        candidates = [value for value in (d_x_upper, d_z_upper) if value is not None]
        d_upper = min(candidates) if candidates else None

    return DistanceReport(
        d_literature=d_literature,
        d_x_exact=d_x_exact,
        d_z_exact=d_z_exact,
        d_exact=d_exact,
        d_x_upper=d_x_upper,
        d_z_upper=d_z_upper,
        d_upper=d_upper,
    )
