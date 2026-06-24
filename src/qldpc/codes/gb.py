"""Generalized bicycle (GB) code construction.

Generalized bicycle codes (Panteleev and Kalachev, 2021) are the one-dimensional
analogue of bivariate bicycle codes. Over a cyclic group of order ``n`` they are
built from two circulant matrices ``A = a(S)`` and ``B = b(S)``, where ``S`` is
the cyclic shift and ``a``, ``b`` are polynomials given by their nonzero
exponents. The CSS checks are ``H_X = [A | B]`` and ``H_Z = [B^T | A^T]``;
circulants commute, so the CSS condition holds automatically.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .css import CSSCode


def _circulant(n: int, exponents: Sequence[int]) -> np.ndarray:
    """Circulant ``sum_k S^k`` over the cyclic group of order ``n``."""
    shift = np.roll(np.eye(n, dtype=np.uint8), 1, axis=1)
    total = np.zeros((n, n), dtype=np.uint8)
    for exponent in exponents:
        power = np.eye(n, dtype=np.uint8)
        for _ in range(exponent % n):
            power = (power @ shift) & 1
        total ^= power
    return total


def build_generalized_bicycle(
    *,
    n: int,
    a_exponents: Sequence[int],
    b_exponents: Sequence[int],
    name: str | None = None,
    d_literature: int | None = None,
) -> CSSCode:
    """Construct a generalized bicycle CSS code on a cyclic group of order ``n``.

    The resulting code has ``2 n`` physical qubits. ``a_exponents`` and
    ``b_exponents`` list the nonzero powers of the cyclic shift in the
    polynomials ``a`` and ``b``.
    """
    a = _circulant(n, a_exponents)
    b = _circulant(n, b_exponents)
    check_x = np.concatenate([a, b], axis=1) & 1
    check_z = np.concatenate([b.T, a.T], axis=1) & 1
    return CSSCode(
        check_x=check_x,
        check_z=check_z,
        name=name or f"GB(n={n})",
        d_literature=d_literature,
    )


#: Small generalized bicycle instances.
GB_PRESETS: dict[str, dict] = {
    "gb46": {
        "n": 23,
        "a_exponents": [0, 5, 8, 12],
        "b_exponents": [0, 1, 5, 7],
        "d_literature": 6,
    },
    "gb48": {
        "n": 24,
        "a_exponents": [0, 2, 8, 15],
        "b_exponents": [0, 2, 12, 17],
        "d_literature": 6,
    },
}


def build_preset(name: str) -> CSSCode:
    """Build a generalized bicycle code from :data:`GB_PRESETS`."""
    if name not in GB_PRESETS:
        raise KeyError(f"unknown GB preset '{name}'; available: {sorted(GB_PRESETS)}")
    return build_generalized_bicycle(name=name, **GB_PRESETS[name])
