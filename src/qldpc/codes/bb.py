"""Bivariate bicycle (BB) code construction.

Bivariate bicycle codes (Bravyi, Cross, Gambetta, Maslov, Rall and Yoder, 2024)
are quantum LDPC codes defined on a two-dimensional torus of size ``l x m``.
Writing ``x`` for the cyclic shift on the first factor and ``y`` for the cyclic
shift on the second, the code is built from two matrices

    A = sum of monomials x^i y^j,
    B = sum of monomials x^i y^j,

and the CSS checks are ``H_X = [A | B]`` and ``H_Z = [B^T | A^T]``. Because the
shifts commute, ``A`` and ``B`` commute, which guarantees the CSS commutation
condition. These codes reach far higher encoding rates than the surface code at
comparable distance, which is the property this package sets out to illustrate.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .css import CSSCode

# A monomial is (x_power, y_power).
Monomial = tuple[int, int]


def _cyclic_shift(size: int) -> np.ndarray:
    """The ``size x size`` cyclic permutation matrix (a single right shift)."""
    return np.roll(np.eye(size, dtype=np.uint8), 1, axis=1)


def _monomial_matrix(ell: int, m: int, x_power: int, y_power: int) -> np.ndarray:
    """Build ``x^a y^b`` as an ``(ell*m) x (ell*m)`` matrix, with ``x = S_ell (x) I_m``."""
    shift_l = _cyclic_shift(ell)
    shift_m = _cyclic_shift(m)
    x = np.kron(shift_l, np.eye(m, dtype=np.uint8))
    y = np.kron(np.eye(ell, dtype=np.uint8), shift_m)
    result = np.eye(ell * m, dtype=np.uint8)
    for _ in range(x_power % ell):
        result = (result @ x) & 1
    for _ in range(y_power % m):
        result = (result @ y) & 1
    return result


def _sum_monomials(ell: int, m: int, terms: Sequence[Monomial]) -> np.ndarray:
    total = np.zeros((ell * m, ell * m), dtype=np.uint8)
    for x_power, y_power in terms:
        total ^= _monomial_matrix(ell, m, x_power, y_power)
    return total


def build_bivariate_bicycle(
    *,
    ell: int,
    m: int,
    a_terms: Sequence[Monomial],
    b_terms: Sequence[Monomial],
    name: str | None = None,
) -> CSSCode:
    """Construct a bivariate bicycle CSS code on an ``ell x m`` torus.

    ``a_terms`` and ``b_terms`` are lists of ``(x_power, y_power)`` monomials
    summed (mod 2) to form the matrices ``A`` and ``B``. The resulting code has
    ``n = 2 * ell * m`` physical qubits.
    """
    a = _sum_monomials(ell, m, a_terms)
    b = _sum_monomials(ell, m, b_terms)
    check_x = np.concatenate([a, b], axis=1) & 1
    check_z = np.concatenate([b.T, a.T], axis=1) & 1
    return CSSCode(check_x=check_x, check_z=check_z, name=name or f"BB(l={ell},m={m})")


#: Small, well-known bivariate bicycle instances suitable for code-capacity sweeps.
BB_PRESETS: dict[str, dict] = {
    # [[72, 12, 6]] -- a compact instance from the Bravyi et al. bivariate-bicycle family.
    "bb72": {
        "ell": 6,
        "m": 6,
        "a_terms": [(3, 0), (0, 1), (0, 2)],
        "b_terms": [(0, 3), (1, 0), (2, 0)],
    },
    # [[90, 8, 10]] on a 15x3 torus (A = x^9 + y + y^2, B = 1 + x^2 + x^7).
    "bb90": {
        "ell": 15,
        "m": 3,
        "a_terms": [(9, 0), (0, 1), (0, 2)],
        "b_terms": [(0, 0), (2, 0), (7, 0)],
    },
    # [[108, 8, 10]] on a 9x6 torus.
    "bb108": {
        "ell": 9,
        "m": 6,
        "a_terms": [(3, 0), (0, 1), (0, 2)],
        "b_terms": [(0, 3), (1, 0), (2, 0)],
    },
}


def build_preset(name: str) -> CSSCode:
    """Build a bivariate bicycle code from :data:`BB_PRESETS`."""
    if name not in BB_PRESETS:
        raise KeyError(f"unknown BB preset '{name}'; available: {sorted(BB_PRESETS)}")
    return build_bivariate_bicycle(name=name, **BB_PRESETS[name])
