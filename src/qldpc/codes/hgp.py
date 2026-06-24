"""Hypergraph-product codes, used here to build a surface-code baseline.

The hypergraph product (Tillich and Zemor, 2009) turns two classical codes into
a CSS quantum code. Applied to two cyclic repetition codes it yields the toric
code, a ``[[2 d^2, 2, d]]`` family whose encoding rate ``2 / (2 d^2)`` falls as
``1 / d^2``. That vanishing rate is exactly the baseline against which the
constant-rate bivariate-bicycle codes are compared.
"""

from __future__ import annotations

import numpy as np

from .css import CSSCode


def hypergraph_product(h1: np.ndarray, h2: np.ndarray, *, name: str = "hgp") -> CSSCode:
    """CSS code from the hypergraph product of classical checks ``h1`` and ``h2``."""
    h1 = (np.asarray(h1) & 1).astype(np.uint8)
    h2 = (np.asarray(h2) & 1).astype(np.uint8)
    m1, n1 = h1.shape
    m2, n2 = h2.shape

    check_x = (
        np.concatenate(
            [np.kron(h1, np.eye(n2, dtype=np.uint8)), np.kron(np.eye(m1, dtype=np.uint8), h2.T)],
            axis=1,
        )
        & 1
    )
    check_z = (
        np.concatenate(
            [np.kron(np.eye(n1, dtype=np.uint8), h2), np.kron(h1.T, np.eye(m2, dtype=np.uint8))],
            axis=1,
        )
        & 1
    )
    return CSSCode(check_x=check_x, check_z=check_z, name=name)


def _repetition_check(d: int) -> np.ndarray:
    """Cyclic repetition-code parity check (``I + S``) of size ``d x d``."""
    return (np.eye(d, dtype=np.uint8) ^ np.roll(np.eye(d, dtype=np.uint8), 1, axis=1)) & 1


def build_toric_code(distance: int) -> CSSCode:
    """Build the ``[[2 d^2, 2, d]]`` toric code as a surface-code baseline."""
    check = _repetition_check(distance)
    return hypergraph_product(check, check, name=f"toric(d={distance})")
