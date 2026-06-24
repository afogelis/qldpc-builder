"""Binary (GF(2)) linear algebra used throughout the package.

Quantum LDPC codes are defined by sparse binary parity-check matrices, and
every quantity we care about -- the number of logical qubits, the logical
operators, and the ordered-statistics decoding step -- reduces to linear algebra
over the field with two elements. NumPy has no GF(2) solver, so the small set of
routines needed here are implemented directly: row reduction, rank, null space,
CSS logical-operator extraction, and the order-0 OSD solve.

All matrices are ``uint8`` arrays whose entries are taken modulo two.
"""

from __future__ import annotations

import numpy as np


def _as_gf2(matrix: np.ndarray) -> np.ndarray:
    return (np.asarray(matrix) & 1).astype(np.uint8)


def row_reduce(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return the reduced row-echelon form over GF(2) and its pivot columns."""
    a = _as_gf2(matrix).copy()
    rows, cols = a.shape
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        pivot = -1
        for i in range(r, rows):
            if a[i, c]:
                pivot = i
                break
        if pivot == -1:
            continue
        a[[r, pivot]] = a[[pivot, r]]
        mask = a[:, c].astype(bool).copy()
        mask[r] = False
        a[mask] ^= a[r]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return a, pivots


def rank_gf2(matrix: np.ndarray) -> int:
    """Rank of a binary matrix over GF(2)."""
    if matrix.size == 0:
        return 0
    return len(row_reduce(matrix)[1])


def nullspace_gf2(matrix: np.ndarray) -> np.ndarray:
    """Basis (as rows) of the right null space ``{v : matrix @ v = 0 (mod 2)}``."""
    a = _as_gf2(matrix)
    if a.size == 0:
        cols = a.shape[1] if a.ndim == 2 else 0
        return np.eye(cols, dtype=np.uint8)
    reduced, pivots = row_reduce(a)
    cols = a.shape[1]
    pivot_set = set(pivots)
    free_cols = [c for c in range(cols) if c not in pivot_set]
    basis = []
    pivot_rows = {col: row for row, col in enumerate(pivots)}
    for free in free_cols:
        vector = np.zeros(cols, dtype=np.uint8)
        vector[free] = 1
        for pivot_col, pivot_row in pivot_rows.items():
            vector[pivot_col] = reduced[pivot_row, free]
        basis.append(vector)
    if not basis:
        return np.zeros((0, cols), dtype=np.uint8)
    return np.asarray(basis, dtype=np.uint8)


class _GF2Span:
    """Incrementally maintained echelon basis of a binary row space."""

    def __init__(self) -> None:
        self._rows: list[np.ndarray] = []
        self._pivots: list[int] = []

    def reduce(self, vector: np.ndarray) -> np.ndarray:
        v = _as_gf2(vector).copy()
        for row, pivot in zip(self._rows, self._pivots):
            if v[pivot]:
                v ^= row
        return v

    def add(self, vector: np.ndarray) -> bool:
        """Add ``vector`` to the span; return True iff it was independent."""
        residual = self.reduce(vector)
        nonzero = np.nonzero(residual)[0]
        if nonzero.size == 0:
            return False
        self._rows.append(residual)
        self._pivots.append(int(nonzero[0]))
        return True


def css_logicals(check_x: np.ndarray, check_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return logical operator bases ``(logical_x, logical_z)`` for a CSS code.

    The logical Z operators are a basis of ``ker(check_x)`` modulo the row space
    of ``check_z`` (Z-type operators that commute with every X stabilizer but are
    not themselves stabilizers); the logical X operators are the symmetric
    construction. Each returned array has ``k`` rows, where ``k`` is the number
    of logical qubits.
    """
    cols = check_x.shape[1]

    logical_z = []
    span_z = _GF2Span()
    for row in _as_gf2(check_z):
        span_z.add(row)
    for vector in nullspace_gf2(check_x):
        if span_z.add(vector):
            logical_z.append(vector)

    logical_x = []
    span_x = _GF2Span()
    for row in _as_gf2(check_x):
        span_x.add(row)
    for vector in nullspace_gf2(check_z):
        if span_x.add(vector):
            logical_x.append(vector)

    lx = np.asarray(logical_x, dtype=np.uint8) if logical_x else np.zeros((0, cols), np.uint8)
    lz = np.asarray(logical_z, dtype=np.uint8) if logical_z else np.zeros((0, cols), np.uint8)
    return lx, lz


def osd0_solve(check: np.ndarray, syndrome: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Order-0 ordered-statistics solve of ``check @ e = syndrome`` over GF(2).

    Columns are considered in the priority sequence ``order`` (most-likely-error
    first). Gaussian elimination greedily selects the first ``rank(check)``
    linearly independent columns in that order to form an information set, and
    the syndrome is satisfied on exactly those columns. The result is a valid
    error pattern (it always reproduces ``syndrome``) concentrated on the bits
    the soft decoder considered least reliable.
    """
    h = _as_gf2(check)
    rows, cols = h.shape
    syndrome = _as_gf2(syndrome).reshape(rows, 1)
    augmented = np.concatenate([h.copy(), syndrome], axis=1)

    selected: list[int] = []
    pivot_row = 0
    for col in order:
        if pivot_row >= rows:
            break
        candidates = np.nonzero(augmented[pivot_row:, col])[0]
        if candidates.size == 0:
            continue
        source = pivot_row + int(candidates[0])
        augmented[[pivot_row, source]] = augmented[[source, pivot_row]]
        mask = augmented[:, col].astype(bool).copy()
        mask[pivot_row] = False
        augmented[mask] ^= augmented[pivot_row]
        selected.append(int(col))
        pivot_row += 1

    error = np.zeros(cols, dtype=np.uint8)
    for row, col in enumerate(selected):
        error[col] = augmented[row, -1]
    return error
