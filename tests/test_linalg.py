"""Tests for the GF(2) linear algebra primitives."""

import numpy as np
import pytest

from qldpc.linalg import nullspace_gf2, osd0_solve, rank_gf2, row_reduce


def test_rank_matches_known_value():
    matrix = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.uint8)
    # Third row is the XOR of the first two, so the rank is 2.
    assert rank_gf2(matrix) == 2


def test_nullspace_vectors_are_annihilated():
    rng = np.random.default_rng(0)
    matrix = rng.integers(0, 2, size=(6, 10)).astype(np.uint8)
    null = nullspace_gf2(matrix)
    assert null.shape[0] == matrix.shape[1] - rank_gf2(matrix)
    for vector in null:
        assert not np.any((matrix @ vector) & 1)


def test_row_reduce_is_idempotent_in_rank():
    rng = np.random.default_rng(1)
    matrix = rng.integers(0, 2, size=(8, 8)).astype(np.uint8)
    reduced, pivots = row_reduce(matrix)
    assert len(pivots) == rank_gf2(matrix)
    assert rank_gf2(reduced) == len(pivots)


@pytest.mark.parametrize("seed", range(5))
def test_osd0_solution_reproduces_syndrome(seed):
    rng = np.random.default_rng(seed)
    check = rng.integers(0, 2, size=(7, 14)).astype(np.uint8)
    error = (rng.random(14) < 0.2).astype(np.uint8)
    syndrome = (check @ error) & 1
    order = rng.permutation(14)
    estimate = osd0_solve(check, syndrome, order)
    assert np.array_equal((check @ estimate) & 1, syndrome)
