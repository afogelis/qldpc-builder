"""Tests for phenomenological noise simulation."""

from qldpc.codes import build_toric_code
from qldpc.simulation import phenomenological_ler


def test_phenomenological_ler_runs():
    code = build_toric_code(4)
    estimate = phenomenological_ler(code, px=0.01, pz=0.01, shots=200, seed=3, backend="scratch")
    assert 0.0 <= estimate.logical_error_rate <= 1.0
