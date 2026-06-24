"""Tests for Tanner-graph connectivity statistics."""

from qldpc.codes import build_bb_preset, build_toric_code
from qldpc.connectivity import connectivity_stats


def test_bb72_has_sparse_checks():
    code = build_bb_preset("bb72")
    stats = connectivity_stats(code.check_x, code.check_z)
    assert stats.max_x_check_weight <= 12
    assert stats.max_qubit_degree >= stats.max_x_check_weight


def test_toric_code_connectivity():
    code = build_toric_code(4)
    stats = connectivity_stats(code.check_x, code.check_z)
    assert stats.num_x_checks == stats.num_z_checks
    assert stats.max_x_check_weight == 4
