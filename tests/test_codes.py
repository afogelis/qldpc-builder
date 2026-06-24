"""Tests for the quantum LDPC code constructions."""

import numpy as np
import pytest

from qldpc.codes import build_bb_preset, build_gb_preset, build_toric_code
from qldpc.codes.bb import BB_PRESETS
from qldpc.codes.gb import GB_PRESETS


def _assert_valid_css(code):
    # CSS commutation is enforced in the constructor; re-check explicitly.
    assert not np.any((code.check_x @ code.check_z.T) & 1)
    assert code.num_logicals >= 1
    # Logical operators must commute with the stabilizers of the opposite type.
    assert not np.any((code.check_x @ code.logical_z.T) & 1)
    assert not np.any((code.check_z @ code.logical_x.T) & 1)
    # The number of logical operators of each type equals k.
    assert code.logical_x.shape[0] == code.num_logicals
    assert code.logical_z.shape[0] == code.num_logicals


@pytest.mark.parametrize("name", sorted(BB_PRESETS))
def test_bivariate_bicycle_presets_are_valid(name):
    code = build_bb_preset(name)
    assert code.num_qubits == 2 * BB_PRESETS[name]["ell"] * BB_PRESETS[name]["m"]
    _assert_valid_css(code)


@pytest.mark.parametrize("name", sorted(GB_PRESETS))
def test_generalized_bicycle_presets_are_valid(name):
    code = build_gb_preset(name)
    assert code.num_qubits == 2 * GB_PRESETS[name]["n"]
    _assert_valid_css(code)


@pytest.mark.parametrize("distance", [3, 4, 5])
def test_toric_code_parameters(distance):
    code = build_toric_code(distance)
    assert code.num_qubits == 2 * distance * distance
    # The toric code encodes exactly two logical qubits.
    assert code.num_logicals == 2
    _assert_valid_css(code)


def test_bivariate_bicycle_has_higher_rate_than_toric():
    bb = build_bb_preset("bb72")
    toric = build_toric_code(6)
    assert bb.rate > toric.rate
