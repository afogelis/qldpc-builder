"""Tests for Stim circuit export."""

import pytest

from qldpc.codes import build_toric_code
from qldpc.stim_export import build_syndrome_circuit, stim_is_available


@pytest.mark.skipif(not stim_is_available(), reason="stim not installed")
def test_build_syndrome_circuit_compiles():
    code = build_toric_code(3)
    circuit = build_syndrome_circuit(code, px=0.01, mode="x_only")
    dem = circuit.detector_error_model(decompose_errors=False)
    assert dem.num_detectors > 0


@pytest.mark.skipif(not stim_is_available(), reason="stim not installed")
def test_export_bundle_includes_stim(tmp_path):
    code = build_toric_code(3)
    target = code.export_matrices(tmp_path / "bundle", include_stim=True, px=0.02)
    assert (target / "syndrome.stim").is_file()
