"""Tests for the from-scratch BP+OSD decoder and the code-capacity simulation."""

import numpy as np
import pytest

from qldpc.codes import build_bb_preset, build_toric_code
from qldpc.decoders import BpOsdDecoder, ldpc_is_available, make_decoder
from qldpc.simulation import code_capacity_ler


def test_bposd_correction_is_always_syndrome_consistent():
    # OSD guarantees the returned error reproduces the syndrome, even when BP
    # alone would not converge.
    code = build_toric_code(4)
    rng = np.random.default_rng(3)
    priors = np.full(code.num_qubits, 0.05)
    decoder = BpOsdDecoder(code.check_z, priors)
    for _ in range(50):
        error = (rng.random(code.num_qubits) < 0.08).astype(np.uint8)
        syndrome = (code.check_z @ error) & 1
        estimate = decoder.decode(syndrome)
        assert np.array_equal((code.check_z @ estimate) & 1, syndrome)


def test_bposd_corrects_low_weight_errors_on_toric_code():
    code = build_toric_code(5)
    priors = np.full(code.num_qubits, 0.02)
    decoder = BpOsdDecoder(code.check_z, priors)
    logical = code.logical_z
    successes = 0
    trials = 40
    rng = np.random.default_rng(7)
    for _ in range(trials):
        error = np.zeros(code.num_qubits, dtype=np.uint8)
        error[rng.integers(0, code.num_qubits)] = 1  # single-qubit error
        syndrome = (code.check_z @ error) & 1
        estimate = decoder.decode(syndrome)
        residual = (error ^ estimate) & 1
        if not np.any((logical @ residual) & 1):
            successes += 1
    # Distance-5 code must correct all weight-1 errors.
    assert successes == trials


def test_code_capacity_ler_low_below_threshold():
    code = build_bb_preset("bb72")
    estimate = code_capacity_ler(code, p=0.01, shots=300, seed=1, backend="scratch")
    # Well below threshold the logical error rate should be small.
    assert estimate.logical_error_rate < 0.5


def test_make_decoder_falls_back_to_scratch_without_ldpc():
    code = build_toric_code(3)
    priors = np.full(code.num_qubits, 0.05)
    decoder = make_decoder(code.check_z, priors, backend="auto")
    if ldpc_is_available():
        pytest.skip("ldpc installed; auto backend uses the compiled decoder")
    assert isinstance(decoder, BpOsdDecoder)
