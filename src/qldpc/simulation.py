"""Code-capacity Monte Carlo estimation of the logical error rate.

The simplest meaningful benchmark for a quantum LDPC code is the *code-capacity*
model: independent bit-flip (X) errors on each qubit with probability ``p``,
perfect syndrome measurement, and a single decoding attempt. X errors are
detected by the Z-checks, so a shot is decoded by solving the Z-check syndrome
and is counted as a logical failure when the residual error anticommutes with a
logical Z operator. This isolates the decoder-plus-code performance from
measurement noise and is the standard setting for comparing qLDPC code families.
"""

from __future__ import annotations

import numpy as np

from .codes.css import CSSCode
from .decoders.bposd import make_decoder
from .metrics import RateEstimate, wilson_interval


def code_capacity_ler(
    code: CSSCode,
    *,
    p: float,
    shots: int,
    seed: int = 2026,
    backend: str = "auto",
    max_iterations: int = 60,
) -> RateEstimate:
    """Estimate the bit-flip code-capacity logical error rate of ``code`` at ``p``.

    Independent X errors are sampled at rate ``p``, decoded from the Z-check
    syndrome with BP+OSD, and scored against the logical Z operators.
    """
    rng = np.random.default_rng(seed)
    check = code.check_z  # Z-checks detect X errors
    logical = code.logical_z
    num_qubits = code.num_qubits
    priors = np.full(num_qubits, p, dtype=float)
    decoder = make_decoder(check, priors, backend=backend, max_iterations=max_iterations)

    failures = 0
    for _ in range(shots):
        error = (rng.random(num_qubits) < p).astype(np.uint8)
        syndrome = (check @ error) & 1
        if not syndrome.any():
            estimate = error  # no detected error -> trivial correction
        else:
            estimate = decoder.decode(syndrome)
        residual = (error ^ estimate) & 1
        if logical.shape[0] and np.any((logical @ residual) & 1):
            failures += 1

    return wilson_interval(failures, shots)


def phenomenological_ler(
    code: CSSCode,
    *,
    px: float,
    pz: float,
    shots: int,
    seed: int = 2026,
    backend: str = "auto",
    max_iterations: int = 60,
) -> RateEstimate:
    """Estimate the logical error rate under independent X and Z Pauli noise.

    X errors are decoded from the Z-check syndrome; Z errors from the X-check
    syndrome. A shot fails when the residual anticommutes with any logical
    operator of the opposite type.
    """
    rng = np.random.default_rng(seed)
    decoder_x = make_decoder(
        code.check_z,
        np.full(code.num_qubits, px, dtype=float),
        backend=backend,
        max_iterations=max_iterations,
    )
    decoder_z = make_decoder(
        code.check_x,
        np.full(code.num_qubits, pz, dtype=float),
        backend=backend,
        max_iterations=max_iterations,
    )
    logical_x = code.logical_x
    logical_z = code.logical_z

    failures = 0
    for _ in range(shots):
        x_error = (rng.random(code.num_qubits) < px).astype(np.uint8)
        z_error = (rng.random(code.num_qubits) < pz).astype(np.uint8)

        syndrome_z = (code.check_z @ x_error) & 1
        if syndrome_z.any():
            x_estimate = decoder_x.decode(syndrome_z)
        else:
            x_estimate = x_error

        syndrome_x = (code.check_x @ z_error) & 1
        if syndrome_x.any():
            z_estimate = decoder_z.decode(syndrome_x)
        else:
            z_estimate = z_error

        residual_x = (x_error ^ x_estimate) & 1
        residual_z = (z_error ^ z_estimate) & 1
        z_flip = logical_z.shape[0] and np.any((logical_z @ residual_x) & 1)
        x_flip = logical_x.shape[0] and np.any((logical_x @ residual_z) & 1)
        if z_flip or x_flip:
            failures += 1

    return wilson_interval(failures, shots)
