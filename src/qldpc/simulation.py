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
