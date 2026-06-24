"""Belief propagation with order-0 ordered-statistics post-processing (BP+OSD).

Plain belief propagation frequently fails to converge on quantum LDPC codes
because their Tanner graphs are degenerate and full of short cycles. The
standard remedy is to follow BP with ordered-statistics decoding: use the BP
posterior to rank qubits by how likely they are to be in error, then solve the
syndrome equation exactly on an information set chosen in that order. The result
is always syndrome-consistent, which is the property a stand-alone BP decoder
lacks.

This module implements BP+OSD-0 from scratch (so it runs on any interpreter and
needs no compiled dependency). When the optional ``ldpc`` package is installed,
:func:`make_decoder` can instead return a wrapper around its faster, higher-order
``BpOsdDecoder`` for cross-checking.
"""

from __future__ import annotations

import numpy as np

from ..linalg import osd0_solve
from .bp import BeliefPropagation


def ldpc_is_available() -> bool:
    """Return True if the optional ``ldpc`` package can be imported."""
    try:
        import ldpc  # noqa: F401
    except Exception:
        return False
    return True


class BpOsdDecoder:
    """From-scratch BP+OSD-0 decoder for a fixed check matrix and prior."""

    def __init__(self, check: np.ndarray, priors: np.ndarray, *, max_iterations: int = 60) -> None:
        self.check = (np.asarray(check) & 1).astype(np.uint8)
        self.priors = np.asarray(priors, dtype=float)
        self._bp = BeliefPropagation(self.check, self.priors, max_iterations=max_iterations)

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode ``syndrome`` to an error estimate that always reproduces it."""
        syndrome = (np.asarray(syndrome) & 1).astype(np.uint8)
        hard, posterior_prob = self._bp.decode(syndrome)
        if np.array_equal((self.check @ hard) & 1, syndrome):
            return hard
        # OSD-0: order qubits most-likely-error first and solve exactly.
        order = np.argsort(-posterior_prob, kind="stable")
        return osd0_solve(self.check, syndrome, order)


class _LdpcBpOsdDecoder:
    """Thin wrapper around ``ldpc``'s compiled BP-OSD decoder (optional backend)."""

    def __init__(self, check: np.ndarray, priors: np.ndarray, *, osd_order: int = 4) -> None:
        from ldpc import BpOsdDecoder as _Backend

        self._backend = _Backend(
            (np.asarray(check) & 1).astype(np.uint8),
            error_channel=[float(p) for p in priors],
            bp_method="product_sum",
            osd_method="osd_cs",
            osd_order=osd_order,
        )

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        return np.asarray(self._backend.decode(np.asarray(syndrome) & 1), dtype=np.uint8)


def make_decoder(
    check: np.ndarray, priors: np.ndarray, *, backend: str = "auto", max_iterations: int = 60
):
    """Return a BP-OSD decoder.

    ``backend`` is ``"scratch"`` for the in-repo implementation, ``"ldpc"`` to
    require the optional compiled backend, or ``"auto"`` to prefer ``ldpc`` when
    available and fall back to the from-scratch decoder otherwise.
    """
    if backend == "scratch":
        return BpOsdDecoder(check, priors, max_iterations=max_iterations)
    if backend == "ldpc":
        return _LdpcBpOsdDecoder(check, priors)
    if backend == "auto":
        if ldpc_is_available():
            return _LdpcBpOsdDecoder(check, priors)
        return BpOsdDecoder(check, priors, max_iterations=max_iterations)
    raise ValueError(f"unknown backend '{backend}'")
