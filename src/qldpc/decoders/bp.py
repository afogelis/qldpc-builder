"""From-scratch belief propagation over a binary parity-check matrix.

Sum-product belief propagation passes log-likelihood-ratio messages on the
Tanner graph of a parity-check matrix. On its own it is known to struggle on
the highly degenerate graphs of quantum codes, but its *soft output* -- a
posterior error probability for every qubit -- is exactly the reliability
ordering that the ordered-statistics post-processing step needs. This module
returns both the hard decision and that soft output.
"""

from __future__ import annotations

import numpy as np

_TANH_CLIP = 1.0 - 1e-12


class BeliefPropagation:
    """Log-domain sum-product decoder for a fixed check matrix and prior."""

    def __init__(self, check: np.ndarray, priors: np.ndarray, *, max_iterations: int = 60) -> None:
        self.check = (np.asarray(check) & 1).astype(np.uint8)
        self.num_checks, self.num_bits = self.check.shape
        priors = np.clip(np.asarray(priors, dtype=float), 1e-15, 0.5 - 1e-15)
        self.prior_llr = np.log((1.0 - priors) / priors)
        self.max_iterations = max_iterations

        checks, bits = np.nonzero(self.check)
        self._check_of_edge = checks.astype(np.int64)
        self._bit_of_edge = bits.astype(np.int64)

    def decode(self, syndrome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(hard_decision, posterior_error_probability)`` for ``syndrome``.

        ``hard_decision`` is a length-``num_bits`` ``uint8`` error estimate;
        ``posterior_error_probability`` gives each bit's marginal probability of
        being in error and is used to order bits for OSD.
        """
        syndrome = (np.asarray(syndrome) & 1).astype(np.uint8)
        check_of_edge = self._check_of_edge
        bit_of_edge = self._bit_of_edge
        sign = np.where(syndrome[check_of_edge] == 1, -1.0, 1.0)

        msg_bit_to_check = self.prior_llr[bit_of_edge].copy()
        total = self.prior_llr.copy()
        best_decision = (self.prior_llr < 0.0).astype(np.uint8)

        for _ in range(self.max_iterations):
            tanh_half = np.clip(np.tanh(0.5 * msg_bit_to_check), -_TANH_CLIP, _TANH_CLIP)
            tanh_half = np.where(tanh_half == 0.0, 1e-12, tanh_half)
            product_per_check = np.ones(self.num_checks, dtype=float)
            np.multiply.at(product_per_check, check_of_edge, tanh_half)
            excluded = np.clip(
                product_per_check[check_of_edge] / tanh_half, -_TANH_CLIP, _TANH_CLIP
            )
            msg_check_to_bit = sign * 2.0 * np.arctanh(excluded)

            total = self.prior_llr.copy()
            np.add.at(total, bit_of_edge, msg_check_to_bit)
            msg_bit_to_check = total[bit_of_edge] - msg_check_to_bit

            decision = (total < 0.0).astype(np.uint8)
            if np.array_equal((self.check @ decision) & 1, syndrome):
                best_decision = decision
                break
            best_decision = decision

        posterior_prob = 1.0 / (1.0 + np.exp(np.clip(total, -700, 700)))
        return best_decision, posterior_prob
