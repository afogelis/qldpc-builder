"""A Calderbank-Shor-Steane (CSS) quantum code held as two check matrices.

A CSS code is fixed by an X-check matrix and a Z-check matrix whose rows must
commute (``check_x @ check_z.T = 0`` over GF(2)). From those two matrices every
other property follows: the number of physical qubits, the number of logical
qubits, the encoding rate, and the logical operators used to score decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from ..linalg import css_logicals, rank_gf2


@dataclass(frozen=True)
class CSSCode:
    """A CSS code defined by its X and Z parity-check matrices."""

    check_x: np.ndarray  # (num_x_checks, n) uint8
    check_z: np.ndarray  # (num_z_checks, n) uint8
    name: str = "css"

    def __post_init__(self) -> None:
        if self.check_x.shape[1] != self.check_z.shape[1]:
            raise ValueError("X and Z checks must act on the same number of qubits")
        product = (self.check_x @ self.check_z.T) & 1
        if np.any(product):
            raise ValueError(f"CSS commutation violated for code '{self.name}'")

    @property
    def num_qubits(self) -> int:
        return int(self.check_x.shape[1])

    @cached_property
    def num_logicals(self) -> int:
        """Number of logical qubits, ``k = n - rank(check_x) - rank(check_z)``."""
        return self.num_qubits - rank_gf2(self.check_x) - rank_gf2(self.check_z)

    @property
    def rate(self) -> float:
        """Encoding rate ``k / n``."""
        return self.num_logicals / self.num_qubits

    @cached_property
    def logicals(self) -> tuple[np.ndarray, np.ndarray]:
        """Logical operator bases ``(logical_x, logical_z)``."""
        return css_logicals(self.check_x, self.check_z)

    @property
    def logical_x(self) -> np.ndarray:
        return self.logicals[0]

    @property
    def logical_z(self) -> np.ndarray:
        return self.logicals[1]

    def summary(self) -> str:
        return f"[[{self.num_qubits}, {self.num_logicals}]] {self.name} (rate {self.rate:.3f})"
