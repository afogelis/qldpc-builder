"""A Calderbank-Shor-Steane (CSS) quantum code held as two check matrices.

A CSS code is fixed by an X-check matrix and a Z-check matrix whose rows must
commute (``check_x @ check_z.T = 0`` over GF(2)). From those two matrices every
other property follows: the number of physical qubits, the number of logical
qubits, the encoding rate, and the logical operators used to score decoding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from ..connectivity import ConnectivityStats, connectivity_stats
from ..distance import DistanceReport, distance_report
from ..linalg import css_logicals, rank_gf2


@dataclass(frozen=True)
class CSSCode:
    """A CSS code defined by its X and Z parity-check matrices."""

    check_x: np.ndarray  # (num_x_checks, n) uint8
    check_z: np.ndarray  # (num_z_checks, n) uint8
    name: str = "css"
    d_literature: int | None = None

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

    def connectivity(self) -> ConnectivityStats:
        """Tanner-graph degree and check-weight summaries."""
        return connectivity_stats(self.check_x, self.check_z)

    def distance(self, *, search_trials: int = 4000, seed: int = 2026) -> DistanceReport:
        """Literature, exact, and search-based distance information."""
        return distance_report(
            self.logical_x,
            self.logical_z,
            d_literature=self.d_literature,
            search_trials=search_trials,
            seed=seed,
        )

    def summary(self) -> str:
        report = self.distance(search_trials=500, seed=0)
        distance_text = ""
        if report.d_exact is not None:
            distance_text = f", d={report.d_exact}"
        elif report.d_literature is not None:
            distance_text = f", d~{report.d_literature} (literature)"
        elif report.d_upper is not None:
            distance_text = f", d<={report.d_upper} (search upper bound)"
        return (
            f"[[{self.num_qubits}, {self.num_logicals}{distance_text}]] "
            f"{self.name} (rate {self.rate:.3f})"
        )

    def export_matrices(
        self,
        path: Path | str,
        *,
        include_stim: bool = False,
        px: float = 0.03,
        pz: float = 0.001,
    ) -> Path:
        """Write parity-check matrices and metadata for downstream tools.

        Saves ``Hx.npy``, ``Hz.npy``, ``logical_x.npy``, ``logical_z.npy``, and
        ``metadata.json``. When ``include_stim=True`` and Stim is installed, also
        writes ``syndrome.stim``.
        """
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)

        np.save(target / "Hx.npy", self.check_x)
        np.save(target / "Hz.npy", self.check_z)
        np.save(target / "logical_x.npy", self.logical_x)
        np.save(target / "logical_z.npy", self.logical_z)

        report = self.distance()
        metadata = {
            "name": self.name,
            "n": self.num_qubits,
            "k": self.num_logicals,
            "rate": self.rate,
            "d_literature": self.d_literature,
            "d_exact": report.d_exact,
            "d_upper": report.d_upper,
            "connectivity": self.connectivity().to_dict(),
        }
        with (target / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        if include_stim:
            from ..stim_export import export_stim

            export_stim(self, target / "syndrome.stim", px=px, mode="x_only")

        return target
