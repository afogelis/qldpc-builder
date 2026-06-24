"""Tanner-graph connectivity statistics for CSS parity-check matrices."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ConnectivityStats:
    """Summary of check and qubit degrees in the Tanner graph."""

    num_x_checks: int
    num_z_checks: int
    max_x_check_weight: int
    max_z_check_weight: int
    mean_x_check_weight: float
    mean_z_check_weight: float
    max_qubit_x_degree: int
    max_qubit_z_degree: int
    max_qubit_degree: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _row_weights(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(0, dtype=int)
    return matrix.sum(axis=1).astype(int)


def _col_weights(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(matrix.shape[1] if matrix.ndim == 2 else 0, dtype=int)
    return matrix.sum(axis=0).astype(int)


def connectivity_stats(check_x: np.ndarray, check_z: np.ndarray) -> ConnectivityStats:
    """Compute check-weight and qubit-degree summaries."""
    x_rows = _row_weights(check_x)
    z_rows = _row_weights(check_z)
    x_cols = _col_weights(check_x)
    z_cols = _col_weights(check_z)
    total_degree = x_cols + z_cols

    return ConnectivityStats(
        num_x_checks=int(check_x.shape[0]),
        num_z_checks=int(check_z.shape[0]),
        max_x_check_weight=int(x_rows.max()) if x_rows.size else 0,
        max_z_check_weight=int(z_rows.max()) if z_rows.size else 0,
        mean_x_check_weight=float(x_rows.mean()) if x_rows.size else 0.0,
        mean_z_check_weight=float(z_rows.mean()) if z_rows.size else 0.0,
        max_qubit_x_degree=int(x_cols.max()) if x_cols.size else 0,
        max_qubit_z_degree=int(z_cols.max()) if z_cols.size else 0,
        max_qubit_degree=int(total_degree.max()) if total_degree.size else 0,
    )
