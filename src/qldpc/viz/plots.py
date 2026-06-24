"""Headline figures for the qLDPC scaling study."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ..types import SweepResult


def _is_surface(code_name: str) -> bool:
    return code_name.startswith("toric")


def plot_rate_vs_ler(result: SweepResult, *, p: float, ax: Axes | None = None) -> Axes:
    """Scatter encoding rate against logical error rate at a fixed physical rate.

    Bivariate-bicycle codes sit at high encoding rate; the toric (surface-family)
    baseline sits near zero rate. At comparable logical error rate the qLDPC
    codes encode far more logical qubits per physical qubit, which is the whole
    point of the family.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 5.5))

    for record in result.records:
        if abs(record.p - p) > 1e-12:
            continue
        is_surface = _is_surface(record.code)
        ax.scatter(
            record.rate,
            max(record.logical_error_rate, 1e-6),
            s=90,
            marker="s" if is_surface else "o",
            color="C1" if is_surface else "C0",
        )
        ax.annotate(
            f"{record.code}\n[[{record.num_qubits},{record.num_logicals}]]",
            (record.rate, max(record.logical_error_rate, 1e-6)),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Encoding rate  k / n")
    ax.set_ylabel(f"Logical error rate at p = {p}")
    ax.set_title("qLDPC (bivariate bicycle) vs surface-code baseline: rate vs logical error rate")
    ax.scatter([], [], marker="o", color="C0", label="qLDPC (bivariate / generalized bicycle)")
    ax.scatter([], [], marker="s", color="C1", label="surface family (toric)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_ler_vs_p(result: SweepResult, *, ax: Axes | None = None) -> Axes:
    """Plot logical error rate versus physical error rate for each code."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 5.5))

    codes = sorted({record.code for record in result.records})
    for code_name in codes:
        rows = sorted(result.for_code(code_name), key=lambda r: r.p)
        ax.plot(
            [r.p for r in rows],
            [max(r.logical_error_rate, 1e-6) for r in rows],
            marker="s" if _is_surface(code_name) else "o",
            label=code_name,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical error rate p")
    ax.set_ylabel("Logical error rate")
    ax.set_title("Code-capacity logical error rate vs physical error rate")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    return ax
