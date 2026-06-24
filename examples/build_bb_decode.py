"""Build qLDPC codes, decode them with BP+OSD, and plot rate vs logical error.

Runs a code-capacity sweep over bivariate-bicycle codes and a toric (surface
family) baseline, then writes the two headline figures to docs/.

    python examples/build_bb_decode.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qldpc import SweepConfig, build_code, ldpc_is_available
from qldpc.experiments import run_sweep
from qldpc.viz import plot_ler_vs_p, plot_rate_vs_ler


def main() -> None:
    os.makedirs("docs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    backend = "auto"  # uses ldpc if present, else the from-scratch BP+OSD
    print(f"BP+OSD backend: {'ldpc' if ldpc_is_available() else 'from-scratch (pure Python)'}")

    headline_p = 0.04
    codes = ["bb72", "bb90", "bb108", "toric:4", "toric:5", "toric:6"]
    for name in codes:
        print(f"  {build_code(name).summary()}")

    rate_sweep = run_sweep(
        SweepConfig(codes=codes, error_rates=[headline_p], shots=3000, seed=11, backend=backend)
    )
    with open("outputs/rate_sweep.json", "w", encoding="utf-8") as handle:
        json.dump(json.loads(rate_sweep.model_dump_json()), handle, indent=2)

    ax = plot_rate_vs_ler(rate_sweep, p=headline_p)
    ax.figure.tight_layout()
    ax.figure.savefig("docs/qldpc_vs_surface_rate_ler.png", dpi=150)
    plt.close(ax.figure)

    p_sweep = run_sweep(
        SweepConfig(
            codes=["bb72", "toric:5"],
            error_rates=[0.02, 0.04, 0.06, 0.08],
            shots=1500,
            seed=21,
            backend=backend,
        )
    )
    ax = plot_ler_vs_p(p_sweep)
    ax.figure.tight_layout()
    ax.figure.savefig("docs/ler_vs_p.png", dpi=150)
    plt.close(ax.figure)

    print("saved docs/qldpc_vs_surface_rate_ler.png and docs/ler_vs_p.png")


if __name__ == "__main__":
    main()
