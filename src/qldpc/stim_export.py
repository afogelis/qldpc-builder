"""Stim circuit export for CSS syndrome extraction.

These circuits are simplified *code-capacity* models: independent X errors followed
by a single round of Z-check ``MPP`` measurements without ancilla scheduling or
flag qubits. They cross-check the matrix sampler and feed a detector error model
into ``decoder-benchmark``. Full phenomenological X/Z noise is supported in
:func:`qldpc.simulation.phenomenological_ler`, not in this Stim front-end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .codes.css import CSSCode


def stim_is_available() -> bool:
    try:
        import stim  # noqa: F401

        return True
    except ImportError:
        return False


def _require_stim():
    try:
        import stim
    except ImportError as exc:
        raise ImportError(
            "Stim export requires the optional 'stim' extra: pip install 'qldpc-builder[stim]'"
        ) from exc
    return stim


def _mpp_z_targets(row: np.ndarray) -> list:
    stim = _require_stim()
    targets: list = []
    for qubit in np.nonzero(row)[0]:
        if targets:
            targets.append(stim.target_combiner())
        targets.append(stim.target_z(int(qubit)))
    return targets


def build_syndrome_circuit(
    code: CSSCode,
    *,
    px: float = 0.0,
    mode: str = "x_only",
) -> object:
    """Build a Stim code-capacity circuit for independent X errors.

    Applies ``X_ERROR(px)`` on every data qubit, measures all Z-check stabilizers
    with ``MPP``, records detectors, and declares the first logical Z operator as
    the observable.
    """
    if mode != "x_only":
        raise ValueError("Stim export supports mode='x_only' (code-capacity X noise)")

    stim = _require_stim()
    circuit = stim.Circuit()

    for qubit in range(code.num_qubits):
        if px > 0:
            circuit.append("X_ERROR", [qubit], px)

    for row in code.check_z:
        if np.any(row):
            circuit.append("MPP", _mpp_z_targets(row))

    num_z_checks = int(code.check_z.shape[0])
    for offset in range(num_z_checks):
        circuit.append("DETECTOR", [stim.target_rec(-num_z_checks + offset)])

    if code.logical_z.shape[0]:
        circuit.append("MPP", _mpp_z_targets(code.logical_z[0]))
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)

    return circuit


def export_stim(
    code: CSSCode,
    path: Path | str,
    *,
    px: float = 0.0,
    mode: str = "x_only",
) -> Path:
    """Write a Stim circuit file for ``code``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    circuit = build_syndrome_circuit(code, px=px, mode=mode)
    target.write_text(str(circuit), encoding="utf-8")
    return target
