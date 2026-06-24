"""Sweep logical error rate across code families and physical error rates."""

from __future__ import annotations

from ..registry import build_code
from ..simulation import code_capacity_ler
from ..types import LerRecord, SweepConfig, SweepResult


def run_sweep(config: SweepConfig) -> SweepResult:
    """Run a code-capacity sweep described by ``config`` and collect records."""
    records: list[LerRecord] = []
    for code_name in config.codes:
        code = build_code(code_name)
        for i, p in enumerate(config.error_rates):
            estimate = code_capacity_ler(
                code,
                p=p,
                shots=config.shots,
                seed=config.seed + i,
                backend=config.backend,
                max_iterations=config.max_iterations,
            )
            records.append(
                LerRecord(
                    code=code_name,
                    num_qubits=code.num_qubits,
                    num_logicals=code.num_logicals,
                    rate=code.rate,
                    p=p,
                    shots=config.shots,
                    num_failures=int(round(estimate.logical_error_rate * config.shots)),
                    logical_error_rate=estimate.logical_error_rate,
                    ci_low=estimate.ci_low,
                    ci_high=estimate.ci_high,
                )
            )
    return SweepResult(records=records)
