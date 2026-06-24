"""Typed configuration and result records for code-capacity experiments."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SweepConfig(BaseModel):
    """Inputs describing a code-capacity logical-error-rate sweep."""

    model_config = {"frozen": True}

    codes: list[str] = Field(..., min_length=1, description="Code names (presets or 'toric:d').")
    error_rates: list[float] = Field(..., min_length=1)
    shots: int = Field(default=2000, ge=1)
    seed: int = Field(default=2026)
    backend: str = Field(default="auto", description="'auto', 'scratch' or 'ldpc'.")
    max_iterations: int = Field(default=60, ge=1)


class LerRecord(BaseModel):
    """One (code, physical error rate) logical-error-rate measurement."""

    model_config = {"frozen": True}

    code: str
    num_qubits: int
    num_logicals: int
    rate: float
    p: float
    shots: int
    num_failures: int
    logical_error_rate: float
    ci_low: float
    ci_high: float


class SweepResult(BaseModel):
    """A full code-capacity sweep."""

    records: list[LerRecord]

    def for_code(self, name: str) -> list[LerRecord]:
        return [record for record in self.records if record.code == name]
