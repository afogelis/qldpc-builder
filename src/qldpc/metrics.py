"""Statistical helpers for logical-error-rate estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RateEstimate:
    """A logical error rate with a Wilson score confidence interval."""

    logical_error_rate: float
    ci_low: float
    ci_high: float


def wilson_interval(failures: int, shots: int, *, z: float = 1.96) -> RateEstimate:
    """Wilson score interval for a binomial proportion.

    The Wilson interval is well-behaved for the small failure counts typical of
    below-threshold logical error rates, where the normal approximation breaks
    down.
    """
    if shots <= 0:
        return RateEstimate(float("nan"), float("nan"), float("nan"))
    p_hat = failures / shots
    denom = 1.0 + z * z / shots
    center = (p_hat + z * z / (2 * shots)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / shots + z * z / (4 * shots * shots))
    return RateEstimate(p_hat, max(0.0, center - margin), min(1.0, center + margin))
