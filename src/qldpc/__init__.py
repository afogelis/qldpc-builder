"""qLDPC code construction and BP+OSD decoding."""

from .codes import (
    CSSCode,
    build_bivariate_bicycle,
    build_generalized_bicycle,
    build_toric_code,
    hypergraph_product,
)
from .decoders import BeliefPropagation, BpOsdDecoder, ldpc_is_available, make_decoder
from .registry import available_codes, build_code
from .simulation import code_capacity_ler
from .types import LerRecord, SweepConfig, SweepResult

__version__ = "0.1.0"

__all__ = [
    "BeliefPropagation",
    "BpOsdDecoder",
    "CSSCode",
    "LerRecord",
    "SweepConfig",
    "SweepResult",
    "available_codes",
    "build_bivariate_bicycle",
    "build_code",
    "build_generalized_bicycle",
    "build_toric_code",
    "code_capacity_ler",
    "hypergraph_product",
    "ldpc_is_available",
    "make_decoder",
]
