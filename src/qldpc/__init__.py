"""qLDPC code construction and BP+OSD decoding."""

from .codes import (
    CSSCode,
    build_bivariate_bicycle,
    build_generalized_bicycle,
    build_toric_code,
    hypergraph_product,
)
from .connectivity import ConnectivityStats, connectivity_stats
from .decoders import BeliefPropagation, BpOsdDecoder, ldpc_is_available, make_decoder
from .distance import DistanceReport, distance_report
from .registry import available_codes, build_code
from .simulation import code_capacity_ler, phenomenological_ler
from .stim_export import build_syndrome_circuit, export_stim, stim_is_available
from .types import LerRecord, SweepConfig, SweepResult

__version__ = "0.2.0"

__all__ = [
    "BeliefPropagation",
    "BpOsdDecoder",
    "CSSCode",
    "ConnectivityStats",
    "DistanceReport",
    "LerRecord",
    "SweepConfig",
    "SweepResult",
    "available_codes",
    "build_bivariate_bicycle",
    "build_code",
    "build_generalized_bicycle",
    "build_syndrome_circuit",
    "build_toric_code",
    "code_capacity_ler",
    "connectivity_stats",
    "distance_report",
    "export_stim",
    "hypergraph_product",
    "ldpc_is_available",
    "make_decoder",
    "phenomenological_ler",
    "stim_is_available",
]
