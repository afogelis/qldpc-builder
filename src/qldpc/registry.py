"""Resolve human-friendly code names into :class:`CSSCode` instances."""

from __future__ import annotations

from .codes.bb import BB_PRESETS
from .codes.bb import build_preset as _build_bb
from .codes.css import CSSCode
from .codes.gb import GB_PRESETS
from .codes.gb import build_preset as _build_gb
from .codes.hgp import build_toric_code


def build_code(name: str) -> CSSCode:
    """Build a code from a preset name or a ``toric:d`` specification."""
    if name.startswith("toric:"):
        distance = int(name.split(":", 1)[1])
        return build_toric_code(distance)
    if name in BB_PRESETS:
        return _build_bb(name)
    if name in GB_PRESETS:
        return _build_gb(name)
    raise KeyError(
        f"unknown code '{name}'; available presets: "
        f"{sorted(BB_PRESETS) + sorted(GB_PRESETS)} or 'toric:<d>'"
    )


def available_codes() -> list[str]:
    """Names of all preset codes (toric codes are parameterised separately)."""
    return sorted(BB_PRESETS) + sorted(GB_PRESETS)
