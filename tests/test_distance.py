"""Tests for distance estimation."""

from qldpc.codes import build_bb_preset, build_toric_code
from qldpc.distance import distance_report


def test_toric_distance_is_exact():
    code = build_toric_code(5)
    report = distance_report(code.logical_x, code.logical_z, d_literature=code.d_literature)
    assert report.d_exact == 5
    assert report.d_literature == 5


def test_bb72_reports_literature_distance():
    code = build_bb_preset("bb72")
    report = distance_report(
        code.logical_x,
        code.logical_z,
        d_literature=code.d_literature,
        search_trials=1000,
        seed=1,
    )
    assert report.d_literature == 6
    assert report.d_exact is None or report.d_exact <= report.d_literature
