"""Tests for matrix export bundles."""

import json
from pathlib import Path

from qldpc.codes import build_toric_code


def test_export_matrices_writes_bundle(tmp_path: Path):
    code = build_toric_code(3)
    target = code.export_matrices(tmp_path / "toric3")
    assert (target / "Hx.npy").is_file()
    assert (target / "Hz.npy").is_file()
    assert (target / "metadata.json").is_file()
    with (target / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["n"] == code.num_qubits
    assert metadata["k"] == code.num_logicals
    assert metadata["d_exact"] == 3
    assert "connectivity" in metadata
