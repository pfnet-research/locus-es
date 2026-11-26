"""GaAs (ICSD 107946) on LOCUS: E_S from the GPAW exporter's input (ground state of the paper) against the paper
(gap 1.161 -> 1.3566 eV, g 0.08752; Ga 3d band 17.951 eV below the VBM, Fig. 3b)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from locus_es import run

DATA: Path = Path(__file__).parent / "data"


def test_gaas_paper_values() -> None:
    r: dict[str, Any] = run(DATA / "GaAs_input.npz")
    assert r["gap_base_ev"] == pytest.approx(1.16098, abs=1e-4)
    assert r["gap_es_ev"] == pytest.approx(1.356626, abs=1e-5)
    assert r["g"] == pytest.approx(0.087516, abs=1e-6)
    assert r["edges"]["vbm"]["soc_ev"] == pytest.approx(0.116282, abs=1e-5)
    assert r["d_center"]["element"] == "Ga"
    assert r["d_center"]["shift_ev"] == pytest.approx(-0.859916, abs=1e-5)
    assert r["d_center"]["es_below_vbm_ev"] == pytest.approx(17.950675, abs=1e-5)
