"""Cl2 (GW100 7782-50-5, def2-TZVPP, grid level 4) against the LOCUS values of the paper (E_tot -920.2772622182 Ha,
HOMO -7.529697 eV; shell term E_LOCUS - E_r2SCAN+VV10 = -0.0027151446 Ha)."""

from __future__ import annotations

from typing import Any

import pytest

pyscf: Any = pytest.importorskip("pyscf")


def test_cl2_paper_values() -> None:
    from pyscf import gto

    from locus_shell.pyscf_locus import run_locus

    mol: Any = gto.M(atom="Cl 0 0 0; Cl 0 0 1.9878", basis="def2-tzvpp", unit="Angstrom", verbose=0)
    e: float
    mf: Any
    info: dict[str, Any]
    e, mf, info = run_locus(mol, grid_level=4, log=False)
    homo: float = mf.mo_energy[mol.nelectron // 2 - 1] * 27.211386245988
    assert info["converged"]
    assert e == pytest.approx(-920.2772622182, abs=2e-7)
    assert e - info["e_r2scan_vv10"] == pytest.approx(-0.0027151446, abs=1e-8)
    assert homo == pytest.approx(-7.529697, abs=1e-4)
