"""Ground state of the paper's protocol: PBE, then LOCUS from the PBE state (data/PROTOCOL.md).

    python -m locus_gpaw.protocol data/structures/crystals.json GaAs_107946 -o gs.gpw [--model LOCUS|PBE|r2SCAN]

Plane waves 600 eV, Fermi-Dirac 0.01 eV, nbands = nocc + max(8, nocc / 2), symmetry 'full', the k-mesh and setups stored with the
structure.  PBE stage: energy 5e-4, density 1e-6, eigenstates 1e-9 (all bands but the top 4), at most 333 iterations; LOCUS stage:
energy 1e-6, density 1e-5, eigenstates 1e-7, plain mixer beta 0.02 / nmaxold 8 / weight 100, at most 300 iterations (spin-polarized:
the summed-density mixer with the same parameters, at most 600 iterations).  A stage that reaches its iteration limit is
continued from (as in the paper's runs).  GPL-3.0-or-later.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Union

if TYPE_CHECKING:
    from ._types import FloatArray

CONV_PBE: dict[str, float] = {"energy": 5e-4, "density": 1e-6, "eigenstates": 1e-9, "bands": -4}
CONV_LOCUS: dict[str, float] = {"energy": 1e-6, "density": 1e-5, "eigenstates": 1e-7, "bands": -4}


def atoms_of(entry: dict[str, Any]) -> Any:
    from ase import Atoms

    a: Any = Atoms(entry["symbols"], positions=entry["positions_A"], cell=entry["cell_A"], pbc=entry.get("pbc", True))
    if entry.get("initial_magnetic_moments"):
        a.set_initial_magnetic_moments(entry["initial_magnetic_moments"])
    return a


def _energy(atoms: Any, calc: Any) -> tuple[float, bool]:
    from gpaw import KohnShamConvergenceError

    try:
        return atoms.get_potential_energy(), True
    except KohnShamConvergenceError:
        calc.scf.converged = True
        return float(calc.hamiltonian.e_total_extrapolated * 27.211386024367243), False


def _say(*args: Any, **kw: Any) -> None:
    from gpaw.mpi import world

    if world.rank == 0:
        print(*args, **kw)


def ground_state(
    atoms: Any,
    kpts: Sequence[int],
    setups: Union[str, dict[str, str]],
    ecut: float = 600.0,
    model: Union[str, Any] = "LOCUS",
    txt: Any = "-",
    log: Callable[..., None] = _say,
    conv_bands_extra: Optional[int] = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """the ground state of the protocol; conv_bands_extra: converge only the occupied bands + this many (large Gamma-only cells,
    data/PROTOCOL.md; the E_S exporter builds the empty bands itself) instead of all bands but the top 4"""
    from gpaw import GPAW, PW, FermiDirac, Mixer, MixerSum

    from . import LOCUS

    nocc_guess: Optional[int] = None
    calc: Any = GPAW(
        mode=PW(ecut),
        xc="PBE",
        kpts={"size": list(kpts), "gamma": True},
        occupations=FermiDirac(0.01),
        setups=setups,
        symmetry={},
        nbands=None,
        maxiter=333,
        convergence=CONV_PBE,
        txt=txt,
    )
    atoms.calc = calc
    calc.initialize(atoms)
    nocc_guess = int(round(calc.wfs.nvalence / 2))
    calc.set(nbands=-max(8, nocc_guess // 2))
    conv_pbe: dict[str, float] = dict(CONV_PBE)
    conv_locus: dict[str, float] = dict(CONV_LOCUS)
    if conv_bands_extra:
        nconv: int = min(nocc_guess + int(conv_bands_extra), nocc_guess + max(8, nocc_guess // 2) - 4)
        conv_pbe["bands"] = conv_locus["bands"] = nconv
        calc.set(convergence=conv_pbe)
    recs: list[dict[str, Any]] = []
    t0: float = time.time()
    e: float
    ok: bool
    e, ok = _energy(atoms, calc)
    recs.append(dict(stage="PBE", E=e, converged=ok, niter=int(calc.scf.niter), seconds=time.time() - t0))
    log(f"PBE: E {e:.6f} eV converged {ok} ({recs[-1]['niter']} iterations, {recs[-1]['seconds']:.0f} s)")
    if model != "PBE":
        locus: bool
        locus = (model == "LOCUS") or not isinstance(model, str)  # an xc object: a LOCUS-type functional given directly
        xc: Any = LOCUS() if model == "LOCUS" else model
        spin: bool = calc.wfs.nspins == 2
        kw: dict[str, Any] = dict(xc=xc, maxiter=600 if spin else 300, convergence=conv_locus)
        if locus:  # spin-polarized (open-shell oxides): the mixer of the summed density, as in the paper's runs
            kw["mixer"] = (MixerSum if spin else Mixer)(beta=0.02, nmaxold=8, weight=100.0)
        calc.set(**kw)
        t0 = time.time()
        e, ok = _energy(atoms, calc)
        recs.append(
            dict(
                stage=model if isinstance(model, str) else "LOCUS",
                E=e,
                converged=ok,
                niter=int(calc.scf.niter),
                seconds=time.time() - t0,
            )
        )
        log(
            f"{recs[-1]['stage']}: E {e:.6f} eV converged {ok} ({recs[-1]['niter']} iterations, {recs[-1]['seconds']:.0f} s)"
        )
    return calc, recs


def ks_gap(calc: Any) -> float:
    """Kohn-Sham gap over all k-points (eV; also with GPAW's k-point parallelisation)"""
    import numpy as np

    nocc: int = int(round(calc.get_number_of_electrons() / 2))
    a: FloatArray = np.array(
        [max(float(k.eps_n[nocc - 1]) for k in calc.wfs.kpt_u), -min(float(k.eps_n[nocc]) for k in calc.wfs.kpt_u)]
    )
    calc.wfs.kd.comm.max(a)
    return float((-a[1] - a[0]) * 27.211386024367243)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap: argparse.ArgumentParser = argparse.ArgumentParser(prog="locus_gpaw.protocol")
    ap.add_argument("structures", help="a structure file of data/structures")
    ap.add_argument("key")
    ap.add_argument("-o", "--out", required=True, help="gpw file (mode 'all')")
    ap.add_argument(
        "--model",
        default="LOCUS",
        help="LOCUS (default), PBE (stop after the PBE stage) or a GPAW xc name for the second stage",
    )
    ap.add_argument("--txt", default="-")
    ap.add_argument(
        "--large-cell",
        action="store_true",
        help="large Gamma-only cells with a band gap: converge only the occupied bands + 8 (data/PROTOCOL.md); run under MPI, "
        "GPAW splits the cell over real-space domains",
    )
    a: argparse.Namespace = ap.parse_args(argv)
    e: dict[str, Any] = json.load(open(a.structures))["entries"][a.key]
    atoms: Any = atoms_of(e)
    calc: Any
    recs: list[dict[str, Any]]
    calc, recs = ground_state(
        atoms,
        e["kpts"],
        e.get("setups") or "paw",
        float(e.get("ecut") or 600.0),
        a.model,
        a.txt,
        conv_bands_extra=8 if a.large_cell else None,
    )
    calc.write(a.out, mode="all")
    _say(json.dumps(dict(key=a.key, kpts=e["kpts"], setups=e.get("setups"), stages=recs, ks_gap_ev=ks_gap(calc))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
