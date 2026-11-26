"""Matrices of weight fields between the states of a window around the gap, per k-point (E_S with a screening g(r), interfaces).
GPL-3.0-or-later (uses GPAW).

    python -m locus_es_gpaw window gs.gpw --fields fields.npz -o window.npz [--occ 24] [--emp 16]

For every field F of the fields file (meta 'matrix_fields') and every k-point / spin entry: <m|F|n> = int psi_m^* psi_n
F over the `occ` highest occupied and over the `emp` lowest empty bands, on the exporter's points (pseudo part on the
fine grid, one-centre parts all-electron minus pseudo).  The matrices are computed at the irreducible k-points without
symmetrization: the fields must have the symmetry of the crystal.  The empty bands of the window are converged first
(bands.locked_band_extension, eigenstate error `tol` in eV^2 per valence electron).  MPI: over k-points.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from . import bands as B
from . import export as X
from . import kpar
from ._types import BoolArray, ComplexArray, FloatArray, IntArray
from .points import PointSet, sphere_blocks


def converge_window(calc: Any, n_emp: int, guard: int = 8, tol: float = 1e-4, check: float | None = 1e-2) -> int:
    """bands up to the highest occupied + n_emp + guard at the ground-state density, the occupied ones kept, the others converged"""
    wfs: Any = calc.wfs
    nocc: list[int] = kpar.gather(calc, [int((k.f_n / k.weight > 0.5).sum()) for k in wfs.kpt_u])
    nlo: int
    nhi: int
    nlo, nhi = min(nocc), max(nocc)
    calc.density.update(wfs)
    calc.hamiltonian.update(calc.density)
    if check is not None:
        r: float = B.occupied_residual(calc, nlo)
        if not r < check:
            raise RuntimeError(
                f"occupied-band residual {r:.2e} eV^2/e under the rebuilt Hamiltonian: inconsistent ground state"
            )
    nb: int = nhi + n_emp + guard
    B.locked_band_extension(calc, nb, nhi + n_emp, tol=tol, nlock=nlo)
    return nb


def one_centre_tables(
    calc: Any, ps: PointSet, Fw: FloatArray
) -> dict[int, tuple[tuple[IntArray, IntArray], FloatArray, FloatArray]]:
    """per atom with a PAW correction: (packed indices, factors, T (fields, packed)) with T[f, p] = sum over the atom's sphere points of
    Fw times the density of the packed density-matrix element p (all-electron minus pseudo, no core)"""
    wfs: Any = calc.wfs
    first: dict[int, int] = {}
    o: int = ps.ngrid
    b: dict[str, Any]
    for b in ps.blocks:
        first.setdefault(b["atom"], o)
        o += b["w"].size
    out: dict[int, tuple[tuple[IntArray, IntArray], FloatArray, FloatArray]] = {}
    a: int
    o0: int
    for a, o0 in first.items():
        ni: int = wfs.setups[a].ni
        npk: int = ni * (ni + 1) // 2
        blk: list[dict[str, Any]] = sphere_blocks(calc, ps.xc, {a: np.eye(npk)}, core=False, kinetic=False)
        nmat: FloatArray = np.concatenate([bb["n"].transpose(1, 0, 2).reshape(npk, -1) for bb in blk], axis=1)
        iu: tuple[IntArray, IntArray] = np.triu_indices(ni)
        out[a] = (iu, np.where(iu[0] == iu[1], 1.0, 2.0), Fw[:, o0 : o0 + nmat.shape[1]] @ nmat.T)
    return out


def block_matrices(
    calc: Any,
    ps: PointSet,
    q: int,
    bands: IntArray,
    Fw: FloatArray,
    tables: dict[int, tuple[tuple[IntArray, IntArray], FloatArray, FloatArray]],
) -> ComplexArray:
    """(fields, nb, nb) complex Hermitian matrices <m|F|n> of the given bands at this rank's k-point kpt_u[q]"""
    wfs: Any
    dens: Any
    wfs, dens = calc.wfs, calc.density
    kpt: Any = wfs.kpt_u[q]
    nb: int
    nf: int
    nb, nf = len(bands), len(Fw)
    out: ComplexArray = np.zeros((nf, nb, nb), complex)
    psi: list[FloatArray | ComplexArray] = [wfs.pd.ifft(kpt.psit_nG[n], kpt.q) for n in bands]
    Fg: FloatArray = Fw[:, : ps.ngrid]
    x_g: FloatArray = dens.finegd.empty()
    i: int
    for i in range(nb):
        j: int
        for j in range(i, nb):
            r: FloatArray | ComplexArray = np.conj(psi[i]) * psi[j]
            dens.distribute_and_interpolate(np.ascontiguousarray(r.real), x_g)
            out[:, i, j] = Fg @ x_g.ravel()
            if np.iscomplexobj(r):
                dens.distribute_and_interpolate(np.ascontiguousarray(r.imag), x_g)
                out[:, i, j] += 1j * (Fg @ x_g.ravel())
    a: int
    iu: tuple[IntArray, IntArray]
    fac: FloatArray
    T: FloatArray
    for a, (iu, fac, T) in tables.items():
        P: FloatArray | ComplexArray = np.asarray(kpt.projections[a])[bands]
        D: FloatArray | ComplexArray = np.einsum("mi,nj->mnij", P.conj(), P)
        Sp: FloatArray | ComplexArray = 0.5 * (D + D.transpose(0, 1, 3, 2))[:, :, iu[0], iu[1]] * fac
        out += np.einsum("mnp,fp->fmn", Sp, T)
    il: tuple[IntArray, IntArray] = np.tril_indices(nb, -1)
    out[:, il[0], il[1]] = np.conj(out[:, il[1], il[0]])
    return out


def write_window(
    calc: Any,
    fields_path: str,
    path: str,
    n_occ: int = 24,
    n_emp: int = 16,
    tol: float = 1e-4,
    ps: PointSet | None = None,
    say: Callable[..., None] = print,
) -> dict[str, Any] | None:
    nb: int = converge_window(calc, n_emp, tol=tol)
    ps = ps or PointSet(calc)
    fm: dict[str, Any]
    fa: dict[str, FloatArray]
    fm, fa = X._load(fields_path, "locus-es-fields")
    fp: FloatArray
    fq: FloatArray
    fp, fq = np.array(fm["points_fingerprint"]), np.array(X._fingerprint(ps))
    if fp[0] != fq[0] or np.max(np.abs(fp[1:] - fq[1:]) / np.maximum(np.abs(fq[1:]), 1e-30)) > 1e-8:
        raise ValueError("the fields were computed on other points (different calculation or settings)")
    names: list[str] | None = fm.get("matrix_fields")
    if not names:
        raise ValueError("the fields file names no matrix fields: write it with 'locus-es fields --layers layers.json'")
    Fw: FloatArray = np.array([ps.w * fa[n] for n in names])
    tables: dict[int, tuple[tuple[IntArray, IntArray], FloatArray, FloatArray]] = one_centre_tables(calc, ps, Fw)
    rows: list[tuple[IntArray, IntArray, FloatArray, FloatArray, ComplexArray, ComplexArray]] = []
    q: int
    kpt: Any
    for q, kpt in enumerate(calc.wfs.kpt_u):
        occ: BoolArray = kpt.f_n / kpt.weight > 0.5
        io_: IntArray
        ie: IntArray
        io_, ie = np.flatnonzero(occ), np.flatnonzero(~occ)
        if len(io_) < n_occ or len(ie) < n_emp:
            raise ValueError(
                f"k-point {kpt.k}: {len(io_)} occupied / {len(ie)} empty bands, the window needs {n_occ} / {n_emp}"
            )
        bo: IntArray
        be: IntArray
        bo, be = io_[-n_occ:], ie[:n_emp]
        Mo: ComplexArray
        Me: ComplexArray
        Mo, Me = block_matrices(calc, ps, q, bo, Fw, tables), block_matrices(calc, ps, q, be, Fw, tables)
        rows.append((bo, be, kpt.eps_n[bo].copy(), kpt.eps_n[be].copy(), Mo, Me))
        say(f"k-point {kpt.k} spin {kpt.s}: window matrices done", flush=True)
    rows = kpar.gather(calc, rows)
    ka: dict[str, npt.NDArray[Any]] = X.kpoint_arrays(calc)
    if not kpar.is_master(calc):
        return None
    arrays: dict[str, npt.NDArray[Any]] = dict(
        w_k=ka["w_k"],
        spin_k=ka["spin_k"],
        ibz_of_u=ka["ibz_of_u"],
        bands_occ=np.array([r[0] for r in rows]),
        bands_emp=np.array([r[1] for r in rows]),
        eps_occ=np.array([r[2] for r in rows]),
        eps_emp=np.array([r[3] for r in rows]),
    )
    f: int
    n: str
    for f, n in enumerate(names):
        arrays[f"M:{n}:occ"] = np.array([r[4][f] for r in rows])
        arrays[f"M:{n}:emp"] = np.array([r[5][f] for r in rows])
    meta: dict[str, Any] = dict(
        format="locus-es-window",
        version=X.VERSION,
        producer="locus_es_gpaw",
        points_checksum=fm["points_checksum"],
        fields_checksum=fm["checksum"],
        matrix_fields=list(names),
        layers=fm.get("layers"),
        n_occ=int(n_occ),
        n_emp=int(n_emp),
        nbands_converged=int(nb),
        band_tol=float(tol),
        spin_paired=bool(calc.wfs.nspins == 1),
        base_functional=str(calc.parameters.xc),
    )
    X._save(path, meta, **arrays)
    return meta
