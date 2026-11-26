"""Reading a ground state and adding empty bands at the converged density (no SCF); expectation values of other functionals.
GPL-3.0-or-later (uses GPAW internals).

extend_empty_bands: nocc + max(16, 2 nocc) bands by a locked block LOBPCG on the fixed Hamiltonian.  The bands the
ground state converged (its convergence 'bands' setting) below the top 4 are locked; the others and random new ones are
iterated S-orthogonal to them, per k-point in blocks of at most 128 bands, to an eigenstate error of 1e-2 eV^2 per
valence electron (the screening and the corrected levels are the same as at 1e-9).  Plane-wave mode, collinear; MPI over
k-points and real-space domains (kpar.LAYOUT).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from ase.units import Hartree as HA

from . import kpar
from ._types import BoolArray, ComplexArray, FloatArray, IntArray


def open_gpw(gpw: str, **kwargs: Any) -> Any:
    """GPAW(gpw) with the positions initialised and all wave functions in memory"""
    from gpaw import GPAW
    from gpaw.mpi import world

    if world.size > 1:
        kwargs = dict(dict(parallel=kpar.LAYOUT), **kwargs)
    calc: Any = GPAW(str(gpw), **dict(dict(txt=None), **kwargs))
    kpar.check_layout(calc)
    if "energy" not in calc.results:  # stored after an SCF that stopped at maxiter: the stored state is used as it is
        calc.scf.converged = True
        calc.results["energy"] = float(calc.hamiltonian.e_total_extrapolated * HA)
    atoms: Any = calc.get_atoms()
    atoms.get_potential_energy()
    calc.initialize_positions(atoms)
    kpt: Any
    for kpt in calc.wfs.kpt_u:
        if not kpt.psit.in_memory:
            kpt.psit.read_from_file()
    return calc


class _Projectors:
    """the projector functions of one k-point expanded in plane waves and kept: projections and their addition as matrix
    products, with the arithmetic of PWLFC.integrate / PWLFC.add.  Memory per rank: local plane waves x projectors (x 2
    real numbers)."""

    pt: Any
    q: int
    f_GI: npt.NDArray[Any]
    real: bool

    def __init__(self, pt: Any, q: int) -> None:
        self.pt, self.q = pt, q
        self.real = pt.pd.dtype == float
        # complex: (nG, nI); real wave functions: (2 nG, nI), real and imaginary parts in alternate rows
        self.f_GI = pt.expand(q)

    @staticmethod
    def gib(pt: Any, q: int) -> float:
        ng: int = int(pt.Y_qGL[q].shape[0])
        return float(ng * pt.nI * 16) / 2**30

    def integrate(self, a_xG: ComplexArray) -> dict[int, npt.NDArray[Any]]:
        """<p_i^a|a_x> of this rank's atoms, {a: (x, i)}"""
        pt: Any = self.pt
        alpha: float = 1.0 / float(pt.pd.gd.N_c.prod())
        b: npt.NDArray[Any]
        if self.real:
            a: FloatArray = a_xG.view(float)
            b = (2.0 * alpha) * (a @ self.f_GI)
            if pt.comm.rank == 0:  # G = 0 counted once
                b -= alpha * np.outer(a[:, 0], self.f_GI[0])
        else:
            b = alpha * (a_xG.conj() @ self.f_GI).conj()
        pt.comm.sum(b)
        e: npt.NDArray[Any] = pt.eikR_qa[self.q]
        return {a_: e[a_] * b[:, I1:I2] for a_, I1, I2 in pt.my_indices}

    def add(self, a_xG: ComplexArray, c_axi: dict[int, npt.NDArray[Any]]) -> None:
        """a_x += sum_ai c_axi p_i^a"""
        pt: Any = self.pt
        c: npt.NDArray[Any] = np.zeros((len(a_xG), pt.nI), pt.pd.dtype)
        e: npt.NDArray[Any] = pt.eikR_qa[self.q]
        a_: int
        I1: int
        I2: int
        for a_, I1, I2 in pt.my_indices:
            c[:, I1:I2] = c_axi[a_] * np.conj(e[a_])
        if pt.comm.size != 1:
            pt.comm.sum(c)
        av: npt.NDArray[Any] = a_xG.view(pt.pd.dtype)
        av += (1.0 / float(pt.pd.gd.dv)) * (c @ self.f_GI.T)


def paw_ops(
    calc: Any,
) -> tuple[
    Callable[[Any, ComplexArray], Any],
    Callable[..., ComplexArray],
    Callable[..., tuple[ComplexArray, ComplexArray]],
    Callable[[ComplexArray, ComplexArray], FloatArray],
]:
    """projections, S and H (the calculator's current Hamiltonian) on blocks of plane-wave coefficients of one k-point, and the inner
    product of columns (complex and real Gamma-point wave functions)"""
    from gpaw.projections import Projections
    from gpaw.wavefunctions.arrays import PlaneWaveExpansionWaveFunctions

    wfs: Any
    ham: Any
    wfs, ham = calc.wfs, calc.hamiltonian
    pd: Any = wfs.pd
    nproj_a: list[int] = [s_.ni for s_ in wfs.setups]
    # LOCUS_ES_PROJ_GIB: largest expanded projectors per rank (GiB); above it GPAW's own routines are used
    max_gib: float = float(os.environ.get("LOCUS_ES_PROJ_GIB", "8"))
    cache: dict[int, _Projectors] = {}

    def projectors(q: int) -> _Projectors | None:
        """the expanded projectors of k-point q (one k-point kept at a time), None above max_gib"""
        if q not in cache:
            cache.clear()
            if _Projectors.gib(wfs.pt, q) > max_gib:
                return None
            cache[q] = _Projectors(wfs.pt, q)
        return cache[q]

    def proj(kpt: Any, a_xG: ComplexArray) -> Any:
        P: Any = Projections(
            len(a_xG), nproj_a, wfs.atom_partition, wfs.bd.comm, collinear=True, spin=kpt.s, dtype=wfs.dtype
        )
        pr: _Projectors | None = projectors(kpt.q)
        if pr is None:
            PlaneWaveExpansionWaveFunctions(
                len(a_xG), pd, wfs.dtype, data=a_xG, kpt=kpt.q, dist=(wfs.bd.comm, -1, 1), spin=kpt.s
            ).matrix_elements(wfs.pt, out=P)
        else:
            c: dict[int, npt.NDArray[Any]] = pr.integrate(a_xG)
            a: int
            P_ni: npt.NDArray[Any]
            for a, P_ni in P.items():
                P_ni[:] = c[a]
        return P

    def add(kpt: Any, a_xG: ComplexArray, c_axi: dict[int, npt.NDArray[Any]]) -> None:
        pr: _Projectors | None = projectors(kpt.q)
        if pr is None:
            wfs.pt.add(a_xG, c_axi, kpt.q)
        else:
            pr.add(a_xG, c_axi)

    def apply_s(kpt: Any, a_xG: ComplexArray, P: Any = None) -> ComplexArray:
        P = proj(kpt, a_xG) if P is None else P
        S: ComplexArray = a_xG.copy()
        add(kpt, S, {a: P_ni @ wfs.setups[a].dO_ii for a, P_ni in P.items()})
        return S

    import inspect

    # GPAW builds with an extra projections argument
    with_P: bool = "P" in inspect.signature(wfs.apply_pseudo_hamiltonian).parameters

    def apply_hs(kpt: Any, a_xG: ComplexArray, P: Any = None) -> tuple[ComplexArray, ComplexArray]:
        """H a and S a; P: the projections of a if known"""
        P = proj(kpt, a_xG) if P is None else P
        H: ComplexArray = np.empty_like(a_xG)
        if with_P:
            wfs.apply_pseudo_hamiltonian(kpt, ham, P, a_xG, H)
        else:
            wfs.apply_pseudo_hamiltonian(kpt, ham, a_xG, H)
        C: Any = P.new()
        ham.dH(P, out=C)
        add(kpt, H, {a: C_ni for a, C_ni in C.items()})
        return H, apply_s(kpt, a_xG, P)

    alpha: float = wfs.gd.dv / wfs.gd.N_c.prod()
    real: bool = wfs.dtype == float

    gcomm: Any = wfs.gd.comm

    def cdot(a: ComplexArray, b: ComplexArray) -> FloatArray:
        r: FloatArray = alpha * np.einsum("xG,xG->x", a.conj(), b).real
        if real:
            r *= 2.0
            if gcomm.rank == 0:  # G = 0, held by the first domain rank
                r -= alpha * (a[:, 0].real * b[:, 0].real)
        gcomm.sum(r)
        return r

    return proj, apply_s, apply_hs, cdot


def occupied_residual(calc: Any, nocc: int) -> float:
    apply_hs: Callable[[Any, ComplexArray], tuple[ComplexArray, ComplexArray]]
    cdot: Callable[[ComplexArray, ComplexArray], FloatArray]
    _, _, apply_hs, cdot = paw_ops(calc)
    err: float = 0.0
    kpt: Any
    for kpt in calc.wfs.kpt_u:
        V: ComplexArray = kpt.psit.array[:nocc]
        H: ComplexArray
        S: ComplexArray
        H, S = apply_hs(kpt, V)
        R: ComplexArray = H - kpt.eps_n[:nocc, None] * S
        err += kpt.weight * float(cdot(R, R).sum())
    return cast(float, kpar.kcomm(calc).sum_scalar(err) * HA**2 / calc.wfs.nvalence)


def _add_random_bands(calc: Any, nb: int) -> int:
    from gpaw.band_descriptor import BandDescriptor
    from gpaw.wavefunctions.arrays import PlaneWaveExpansionWaveFunctions

    wfs: Any = calc.wfs
    nb0: int = wfs.bd.nbands
    if nb <= nb0:
        return nb0
    bd: Any = BandDescriptor(nb, wfs.bd.comm)
    kpt: Any
    pd: Any = wfs.pd
    g1: int = int(wfs.gd.comm.rank * pd.maxmyng)  # this rank's plane waves in the global order of the k-point
    for kpt in wfs.kpt_u:
        new: Any = PlaneWaveExpansionWaveFunctions(
            nb, pd, wfs.dtype, kpt=kpt.q, dist=(bd.comm, -1, 1), spin=kpt.s, collinear=wfs.collinear
        )
        new.array[:nb0] = kpt.psit.array
        arr: ComplexArray = new.array[nb0:]
        ng: int = int(pd.ng_q[kpt.q])
        g2: int = g1 + int(pd.myng_q[kpt.q])
        w: FloatArray = 1.0 / (1.0 + pd.G2_qG[kpt.q])
        j: int
        # per (k-point, spin, band) over all plane waves: the same start on any number of ranks
        for j in range(nb - nb0):
            rs: np.random.RandomState = np.random.RandomState([1234, int(kpt.k), int(kpt.s), nb0 + j])
            re: FloatArray = rs.uniform(-1, 1, ng)
            arr[j].real = re[g1:g2] * w
            if np.iscomplexobj(arr):
                im: FloatArray = rs.uniform(-1, 1, ng)
                arr[j].imag = im[g1:g2] * w
        if wfs.dtype == float and wfs.gd.comm.rank == 0:  # real wave functions: the G = 0 coefficient is real
            arr[:, 0] = arr[:, 0].real
        kpt.psit = new
        kpt.eps_n = np.concatenate([kpt.eps_n, np.full(nb - nb0, kpt.eps_n[-1])])
        kpt.f_n = np.concatenate([kpt.f_n, np.zeros(nb - nb0)])
    wfs.bd = bd
    if hasattr(wfs.occupations, "bd"):
        wfs.occupations.bd = bd
    wfs._work_array = None
    wfs._work_matrix_nn = None
    wfs.kpt_u[0].projections = None
    wfs.allocate_arrays_for_projections(wfs.pt.my_atom_indices)
    for kpt in wfs.kpt_u:
        kpt.psit.matrix_elements(wfs.pt, out=kpt.projections)
    wfs.orthonormalized = False
    calc.parameters.nbands = nb
    return nb0


def _gen_eigh(
    H: FloatArray | ComplexArray, S: FloatArray | ComplexArray, n: int
) -> tuple[FloatArray, FloatArray | ComplexArray]:
    H = 0.5 * (H + H.conj().T)
    S = 0.5 * (S + S.conj().T)
    s_: FloatArray
    U: FloatArray | ComplexArray
    s_, U = np.linalg.eigh(S)
    keep: BoolArray = s_ > 1e-10 * s_.max()
    T: FloatArray | ComplexArray = U[:, keep] / np.sqrt(s_[keep])
    e: FloatArray
    V: FloatArray | ComplexArray
    e, V = np.linalg.eigh(T.conj().T @ H @ T)
    return e[:n], (T @ V)[:, :n]


def _gen_eigh_shared(
    comm: Any, H: FloatArray | ComplexArray, S: FloatArray | ComplexArray, n: int
) -> tuple[FloatArray, FloatArray | ComplexArray]:
    """_gen_eigh solved on rank 0 of comm and sent to the others: every domain rank applies the same rotation to its plane waves
    (separate solutions could differ within degenerate subspaces)"""
    if comm.size == 1:
        return _gen_eigh(H, S, n)
    e: FloatArray
    C: FloatArray | ComplexArray
    shape: IntArray = np.zeros(2, np.int64)
    if comm.rank == 0:
        e, C = _gen_eigh(H, S, n)
        shape[:] = C.shape
    comm.broadcast(shape, 0)
    if comm.rank != 0:
        e, C = np.empty(int(shape[1])), np.empty((int(shape[0]), int(shape[1])), dtype=np.result_type(H, S))
    C = np.ascontiguousarray(C)
    comm.broadcast(e, 0)
    comm.broadcast(C, 0)
    return e, C


EXTENSION_STATS: dict[str, int] = {}  # of the last locked_band_extension on this rank: blocks, iterations (sweeps)


def locked_band_extension(
    calc: Any,
    nb: int,
    nconv: int,
    tol: float = 1e-2,
    nlock: int | None = None,
    maxiter: int = 200,
    reorth: int = 8,
    block: int = 128,
    guard: int = 4,
) -> None:
    wfs: Any = calc.wfs
    pd: Any = wfs.pd
    if wfs.bd.comm.size != 1 or not wfs.collinear:
        raise NotImplementedError("locked_band_extension: plane waves, collinear, no band parallelisation")
    nb0: int = wfs.bd.nbands
    nlock = nb0 - 4 if nlock is None else nlock
    _add_random_bands(calc, nb)
    precond: Any = wfs.make_preconditioner(1)
    proj: Callable[[Any, ComplexArray], Any]
    apply_hs: Callable[..., tuple[ComplexArray, ComplexArray]]
    proj, _, apply_hs, _ = paw_ops(calc)
    gcomm: Any = wfs.gd.comm

    def paw_overlap(PL: npt.NDArray[Any], PX: Any) -> npt.NDArray[Any]:
        """sum_a <L|p^a> dO^a <p^a|X> (L: projection rows of the locked bands, same atom layout as PX), summed over the domains"""
        M: npt.NDArray[Any] = np.zeros((len(PL), len(PX.array)), dtype=PX.array.dtype)
        a: int
        I1: int
        I2: int
        for a, (I1, I2) in PX.map.items():
            M += PL[:, I1:I2].conj() @ (wfs.setups[a].dO_ii @ PX.array[:, I1:I2].T)
        gcomm.sum(M)
        return M

    blocks: list[tuple[int, int, int]]
    lo: int
    blocks, lo = [], nlock
    while lo < nb:
        hi: int = nb if nb - lo <= block + guard else lo + block
        blocks.append((lo, hi, min(nb, hi + guard)))
        lo = hi
    nconv_all: int = max(nconv - nlock, 1)
    wtot: float = kpar.kcomm(calc).sum_scalar(float(sum(k_.weight for k_ in wfs.kpt_u)))
    bad: int = 0
    EXTENSION_STATS.clear()
    EXTENSION_STATS.update(blocks=0, iterations=0)
    kpt: Any
    for kpt in wfs.kpt_u:
        tol_k: float = tol * kpt.weight / wtot
        # projections of the bands (rows), kept current for the locked ones
        PL_all: npt.NDArray[Any] = kpt.projections.array
        top: int
        for lo, hi, top in blocks:
            nact: int = top - lo
            ncv: int = min(hi, nconv) - lo
            L: ComplexArray = kpt.psit.array[:lo]
            PL: npt.NDArray[Any] = PL_all[:lo]
            st: dict[str, Any] = {}

            def ortho_locked(X: ComplexArray, PX: Any) -> tuple[ComplexArray, Any]:
                """X - L <L|S|X>, with <L|S|X> = <L|X> + PAW term from the stored projections of L; the projections of the result"""
                M: npt.NDArray[Any] = pd.integrate(L, X) + paw_overlap(PL, PX)
                P2: Any = PX.new()
                P2.array[:] = PX.array - M.T @ PL
                return cast(ComplexArray, X - M.T @ L), P2

            def restart(A: ComplexArray) -> None:
                PA: Any
                A, PA = ortho_locked(A, proj(kpt, A))
                HA_: ComplexArray
                SA: ComplexArray
                HA_, SA = apply_hs(kpt, A, PA)
                e: FloatArray
                C: FloatArray | ComplexArray
                e, C = _gen_eigh_shared(wfs.gd.comm, pd.integrate(A, HA_), pd.integrate(A, SA), nact)
                st.update(A=C.T @ A, HA_=C.T @ HA_, SA=C.T @ SA, eps=e, P=None)

            restart(kpt.psit.array[lo:top].copy())
            tol_b: float = tol_k * max(ncv, 0) / nconv_all
            it: int
            for it in range(1, maxiter + 1):
                if it % reorth == 0:
                    restart(st["A"])
                A: ComplexArray
                HA_: ComplexArray
                SA: ComplexArray
                e: FloatArray
                A, HA_, SA, e = st["A"], st["HA_"], st["SA"], st["eps"]
                R: ComplexArray = HA_ - e[:, None] * SA
                err: float = (
                    kpt.weight * float(np.sum(pd.integrate(R[:ncv], R[:ncv]).diagonal().real)) * HA**2 / wfs.nvalence
                    if ncv > 0
                    else 0.0
                )
                if ncv <= 0 or err < tol_b:
                    break
                X: ComplexArray = precond(R, kpt, precond.calculate_kinetic_energy(A, kpt))
                X = X - pd.integrate(SA, X).T @ A  # A is S-orthogonal to the locked bands, so this order is equivalent
                PX: Any
                X, PX = ortho_locked(X, proj(kpt, X))
                HX: ComplexArray
                SX: ComplexArray
                HX, SX = apply_hs(kpt, X, PX)
                V: list[ComplexArray]
                HV: list[ComplexArray]
                SV: list[ComplexArray]
                V, HV, SV = [A, X], [HA_, HX], [SA, SX]
                if st.get("P") is not None:
                    V.append(st["P"])
                    HV.append(st["HP"])
                    SV.append(st["SP"])
                H2: FloatArray | ComplexArray = np.block([[pd.integrate(a, hb) for hb in HV] for a in V])
                S2: FloatArray | ComplexArray = np.block([[pd.integrate(a, sb) for sb in SV] for a in V])
                e2: FloatArray
                C: FloatArray | ComplexArray
                e2, C = _gen_eigh_shared(wfs.gd.comm, H2, S2, nact)
                Cs: list[npt.NDArray[np.inexact[Any]]] = np.split(C, np.cumsum([len(v) for v in V])[:-1])
                P: ComplexArray = cast(ComplexArray, sum(c.T @ v for c, v in zip(Cs[1:], V[1:])))
                HP: ComplexArray = cast(ComplexArray, sum(c.T @ v for c, v in zip(Cs[1:], HV[1:])))
                SP: ComplexArray = cast(ComplexArray, sum(c.T @ v for c, v in zip(Cs[1:], SV[1:])))
                st.update(A=Cs[0].T @ A + P, HA_=Cs[0].T @ HA_ + HP, SA=Cs[0].T @ SA + SP, eps=e2, P=P, HP=HP, SP=SP)
            else:
                bad += 1
            EXTENSION_STATS["blocks"] += 1
            EXTENSION_STATS["iterations"] += it
            kpt.psit.array[lo:top] = st["A"]
            kpt.eps_n[lo:top] = st["eps"]
            PL_all[lo:top] = proj(kpt, st["A"]).array
        if len(blocks) > 1:
            idx: IntArray = nlock + np.argsort(kpt.eps_n[nlock:nb], kind="stable")
            if not np.array_equal(idx, np.arange(nlock, nb)):
                kpt.psit.array[nlock:nb] = kpt.psit.array[idx]
                kpt.eps_n[nlock:nb] = kpt.eps_n[idx]
        kpt.psit.matrix_elements(wfs.pt, out=kpt.projections)
    wfs.calculate_occupation_numbers()
    bad = int(kpar.kcomm(calc).sum_scalar(bad))
    if bad:
        raise RuntimeError(f"locked_band_extension: {bad} (k-point, block) pairs not converged after {maxiter} sweeps")


def converged_bands(calc: Any, nocc: int) -> int:
    """the number of bands the ground state converged: GPAW's convergence 'bands' ('occupied' by default, 'all', n or -n)"""
    nb0: int = int(calc.wfs.bd.nbands)
    conv: Any = (calc.parameters.get("convergence") or {}).get("bands", "occupied")
    if conv == "occupied":
        return nocc
    if conv == "all":
        return nb0
    n: int = int(conv)
    return nb0 + n if n < 0 else min(n, nb0)


def extend_empty_bands(
    calc: Any, factor: float = 2.0, min_extra: int = 16, tol: float = 1e-2, check: float | None = 1e-2
) -> int:
    nocc: int = int(round(calc.get_number_of_electrons() / 2))
    nb: int = nocc + max(min_extra, int(round(factor * nocc)))
    if calc.wfs.bd.nbands >= nb:
        return cast(int, calc.wfs.bd.nbands)
    calc.density.update(calc.wfs)
    calc.hamiltonian.update(calc.density)
    if check is not None:
        r: float = occupied_residual(calc, nocc)
        if not r < check:
            raise RuntimeError(
                f"occupied-band residual {r:.2e} eV^2/e under the rebuilt Hamiltonian: inconsistent ground state"
            )
    nb0: int = int(calc.wfs.bd.nbands)
    locked_band_extension(calc, nb, nb - 4, tol=tol, nlock=min(nb0 - 4, max(nocc, converged_bands(calc, nocc))))
    return nb


# ------------------------------------------------------------------------------------- expectation values of other functionals
class _HamState:
    """every array / atomic-matrix / scalar attribute of a GPAW Hamiltonian and its xc, so that another functional's Hamiltonian can be
    built and the original put back without re-evaluating it"""

    ham: Any
    xc: Any
    saved: dict[str, tuple[Any, Any]]

    def __init__(self, ham: Any) -> None:
        self.ham, self.xc, self.saved = ham, ham.xc, {}
        k: str
        v: Any
        for k, v in vars(ham).items():
            if k == "xc":
                continue
            if isinstance(v, np.ndarray):
                self.saved[k] = (v, v.copy())
            elif hasattr(v, "partition") and hasattr(v, "items"):
                self.saved[k] = (v, {a: np.array(x, copy=True) for a, x in v.items()})
            elif isinstance(v, (float, int, np.floating, np.integer)) and not isinstance(v, bool):
                self.saved[k] = (v, None)

    def restore(self) -> None:
        k: str
        obj: Any
        data: Any
        for k, (obj, data) in self.saved.items():
            if isinstance(obj, np.ndarray):
                obj[...] = data
            elif data is not None:
                a: int
                x: FloatArray
                for a, x in data.items():
                    obj[a][...] = x
            setattr(self.ham, k, obj)
        self.ham.xc = self.xc


def group_expectations(
    calc: Any, groups: Sequence[dict[str, Any]], names: Sequence[str | dict[str, Any]]
) -> dict[str, FloatArray]:
    """band-averaged <psi|H_name|psi> / <psi|S|psi> (Ha) of the groups (dicts with the global u and bands) for each functional (GPAW xc
    names or dicts) at the current density; the calculator's Hamiltonian is restored afterwards.  Collective over the k-point ranks.
    """
    from gpaw.xc import XC

    ham: Any
    dens: Any
    wfs: Any
    ham, dens, wfs = calc.hamiltonian, calc.density, calc.wfs
    dens.calculate_pseudo_charge()
    ham.update(dens)
    base: _HamState = _HamState(ham)
    out: dict[str, FloatArray] = {}
    try:
        name: str | dict[str, Any]
        for name in names:
            xc: Any = XC(name)
            xc.set_grid_descriptor(ham.finegd)
            xc.initialize(dens, ham, wfs)
            xc.set_positions(calc.spos_ac)
            ham.xc = xc
            dens.calculate_pseudo_charge()
            ham.update(dens)
            apply_hs: Callable[[Any, ComplexArray], tuple[ComplexArray, ComplexArray]]
            cdot: Callable[[ComplexArray, ComplexArray], FloatArray]
            _, _, apply_hs, cdot = paw_ops(calc)
            vals: FloatArray = np.zeros(len(groups))
            i: int
            gr: dict[str, Any]
            for i, gr in enumerate(groups):
                q: int | None = kpar.local(calc, gr["u"])
                if q is None:
                    continue
                kpt: Any = wfs.kpt_u[q]
                V: ComplexArray = np.ascontiguousarray(kpt.psit.array[gr["bands"]])
                H: ComplexArray
                S: ComplexArray
                H, S = apply_hs(kpt, V)
                vals[i] = float(np.mean(cdot(V, H) / cdot(V, S)))
            kpar.kcomm(calc).sum(vals)
            out[str(name)] = vals
            base.restore()
    finally:
        base.restore()
    return out


def reference_shifts(
    calc: Any, groups: Sequence[dict[str, Any]], ref_xc: str | dict[str, Any], base_xc: str | dict[str, Any] = "PBE"
) -> list[float]:
    """first-order level shifts <H_ref> - <H_PBE> (eV) of the groups at the current density"""
    ex: dict[str, FloatArray] = group_expectations(calc, groups, [base_xc, ref_xc])
    return [float(v) for v in (ex[str(ref_xc)] - ex[str(base_xc)]) * HA]


def soc_edge_shifts(calc: Any, window: int | None = 8) -> dict[str, Any]:
    """second-variation spin-orbit shifts (Ha) of the VBM and CBM, PBE radial dV/dr in the spheres, basis [nocc - window, nocc + window)
    (window None: all bands)"""
    from gpaw.spinorbit import soc_eigenstates
    from gpaw.xc import XC

    nocc: int = int(round(calc.get_number_of_electrons() / 2))
    eps: FloatArray = kpar.gather_array(calc, [k.eps_n for k in calc.wfs.kpt_u]) * HA
    vbm0: float
    cbm0: float
    vbm0, cbm0 = float(eps[:, nocc - 1].max()), float(eps[:, nocc].min())
    ham: Any = calc.hamiltonian
    xc0: Any = ham.xc
    xc: Any = XC("PBE")
    xc.initialize(calc.density, ham, calc.wfs)
    xc.set_positions(calc.spos_ac)
    ham.xc = xc
    nb: int = calc.wfs.bd.nbands
    n1: int
    n2: int
    n1, n2 = (0, nb) if window is None else (max(0, nocc - int(window)), min(nb, nocc + int(window)))
    try:
        e_km: FloatArray = soc_eigenstates(calc, n1=n1, n2=n2).eigenvalues()
    finally:
        ham.xc = xc0
    iv: int = 2 * (nocc - n1)
    return dict(vbm=(float(e_km[:, iv - 1].max()) - vbm0) / HA, cbm=(float(e_km[:, iv].min()) - cbm0) / HA)
