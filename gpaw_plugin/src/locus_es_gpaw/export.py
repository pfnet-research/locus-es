"""Exporter of the E_S input from a converged GPAW calculation (see docs/ES_INPUT_FORMAT.md).  GPL-3.0-or-later (uses GPAW).

    step 1:  python -m locus_es_gpaw points gs.gpw -o points.npz
    step 2:  locus-es fields points.npz -o fields.npz                 (the E_S package)
    step 3:  python -m locus_es_gpaw export gs.gpw --fields fields.npz -o es_input.npz [--window 40 16] [--soc-window N]
    step 4:  locus-es run es_input.npz -o es.json                      (the E_S package)

The exporter extracts data only: points and state integrals of the weight fields it is given, eigenvalues, momentum matrix elements,
projection weights, free-atom shells, spin-orbit edge shifts and the first-order shifts of a reference functional.  The model is in
the E_S package.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from math import pi
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from ase.units import Hartree as HA

from . import kpar
from ._types import BoolArray, ComplexArray, FloatArray, IntArray
from .points import PointSet

VERSION: int = 1


# ------------------------------------------------------------------------------------------------------------------------------ io
def _checksum(*arrays: FloatArray) -> str:
    h: hashlib._Hash = hashlib.sha256()
    a: FloatArray
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def _fingerprint(ps: PointSet) -> list[float]:
    """number of points and weighted sums of the fields over the whole calculation (PointSet.fingerprint)"""
    return ps.fingerprint()


def _save(path: str, meta: dict[str, Any], **arrays: npt.NDArray[Any]) -> None:
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)


def _load(path: str, fmt: str) -> tuple[dict[str, Any], dict[str, npt.NDArray[Any]]]:
    z: np.lib.npyio.NpzFile = np.load(path, allow_pickle=False)
    meta: dict[str, Any] = json.loads(str(z["meta"]))
    if meta.get("format") != fmt or int(meta.get("version", 0)) != VERSION:
        raise ValueError(f"{path}: not a {fmt} file of version {VERSION}")
    return meta, {k: z[k] for k in z.files if k != "meta"}


# ------------------------------------------------------------------------------------------------------------------------ k-points
def kpoint_arrays(calc: Any) -> dict[str, npt.NDArray[Any]]:
    """per-k arrays over all k-points (gathered from the k-point ranks, global order u = k * nspins + s)"""
    wfs: Any = calc.wfs
    kd: Any = wfs.kd
    rows: list[tuple[FloatArray, FloatArray, float, int, int]] = kpar.gather(
        calc,
        [
            (np.array(k.eps_n, float), np.array(k.f_n / k.weight, float), float(k.weight), int(k.s), int(k.k))
            for k in wfs.kpt_u
        ],
    )
    w: FloatArray = np.array([r[2] for r in rows], float)
    return dict(
        eps_kn=np.array([r[0] for r in rows]),
        occ_kn=np.array([r[1] for r in rows]),
        w_k=w,
        w_k_raw=w.copy(),
        spin_k=np.array([r[3] for r in rows], int),
        ibz_of_u=np.array([r[4] for r in rows], int),
        ibzk_kc=np.asarray(kd.ibzk_kc, float),
        bzk_kc=np.asarray(kd.bzk_kc, float),
        bz2ibz_k=np.asarray(kd.bz2ibz_k, int),
    )


def _state_info(calc: Any, q: int, bands: list[int]) -> dict[str, float]:
    kpt: Any = calc.wfs.kpt_u[q]
    return dict(
        eps=float(np.mean(kpt.eps_n[bands])),
        weight=float(kpt.weight),
        occupation=float(np.clip(np.mean([kpt.f_n[b] / kpt.weight for b in bands]), 0, 1)),
    )


def band_edges(calc: Any, tol_ev: float = 1e-3) -> dict[str, dict[str, Any]]:
    """the VBM and CBM over all k-points (the first in global order among equal extrema), with the degenerate bands at that k-point"""
    wfs: Any = calc.wfs
    u0: int = kpar.ks0(calc)
    cand: list[tuple[float, float, int]] = []
    q: int | None
    kpt: Any
    for q, kpt in enumerate(wfs.kpt_u):
        occ: BoolArray = kpt.f_n / kpt.weight > 0.5
        cand.append(
            (
                float(kpt.eps_n[occ].max()) if occ.any() else -np.inf,
                float(kpt.eps_n[~occ].min()) if (~occ).any() else np.inf,
                u0 + q,
            )
        )
    cand = kpar.gather(calc, cand)
    out: dict[str, dict[str, Any]] = {}
    key: str
    pick: Callable[[list[tuple[float, float, int]]], tuple[float, float, int]]
    for key, pick in (
        ("vbm", lambda c: max(c, key=lambda t: (t[0], -t[2]))),
        ("cbm", lambda c: min(c, key=lambda t: (t[1], t[2]))),
    ):
        c: tuple[float, float, int] = pick(cand)
        eps: float = c[0] if key == "vbm" else c[1]
        if not np.isfinite(eps):
            raise ValueError(f"no {key} (metal or no empty bands)")
        u: int = int(c[2])
        q = kpar.local(calc, u)
        e: dict[str, Any] | None = None
        if q is not None:
            kpt = wfs.kpt_u[q]
            bands: list[int] = [int(b) for b in np.where(np.abs(kpt.eps_n - eps) * HA < tol_ev)[0]]
            e = dict(u=u, k=int(kpt.k), s=int(kpt.s), bands=bands, **_state_info(calc, q, bands))
        out[key] = kpar.owner_value(calc, e, q is not None)
    return out


# ------------------------------------------------------------------------------------------------------------------- projections
def _proj2(calc: Any, kpt: Any) -> dict[int, FloatArray]:
    """|P_ni|^2 of the atoms whose projections this rank holds (all atoms without domain decomposition)"""
    out: dict[int, FloatArray] = {}
    a: int
    P_ni: Any
    for a, P_ni in kpt.projections.items():
        out[int(a)] = np.abs(np.asarray(P_ni)) ** 2
    return out


def _bound_d_mask(setup: Any) -> BoolArray:
    f_j: Any = getattr(setup, "f_j", None)
    if f_j is None:
        f_j = setup.data.f_j
    m: list[bool] = []
    ell: int
    f: float
    for ell, f in zip(setup.l_j, f_j):
        m += [bool(ell == 2 and f > 0)] * (2 * ell + 1)
    return np.array(m, bool)


def d_groups(calc: Any, xmin: float = 0.5) -> list[dict[str, Any]]:
    """per k-point and element: the occupied bands with bound-d character >= xmin on that element"""
    wfs: Any = calc.wfs
    out: list[dict[str, Any]] = []
    masks: list[BoolArray] = [_bound_d_mask(s_) for s_ in wfs.setups]
    u0: int = kpar.ks0(calc)
    q: int
    kpt: Any
    elems: list[str] = sorted({s_.symbol for s_, m_ in zip(wfs.setups, masks) if m_.any()})
    for q, kpt in enumerate(wfs.kpt_u):
        P2: dict[int, FloatArray] = _proj2(calc, kpt)
        # row 0: all atoms; then the bound d of each element
        acc: FloatArray = np.zeros((1 + len(elems), len(kpt.eps_n)))
        a: int
        P: FloatArray
        for a, P in P2.items():
            acc[0] += P.sum(1)
            if masks[a].any():
                acc[1 + elems.index(wfs.setups[a].symbol)] += P[:, masks[a]].sum(1)
        kpar.domain_sum(calc, acc)
        tot: FloatArray = np.maximum(acc[0], 1e-30)
        occ: BoolArray = kpt.f_n / kpt.weight > 0.5
        j: int
        el: str
        for j, el in enumerate(elems):
            x: FloatArray = acc[1 + j]
            if not x.any():
                continue
            bands: list[int] = [int(n) for n in np.where(occ & (x / tot >= xmin))[0]]
            if bands:
                out.append(
                    dict(
                        kind="d",
                        u=u0 + q,
                        k=int(kpt.k),
                        bands=bands,
                        element=el,
                        occupied=True,
                        **_state_info(calc, q, bands),
                    )
                )
    return kpar.gather(calc, out)


def window_groups(calc: Any, a: int, b: int, tol_ev: float = 1e-3) -> list[dict[str, Any]]:
    """every band of [nocc - a, nocc + b) at every k-point, degenerate bands grouped"""
    out: list[dict[str, Any]] = []
    u0: int = kpar.ks0(calc)
    q: int
    kpt: Any
    for q, kpt in enumerate(calc.wfs.kpt_u):
        e: FloatArray = kpt.eps_n
        nocc_k: int = int((kpt.f_n / kpt.weight > 0.5).sum())
        n: int
        n2: int
        n, n2 = max(0, nocc_k - a), min(len(e), nocc_k + b)
        while n > 0 and abs(e[n - 1] - e[n]) * HA < tol_ev:
            n -= 1
        while n2 < len(e) and abs(e[n2] - e[n2 - 1]) * HA < tol_ev:
            n2 += 1
        while n < n2:
            grp: list[int] = [n]
            while grp[-1] + 1 < n2 and abs(e[grp[-1] + 1] - e[n]) * HA < tol_ev:
                grp.append(grp[-1] + 1)
            out.append(
                dict(
                    kind="window",
                    u=u0 + q,
                    k=int(kpt.k),
                    bands=grp,
                    occupied=bool(grp[0] < nocc_k),
                    **_state_info(calc, q, grp),
                )
            )
            n = grp[-1] + 1
    return kpar.gather(calc, out)


def channel_weights(calc: Any) -> tuple[list[list[Any]], FloatArray]:
    """channels (element, l) of the bound projectors and |P|^2 summed over them, per (k, band, channel)"""
    from gpaw.dos import get_projector_numbers

    wfs: Any = calc.wfs
    chans: list[tuple[str, int]]
    idx: list[tuple[int, int, list[int]]]
    chans, idx = [], []
    a: int
    setup: Any
    for a, setup in enumerate(wfs.setups):
        ell: int
        for ell in sorted(set(setup.l_j)):
            pn: list[int] = get_projector_numbers(setup, int(ell))
            if not pn:
                continue
            key: tuple[str, int] = (setup.symbol, int(ell))
            if key not in chans:
                chans.append(key)
            idx.append((a, chans.index(key), list(pn)))
    out: FloatArray = np.zeros((len(wfs.kpt_u), len(wfs.kpt_u[0].eps_n), len(chans)))
    q: int
    kpt: Any
    for q, kpt in enumerate(wfs.kpt_u):
        P2: dict[int, FloatArray] = _proj2(calc, kpt)
        c: int
        for a, c, pn in idx:
            if a in P2:
                out[q, :, c] += P2[a][:, pn].sum(1)
    kpar.domain_sum(calc, out)
    return [list(c) for c in chans], kpar.gather_array(calc, list(out))


def atom_weights(calc: Any, groups: list[dict[str, Any]]) -> FloatArray:
    """|P|^2 per atom summed over the bands of each group (global u), every group from the rank that holds its k-point"""
    out: FloatArray = np.zeros((len(groups), len(calc.atoms)))
    i: int
    gr: dict[str, Any]
    for i, gr in enumerate(groups):
        q: int | None = kpar.local(calc, gr["u"])
        if q is not None:
            P2: dict[int, FloatArray] = _proj2(calc, calc.wfs.kpt_u[q])
            a: int
            P: FloatArray
            for a, P in P2.items():
                out[i, a] = float(P[gr["bands"]].sum())
    kpar.domain_sum(calc, out)
    kpar.kcomm(calc).sum(out)
    return out


def symmetry_orbits(calc: Any) -> IntArray | None:
    """orbit of every atom under the symmetry operations of the calculation (None without symmetry)"""
    nat: int = len(calc.atoms)
    a_sa: IntArray = np.asarray(getattr(calc.wfs.kd.symmetry, "a_sa", np.arange(nat)[None, :]))
    if not (a_sa.ndim == 2 and a_sa.shape[0] > 1 and a_sa.shape[1] == nat):
        return None
    cls: IntArray = np.arange(nat)

    def root(x: int) -> int:
        while cls[x] != x:
            x = cls[x]
        return x

    perm: IntArray
    for perm in a_sa:
        a: int
        b: int
        for a, b in enumerate(perm):
            ra: int
            rb: int
            ra, rb = root(a), root(b)
            if ra != rb:
                cls[max(ra, rb)] = min(ra, rb)
    return np.array([root(a) for a in range(nat)])


def atom_classes(calc: Any) -> IntArray:
    """classes of equivalent atoms: orbits of the symmetry operations, or the chemical species without symmetry"""
    orb: IntArray | None = symmetry_orbits(calc)
    if orb is not None:
        return orb
    names: list[str] = calc.atoms.get_chemical_symbols()
    return np.array([sorted(set(names)).index(x) for x in names])


# --------------------------------------------------------------------------------------------------------------- free-atom shells
def _shell_n(sym: str, ell: int) -> int:
    from gpaw.atom.configurations import configurations

    cfg: list[tuple[int, int, float, float]] = configurations[sym][1]
    occ: list[int] = [n for n, ll, f, e in cfg if ll == ell and f > 0]
    if occ:
        return max(occ)
    ns: int = max(n for n, ll, f, e in cfg if ll == 0 and f > 0)
    return max(ell + 1, {0: ns + 1, 1: ns, 2: ns - 1, 3: ns - 2}[ell])


def _atom_tau(at: Any) -> FloatArray:
    """kinetic-energy density of the atom, sum over the occupied orbitals of f [(dphi/dr)^2 + l (l + 1) (phi / r)^2] / (8 pi)"""
    rgd: Any = at.rgd
    tau: FloatArray = np.zeros(rgd.N)
    ch: Any
    for ch in at.channels:
        r: FloatArray = ch.basis.rgd.r_g
        n: int
        f: float
        for n, f in enumerate(ch.f_n):
            if f == 0:
                continue
            phi: FloatArray = ch.phi_ng[n]
            t: FloatArray = ch.basis.rgd.derivative(phi) ** 2 / (8 * pi)
            if ch.l > 0:
                t[1:] += ch.l * (ch.l + 1) * (phi[1:] / r[1:]) ** 2 / (8 * pi)
            tau += f * t
    return tau


def atom_shell(sym: str, l: int) -> dict[str, FloatArray]:  # noqa: E741
    """the neutral all-electron atom (PBE, spin-paired, scalar-relativistic): normalized density of the shell (sym, l) and the
    atomic density, |grad n|^2 and tau on its radial grid, with the weights 4 pi r^2 dr"""
    ell: int = l
    import io as _io

    from gpaw.atom.aeatom import AllElectronAtom
    from gpaw.atom.configurations import configurations

    n: int = _shell_n(sym, ell)
    at: Any = AllElectronAtom(sym, xc="PBE", log=_io.StringIO())
    if not any(nn == n and ll == ell for nn, ll, f, e in configurations[sym][1]):
        at.add(n, ell, 0.0)
    at.initialize()
    at.run()
    at.scalar_relativistic = True
    at.refine()
    rgd: Any = at.rgd
    ch: Any = [c for c in at.channels if c.l == ell and c.s == 0][0]
    rho: FloatArray = ch.phi_ng[n - ell - 1] ** 2 / (4 * pi)
    rho = rho / rgd.integrate(rho)
    tau_g: FloatArray = _atom_tau(at)
    w_g: FloatArray = 4 * pi * rgd.r_g**2 * rgd.dr_g
    w_g[0] = 0.0
    return dict(
        r_g=np.array(rgd.r_g),
        w_g=w_g,
        rho_g=np.array(rho),
        n_g=np.clip(at.n_sg.sum(0), 1e-12, None),
        sigma_g=rgd.derivative(at.n_sg.sum(0)) ** 2,
        tau_g=np.clip(tau_g, 0.0, None),
    )


def valence_sp_density(calc: Any) -> float:
    from gpaw.atom.configurations import configurations

    tot: float = 0.0
    s_: str
    for s_ in calc.atoms.get_chemical_symbols():
        cfg: list[tuple[int, int, float, float]] = configurations[s_][1]
        ns: int = max(n for n, ll, f, e in cfg if ll == 0 and f > 0)
        tot += sum(f for n, ll, f, e in cfg if n == ns and ll in (0, 1))
    return cast(float, tot / abs(np.linalg.det(calc.wfs.gd.cell_cv)))


# ---------------------------------------------------------------------------------------------------------- momentum matrix elements
def band_pairs(occ_kn: FloatArray) -> tuple[IntArray, IntArray]:
    frac: FloatArray = np.clip(occ_kn, 0.0, 1.0)
    iu: IntArray
    ju: IntArray
    iu, ju = np.triu_indices(frac.shape[1], k=1)
    keep: BoolArray = (frac[:, iu] * (1.0 - frac[:, ju])).max(0) > 1e-8
    return iu[keep], ju[keep]


def _rows(C: ComplexArray, idx: IntArray) -> ComplexArray:
    """C[idx] as a view when idx is a contiguous range (the occupied or the empty bands), a copy otherwise"""
    if idx.size and int(idx[-1]) - int(idx[0]) + 1 == idx.size:
        return C[int(idx[0]) : int(idx[-1]) + 1]
    return C[idx]


def momentum_sq(calc: Any, kpt: Any, iu: IntArray, ju: IntArray) -> FloatArray:
    """|<n|p|m>|^2 averaged over the three directions for the pairs (iu, ju) at one k-point (plane-wave part + PAW correction).
    Collective over the domain ranks of the k-point."""
    wfs: Any = calc.wfs
    pd: Any = wfs.pd
    rows: IntArray = np.unique(iu)
    cols: IntArray = np.unique(ju)
    ri: IntArray = np.searchsorted(rows, iu)
    ci: IntArray = np.searchsorted(cols, ju)
    C: ComplexArray = np.asarray(kpt.psit_nG)
    Cr: ComplexArray = _rows(C, rows)
    Cc: ComplexArray = _rows(C, cols)
    kG: FloatArray = pd.get_reciprocal_vectors(q=kpt.q, add_q=True)
    alpha: float = wfs.gd.dv / wfs.gd.N_c.prod()
    paw: list[tuple[Any, FloatArray | ComplexArray]] = []
    a: int
    P_ni: Any
    for a, P_ni in kpt.projections.items():
        setup: Any = wfs.setups[int(a)]
        if getattr(setup, "nabla_iiv", None) is not None:
            paw.append((setup, np.asarray(P_ni)))
    Pc: FloatArray | ComplexArray | None = np.hstack([P[cols] for _, P in paw]) if paw else None
    out: FloatArray = np.zeros(len(iu))
    v: int
    for v in range(3):
        M: ComplexArray = (Cr.conj() * kG[:, v]) @ Cc.T
        M = 2j * M.imag * alpha if pd.dtype == float else M * alpha
        if Pc is not None:
            Pr: FloatArray | ComplexArray = np.hstack([P[rows].conj() @ s_.nabla_iiv[:, :, v] for s_, P in paw])
            M -= 1j * (Pr @ Pc.T)
        kpar.domain_sum(calc, M)
        out += np.abs(M[ri, ci]) ** 2
    return out / 3.0


# ----------------------------------------------------------------------------------------------------------------------- steps
def point_geometry(calc: Any) -> dict[str, Any]:
    """where the points are: the whole fine grid (point i at (i / N) . cell for i in [beg, beg + n), C order; beg = 1
    along a non-periodic axis), the scaled atomic positions and the symmetry orbits (for a screening g(r))"""
    gd: Any = calc.hamiltonian.finegd
    pbc: IntArray = np.asarray(gd.pbc_c, int)
    orb: IntArray | None = symmetry_orbits(calc)
    meta: dict[str, Any] = dict(
        grid_N=[int(x) for x in gd.N_c],
        grid_beg=[int(x) for x in 1 - pbc],
        grid_n=[int(x) for x in gd.get_size_of_global_array()],
        cell_bohr=np.asarray(gd.cell_cv).tolist(),
        spos=np.asarray(calc.spos_ac).tolist(),
        symbols=calc.atoms.get_chemical_symbols(),
        atom_orbit=None if orb is None else orb.tolist(),
    )
    return meta


def write_points(calc: Any, path: str, ps: PointSet | None = None) -> PointSet:
    """the points file, in the serial order on any number of ranks"""
    ps = ps or PointSet(calc)
    fp: list[float] = _fingerprint(ps)
    g: dict[str, npt.NDArray[Any]] | None = ps.gather()
    if not kpar.is_master(calc):
        return ps
    if g is None:
        raise RuntimeError("write_points: the master rank is not the root of its domain communicator")
    meta: dict[str, Any] = dict(
        format="locus-es-points",
        version=VERSION,
        producer="locus_es_gpaw",
        checksum=_checksum(g["w"], g["n"], g["sigma"], g["tau"]),
        fingerprint=fp,
        n_points=int(g["w"].size),
        n_grid=int(np.prod(calc.hamiltonian.finegd.get_size_of_global_array())),
        **point_geometry(calc),
    )
    _save(path, meta, w=g["w"], n=g["n"], sigma=g["sigma"], tau=g["tau"], atom_p=g["atom_p"])
    return ps


def write_input(
    calc: Any,
    fields_path: str,
    path: str,
    ps: PointSet | None = None,
    window: Sequence[int] | None = None,
    d_states: bool = True,
    soc: dict[str, Any] | None = None,
    d_ref: Callable[[Any, list[dict[str, Any]]], list[float]] | None = None,
    base_functional: str | None = None,
) -> dict[str, Any] | None:
    """soc: dict(vbm=, cbm=) in Ha (or None); d_ref: function(calc, groups) -> first-order shifts of the reference functional (eV)"""
    ps = ps or PointSet(calc)
    fm: dict[str, Any]
    fa: dict[str, FloatArray]
    fm, fa = _load(fields_path, "locus-es-fields")
    fp: FloatArray
    fq: FloatArray
    fp, fq = np.array(fm["points_fingerprint"]), np.array(_fingerprint(ps))
    if fp[0] != fq[0] or np.max(np.abs(fp[1:] - fq[1:]) / np.maximum(np.abs(fq[1:]), 1e-30)) > 1e-8:
        raise ValueError("the fields were computed on other points (different calculation or settings)")
    idx: IntArray = ps.global_index()  # the fields on this rank's points
    u_v: FloatArray
    u_c: FloatArray
    u_v, u_c = fa["u_v"][idx], fa["u_c"][idx]
    del fa
    edges: dict[str, dict[str, Any]] = band_edges(calc)
    groups: list[dict[str, Any]] = (d_groups(calc) if d_states else []) + (
        window_groups(calc, *window) if window else []
    )
    if d_ref is not None:
        dg: list[dict[str, Any]] = [gr for gr in groups if gr["kind"] == "d"]
        gr: dict[str, Any]
        v: float | FloatArray
        for gr, v in zip(dg, d_ref(calc, dg) if dg else []):
            gr["d_ref_ev"] = float(v)
    states: list[dict[str, Any]] = [edges["vbm"], edges["cbm"]] + groups
    lev: FloatArray
    norm: FloatArray
    lev, norm = np.zeros((len(states), 2)), np.zeros(len(states))
    i: int
    st: dict[str, Any]
    for i, st in enumerate(states):  # each state on the ranks that hold its k-point, each on its points
        q: int | None = kpar.local(calc, st["u"])
        if q is None:
            continue
        dn: FloatArray = ps.state_density(q, st["bands"])
        lev[i] = [float(ps.w @ (dn * u_v)), float(ps.w @ (dn * u_c))]
        norm[i] = float(ps.w @ dn)
    kpar.domain_sum(calc, lev)
    kpar.domain_sum(calc, norm)
    kpar.kcomm(calc).sum(lev)
    kpar.kcomm(calc).sum(norm)
    ka: dict[str, npt.NDArray[Any]] = kpoint_arrays(calc)
    iu: IntArray
    ju: IntArray
    iu, ju = band_pairs(ka["occ_kn"])
    p2: FloatArray = kpar.gather_array(calc, [momentum_sq(calc, k, iu, ju) for k in calc.wfs.kpt_u])
    chans: list[list[Any]]
    chanw: FloatArray
    chans, chanw = channel_weights(calc)
    if window:
        aw: FloatArray = atom_weights(calc, groups)
        aw[[gr["kind"] != "window" for gr in groups]] = 0.0
    if not kpar.is_master(calc):
        return None
    shells: dict[str, dict[str, FloatArray]] = {}
    w: str
    for w in ("vbm", "cbm"):
        e: dict[str, Any] = edges[w]
        jc: int = int(np.argmax(chanw[e["u"], e["bands"]].sum(0)))
        el: str
        ell: int
        el, ell = chans[jc]
        if f"{el}_{ell}" not in shells:
            shells[f"{el}_{ell}"] = atom_shell(el, ell)
    arrays: dict[str, npt.NDArray[Any]] = dict(
        ka, pair_n=iu, pair_m=ju, p2_kp=p2, chanw_knc=chanw, lev=lev, norm=norm, atom_class=atom_classes(calc)
    )
    if window:
        arrays["atomw_gi"] = aw
    ch: str
    sh: dict[str, FloatArray]
    for ch, sh in shells.items():
        k: str
        for k, v in sh.items():
            arrays[f"shell:{ch}:{k}"] = v
    atoms: Any = calc.atoms
    meta: dict[str, Any] = dict(
        format="locus-es-input",
        version=VERSION,
        producer="locus_es_gpaw",
        base_functional=base_functional,
        points_checksum=fm["points_checksum"],
        fields_checksum=fm["checksum"],
        spin_paired=bool(calc.wfs.nspins == 1),
        n_electrons=float(calc.wfs.nvalence),
        symbols=atoms.get_chemical_symbols(),
        positions_A=atoms.positions.tolist(),
        cell_A=np.asarray(atoms.cell).tolist(),
        cell_bohr=np.asarray(calc.wfs.gd.cell_cv).tolist(),
        pbc=[bool(x) for x in atoms.pbc],
        nbzkpts=int(calc.wfs.kd.nbzkpts),
        nv_bohr3=valence_sp_density(calc),
        channels=chans,
        edges={w: {k: edges[w][k] for k in ("u", "k", "bands", "eps", "occupation")} for w in ("vbm", "cbm")},
        groups=groups,
        window=list(window) if window else None,
        soc=soc,
    )
    _save(path, meta, **arrays)
    return meta
