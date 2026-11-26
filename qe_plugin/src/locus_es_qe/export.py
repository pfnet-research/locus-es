"""Writing the E_S input files (docs/ES_INPUT_FORMAT.md, version 1) from a pw.x PAW run.

step 1:  python -m locus_es_qe points <outdir>/<prefix>.save -o points.npz
step 2:  locus-es fields points.npz -o fields.npz                                  (the E_S package)
step 3:  python -m locus_es_qe export <outdir>/<prefix>.save --fields fields.npz -o es_input.npz [--soc VBM CBM]
step 4:  locus-es run es_input.npz -o es.json --s 0.7084                           (PBE base; the E_S package)
"""

from __future__ import annotations

import hashlib
import json
from math import pi
from typing import Any

import numpy as np

from ._types import ComplexArray, FloatArray, IntArray
from .points import AtomBlock, QePoints
from .qe_files import QeRun, read_run
from .tables import NV_SP, SHELL_N
from .upf import PawDataset, core_kinetic, orbital_kinetic, radial_derivative

VERSION: int = 1
HA: float = 27.211386024367243


def _checksum(*arrays: FloatArray) -> str:
    h: Any = hashlib.sha256()
    a: FloatArray
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return str(h.hexdigest()[:16])


def _fingerprint(ps: QePoints) -> list[float]:
    return [int(ps.w.size), float(ps.w.sum()), float(ps.w @ ps.n), float(ps.w @ ps.sigma), float(ps.w @ ps.tau)]


def _save(path: str, meta: dict[str, Any], **arrays: Any) -> None:
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)


def _elements(ps: QePoints) -> list[str]:
    return [ps.datasets[s].element for s in ps.run.species]


# ---------------------------------------------------------------------------------------------------------------- k-points
def kpoint_arrays(ps: QePoints) -> dict[str, Any]:
    run: QeRun = ps.run
    nk: int = len(run.kpts)
    occ: FloatArray = np.zeros((nk, run.nbnd))
    occ[:, : ps.nocc] = 1.0
    ibzk: FloatArray = run.scaled_kpts()
    N: tuple[int, int, int] = run.mp_grid
    s: tuple[int, int, int] = run.mp_shift
    axes: list[FloatArray] = [(np.arange(N[i]) + 0.5 * s[i]) / N[i] for i in range(3)]
    grid: FloatArray = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    bzk: FloatArray = (grid + 0.5) % 1.0 - 0.5
    bz2ibz: IntArray = np.full(len(bzk), -1, dtype=np.int64)
    j: int
    k: FloatArray
    for j, k in enumerate(bzk):
        d_plus: FloatArray = (ibzk - k + 0.5) % 1.0 - 0.5
        d_minus: FloatArray = (ibzk + k + 0.5) % 1.0 - 0.5
        hit: IntArray = np.where((np.abs(d_plus).max(1) < 1e-6) | (np.abs(d_minus).max(1) < 1e-6))[0]
        if hit.size == 0:
            raise ValueError("the k-points of the run are not the full mesh reduced by time reversal only (use nosym)")
        bz2ibz[j] = int(hit[0])
    return dict(
        eps_kn=run.eps.copy(),
        occ_kn=occ,
        w_k=run.weights.copy(),
        w_k_raw=run.weights.copy(),
        spin_k=np.zeros(nk, dtype=np.int64),
        ibz_of_u=np.arange(nk, dtype=np.int64),
        ibzk_kc=ibzk,
        bzk_kc=bzk,
        bz2ibz_k=bz2ibz,
    )


def _state_info(ps: QePoints, ik: int, bands: list[int]) -> dict[str, Any]:
    run: QeRun = ps.run
    return dict(
        eps=float(np.mean(run.eps[ik, bands])),
        weight=float(run.weights[ik]),
        occupation=float(np.mean([1.0 if b < ps.nocc else 0.0 for b in bands])),
    )


def band_edges(ps: QePoints, tol_ev: float = 1e-3) -> dict[str, dict[str, Any]]:
    """VBM and CBM over all k-points (the first k-point among equal extrema), with the bands degenerate at that k-point"""
    eps: FloatArray = ps.run.eps
    out: dict[str, dict[str, Any]] = {}
    key: str
    for key in ("vbm", "cbm"):
        col: FloatArray = eps[:, ps.nocc - 1] if key == "vbm" else eps[:, ps.nocc]
        ik: int = int(np.argmax(col) if key == "vbm" else np.argmin(col))
        e0: float = float(col[ik])
        bands: list[int] = [int(b) for b in np.where(np.abs(eps[ik] - e0) * HA < tol_ev)[0]]
        out[key] = dict(u=ik, k=ik, s=0, bands=bands, **_state_info(ps, ik, bands))
    if out["cbm"]["eps"] <= out["vbm"]["eps"]:
        raise ValueError("no gap: metals are not supported by this exporter")
    return out


# ------------------------------------------------------------------------------------------------------------- projections
def _bound(b: AtomBlock, c: int) -> bool:
    i: int = b.species.channels[c][0]
    return bool(b.species.ds.projectors[i].bound)


def channel_weights(ps: QePoints) -> tuple[list[list[Any]], FloatArray]:
    """channels (element, l) of the bound projectors and |P|^2 summed over them, per (k, band, channel)"""
    chans: list[tuple[str, int]] = []
    idx: list[tuple[AtomBlock, int, list[int]]] = []
    elements: list[str] = _elements(ps)
    b: AtomBlock
    for b in ps.blocks:
        ells: list[int] = sorted({p.ell for p in b.species.ds.projectors})
        ell: int
        for ell in ells:
            cols: list[int] = [
                b.offset + c for c, (i, el, m) in enumerate(b.species.channels) if el == ell and _bound(b, c)
            ]
            if not cols:
                continue
            key: tuple[str, int] = (elements[b.atom], ell)
            if key not in chans:
                chans.append(key)
            idx.append((b, chans.index(key), cols))
    out: FloatArray = np.zeros((len(ps.P), ps.run.nbnd, len(chans)))
    ik: int
    P: ComplexArray
    for ik, P in enumerate(ps.P):
        c: int
        cols2: list[int]
        for b, c, cols2 in idx:
            out[ik, :, c] += np.sum(np.abs(P[:, cols2]) ** 2, axis=1)
    return [list(c) for c in chans], out


def d_groups(ps: QePoints, xmin: float = 0.5) -> list[dict[str, Any]]:
    """per k-point and element: the occupied bands with bound-d character (occupied d projectors of the dataset) >= xmin"""
    elements: list[str] = _elements(ps)
    out: list[dict[str, Any]] = []
    ik: int
    P: ComplexArray
    for ik, P in enumerate(ps.P):
        p2: FloatArray = np.asarray(np.abs(P) ** 2, dtype=float)
        tot: FloatArray = np.maximum(p2.sum(1), 1e-30)
        xd: dict[str, FloatArray] = {}
        b: AtomBlock
        for b in ps.blocks:
            cols: list[int] = [
                b.offset + c
                for c, (i, ell, m) in enumerate(b.species.channels)
                if ell == 2 and b.species.ds.projectors[i].bound and b.species.ds.projectors[i].occupation > 0
            ]
            if cols:
                el: str = elements[b.atom]
                xd[el] = xd.get(el, np.zeros(len(tot))) + p2[:, cols].sum(1)
        el2: str
        x: FloatArray
        for el2, x in xd.items():
            bands: list[int] = [int(n) for n in np.where((np.arange(len(tot)) < ps.nocc) & (x / tot >= xmin))[0]]
            if bands:
                out.append(
                    dict(kind="d", u=ik, k=ik, bands=bands, element=el2, occupied=True, **_state_info(ps, ik, bands))
                )
    return out


# --------------------------------------------------------------------------------------------------------- free-atom shells
def atom_shell(ds: PawDataset, ell: int) -> dict[str, FloatArray] | None:
    """the free atom of the dataset's generation configuration: normalized density of the shell (n, l), n from SHELL_N, and the
    atomic density, |grad n|^2 and tau on the dataset mesh (weights 4 pi r^2 dr); None if the dataset has no such bound state
    """
    n: int = SHELL_N[ds.element][ell]
    label: str = f"{n}{'SPDF'[ell]}"
    r: FloatArray = ds.r
    rpsi: FloatArray | None = None
    dens: FloatArray = ds.nc.copy()
    tau: FloatArray = core_kinetic(ds)
    p: Any
    for p in ds.projectors:
        if not p.bound:
            continue
        norm: float = float(np.sqrt(np.sum(p.ae**2 * ds.rab)))
        dens = dens + p.occupation * (p.ae / norm) ** 2 / (4.0 * pi * r**2)
        tau = tau + orbital_kinetic(p.ae / norm, p.ell, p.occupation, ds)
        if p.label.upper() == label:
            rpsi = p.ae / norm
    c: Any
    for c in ds.core:
        if c.n == n and c.ell == ell:
            rpsi = c.rpsi
    if rpsi is None:
        return None
    w: FloatArray = 4.0 * pi * r**2 * ds.rab
    return dict(
        r_g=r.copy(),
        w_g=w,
        rho_g=rpsi**2 / (4.0 * pi * r**2),
        n_g=np.clip(dens, 1e-12, None),
        sigma_g=radial_derivative(dens, ds) ** 2,
        tau_g=np.clip(tau, 0.0, None),
    )


# -------------------------------------------------------------------------------------------------------------------- steps
def write_points(ps: QePoints, path: str) -> None:
    run: QeRun = ps.run
    meta: dict[str, Any] = dict(
        format="locus-es-points",
        version=VERSION,
        producer="locus_es_qe",
        checksum=_checksum(ps.w, ps.n, ps.sigma, ps.tau),
        fingerprint=_fingerprint(ps),
        n_points=int(ps.w.size),
        n_grid=int(ps.ngrid),
        grid_N=list(run.fft),
        grid_beg=[0, 0, 0],
        grid_n=list(run.fft),
        cell_bohr=run.cell.tolist(),
        spos=(run.positions @ np.linalg.inv(run.cell)).tolist(),
        symbols=_elements(ps),
        atom_orbit=None,
    )
    _save(path, meta, w=ps.w, n=ps.n, sigma=ps.sigma, tau=ps.tau, atom_p=ps.atom_p)


def write_input(ps: QePoints, fields_path: str, path: str, soc_ev: tuple[float, float] | None = None) -> dict[str, Any]:
    z: Any = np.load(fields_path, allow_pickle=False)
    fm: dict[str, Any] = json.loads(str(z["meta"]))
    if fm.get("format") != "locus-es-fields":
        raise ValueError(f"{fields_path}: not a fields file")
    fp: FloatArray = np.array(fm["points_fingerprint"], dtype=float)
    fq: FloatArray = np.array(_fingerprint(ps), dtype=float)
    if fp[0] != fq[0] or np.max(np.abs(fp[1:] - fq[1:]) / np.maximum(np.abs(fq[1:]), 1e-30)) > 1e-8:
        raise ValueError("the fields were computed on other points (different calculation or settings)")
    u_v: FloatArray = z["u_v"]
    u_c: FloatArray = z["u_c"]
    run: QeRun = ps.run
    edges: dict[str, dict[str, Any]] = band_edges(ps)
    groups: list[dict[str, Any]] = d_groups(ps)
    states: list[dict[str, Any]] = [edges["vbm"], edges["cbm"]] + groups
    lev: FloatArray = np.zeros((len(states), 2))
    norm: FloatArray = np.zeros(len(states))
    i: int
    st: dict[str, Any]
    for i, st in enumerate(states):
        dn: FloatArray = ps.state_density(int(st["u"]), list(st["bands"]))
        lev[i] = [float(ps.w @ (dn * u_v)), float(ps.w @ (dn * u_c))]
        norm[i] = float(ps.w @ dn)
    ka: dict[str, Any] = kpoint_arrays(ps)
    chans: list[list[Any]]
    chanw: FloatArray
    chans, chanw = channel_weights(ps)
    elements: list[str] = _elements(ps)
    shells: dict[str, dict[str, FloatArray]] = {}
    missing: list[str] = []
    w: str
    for w in ("vbm", "cbm"):
        e: dict[str, Any] = edges[w]
        jc: int = int(np.argmax(chanw[e["u"], e["bands"]].sum(0)))
        el: str = str(chans[jc][0])
        ell: int = int(chans[jc][1])
        key: str = f"{el}_{ell}"
        if key in shells or key in missing:
            continue
        ds: PawDataset = next(ps.datasets[s] for s, x in zip(run.species, elements) if x == el)
        sh: dict[str, FloatArray] | None = atom_shell(ds, ell)
        if sh is None:
            missing.append(key)
        else:
            shells[key] = sh
    species_order: list[str] = sorted(set(elements))
    arrays: dict[str, Any] = dict(
        ka,
        pair_n=ps.pair_n,
        pair_m=ps.pair_m,
        p2_kp=np.array(ps.p2),
        chanw_knc=chanw,
        lev=lev,
        norm=norm,
        atom_class=np.array([species_order.index(x) for x in elements], dtype=np.int64),
    )
    ch: str
    sh2: dict[str, FloatArray]
    for ch, sh2 in shells.items():
        k: str
        v: FloatArray
        for k, v in sh2.items():
            arrays[f"shell:{ch}:{k}"] = v
    meta: dict[str, Any] = dict(
        format="locus-es-input",
        version=VERSION,
        producer="locus_es_qe",
        base_functional=f"{run.functional} (Quantum ESPRESSO PAW)",
        points_checksum=fm["points_checksum"],
        fields_checksum=fm["checksum"],
        spin_paired=True,
        n_electrons=float(run.nelec),
        symbols=elements,
        positions_A=(run.positions * 0.5291772105638411).tolist(),
        cell_A=(run.cell * 0.5291772105638411).tolist(),
        cell_bohr=run.cell.tolist(),
        pbc=[True, True, True],
        nbzkpts=int(np.prod(run.mp_grid)),
        nv_bohr3=float(sum(NV_SP[x] for x in elements) / run.volume),
        channels=chans,
        edges={x: {k2: edges[x][k2] for k2 in ("u", "k", "bands", "eps", "occupation")} for x in ("vbm", "cbm")},
        groups=groups,
        window=None,
        soc=None if soc_ev is None else {"vbm": soc_ev[0] / HA, "cbm": soc_ev[1] / HA},
        missing_shells=missing,
        norm_error=ps.norm_error,
    )
    _save(path, meta, **arrays)
    return meta


def points_from_save(save: str) -> QePoints:
    return QePoints(read_run(save))
