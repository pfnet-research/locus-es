"""E_S across interfaces (heterojunctions, superlattices): a screening field g(r) from the bulk g of each material.

    g(r) = sum_M g_M h_M(r),      h_M = rho_M / sum_M' rho_M'   (Hirshfeld-type layer weights)

g_M: E_S screening of bulk material M at the lattice of the interface cell; rho_M: Gaussians exp(-r^2 / sigma^2)
(sigma = 1.2 A) on the atoms of M and their periodic images.  On the one-centre (PAW sphere) points of an atom, h_M is 1
for the atom's own material and 0 otherwise.  Correction V_S = (1 - P) W_C (1 - P) - P W_V P, W_X = s kX g(r) u_X(r)
(P: projector on the occupied states), in a window of states at each k-point:

    first order:     eps_n + <n|V_S|n>
    diagonalized:    eigenvalues of diag(eps) + V_S in the occupied block and in the empty block

Layer edges: the highest occupied / lowest empty states with weight <n|h_M|n> > `select` (0.7); an edge is defined
only if its weight is >= `min_weight` (0.75) before and after the diagonalization (otherwise it is resonant with the
other material).

Steps (docs/ES_INPUT_FORMAT.md):

    exporter  points gs.gpw -o points.npz
    E_S       locus-es fields points.npz --layers layers.json -o fields.npz        (u_V, u_C and h_M u_V, h_M u_C, h_M)
    exporter  window gs.gpw --fields fields.npz -o window.npz [--occ 24 --emp 16]   (matrices <m|F|n> in the window)
    E_S       locus-es interface window.npz --layers layers.json -o iface.json
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from math import pi
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from ._types import ComplexArray, FloatArray, IntArray
from .params import BOHR, HA

SIGMA_A: float = 1.2


# ------------------------------------------------------------------------------------------------------------------------- layers
def read_layers(path: str | os.PathLike[str]) -> dict[str, Any]:
    """layers.json -> dict(sigma_A, layers=[{name, atoms, g, bulk}]).

    {"sigma_A": 1.2,
     "layers": [{"name": "AlAs", "atoms": [0, 1, ...], "g": 0.1267},
                {"name": "GaAs", "atoms": "rest",       "g": "GaAs_bulk.es.json"}]}

    atoms: 0-based atom indices, or "rest" (all other atoms; one layer at most).  g: a number, or the es.json of the
    bulk (locus-es run), which also gives the bulk band-edge shifts.  Relative paths are from the directory of
    layers.json.  g is not needed for 'locus-es fields'."""
    spec: dict[str, Any] = json.load(open(path))
    base: str = os.path.dirname(os.path.abspath(path))
    out: list[dict[str, Any]] = []
    L: dict[str, Any]
    for L in spec["layers"]:
        d: dict[str, Any] = dict(name=str(L["name"]), atoms=L["atoms"], g=None, bulk=None)
        g: str | float | None = L.get("g")
        if isinstance(g, str):
            es: dict[str, Any] = json.load(open(g if os.path.isabs(g) else os.path.join(base, g)))
            d["g"] = float(es["g"])
            d["bulk"] = {w: float(es["edges"][w]["shift_ev"]) for w in ("vbm", "cbm")}
        elif g is not None:
            d["g"] = float(g)
        out.append(d)
    return dict(sigma_A=float(spec.get("sigma_A", SIGMA_A)), layers=out)


def layer_index(layers: list[dict[str, Any]], natoms: int, orbits: npt.ArrayLike | None = None) -> IntArray:
    """(natoms,) layer of every atom.  Layers must be unions of symmetry orbits: the window matrices are computed at the
    irreducible k-points without symmetrization."""
    lay: IntArray = np.full(natoms, -1, int)
    rest: list[int] = [i for i, L in enumerate(layers) if isinstance(L["atoms"], str)]
    if len(rest) > 1 or any(layers[i]["atoms"] != "rest" for i in rest):
        raise ValueError("layers: 'atoms' is a list of indices, or \"rest\" for one layer")
    i: int
    L: dict[str, Any]
    for i, L in enumerate(layers):
        if i in rest:
            continue
        idx: IntArray = np.asarray(L["atoms"], int)
        if idx.size and (idx.min() < 0 or idx.max() >= natoms):
            raise ValueError(f"layer {L['name']}: atom index out of range (0..{natoms - 1})")
        if (lay[idx] >= 0).any():
            raise ValueError(f"layer {L['name']}: atoms {sorted(set(idx[lay[idx] >= 0].tolist()))} are in two layers")
        lay[idx] = i
    if rest:
        lay[lay < 0] = rest[0]
    if (lay < 0).any():
        raise ValueError(f"atoms {np.flatnonzero(lay < 0).tolist()} are in no layer")
    if orbits is not None:
        orbits = np.asarray(orbits, int)
        o: int
        for o in set(orbits.tolist()):
            if len(set(lay[orbits == o].tolist())) > 1:
                raise ValueError(
                    "a layer splits a set of symmetry-equivalent atoms: choose layers that are unions of symmetry orbits, "
                    "or run the ground state without symmetry"
                )
    return lay


def promolecular_grid(
    cell_bohr: npt.ArrayLike, spos: npt.ArrayLike, grid_N: Iterable[int], sigma_bohr: float
) -> FloatArray:
    """sum over the given atoms and all periodic images of exp(-|r - R|^2 / sigma^2) on the full periodic grid N (point i at
    (i / N) . cell), exactly through its Fourier series"""
    cell: FloatArray = np.asarray(cell_bohr, float)
    N: list[int] = [int(x) for x in grid_N]
    vol: float = abs(np.linalg.det(cell))
    B: FloatArray = 2 * pi * np.linalg.inv(cell).T  # rows: reciprocal vectors
    m: list[FloatArray] = [np.fft.fftfreq(n, 1.0 / n) for n in N]
    G: FloatArray = (
        m[0][:, None, None, None] * B[0] + m[1][None, :, None, None] * B[1] + m[2][None, None, :, None] * B[2]
    )
    c: ComplexArray = np.zeros(N, complex)
    s: FloatArray
    for s in np.asarray(spos, float).reshape(-1, 3):
        c += (
            np.exp(-2j * pi * m[0] * s[0])[:, None, None]
            * np.exp(-2j * pi * m[1] * s[1])[None, :, None]
            * np.exp(-2j * pi * m[2] * s[2])[None, None, :]
        )
    c *= pi**1.5 * sigma_bohr**3 / vol * np.exp(-(G**2).sum(-1) * sigma_bohr**2 / 4)
    return np.fft.ifftn(c).real * np.prod(N)


def layer_weights(
    meta: dict[str, Any], atom_p: npt.ArrayLike, lay: IntArray, nlayers: int, sigma_A: float = SIGMA_A
) -> FloatArray:
    """(nlayers, P) weights h_M on the points of an exporter: Hirshfeld-type on the grid, 1 / 0 on the one-centre points of the atoms of
    M / of other layers"""
    k: str
    for k in ("grid_N", "grid_beg", "grid_n", "cell_bohr", "spos"):
        if k not in meta:
            raise ValueError(
                f"points file without '{k}': write it again with this version of the exporter (layer weights need the grid)"
            )
    atom_p = np.asarray(atom_p, int)
    beg: list[int]
    n: list[int]
    beg, n = [int(x) for x in meta["grid_beg"]], [int(x) for x in meta["grid_n"]]
    spos: FloatArray = np.asarray(meta["spos"], float)
    ngrid: int = int(np.prod(n))
    if not (atom_p[:ngrid] < 0).all() or (atom_p[ngrid:] < 0).any():
        raise ValueError("points: atom_p must be -1 on the grid points (first) and the atom index on the sphere points")
    sl: tuple[slice, ...] = tuple(slice(b, b + k) for b, k in zip(beg, n))
    rho: FloatArray = np.array(
        [
            promolecular_grid(meta["cell_bohr"], spos[lay == M], meta["grid_N"], sigma_A / BOHR)[sl].ravel()
            for M in range(nlayers)
        ]
    )
    rho = np.clip(rho, 0.0, None)
    h_grid: FloatArray = rho / np.maximum(rho.sum(0), 1e-300)
    h: FloatArray = np.zeros((nlayers, atom_p.size))
    h[:, :ngrid] = h_grid
    sph: IntArray = atom_p[ngrid:]
    M: int
    for M in range(nlayers):
        h[M, ngrid:] = lay[sph] == M
    return h


def layer_fields(
    points_meta: dict[str, Any],
    points: dict[str, npt.NDArray[Any]],
    u_v: FloatArray,
    u_c: FloatArray,
    layers_path: str | os.PathLike[str],
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    """the fields of 'locus-es fields --layers': {uv:<M>, uc:<M>, h:<M>} for every layer M, and the meta entries describing them"""
    spec: dict[str, Any] = read_layers(layers_path)
    layers: list[dict[str, Any]] = spec["layers"]
    if "spos" not in points_meta or "atom_p" not in points:
        raise ValueError("points file without geometry: write it again with this version of the exporter")
    nat: int = len(points_meta["spos"])
    lay: IntArray = layer_index(layers, nat, points_meta.get("atom_orbit"))
    h: FloatArray = layer_weights(points_meta, points["atom_p"], lay, len(layers), spec["sigma_A"])
    arrays: dict[str, FloatArray] = {}
    M: int
    L: dict[str, Any]
    for M, L in enumerate(layers):
        arrays[f"uv:{L['name']}"] = h[M] * u_v
        arrays[f"uc:{L['name']}"] = h[M] * u_c
        arrays[f"h:{L['name']}"] = h[M]
    meta: dict[str, Any] = dict(
        layers=[L["name"] for L in layers],
        layer_of_atom=lay.tolist(),
        sigma_A=spec["sigma_A"],
        matrix_fields=list(arrays),
    )
    return meta, arrays


# ------------------------------------------------------------------------------------------------------------------------ levels
def _edges(
    states: dict[str, list[tuple[float, FloatArray]]], nlayers: int, select: float
) -> dict[tuple[str, int], tuple[float, float] | None]:
    """per layer: the highest occupied / lowest empty (value, weight on the layer) among the states with weight > select on it"""
    out: dict[tuple[str, int], tuple[float, float] | None] = {}
    blk: str
    pick: Callable[..., tuple[float, float]]
    M: int
    for blk, pick in (("occ", max), ("emp", min)):
        for M in range(nlayers):
            c: list[tuple[float, float]] = [(v, w[M]) for v, w in states[blk] if w[M] > select]
            out[(blk, M)] = pick(c, key=lambda x: x[0]) if c else None
    return out


def interface(
    win: dict[str, Any],
    layers: list[dict[str, Any]],
    P: dict[str, Any],
    g_one: float | None = None,
    select: float = 0.7,
    min_weight: float = 0.75,
) -> dict[str, Any]:
    """layer-resolved band edges with g(r) from the window file of the exporter (dict of meta + arrays)"""
    names: list[str] = win["layers"]
    if [L["name"] for L in layers] != list(names):
        raise ValueError(f"layers {[L['name'] for L in layers]} do not match those of the window file {list(names)}")
    if any(L["g"] is None for L in layers):
        raise ValueError("every layer needs its g (a number or the bulk es.json)")
    gM: FloatArray = np.array([L["g"] for L in layers])
    s: float
    kV: float
    kC: float
    s, kV, kC = P["s"], P["kV"], P["kC"]
    nl: int = len(names)
    methods: tuple[str, ...] = ("ks", "one", "local", "diag") if g_one is not None else ("ks", "local", "diag")
    states: dict[str, dict[str, list[tuple[float, FloatArray]]]] = {m: {"occ": [], "emp": []} for m in methods}
    kpts: list[dict[str, Any]] = []
    u: int
    blk: str
    X: str
    sign: float
    ev: FloatArray
    C: ComplexArray
    for u in range(len(win["w_k"])):
        rec: dict[str, Any] = dict(u=u, k=int(win["ibz_of_u"][u]), s=int(win["spin_k"][u]), weight=float(win["w_k"][u]))
        for blk, X, sign in (("occ", "uv", -kV), ("emp", "uc", kC)):
            eps: FloatArray = np.asarray(win[f"eps_{blk}"][u], float) * HA
            WM: ComplexArray = np.array([win[f"M:{X}:{n}:{blk}"][u] for n in names])  # (layers, nb, nb)
            HM: ComplexArray = np.array([win[f"M:h:{n}:{blk}"][u] for n in names])
            V: ComplexArray = s * sign * np.tensordot(gM, WM, 1)  # eV (u in bohr^-1, kX in eV bohr)
            V = 0.5 * (V + V.conj().T)
            first: FloatArray = eps + np.real(np.diag(V))
            ev, C = np.linalg.eigh(np.diag(eps) + V)
            w_ks: FloatArray = np.real(np.einsum("Mnn->nM", HM))
            w_diag: FloatArray = np.real(np.einsum("mi,Mmn,ni->iM", C.conj(), HM, C))
            r: dict[str, Any] = dict(
                bands=np.asarray(win[f"bands_{blk}"][u]).tolist(),
                ks=eps.tolist(),
                local=first.tolist(),
                diag=ev.tolist(),
                weight_ks=w_ks.tolist(),
                weight_diag=w_diag.tolist(),
            )
            states["ks"][blk] += list(zip(eps, w_ks))
            states["local"][blk] += list(zip(first, w_ks))
            states["diag"][blk] += list(zip(ev, w_diag))
            if g_one is not None:
                one: FloatArray = eps + s * sign * g_one * np.real(np.einsum("Mnn->n", WM))
                r["one"] = one.tolist()
                states["one"][blk] += list(zip(one, w_ks))
            rec[blk] = r
        kpts.append(rec)
    E: dict[str, dict[tuple[str, int], Any]] = {m: _edges(states[m], nl, select) for m in methods}
    out: dict[str, Any] = dict(
        layers=[dict(name=L["name"], g=L["g"], bulk_shift_ev=L["bulk"]) for L in layers],
        g_one=g_one,
        s=s,
        kV=kV,
        kC=kC,
        select=select,
        min_weight=min_weight,
        edges={},
        offsets={},
        kpts=kpts,
    )
    edge: str
    M: int
    n: str
    i: int
    a: str
    b: str
    ea: dict[str, Any]
    eb: dict[str, Any]
    ba: dict[str, float] | None
    bb: dict[str, float] | None
    for blk, edge in (("occ", "vbm"), ("emp", "cbm")):
        wmin: float = min(
            (E[m][(blk, M)][1] for m in ("ks", "diag") for M in range(nl) if E[m][(blk, M)] is not None), default=0.0
        )
        defined: bool = all(E["ks"][(blk, M)] is not None for M in range(nl)) and wmin >= min_weight
        per: dict[str, dict[str, Any]] = {}
        for M, n in enumerate(names):
            e: dict[str, Any] = {m: (None if E[m][(blk, M)] is None else E[m][(blk, M)][0]) for m in methods}
            w: dict[str, Any] = {m: (None if E[m][(blk, M)] is None else E[m][(blk, M)][1]) for m in methods}
            sh: dict[str, Any] = {
                m: (None if e[m] is None or e["ks"] is None else e[m] - e["ks"]) for m in methods if m != "ks"
            }
            per[n] = dict(eps_ev=e, weight=w, shift_ev=sh)
        out["edges"][edge] = dict(defined=bool(defined), min_weight=float(wmin), layers=per)
        # offsets E(layer i+1) - E(layer i); vs_bulk: difference of the shifts minus that of the bulk shifts
        offs: list[dict[str, Any]] = []
        for i in range(nl - 1):
            a, b = names[i], names[i + 1]
            ea, eb = per[a]["eps_ev"], per[b]["eps_ev"]
            o: dict[str, Any] = dict(
                pair=[a, b], offset_ev={m: (None if ea[m] is None or eb[m] is None else eb[m] - ea[m]) for m in methods}
            )
            ba, bb = layers[i]["bulk"], layers[i + 1]["bulk"]
            if ba and bb:
                ref: float = bb[edge] - ba[edge]
                o["bulk_shift_difference_ev"] = ref
                o["vs_bulk_ev"] = {
                    m: (
                        None
                        if per[a]["shift_ev"][m] is None or per[b]["shift_ev"][m] is None
                        else per[b]["shift_ev"][m] - per[a]["shift_ev"][m] - ref
                    )
                    for m in methods
                    if m != "ks"
                }
            offs.append(o)
        out["offsets"][edge] = offs
    return out
