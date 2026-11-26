"""Localized-carrier screening of s / p band edges.

A band-edge carrier localized on one atom (narrow band relative to its on-site interaction) is screened at the length scale of its own
atomic shell: g_phi = <eps^-1(q)> over the form factor |rho_shell(q)|^2 of the dominant atomic shell (model dielectric function of
Cappellini et al. with eps_inf = 1/g and the free-atom valence sp density).  Switch:
    t = max(sig((U / W_block - c) / w) [VBM only], sig((U / W_complex - c) / w) sig((m* - m_c) / w_m)),
    U = s (kV <u_V> + kC <u_C>)_atom g_phi,  W = sqrt(12 var) of the channel-weighted levels of the edge band block / complex,
    m* the harmonic-mean edge mass from finite differences on the k grid.   g_edge = (1 - t) g + t g_phi.
"""

from __future__ import annotations

from math import pi
from typing import Any, Callable, cast

import numpy as np

from ._types import FloatArray, IntArray
from .fields import local_variables
from .params import HA, HB2M

_trap: Callable[..., Any] = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # trapz: NumPy < 2.0


def _sig(z: float) -> float:
    return cast(float, 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50))))


def atom_shell_features(sh: dict[str, FloatArray]) -> dict[str, float]:
    """state-weighted lever features of the free-atom shell on its own density (radial grid, weights 4 pi r^2 dr)"""
    v: dict[str, FloatArray] = local_variables(sh["n_g"], sh["sigma_g"], sh["tau_g"])
    gs: FloatArray = cast(FloatArray, 1 + 3 * v["s2"] / (1 + v["s2"] / 4))
    w: FloatArray = sh["w_g"] * sh["rho_g"]
    return dict(kf_e1_s=float(w @ (v["kf"] * np.exp(-v["alpha"]) * gs)), kf_s=float(w @ (v["kf"] * gs)))


def g_phi(sh: dict[str, FloatArray], eps_inf: float, nv_b: float) -> float:
    rq: FloatArray = np.logspace(-4, np.log10(40.0), 400)
    rho: FloatArray = np.interp(rq, sh["r_g"], sh["rho_g"])
    q: FloatArray = np.linspace(1e-4, 60.0, 6000)
    ff: FloatArray = 4 * pi * _trap(np.sinc(np.outer(q, rq) / pi) * (rho * rq**3)[None, :], np.log(rq), axis=1)
    kf: float = (3 * pi**2 * nv_b) ** (1 / 3)
    qtf: float = np.sqrt(4 * kf / pi)
    a: float = 0.0 if not np.isfinite(eps_inf) else 1.0 / max(eps_inf - 1.0, 1e-6)
    eps: FloatArray = 1.0 + 1.0 / (a + 1.563 * (q / qtf) ** 2 + q**4 / (4 * kf**2 * qtf**2))
    w: FloatArray = ff**2
    return float(_trap(w / eps, q) / _trap(w, q))


def _width(eps: FloatArray, kw: FloatArray, sel: list[tuple[int, int]], chanw: FloatArray) -> float | None:
    e_: FloatArray = np.array([eps[u, n] for u, n in sel])
    w_: FloatArray = np.array([kw[u] * chanw[u, n] for u, n in sel])
    if w_.sum() <= 0:
        return None
    m: float = float(w_ @ e_ / w_.sum())
    return float(np.sqrt(12 * (w_ @ (e_ - m) ** 2) / w_.sum()))


def _block(eps: FloatArray, nocc: int, side: str, gap_ev: float) -> list[tuple[int, int]]:
    nk: int
    nb: int
    nk, nb = eps.shape
    lv: list[tuple[float, int, int]] = (
        sorted(((eps[u, n], u, n) for u in range(nk) for n in range(nocc)), reverse=True)
        if side == "v"
        else sorted((eps[u, n], u, n) for u in range(nk) for n in range(nocc, nb))
    )
    out: list[tuple[int, int]] = [lv[0][1:]]
    e0: float
    e1: float
    u: int
    n: int
    for (e0, *_), (e1, u, n) in zip(lv, lv[1:]):
        if abs(e1 - e0) > gap_ev:
            break
        out.append((u, n))
    return out


def _complex(eps: FloatArray, nocc: int, side: str, tol: float) -> list[tuple[int, int]]:
    nk: int
    nb: int
    nk, nb = eps.shape
    start: int
    step: int
    start, step = (nocc - 1, -1) if side == "v" else (nocc, 1)
    idx: list[int] = [start]
    lo: float
    hi: float
    lo, hi = eps[:, start].min(), eps[:, start].max()
    n: int = start + step
    a: float
    b: float
    while 0 <= n < nb:
        a, b = eps[:, n].min(), eps[:, n].max()
        if (step < 0 and b >= lo - tol) or (step > 0 and a <= hi + tol):
            idx.append(n)
            lo = min(lo, a)
            hi = max(hi, b)
            n += step
        else:
            break
    return [(u, m) for u in range(nk) for m in idx]


def _mass_fd(
    eps: FloatArray, bzk: FloatArray, bz2ibz: IntArray, cell_A: FloatArray, band: int, side: str
) -> float | None:
    bzk = np.asarray(bzk, float)
    e: FloatArray = np.asarray(eps, float)[np.asarray(bz2ibz), band]
    N: list[int] = [len(np.unique(np.round(bzk[:, i] % 1.0, 6))) for i in range(3)]
    i0: int = int(np.argmax(e) if side == "v" else np.argmin(e))
    B: FloatArray = 2 * pi * np.linalg.inv(np.asarray(cell_A, float)).T
    key: dict[tuple[float, ...], int] = {tuple(np.round(k % 1.0, 5)): j for j, k in enumerate(bzk)}
    ms: list[float] = []
    i: int
    for i in range(3):
        if N[i] < 2:
            continue
        dv: FloatArray = np.zeros(3)
        dv[i] = 1.0 / N[i]
        jp: int | None = key.get(tuple(np.round((bzk[i0] + dv) % 1.0, 5)))
        jm: int | None = key.get(tuple(np.round((bzk[i0] - dv) % 1.0, 5)))
        if jp is None or jm is None:
            continue
        dk: np.floating[Any] = np.linalg.norm(B[i]) / N[i]
        curv: float = 2.0 * (e[jp] - e[i0]) / dk**2 if N[i] == 2 else (e[jp] + e[jm] - 2 * e[i0]) / dk**2
        curv = -curv if side == "v" else curv
        ms.append(HB2M / curv if curv > 1e-9 else np.inf)
    return (len(ms) / sum(1 / m for m in ms)) if ms and all(m > 0 for m in ms) else None


def localized_edges(inp: dict[str, Any], g: float, P: dict[str, Any]) -> dict[str, Any]:
    """per band edge: dominant channel, U, W, m*, g_phi and the switch t"""
    lp: dict[str, Any] = P["loc_screen"]
    s: float
    kV: float
    kC: float
    s, kV, kC = P["s"], P["kV"], P["kC"]
    nocc: int = int(round(inp["n_electrons"] / 2))
    eps: FloatArray = inp["eps_kn"] * HA
    kw: FloatArray = inp["w_k"] / inp["w_k"].sum()
    out: dict[str, Any] = dict(nv_bohr3=inp["nv_bohr3"], eps_inf=1.0 / g)
    w: str
    side: str
    el: str
    ell: int
    for w, side in (("vbm", "v"), ("cbm", "c")):
        e: dict[str, Any] = inp["edges"][w]
        cw: FloatArray = inp["chanw_knc"][e["u"], e["bands"]].sum(0)
        tot: float = float(cw.sum())
        jc: int = int(np.argmax(cw))
        el, ell = inp["channels"][jc]
        r: dict[str, Any] = dict(channel=f"{el} {'spdf'[ell]}", frac=float(cw[jc] / tot) if tot else None)
        if "spdf"[ell] not in lp["ls"]:
            out[w] = dict(r, t=0.0)
            continue
        sh: dict[str, FloatArray] = inp["atom_shells"][f"{el}_{ell}"]
        feat: dict[str, float] = atom_shell_features(sh)
        gp: float = g_phi(sh, 1.0 / g, inp["nv_bohr3"])
        U: float = s * (kV * feat["kf_e1_s"] + kC * feat["kf_s"]) * gp
        chan: FloatArray = inp["chanw_knc"][:, :, jc]
        Wb: float | None = _width(eps, kw, _block(eps, nocc, side, lp["gap_ev"]), chan)
        Wc: float | None = _width(eps, kw, _complex(eps, nocc, side, float(lp["complex_tol_ev"])), chan)
        m: float | None = _mass_fd(
            eps, inp["bzk_kc"], inp["bz2ibz_k"], inp["cell_A"], nocc - 1 if side == "v" else nocc, side
        )
        tA: float = float(_sig((U / Wb - lp["c"]) / lp["w"])) if (Wb and side == "v") else 0.0
        tC: float = float(_sig((U / Wc - lp["c"]) / lp["w"])) if Wc else 0.0
        tB: float = (
            float(_sig((m - lp["m_c"]) / lp["w_m"]))
            if m is not None and np.isfinite(m)
            else (1.0 if m is not None else 0.0)
        )
        r.update(
            U_ev=U,
            W_block_ev=Wb,
            W_complex_ev=Wc,
            mass=m,
            g_phi=gp,
            t_block=tA,
            t_complex=tC,
            t_mass=tB,
            t=max(tA, tC * tB),
        )
        out[w] = r
    return out
