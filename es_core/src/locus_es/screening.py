"""Screening g = 1 / (eps_KS + eps_TF) of E_S.

eps_KS: independent-particle dielectric constant without local fields from the momentum matrix elements of all band pairs,
    eps_KS - 1 = (8 pi / V) sum_k w_k sum_{n<m} f_n (1 - f_m) 2 |p_nm|^2 / omega_nm^3  x R_nm  (atomic units),
with the k.p cell average R at local minima of the direct gap and the extrapolation of the band sum in the number of empty bands.
Finite systems: a smooth switch on the dispersion-weighted estimate eps_disp (1 for an isolated molecule) blends g towards g_fin; a
cell sampled at a single k-point is treated as isolated only with a vacuum gap along every periodic axis.  Monolayers: the effective
layer dielectric constant eps_s = 1 + (eps_cell - 1) L / d.  Metals: the Thomas-Fermi term eps_TF = 4 pi D_F L^2.
"""

from __future__ import annotations

from math import pi
from typing import Any, Callable, cast

import numpy as np

from ._types import BoolArray, FloatArray, IntArray
from .params import BOHR, HA


def _sig(z: float) -> float:
    return cast(float, 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50))))


def metallic(occ_kn: FloatArray, spin_k: IntArray) -> bool:
    frac: FloatArray = np.clip(occ_kn, 0.0, 1.0)
    part: bool = bool(((frac > 0.02) & (frac < 0.98)).any())
    varies: bool = any(len({int((f > 0.5).sum()) for f in frac[spin_k == s]}) != 1 for s in set(spin_k.tolist()))
    return part or varies


def kp_cell_factor(d: FloatArray, occ_w: FloatArray, inp: dict[str, Any]) -> FloatArray:
    """R (k, pairs) of the k.p cell average (two-band Kane dispersion averaged over a sphere of the k-point cell volume)"""
    nk: int = d.shape[0]
    R: FloatArray = np.ones_like(d)
    occ: BoolArray = occ_w > 0.5
    om0: FloatArray = np.array([d[u][occ[u]].min() if occ[u].any() else np.inf for u in range(nk)])
    cell: FloatArray = inp["cell_bohr"]
    B: FloatArray = 2 * pi * np.linalg.inv(cell).T
    vol: float = abs(np.linalg.det(cell))
    bzk: FloatArray
    bz2ibz: IntArray
    ibzk: FloatArray
    bzk, bz2ibz, ibzk = inp["bzk_kc"], inp["bz2ibz_k"], inp["ibzk_kc"]
    nbz: int = len(bzk)
    k_c: float = (3.0 * (2 * pi) ** 3 / vol / nbz / (4 * pi)) ** (1.0 / 3.0)
    ibz_of_u: IntArray
    s_of_u: IntArray
    ibz_of_u, s_of_u = inp["ibz_of_u"], inp["spin_k"]
    u: int
    for u in range(nk):
        if not np.isfinite(om0[u]):
            continue
        k0: FloatArray = ibzk[ibz_of_u[u]]
        dk: FloatArray = (bzk - k0 + 0.5) % 1.0 - 0.5
        dist: FloatArray = np.linalg.norm(dk @ B, axis=1)
        dist[dist < 1e-9] = np.inf
        dmin: float = dist.min()
        if not np.isfinite(dmin):
            continue
        nb: IntArray = np.where(dist < 1.05 * dmin)[0]
        nb_u: list[int] = [int(np.where((ibz_of_u == bz2ibz[j]) & (s_of_u == s_of_u[u]))[0][0]) for j in nb]
        om_nb: FloatArray = om0[nb_u]
        if not np.all(om_nb > om0[u]):
            continue
        b: float = float(np.mean(om_nb**2) - om0[u] ** 2) / dmin**2
        if not np.isfinite(b) or b <= 0:
            continue
        w0: FloatArray = d[u]
        X: FloatArray = k_c * np.sqrt(b) / w0
        avg: FloatArray = 3.0 / k_c**3 * b ** (-1.5) * (np.arcsinh(X) - X / np.sqrt(1 + X**2))
        R[u] = np.where(X > 1e-3, avg * w0**3, 1.0)
    return R


def eps_ks(inp: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    """momentum-form eps_KS (with the k.p cell average and the band extrapolation) and the dispersion-form eps_disp"""
    frac: FloatArray = np.clip(inp["occ_kn"], 0.0, 1.0)
    eps: FloatArray = inp["eps_kn"]
    w: FloatArray = inp["w_k"] / inp["w_k"].sum()
    vol: float = abs(np.linalg.det(inp["cell_bohr"]))
    nb: int = eps.shape[1]
    if (frac[:, -1] > 0.5).any():
        raise ValueError("E_S: the highest band is occupied; the input needs empty bands")
    met: bool = metallic(inp["occ_kn"], inp["spin_k"])
    iu: IntArray
    ju: IntArray
    iu, ju = inp["pair_n"], inp["pair_m"]
    occ_w: FloatArray = frac[:, iu] * (1.0 - frac[:, ju])
    d: FloatArray = np.maximum(eps[:, ju] - eps[:, iu], p["omega_min_ev"] / HA)
    # dispersion form (finite-system switch): f_nm = Var_k(D) V^(2/3) / mean_k(D)
    dbar: FloatArray = (w[:, None] * d).sum(0)
    var: FloatArray = (w[:, None] * (d - dbar) ** 2).sum(0)
    f_disp: FloatArray = var * vol ** (2 / 3) / dbar
    eps_disp: float = 1.0 + float((8 * pi / vol * (w[:, None] * occ_w * f_disp[None] / d**2)).sum())
    # momentum form
    f: FloatArray = 2.0 * inp["p2_kp"] / d
    ck: FloatArray = 8 * pi / vol * (w[:, None] * occ_w * f / d**2)
    if p["kp_cell"] and not met:
        ck = ck * kp_cell_factor(d, occ_w, inp)
    extrap: dict[str, Any] | None = None
    if p["extrap"]:
        nocc_e: int = int((frac > 0.5).sum(1).max())
        n_emp: int = nb - nocc_e
        n_h: int = n_emp // 2
        if n_h >= 4:
            e_n: float = float(ck.sum())
            e_h: float = float(ck[:, ju < nocc_e + n_h].sum())
            tail: float = (e_n - e_h) / ((n_emp / n_h) ** float(p["extrap_gamma"]) - 1.0)
            if e_n > 0 and tail > 0:
                ck = ck * (1.0 + tail / e_n)
            extrap = dict(eps_trunc=1.0 + e_n, eps_half=1.0 + e_h, n_empty=n_emp, n_half=n_h, tail=float(tail))
    nocc: int = int((frac[0] > 0.5).sum())
    return dict(
        eps=float(1.0 + ck.sum()),
        eps_disp=eps_disp,
        metallic=met,
        nocc=nocc,
        n_empty=int(nb - nocc),
        bands_ok=bool(nb - nocc >= nocc),
        extrap=extrap,
    )


def slab_eps(inp: dict[str, Any], eps_cell: float, opt: dict[str, Any]) -> tuple[float, dict[str, float]]:
    ax: int = int(opt.get("axis", 2))
    cell: FloatArray = np.asarray(inp["cell_A"])
    L: float = float(np.linalg.norm(cell[ax]))
    z: FloatArray = np.asarray(inp["positions_A"]) @ (cell[ax] / L) % L
    zs: FloatArray = np.sort(z)
    gaps: FloatArray = np.diff(np.concatenate([zs, [zs[0] + L]]))
    thick: float = L - float(gaps[int(np.argmax(gaps))])
    dd: float = thick + float(opt.get("delta_A", 3.1))
    eps_s: float = 1.0 + (float(eps_cell) - 1.0) * L / dd
    return eps_s, dict(L_A=L, thick_A=thick, d_A=dd, eps_cell=float(eps_cell), eps_s=eps_s)


def single_k_extended(inp: dict[str, Any], fin: dict[str, Any]) -> dict[str, Any] | None:
    """None, or the reason why a single-k-point cell is not an isolated system (a periodic axis without a vacuum gap of vac_min)"""
    if not fin.get("gamma_guard", True) or int(inp["nbzkpts"]) > 1:
        return None
    cell: FloatArray = np.asarray(inp["cell_A"])
    spos: FloatArray = (np.asarray(inp["positions_A"]) @ np.linalg.inv(cell)) % 1.0
    recip: FloatArray = np.linalg.inv(cell).T
    ext: list[int]
    gaps: list[float]
    ext, gaps = [], []
    ax: int
    for ax in range(3):
        s_: FloatArray = np.sort(spos[:, ax])
        gap: float = float(np.diff(np.concatenate([s_, [s_[0] + 1.0]])).max() / np.linalg.norm(recip[ax]))
        gaps.append(gap)
        if inp["pbc"][ax] and gap < float(fin.get("vac_min_A", 5.0)):
            ext.append(ax)
    return dict(extended_axes=ext, max_gap_A=gaps) if ext else None


def fermi_dos(inp: dict[str, Any], sigma_ev: float) -> float:
    """D_F (1 / (Ha bohr^3)) with a fixed Fermi-Dirac broadening; the Fermi level from the electron count"""
    e: FloatArray = inp["eps_kn"]
    wu: FloatArray = inp["w_k_raw"]  # includes the spin degeneracy
    nel: float = float((wu[:, None] * np.clip(inp["occ_kn"], 0, 1)).sum())
    s: float = sigma_ev / HA
    fd: Callable[[FloatArray], FloatArray] = lambda x: 0.5 * (1.0 - np.tanh(0.5 * np.clip(x, -200, 200)))  # noqa: E731
    lo: float
    hi: float
    lo, hi = float(e.min()) - 1.0, float(e.max()) + 1.0
    mu: float
    n: float
    for _ in range(200):
        mu = 0.5 * (lo + hi)
        n = float((wu[:, None] * fd((e - mu) / s)).sum())
        lo, hi = (mu, hi) if n < nel else (lo, mu)
    f: FloatArray = fd((e - mu) / s)
    vol: float = abs(np.linalg.det(inp["cell_bohr"]))
    return float((wu[:, None] * f * (1 - f)).sum()) / s / vol


def screening(inp: dict[str, Any], P: dict[str, Any]) -> dict[str, Any]:
    p: dict[str, Any] = P["screening"]
    jd: dict[str, Any] = eps_ks(inp, p)
    out: dict[str, Any] = dict(
        eps_ks=jd["eps"],
        eps_disp=jd["eps_disp"],
        metallic=jd["metallic"],
        nocc=jd["nocc"],
        n_empty=jd["n_empty"],
        bands_ok=jd["bands_ok"],
        extrap=jd["extrap"],
    )
    eps: float = jd["eps"]
    if p.get("slab"):
        eps, out["slab"] = slab_eps(inp, eps, p["slab"])
    fin: dict[str, Any] = p["finite"]
    u: float = float(_sig((fin["eps1"] - jd["eps_disp"]) / fin["w1"]))
    guard: dict[str, Any] | None = single_k_extended(inp, fin)
    if guard is not None:
        out["single_k_extended"] = dict(guard, u_raw=u)
        u = 0.0
    g: float = float(np.exp((1 - u) * np.log(1.0 / eps) + u * np.log(fin["g_fin"])))
    D_F: float = fermi_dos(inp, p["metal"]["sigma_ev"]) if jd["metallic"] else 0.0
    eps_tf: float = 4 * pi * D_F * float(p["metal"]["L"]) ** 2
    out.update(eps=eps, u_fin=u, g_ins=g, D_F=D_F, eps_TF=eps_tf, g=float(1.0 / (1.0 / g + eps_tf)))
    return out


__all__ = ["screening", "eps_ks", "BOHR"]
