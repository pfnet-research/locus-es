"""Pointwise kernel of LOCUS: r2SCAN + the shell term

    e_sh = c G(n, p, x) e_x^LDA(n),   e_x^LDA = C_X n^(4/3),   c = 0.2,

with analytic derivatives with respect to n, sigma = |grad n|^2, tau and the shell descriptor x = x_c(4) (the density averaged by a
Gaussian of width min(max(4 / k_F, 0.3), 6) bohr, divided by the local density).  The gate G is the product of smooth factors in ln n,
p = |grad n|^2 / (4 (3 pi^2)^(2/3) n^(8/3)) and x (see SHELL); the term is evaluated on the total density.  The r2SCAN
exchange is rebuilt per spin channel with F_x / 1.174 clipped to [1e-6, 1 - 1e-6].  The caller supplies the libxc r2SCAN
base (base_eval / base_eval_spin: per-particle energies and derivatives), e.g. from locus_shell.pyscf_locus.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Union, cast

import numpy as np

from ._types import BoolArray, FloatArray

PI: float = math.pi
CX: float = -0.75 * (3.0 / PI) ** (1.0 / 3.0)
K_P: float = 4.0 * (3.0 * PI**2) ** (2.0 / 3.0)
C_TAU: float = 0.3 * (3.0 * PI**2) ** (2.0 / 3.0)
ETA: float = 1.0e-3
TINY: float = 1.0e-12
# r2SCAN exchange bound h_x^0: F_x / LO is clipped to [1e-6, 1 - 1e-6] (zero exchange gradient beyond)
LO: float = 1.174
FCLIP: float = 1.0e-6

BASES: dict[str, tuple[str, str]] = {"r2SCAN": ("MGGA_X_R2SCAN", "MGGA_C_R2SCAN")}

SHELL: dict[str, float] = dict(
    c=0.2,
    n_t=1.0,
    w_n=0.3,
    p_t=0.15,
    w_p=0.08,
    a_t=-100.0,
    w_a=0.1,
    x_t=0.33,
    w_x=0.02,
    x_hi=1.0,
    xwin_lo=0.62,
    xwin_hi=0.92,
    w_win=0.03,
    nwin_hi=6.0,
    w_nwin=0.3,
    n3_t=2.5,
    w_n3=0.3,
    x3win_hi=0.60,
)
LOWDENS: dict[str, float] = dict(n_lo=1.0e-4, n_hi=1.0e-3)


class LocusModel:
    """LOCUS model: the shell term on descriptor column 0 (c = 4), base r2SCAN"""

    def __init__(self, shell: Optional[dict[str, float]] = None) -> None:
        st: dict[str, float] = dict(SHELL, **(shell or {}))
        self.shell_term: dict[str, Any] = dict(st, x_col=0, spin_scaled="sym", xhi_compact=True)
        self.deorb: None = None
        self.scale_flow: None = None


def _sig(z: FloatArray) -> FloatArray:
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-z))


def shell_gate(
    st: dict[str, Any], n: FloatArray, p: FloatArray, alpha: FloatArray, x: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """G and its partial derivatives (dG/dlnn, dG/dp, dG/dalpha, dG/dx) for the total density n (> 0), p, alpha, x (clipped)"""
    L: FloatArray = np.log(np.maximum(n, 1e-30))
    fn: FloatArray = _sig((L - math.log(st["n_t"])) / st["w_n"])
    fp: FloatArray = _sig((st["p_t"] - p) / st["w_p"])
    fa: FloatArray = _sig((alpha - st["a_t"]) / st["w_a"])
    fx: FloatArray = _sig((x - st["x_t"]) / st["w_x"])
    span: float = st["x_hi"] - st["xwin_hi"]
    t: FloatArray = np.clip((st["x_hi"] - x) / span, 0.0, 1.0)
    cap: FloatArray = t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
    dcap: FloatArray = np.where((t > 0.0) & (t < 1.0), -30.0 * t * t * (1.0 - t) ** 2 / span, 0.0)
    W: FloatArray = _sig((x - st["xwin_lo"]) / st["w_win"]) * _sig((st["xwin_hi"] - x) / st["w_win"])
    w1: FloatArray
    w2: FloatArray
    w1, w2 = _sig((x - st["xwin_lo"]) / st["w_win"]), _sig((st["xwin_hi"] - x) / st["w_win"])
    dW: FloatArray = (w1 * (1 - w1) * w2 - w1 * w2 * (1 - w2)) / st["w_win"]
    cn: FloatArray = _sig((math.log(st["nwin_hi"]) - L) / st["w_nwin"])
    F4: FloatArray = 1.0 - W * (1.0 - cn)
    W3: FloatArray = _sig((st["x3win_hi"] - x) / st["w_win"])
    dW3: FloatArray = -W3 * (1 - W3) / st["w_win"]
    fl: FloatArray = _sig((L - math.log(st["n3_t"])) / st["w_n3"])
    F3: FloatArray = 1.0 - W3 * (1.0 - fl)
    base: FloatArray = fn * fp * fa * fx
    G: FloatArray = base * cap * F4 * F3
    dG_L: FloatArray = (
        G * (1 - fn) / st["w_n"]
        + base * cap * F3 * W * (-cn * (1 - cn) / st["w_nwin"])
        + base * cap * F4 * W3 * fl * (1 - fl) / st["w_n3"]
    )
    dG_p: FloatArray = -G * (1 - fp) / st["w_p"]
    dG_a: FloatArray = G * (1 - fa) / st["w_a"]
    dG_x: FloatArray = (
        G * (1 - fx) / st["w_x"]
        + base * dcap * F4 * F3
        + base * cap * F3 * (-dW * (1.0 - cn))
        + base * cap * F4 * (-dW3 * (1.0 - fl))
    )
    return G, dG_L, dG_p, dG_a, dG_x


def _shell_terms(
    st: dict[str, Any],
    n: FloatArray,
    sigma_tot: FloatArray,
    tau_tot: FloatArray,
    x: FloatArray,
    ds: Optional[FloatArray] = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """e_sh and its derivatives for the total density (live points: n >= 1e-10); ds: the spin-scaling factor of tau_unif"""
    ns: FloatArray = n + TINY
    p: FloatArray = sigma_tot / (K_P * ns ** (8.0 / 3.0))
    tw: FloatArray = sigma_tot / (8.0 * ns)
    den: FloatArray = C_TAU * ns ** (5.0 / 3.0) * (1.0 if ds is None else ds) + ETA * tw
    a_raw: FloatArray = (tau_tot - tw) / den
    alpha: FloatArray = np.maximum(a_raw, 0.0)
    G: FloatArray
    dG_L: FloatArray
    dG_p: FloatArray
    dG_a: FloatArray
    dG_x: FloatArray
    G, dG_L, dG_p, dG_a, dG_x = shell_gate(st, n, p, alpha, x)
    c: float = st["c"]
    n13: FloatArray = np.maximum(n, 1e-30) ** (1.0 / 3.0)
    e: FloatArray = c * G * CX * n13 * n
    pref: FloatArray = c * CX * n13 * n
    # alpha derivatives (the alpha factor is 1 to machine precision for LOCUS)
    on: BoolArray = a_raw > 0.0
    da_dtau: FloatArray = np.where(on, 1.0 / den, 0.0)
    dtw_dn: FloatArray
    dtw_ds: FloatArray
    dtw_dn, dtw_ds = -sigma_tot / (8.0 * ns * ns), 1.0 / (8.0 * ns)
    dden_dn: FloatArray = C_TAU * (5.0 / 3.0) * ns ** (2.0 / 3.0) * (1.0 if ds is None else ds) + ETA * dtw_dn
    da_dn: FloatArray = np.where(on, (-dtw_dn * den - (tau_tot - tw) * dden_dn) / den**2, 0.0)
    da_ds: FloatArray = np.where(on, (-dtw_ds * den - (tau_tot - tw) * ETA * dtw_ds) / den**2, 0.0)
    dp_dn: FloatArray = -(8.0 / 3.0) * p / ns
    dp_ds: FloatArray = 1.0 / (K_P * ns ** (8.0 / 3.0))
    dedn: FloatArray = c * CX * (4.0 / 3.0) * n13 * G + pref * (
        dG_L / np.maximum(n, 1e-30) + dG_p * dp_dn + dG_a * da_dn
    )
    dedsig: FloatArray = pref * (dG_p * dp_ds + dG_a * da_ds)
    dedtau: FloatArray = pref * dG_a * da_dtau
    dedx: FloatArray = pref * dG_x
    return e, dedn, dedsig, dedtau, dedx


def _exchange_paired(
    n: FloatArray, ns: FloatArray, ex: FloatArray, vrx: FloatArray, vsx: FloatArray, vtx: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """exchange energy per volume and derivatives, with the F_x / LO clip (r2SCAN reaches its bound 1.174 only for alpha = s = 0)"""
    m: FloatArray = ns + TINY
    F: FloatArray = ex / (CX * m ** (1.0 / 3.0))
    fb: FloatArray = F / LO
    clip: BoolArray = (fb < FCLIP) | (fb > 1.0 - FCLIP)
    e: FloatArray = n * ex
    dn: FloatArray
    ds: FloatArray
    dt: FloatArray
    dn, ds, dt = vrx.copy(), vsx.copy(), vtx.copy()
    if clip.any():
        c: FloatArray = np.where(fb[clip] < FCLIP, FCLIP, 1.0 - FCLIP)
        mc: FloatArray = m[clip]
        e[clip] = n[clip] * CX * mc ** (1.0 / 3.0) * LO * c
        dn[clip] = CX * LO * c * (mc ** (1.0 / 3.0) + ns[clip] / 3.0 * mc ** (-2.0 / 3.0))
        ds[clip] = 0.0
        dt[clip] = 0.0
    return e, dn, ds, dt


def constrained_terms(
    model: LocusModel,
    base: str,
    n: FloatArray,
    sigma: FloatArray,
    tau: FloatArray,
    x: FloatArray,
    n_min: float = 1.0e-10,
    x_max: float = 1.0e3,
    base_eval: Optional[Callable[[FloatArray, FloatArray, FloatArray], tuple[FloatArray, ...]]] = None,
    **kw: Any,
) -> dict[str, FloatArray]:
    """spin-paired LOCUS: n, sigma, tau (P,), x (K, P) -> dict(e, dedn, dedsigma, dedtau, dedx); points with n < n_min contribute 0"""
    st: dict[str, Any] = model.shell_term
    n = np.asarray(n, np.float64)
    ns: FloatArray = np.maximum(n, n_min)
    live: BoolArray = n > n_min
    sigma = np.maximum(np.asarray(sigma, np.float64), 0.0)
    tau = np.maximum(np.asarray(tau, np.float64), 0.0)
    x_raw: FloatArray = np.asarray(x, np.float64)
    xclip: BoolArray = (x_raw < 0.0) | (x_raw > x_max)
    x = np.where(live[None, :], np.clip(x_raw, 0.0, x_max), 1.0)
    ex: FloatArray
    vrx: FloatArray
    vsx: FloatArray
    vtx: FloatArray
    ec: FloatArray
    vrc: FloatArray
    vsc: FloatArray
    vtc: FloatArray
    ex, vrx, vsx, vtx, ec, vrc, vsc, vtc = cast(
        "Callable[[FloatArray, FloatArray, FloatArray], tuple[FloatArray, ...]]", base_eval
    )(ns, sigma, tau)
    col: int = int(st["x_col"])
    es: FloatArray
    dn_s: FloatArray
    ds_s: FloatArray
    dt_s: FloatArray
    dx_s: FloatArray
    es, dn_s, ds_s, dt_s, dx_s = _shell_terms(st, ns, sigma, tau, x[col])
    exv: FloatArray
    dnx: FloatArray
    dsx: FloatArray
    dtx: FloatArray
    exv, dnx, dsx, dtx = _exchange_paired(ns, ns, ex, vrx, vsx, vtx)
    e: FloatArray = (exv / ns + ec) * n + es * (n / ns)
    dedn: FloatArray = dnx + vrc + dn_s
    dedsigma: FloatArray = dsx + vsc + ds_s
    dedtau: FloatArray = dtx + vtc + dt_s
    dedx: FloatArray = np.zeros_like(x)
    dedx[col] = dx_s
    dead: BoolArray = ~live
    a: FloatArray
    for a in (e, dedn, dedsigma, dedtau):
        a[dead] = 0.0
    dedx[:, dead] = 0.0
    dedx[xclip] = 0.0
    return dict(e=e, dedn=dedn, dedsigma=dedsigma, dedtau=dedtau, dedx=dedx)


def constrained_terms_spin(
    model: LocusModel,
    base: Union[str, list[FloatArray]],
    n_s: FloatArray,
    sig_s: FloatArray,
    tau_s: FloatArray,
    x_s: FloatArray,
    x_tot: FloatArray,
    n_min: float = 1.0e-10,
    x_max: float = 1.0e3,
    base_eval: Optional[Callable[[FloatArray, FloatArray, FloatArray], tuple[FloatArray, ...]]] = None,
    **kw: Any,
) -> dict[str, FloatArray]:
    """spin-polarized LOCUS: r2SCAN exchange on the channel densities 2 n_s (with the F_x / LO clip), polarized r2SCAN correlation,
    the shell term on the total density -> dict(e, dedn (2, P), dedsigma (3, P), dedtau (2, P), dedx_tot (K, P), dedx_s (K, 2, P)).
    Regularisation: sigma_ab within the Cauchy-Schwarz bound, tau_s >= tau_W of its channel (the floored tau depends on n_s and sigma_ss:
    dE/dtau is routed there), an empty channel (n_s <= n_min) has no derivatives, non-finite base values are 0."""
    st: dict[str, Any] = model.shell_term
    n_s = np.asarray(n_s, np.float64)
    P: int = n_s.shape[1]
    n: FloatArray = n_s.sum(0)
    live: BoolArray = n > n_min
    ns_s: FloatArray = np.maximum(n_s, 0.5 * n_min)
    sig_s = np.asarray(sig_s, np.float64).copy()
    sig_s[0] = np.maximum(sig_s[0], 0.0)
    sig_s[2] = np.maximum(sig_s[2], 0.0)
    sab: FloatArray = np.sqrt(sig_s[0] * sig_s[2])
    viol: BoolArray = np.abs(sig_s[1]) > sab * (1.0 + 1.0e-6) + 1.0e-30
    sig_s[1] = np.where(viol, np.clip(sig_s[1], -sab, sab), sig_s[1])
    tau_s = np.maximum(np.asarray(tau_s, np.float64), 0.0)
    floored: BoolArray = np.zeros((2, P), bool)
    s: int
    for s in range(2):
        tw: FloatArray = sig_s[2 * s] / (8.0 * ns_s[s])
        floored[s] = tau_s[s] < tw
        tau_s[s] = np.where(floored[s], tw, tau_s[s])
    x_raw: FloatArray = np.asarray(x_tot, np.float64)
    xclip: BoolArray = (x_raw < 0.0) | (x_raw > x_max)
    x: FloatArray = np.where(live[None, :], np.clip(x_raw, 0.0, x_max), 1.0)
    base = list(
        cast("Callable[[FloatArray, FloatArray, FloatArray], tuple[FloatArray, ...]]", base_eval)(ns_s, sig_s, tau_s)
    )
    if sum(int((~np.isfinite(np.asarray(a))).sum()) for a in base):
        base = [np.nan_to_num(np.asarray(a, float), nan=0.0, posinf=0.0, neginf=0.0) for a in base]
    ex2: FloatArray
    vrx: FloatArray
    vsx: FloatArray
    vtx: FloatArray
    ec: FloatArray
    vrc: FloatArray
    vsc: FloatArray
    vtc: FloatArray
    ex2, vrx, vsx, vtx, ec, vrc, vsc, vtc = base
    # gradients of Phi = eps (n0 + n1) with respect to the leaves (ns_s, sigma, tau, ex2_s, ec), eps per particle of the leaf density
    nl: FloatArray = ns_s.sum(0)
    nT: FloatArray = nl + TINY
    eps: FloatArray = np.array(ec, float, copy=True)
    g_n: FloatArray = np.zeros((2, P))
    g_sig: FloatArray = np.zeros((3, P))
    g_tau: FloatArray = np.zeros((2, P))
    g_ex2: FloatArray = np.zeros((2, P))
    rest: FloatArray = np.zeros(P)
    for s in range(2):
        m: FloatArray = 2.0 * ns_s[s] + TINY
        R: FloatArray = 0.5 * CX * m ** (4.0 / 3.0) / nT
        F: FloatArray = ex2[s] / (CX * m ** (1.0 / 3.0))
        fb: FloatArray = F / LO
        clip: BoolArray = (fb < FCLIP) | (fb > 1.0 - FCLIP)
        fh: FloatArray = np.where(clip, np.where(fb < FCLIP, FCLIP, 1.0 - FCLIP), fb)
        term: FloatArray = R * LO * fh  # eps_x,s
        eps = eps + term
        # d(term)/d ns_s through R (m) and, unclipped, through F (m); d/d ex2_s unclipped
        dR_dm: FloatArray = 0.5 * CX * (4.0 / 3.0) * m ** (1.0 / 3.0) / nT
        dF_dm: FloatArray = -F / (3.0 * m)
        dterm_dm: FloatArray = dR_dm * LO * fh + np.where(clip, 0.0, R * dF_dm)
        g_n[s] += nl * dterm_dm * 2.0
        g_ex2[s] = nl * np.where(clip, 0.0, R / (CX * m ** (1.0 / 3.0)))
        rest = rest - nl * R * LO * fh / nT  # d(1/nT)/d n_k for both channels (the R factor), times nl
    g_n += (rest + (eps - ec))[None, :]  # d(nl eps_x)/d n_k: eps_x itself and the 1/nT factor of R
    # shell term on the total leaf density
    sig_tot: FloatArray = sig_s[0] + 2.0 * sig_s[1] + sig_s[2]
    zeta: FloatArray = np.clip((ns_s[0] - ns_s[1]) / nT, -1.0, 1.0)
    dsz: FloatArray = 0.5 * ((1.0 + zeta) ** (5.0 / 3.0) + (1.0 - zeta) ** (5.0 / 3.0))
    col: int = int(st["x_col"])
    es: FloatArray
    dn_sh: FloatArray
    ds_sh: FloatArray
    dt_sh: FloatArray
    dx_sh: FloatArray
    es, dn_sh, ds_sh, dt_sh, dx_sh = _shell_terms(st, nl, sig_tot, tau_s.sum(0), x[col], ds=dsz)
    if np.any(dt_sh != 0.0):
        raise RuntimeError("LOCUS spin kernel: the alpha factor of the shell gate is not saturated")
    eps = eps + es / nl
    g_ec: FloatArray = nl
    for s in range(2):
        g_n[s] += ec + dn_sh  # d(nl ec)/d n_s at fixed ec, and the shell term
    g_sig += np.stack([ds_sh, 2.0 * ds_sh, ds_sh])
    nn: FloatArray = np.maximum(n, n_min)
    dedn: FloatArray = np.empty((2, P))
    dedtau: FloatArray = np.empty((2, P))
    dedsigma: FloatArray = np.empty((3, P))
    for s in range(2):
        dedn[s] = g_n[s] + g_ex2[s] * (vrx[s] - ex2[s]) / ns_s[s] + g_ec * (vrc[s] - ec) / nn
        dedtau[s] = g_tau[s] + g_ex2[s] * vtx[s] / ns_s[s] + g_ec * vtc[s] / nn
        dedsigma[2 * s] = g_sig[2 * s] + g_ex2[s] * 2.0 * vsx[s] / ns_s[s] + g_ec * vsc[2 * s] / nn
    dedsigma[1] = g_sig[1] + g_ec * vsc[1] / nn
    for s in range(2):
        empty: BoolArray = n_s[s] <= n_min
        f: BoolArray = floored[s] & ~empty
        if f.any():
            dt: FloatArray = dedtau[s, f]
            dedsigma[2 * s, f] += dt / (8.0 * ns_s[s, f])
            dedn[s, f] -= dt * sig_s[2 * s, f] / (8.0 * ns_s[s, f] ** 2)
        dedtau[s, floored[s]] = 0.0
        dedn[s, empty] = 0.0
        dedsigma[2 * s, empty] = 0.0
        dedtau[s, empty] = 0.0
    e: FloatArray = eps * n
    dedx_tot: FloatArray = np.zeros_like(x)
    dedx_tot[col] = dx_sh
    dead: BoolArray = ~live
    e[dead] = 0.0
    a: FloatArray
    for a in (dedn, dedsigma, dedtau):
        a[:, dead] = 0.0
    dedx_tot[:, dead] = 0.0
    dedx_tot[xclip] = 0.0
    return dict(e=e, dedn=dedn, dedsigma=dedsigma, dedtau=dedtau, dedx_tot=dedx_tot, dedx_s=np.zeros(np.shape(x_s)))
