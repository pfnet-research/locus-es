"""Corrected levels: band edges (+ spin-orbit shifts), d states and, for alloys, every state of a window with the switch per state.

eps'_n = eps_n + s g_n Dhat_n [- sub_n] [+ Delta^SOC_n]
Dhat_n = f_n (-kV <u_V>_n) + (1 - f_n) (kC <u_C>_n)        (f_n the occupation; x k_d for d states)
d states: sub_n = mu_d d_n^ref, |sub_n| <= s g |Dhat_n|      (d^ref: first-order shift of the reference functional, from the input)
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt

from ._types import FloatArray, IntArray
from .localized import localized_edges
from .params import HA
from .screening import screening


def _sig(z: float) -> float:
    return cast(float, 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50))))


def dhat(P: dict[str, Any], lev: FloatArray, f_occ: float, k_scale: float = 1.0) -> float:
    return cast(float, (f_occ * (-P["kV"] * lev[0]) + (1.0 - f_occ) * P["kC"] * lev[1]) * k_scale)


def uniformity(atomw: npt.ArrayLike, cls: IntArray) -> float:
    """f = PR / PR_uniform of the atomic weights of a state over the classes of equivalent atoms (1: Bloch-like, small: localized)"""
    w: FloatArray = np.asarray(atomw, float)
    w = w / max(w.sum(), 1e-30)
    pr: float = 1.0 / max(float(np.sum(w**2)), 1e-30)
    pu: float = 1.0 / sum(w[cls == c].sum() ** 2 / int((cls == c).sum()) for c in set(cls.tolist()))
    return float(pr / pu)


def correct(inp: dict[str, Any], P: dict[str, Any]) -> dict[str, Any]:
    scr: dict[str, Any] = screening(inp, P)
    g: float
    u_fin: float
    g, u_fin = scr["g"], scr["u_fin"]
    s: float = P["s"]
    out: dict[str, Any] = dict(screening=scr, g=g, u_fin=u_fin, edges={})
    loc: dict[str, Any] | None = None
    if len(set(np.asarray(inp["spin_k"]).tolist())) > 1:
        # localized-edge screening for spin-paired systems only (as in the paper)
        loc = {"skipped": "spin-polarized"}
    elif P["loc_screen"].get("on") and u_fin < 0.5 and not scr["metallic"]:
        loc = localized_edges(inp, g, P)
    out["loc_screen"] = loc
    gg: float = (1 - u_fin) * g + u_fin * 1.0
    soc: dict[str, Any] = inp.get("soc") or {}
    i: int
    w: str
    for i, w in enumerate(("vbm", "cbm")):
        e: dict[str, Any] = inp["edges"][w]
        dh: float = dhat(P, inp["lev"][i], e["occupation"])
        g_sol: float = g
        if loc and isinstance(loc.get(w), dict) and loc[w].get("t"):
            g_sol = (1.0 - loc[w]["t"]) * g + loc[w]["t"] * loc[w]["g_phi"]
        g_e: float = (1 - u_fin) * g_sol + u_fin * 1.0
        shift: float = s * g_e * dh
        d_soc: float = float(soc.get(w, 0.0)) * HA
        out["edges"][w] = dict(
            eps_base_ev=e["eps"] * HA,
            dhat_ev=dh,
            g=g_e,
            shift_ev=shift,
            soc_ev=d_soc,
            eps_es_ev=e["eps"] * HA + shift + d_soc,
        )
    out["gap_base_ev"] = (inp["edges"]["cbm"]["eps"] - inp["edges"]["vbm"]["eps"]) * HA
    out["gap_es_ev"] = out["edges"]["cbm"]["eps_es_ev"] - out["edges"]["vbm"]["eps_es_ev"]

    # d states: per k-point and element, the occupied bands of bound-d character >= 0.5 (one group each)
    groups: list[dict[str, Any]] = inp.get("groups") or []
    dg: list[dict[str, Any]] = []
    j: int
    gr: dict[str, Any]
    for j, gr in enumerate(groups):
        if gr["kind"] != "d":
            continue
        lev: FloatArray = inp["lev"][2 + j]
        dh = dhat(P, lev, gr["occupation"], P["k_d"])
        if P["mu_d"] and "d_ref_ev" not in gr:
            raise ValueError("mu_d > 0 needs the reference shifts of the d states: export with --ref-xc")
        sub: float = P["mu_d"] * gr.get("d_ref_ev", 0.0)
        if P["mu_cap"]:
            sub = float(np.sign(sub) * min(abs(sub), s * gg * abs(dh)))
        shift = -sub + s * gg * dh
        dg.append(
            dict(
                gr, dhat_ev=dh, sub_ev=sub, shift_ev=shift, eps_base_ev=gr["eps"] * HA, eps_es_ev=gr["eps"] * HA + shift
            )
        )
    if dg:
        vb0: float
        vb1: float
        vb0, vb1 = out["edges"]["vbm"]["eps_base_ev"], out["edges"]["vbm"]["eps_es_ev"]
        byel: dict[str, dict[str, float]] = {}
        el: str
        for el in sorted({x["element"] for x in dg}):
            xs: list[dict[str, Any]] = [x for x in dg if x["element"] == el]
            ws: FloatArray = np.array([x["weight"] * len(x["bands"]) for x in xs])
            c0: float = float(ws @ [x["eps_base_ev"] for x in xs] / ws.sum())
            c1: float = float(ws @ [x["eps_es_ev"] for x in xs] / ws.sum())
            byel[el] = dict(base_below_vbm_ev=vb0 - c0, es_below_vbm_ev=vb1 - c1, shift_ev=c1 - c0)
        out["d_center_by_element"] = byel
        el0: str = min(byel, key=lambda e_: byel[e_]["base_below_vbm_ev"])
        out["d_center"] = dict(byel[el0], element=el0)
        out["d_groups"] = dg

    # alloys: every state of the window, shifted with the global g, then switched per state by its edge uniformity
    wg: list[tuple[int, dict[str, Any]]] = [(j, gr) for j, gr in enumerate(groups) if gr["kind"] == "window"]
    if wg and P.get("state_switch"):
        ss: dict[str, Any] = P["state_switch"]
        cls: IntArray = np.asarray(inp["atom_class"])
        gphi: dict[bool, float] = {
            True: ((loc or {}).get("vbm") or {}).get("g_phi") or g,
            False: ((loc or {}).get("cbm") or {}).get("g_phi") or g,
        }
        rows: list[dict[str, Any]] = []
        for j, gr in wg:
            dh = dhat(P, inp["lev"][2 + j], gr["occupation"])
            f: float = uniformity(inp["atomw_gi"][j], cls)
            t: float = float(_sig((ss["f_c"] - f) / ss["w"]))
            gn: float = (1 - t) * gg + t * gphi[bool(gr["occupied"])]
            eb: float = gr["eps"] * HA
            rows.append(
                dict(
                    k=gr["k"],
                    bands=gr["bands"],
                    occ=bool(gr["occupied"]),
                    f=f,
                    t=t,
                    ks=eb,
                    es=eb + s * gg * dh,
                    sw=eb + s * gn * dh,
                )
            )
        vb: list[dict[str, Any]] = [q for q in rows if q["occ"]]
        cb: list[dict[str, Any]] = [q for q in rows if not q["occ"]]
        sw: dict[str, Any] = dict(
            window=inp.get("window"), f_c=ss["f_c"], w=ss["w"], g_phi_v=gphi[True], g_phi_c=gphi[False]
        )
        lab: str
        for lab in ("ks", "es", "sw"):
            sw[f"gap_{lab}"] = min(q[lab] for q in cb) - max(q[lab] for q in vb)
            ve: float = max((q[lab] for q in vb if q["f"] >= ss["f_c"]), default=np.nan)
            ce: float = min((q[lab] for q in cb if q["f"] >= ss["f_c"]), default=np.nan)
            sw[f"gap_mob_{lab}"] = ce - ve
        sw["soc_gap"] = out["edges"]["cbm"]["soc_ev"] - out["edges"]["vbm"]["soc_ev"]
        sw["gap"] = sw["gap_sw"] + sw["soc_gap"]
        out["state_switch"] = sw
    return out
