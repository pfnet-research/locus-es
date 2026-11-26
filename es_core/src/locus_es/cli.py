"""Command line of E_S.

    locus-es fields points.npz -o fields.npz          # step 2: lever weight fields on the exporter's points
    locus-es run input.npz -o es.json [--params p.json] [--s 0.7084] [--slab]

Interfaces (a screening g(r) from the bulk g of each material; see interface.py):
    locus-es fields points.npz --layers layers.json -o fields.npz
    locus-es interface window.npz --layers layers.json -o iface.json [--g-one G | --g-one cell.es.json]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from . import interface as IF
from . import io
from ._types import FloatArray
from .fields import lever_fields
from .levels import correct
from .params import DEFAULT


def _deep_update(base: dict[str, Any], upd: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = copy.deepcopy(base)
    k: str
    v: Any
    for k, v in (upd or {}).items():
        out[k] = _deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def params(user: dict[str, Any] | None = None) -> dict[str, Any]:
    return _deep_update(DEFAULT, user)


def fields(
    points_path: str | os.PathLike[str], out_path: str | os.PathLike[str], layers: str | os.PathLike[str] | None = None
) -> None:
    meta: dict[str, Any]
    a: dict[str, npt.NDArray[Any]]
    meta, a = io.load(points_path, "locus-es-points")
    u_v: FloatArray
    u_c: FloatArray
    u_v, u_c = lever_fields(a["n"], a["sigma"], a["tau"])
    out: dict[str, Any] = dict(
        format="locus-es-fields",
        version=io.VERSION,
        points_checksum=meta["checksum"],
        points_fingerprint=meta.get("fingerprint"),
    )
    extra: dict[str, FloatArray] = {}
    lm: dict[str, Any]
    if layers:
        lm, extra = IF.layer_fields(meta, a, u_v, u_c, layers)
        out.update(lm)
    out["checksum"] = io.checksum(u_v, u_c, *extra.values())
    io.save(out_path, out, u_v=u_v, u_c=u_c, **extra)


def interface(
    window_path: str | os.PathLike[str],
    layers_path: str | os.PathLike[str],
    user_params: dict[str, Any] | None = None,
    g_one: float | str | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any]
    arr: dict[str, npt.NDArray[Any]]
    meta, arr = io.load(window_path, "locus-es-window")
    spec: dict[str, Any] = IF.read_layers(layers_path)
    if isinstance(g_one, str):
        try:
            g_one = float(g_one)
        except ValueError:
            g_one = float(json.load(open(cast(str, g_one)))["g"])
    return IF.interface(dict(meta, **arr), spec["layers"], params(user_params), g_one=g_one)


def run(input_path: str | os.PathLike[str], user_params: dict[str, Any] | None = None) -> dict[str, Any]:
    inp: dict[str, Any] = io.load_input(input_path)
    return correct(inp, params(user_params))


def _json(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def main(argv: list[str] | None = None) -> int:
    ap: argparse.ArgumentParser = argparse.ArgumentParser(prog="locus-es")
    sub: argparse._SubParsersAction[argparse.ArgumentParser] = ap.add_subparsers(dest="cmd", required=True)
    f: argparse.ArgumentParser = sub.add_parser("fields", help="lever weight fields on the points of an exporter")
    f.add_argument("points")
    f.add_argument("-o", "--out", required=True)
    f.add_argument(
        "--layers", default=None, help="layers.json: also the fields of a screening g(r) for interfaces (interface.py)"
    )
    i: argparse.ArgumentParser = sub.add_parser(
        "interface", help="layer-resolved band edges with a screening g(r) from the bulk g of each material"
    )
    i.add_argument("window", help="window file of the exporter (python -m locus_es_gpaw window)")
    i.add_argument("--layers", required=True, help="layers.json with the g (or the bulk es.json) of every layer")
    i.add_argument("-o", "--out", default=None)
    i.add_argument("--params", default=None, help="JSON file merged over the defaults")
    i.add_argument(
        "--g-one",
        default=None,
        help="also the first-order levels with one g for the whole cell (a number or an es.json)",
    )
    r: argparse.ArgumentParser = sub.add_parser("run", help="corrected levels from an E_S input file")
    r.add_argument("input")
    r.add_argument("-o", "--out", default=None)
    r.add_argument("--params", default=None, help="JSON file merged over the defaults")
    r.add_argument("--s", type=float, default=None, help="screening scale of the base functional (default LOCUS)")
    r.add_argument(
        "--slab", action="store_true", help="monolayer: effective layer dielectric constant (d = thickness + 3.1 A)"
    )
    r.add_argument(
        "--no-loc",
        action="store_true",
        help="no localized-carrier screening (molecular-dynamics snapshots, amorphous cells)",
    )
    a: argparse.Namespace = ap.parse_args(argv)
    if a.cmd == "fields":
        fields(a.points, a.out, a.layers)
        return 0
    up: dict[str, Any] = json.load(open(a.params)) if a.params else {}
    res: dict[str, Any]
    txt: str
    e: dict[str, Any]
    if a.cmd == "interface":
        res = _json(interface(a.window, a.layers, up, a.g_one))
        txt = json.dumps(res, indent=1)
        if a.out:
            open(a.out, "w").write(txt)
        edge: str
        o: dict[str, Any]
        for edge, e in res["edges"].items():
            for o in res["offsets"][edge]:
                tag: str = " (not defined: edge weight %.2f)" % e["min_weight"] if not e["defined"] else ""
                ms: str = " | ".join(f"{m} {v:+.4f}" for m, v in o["offset_ev"].items() if v is not None)
                vb: str = (
                    (
                        " | vs bulk lineup: "
                        + " ".join(f"{m} {v:+.4f}" for m, v in o["vs_bulk_ev"].items() if v is not None)
                    )
                    if o.get("vs_bulk_ev")
                    else ""
                )
                print(f"{edge} {o['pair'][1]} - {o['pair'][0]}: {ms}{vb}{tag}")
        return 0
    if a.s is not None:
        up["s"] = a.s
    if a.no_loc:
        up.setdefault("loc_screen", {})["on"] = False
    if a.slab:
        up.setdefault("screening", {})["slab"] = {"delta_A": 3.1, "axis": 2}
    res = _json(run(a.input, up))
    txt = json.dumps(res, indent=1)
    if a.out:
        open(a.out, "w").write(txt)
    e = res["edges"]
    print(
        f"gap {res['gap_base_ev']:.4f} -> {res['gap_es_ev']:.4f} eV | g {res['g']:.5f} | VBM {e['vbm']['shift_ev']:+.4f} "
        f"(SOC {e['vbm']['soc_ev']:+.4f}) | CBM {e['cbm']['shift_ev']:+.4f} (SOC {e['cbm']['soc_ev']:+.4f})"
        + (f" | d ({res['d_center']['element']}) {res['d_center']['shift_ev']:+.4f}" if res.get("d_center") else "")
        + (
            f" | window states: gap with the switch per state {res['state_switch']['gap']:.4f} eV (SQS alloys), mobility gap "
            f"{res['state_switch']['gap_mob_es']:.4f} eV (KS {res['state_switch']['gap_mob_ks']:.4f}; amorphous cells)"
            if res.get("state_switch")
            else ""
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
