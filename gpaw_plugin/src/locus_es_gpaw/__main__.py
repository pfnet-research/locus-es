"""python -m locus_es_gpaw points gs.gpw -o points.npz
python -m locus_es_gpaw export gs.gpw --fields fields.npz -o es_input.npz [--soc-window 8 | --soc-window nsites | --no-soc]
                                     [--window 40 16] [--ref-xc JSON] [--no-d]
python -m locus_es_gpaw window gs.gpw --fields fields.npz -o window.npz [--occ 24] [--emp 16]   (interfaces: 'locus-es fields --layers')
In parallel:  mpiexec -n N python -m locus_es_gpaw ...  (k-points, then real-space domains; 'window': k-points only).
Over k-points the output does not depend on N; with domains it agrees within the convergence of the empty bands."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

import locus_gpaw  # noqa: F401  (registers LOCUS with GPAW)

from . import bands as B
from . import export as X
from . import window as W

# r2SCAN + rVV10: LOCUS without its shell term (for --ref-xc default)
REF_XC: dict[str, Any] = {"name": "LOCUS", "shell": False}


def main(argv: Sequence[str] | None = None) -> int:
    ap: argparse.ArgumentParser = argparse.ArgumentParser(prog="locus_es_gpaw")
    sub: argparse._SubParsersAction[argparse.ArgumentParser] = ap.add_subparsers(dest="cmd", required=True)
    p: argparse.ArgumentParser = sub.add_parser(
        "points", help="integration points and the density, |grad n|^2, tau on them"
    )
    p.add_argument("gpw")
    p.add_argument("-o", "--out", required=True)
    e: argparse.ArgumentParser = sub.add_parser("export", help="the E_S input (adds the empty bands first)")
    e.add_argument("gpw")
    e.add_argument("--fields", required=True, help="lever weight fields from 'locus-es fields'")
    e.add_argument("-o", "--out", required=True)
    e.add_argument(
        "--soc-window",
        default="8",
        help="bands on each side of the gap in the spin-orbit basis: a number, 'nsites' "
        "(max(8, number of atoms), for supercells) or 'all'",
    )
    e.add_argument("--no-soc", action="store_true", help="no spin-orbit coupling (e.g. setups with f projectors)")
    e.add_argument(
        "--window",
        type=int,
        nargs=2,
        default=None,
        metavar=("BELOW", "ABOVE"),
        help="every state of [nocc - BELOW, nocc + ABOVE) at every k-point (SQS alloys: 40 16)",
    )
    e.add_argument("--no-d", action="store_true", help="no d-state groups")
    e.add_argument(
        "--ref-xc",
        default=None,
        nargs="?",
        const="default",
        help="also write the first-order d-state shifts of a reference functional (GPAW xc name or JSON dict; without a value "
        "r2SCAN + rVV10, i.e. LOCUS without its shell term).  Only needed for E_S runs with mu_d > 0; the paper uses mu_d = 0",
    )
    e.add_argument("--band-factor", type=float, default=2.0)
    e.add_argument(
        "--check",
        type=float,
        default=1e-2,
        help="largest occupied-band residual (eV^2 per valence electron) accepted under the "
        "rebuilt Hamiltonian before the empty bands are added (guards against inconsistent wave functions)",
    )
    w: argparse.ArgumentParser | int | None  # the parser of 'window', later the spin-orbit window
    w = sub.add_parser(
        "window", help="matrices of the layer fields between the states of a window (E_S with a screening g(r))"
    )
    w.add_argument("gpw")
    w.add_argument("--fields", required=True, help="fields from 'locus-es fields --layers layers.json'")
    w.add_argument("-o", "--out", required=True)
    w.add_argument("--occ", type=int, default=24, help="highest occupied bands per k-point")
    w.add_argument("--emp", type=int, default=16, help="lowest empty bands per k-point")
    w.add_argument(
        "--tol", type=float, default=1e-4, help="eigenstate error of the empty bands (eV^2 per valence electron)"
    )
    a: argparse.Namespace = ap.parse_args(argv)
    from gpaw.mpi import world

    say: Callable[..., None] = print if world.rank == 0 else (lambda *x, **k: None)
    t0: float = time.time()
    calc: Any = B.open_gpw(a.gpw, parallel={"domain": 1, "band": 1}) if a.cmd == "window" else B.open_gpw(a.gpw)
    if a.cmd == "points":
        ps: X.PointSet = X.write_points(calc, a.out)
        ngrid: int = int(np.prod(calc.hamiltonian.finegd.get_size_of_global_array()))
        say(f"{ps.nglobal} points ({ngrid} grid) in {time.time() - t0:.0f} s")
        return 0
    if a.cmd == "window":
        W.write_window(calc, a.fields, a.out, n_occ=a.occ, n_emp=a.emp, tol=a.tol, say=say)
        say(
            f"window {a.occ} + {a.emp} bands in {time.time() - t0:.0f} s -> {a.out}"
            + (f" ({world.size} ranks)" if world.size > 1 else "")
        )
        return 0
    nb: int = B.extend_empty_bands(calc, factor=a.band_factor, check=a.check)
    t1: float = time.time()
    soc: dict[str, Any] | None = None
    if not a.no_soc:
        w = (
            None
            if a.soc_window == "all"
            else (max(8, len(calc.atoms)) if a.soc_window == "nsites" else int(a.soc_window))
        )
        soc = B.soc_edge_shifts(calc, window=w)
        soc["window"] = w
    ref: Callable[[Any, list[dict[str, Any]]], list[float]] | None = None
    if not a.no_d and a.ref_xc is not None:
        rx: str | dict[str, Any] = (
            REF_XC
            if a.ref_xc == "default"
            else (json.loads(a.ref_xc) if a.ref_xc.strip().startswith("{") else a.ref_xc)
        )
        ref = lambda calc_, groups: B.reference_shifts(calc_, groups, rx)  # noqa: E731
    X.write_input(
        calc,
        a.fields,
        a.out,
        soc=soc,
        d_ref=ref,
        window=a.window,
        d_states=not a.no_d,
        base_functional=str(calc.parameters.xc),
    )
    say(
        f"bands {nb} in {t1 - t0:.0f} s"
        + (
            f" ({B.EXTENSION_STATS['blocks']} blocks, {B.EXTENSION_STATS['iterations']} sweeps)"
            if B.EXTENSION_STATS
            else ""
        )
        + f", export {time.time() - t1:.0f} s -> {a.out}"
        + (f" ({world.size} ranks)" if world.size > 1 else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
