"""Command line:  python -m locus_es_qe points SAVE -o points.npz
python -m locus_es_qe export SAVE --fields fields.npz -o es_input.npz [--soc VBM CBM]"""

from __future__ import annotations

import argparse
import sys
import time

from .export import points_from_save, write_input, write_points
from .points import QePoints


def main(argv: list[str] | None = None) -> int:
    ap: argparse.ArgumentParser = argparse.ArgumentParser(prog="python -m locus_es_qe")
    sub: argparse._SubParsersAction[argparse.ArgumentParser] = ap.add_subparsers(dest="cmd", required=True)
    p: argparse.ArgumentParser = sub.add_parser("points", help="integration points and ground-state fields")
    p.add_argument("save", help="<outdir>/<prefix>.save of the nscf run (nosym, empty bands)")
    p.add_argument("-o", "--out", required=True)
    e: argparse.ArgumentParser = sub.add_parser("export", help="the E_S input file")
    e.add_argument("save")
    e.add_argument("--fields", required=True, help="fields.npz from 'locus-es fields'")
    e.add_argument("-o", "--out", required=True)
    e.add_argument(
        "--soc", nargs=2, type=float, default=None, metavar=("VBM", "CBM"), help="spin-orbit edge shifts in eV"
    )
    a: argparse.Namespace = ap.parse_args(argv)
    t0: float = time.time()
    ps: QePoints = points_from_save(a.save)
    if ps.norm_error > 1e-4:
        print(f"warning: the projections reproduce <psi|S|psi> = 1 only to {ps.norm_error:.1e}", file=sys.stderr)
    if a.cmd == "points":
        write_points(ps, a.out)
        print(
            f"{ps.w.size} points ({ps.ngrid} grid), {ps.electrons():.5f} valence electrons, {time.time() - t0:.0f} s -> {a.out}"
        )
        return 0
    meta: dict[str, object] = write_input(ps, a.fields, a.out, None if a.soc is None else (a.soc[0], a.soc[1]))
    print(f"E_S input -> {a.out} ({time.time() - t0:.0f} s); shells missing: {meta['missing_shells'] or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
