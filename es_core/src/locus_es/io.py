"""Files exchanged with a DFT-code exporter (docs/ES_INPUT_FORMAT.md): NumPy .npz archives with a JSON 'meta' entry.

points (exporter, step 1):  w, n, sigma, tau on the integration points          meta.format = "locus-es-points"
fields (E_S, step 2):       u_v, u_c on the same points                          meta.format = "locus-es-fields"
input  (exporter, step 3):  eigenvalues, transitions, state integrals, ...       meta.format = "locus-es-input"
window (exporter, interfaces): matrices of the layer fields near the gap        meta.format = "locus-es-window"
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import numpy as np
import numpy.typing as npt

VERSION: int = 1


def checksum(*arrays: npt.ArrayLike) -> str:
    h: hashlib._Hash = hashlib.sha256()
    a: npt.ArrayLike
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def save(path: str | os.PathLike[str], meta: dict[str, Any], **arrays: npt.ArrayLike) -> None:
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)


def load(path: str | os.PathLike[str], fmt: str) -> tuple[dict[str, Any], dict[str, npt.NDArray[Any]]]:
    z: np.lib.npyio.NpzFile = np.load(path, allow_pickle=False)
    meta: dict[str, Any] = json.loads(str(z["meta"]))
    if meta.get("format") != fmt:
        raise ValueError(f"{path}: format {meta.get('format')!r}, expected {fmt!r}")
    if int(meta.get("version", 0)) != VERSION:
        raise ValueError(f"{path}: version {meta.get('version')}, this code reads version {VERSION}")
    return meta, {k: z[k] for k in z.files if k != "meta"}


def load_input(path: str | os.PathLike[str]) -> dict[str, Any]:
    """the input of E_S as one dict (meta entries + arrays; atom shells regrouped)"""
    meta: dict[str, Any]
    arr: dict[str, npt.NDArray[Any]]
    meta, arr = load(path, "locus-es-input")
    inp: dict[str, Any] = dict(meta)
    shells: dict[str, dict[str, npt.NDArray[Any]]] = {}
    k: str
    v: npt.NDArray[Any]
    ch: str
    name: str
    for k, v in arr.items():
        if k.startswith("shell:"):
            _, ch, name = k.split(":")
            shells.setdefault(ch, {})[name] = v
        else:
            inp[k] = v
    inp["atom_shells"] = shells
    for k in ("cell_bohr", "cell_A", "positions_A"):
        inp[k] = np.asarray(inp[k], float)
    inp["pbc"] = [bool(x) for x in inp["pbc"]]
    inp["channels"] = [tuple(c) for c in inp["channels"]]
    return inp
