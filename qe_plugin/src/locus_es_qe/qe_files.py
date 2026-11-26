"""Reading a pw.x run: data-file-schema.xml and the wave functions wfc<k>.hdf5 of <outdir>/<prefix>.save.

Units on return: bohr, bohr^-1 and hartree.  k-point weights include the spin degeneracy (they sum to 2 for spin-paired runs).
"""

from __future__ import annotations

import dataclasses
import xml.etree.ElementTree as ET
from math import pi
from pathlib import Path
from typing import Any

import numpy as np

from ._types import ComplexArray, FloatArray, IntArray


@dataclasses.dataclass
class QeRun:
    save: Path
    cell: FloatArray  # (3, 3) lattice vectors as rows, bohr
    species: list[str]  # species name of every atom
    positions: FloatArray  # (nat, 3) bohr
    pseudo_files: dict[str, str]  # species name -> UPF file name in the save directory
    nelec: float
    nbnd: int
    kpts: FloatArray  # (nk, 3) Cartesian, bohr^-1
    weights: FloatArray  # (nk,)
    eps: FloatArray  # (nk, nbnd) hartree
    mp_grid: tuple[int, int, int]
    mp_shift: tuple[int, int, int]
    fft: tuple[int, int, int]
    functional: str
    flags: dict[str, bool]

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.cell)))

    @property
    def recip(self) -> FloatArray:
        """reciprocal lattice vectors as rows, bohr^-1"""
        out: FloatArray = 2.0 * pi * np.linalg.inv(self.cell).T
        return out

    def scaled_kpts(self) -> FloatArray:
        out: FloatArray = self.kpts @ self.cell.T / (2.0 * pi)
        return out

    def wavefunctions(self, ik: int) -> tuple[IntArray, ComplexArray]:
        """Miller indices (npw, 3) and coefficients (nbnd, npw) of k-point ik (0-based)"""
        h5: Path = self.save / f"wfc{ik + 1}.hdf5"
        if not h5.exists():
            raise FileNotFoundError(f"{h5}: only wave functions in HDF5 files are read (pw.x built with HDF5)")
        import h5py  # optional dependency, imported only when wave functions are read

        f: Any
        with h5py.File(h5, "r") as f:
            if int(f.attrs["npol"]) != 1 or str(f.attrs["gamma_only"].decode()).upper() != ".FALSE.":
                raise ValueError(f"{h5}: collinear runs without the Gamma-point trick only")
            mill: IntArray = np.asarray(f["MillerIndices"][:], dtype=np.int64)
            raw: FloatArray = np.asarray(f["evc"][:], dtype=float)
        coef: ComplexArray = raw[:, 0::2] + 1j * raw[:, 1::2]
        return mill, coef


def _vec(e: ET.Element | None) -> FloatArray:
    if e is None or e.text is None:
        raise ValueError("missing entry in data-file-schema.xml")
    return np.array(e.text.split(), dtype=float)


def _req(root: ET.Element, path: str) -> ET.Element:
    e: ET.Element | None = root.find(path)
    if e is None:
        raise ValueError(f"data-file-schema.xml: no {path}")
    return e


def _flag(root: ET.Element, path: str) -> bool:
    e: ET.Element | None = root.find(path)
    return e is not None and (e.text or "").strip().lower() == "true"


def read_run(save: str | Path) -> QeRun:
    save_dir: Path = Path(save)
    root: ET.Element = ET.parse(save_dir / "data-file-schema.xml").getroot()
    out: ET.Element = _req(root, "output")
    st: ET.Element = _req(out, "atomic_structure")
    alat: float = float(st.get("alat", "0"))
    cell: FloatArray = np.array([_vec(_req(st, f"cell/a{i}")) for i in (1, 2, 3)])
    atoms: list[ET.Element] = st.findall("atomic_positions/atom")
    species: list[str] = [a.get("name", "") for a in atoms]
    positions: FloatArray = np.array([_vec(a) for a in atoms])
    pseudo: dict[str, str] = {
        s.get("name", ""): (_req(s, "pseudo_file").text or "").strip() for s in out.findall("atomic_species/species")
    }
    bs: ET.Element = _req(out, "band_structure")
    ks: list[ET.Element] = bs.findall("ks_energies")
    tpiba: float = 2.0 * pi / alat
    kpts: FloatArray = np.array([_vec(_req(k, "k_point")) for k in ks]) * tpiba
    weights: FloatArray = np.array([float(_req(k, "k_point").get("weight", "0")) for k in ks])
    eps: FloatArray = np.array([_vec(_req(k, "eigenvalues")) for k in ks])
    mp: ET.Element | None = bs.find("starting_k_points/monkhorst_pack")
    if mp is None:
        raise ValueError("the k-points must come from an automatic (Monkhorst-Pack) mesh")
    grid: tuple[int, int, int] = (int(mp.get("nk1", "0")), int(mp.get("nk2", "0")), int(mp.get("nk3", "0")))
    shift: tuple[int, int, int] = (int(mp.get("k1", "0")), int(mp.get("k2", "0")), int(mp.get("k3", "0")))
    fe: ET.Element = _req(out, "basis_set/fft_grid")
    fft: tuple[int, int, int] = (int(fe.get("nr1", "0")), int(fe.get("nr2", "0")), int(fe.get("nr3", "0")))
    func: ET.Element | None = out.find("dft/functional")
    flags: dict[str, bool] = {
        "lsda": _flag(bs, "lsda"),
        "noncolin": _flag(bs, "noncolin"),
        "spinorbit": _flag(bs, "spinorbit"),
        "nosym": _flag(root, "input/symmetry_flags/nosym"),
        "noinv": _flag(root, "input/symmetry_flags/noinv"),
    }
    return QeRun(
        save=save_dir,
        cell=cell,
        species=species,
        positions=positions,
        pseudo_files=pseudo,
        nelec=float(_req(bs, "nelec").text or "0"),
        nbnd=int(_req(bs, "nbnd").text or "0"),
        kpts=kpts,
        weights=weights,
        eps=eps,
        mp_grid=grid,
        mp_shift=shift,
        fft=fft,
        functional=(func.text or "").strip() if func is not None else "",
        flags=flags,
    )
