"""PAW datasets in the UPF v2 format (as written by ld1.x, e.g. pslibrary), with the GIPAW core orbitals.

Radial functions on the dataset's mesh r (bohr), with the integration weights rab = dr/di:
    partial waves and projectors as r phi(r), r phi~(r), r beta(r);  core densities n_c(r), n~_c(r) (not multiplied by 4 pi r^2);
    core orbitals as r psi_c(r).
Bound projectors are those generated at the eigenvalue of an atomic state (energy 0.00 in the generation input of ld1.x).
"""

from __future__ import annotations

import dataclasses
import re
import xml.etree.ElementTree as ET
from math import pi
from pathlib import Path

import numpy as np

from ._types import FloatArray, IntArray


@dataclasses.dataclass
class Projector:
    label: str
    ell: int
    bound: bool
    occupation: float
    beta: FloatArray  # r beta(r)
    ae: FloatArray  # r phi(r)
    ps: FloatArray  # r phi~(r)
    cutoff_index: int


@dataclasses.dataclass
class CoreOrbital:
    label: str
    n: int
    ell: int
    rpsi: FloatArray


@dataclasses.dataclass
class PawDataset:
    path: str
    element: str
    z_valence: float
    functional: str
    relativistic: str
    r: FloatArray
    rab: FloatArray
    projectors: list[Projector]
    nc: FloatArray  # all-electron core density
    nct: FloatArray  # pseudo core density
    core: list[CoreOrbital]
    config: str
    n_sphere: int  # number of mesh points of the augmentation sphere (all-electron and pseudo parts differ only inside)

    @property
    def n_core_electrons(self) -> float:
        return float(np.sum(4.0 * pi * self.r**2 * self.nc * self.rab))


def _floats(e: ET.Element | None) -> FloatArray:
    if e is None or e.text is None:
        raise ValueError("missing UPF array")
    return np.array(e.text.split(), dtype=float)


def _generation_lines(info: str, labels: list[str]) -> list[tuple[str, float, float]]:
    """(label, occupation, energy) of every projector from the ld1.x input stored in PP_INPUTFILE: the block of states after the
    namelists, matched in order to the projector labels (states that are not projectors are skipped)"""
    lines: list[list[str]] = [ln.split() for ln in info.splitlines()]
    i: int
    for i in range(len(lines)):
        row: list[str] = lines[i]
        if len(row) != 1 or not row[0].isdigit():
            continue
        block: list[list[str]] = lines[i + 1 : i + 1 + int(row[0])]
        if not block or not all(len(x) >= 5 and re.fullmatch(r"\d[SPDF]", x[0].upper()) for x in block):
            continue
        out: list[tuple[str, float, float]] = []
        x: list[str]
        for x in block:
            if len(out) < len(labels) and x[0].upper() == labels[len(out)].upper():
                out.append((x[0].upper(), float(x[3]), float(x[4])))
        if len(out) == len(labels):
            return out
    raise ValueError("the generation input of the projectors (PP_INPUTFILE of ld1.x) was not found")


def _last_mismatch(a: FloatArray, b: FloatArray) -> int:
    scale: float = float(max(np.abs(a).max(), np.abs(b).max(), 1e-300))
    idx: IntArray = np.where(np.abs(a - b) > 1e-12 * scale)[0]
    return int(idx.max()) if idx.size else 0


def read_upf(path: str | Path) -> PawDataset:
    text: str = Path(path).read_text()
    root: ET.Element = ET.fromstring(text.replace("&amp;", "&").replace("&", "&amp;"))
    head: ET.Element | None = root.find("PP_HEADER")
    if head is None:
        raise ValueError(f"{path}: not a UPF v2 file")
    if head.get("is_paw", "").lower() not in ("true", "t", ".true."):
        raise ValueError(f"{path}: not a PAW dataset (this exporter supports PAW only)")
    if head.get("has_gipaw", "").lower() not in ("true", "t", ".true."):
        raise ValueError(f"{path}: no GIPAW core orbitals (the core kinetic-energy density is built from them)")
    element: str = head.get("element", "").strip()
    r: FloatArray = _floats(root.find("PP_MESH/PP_R"))
    rab: FloatArray = _floats(root.find("PP_MESH/PP_RAB"))
    nproj: int = int(head.get("number_of_proj", "0"))
    info_el: ET.Element | None = root.find("PP_INFO/PP_INPUTFILE")
    info: str = info_el.text if info_el is not None and info_el.text else ""
    labels: list[str] = []
    k: int
    for k in range(1, nproj + 1):
        bl: ET.Element | None = root.find(f"PP_NONLOCAL/PP_BETA.{k}")
        labels.append(bl.get("label", "").strip() if bl is not None else "")
    gen: list[tuple[str, float, float]] = _generation_lines(info, labels)
    cfg_match: re.Match[str] | None = re.search(r"config\s*=\s*'([^']*)'", info)
    projectors: list[Projector] = []
    i: int
    for i in range(1, nproj + 1):
        beta_el: ET.Element | None = root.find(f"PP_NONLOCAL/PP_BETA.{i}")
        ae_el: ET.Element | None = root.find(f"PP_FULL_WFC/PP_AEWFC.{i}")
        ps_el: ET.Element | None = root.find(f"PP_FULL_WFC/PP_PSWFC.{i}")
        if beta_el is None or ae_el is None or ps_el is None:
            raise ValueError(f"{path}: projector {i} or its partial waves are missing")
        beta: FloatArray = _floats(beta_el)
        ell: int = int(beta_el.get("angular_momentum", "0"))
        label: str
        occ: float
        energy: float
        label, occ, energy = gen[i - 1]
        projectors.append(
            Projector(
                label=label,
                ell=ell,
                bound=bool(energy == 0.0),
                occupation=occ,
                beta=beta,
                ae=_floats(ae_el),
                ps=_floats(ps_el),
                cutoff_index=int(beta_el.get("cutoff_radius_index", str(len(r)))),
            )
        )
    nc: FloatArray = _floats(root.find("PP_PAW/PP_AE_NLCC"))
    nct_el: ET.Element | None = root.find("PP_NLCC")
    nct: FloatArray = _floats(nct_el) if nct_el is not None else np.zeros_like(r)
    gipaw: list[CoreOrbital] = []
    c: ET.Element
    gip: ET.Element | None = root.find("PP_GIPAW/PP_GIPAW_CORE_ORBITALS")
    for c in gip if gip is not None else []:
        gipaw.append(
            CoreOrbital(
                label=c.get("label", "").strip(),
                n=int(float(c.get("n", "0"))),
                ell=int(float(c.get("l", "0"))),
                rpsi=_floats(c),
            )
        )
    core: list[CoreOrbital] = _core_orbitals(gipaw, projectors, nc, r, rab, str(path))
    n_sphere: int = max([_last_mismatch(p.ae, p.ps) for p in projectors] + [_last_mismatch(nc, nct)])
    ds: PawDataset = PawDataset(
        path=str(path),
        element=element,
        z_valence=float(head.get("z_valence", "0")),
        functional=" ".join(head.get("functional", "").split()),
        relativistic=head.get("relativistic", ""),
        r=r,
        rab=rab,
        projectors=projectors,
        nc=nc,
        nct=nct,
        core=core,
        config=cfg_match.group(1) if cfg_match else "",
        n_sphere=min(n_sphere + 3, len(r)),
    )
    return ds


def _core_orbitals(
    gipaw: list[CoreOrbital], projectors: list[Projector], nc: FloatArray, r: FloatArray, rab: FloatArray, path: str
) -> list[CoreOrbital]:
    """the core shells: the GIPAW core orbitals that are not valence states (some datasets list valence orbitals there and omit one
    core shell), plus at most one missing closed shell taken from the rest of the all-electron core density"""
    valence: set[tuple[int, int]] = {(int(p.label[0]), "SPDF".index(p.label[1].upper())) for p in projectors}
    kept: list[CoreOrbital] = [c for c in gipaw if (c.n, c.ell) not in valence]
    w: FloatArray = 4.0 * pi * r**2 * rab
    rho: FloatArray = np.zeros_like(r)
    c: CoreOrbital
    for c in kept:
        rho = rho + _closed_shell_occupation(c.ell) * c.rpsi**2 / (4.0 * pi * r**2)
    rest: FloatArray = nc - rho
    nrest: float = float(np.sum(w * rest))
    if abs(nrest) < 1e-4:
        return kept
    ell: int
    for ell in range(4):
        occ: float = _closed_shell_occupation(ell)
        if abs(nrest - occ) < 1e-3 and float(rest.min()) > -1e-6 * float(nc.max()):
            radial: FloatArray = np.sqrt(np.clip(rest, 0.0, None) * 4.0 * pi / occ)
            return kept + [CoreOrbital(label=f"core l={ell} (from the core density)", n=0, ell=ell, rpsi=radial * r)]
    raise ValueError(f"{path}: the core orbitals miss {nrest:.4f} core electrons, which is not one closed shell")


def _closed_shell_occupation(ell: int) -> float:
    return 2.0 * (2 * ell + 1)


def core_density(ds: PawDataset) -> FloatArray:
    """sum over the core orbitals of f |psi_c|^2 (closed shells), on the mesh"""
    out: FloatArray = np.zeros_like(ds.r)
    c: CoreOrbital
    for c in ds.core:
        out = out + _closed_shell_occupation(c.ell) * c.rpsi**2
    return out / (4.0 * pi * ds.r**2)


def radial_derivative(f: FloatArray, ds: PawDataset) -> FloatArray:
    """df/dr on the mesh (second-order finite differences in the mesh index)"""
    out: FloatArray = np.gradient(f) / ds.rab
    return out


def orbital_kinetic(rpsi: FloatArray, ell: int, occupation: float, ds: PawDataset) -> FloatArray:
    """kinetic-energy density f/(8 pi) [R'^2 + l (l + 1) (R / r)^2] of a spherically averaged shell, R = r psi / r"""
    rr: FloatArray = ds.r
    R: FloatArray = rpsi / rr
    dR: FloatArray = radial_derivative(R, ds)
    return occupation / (8.0 * pi) * (dR**2 + ell * (ell + 1) * (R / rr) ** 2)


def core_kinetic(ds: PawDataset) -> FloatArray:
    """kinetic-energy density of the core, from the GIPAW core orbitals"""
    out: FloatArray = np.zeros_like(ds.r)
    c: CoreOrbital
    for c in ds.core:
        out = out + orbital_kinetic(c.rpsi, c.ell, _closed_shell_occupation(c.ell), ds)
    return out


def pseudo_core_kinetic(ds: PawDataset) -> FloatArray:
    """pseudo core kinetic-energy density tau~_c = tau_c (n~_c / n_c)^(5/3): smooth where the core density is pseudized and equal
    to tau_c outside (used identically on the grid and the pseudo sphere points, so it cancels in the PAW sum)"""
    ratio: FloatArray = np.where(ds.nc > 1e-300, ds.nct / np.maximum(ds.nc, 1e-300), 0.0)
    out: FloatArray = core_kinetic(ds) * np.clip(ratio, 0.0, 1.0) ** (5.0 / 3.0)
    return out
