"""Integration points of a GPAW PAW calculation and the densities of single states on them.

Points: the fine real-space grid (weight dv) and, for every atom with a PAW correction, the radial x Lebedev points of
its augmentation sphere twice -- all-electron (weight +) and pseudo (weight -) -- so that sum_p w_p F(n_p) ...
integrates a local function of the all-electron density.  On every point: the spin-summed density n, sigma = |grad n|^2
and the kinetic-energy density tau (core densities included).  State densities: one electron spread over a group of
degenerate bands at one k-point (symmetrised over the star of k).  Serial order of the points: the fine grid in C order,
then the + and - blocks of every atom in atom order; with domain decomposition every rank holds its part of the grid and
the spheres of its atoms.  GPL-3.0-or-later (uses GPAW).
"""

from __future__ import annotations

from collections.abc import Mapping
from math import pi, sqrt
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from . import kpar
from ._types import FloatArray, IntArray


def frozen_density(calc: Any) -> None:
    """pseudo density, atomic density matrices and fine-grid density rebuilt from the current orbitals and occupations"""
    dens: Any
    wfs: Any
    dens, wfs = calc.density, calc.wfs
    dens.calculate_pseudo_density(wfs)
    wfs.calculate_atomic_density_matrices(dens.D_asp)
    dens.interpolate_pseudo_density()


def aux_mgga(calc: Any, name: str = "TPSS") -> Any:
    """an auxiliary meta-GGA bound to the calculator: its gradient stencils, kinetic-energy interpolation and PAW kinetic expansions
    give n, sigma, tau on the points (its energy is not used)"""
    from gpaw.xc import XC

    rx: Any = XC(name)
    rx.set_grid_descriptor(calc.hamiltonian.finegd)
    rx.initialize(calc.density, calc.hamiltonian, calc.wfs)
    rx.set_positions(calc.spos_ac)
    return rx


def ground_arrays(calc: Any, xc: Any) -> tuple[FloatArray, FloatArray, dict[int, FloatArray]]:
    dens: Any
    wfs: Any
    dens, wfs = calc.density, calc.wfs
    taut_sG: FloatArray = wfs.calculate_kinetic_energy_density()
    if xc.tauct_G is None:
        xc.set_positions(calc.spos_ac)
    taut_sg: FloatArray = np.empty_like(dens.nt_sg)
    taut_G: FloatArray
    taut_g: FloatArray
    for taut_G, taut_g in zip(taut_sG, taut_sg):
        taut_G += 1.0 / wfs.nspins * xc.tauct_G
        xc.distribute_and_interpolate(taut_G, taut_g)
    return dens.nt_sg.copy(), taut_sg, {a: np.array(D_sp, float) for a, D_sp in dens.D_asp.items()}


def sphere_blocks(
    calc: Any, xc: Any, D_asp: Mapping[int, FloatArray], core: bool = True, kinetic: bool = True
) -> list[dict[str, Any]]:
    """per atom with a PAW correction: an all-electron (+) and a pseudo (-) block with n, radial derivative a, angular gradient b and
    (kinetic) tau on all radial x Lebedev points; core=False for a density-matrix change (no core parts)"""
    from gpaw.sphere.lebedev import Y_nL, weight_n
    from gpaw.xc.pawcorrection import rnablaY_nLv

    out: list[dict[str, Any]] = []
    a: int
    for a in D_asp.keys():
        xcc: Any = calc.wfs.setups[a].xc_correction
        if xcc is None:
            continue
        if kinetic and xcc.tau_npg is None:
            xcc.tau_npg, xcc.taut_npg = xc.initialize_kinetic(xcc)
        rgd: Any = xcc.rgd
        D_sp: FloatArray = np.asarray(D_asp[a], float)
        ns: int = len(D_sp)
        D_sLq: FloatArray = np.inner(D_sp, xcc.B_pqL.T)
        Lmax: int = D_sLq.shape[1]
        Y: FloatArray = Y_nL[:, :Lmax]
        rY: FloatArray = np.ascontiguousarray(rnablaY_nLv[:, :Lmax].transpose(0, 2, 1))
        w_g: FloatArray = 4.0 * pi * rgd.r_g**2 * rgd.dr_g
        w_g[0] = 0.0
        nq: FloatArray
        ncore: FloatArray
        tau_npg: Any
        tauc: Any
        sign: float
        for nq, ncore, tau_npg, tauc, sign in (
            (xcc.n_qg, xcc.nc_g, xcc.tau_npg, xcc.tauc_g, 1.0),
            (xcc.nt_qg, xcc.nct_g, xcc.taut_npg, xcc.tauct_g, -1.0),
        ):
            n_sLg: FloatArray = D_sLq @ nq
            if core:
                n_sLg[:, 0] += sqrt(4 * pi) / ns * ncore
            blk: dict[str, Any] = dict(
                n=np.matmul(Y, n_sLg).transpose(1, 0, 2),
                w=sign * weight_n[:, None] * w_g[None],
                sign=int(sign),
                atom=a,
                r2=rgd.r_g**2,
                m=len(weight_n) * rgd.N,
            )
            if kinetic:
                dndr_sLg: FloatArray = np.empty_like(n_sLg)
                s_: int
                for s_ in range(ns):
                    L: int
                    for L in range(Lmax):
                        rgd.derivative(n_sLg[s_, L], dndr_sLg[s_, L])
                blk["a"] = np.matmul(Y, dndr_sLg).transpose(1, 0, 2)
                blk["b"] = np.stack([np.matmul(rY[:, v], n_sLg) for v in range(3)]).transpose(2, 0, 1, 3)
                tau_nsg: FloatArray = np.stack([D_sp @ tau_npg[i_] for i_ in range(len(tau_npg))])
                if core:
                    tau_nsg = tau_nsg + tauc / (sqrt(4 * pi) * ns)
                blk["tau"] = tau_nsg
            out.append(blk)
    return out


def _sigma_total(a: FloatArray, b: FloatArray, r2: FloatArray, ns: int) -> FloatArray:
    """|grad n_total|^2 (n, g) of the sphere expansion (the r = 0 point of the angular part copied from r_1)"""
    at: FloatArray = a.sum(1)
    bt: FloatArray = b.sum(2)  # (n, v, g)
    ang: FloatArray = (bt * bt).sum(1)
    ang[:, 1:] /= r2[1:]
    ang[:, 0] = ang[:, 1]
    return ang + at * at


def _flat_total(x: FloatArray) -> FloatArray:
    """(n, s, g) -> spin sum, n-major flat"""
    return cast(FloatArray, x.sum(1).reshape(-1))


def sphere_layout(calc: Any) -> list[tuple[int, int]]:
    """(atom, points of each of its two blocks) for every atom with a PAW correction, in the serial order of the blocks"""
    from gpaw.sphere.lebedev import weight_n

    out: list[tuple[int, int]] = []
    a: int
    setup: Any
    for a, setup in enumerate(calc.wfs.setups):
        if setup.xc_correction is not None:
            out.append((a, len(weight_n) * int(setup.xc_correction.rgd.N)))
    return out


class PointSet:
    """the point set of a converged calculator and the state densities on it"""

    calc: Any
    xc: Any
    blocks: list[dict[str, Any]]
    w: FloatArray
    n: FloatArray
    sigma: FloatArray
    tau: FloatArray
    ngrid: int
    ns: int
    atom: IntArray

    def __init__(self, calc: Any) -> None:
        from gpaw.xc.gga import calculate_sigma

        frozen_density(calc)
        self.calc = calc
        xc: Any
        self.xc = xc = aux_mgga(calc)
        nt_sg: FloatArray
        taut_sg: FloatArray
        D_asp: dict[int, FloatArray]
        nt_sg, taut_sg, D_asp = ground_arrays(calc, xc)
        gd: Any = calc.hamiltonian.finegd
        ns: int = len(nt_sg)
        sigma_xg: FloatArray
        sigma_xg, _ = calculate_sigma(gd, xc.grad_v, np.ascontiguousarray(nt_sg, float).copy())
        sig_tot: FloatArray = sigma_xg[0] if ns == 1 else sigma_xg[0] + 2 * sigma_xg[1] + sigma_xg[2]
        w: list[FloatArray]
        n: list[FloatArray]
        s: list[FloatArray]
        t: list[FloatArray]
        w, n, s, t = (
            [np.full(nt_sg[0].size, gd.dv)],
            [nt_sg.sum(0).ravel()],
            [sig_tot.ravel()],
            [taut_sg.sum(0).ravel()],
        )
        self.blocks = sphere_blocks(calc, xc, D_asp)
        b: dict[str, Any]
        for b in self.blocks:
            w.append(b["w"].ravel())
            n.append(_flat_total(b["n"]))
            s.append(_sigma_total(b["a"], b["b"], b["r2"], ns).reshape(-1))
            t.append(b["tau"].sum(1).reshape(-1))
        self.w, self.n, self.sigma, self.tau = (np.concatenate(x) for x in (w, n, s, t))
        self.ngrid = nt_sg[0].size
        self.ns = ns
        self.atom = np.concatenate(
            [np.full(self.ngrid, -1, np.int32)] + [np.full(b["w"].size, b["atom"], np.int32) for b in self.blocks]
        )
        self._index: IntArray | None = None

    @property
    def nglobal(self) -> int:
        """number of points of the whole calculation (serial count)"""
        gd: Any = self.calc.hamiltonian.finegd
        return int(np.prod(gd.get_size_of_global_array())) + 2 * sum(m for _, m in sphere_layout(self.calc))

    def global_index(self) -> IntArray:
        """position of every local point in the serial order of the points"""
        if self._index is not None:
            return self._index
        gd: Any = self.calc.hamiltonian.finegd
        # the grid without the boundary point of non-periodic axes
        N: IntArray = np.asarray(gd.get_size_of_global_array(), np.int64)
        beg: IntArray = np.asarray(gd.beg_c, np.int64) - (1 - np.asarray(gd.pbc_c, np.int64))
        nloc: IntArray = np.asarray(gd.n_c, np.int64)
        ax: list[IntArray] = [np.arange(beg[c], beg[c] + nloc[c], dtype=np.int64) for c in range(3)]
        grid: IntArray = ((ax[0][:, None, None] * N[1] + ax[1][None, :, None]) * N[2] + ax[2][None, None, :]).ravel()
        start: dict[tuple[int, int], int] = {}
        o: int = int(np.prod(N))
        a: int
        m: int
        for a, m in sphere_layout(self.calc):
            start[(a, 1)] = o
            start[(a, -1)] = o + m
            o += 2 * m
        parts: list[IntArray] = [grid]
        b: dict[str, Any]
        for b in self.blocks:
            parts.append(start[(int(b["atom"]), int(b["sign"]))] + np.arange(b["w"].size, dtype=np.int64))
        self._index = np.concatenate(parts)
        return self._index

    def fingerprint(self) -> list[float]:
        """number of points and weighted sums of the fields over the whole calculation (the same values, to rounding, on any number of
        domain ranks and before and after the empty bands are added)"""
        f: FloatArray = np.array(
            [self.w.size, self.w.sum(), self.w @ self.n, self.w @ self.sigma, self.w @ self.tau], dtype=np.float64
        )
        kpar.domain_sum(self.calc, f)
        return [int(round(f[0]))] + [float(x) for x in f[1:]]

    def gather(self) -> dict[str, npt.NDArray[Any]] | None:
        """w, n, sigma, tau and the atom of every point in the serial order, on the domain root (None on the other ranks)"""
        parts: list[tuple[IntArray, FloatArray, FloatArray, FloatArray, FloatArray, IntArray]] | None = (
            kpar.gather_root(
                kpar.gcomm(self.calc), (self.global_index(), self.w, self.n, self.sigma, self.tau, self.atom)
            )
        )
        if parts is None:
            return None
        n: int = self.nglobal
        out: dict[str, npt.NDArray[Any]] = dict(
            w=np.empty(n), n=np.empty(n), sigma=np.empty(n), tau=np.empty(n), atom_p=np.empty(n, np.int32)
        )
        seen: npt.NDArray[np.bool_] = np.zeros(n, bool)
        part: tuple[IntArray, FloatArray, FloatArray, FloatArray, FloatArray, IntArray]
        for part in parts:
            i: IntArray = part[0]
            out["w"][i], out["n"][i], out["sigma"][i], out["tau"][i], out["atom_p"][i] = part[1:]
            seen[i] = True
        if not seen.all():
            raise RuntimeError(f"points: {int((~seen).sum())} of {n} points not held by any domain rank")
        return out

    def state_density(self, q: int, bands: list[int]) -> FloatArray:
        """density of one electron spread over `bands` at this rank's k-point kpt_u[q] (symmetrised over the star of k)
        on this rank's points; collective over the domain ranks of the k-point"""
        from gpaw.utilities import pack

        calc: Any = self.calc
        dens: Any
        wfs: Any
        dens, wfs = calc.density, calc.wfs
        kpt: Any = wfs.kpt_u[q]
        f_n: FloatArray = np.zeros(len(kpt.f_n))
        b: int
        for b in bands:
            f_n[b] += 1.0 / len(bands)
        nt_sG: FloatArray = wfs.gd.zeros(wfs.nspins)
        wfs.add_to_density_from_k_point_with_occupation(nt_sG, kpt, f_n)
        a_G: FloatArray
        for a_G in nt_sG:
            wfs.kd.symmetry.symmetrize(a_G, wfs.gd)
        nt_sg: FloatArray = np.empty_like(dens.nt_sg)
        x_G: FloatArray
        x_g: FloatArray
        for x_G, x_g in zip(nt_sG, nt_sg):
            dens.distribute_and_interpolate(x_G, x_g)
        D: Any = wfs.setups.empty_atomic_matrix(wfs.nspins, wfs.atom_partition)
        a: int
        D_sp: FloatArray
        for a, D_sp in D.items():
            ni: int = wfs.setups[a].ni
            D_sii: FloatArray = np.zeros((len(D_sp), ni, ni))
            wfs.calculate_atomic_density_matrices_k_point(D_sii, kpt, a, f_n)
            D_sp[:] = [pack(D_ii) for D_ii in D_sii]
        wfs.symmetrize_atomic_density_matrices(D)
        dD: dict[int, FloatArray] = {a: np.array(D[a], float) for a in dens.D_asp.keys()}
        parts: list[FloatArray] = [nt_sg.sum(0).ravel()] + [
            _flat_total(b["n"]) for b in sphere_blocks(calc, self.xc, dD, core=False, kinetic=False)
        ]
        return np.concatenate(parts)
