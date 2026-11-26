"""Integration points of a pw.x PAW run, the ground-state fields on them, and the densities of single states.

Points: the dense FFT grid (weight V / N), then for every atom its sphere points twice, all-electron (weight +) and
pseudo (weight -), so that  sum_p w_p F(fields_p)  integrates a local function of the all-electron density: grid n~,
spheres n^1 + n_c and n~^1 (the pseudo core n~_c only with pseudo_core; SPEC.md section 6).
Fields: spin-summed density n, sigma = |grad n|^2 and tau, with n_c and tau_c on the all-electron sphere points.
k-points are reduced only by time reversal: the partners give the same |psi|^2 and the real part of the density matrix.
"""

from __future__ import annotations

import dataclasses
import sys
import xml.etree.ElementTree as ET
from math import pi, sqrt

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn

from ._types import ComplexArray, FloatArray, IntArray
from .paw import SphereSpecies, build_species, projector_matrix, sphere_density, sphere_fields
from .qe_files import QeRun
from .upf import PawDataset, pseudo_core_kinetic, read_upf


def _ecutwfc(run: QeRun) -> float:
    root: ET.Element = ET.parse(run.save / "data-file-schema.xml").getroot()
    e: ET.Element | None = root.find("output/basis_set/ecutwfc")
    if e is None or e.text is None:
        raise ValueError("no ecutwfc in data-file-schema.xml")
    return float(e.text)


@dataclasses.dataclass
class AtomBlock:
    atom: int
    species: SphereSpecies
    offset: int  # first projection column of the atom


class QePoints:
    """the point set of a pw.x run and the state densities on it"""

    def __init__(self, run: QeRun, check_norm: bool = True, pseudo_core: bool = False) -> None:
        if run.flags["lsda"] or run.flags["noncolin"] or run.flags["spinorbit"]:
            raise ValueError("spin-paired collinear runs only")
        if not run.flags["nosym"]:
            raise ValueError(
                "the run with the empty bands must use nosym = .true. (k-points reduced only by time reversal)"
            )
        nocc_f: float = run.nelec / 2.0
        if abs(nocc_f - round(nocc_f)) > 1e-8:
            raise ValueError("an even number of electrons is needed (insulators, spin-paired)")
        self.run: QeRun = run
        self.pseudo_core: bool = pseudo_core
        self.nocc: int = int(round(nocc_f))
        self.datasets: dict[str, PawDataset] = {s: read_upf(run.save / f) for s, f in run.pseudo_files.items()}
        qmax: float = sqrt(2.0 * _ecutwfc(run)) + float(np.linalg.norm(run.kpts, axis=1).max()) + 1.0
        sp: dict[str, SphereSpecies] = {s: build_species(ds, qmax) for s, ds in self.datasets.items()}
        self.blocks: list[AtomBlock] = []
        off: int = 0
        a: int
        s: str
        for a, s in enumerate(run.species):
            self.blocks.append(AtomBlock(atom=a, species=sp[s], offset=off))
            off += sp[s].nchan
        self.nproj: int = off
        self.ngrid: int = int(np.prod(run.fft))
        self._ground_state(check_norm)

    # ------------------------------------------------------------------------------------------------------------ plane waves
    def _q(self, ik: int, mill: IntArray) -> FloatArray:
        out: FloatArray = self.run.kpts[ik][None, :] + mill @ self.run.recip
        return out

    def _to_grid(self, mill: IntArray, coef: ComplexArray) -> ComplexArray:
        """periodic parts u_n(r) on the dense grid for the rows of coef (nb, npw): u(r_j) = sum_G c_G exp(i G.r_j)"""
        N: tuple[int, int, int] = self.run.fft
        idx: tuple[IntArray, IntArray, IntArray] = (mill[:, 0] % N[0], mill[:, 1] % N[1], mill[:, 2] % N[2])
        out: ComplexArray = np.zeros((coef.shape[0],) + N, dtype=complex)
        out[(slice(None),) + idx] = coef
        res: ComplexArray = np.fft.ifftn(out, axes=(1, 2, 3)) * self.ngrid
        return res

    def projections(self, ik: int, mill: IntArray, coef: ComplexArray) -> ComplexArray:
        """<p~_p^a|psi~_n> (nbnd, nproj) at k-point ik"""
        q: FloatArray = self._q(ik, mill)
        out: ComplexArray = np.zeros((coef.shape[0], self.nproj), dtype=complex)
        b: AtomBlock
        for b in self.blocks:
            B: ComplexArray = projector_matrix(b.species, q, self.run.positions[b.atom], self.run.volume)
            out[:, b.offset : b.offset + b.species.nchan] = coef @ B.T
        return out

    def _block(self, P: ComplexArray, b: AtomBlock) -> ComplexArray:
        out: ComplexArray = P[:, b.offset : b.offset + b.species.nchan]
        return out

    # ----------------------------------------------------------------------------------------------------------- ground state
    def _ground_state(self, check_norm: bool) -> None:
        run: QeRun = self.run
        N: tuple[int, int, int] = run.fft
        vol: float = run.volume
        nt: FloatArray = np.zeros(N)
        taut: FloatArray = np.zeros(N)
        D: list[FloatArray] = [np.zeros((b.species.nchan, b.species.nchan)) for b in self.blocks]
        self.P: list[ComplexArray] = []
        self.p2: list[FloatArray] = []
        iu: IntArray
        ju: IntArray
        iu, ju = np.triu_indices(run.nbnd, k=1)
        keep: np.ndarray = (iu < self.nocc) & (ju >= self.nocc)
        self.pair_n: IntArray = iu[keep]
        self.pair_m: IntArray = ju[keep]
        self.norm_error: float = 0.0
        ik: int
        for ik in range(len(run.kpts)):
            mill: IntArray
            coef: ComplexArray
            mill, coef = run.wavefunctions(ik)
            q: FloatArray = self._q(ik, mill)
            P: ComplexArray = self.projections(ik, mill, coef)
            self.P.append(P)
            if check_norm:
                snorm: FloatArray = np.sum(np.abs(coef) ** 2, axis=1)
                b: AtomBlock
                for b in self.blocks:
                    Pa: ComplexArray = self._block(P, b)
                    snorm = snorm + np.real(np.einsum("np,pq,nq->n", Pa.conj(), b.species.overlap, Pa))
                self.norm_error = max(self.norm_error, float(np.abs(snorm - 1.0).max()))
            wk: float = float(run.weights[ik])
            occ: ComplexArray = coef[: self.nocc]
            u: ComplexArray = self._to_grid(mill, occ)
            nt += wk * np.sum(np.abs(u) ** 2, axis=0) / vol
            v: int
            for v in range(3):
                du: ComplexArray = self._to_grid(mill, occ * (1j * q[:, v])[None, :])
                taut += 0.5 * wk * np.sum(np.abs(du) ** 2, axis=0) / vol
            Da: FloatArray
            for b, Da in zip(self.blocks, D):
                Po: ComplexArray = self._block(P, b)[: self.nocc]
                Da += wk * np.real(Po.conj().T @ Po)
            # momentum matrix elements of the band pairs (plane-wave part + PAW correction), averaged over the directions
            p2: FloatArray = np.zeros(len(self.pair_n))
            for v in range(3):
                M: ComplexArray = (coef.conj() * q[:, v][None, :]) @ coef.T
                for b in self.blocks:
                    Pb: ComplexArray = self._block(P, b)
                    M = M - 1j * (Pb.conj() @ b.species.nabla[:, :, v] @ Pb.T)
                p2 += np.abs(M[self.pair_n, self.pair_m]) ** 2
            self.p2.append(p2 / 3.0)
        if check_norm and self.norm_error > 1e-4:
            print(f"warning: <psi|S|psi> deviates from 1 by up to {self.norm_error:.2e}", file=sys.stderr)
        self.D: list[FloatArray] = D
        nct: FloatArray
        dnct: FloatArray
        tauct: FloatArray
        nct, dnct, tauct = self._pseudo_core()
        if not self.pseudo_core:  # the core only in the all-electron sphere parts
            nct, dnct, tauct = np.zeros_like(nct), np.zeros_like(dnct), np.zeros_like(tauct)
        dnt: FloatArray = self._gradient(nt)
        self.grid_core: FloatArray = nct.ravel()
        self.grid_n: FloatArray = nt + nct
        grad: FloatArray = dnt + dnct
        w: list[FloatArray] = [np.full(self.ngrid, vol / self.ngrid)]
        n: list[FloatArray] = [self.grid_n.ravel()]
        sg: list[FloatArray] = [np.sum(grad**2, axis=0).ravel()]
        tau: list[FloatArray] = [(taut + tauct).ravel()]
        atom_p: list[IntArray] = [np.full(self.ngrid, -1, dtype=np.int64)]
        for b, Da in zip(self.blocks, D):
            sign: float
            kind: str
            for kind, sign in (("ae", 1.0), ("ps", -1.0)):
                nn: FloatArray
                ss: FloatArray
                tt: FloatArray
                nn, ss, tt = sphere_fields(b.species, Da, kind, core=kind == "ae" or self.pseudo_core)
                w.append(sign * b.species.weight)
                n.append(nn)
                sg.append(ss)
                tau.append(tt)
                atom_p.append(np.full(b.species.npts, b.atom, dtype=np.int64))
        self.w: FloatArray = np.concatenate(w)
        self.n: FloatArray = np.concatenate(n)
        self.sigma: FloatArray = np.concatenate(sg)
        self.tau: FloatArray = np.concatenate(tau)
        self.atom_p: IntArray = np.concatenate(atom_p)

    def _gvectors(self) -> FloatArray:
        N: tuple[int, int, int] = self.run.fft
        m: list[FloatArray] = [np.fft.fftfreq(n) * n for n in N]
        mm: tuple[FloatArray, ...] = tuple(np.meshgrid(*m, indexing="ij"))
        G: FloatArray = np.einsum("i...,ij->j...", np.array(mm), self.run.recip)
        return G

    def _gradient(self, f: FloatArray) -> FloatArray:
        G: FloatArray = self._gvectors()
        fg: ComplexArray = np.fft.fftn(f)
        out: FloatArray = np.array([np.real(np.fft.ifftn(1j * G[v] * fg)) for v in range(3)])
        return out

    def _pseudo_core(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """pseudo core density on the grid, its gradient and the pseudo core kinetic-energy density, from their Fourier series"""
        run: QeRun = self.run
        G: FloatArray = self._gvectors()
        gn: FloatArray = np.sqrt(np.sum(G**2, axis=0))
        rho: ComplexArray = np.zeros(run.fft, dtype=complex)
        tau: ComplexArray = np.zeros(run.fft, dtype=complex)
        q: FloatArray = np.linspace(0.0, float(gn.max()) + 1e-6, 4000)
        s: str
        ds: PawDataset
        for s, ds in self.datasets.items():
            w: FloatArray = 4.0 * pi * ds.r**2 * ds.rab
            j0: FloatArray = spherical_jn(0, np.outer(q, ds.r))
            fq: FloatArray = CubicSpline(q, j0 @ (ds.nct * w))(gn)
            tq: FloatArray = CubicSpline(q, j0 @ (pseudo_core_kinetic(ds) * w))(gn)
            a: int
            name: str
            for a, name in enumerate(run.species):
                if name == s:
                    ph: ComplexArray = np.exp(-1j * np.einsum("v...,v->...", G, run.positions[a]))
                    rho += ph * fq
                    tau += ph * tq
        rho /= run.volume
        tau /= run.volume
        nct: FloatArray = np.real(np.fft.ifftn(rho)) * self.ngrid
        dn: FloatArray = np.array([np.real(np.fft.ifftn(1j * G[v] * rho)) * self.ngrid for v in range(3)])
        tct: FloatArray = np.real(np.fft.ifftn(tau)) * self.ngrid
        return nct, dn, tct

    def state_density(self, ik: int, bands: list[int]) -> FloatArray:
        """density of one electron spread over the bands at k-point ik, on the points (no core)"""
        mill: IntArray
        coef: ComplexArray
        mill, coef = self.run.wavefunctions(ik)
        u: ComplexArray = self._to_grid(mill, coef[bands])
        parts: list[FloatArray] = [(np.sum(np.abs(u) ** 2, axis=0) / self.run.volume / len(bands)).ravel()]
        P: ComplexArray = self.P[ik]
        b: AtomBlock
        for b in self.blocks:
            Pb: ComplexArray = self._block(P, b)[bands]
            Ds: FloatArray = np.real(Pb.conj().T @ Pb) / len(bands)
            parts.append(sphere_density(b.species, Ds, "ae"))
            parts.append(sphere_density(b.species, Ds, "ps"))
        out: FloatArray = np.concatenate(parts)
        return out

    def electrons(self) -> float:
        """number of valence electrons integrated over the points (a check of the projections and the sphere parts)"""
        nv: FloatArray = self.n.copy()
        nv[: self.ngrid] -= self.grid_core
        o: int = self.ngrid
        b: AtomBlock
        for b in self.blocks:
            npt: int = b.species.npts
            nv[o : o + npt] -= b.species.core["ae"]
            if self.pseudo_core:
                nv[o + npt : o + 2 * npt] -= b.species.core["ps"]
            o += 2 * npt
        return float(self.w @ nv)
