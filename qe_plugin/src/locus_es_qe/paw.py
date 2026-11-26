"""PAW quantities of one species: projections of plane-wave states, partial waves on the sphere points, and their matrix elements.

Channels p = (projector i, m) with the real harmonics of harmonics.py.  For atom a at R_a and a state with coefficients c_G at k:
    <p~_p|psi~> = 4 pi / sqrt(V) i^l sum_G c_G Y_lm(q^) beta~_i(|q|) exp(i q.R_a),   q = k + G,
    beta~_i(q) = int r beta_i(r) j_l(q r) r dr                                        (UPF stores r beta).
Sphere points: the dataset's radial mesh up to the radius beyond which the all-electron and pseudo partial waves and core densities
coincide, times a Lebedev rule of degree 11 (50 directions); weights r^2 dr (Simpson in the mesh index) x the Lebedev weights.
"""

from __future__ import annotations

import dataclasses
from math import pi, sqrt

import numpy as np
from scipy.integrate import lebedev_rule
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn

from ._types import ComplexArray, FloatArray
from .harmonics import harmonics, polynomial_gradients
from .upf import PawDataset, Projector, core_kinetic, pseudo_core_kinetic, radial_derivative

LEBEDEV_DEGREE: int = 11


def _simpson(n: int) -> FloatArray:
    """Simpson weights on n points of a uniform index grid (trapezoid on the last interval if n is even)"""
    w: FloatArray = np.ones(n)
    if n < 3:
        return w if n == 1 else np.full(n, 0.5)
    m: int = n if n % 2 == 1 else n - 1
    w[:m] = 1.0
    w[1 : m - 1 : 2] = 4.0
    w[2 : m - 1 : 2] = 2.0
    w[:m] /= 3.0
    if m < n:
        w[m - 1] += 0.5
        w[n - 1] = 0.5
    return w


@dataclasses.dataclass
class SphereSpecies:
    """partial waves of one species on its sphere points (np points = nr x nang, radial index first)"""

    ds: PawDataset
    channels: list[tuple[int, int, int]]  # (projector index, l, m index)
    unit: FloatArray  # (3, nang)
    r: FloatArray  # (nr,)
    weight: FloatArray  # (nr * nang,)  r^2 dr dOmega
    phi: dict[str, FloatArray]  # "ae", "ps": (nchan, npts)
    dphi: dict[str, FloatArray]  # (nchan, 3, npts)
    core: dict[str, FloatArray]  # core density (npts,)
    dcore: dict[str, FloatArray]  # its gradient (3, npts)
    tau_core: dict[str, FloatArray]  # core kinetic-energy density (npts,): tau_c (all-electron) and tau~_c (pseudo)
    overlap: FloatArray  # (nchan, nchan) <phi_p|phi_p'> - <phi~_p|phi~_p'>
    nabla: FloatArray  # (nchan, nchan, 3) <phi_p|grad|phi_p'> - <phi~_p|grad|phi~_p'>
    beta_q: list[CubicSpline]  # beta~_i(q) per projector

    @property
    def nchan(self) -> int:
        return len(self.channels)

    @property
    def npts(self) -> int:
        return int(self.weight.size)


def _beta_table(p: Projector, ds: PawDataset, qmax: float) -> CubicSpline:
    n: int = min(p.cutoff_index + 1, len(ds.r))
    r: FloatArray = ds.r[:n]
    w: FloatArray = ds.rab[:n] * _simpson(n)
    q: FloatArray = np.linspace(0.0, qmax, int(qmax / 0.005) + 2)
    jl: FloatArray = spherical_jn(p.ell, np.outer(q, r))
    vals: FloatArray = jl @ (p.beta[:n] * r * w)
    return CubicSpline(q, vals)


def build_species(ds: PawDataset, qmax: float) -> SphereSpecies:
    xyz: FloatArray
    wang: FloatArray
    xyz, wang = lebedev_rule(LEBEDEV_DEGREE)
    nr: int = ds.n_sphere
    r: FloatArray = ds.r[:nr]
    wrad: FloatArray = r**2 * ds.rab[:nr] * _simpson(nr)
    weight: FloatArray = (wrad[:, None] * wang[None, :]).ravel()
    rhat: FloatArray = np.repeat(xyz[:, None, :], nr, axis=1).reshape(3, -1)  # (3, npts), radial index first
    rr: FloatArray = np.repeat(r, xyz.shape[1])
    channels: list[tuple[int, int, int]] = []
    phi: dict[str, list[FloatArray]] = {"ae": [], "ps": []}
    dphi: dict[str, list[FloatArray]] = {"ae": [], "ps": []}
    i: int
    p: Projector
    for i, p in enumerate(ds.projectors):
        y: FloatArray = harmonics(p.ell, xyz)  # (2l + 1, nang)
        gs: FloatArray = polynomial_gradients(p.ell, xyz)  # (2l + 1, 3, nang)
        kind: str
        rf: FloatArray
        for kind, rf in (("ae", p.ae), ("ps", p.ps)):
            f: FloatArray = rf[:nr] / r
            df: FloatArray = radial_derivative(rf / ds.r, ds)[:nr]
            m: int
            for m in range(2 * p.ell + 1):
                ym: FloatArray = np.tile(y[m], nr)
                gm: FloatArray = np.tile(gs[m], (1, nr)) if p.ell > 0 else np.zeros((3, rr.size))
                fr: FloatArray = np.repeat(f, xyz.shape[1])
                dfr: FloatArray = np.repeat(df, xyz.shape[1])
                phi[kind].append(fr * ym)
                dphi[kind].append(dfr * ym * rhat + fr / rr * (gm - p.ell * ym * rhat))
                if kind == "ae":
                    channels.append((i, p.ell, m))
    tauc: FloatArray = core_kinetic(ds)
    core: dict[str, FloatArray] = {}
    dcore: dict[str, FloatArray] = {}
    nc: FloatArray
    for kind, nc in (("ae", ds.nc), ("ps", ds.nct)):
        core[kind] = np.repeat(nc[:nr], xyz.shape[1])
        dcore[kind] = np.repeat(radial_derivative(nc, ds)[:nr], xyz.shape[1])[None, :] * rhat
    phi_a: dict[str, FloatArray] = {k: np.array(v) for k, v in phi.items()}
    dphi_a: dict[str, FloatArray] = {k: np.array(v) for k, v in dphi.items()}
    overlap: FloatArray = (phi_a["ae"] * weight) @ phi_a["ae"].T - (phi_a["ps"] * weight) @ phi_a["ps"].T
    nabla: FloatArray = np.stack(
        [
            (phi_a["ae"] * weight) @ dphi_a["ae"][:, v].T - (phi_a["ps"] * weight) @ dphi_a["ps"][:, v].T
            for v in range(3)
        ],
        axis=-1,
    )
    return SphereSpecies(
        ds=ds,
        channels=channels,
        unit=xyz,
        r=r,
        weight=weight,
        phi=phi_a,
        dphi=dphi_a,
        core=core,
        dcore=dcore,
        tau_core={
            "ae": np.repeat(tauc[:nr], xyz.shape[1]),
            "ps": np.repeat(pseudo_core_kinetic(ds)[:nr], xyz.shape[1]),
        },
        overlap=overlap,
        nabla=nabla,
        beta_q=[_beta_table(p, ds, qmax) for p in ds.projectors],
    )


def projector_matrix(sp: SphereSpecies, q: FloatArray, pos: FloatArray, volume: float) -> ComplexArray:
    """(nchan, npw): <p~_p| applied to plane waves exp(i q.r) / sqrt(V) of an atom at pos; <p~|psi~> = B @ c"""
    qn: FloatArray = np.linalg.norm(q, axis=1)
    qhat: FloatArray = np.where(qn[:, None] > 1e-12, q / np.maximum(qn, 1e-12)[:, None], np.array([0.0, 0.0, 1.0]))
    phase: ComplexArray = np.exp(1j * (q @ pos))
    out: ComplexArray = np.zeros((sp.nchan, len(q)), dtype=complex)
    cache: dict[int, FloatArray] = {}
    c: int
    i: int
    ell: int
    m: int
    for c, (i, ell, m) in enumerate(sp.channels):
        if ell not in cache:
            cache[ell] = harmonics(ell, qhat.T)
        out[c] = 4.0 * pi / sqrt(volume) * (1j**ell) * cache[ell][m] * sp.beta_q[i](qn) * phase
    return out


def sphere_fields(sp: SphereSpecies, D: FloatArray, kind: str, core: bool) -> tuple[FloatArray, FloatArray, FloatArray]:
    """density, |grad n|^2 and kinetic-energy density on the sphere points from a (real, symmetric) density matrix D (nchan, nchan);
    with the core parts if core"""
    ph: FloatArray = sp.phi[kind]
    dph: FloatArray = sp.dphi[kind]
    a: FloatArray = D @ ph  # (nchan, npts)
    n: FloatArray = np.sum(ph * a, axis=0)
    grad: FloatArray = 2.0 * np.einsum("cvp,cp->vp", dph, a)
    tau: FloatArray = 0.5 * np.einsum("cvp,cvp->p", dph, np.einsum("cd,dvp->cvp", D, dph))
    if core:
        n = n + sp.core[kind]
        grad = grad + sp.dcore[kind]
        tau = tau + sp.tau_core[kind]
    return n, np.sum(grad**2, axis=0), tau


def sphere_density(sp: SphereSpecies, D: FloatArray, kind: str) -> FloatArray:
    """density on the sphere points from a density matrix (no core)"""
    out: FloatArray = np.sum(sp.phi[kind] * (D @ sp.phi[kind]), axis=0)
    return out
