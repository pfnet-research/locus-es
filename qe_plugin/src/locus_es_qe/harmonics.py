"""Real spherical harmonics up to l = 3 as homogeneous polynomials, with their Cartesian gradients.

Y_lm(r^) = S_lm(r^) with S_lm a homogeneous polynomial of degree l (real solid harmonic), orthonormal on the unit sphere.
For f(r) Y_lm(r^):   grad = f'(r) Y_lm r^ + f(r) / r [grad S_lm(r^) - l Y_lm(r^) r^]   (Euler's theorem for S_lm).
The order of m is -l, ..., l.  The sign convention is internal to this package; the densities and matrix elements built
from it do not depend on it.
"""

from __future__ import annotations

from math import pi, sqrt

import numpy as np

from ._types import FloatArray

Monomials = dict[tuple[int, int, int], float]

_C0: float = 0.5 / sqrt(pi)
_C1: float = sqrt(3.0 / (4.0 * pi))
_C2A: float = 0.5 * sqrt(15.0 / pi)
_C2B: float = 0.25 * sqrt(5.0 / pi)
_C2C: float = 0.25 * sqrt(15.0 / pi)
_C3A: float = 0.25 * sqrt(35.0 / (2.0 * pi))
_C3B: float = 0.5 * sqrt(105.0 / pi)
_C3C: float = 0.25 * sqrt(21.0 / (2.0 * pi))
_C3D: float = 0.25 * sqrt(7.0 / pi)
_C3E: float = 0.25 * sqrt(105.0 / pi)

# exponents (a, b, c) of x^a y^b z^c -> coefficient
POLYNOMIALS: list[list[Monomials]] = [
    [{(0, 0, 0): _C0}],
    [{(0, 1, 0): _C1}, {(0, 0, 1): _C1}, {(1, 0, 0): _C1}],
    [
        {(1, 1, 0): _C2A},
        {(0, 1, 1): _C2A},
        {(0, 0, 2): 2.0 * _C2B, (2, 0, 0): -_C2B, (0, 2, 0): -_C2B},
        {(1, 0, 1): _C2A},
        {(2, 0, 0): _C2C, (0, 2, 0): -_C2C},
    ],
    [
        {(2, 1, 0): 3.0 * _C3A, (0, 3, 0): -_C3A},
        {(1, 1, 1): _C3B},
        {(0, 1, 2): 4.0 * _C3C, (2, 1, 0): -_C3C, (0, 3, 0): -_C3C},
        {(0, 0, 3): 2.0 * _C3D, (2, 0, 1): -3.0 * _C3D, (0, 2, 1): -3.0 * _C3D},
        {(1, 0, 2): 4.0 * _C3C, (3, 0, 0): -_C3C, (1, 2, 0): -_C3C},
        {(2, 0, 1): _C3E, (0, 2, 1): -_C3E},
        {(3, 0, 0): _C3A, (1, 2, 0): -3.0 * _C3A},
    ],
]

LMAX: int = len(POLYNOMIALS) - 1


def _evaluate(poly: Monomials, xyz: FloatArray) -> FloatArray:
    """value of a polynomial at the points xyz (3, N)"""
    out: FloatArray = np.zeros(xyz.shape[1])
    exps: tuple[int, int, int]
    coef: float
    for exps, coef in poly.items():
        out = out + coef * xyz[0] ** exps[0] * xyz[1] ** exps[1] * xyz[2] ** exps[2]
    return out


def _derivative(poly: Monomials, axis: int) -> Monomials:
    out: Monomials = {}
    exps: tuple[int, int, int]
    coef: float
    for exps, coef in poly.items():
        if exps[axis] == 0:
            continue
        new: list[int] = list(exps)
        new[axis] -= 1
        key: tuple[int, int, int] = (new[0], new[1], new[2])
        out[key] = out.get(key, 0.0) + coef * exps[axis]
    return out


def harmonics(ell: int, xyz: FloatArray) -> FloatArray:
    """Y_(ell m) at unit vectors xyz (3, N) -> (2 ell + 1, N)"""
    if ell > LMAX:
        raise ValueError(f"angular momentum {ell} > {LMAX} is not supported")
    return np.array([_evaluate(p, xyz) for p in POLYNOMIALS[ell]])


def polynomial_gradients(ell: int, xyz: FloatArray) -> FloatArray:
    """grad S_lm at the points xyz (3, N) -> (2 ell + 1, 3, N)"""
    if ell > LMAX:
        raise ValueError(f"angular momentum {ell} > {LMAX} is not supported")
    return np.array([[_evaluate(_derivative(p, v), xyz) for v in range(3)] for p in POLYNOMIALS[ell]])
