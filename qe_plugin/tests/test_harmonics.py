"""Real spherical harmonics of the QE exporter: orthonormal on the sphere, gradients equal to finite differences."""

from __future__ import annotations

import numpy as np
from scipy.integrate import lebedev_rule

from locus_es_qe.harmonics import LMAX, POLYNOMIALS, _evaluate, harmonics, polynomial_gradients
from locus_es_qe.paw import _simpson


def test_orthonormal() -> None:
    xyz: np.ndarray
    w: np.ndarray
    xyz, w = lebedev_rule(17)
    y: np.ndarray = np.concatenate([harmonics(ell, xyz) for ell in range(LMAX + 1)])
    assert np.allclose((y * w) @ y.T, np.eye(len(y)), atol=1e-12)


def test_gradients() -> None:
    pts: np.ndarray = np.random.default_rng(0).normal(size=(3, 7))
    h: float = 1e-6
    ell: int
    for ell in range(LMAX + 1):
        g: np.ndarray = polynomial_gradients(ell, pts)
        v: int
        for v in range(3):
            e: np.ndarray = np.zeros((3, 1))
            e[v] = h
            fd: np.ndarray = np.array(
                [(_evaluate(p, pts + e) - _evaluate(p, pts - e)) / (2 * h) for p in POLYNOMIALS[ell]]
            )
            assert np.allclose(fd, g[:, v], atol=1e-6)


def test_simpson() -> None:
    """exact for x^2 on an odd number of points; an even number adds the trapezoid error h^3 / 6 of the last interval"""
    n: int
    for n in (5, 6, 101):
        x: np.ndarray = np.linspace(0.0, 1.0, n)
        h: float = float(x[1] - x[0])
        err: float = abs(float(np.sum(_simpson(n) * x**2)) * h - 1.0 / 3.0)
        assert err < (1e-12 if n % 2 == 1 else h**3 / 6.0 + 1e-12)
