"""Local variables and the lever weight fields of E_S on a point set.

u_V = k_F exp(-alpha) G(s)     (occupied states)
u_C = k_F G(s)                 (empty states)
G(s) = 1 + 3 s^2 / (1 + s^2 / 4),   k_F = (3 pi^2 n)^(1/3),   alpha = (tau - tau_W) / tau_unif (clipped to [0, 20])
"""

from __future__ import annotations

from math import pi
from typing import cast

import numpy as np

from ._types import FloatArray


def local_variables(n: FloatArray, sigma: FloatArray, tau: FloatArray) -> dict[str, FloatArray]:
    """n, sigma = |grad n|^2 and tau of the total density (atomic units) -> dict of the local variables"""
    n = np.clip(np.asarray(n, float), 1e-12, None)
    sigma = np.clip(np.asarray(sigma, float), 0.0, None)
    tau = np.clip(np.asarray(tau, float), 0.0, None)
    kf: FloatArray = (3 * pi**2 * n) ** (1 / 3)
    s2: FloatArray = sigma / (2 * kf * n) ** 2
    tw: FloatArray = sigma / (8 * n)
    tu: FloatArray = 0.3 * (3 * pi**2) ** (2 / 3) * n ** (5 / 3)
    alpha: FloatArray = np.clip((tau - tw) / tu, 0.0, 20.0)
    return dict(n=n, sigma=sigma, tau=tau, kf=kf, s2=s2, alpha=alpha)


def lever_fields(n: FloatArray, sigma: FloatArray, tau: FloatArray) -> tuple[FloatArray, FloatArray]:
    """(u_V, u_C) in bohr^-1 at every point"""
    v: dict[str, FloatArray] = local_variables(n, sigma, tau)
    gs: FloatArray = cast(FloatArray, 1 + 3 * v["s2"] / (1 + v["s2"] / 4))
    return v["kf"] * np.exp(-v["alpha"]) * gs, v["kf"] * gs
