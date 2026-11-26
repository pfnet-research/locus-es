"""E_S with a screening g(r) (interface.py) on synthetic data: the layer weights and the levels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from locus_es import interface as IF
from locus_es._types import ComplexArray, FloatArray, IntArray
from locus_es.params import BOHR, DEFAULT


def test_promolecular_grid_matches_real_space_sum() -> None:
    cell: FloatArray = np.array([[7.0, 0.0, 0.0], [2.5, 6.5, 0.0], [1.0, 1.5, 9.0]])  # bohr, not orthogonal
    spos: FloatArray = np.array([[0.1, 0.2, 0.3], [0.6, 0.55, 0.8]])
    N: tuple[int, int, int] = (18, 16, 24)
    sig: float = 1.2 / BOHR
    rho: FloatArray = IF.promolecular_grid(cell, spos, N, sig)
    rng: np.random.Generator = np.random.default_rng(0)
    s: FloatArray
    t: tuple[int, ...]
    for _ in range(5):
        i: list[int] = [int(rng.integers(n)) for n in N]
        r: FloatArray = (np.array(i) / N) @ cell
        ref: float = 0.0
        for s in spos:
            R: FloatArray = s @ cell
            for t in np.ndindex(7, 7, 7):
                d: FloatArray = r - R - (np.array(t) - 3) @ cell
                ref += np.exp(-(d @ d) / sig**2)
        assert rho[tuple(i)] == pytest.approx(ref, rel=1e-8)


def test_layer_weights_partition() -> None:
    cell: FloatArray = np.diag([8.0, 8.0, 16.0])
    spos: FloatArray = np.array([[0, 0, 0.1], [0.5, 0.5, 0.2], [0, 0, 0.6], [0.5, 0.5, 0.7]])
    meta: dict[str, Any] = dict(
        grid_N=[12, 12, 24], grid_beg=[0, 0, 0], grid_n=[12, 12, 24], cell_bohr=cell.tolist(), spos=spos.tolist()
    )
    ngrid: int = 12 * 12 * 24
    atom_p: IntArray = np.concatenate([np.full(ngrid, -1), np.repeat([0, 1, 2, 3], 5)])
    lay: IntArray = np.array([0, 0, 1, 1])
    h: FloatArray = IF.layer_weights(meta, atom_p, lay, 2)
    assert np.allclose(h.sum(0), 1.0)
    assert np.allclose(h[0, ngrid:], np.repeat([1, 1, 0, 0], 5))
    z: FloatArray = np.arange(24) / 24
    hz: FloatArray = h[0, :ngrid].reshape(12, 12, 24).mean((0, 1))
    assert hz[np.argmin(abs(z - 0.15))] > 0.95 and hz[np.argmin(abs(z - 0.65))] < 0.05


def test_layers_must_respect_symmetry() -> None:
    layers: list[dict[str, Any]] = [dict(name="A", atoms=[0, 2]), dict(name="B", atoms="rest")]
    with pytest.raises(ValueError):
        IF.layer_index(layers, 4, orbits=[0, 0, 2, 2])
    assert IF.layer_index(layers, 4, orbits=[0, 1, 0, 3]).tolist() == [0, 1, 0, 1]


def _window(
    eps_occ: FloatArray,
    eps_emp: FloatArray,
    h_occ: list[tuple[float, float]],
    h_emp: list[tuple[float, float]],
    uv: dict[str, ComplexArray],
    uc: dict[str, ComplexArray],
) -> dict[str, Any]:
    """one k-point; layer fields diagonal in the bands"""
    w: dict[str, Any] = dict(
        layers=["A", "B"],
        w_k=np.array([1.0]),
        spin_k=np.array([0]),
        ibz_of_u=np.array([0]),
        bands_occ=np.array([np.arange(len(eps_occ))]),
        bands_emp=np.array([np.arange(len(eps_emp)) + len(eps_occ)]),
        eps_occ=np.array([eps_occ]),
        eps_emp=np.array([eps_emp]),
    )
    blk: str
    hh: list[tuple[float, float]]
    u: dict[str, ComplexArray]
    M: int
    n: str
    X: str
    for blk, hh, u in (("occ", h_occ, uv), ("emp", h_emp, uc)):
        for M, n in enumerate("AB"):
            H: ComplexArray = np.diag([x[M] for x in hh]).astype(complex)
            w[f"M:h:{n}:{blk}"] = H[None]
            used: str = {"occ": "uv", "emp": "uc"}[blk]
            for X in ("uv", "uc"):
                w[f"M:{X}:{n}:{blk}"] = (u[n] if X == used else np.zeros_like(H))[None]
    return w


def test_interface_levels() -> None:
    HA: float = 27.211386024367243
    P: dict[str, Any] = dict(DEFAULT)
    eps_occ: FloatArray
    eps_occ = np.array([-0.30, -0.25, -0.20]) / HA  # the top two: edges of layer B (-0.25) and layer A (-0.20)
    eps_emp: FloatArray = np.array([0.05, 0.08]) / HA  # layer A, layer B
    h_occ: list[tuple[float, float]] = [(0.5, 0.5), (0.0, 1.0), (1.0, 0.0)]
    h_emp: list[tuple[float, float]] = [(1.0, 0.0), (0.0, 1.0)]
    uv: dict[str, ComplexArray] = {
        "A": np.diag([0.3, 0.0, 0.6]).astype(complex),
        "B": np.diag([0.3, 0.5, 0.0]).astype(complex),
    }
    uc: dict[str, ComplexArray] = {"A": np.diag([0.4, 0.0]).astype(complex), "B": np.diag([0.0, 0.2]).astype(complex)}
    layers: list[dict[str, Any]] = [
        dict(name="A", g=0.1, bulk={"vbm": -0.5, "cbm": 0.1}),
        dict(name="B", g=0.2, bulk={"vbm": -0.3, "cbm": 0.05}),
    ]
    r: dict[str, Any] = IF.interface(_window(eps_occ, eps_emp, h_occ, h_emp, uv, uc), layers, P, g_one=0.15)
    s: float
    kV: float
    kC: float
    s, kV, kC = P["s"], P["kV"], P["kC"]
    e: dict[str, Any] = r["edges"]["vbm"]["layers"]
    assert e["A"]["shift_ev"]["local"] == pytest.approx(-s * kV * 0.1 * 0.6)
    assert e["B"]["shift_ev"]["local"] == pytest.approx(-s * kV * 0.2 * 0.5)
    assert e["A"]["shift_ev"]["one"] == pytest.approx(-s * kV * 0.15 * 0.6)
    # no coupling: diagonalization = first order
    assert e["A"]["shift_ev"]["diag"] == pytest.approx(e["A"]["shift_ev"]["local"])
    c: dict[str, Any] = r["edges"]["cbm"]["layers"]
    assert c["B"]["shift_ev"]["local"] == pytest.approx(s * kC * 0.2 * 0.2)
    o: dict[str, Any] = r["offsets"]["vbm"][0]
    assert o["offset_ev"]["ks"] == pytest.approx((-0.25 + 0.20))
    assert o["vs_bulk_ev"]["local"] == pytest.approx(
        e["B"]["shift_ev"]["local"] - e["A"]["shift_ev"]["local"] - (-0.3 + 0.5)
    )
    assert r["edges"]["vbm"]["defined"] and r["edges"]["cbm"]["defined"]
    # equal g everywhere: g(r) = one g
    layers2: list[dict[str, Any]] = [dict(L, g=0.15) for L in layers]
    r2: dict[str, Any] = IF.interface(_window(eps_occ, eps_emp, h_occ, h_emp, uv, uc), layers2, P, g_one=0.15)
    k: dict[str, Any]
    for k in r2["kpts"]:
        assert np.allclose(k["occ"]["local"], k["occ"]["one"]) and np.allclose(k["emp"]["local"], k["emp"]["one"])
