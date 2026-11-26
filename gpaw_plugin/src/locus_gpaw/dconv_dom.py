"""Distributed fine-grid layout of the DCONV functional (dconv) for GPAW's domain decomposition.

GPAW hands the XC functional its local 3D block of the fine grid (gd.comm, with k-point / band replicas of the same
block on the other ranks).  DCONV works on its own layout over the whole world communicator, replicas included:
  * X layout (real space): rank r owns the x planes [xb[r], xb[r+1]) of the (N0, N1, N2) grid, a contiguous range of the
    C-ordered flat index;
  * Y layout (reciprocal space): rank r owns the ky rows [yb[r], yb[r+1]) for all kx, kz.
The blocks <-> X slabs redistribution and the X <-> Y transposes are single alltoallv calls; the distributed FFTs equal
the serial 3D FFTs up to the summation order."""

from __future__ import annotations

from typing import Any, Sequence, Union, cast

import numpy as np
import numpy.typing as npt

from ._types import AnyArray, ComplexArray, FloatArray, IntArray


def _a2a(comm: Any, parts: Sequence[AnyArray], dtype: type) -> list[AnyArray]:
    """alltoallv of a list of 1-D arrays (parts[q] to rank q) -> list over source ranks; counts exchanged with one sum"""
    R: int = comm.size
    sc: IntArray = np.array([p.size for p in parts], dtype=int)
    cnt: IntArray = np.zeros((R, R), dtype=int)
    cnt[comm.rank] = sc
    comm.sum(cnt)
    rc: IntArray = np.ascontiguousarray(cnt[:, comm.rank])
    sc2: IntArray
    rc2: IntArray
    if dtype == complex:
        sbuf: AnyArray
        rbuf: AnyArray
        sbuf = np.concatenate([np.ascontiguousarray(p).view(float).ravel() for p in parts]) if sc.sum() else np.zeros(1)
        rbuf = np.empty(max(2 * int(rc.sum()), 1))
        sc2, rc2 = 2 * sc, 2 * rc
    else:
        sbuf = np.concatenate([np.asarray(p, dtype).ravel() for p in parts]) if sc.sum() else np.zeros(1, dtype)
        rbuf = np.empty(max(int(rc.sum()), 1), dtype)
        sc2, rc2 = sc, rc
    comm.alltoallv(
        sbuf, sc2, np.r_[0, np.cumsum(sc2)[:-1]].astype(int), rbuf, rc2, np.r_[0, np.cumsum(rc2)[:-1]].astype(int)
    )
    off: IntArray = np.r_[0, np.cumsum(rc2)]
    out: list[AnyArray] = [rbuf[off[q] : off[q + 1]] for q in range(R)]
    return [o.view(complex) for o in out] if dtype == complex else out


class DomLayout:
    """the X / Y slab layout of the world communicator for an (N0, N1, N2) grid and the GPAW XC grid descriptor xcgd (its blocks)"""

    def __init__(self, world: Any, xcgd: Any, N: Sequence[int]) -> None:
        self.comm: Any = world
        self.R: int
        self.rank: int
        self.R, self.rank = world.size, world.rank
        self.N: tuple[int, ...] = tuple(int(x) for x in N)
        N0: int
        N1: int
        N2: int
        N0, N1, N2 = self.N
        R: int
        r: int
        R, r = self.R, self.rank
        if R > min(N0, N1):
            raise ValueError(
                f"DCONV distributed grid: {R} ranks need at least as many x and y planes (fine grid {N0} x {N1} x {N2})"
            )
        self.xb: list[int] = [N0 * q // R for q in range(R + 1)]
        self.yb: list[int] = [N1 * q // R for q in range(R + 1)]
        self.x0: int
        self.x1: int
        self.x0, self.x1 = self.xb[r], self.xb[r + 1]
        self.y0: int
        self.y1: int
        self.y0, self.y1 = self.yb[r], self.yb[r + 1]
        self.nx: int
        self.ny: int
        self.nx, self.ny = self.x1 - self.x0, self.y1 - self.y0
        self.Pl: int = self.nx * N1 * N2  # points of this rank's X slab
        self.p0: int
        self.p1: int
        self.p0, self.p1 = self.x0 * N1 * N2, self.x1 * N1 * N2  # its flat range
        self.PlG: int = N0 * self.ny * N2  # reciprocal-space points of its Y slab
        self.bounds: list[int]
        self.bounds = [self.xb[q] * N1 * N2 for q in range(R + 1)]  # owner ranges of the flat index (X layout)
        # GPAW blocks of the XC grid: gd.comm rank of every world rank, block extents, the first replica of every block
        info: IntArray = np.zeros((R, 1), int)
        info[r, 0] = xcgd.comm.rank
        world.sum(info)
        self.drank: IntArray = info[:, 0]
        self.blocks: dict[int, tuple[IntArray, IntArray]] = {}
        d: int
        for d in range(xcgd.comm.size):
            pos: Any = xcgd.get_processor_position_from_rank(d)
            self.blocks[d] = (
                np.array([xcgd.n_cp[c][pos[c]] for c in range(3)]),
                np.array([xcgd.n_cp[c][pos[c] + 1] for c in range(3)]),
            )
        assert all(int(b[0][c]) >= 0 for b in self.blocks.values() for c in range(3)), "periodic grids only"
        self.primary: dict[int, int] = {d: min(w for w in range(R) if self.drank[w] == d) for d in self.blocks}
        self.myblock: tuple[IntArray, IntArray] = self.blocks[xcgd.comm.rank]
        self.block_shape: tuple[int, ...] = tuple(int(x) for x in (self.myblock[1] - self.myblock[0]))
        self._gcache: dict[str, Any] = {}  # arrays, and the node dampings of dconv under "damp"

    # ---------------------------------------------------------------- GPAW blocks <-> X slabs
    def to_slab(self, a: FloatArray) -> FloatArray:
        """GPAW block arrays (k, *block) of this rank (only the first replica of each block is read) -> (k, Pl) X-slab arrays"""
        a = np.asarray(a)
        k: int = a.shape[0]
        N0: int
        N1: int
        N2: int
        N0, N1, N2 = self.N
        beg: IntArray
        end: IntArray
        beg, end = self.myblock
        parts: list[FloatArray] = []
        me_primary: bool = self.primary[self.drank[self.rank]] == self.rank
        q: int
        for q in range(self.R):
            lo: int
            hi: int
            lo, hi = max(beg[0], self.xb[q]), min(end[0], self.xb[q + 1])
            if me_primary and hi > lo:
                parts.append(np.ascontiguousarray(a[:, lo - beg[0] : hi - beg[0]]).ravel())
            else:
                parts.append(np.zeros(0))
        got: list[AnyArray] = _a2a(self.comm, parts, float)
        out: FloatArray = np.empty((k, self.nx, N1, N2))
        w: int
        for w in range(self.R):
            if self.primary[self.drank[w]] != w:
                continue
            b: IntArray
            e: IntArray
            b, e = self.blocks[self.drank[w]]
            lo, hi = max(b[0], self.x0), min(e[0], self.x1)
            if hi > lo:
                out[:, lo - self.x0 : hi - self.x0, b[1] : e[1], b[2] : e[2]] = got[w].reshape(
                    k, hi - lo, e[1] - b[1], e[2] - b[2]
                )
        return out.reshape(k, self.Pl)

    def to_blocks(self, s: FloatArray) -> FloatArray:
        """(k, Pl) X-slab arrays -> (k, *block) arrays of this rank's GPAW block (every replica receives its block)"""
        s = np.asarray(s)
        k: int = s.shape[0]
        N0: int
        N1: int
        N2: int
        N0, N1, N2 = self.N
        sl: FloatArray = s.reshape(k, self.nx, N1, N2)
        parts: list[FloatArray] = []
        w: int
        for w in range(self.R):
            b: IntArray
            e: IntArray
            b, e = self.blocks[self.drank[w]]
            lo: int
            hi: int
            lo, hi = max(b[0], self.x0), min(e[0], self.x1)
            if hi > lo:
                parts.append(np.ascontiguousarray(sl[:, lo - self.x0 : hi - self.x0, b[1] : e[1], b[2] : e[2]]).ravel())
            else:
                parts.append(np.zeros(0))
        got: list[AnyArray] = _a2a(self.comm, parts, float)
        beg: IntArray
        end: IntArray
        beg, end = self.myblock
        out: FloatArray = np.empty((k,) + self.block_shape)
        q: int
        for q in range(self.R):
            lo, hi = max(beg[0], self.xb[q]), min(end[0], self.xb[q + 1])
            if hi > lo:
                out[:, lo - beg[0] : hi - beg[0]] = got[q].reshape(k, hi - lo, end[1] - beg[1], end[2] - beg[2])
        return out

    # ---------------------------------------------------------------- distributed 3D FFT (complex)
    def _x2y(self, b: ComplexArray) -> ComplexArray:
        N0: int
        N1: int
        N2: int
        N0, N1, N2 = self.N
        parts: list[ComplexArray] = [
            np.ascontiguousarray(b[:, self.yb[q] : self.yb[q + 1], :]).ravel() for q in range(self.R)
        ]
        got: list[AnyArray] = _a2a(self.comm, parts, complex)
        c: ComplexArray = np.empty((N0, self.ny, N2), complex)
        q: int
        for q in range(self.R):
            c[self.xb[q] : self.xb[q + 1]] = got[q].reshape(self.xb[q + 1] - self.xb[q], self.ny, N2)
        return c

    def _y2x(self, c: ComplexArray) -> ComplexArray:
        N0: int
        N1: int
        N2: int
        N0, N1, N2 = self.N
        parts: list[ComplexArray] = [
            np.ascontiguousarray(c[self.xb[q] : self.xb[q + 1]]).ravel() for q in range(self.R)
        ]
        got: list[AnyArray] = _a2a(self.comm, parts, complex)
        b: ComplexArray = np.empty((self.nx, N1, N2), complex)
        q: int
        for q in range(self.R):
            b[:, self.yb[q] : self.yb[q + 1], :] = got[q].reshape(self.nx, self.yb[q + 1] - self.yb[q], N2)
        return b

    def fft(self, a: Union[FloatArray, ComplexArray]) -> ComplexArray:
        """X-slab (Pl,) real or complex -> Y-slab spectrum (PlG,) complex: fft over (y, z) on the planes, transpose, fft over x (the
        unnormalised forward transform, as numpy / scipy fftn)"""
        import scipy.fft as sfft

        from .dconv import _fft_workers

        N0: int
        N1: int
        N2: int
        N0, N1, N2 = self.N
        wk: int = _fft_workers()
        b: ComplexArray = sfft.fftn(np.asarray(a, complex).reshape(self.nx, N1, N2), axes=(1, 2), workers=wk)
        c: ComplexArray = self._x2y(b) if self.R > 1 else b
        return cast(ComplexArray, sfft.fft(c, axis=0, workers=wk, overwrite_x=True).reshape(-1))

    def ifft(self, A: ComplexArray) -> ComplexArray:
        """Y-slab spectrum (PlG,) -> X-slab (Pl,) complex: ifft over x, transpose, ifft over (y, z); normalised by
        1 / P once, between the passes (as scipy.fft.ifftn)"""
        import scipy.fft as sfft

        from .dconv import _fft_workers

        N0: int
        N1: int
        N2: int
        N0, N1, N2 = self.N
        wk: int = _fft_workers()
        c: ComplexArray
        c = sfft.ifft(np.asarray(A).reshape(N0, self.ny, N2), axis=0, norm="forward", workers=wk) / float(N0 * N1 * N2)
        b: ComplexArray = self._y2x(c) if self.R > 1 else c
        return cast(ComplexArray, sfft.ifftn(b, axes=(1, 2), norm="forward", workers=wk, overwrite_x=True).reshape(-1))

    # ---------------------------------------------------------------- reciprocal-space quantities of the Y slab
    def gidx(self) -> IntArray:
        """global flat indices (C order of the full grid) of this rank's Y-slab G points, in the Y-slab order"""
        if "gidx" not in self._gcache:
            N0: int
            N1: int
            N2: int
            N0, N1, N2 = self.N
            i0: IntArray = np.arange(N0)[:, None, None]
            i1: IntArray = np.arange(self.y0, self.y1)[None, :, None]
            i2: IntArray = np.arange(N2)[None, None, :]
            self._gcache["gidx"] = ((i0 * N1 + i1) * N2 + i2).reshape(-1)
        return cast(AnyArray, self._gcache["gidx"])

    def gvec(self, kk: Sequence[FloatArray], B: FloatArray) -> FloatArray:
        """G vectors (PlG, 3) of the Y slab (the expression of dconv's G array, element by element)"""
        N0: int
        N1: int
        N2: int
        N0, N1, N2 = self.N
        k0: FloatArray = kk[0][:, None, None, None]
        k1: FloatArray = kk[1][self.y0 : self.y1][None, :, None, None]
        k2: FloatArray = kk[2][None, None, :, None]
        return cast(FloatArray, (k0 * B[0] + k1 * B[1] + k2 * B[2]).reshape(-1, 3))

    def g2(self, kk: Sequence[FloatArray], B: FloatArray) -> FloatArray:
        if "g2" not in self._gcache:
            self._gcache["g2"] = (self.gvec(kk, B) ** 2).sum(-1)
        return cast(AnyArray, self._gcache["g2"])

    def bspline(self) -> FloatArray:
        """transfer function of the cubic B-spline interpolation at the grid points, prod_c (2 + cos(2 pi i_c / N_c)) / 3: dividing a
        spectrum by it gives the spline coefficients (the periodic spline prefilter in reciprocal space)"""
        if "bsp" not in self._gcache:
            N0: int
            N1: int
            N2: int
            N0, N1, N2 = self.N
            f: list[FloatArray] = [(2.0 + np.cos(2 * np.pi * np.arange(n) / n)) / 3.0 for n in self.N]
            self._gcache["bsp"] = (
                f[0][:, None, None] * f[1][self.y0 : self.y1][None, :, None] * f[2][None, None, :]
            ).reshape(-1)
        return cast(AnyArray, self._gcache["bsp"])

    def prefilter(self, A: ComplexArray) -> FloatArray:
        """spline coefficients (X slab) of the field with spectrum A"""
        return np.real(self.ifft(A / self.bspline()))

    # ---------------------------------------------------------------- reductions
    def allsum(self, x: npt.ArrayLike) -> FloatArray:
        """sum over the world communicator, bit-identical on every rank (reduce to 0, broadcast)"""
        x = np.ascontiguousarray(np.asarray(x, float))
        self.comm.sum(x, 0)
        self.comm.broadcast(x, 0)
        return x

    def allmax(self, v: float) -> float:
        x: FloatArray = np.array([float(v)])
        buf: FloatArray = np.zeros(self.R)
        buf[self.rank] = x[0]
        self.comm.sum(buf)
        return float(buf.max())
