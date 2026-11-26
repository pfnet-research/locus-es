"""LOCUS in GPAW: r2SCAN (libxc) + rVV10 + the shell term, whose gate reads the density-convolved descriptor
x_c = sum_a p_a(n) (G_a * n) / n (c = 4: Gaussians of width ~ c / k_F, interpolated between the fixed node widths
DENSITY_NODES), self-consistent in the PAW method.  GPL-3.0-or-later.

The smooth field S_w = G_w * [pseudo density + one-centre differences n^a - nt^a incl. the core] lives on the fine grid
and is read at the grid, PS and AE points; the one-centre (PAW) part feeds dE/dx back to the grid through sphere -> grid
deposits and the tails adjoint.  Spin-polarized: the shell term reads the total-density descriptor only.  Parallel:
k-points (_KPar, parallel={'domain': 1, 'band': 1}) or distributed fine grids (dconv_dom, parallel={'domain': N}).

    from locus_gpaw import LOCUS
    xc = LOCUS()
    calc = GPAW(mode=PW(600), xc=xc, ...)
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence, TypeVar, Union, cast

import numpy as np
from gpaw.sphere.lebedev import R_nv
from gpaw.xc.gga import Y_nL, weight_n
from gpaw.xc.mgga import MGGA
from gpaw.xc.pawcorrection import rnablaY_nLv
from scipy.interpolate import CubicSpline
from scipy.ndimage import map_coordinates
from scipy.special import ive, spherical_jn

from ._types import BoolArray, ComplexArray, FloatArray, IntArray

if TYPE_CHECKING:
    from types import ModuleType

    from _hashlib import HASH
    from numpy.lib.npyio import NpzFile
    from scipy.interpolate import PPoly

    from .dconv_dom import DomLayout
    from .kernel import LocusModel
    from .rvv10 import RVV10

    ArrT = TypeVar("ArrT", FloatArray, ComplexArray, IntArray)  # the arrays the MPI helpers of _KPar pass through
    BaseEval = Callable[[FloatArray, FloatArray, FloatArray], tuple[FloatArray, ...]]  # DConvKernel.base_eval(_spin)
    KmatKey = tuple[str, float, int, int, int, float]  # (symbol, w, l, len(ru), len(r_g), r_g[-1])
    TabKey = tuple[Union[str, int], ...]  # species-table keys (symbol, len(r_g)) and ("kin", symbol, len(r_g))
    Dirs = Optional[tuple[int, int]]  # a chunk (n0, n1) of the Lebedev directions, None: all
    # (atom, routes, directions): a work item of the one-centre pass
    PawItem = tuple[int, tuple[bool, ...], Optional[tuple[int, int]]]
    PawCache = dict[int, tuple[FloatArray, float, FloatArray]]  # {a: (D_sp, E_a, dE/dD)}

PI: float = math.pi
RU_STEP: float = 0.01
EXACT_TAILS_MIN_W: float = 0.25  # bohr: narrowest node with exact (non-pseudized) tails on the fine grid
TAILS_NEGL: float = (
    1e-10  # one-centre tails with sum_ql int |PU_ql| r^2 dr below this (electrons per unit D) are rounding noise
)


def _fft_workers() -> int:
    """threads of the grid FFTs: OMP_NUM_THREADS, capped by the cgroup CPU quota"""
    try:
        n: int = int(os.environ.get("OMP_NUM_THREADS", "1"))
    except ValueError:
        n = 1
    return max(1, min(n, _cpu_quota()))


def _cpu_quota() -> int:
    """CPUs of the cgroup quota (cpu.max), else os.cpu_count()"""
    try:
        q: str
        p: str
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if q != "max":
            return max(1, int(int(q) // int(p)))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


def _limit_threads(nranks: int) -> None:
    """torch / BLAS / OpenMP threads <= cgroup CPU quota / ranks; settings are only lowered, never raised"""
    n: int = max(1, _cpu_quota() // max(1, nranks))
    try:
        try:
            import torch
        except ImportError:
            return
        if torch.get_num_threads() > n:
            torch.set_num_threads(n)
    except Exception:
        pass
    try:
        from threadpoolctl import threadpool_info, threadpool_limits

        if any(int(i.get("num_threads", 0)) > n for i in threadpool_info()):
            threadpool_limits(limits=n)
    except Exception:
        pass


def _disk_cache(
    tag: str, parts: Iterable[object], compute: Callable[[], dict[str, FloatArray]]
) -> dict[str, FloatArray]:
    """compute() -> dict of arrays, cached on disk: directory $DCONV_CACHE_DIR (default ~/.cache/gpaw_dconv; '0'
    disables), file name = tag + sha1 of the key parts; atomic write (tmp + rename), safe for concurrent processes / MPI
    ranks"""
    import hashlib

    d: str = os.environ.get("DCONV_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "gpaw_dconv"))
    if d == "0":
        return compute()
    h: HASH = hashlib.sha1(tag.encode())
    p: object
    for p in parts:
        h.update(np.ascontiguousarray(p).tobytes() if isinstance(p, np.ndarray) else repr(p).encode())
    f: str = os.path.join(d, f"{tag}_{h.hexdigest()[:24]}.npz")
    try:
        z: NpzFile
        with np.load(f) as z:
            return {k: z[k] for k in z.files}
    except Exception:
        pass
    out: dict[str, FloatArray] = compute()
    try:
        os.makedirs(d, exist_ok=True)
        tmp: str = f"{f[:-4]}.{os.getpid()}.tmp.npz"
        np.savez(tmp, **out)
        os.replace(tmp, f)
    except OSError:
        pass
    return out


def _fftn(a: FloatArray | ComplexArray) -> ComplexArray:
    """complex FFT of the grid convolutions (scipy.fft, threads from _fft_workers)"""
    import scipy.fft as sfft

    return cast(ComplexArray, sfft.fftn(a, workers=_fft_workers()))


def _ifftn(a: ComplexArray) -> ComplexArray:
    import scipy.fft as sfft

    return cast(ComplexArray, sfft.ifftn(a, workers=_fft_workers()))


class _KPar:
    """k-point parallel work split of the DCONV functional.  Every rank holds the whole fine-grid input
    (parallel={'domain': 1, 'band': 1}); contiguous index ranges (G vectors, grid points, (spin, node) items, rVV10
    channels, G slabs) are computed by one rank each.  The outputs GPAW reads are all-gathered; sums are reduced to rank
    0 and broadcast (reduce_bcast)."""

    comm: Any  # the MPI communicator (GPAW's world)
    size: int
    rank: int

    def __init__(self, comm: Any) -> None:
        self.comm, self.size, self.rank = comm, comm.size, comm.rank

    def rng(self, n: int, rank: int | None = None) -> tuple[int, int]:
        r: int = self.rank if rank is None else rank
        return n * r // self.size, n * (r + 1) // self.size

    def gather(self, local: ArrT, n: int) -> ArrT:
        """rows rng(n) of an (n, ...) array on each rank -> the full (n, ...) array on every rank"""
        local = np.ascontiguousarray(local)
        b: int = -(-n // self.size)
        buf: ArrT = np.zeros((b,) + local.shape[1:], local.dtype)
        buf[: len(local)] = local
        out: ArrT = np.empty((self.size,) + buf.shape, local.dtype)
        self.comm.all_gather(buf, out)
        return np.concatenate([out[r, : self.rng(n, r)[1] - self.rng(n, r)[0]] for r in range(self.size)])

    def fill_last(self, arr: FloatArray) -> None:
        """arr (..., P), C-contiguous, holds valid values in this rank's columns rng(P): fill the other ranks' columns in place"""
        assert arr.flags.c_contiguous
        P: int = arr.shape[-1]
        flat: FloatArray = arr.reshape(-1, P)
        p0: int
        p1: int
        p0, p1 = self.rng(P)
        flat[:] = self.gather(np.ascontiguousarray(flat[:, p0:p1].T), P).T

    def transpose(self, local: FloatArray, n: int, P: int) -> FloatArray:
        """rows rng(n) of an (n, P) array on each rank -> all n rows of this rank's columns rng(P) (one alltoallv)"""
        local = np.ascontiguousarray(local)
        r0: int
        r1: int
        r0, r1 = self.rng(n)
        p0: int
        p1: int
        p0, p1 = self.rng(P)
        cols: list[tuple[int, int]] = [self.rng(P, r) for r in range(self.size)]
        sbuf: FloatArray = np.concatenate([local[:, c0:c1].ravel() for c0, c1 in cols])
        scounts: IntArray = np.array([(r1 - r0) * (c1 - c0) for c0, c1 in cols], dtype=int)
        rcounts: IntArray = np.array(
            [(b - a) * (p1 - p0) for a, b in (self.rng(n, r) for r in range(self.size))], dtype=int
        )
        rbuf: FloatArray = np.empty(int(rcounts.sum()), local.dtype)
        self.comm.alltoallv(
            sbuf,
            scounts,
            np.r_[0, np.cumsum(scounts)[:-1]].astype(int),
            rbuf,
            rcounts,
            np.r_[0, np.cumsum(rcounts)[:-1]].astype(int),
        )
        return rbuf.reshape(n, p1 - p0)

    def transpose_back(self, slab: FloatArray, n: int, P: int) -> FloatArray:
        """inverse of transpose: all n rows of this rank's columns rng(P) -> rows rng(n) of the (n, P) array"""
        slab = np.ascontiguousarray(slab)
        r0: int
        r1: int
        r0, r1 = self.rng(n)
        p0: int
        p1: int
        p0, p1 = self.rng(P)
        rows: list[tuple[int, int]] = [self.rng(n, r) for r in range(self.size)]
        cols: list[tuple[int, int]] = [self.rng(P, r) for r in range(self.size)]
        sbuf: FloatArray = np.ascontiguousarray(
            slab.reshape(n, p1 - p0)
        )  # rows in rank order: rank r gets rows rng(n, r) of this rank's columns
        scounts: IntArray = np.array([(b - a) * (p1 - p0) for a, b in rows], dtype=int)
        rcounts: IntArray = np.array([(r1 - r0) * (c1 - c0) for c0, c1 in cols], dtype=int)
        rbuf: FloatArray = np.empty(int(rcounts.sum()), slab.dtype)
        self.comm.alltoallv(
            sbuf.reshape(-1),
            scounts,
            np.r_[0, np.cumsum(scounts)[:-1]].astype(int),
            rbuf,
            rcounts,
            np.r_[0, np.cumsum(rcounts)[:-1]].astype(int),
        )
        out: FloatArray = np.empty((r1 - r0, P), slab.dtype)
        off: int = 0
        c0: int
        c1: int
        for c0, c1 in cols:
            k: int = (r1 - r0) * (c1 - c0)
            out[:, c0:c1] = rbuf[off : off + k].reshape(r1 - r0, c1 - c0)
            off += k
        return out

    def sum(self, arr: FloatArray) -> None:
        self.comm.sum(arr)

    def reduce_bcast(self, arr: FloatArray) -> FloatArray:
        """sum over the ranks, bit-identical on every rank (reduce to rank 0, then broadcast; an allreduce may differ in
        the last bits between ranks).  arr: C-contiguous, in place"""
        self.comm.sum(arr, 0)
        self.comm.broadcast(arr, 0)
        return arr

    def alltoallv_list(self, parts: Sequence[ArrT]) -> list[ArrT]:
        """parts[q]: 1-D array for rank q (same dtype on all ranks) -> list over source ranks r of the arrays rank r sent here"""
        dt: np.dtype[Any] = parts[0].dtype if len(parts) else np.dtype(np.int64)
        sc: IntArray = np.array([len(x) for x in parts], dtype=int)
        rc: IntArray = np.zeros(self.size, dtype=int)
        cnt: IntArray = np.zeros((self.size, self.size), dtype=int)
        cnt[self.rank] = sc
        self.comm.sum(cnt)
        rc[:] = cnt[:, self.rank]
        sbuf: ArrT = np.concatenate([np.asarray(x, dt).ravel() for x in parts]) if int(sc.sum()) else np.zeros(1, dt)
        rbuf: ArrT = np.empty(max(int(rc.sum()), 1), dt)
        self.comm.alltoallv(
            sbuf, sc, np.r_[0, np.cumsum(sc)[:-1]].astype(int), rbuf, rc, np.r_[0, np.cumsum(rc)[:-1]].astype(int)
        )
        off: IntArray = np.r_[0, np.cumsum(rc)]
        return [rbuf[off[r] : off[r + 1]] for r in range(self.size)]

    def bcast(self, arr: ArrT) -> ArrT:
        """rank 0's values of the C-contiguous array arr on every rank (in place)"""
        assert arr.flags.c_contiguous
        self.comm.broadcast(arr, 0)
        return arr


def _einsum(sub: str, *ops: FloatArray) -> FloatArray:
    """np.einsum with path optimisation for three or more operands"""
    return cast(FloatArray, np.einsum(sub, *ops, optimize=len(ops) >= 3))


_KMAT_CACHE: dict[tuple[str, float, int, int, int, float, float, float], FloatArray] = {}


def _kernel_modules() -> tuple[ModuleType, ModuleType]:
    """the LOCUS kernel module (locus_gpaw.kernel) in the roles of the model core and the SCF terms"""
    from . import kernel

    return kernel, kernel


DENSITY_NODES: tuple[float, ...] = (0.3, 0.4243, 0.6, 0.8485, 1.2, 1.6971, 2.4, 3.3941, 4.8, 6.0)


def _kernel_matrix(r_out: FloatArray, r_in: FloatArray, dr_in: FloatArray, ell: int, s: float) -> FloatArray:
    """(g_l * f)(r_out) = K @ f(r_in): 4 pi (2 pi s)^-3/2 e^{-(r^2 + r'^2)/2s} i_l(r r'/s) r'^2 dr'"""
    ro: FloatArray
    ri: FloatArray
    ro, ri = r_out[:, None], r_in[None, :]
    z: FloatArray = ro * ri / s
    il: FloatArray = np.where(
        z > 1e-12,
        np.sqrt(PI / (2 * np.maximum(z, 1e-12))) * ive(ell + 0.5, np.maximum(z, 1e-12)),
        1.0 if ell == 0 else 0.0,
    )
    return cast(
        FloatArray, 4 * PI * (2 * PI * s) ** -1.5 * np.exp(-((ro - ri) ** 2) / (2 * s)) * il * ri**2 * dr_in[None, :]
    )


KCHUNK: int = int(
    os.environ.get("DCONV_KCHUNK", "65536")
)  # points per pointwise chunk (grid kernel, feedback weights; rVV10 KCHUNK / 8): bounds the temporaries
# marker of a lagged deposit already added to the feedback (not None: the next update is not a first one)
_DEP_USED: FloatArray = np.zeros(0)
_PREF: str = "convolved fields"


def _local_ranks(size: int) -> int:
    """MPI ranks on this node (OpenMPI / MPICH / Slurm environment; the whole communicator if none is set): the read-out
    operator budget DCONV_READOUT_MB is per node"""
    k: str
    for k in ("OMPI_COMM_WORLD_LOCAL_SIZE", "MPI_LOCALNRANKS", "MPICH_LOCALNRANKS", "SLURM_NTASKS_PER_NODE"):
        try:
            return max(1, int(os.environ[k]))
        except (KeyError, ValueError):
            pass
    return max(1, int(size))


def _gridcache_ok(gc: dict[str, Any], nbytes: int) -> bool:
    """True (and the bytes booked) if a rebuildable per-cell grid table of nbytes fits into DCONV_GRIDCACHE_MB
    (default 128 per process); tables beyond it are rebuilt on every use"""
    used: int = gc.setdefault("cache_used", 0)
    if used + nbytes > float(os.environ.get("DCONV_GRIDCACHE_MB", "128")) * 2**20:
        return False
    gc["cache_used"] = used + nbytes
    return True


class _LazyDamp:
    """the node dampings exp(-G^2 w^2 / 2) on the flat fine grid: kept within the grid-table budget (_gridcache_ok),
    evaluated on access beyond it"""

    G2: FloatArray
    nodes: tuple[float, ...]
    gc: dict[str, Any]
    cache: dict[int, FloatArray]

    def __init__(self, G2: FloatArray, nodes: Iterable[float], gc: dict[str, Any]) -> None:
        self.G2, self.nodes, self.gc, self.cache = G2, tuple(nodes), gc, {}

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, i: int) -> FloatArray:
        if i in self.cache:
            return self.cache[i]
        w: float = self.nodes[i]
        d: FloatArray = np.exp(-0.5 * self.G2 * w * w)
        if _gridcache_ok(self.gc, d.nbytes):
            self.cache[i] = d
        return d

    def __iter__(self) -> Iterator[FloatArray]:
        return (self[i] for i in range(len(self.nodes)))


def _ylm(vec: FloatArray, Lmax: int) -> FloatArray:
    """real spherical harmonics Y_L (L < Lmax) of the directions of vec (N, 3), GPAW ordering"""
    from gpaw.spherical_harmonics import Y

    n: FloatArray = np.linalg.norm(vec, axis=1)
    u: FloatArray = vec / np.where(n > 0, n, 1.0)[:, None]
    out: FloatArray = np.empty((len(vec), Lmax))
    L: int
    for L in range(Lmax):
        out[:, L] = Y(L, u[:, 0], u[:, 1], u[:, 2])
    out[n == 0, 1:] = 0.0
    return out


class DConvKernel:
    """pointwise LOCUS kernel on the grid: calculate(e, n, dedn(+=), sigma, dedsigma, tau, dedtau, x, dedx[, x_s, dedx_s])"""

    def __init__(self, model: LocusModel, base: str, cs: Sequence[float], nl_c: Sequence[float] | None = None) -> None:
        self.name: str = "DCONV"
        self.type: str = "MGGA"
        self.model: LocusModel = model
        self.base: str = base
        self.cs: list[float] = [float(c) for c in cs]  # every computed c (ascending)
        self.sel: list[int] = [self.cs.index(float(c)) for c in (nl_c if nl_c is not None else cs)]
        self._base_eval: BaseEval | None = None

    def base_eval(self) -> BaseEval:
        """base functional through GPAW's own LibXC kernels (eps per particle and d(n eps)/d(n, sigma, tau))"""
        if self._base_eval is None:
            from gpaw.xc.libxc import LibXC

            BASES: dict[str, tuple[str, str]] = _kernel_modules()[0].BASES

            kx: Any
            kc: Any
            kx, kc = LibXC(BASES[self.base][0]), LibXC(BASES[self.base][1])

            def ev(n: FloatArray, sigma: FloatArray, tau: FloatArray) -> tuple[FloatArray, ...]:
                out: list[FloatArray] = []
                k: Any
                for k in (kx, kc):
                    e: FloatArray = np.zeros_like(n)
                    dedn: FloatArray = np.zeros((1,) + n.shape)
                    dedsig: FloatArray = np.zeros((1,) + n.shape)
                    dedtau: FloatArray = np.zeros((1,) + n.shape)
                    k.calculate(e, n[None], dedn, sigma[None], dedsig, tau[None], dedtau)
                    out += [e / n, dedn[0], dedsig[0], dedtau[0]]
                return tuple(out)

            self._base_eval = ev
        return self._base_eval

    def base_eval_spin(self) -> BaseEval:
        """spin-polarized base evaluator through GPAW's LibXC kernels: exchange unpolarized on the channel densities 2 n_s, correlation
        polarized"""
        if getattr(self, "_base_eval_spin", None) is None:
            from gpaw.xc.libxc import LibXC

            BASES: dict[str, tuple[str, str]] = _kernel_modules()[0].BASES
            kx: Any
            kc: Any
            kx, kc = LibXC(BASES[self.base][0]), LibXC(BASES[self.base][1])

            def ev(n_s: FloatArray, sig_s: FloatArray, tau_s: FloatArray) -> tuple[FloatArray, ...]:
                P: int = n_s.shape[1]
                ex2: FloatArray
                vrx: FloatArray
                vsx: FloatArray
                vtx: FloatArray
                ex2, vrx, vsx, vtx = np.empty((2, P)), np.empty((2, P)), np.empty((2, P)), np.empty((2, P))
                s: int
                for s in range(2):
                    m: FloatArray
                    sm: FloatArray
                    tm: FloatArray
                    m, sm, tm = 2.0 * n_s[s], 4.0 * sig_s[2 * s], 2.0 * tau_s[s]
                    e: FloatArray = np.zeros(P)
                    dn: FloatArray = np.zeros((1, P))
                    ds: FloatArray = np.zeros((1, P))
                    dt: FloatArray = np.zeros((1, P))
                    kx.calculate(e, m[None], dn, sm[None], ds, tm[None], dt)
                    ex2[s], vrx[s], vsx[s], vtx[s] = e / m, dn[0], ds[0], dt[0]
                e = np.zeros(P)
                dn = np.zeros((2, P))
                ds = np.zeros((3, P))
                dt = np.zeros((2, P))
                kc.calculate(
                    e, np.ascontiguousarray(n_s), dn, np.ascontiguousarray(sig_s), ds, np.ascontiguousarray(tau_s), dt
                )
                return ex2, vrx, vsx, vtx, e / n_s.sum(0), dn, ds, dt

            self._base_eval_spin: BaseEval = ev
        return self._base_eval_spin

    lowdens: bool = (
        True  # blend into PBE below n_hi (kernel.LOWDENS); bulk solids are untouched (the switch returns early)
    )

    def calculate(
        self,
        e_g: FloatArray,
        n_sg: FloatArray,
        dedn_sg: FloatArray,
        sigma_xg: FloatArray,
        dedsigma_xg: FloatArray,
        tau_sg: FloatArray,
        dedtau_sg: FloatArray,
        x_cg: FloatArray,
        dedx_cg: FloatArray,
        xs_csg: FloatArray | None = None,
        dedxs_csg: FloatArray | None = None,
    ) -> None:
        """the kernel, blended into PBE at low density: e = w e_LOCUS + (1 - w) e_PBE, w(n) a smoothstep in log10 n between n_lo and n_hi
        of the total density; dE/dn picks up w' (e_LOCUS - e_PBE), dE/dsigma is blended, dE/dtau and dE/dx are scaled by w
        """
        from . import kernel as _k

        ntot: FloatArray = n_sg.sum(0) if len(n_sg) == 2 else n_sg[0]
        if not self.lowdens or float(np.min(ntot)) >= _k.LOWDENS["n_hi"]:
            return self._calculate_raw(
                e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg, tau_sg, dedtau_sg, x_cg, dedx_cg, xs_csg, dedxs_csg
            )
        w: FloatArray
        dw: FloatArray
        w, dw = _k.lowdens_switch(ntot)
        vD: FloatArray = np.zeros_like(dedn_sg)
        self._calculate_raw(e_g, n_sg, vD, sigma_xg, dedsigma_xg, tau_sg, dedtau_sg, x_cg, dedx_cg, xs_csg, dedxs_csg)
        eD: FloatArray
        sD: FloatArray
        eD, sD = e_g.copy(), dedsigma_xg.copy()
        from gpaw.xc.libxc import LibXC

        if getattr(self, "_pbe", None) is None:
            self._pbe: Any = LibXC("GGA_X_PBE+GGA_C_PBE")
        eP: FloatArray = np.zeros_like(e_g)
        vP: FloatArray = np.zeros_like(dedn_sg)
        sP: FloatArray = np.zeros_like(dedsigma_xg)
        self._pbe.calculate(eP, np.ascontiguousarray(n_sg).copy(), vP, np.ascontiguousarray(sigma_xg).copy(), sP)
        e_g[:] = w * eD + (1.0 - w) * eP
        s: int
        for s in range(len(n_sg)):
            dedn_sg[s] += w * vD[s] + (1.0 - w) * vP[s] + dw * (eD - eP)
        dedsigma_xg[:] = w * sD + (1.0 - w) * sP
        dedtau_sg *= w
        dedx_cg *= w
        if dedxs_csg is not None and dedxs_csg.size:
            dedxs_csg *= w

    def _calculate_raw(
        self,
        e_g: FloatArray,
        n_sg: FloatArray,
        dedn_sg: FloatArray,
        sigma_xg: FloatArray,
        dedsigma_xg: FloatArray,
        tau_sg: FloatArray,
        dedtau_sg: FloatArray,
        x_cg: FloatArray,
        dedx_cg: FloatArray,
        xs_csg: FloatArray | None = None,
        dedxs_csg: FloatArray | None = None,
    ) -> None:
        """x_cg: total-density descriptors (K, ...); spin-polarized: xs_csg (K, 2, ...) the (placeholder) channel descriptors and dedxs_csg
        their gradients; dedx_cg then holds d/dx_c of the total-density descriptors"""
        st: ModuleType = _kernel_modules()[1]
        shp: tuple[int, ...] = e_g.shape
        K: int = len(self.cs)
        if len(n_sg) == 1:
            t: dict[str, FloatArray] = st.constrained_terms(
                self.model,
                self.base,
                n_sg[0].ravel(),
                sigma_xg[0].ravel(),
                tau_sg[0].ravel(),
                x_cg.reshape(K, -1),
                base_eval=self.base_eval(),
            )
            e_g[:] = t["e"].reshape(shp)
            dedn_sg[0] += t["dedn"].reshape(shp)
            dedsigma_xg[0] = t["dedsigma"].reshape(shp)
            dedtau_sg[0] = t["dedtau"].reshape(shp)
            dedx_cg[:] = t["dedx"].reshape(x_cg.shape)
            return
        assert xs_csg is not None and dedxs_csg is not None, "spin-polarized DConvKernel needs the channel descriptors"
        n_s: FloatArray = np.stack([n_sg[0].ravel(), n_sg[1].ravel()])
        sig_s: FloatArray = np.stack([sigma_xg[0].ravel(), sigma_xg[1].ravel(), sigma_xg[2].ravel()])
        tau_s: FloatArray = np.stack([tau_sg[0].ravel(), tau_sg[1].ravel()])
        t = st.constrained_terms_spin(
            self.model,
            self.base,
            n_s,
            sig_s,
            tau_s,
            xs_csg.reshape(K, 2, -1),
            x_cg.reshape(K, -1),
            base_eval=self.base_eval_spin(),
        )
        e_g[:] = t["e"].reshape(shp)
        s: int
        for s in range(2):
            dedn_sg[s] += t["dedn"][s].reshape(shp)
            dedtau_sg[s] = t["dedtau"][s].reshape(shp)
        k: int
        for k in range(3):
            dedsigma_xg[k] = t["dedsigma"][k].reshape(shp)
        dedx_cg[:] = t["dedx_tot"].reshape(x_cg.shape)
        dedxs_csg[:] = t["dedx_s"].reshape(xs_csg.shape)


class DensityConvMGGA(MGGA):  # type: ignore[misc]  # GPAW's MGGA is untyped (Any)
    _native_kpar: bool = True  # k-point parallel work split built in
    _native_domain: bool = True  # distributed fine grids (parallel={"domain": N}, augment_grids) supported (dconv_dom)

    # instance attributes (set in the methods below; MGGA provides wfs, gd, kernel, tauct_G, fixed_ke, ...)
    kernel: DConvKernel
    name: str
    vv10: dict[str, float] | None
    _rvv10: RVV10 | None
    cs: list[float]
    nodes: tuple[float, ...]
    _spline: CubicSpline
    _dspline: PPoly
    _kmat: dict[KmatKey, FloatArray]
    _tabs: dict[TabKey, dict[str, Any]]
    _jtabs: dict[TabKey, tuple[FloatArray, FloatArray]]
    _gcell: dict[str, Any] | None  # cell part of the geometry (_geometry)
    _geom: dict[str, Any] | None  # geometry: cell part + atom positions + per-geometry caches ("pts")
    _deposit_wg: FloatArray | None
    _deposit_new: FloatArray | None
    _vfb_wg: FloatArray | None
    _vfb_swg: FloatArray | None
    _vfb_pref: list[FloatArray] | None
    _vfb_pref_dom: FloatArray | None
    _S_wg: FloatArray | None
    _S_swg: FloatArray | None
    _S_prefT: FloatArray | None
    _S_pref: list[list[FloatArray]] | None
    timings: dict[str, Any]
    dEdD_asp: dict[int, FloatArray]
    density: Any
    finegd: Any
    spos_ac: FloatArray | None
    _dom_on: bool
    _spinsum: bool
    _xoff: bool
    _t_tick: float
    _taut_gradv_init: bool
    _D_asp: dict[int, FloatArray] | None
    _tails: dict[int, dict[str, Any]]
    _slab: tuple[int, int, bool]
    _pref_own: FloatArray | None
    _pref_rng: tuple[int, int]
    _paw_cache: PawCache | None
    _x_stats: dict[str, Any]
    dedtaut_sG: FloatArray
    ekin: float
    _domL: DomLayout
    _domL_key: tuple[tuple[int, ...], int, int]
    _dom_par: _KPar
    _pref_dom: FloatArray | None
    _tadj: dict[int, FloatArray]
    _par: _KPar
    _owner: tuple[tuple[int, int, str], list[PawItem], list[int]]
    _prefC: FloatArray | None
    _dep_acc: FloatArray | None
    _lay: dict[str, Any] | None
    _fdep: dict[tuple[int, Dirs], dict[int, FloatArray]]
    _fdep_owner: FloatArray | None
    _fwarned: bool

    def __init__(
        self,
        model: LocusModel,
        cs: Sequence[float],
        base: str = "r2SCAN",
        nodes: Iterable[float] = DENSITY_NODES,
        stencil: int = 2,
        nl_c: Sequence[float] | None = None,
        vv10: Mapping[str, float] | None = None,
    ) -> None:
        kernel: DConvKernel = DConvKernel(model, base, cs, nl_c=nl_c)
        MGGA.__init__(self, kernel, stencil)
        self.name = "DCONV"
        # rVV10 non-local correlation (O(N log N) FFT) on the total smooth density of the fine grid, added to e_g, v_sg
        # and dE/dsigma before the gradient correction; not in the PAW spheres (as GPAW's vdW-DF)
        self.vv10 = dict(vv10) if vv10 else None
        self._rvv10 = None
        self.cs = kernel.cs
        self.nodes = tuple(float(x) for x in nodes)
        self._spline = CubicSpline(np.log(self.nodes), np.eye(len(self.nodes)), bc_type="natural")
        self._dspline = self._spline.derivative()
        self._kmat = {}
        self._tabs = {}
        self._jtabs = {}
        self._gcell = None
        self._deposit_wg = None
        self._deposit_new = None
        self._vfb_wg = None
        self._vfb_pref = None
        self._S_wg = None  # smooth convolved fields of the last grid evaluation (K, N) on the full fine grid
        self._geom = None
        self.timings = {}

    def todict(self) -> dict[str, Any]:
        """the xc spec of the gpw parameters ({'name': 'LOCUS', ...}, set by locus_gpaw.LOCUS; gpaw.xc.XC rebuilds the functional from it
        once locus_gpaw is imported); otherwise the generic MGGA representation"""
        spec: dict[str, Any] | None = getattr(self, "_run_spec", None)
        return dict(spec) if spec else MGGA.todict(self)

    def initialize(self, density: Any, hamiltonian: Any, wfs: Any) -> None:
        MGGA.initialize(self, density, hamiltonian, wfs)
        _limit_threads(wfs.world.size)
        self.dEdD_asp = {}
        self.density = density
        self.finegd = density.finegd
        self._geom = None
        self._gcell = None
        self._kmat = {}
        self._tabs = {}
        self._S_wg = None
        self._deposit_wg = None
        self.spos_ac = None

    def set_positions(self, spos_ac: FloatArray) -> None:
        MGGA.set_positions(self, spos_ac)
        self.spos_ac = np.array(spos_ac, float)
        self._geom = None

    def _spos(self) -> FloatArray:
        if getattr(self, "spos_ac", None) is None:
            self.spos_ac = np.array(
                self.wfs.spos_ac if getattr(self.wfs, "spos_ac", None) is not None else self.density.spos_ac, float
            )
        return cast(FloatArray, self.spos_ac)

    def get_setup_name(self) -> str:
        return "PBE"

    def get_description(self) -> str:
        return f"{getattr(self, 'name', 'DCONV')} ({self.kernel.base} + shell term on the density-convolved descriptor, c = {self.cs})"

    # ------------------------------------------------------------------ helpers
    def _st(self) -> ModuleType:
        return _kernel_modules()[1]

    def _x_on(self) -> bool:
        """the kernel reads the convolved descriptors (kernel.x_needed: shell term on; not for r2SCAN + rVV10 alone)"""
        k: DConvKernel = self.kernel
        return bool(self._st().x_needed(k.model, k.sel, None))

    def _check_skips(self, xoff: bool, dedx: FloatArray, dedx_s: FloatArray | None, D: dict[str, Any]) -> None:
        """the skipped descriptor work must be unread by the kernel: dE/dx = 0 (x_needed False) and dE/dx_s = 0
        (channels not formed); a model term that reads them but is missing from kernel.x_needed / channel_x_needed shows
        up here"""
        if xoff and np.any(dedx):
            raise RuntimeError(
                "DCONV: a model term reads the convolved descriptors, but kernel.x_needed says it does not (update kernel.x_needed)"
            )
        if dedx_s is not None and D.get("dx_s") is None and len(dedx_s) and np.any(dedx_s):
            raise RuntimeError(
                "DCONV: a model term reads the channel descriptors, but kernel.channel_x_needed says it does not (update it)"
            )

    def _descriptors_off(self, n_s: FloatArray) -> dict[str, Any]:
        """placeholder descriptors (x = 1, dx = 0) when no term reads them (x_needed False)"""
        nspins: int
        P: int
        nspins, P = n_s.shape
        K: int = len(self.cs)
        x: FloatArray = np.ones((K, P))
        out: dict[str, Any] = dict(x_tot=x, dx_tot=np.zeros((K, P)), P_tot=None, live=n_s.sum(0) > 1e-10)
        if nspins == 2:
            out.update(x_s=np.broadcast_to(x[:, None, :], (K, 2, P)), dx_s=None, P_s=None)
        return out

    def _spinsum_on(self) -> bool:
        """total-density node convolution only (the kernel reads no channel descriptors); DCONV_SPINSUM=0: off"""
        return os.environ.get("DCONV_SPINSUM", "1") != "0"

    def _phi_adj(self, n: FloatArray, W: FloatArray, P_arr: FloatArray) -> FloatArray:
        """sum_c W_c p_c,a(n) (node-weight adjoint, (A, P)) with the node-weight arrays P_arr (K, A, P)"""
        return _einsum("cp,cap->ap", W, P_arr)

    def _weights(self, n: FloatArray, c: float) -> tuple[FloatArray, FloatArray]:
        kf: FloatArray = np.cbrt(3.0 * PI**2 * np.maximum(n, 1e-30))
        r0: FloatArray = c / kf
        inside: BoolArray = (r0 > self.nodes[0]) & (r0 < self.nodes[-1])
        u: FloatArray = np.log(np.clip(r0, self.nodes[0], self.nodes[-1]))
        chain: FloatArray = inside.astype(float)
        p: FloatArray = self._spline(u).T
        dp: FloatArray = self._dspline(u).T * (-chain / (3.0 * np.maximum(n, 1e-30)))[None, :]
        return p, dp

    def _geometry(self, gd: Any, full: bool = True) -> dict[str, Any]:
        """reciprocal vectors of the fine grid, Y_L(G), node dampings (cached per cell / grid) and the atom positions
        (cached per positions); the atom phases e^{-i G.R_a} are built per chunk from 1D factors (_phase1d)"""
        cell: FloatArray = gd.cell_cv
        N: tuple[int, ...] = tuple(int(x) for x in gd.N_c)
        pos: FloatArray = self._spos() @ cell
        key: tuple[tuple[int, ...], bytes, bytes, bool] = (N, cell.tobytes(), pos.tobytes(), full)
        if self._geom is not None and self._geom["key"] == key:
            return self._geom
        ckey: tuple[tuple[int, ...], bytes, tuple[float, ...], bool] = (N, cell.tobytes(), tuple(self.nodes), full)
        gc: dict[str, Any] | None = getattr(self, "_gcell", None)
        if gc is None or gc["ckey"] != ckey:
            inv: FloatArray = np.linalg.inv(cell)
            B: FloatArray = 2 * PI * inv.T
            kk: list[FloatArray] = [np.fft.fftfreq(N[i]) * N[i] for i in range(3)]
            # only |G|^2 is kept (built per x plane); G vectors (_gvec), |G| and the node dampings (_LazyDamp) are
            # rebuilt where used.  full=False (distributed grids): no grid array; |G|^2 per plane only for its maximum
            # (gline)
            G2: FloatArray | None = np.empty(N) if full else None
            g2max: float = 0.0
            i: int
            for i in range(N[0]):
                g2i: FloatArray = (
                    (kk[0][i] * B[0] + kk[1][:, None, None] * B[1] + kk[2][None, :, None] * B[2]) ** 2
                ).sum(-1)
                if full:
                    cast(FloatArray, G2)[i] = g2i
                else:
                    g2max = max(g2max, float(g2i.max()))
            if full:
                G2 = cast(FloatArray, G2).reshape(-1)
                g2max = G2.max()
            Lmax: int = max(s.xc_correction.Lmax for s in self.wfs.setups)
            gc = dict(
                ckey=ckey,
                N=N,
                cell=cell,
                inv=inv,
                B=B,
                kk=kk,
                G2=G2,
                Lmax=Lmax,
                vol=abs(np.linalg.det(cell)),
                gline=np.linspace(0, np.sqrt(g2max) * 1.0001, 3000),
                tails_cache={},
            )
            gc["damp"] = _LazyDamp(cast(FloatArray, G2), self.nodes, gc) if full else None
            self._gcell = gc
            self._jtabs = {}
        self._geom = dict(gc, key=key, pos=pos, pts={})
        return self._geom

    def _gvec(self, idx: IntArray | None = None) -> FloatArray:
        """G vectors (n, 3) of the flat fine-grid indices idx (all when None)"""
        geo: dict[str, Any] = cast("dict[str, Any]", self._geom)
        kk: list[FloatArray]
        B: FloatArray
        N: tuple[int, ...]
        kk, B, N = geo["kk"], geo["B"], geo["N"]
        if idx is None:
            return cast(
                FloatArray,
                (
                    kk[0][:, None, None, None] * B[0]
                    + kk[1][None, :, None, None] * B[1]
                    + kk[2][None, None, :, None] * B[2]
                ).reshape(-1, 3),
            )
        i0: IntArray
        i1: IntArray
        i2: IntArray
        i0, i1, i2 = np.unravel_index(idx, N)
        return cast(FloatArray, kk[0][i0][:, None] * B[0] + kk[1][i1][:, None] * B[1] + kk[2][i2][:, None] * B[2])

    def _phase1d(self, a: int) -> list[ComplexArray]:
        """1D factors of e^{-i G.R_a} = p0[k0] p1[k1] p2[k2] (G = sum_c k_c B_c, so G.R = sum_c k_c B_c.R)"""
        geo: dict[str, Any] = cast("dict[str, Any]", self._geom)
        return [np.exp(-1j * geo["kk"][c] * float(geo["B"][c] @ geo["pos"][a])) for c in range(3)]

    def _onecentre_tails(self, a: int) -> dict[str, Any]:
        """per spin: the one-centre coefficients D_sLq (nspins, Lmax, nq) of atom a.  Delta n_L = D_sLq Delta_q (+ the
        core difference, shared equally by the spins) and its form factors are linear in D_sLq; the D-independent radial
        work is tabulated per species (_species_tables, _jtab)"""
        xcc: Any = self.wfs.setups[a].xc_correction
        D_sp: FloatArray = self._dasp(a)
        D_sLq: FloatArray = np.inner(D_sp, xcc.B_pqL.T)  # (nspins, Lmax, nq)
        return dict(D_sLq=D_sLq, Lmax=xcc.Lmax, lmax=int(round(math.sqrt(xcc.Lmax))) - 1, nspins=len(D_sp))

    def _species_tables(self, a: int) -> dict[str, Any]:
        """D-independent radial tables of atom a's setup (cached per setup): on ru (step RU_STEP, to rcut + 0.5) U
        (nq, nru) = Delta_q = n_q - nt_q, PU (nq, nl, nru) its tails per l (= U: exact tails), Uc / PUc the same for the
        core difference sqrt(4 pi) (nc - nct); PUr = PU ru^2 dru, PUcr; PUrq / PUcrq the same on the setup grid
        (quadrature of the tails transforms)"""
        setup: Any = self.wfs.setups[a]
        xcc: Any = setup.xc_correction
        r_g: FloatArray = xcc.rgd.r_g
        key: TabKey = (setup.symbol, len(r_g))
        if key in self._tabs:
            return self._tabs[key]
        rcut: float = float(max(setup.rcut_j))
        if min(self.nodes) < EXACT_TAILS_MIN_W:
            raise ValueError(
                f"dconv: the PAW-consistent tails need node widths >= {EXACT_TAILS_MIN_W} bohr (have {min(self.nodes)}); narrower kernels "
                "are not implemented"
            )
        ru: FloatArray = np.arange(0.0, rcut + 0.5, RU_STEP)
        nq: int = xcc.n_qg.shape[0]
        lmax: int = int(round(math.sqrt(xcc.Lmax))) - 1
        # PAW-consistent tails: the one-centre differences Delta_q = n_q - nt_q and the core difference
        # sqrt(4 pi)(nc - nct) enter the smooth field exactly (no pseudization).  With node widths >= EXACT_TAILS_MIN_W
        # the convolved field is band limited on the fine grid (exp(-G^2 w^2 / 2)), so grid, PS and AE points all carry
        # the all-electron field and the own-atom AE correction (T_ae - T_ps, Tc) vanishes.
        U: FloatArray = np.array([np.interp(ru, r_g, xcc.n_qg[q] - xcc.nt_qg[q], right=0.0) for q in range(nq)])
        PU: FloatArray = np.repeat(U[:, None, :], lmax + 1, axis=1)
        Uc: FloatArray = np.interp(ru, r_g, math.sqrt(4 * PI) * (xcc.nc_g - xcc.nct_g), right=0.0)
        PUc: FloatArray = Uc.copy()
        w2: FloatArray = ru**2 * RU_STEP
        # tails quadrature (Bessel transforms, adjoint) on the setup's logarithmic grid: resolves the core peaks the uniform ru step does not
        gq: int = int(np.searchsorted(r_g, rcut + 0.5)) + 1
        rq: FloatArray = np.ascontiguousarray(r_g[:gq])
        wq: FloatArray = rq**2 * xcc.rgd.dr_g[:gq]
        Uq: FloatArray = np.array([(xcc.n_qg[q] - xcc.nt_qg[q])[:gq] for q in range(nq)])
        Ucq: FloatArray = (math.sqrt(4 * PI) * (xcc.nc_g - xcc.nct_g))[:gq]
        PUrq: FloatArray = np.repeat((Uq * wq)[:, None, :], lmax + 1, axis=1)
        PUcrq: FloatArray = Ucq * wq
        # negligible tails (sum_q int |PU_q| r^2 dr + int |PUc| r^2 dr <= TAILS_NEGL): the species skips the tails form
        # factors (_tails_fsum) and their adjoint (_tails_adjoint); DCONV_TAILS_EXACT=1 evaluates them anyway
        negl: bool = (
            not os.environ.get("DCONV_TAILS_EXACT")
            and float(np.sum(np.abs(PU) * w2) + np.sum(np.abs(PUc) * w2)) <= TAILS_NEGL
        )
        self._tabs[key] = dict(
            ru=ru,
            rcut=rcut,
            U=U,
            PU=PU,
            Uc=Uc,
            PUc=PUc,
            PUr=PU * w2,
            PUcr=PUc * w2,
            negl=negl,
            rq=rq,
            PUrq=PUrq,
            PUcrq=PUcrq,
            lofL=np.array([int(math.isqrt(L)) for L in range(xcc.Lmax)]),
            key=key,
        )
        return self._tabs[key]

    def _species_kprod(self, a: int) -> dict[str, Any]:
        """the kernel products T_ae = g_{l,w} * U_q, T_ps = g_{l,w} * PU_ql on r_g (nw, nl, nq, ng) and Tc (nw, ng),
        built on first use (k-parallel: only on the ranks that evaluate atoms of the species) and cached on disk
        (_disk_cache)"""
        st: dict[str, Any] = self._species_tables(a)
        if "T_ae" not in st:
            r_g: FloatArray = self.wfs.setups[a].xc_correction.rgd.r_g
            ru: FloatArray
            U: FloatArray
            PU: FloatArray
            Uc: FloatArray
            ru, U, PU, Uc = st["ru"], st["U"], st["PU"], st["Uc"]

            def compute() -> dict[str, FloatArray]:
                nl: int = PU.shape[1]
                T_ae: FloatArray = np.zeros((len(self.nodes), nl, U.shape[0], len(r_g)))
                T_ps: FloatArray = np.zeros_like(T_ae)
                Tc: FloatArray = np.zeros((len(self.nodes), len(r_g)))
                wi: int
                w: float
                for wi, w in enumerate(self.nodes):
                    ell: int
                    for ell in range(nl):
                        Kl: FloatArray = self._kernel(a, w, ell, ru)
                        T_ae[wi, ell] = U @ Kl.T
                T_ps[:] = T_ae  # exact tails: the PS route sees the same own-atom field, Tc = g * (Uc - PUc) = 0
                return dict(T_ae=T_ae, T_ps=T_ps, Tc=Tc)

            tb: dict[str, FloatArray] = _disk_cache(
                "dconv_kprod_v2_exact", [r_g, ru, U, Uc, np.array(self.nodes)], compute
            )
            st.update(tb)  # T_d = T_ae - T_ps is formed per l where used
        return st

    def _jtab(self, a: int) -> tuple[FloatArray, FloatArray]:
        """spherical Bessel transforms of the tails on gline: J (nq, nl, ngl) = int j_l(g r) PU_ql(r) r^2 dr, Jc (ngl)
        (core, l = 0); cached per species and gline (i.e. per cell / grid)"""
        st: dict[str, Any] = self._species_tables(a)
        geo: dict[str, Any] = cast("dict[str, Any]", self._geom)
        if st["key"] not in self._jtabs:
            rq: FloatArray
            nl: int
            rq, nl = st["rq"], st["PU"].shape[1]
            J: FloatArray = np.empty((st["PU"].shape[0], nl, len(geo["gline"])))
            ell: int
            for ell in range(nl):
                jl: FloatArray = spherical_jn(ell, geo["gline"][:, None] * rq[None, :])
                J[:, ell] = st["PUrq"][:, ell] @ jl.T
                if ell == 0:
                    Jc: FloatArray = jl @ st["PUcrq"]
            self._jtabs[st["key"]] = (J, Jc)
        return self._jtabs[st["key"]]

    def _tails_half(self) -> dict[str, Any]:
        """half-space G set of the tails form factors (cached per cell): the real tails have F(-G) = conj F(G) exactly
        (Y_L(-G) = (-1)^l Y_L(G), e^{iG.R} = conj e^{-iG.R}), so F is computed for i2 <= N2 // 2 (and the Nyquist rows
        of axes 0 / 1, whose partner index is not -G) and conjugated onto the rest.  Only the index sets are kept
        (int32 below 2^31 points)."""
        gc: dict[str, Any] = cast("dict[str, Any]", self._gcell)
        if "hidx" not in gc:
            geo: dict[str, Any] = cast("dict[str, Any]", self._geom)
            N: tuple[int, ...] = geo["N"]
            it: type[np.signedinteger[Any]] = np.int32 if int(np.prod(N)) < 2**31 else np.int64
            i0: IntArray
            i1: IntArray
            i2: IntArray
            i0, i1, i2 = np.unravel_index(np.arange(int(np.prod(N))), N)
            nyq: BoolArray = ((N[0] % 2 == 0) & (i0 == N[0] // 2)) | ((N[1] % 2 == 0) & (i1 == N[1] // 2))
            direct: BoolArray = (i2 <= N[2] // 2) | nyq
            hidx: IntArray = np.flatnonzero(direct)
            cdst: IntArray = np.flatnonzero(~direct)
            csrc: IntArray = np.ravel_multi_index(((-i0[cdst]) % N[0], (-i1[cdst]) % N[1], (-i2[cdst]) % N[2]), N)
            gc.update(hidx=hidx.astype(it), cdst=cdst.astype(it), csrc=csrc.astype(it))
        return gc

    def _tails_fsum(self, g0: int, g1: int, atoms: Iterable[int] | None = None) -> ComplexArray:
        """sum_a e^{-i G.R_a} F_a,s(G) for the entries [g0, g1) of the half-space set _tails_half()["hidx"]:
        (nspins, g1 - g0) complex, with the form factors of the tails F_a,s(G) = 4 pi sum_L (-i)^l Y_L(G) IL_a,sL(|G|),
        IL = D_sLq J_ql (+ Jc / nspins for L = 0), linearly interpolated in |G| on gline (np.interp's slope form).
        Every G element is computed from its own data only (atom order fixed), so any split of the range
        (chunks, MPI ranks) gives bit-identical values."""
        geo: dict[str, Any] = cast("dict[str, Any]", self._geom)
        hs: dict[str, Any] = self._tails_half()
        nspins: int = self._tails[0]["nspins"]
        atoms = [a for a in (range(len(geo["pos"])) if atoms is None else atoms) if not self._species_tables(a)["negl"]]
        out: ComplexArray = np.zeros((nspins, g1 - g0), complex)
        dg: FloatArray = np.diff(geo["gline"])
        per: list[tuple[int, FloatArray, FloatArray, IntArray, list[ComplexArray]]] = []
        a: int
        for a in atoms:
            tl: dict[str, Any] = self._tails[a]
            J: FloatArray
            Jc: FloatArray
            J, Jc = self._jtab(a)
            lofL: IntArray = self._species_tables(a)["lofL"]
            IL: FloatArray = np.empty((nspins, tl["Lmax"], J.shape[-1]))
            ell: int
            for ell in range(tl["lmax"] + 1):
                Ls: slice = slice(ell * ell, (ell + 1) ** 2)
                IL[:, Ls] = np.einsum("sLq,qk->sLk", tl["D_sLq"][:, Ls], J[:, ell])
            IL[:, 0] += Jc / nspins
            per.append((a, IL, (IL[..., 1:] - IL[..., :-1]) / dg, lofL, self._phase1d(a)))
        CH: int = 32768
        gline: FloatArray = geo["gline"]
        c0: int
        for c0 in range(g0, g1, CH):
            c1: int = min(c0 + CH, g1)
            hid: IntArray = hs["hidx"][c0:c1]
            i0: IntArray
            i1: IntArray
            i2: IntArray
            i0, i1, i2 = np.unravel_index(hid, geo["N"])
            tc: tuple[IntArray, FloatArray, FloatArray] | None = hs["tails_cache"].get((c0, c1))
            if (
                tc is None
            ):  # G, |G| (np.interp's slope form on gline) and Y_L(G) of the chunk, kept within DCONV_GRIDCACHE_MB
                Gv: FloatArray = self._gvec(hid)
                Gabs: FloatArray = np.sqrt((Gv**2).sum(1))
                j: IntArray = np.clip(np.searchsorted(gline, Gabs, side="right") - 1, 0, len(gline) - 2)
                tc = (j.astype(np.int32), Gabs - gline[j], np.ascontiguousarray(_ylm(Gv, geo["Lmax"]).T))
                del Gv, Gabs
                if _gridcache_ok(hs, sum(x.nbytes for x in tc)):
                    hs["tails_cache"][(c0, c1)] = tc
            dx: FloatArray
            YGt: FloatArray
            j, dx, YGt = tc
            sl: FloatArray
            p1d: list[ComplexArray]
            for a, IL, sl, lofL, p1d in per:
                ph: ComplexArray = p1d[0][i0] * p1d[1][i1] * p1d[2][i2]
                s_: int
                for s_ in range(nspins):
                    Ich: FloatArray = sl[s_][:, j] * dx + IL[s_][:, j]  # np.interp: slope (x - xp_j) + fp_j
                    fr: FloatArray = np.zeros(c1 - c0)
                    fi: FloatArray = np.zeros(c1 - c0)
                    L: int
                    for L, ell in enumerate(lofL):
                        c: complex = 4 * PI * (-1j) ** ell
                        if ell % 2 == 0:
                            fr += (c.real * YGt[L]) * Ich[L]
                        else:
                            fi += (c.imag * YGt[L]) * Ich[L]
                    out[s_, c0 - g0 : c1 - g0] += ph * (fr + 1j * fi)
        return out

    def _kernel(self, a: int, w: float, ell: int, ru: FloatArray) -> FloatArray:
        r_g: FloatArray = self.wfs.setups[a].xc_correction.rgd.r_g
        key: KmatKey = (self.wfs.setups[a].symbol, w, ell, len(ru), len(r_g), float(r_g[-1]))
        if key not in self._kmat:
            gk: tuple[str, float, int, int, int, float, float, float] = key + (float(r_g[1]), float(ru[-1]))
            if gk not in _KMAT_CACHE:
                _KMAT_CACHE[gk] = _kernel_matrix(r_g, ru, np.full(len(ru), RU_STEP), ell, w * w)
            self._kmat[key] = _KMAT_CACHE[gk]
        return self._kmat[key]

    # ------------------------------------------------------------------ grid part
    def _descriptors(self, n_s: FloatArray, S_s: FloatArray, keep_P: bool = True) -> dict[str, Any]:
        """x_tot (K, P), dxloc_tot (K, P) [d x_tot / d n_s, equal for both spins], P_tot (K, A, P) and, for nspins = 2,
        x_s (K, 2, P), dxloc_s (K, 2, P) [d x_s / d n_s, same spin], P_s (K, 2, A, P) with the channel densities 2 n_s.
        S_s: convolved fields per spin (nspins, A, P); the one-spin case reduces to the spin-paired construction.
        keep_P=False (grid pass): P_tot / P_s are not kept; _feedback_terms recomputes them per chunk."""
        nspins: int = n_s.shape[0]
        n: FloatArray = n_s.sum(0)
        ns: FloatArray = np.maximum(n, 1e-30)
        S: FloatArray = S_s.sum(0)
        K: int = len(self.cs)
        x_tot: FloatArray
        dx_tot: FloatArray
        P_tot: list[FloatArray]
        x_tot, dx_tot, P_tot = np.empty((K, n.size)), np.empty((K, n.size)), []
        ic: int
        c: float
        for ic, c in enumerate(self.cs):
            p: FloatArray
            dp: FloatArray
            p, dp = self._weights(n, c)
            if keep_P:
                P_tot.append(p)
            x_tot[ic] = _einsum("ap,ap->p", p, S) / ns
            dx_tot[ic] = _einsum("ap,ap->p", dp, S) / ns - x_tot[ic] / ns
            del p, dp
        live: BoolArray = n > 1e-10
        dx_tot[:, ~live] = 0.0
        out: dict[str, Any] = dict(x_tot=x_tot, dx_tot=dx_tot, P_tot=np.array(P_tot) if keep_P else None, live=live)
        if nspins == 2:
            # the kernel does not read the channel descriptors (dE/dx_s = 0 identically): not formed (placeholder values)
            out.update(x_s=np.broadcast_to(x_tot[:, None, :], (K, 2, n.size)), dx_s=None, P_s=None)
        return out

    def _full_row(self, v: FloatArray, par: _KPar | None) -> FloatArray:
        """a slab vector of the distributed layout (this rank's points rng(P)) -> the whole grid on every rank; identity otherwise"""
        p0: int
        p1: int
        dist: bool
        p0, p1, dist = getattr(self, "_slab", (0, len(v), False))
        if not dist:
            return v
        P: int = int(np.prod(cast("dict[str, Any]", self._geom)["N"]))
        full: FloatArray = np.zeros((1, P))
        full[0, p0:p1] = v
        cast(_KPar, par).fill_last(full)
        return cast(FloatArray, full[0])

    def _feedback_terms(
        self, n_s: FloatArray, dedx: FloatArray, dedx_s: FloatArray | None, D: dict[str, Any]
    ) -> tuple[FloatArray, FloatArray]:
        """the local adjoint vx_s (nspins, P) = sum_c dE/dx_c dx_c/dn_s and the feedback sources F_s
        (nspins, nodes, P) = sum_c dE/dx_c p_w(n) / n of the grid pass (the total-density descriptors only: the same for
        both spins); p_w is recomputed per chunk of KCHUNK points"""
        nspins: int
        P: int
        nspins, P = n_s.shape
        n: FloatArray = n_s.sum(0)
        ns: FloatArray = np.maximum(n, 1e-10)
        F_s: FloatArray = np.zeros((nspins, len(self.nodes), P))
        vx_s: FloatArray = np.zeros((nspins, P))
        s_: int
        for s_ in range(nspins):
            vx_s[s_] = _einsum("cp,cp->p", dedx, D["dx_tot"])
        c0: int
        for c0 in range(0, P, KCHUNK):
            sl: slice = slice(c0, min(c0 + KCHUNK, P))
            P_tot: FloatArray = np.array([self._weights(n[sl], c)[0] for c in self.cs])
            for s_ in range(nspins):
                F_s[s_, :, sl] = _einsum("cp,cap->ap", dedx[:, sl] / ns[None, sl], P_tot)
        return vx_s, F_s

    def _dom_mode(self) -> bool:
        """the distributed-grid layout (dconv_dom): GPAW's XC grid is domain-decomposed (parallel={'domain': N},
        augment_grids) or DCONV_DOMAIN=1 (the same layout on undistributed grids: k-point parallel or serial runs)"""
        gd: Any = getattr(self, "gd", None)
        return gd is not None and (gd.comm.size > 1 or os.environ.get("DCONV_DOMAIN", "0") == "1")

    def process_mgga(
        self, e_g: FloatArray, nt_sg: FloatArray, v_sg: FloatArray, sigma_xg: FloatArray, dedsigma_xg: FloatArray
    ) -> None:
        import time

        self._dom_on = self._dom_mode()
        self._spinsum = (
            False  # node fields of the total density only (set by the domain path, read by _onecentre_fields)
        )
        if self._dom_on:
            return self._process_mgga_dom(e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg)
        t0: float = time.time()
        par: _KPar | None = self._kpar()
        self._t_tick = time.perf_counter()
        nspins: int = len(nt_sg)
        if self.fixed_ke:
            taut_sG: FloatArray | None = (
                self._fixed_taut_sG.copy()
            )  # the core part is added in place below: never accumulate it into the fixed array
            if not self._taut_gradv_init:
                self._taut_gradv_init = True
                self.wfs.calculate_kinetic_energy_density()
        else:
            taut_sG = self.wfs.calculate_kinetic_energy_density()
        if taut_sG is None:
            # no wave functions yet (LCAO start): von Weizsaecker kinetic energy density, as GPAW's MGGA (gpaw issue 124).
            # tau = 0 would put the iso-orbital indicator z = tau_W / tau and its tau-derivative (x^z channels) out of range
            nt0_sg: FloatArray = nt_sg.copy()
            nt0_sg[nt0_sg < 1e-10] = np.inf
            tw_sg: FloatArray = sigma_xg[::2] / 8 / nt0_sg
            taut_sG = self.wfs.gd.empty(len(nt_sg))
            tw_g: FloatArray
            taut_G: FloatArray
            for tw_g, taut_G in zip(tw_sg, taut_sG):
                self.restrict_and_collect(tw_g, taut_G)
        taut_sg: FloatArray = np.empty_like(nt_sg)
        taut_g: FloatArray
        for taut_G, taut_g in zip(taut_sG, taut_sg):
            taut_G += 1.0 / self.wfs.nspins * self.tauct_G
            self.distribute_and_interpolate(taut_G, taut_g)
        self._D_asp = None
        if par is not None:
            # GPAW's k-parallel density, sigma and D_asp can differ between the ranks in the last bits (an allreduce is
            # not bitwise identical on all ranks); the split work must use one input -> rank 0's copies
            # (GPAW's arrays untouched)
            nt_sg = par.bcast(np.array(nt_sg))
            sigma_xg = par.bcast(np.array(sigma_xg))
            taut_sg = par.bcast(taut_sg)
            self._D_asp = {a: par.bcast(np.array(self.density.D_asp[a], float)) for a in range(len(self.wfs.setups))}
        self._tick("tau")
        gd: Any = self.finegd
        geo: dict[str, Any] = self._geometry(gd)
        N: tuple[int, ...] = geo["N"]
        n_s: FloatArray = np.clip(nt_sg, 0.0, None).reshape(nspins, -1)
        assert nt_sg[0].shape == N, (nt_sg[0].shape, N)
        self._tails = {a: self._onecentre_tails(a) for a in range(len(geo["pos"]))}
        from scipy.ndimage import spline_filter

        # no term reads the convolved descriptors (kernel.x_needed, e.g. r2SCAN + rVV10): node convolutions,
        # descriptors, feedback, deposits and the one-centre fields are skipped (dE/dx = 0 identically)
        xoff: bool
        xoff = self._xoff = not self._x_on()
        # convolved fields S_s,w = G_w * (pseudo density + one-centre tails) and their spline prefilters; k-parallel:
        # the tails form factors split over G, the (spin, node) items over the ranks
        nw: int
        NG: int
        nw, NG = len(self.nodes), int(np.prod(N))
        Fsum_s: ComplexArray | None = None
        if not xoff and any(not self._species_tables(a)["negl"] for a in range(len(geo["pos"]))):
            hs: dict[str, Any] = self._tails_half()
            nh: int = len(hs["hidx"])
            g0: int
            g1: int
            g0, g1 = (0, nh) if par is None else par.rng(nh)
            Fh: ComplexArray = self._tails_fsum(g0, g1)
            if par is not None:
                Fh = par.gather(Fh.T, nh).T
            Fsum_s = np.empty((nspins, NG), complex)
            Fsum_s[:, hs["hidx"]] = Fh
            Fsum_s[:, hs["cdst"]] = np.conj(Fsum_s[:, hs["csrc"]])
            Fsum_s *= np.prod(N) / geo["vol"]
        items: list[tuple[int, int]] = [(s_, w) for s_ in range(nspins) for w in range(nw)]
        i0: int
        i1: int
        i0, i1 = (0, len(items)) if par is None else par.rng(len(items))
        P: int = n_s.shape[1]
        # distributed layout (k-parallel; DCONV_DIST=0: replicated layout): the (spin, node) items stay on their owner
        # rank as whole grids (FFT side), every pointwise quantity (descriptors, kernel, feedback sources, deposits)
        # lives on the rank's point slab rng(P) (alltoallv transposes between the two); the prefiltered fields are
        # served to the one-centre pass at the grid points it reads (_pref_fetch).  Serial: the prefiltered fields go
        # straight into the (P, nspins K) read-out stack; one spin's density spectrum at a time (items are spin-major).
        dist: bool = (
            par is not None
            and os.environ.get("DCONV_DIST", "1") != "0"
            and os.environ.get("DCONV_SPLAT", "linear") != "cubic"
        )
        p0: int
        p1: int
        p0, p1 = cast(_KPar, par).rng(P) if dist else (0, P)
        Pl: int = p1 - p0
        self._slab = (p0, p1, dist)
        S_x: FloatArray = np.empty((0 if xoff else i1 - i0, P))
        prefT: FloatArray | None = None if (dist or xoff) else np.empty((P, len(items)))
        pref_own: FloatArray | None = np.empty((i1 - i0, P)) if (dist and not xoff) else None
        mk: dict[int, ComplexArray] = {}
        i: int
        for i in range(i0, i0 if xoff else i1):
            s_: int
            w: int
            s_, w = items[i]
            if s_ not in mk:
                mk = {s_: _fftn(n_s[s_].reshape(N)).reshape(-1) + (Fsum_s[s_] if Fsum_s is not None else 0.0)}
            S_x[i - i0] = np.real(_ifftn((mk[s_] * geo["damp"][w]).reshape(N))).reshape(-1)
            pf: FloatArray = spline_filter(S_x[i - i0].reshape(N), order=3, mode="grid-wrap").reshape(-1)
            if dist:
                cast(FloatArray, pref_own)[i - i0] = pf
            else:
                cast(FloatArray, prefT)[:, i] = pf
            del pf
        del mk, Fsum_s
        if xoff:
            S_slab: FloatArray | None = None
        elif dist:
            S_slab = cast(_KPar, par).transpose(S_x, len(items), P)  # (items, P_slab)
        elif par is not None:
            S_slab = par.gather(S_x, len(items))
            prefT = np.ascontiguousarray(
                par.gather(np.ascontiguousarray(cast(FloatArray, prefT)[:, i0:i1].T), len(items)).T
            )
        else:
            S_slab = S_x
        del S_x
        self._S_wg = self._S_swg = None
        # (P, nspins K) C-ordered stack for the sparse one-centre read-out (_read); the per-node fields are views of its
        # columns
        self._S_prefT = prefT
        self._pref_own, self._pref_rng = pref_own, (i0, i1)
        del prefT, pref_own
        self._S_pref = (
            None
            if self._S_prefT is None
            else [[self._S_prefT[:, s_ * nw + w].reshape(N) for w in range(nw)] for s_ in range(nspins)]
        )
        self._tick("tails+S")
        D: dict[str, Any] = (
            self._descriptors_off(n_s[:, p0:p1])
            if xoff
            else self._descriptors(n_s[:, p0:p1], cast(FloatArray, S_slab).reshape(nspins, nw, -1), keep_P=False)
        )
        del S_slab
        K: int = len(self.cs)
        live: BoolArray = D["live"]  # this rank's slab
        dedtaut_sg: FloatArray = np.zeros_like(nt_sg)
        dedx: FloatArray = np.zeros((K, Pl))
        if nspins == 1:
            self._kernel_calculate(
                par, e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg, taut_sg, dedtaut_sg, D["x_tot"], dedx, xoff=p0
            )
            dedx_s: FloatArray | None = None
        else:
            dedx_s = np.zeros((K, 2, Pl))
            self._kernel_calculate(
                par,
                e_g,
                nt_sg,
                v_sg,
                sigma_xg,
                dedsigma_xg,
                taut_sg,
                dedtaut_sg,
                D["x_tot"],
                dedx,
                D["x_s"],
                dedx_s,
                xoff=p0,
            )
        self._check_skips(xoff, dedx, dedx_s, D)
        self._tick("kernel")
        if self.vv10:
            from .rvv10 import RVV10

            if self._rvv10 is None:
                self._rvv10 = RVV10(b=float(self.vv10.get("b", 10.0)))
            n_tot: FloatArray = np.clip(nt_sg.sum(0), 0.0, None)
            sig_tot: FloatArray = sigma_xg[0] if nspins == 1 else sigma_xg[0] + 2.0 * sigma_xg[1] + sigma_xg[2]
            E_nl: float
            vn_g: FloatArray
            vs_g: FloatArray
            E_nl, vn_g, vs_g = self._rvv10.calculate(n_tot, np.clip(sig_tot, 0.0, None), gd.cell_cv, par=par)
            e_g += cast(FloatArray, self._rvv10.last_e)
            for s_ in range(nspins):
                v_sg[s_] += vn_g
            if nspins == 1:
                dedsigma_xg[0] += vs_g
            else:
                dedsigma_xg[0] += vs_g
                dedsigma_xg[1] += 2.0 * vs_g
                dedsigma_xg[2] += vs_g
            self.timings["E_nl_vv10"] = E_nl
            self._rvv10.last_e = None  # release rVV10's output array once added
            del vn_g, vs_g
        # sphere -> grid feedback, lagged by default: the one-centre deposits of the previous Hamiltonian update enter
        # this potential.  DCONV_SYNC=1: synchronous, the one-centre (PAW) pass of every atom runs here on the current
        # D_asp and fields, so its deposits enter this potential; the results are cached for GPAW's
        # calculate_paw_correction calls of this update (same D_sp), which then only add the tails adjoint of the new
        # feedback potential.  The first evaluation of this xc object (no lagged deposit yet: after a gpw read or an xc
        # switch) is synchronous, so that a single Hamiltonian rebuild is consistent with the stored state
        # (DCONV_FIRST_SYNC=0: lagged from the start).
        self._tick("vv10")
        first: bool = self._deposit_wg is None and os.environ.get("DCONV_FIRST_SYNC", "1") != "0"
        sync: bool = (
            (bool(os.environ.get("DCONV_SYNC")) or first)
            and not os.environ.get("DCONV_NO_DEPOSIT")
            and getattr(self, "density", None) is not None
        )
        self._paw_cache = None
        if sync:
            self._deposit_new = (
                None if xoff else np.zeros((nspins, len(self.nodes), Pl))
            )  # this rank's slab (all points in serial / replicated runs)
            cache: PawCache = self._paw_pass(par)
            self._deposit_wg = self._deposit_new
            self._deposit_new = None  # the cached PAW calls of this update must not deposit again
            self._paw_cache = cache
        self._tick("sync")
        if (
            xoff
        ):  # no descriptor feedback (dE/dx = 0): no node spectra, no deposits; the potential is complete on every rank
            self._vfb_swg = np.zeros((nspins, 1, P))
            self._vfb_wg = self._vfb_swg[0]
            self._vfb_pref = None
            self._deposit_new = None
            self._deposit_wg = _DEP_USED
        else:
            vx_s: FloatArray
            F_s: FloatArray
            vx_s, F_s = self._feedback_terms(n_s[:, p0:p1], dedx, dedx_s, D)
            F_s[:, :, ~live] = 0.0
            F_shape: tuple[int, ...] = F_s.shape
            if (
                getattr(self, "_deposit_new", None) is not None
                and cast(FloatArray, self._deposit_new).shape == F_s.shape
            ):
                self._deposit_wg = self._deposit_new  # sphere weights accumulated during the previous PAW pass
            if (
                getattr(self, "_deposit_wg", None) is not None
                and cast(FloatArray, self._deposit_wg).shape == F_s.shape
                and not os.environ.get("DCONV_NO_DEPOSIT")
            ):
                F_s += self._deposit_filter(
                    cast(FloatArray, self._deposit_wg), N
                )  # sphere -> grid feedback (one-centre dE/dC through S~ at the atomic points; lagged one SCF step)
            if (
                not sync
                and isinstance(getattr(self, "_deposit_wg", None), np.ndarray)
                and cast(FloatArray, self._deposit_wg).size
            ):
                # released once added when this update's PAW pass refills _deposit_new (after a synchronous update the
                # next one reuses it); not None: the next update is lagged, not a first one
                self._deposit_wg = _DEP_USED
            # feedback potential sum_w G_w * F_s,w: only the node sum enters v_s and (linearly) the tails adjoint -> one
            # inverse FFT per spin (_vfb_swg: the (nspins, 1, P) node-summed field).  k-parallel: the (spin, node)
            # spectra of the rank's items summed in node order and reduced over the ranks; the distributed layout brings
            # the slab sources to the item owners first (transpose_back)
            if dist:
                F_own: FloatArray = cast(_KPar, par).transpose_back(F_s.reshape(len(items), Pl), len(items), P)
                del F_s
            else:
                F_own = F_s.reshape(len(items), P)[i0:i1]
            spec_s: ComplexArray = np.zeros((nspins, NG), complex)
            for i in range(i0, i1):
                s_, w = items[i]
                spec_s[s_] += _fftn(F_own[i - i0].reshape(N)).reshape(-1) * geo["damp"][w]
            del F_own
            if par is not None:
                par.reduce_bcast(spec_s.view(float))
            self._vfb_swg = np.array([[np.real(_ifftn(spec_s[s_].reshape(N))).reshape(-1)] for s_ in range(nspins)])
            del spec_s
            self._vfb_wg = self._vfb_swg[0]
            self._vfb_pref = None
            self._deposit_new = None if sync else np.zeros(F_shape)
            for s_ in range(
                nspins
            ):  # the local adjoint is known on the slab: v there, then the slabs all-gathered (distributed layout)
                vf: FloatArray = v_sg[s_].reshape(-1)
                vf[p0:p1] += vx_s[s_] + self._vfb_swg[s_].sum(0)[p0:p1]
            if dist:
                cast(_KPar, par).fill_last(v_sg.reshape(nspins, -1))
        self._tick("feedback")
        self.dedtaut_sG = self.wfs.gd.empty(self.wfs.nspins)
        self.ekin = 0.0
        for s_ in range(self.wfs.nspins):
            self.restrict_and_collect(dedtaut_sg[s_], self.dedtaut_sG[s_])
            self.ekin -= self.wfs.gd.integrate(self.dedtaut_sG[s_] * (taut_sG[s_] - self.tauct_G / self.wfs.nspins))
        self._tick("restrict")
        self.timings["grid"] = time.time() - t0
        vfb: FloatArray = self._vfb_swg.sum((0, 1))
        live_f: BoolArray = n_s.sum(0) > 1e-10
        xt: FloatArray = D["x_tot"] if not dist else np.array([self._full_row(r, par) for r in D["x_tot"]])
        self._x_stats = dict(x_med=np.median(xt[:, live_f], axis=1), vfb_rms=float(np.sqrt(np.mean(vfb[live_f] ** 2))))
        del xt, D
        if (
            par is not None
            and not sync
            and getattr(self, "density", None) is not None
            and not os.environ.get("DCONV_NO_DEPOSIT")
        ):
            # k-parallel, lagged deposits: the one-centre pass of this update runs here, distributed over the ranks by
            # cost, on the state GPAW's per-atom calculate_paw_correction calls see next; its results are cached for
            # those calls (which add only the tails adjoint) and the deposits are complete (summed over the ranks) for
            # the next update.  Serial runs keep the per-atom evaluation in GPAW's calls.
            t1: float = time.time()
            self._paw_cache = self._paw_pass(par)
            self.timings["paw_pass"] = time.time() - t1
            self._tick("paw_pass")

    # ------------------------------------------------------------------ distributed fine grid (domain decomposition)
    def _dom_damp(self, w: int) -> FloatArray:
        """node damping exp(-G^2 w^2 / 2) of node index w on the Y slab of the distributed layout (kept within DCONV_GRIDCACHE_MB)"""
        dom: DomLayout = self._domL
        c: dict[int, FloatArray] = dom._gcache.setdefault("damp", {})
        if w in c:
            return c[w]
        wv: float = self.nodes[w]
        d: FloatArray = np.exp(
            -0.5 * dom.g2(cast("dict[str, Any]", self._geom)["kk"], cast("dict[str, Any]", self._geom)["B"]) * wv * wv
        )
        if _gridcache_ok(cast("dict[str, Any]", self._gcell), d.nbytes):
            c[w] = d
        return d

    def _gather_dasp_dom(self, dom: DomLayout) -> dict[int, FloatArray]:
        """D_sp of every atom on every rank (distributed grids: GPAW keeps an atom's D_sp on its domain's ranks): the first k-point / band
        replica of each owner contributes, one exact sum (a single non-zero term per entry)"""
        D_asp: Any = self.density.D_asp
        nat: int = len(self.wfs.setups)
        nc: int = int(self.density.ncomponents)
        sz: list[int] = [nc * self.wfs.setups[a].ni * (self.wfs.setups[a].ni + 1) // 2 for a in range(nat)]
        off: IntArray = np.r_[0, np.cumsum(sz)]
        buf: FloatArray = np.zeros(int(off[-1]))
        if self.wfs.kptband_comm.rank == 0:
            a: int
            D: FloatArray
            for a, D in D_asp.items():
                buf[off[a] : off[a + 1]] = np.asarray(D, float).ravel()
        dom.allsum(buf)
        return {a: buf[off[a] : off[a + 1]].reshape(nc, -1).copy() for a in range(nat)}

    TAILS_CH: int = 32768  # the chunk length of the numpy loops (_tails_fsum / _tails_F_np)

    def _tails_F_idx(self, gidx: IntArray) -> ComplexArray:
        """sum_a e^{-i G.R_a} F_a,s(G) at the global flat G indices gidx (both half-spaces; the Y slab of the
        distributed layout): the expressions of _tails_fsum element by element, the half-space conjugation replaced by
        the direct evaluation (~1e-16)"""
        return self._tails_F_np(gidx)

    def _tails_F_np(self, gidx: IntArray, cache: bool = True) -> ComplexArray:
        """the numpy loop of _tails_F_idx (cache=False: the chunk tables are neither read from nor written to the cell
        cache)"""
        geo: dict[str, Any] = cast("dict[str, Any]", self._geom)
        gc: dict[str, Any] = cast("dict[str, Any]", self._gcell)
        nspins: int = self._tails[0]["nspins"]
        atoms: list[int] = [a for a in range(len(geo["pos"])) if not self._species_tables(a)["negl"]]
        out: ComplexArray = np.zeros((nspins, len(gidx)), complex)
        dg: FloatArray = np.diff(geo["gline"])
        per: list[tuple[int, FloatArray, FloatArray, IntArray, list[ComplexArray]]] = []
        a: int
        for a in atoms:
            tl: dict[str, Any] = self._tails[a]
            J: FloatArray
            Jc: FloatArray
            J, Jc = self._jtab(a)
            lofL: IntArray = self._species_tables(a)["lofL"]
            IL: FloatArray = np.empty((nspins, tl["Lmax"], J.shape[-1]))
            ell: int
            for ell in range(tl["lmax"] + 1):
                Ls: slice = slice(ell * ell, (ell + 1) ** 2)
                IL[:, Ls] = np.einsum("sLq,qk->sLk", tl["D_sLq"][:, Ls], J[:, ell])
            IL[:, 0] += Jc / nspins
            per.append((a, IL, (IL[..., 1:] - IL[..., :-1]) / dg, lofL, self._phase1d(a)))
        CH: int = 32768
        gline: FloatArray = geo["gline"]
        c0: int
        for c0 in range(0, len(gidx), CH):
            c1: int = min(c0 + CH, len(gidx))
            hid: IntArray = gidx[c0:c1]
            i0: IntArray
            i1: IntArray
            i2: IntArray
            i0, i1, i2 = np.unravel_index(hid, geo["N"])
            tc: tuple[IntArray, FloatArray, FloatArray] | None = gc["tails_cache"].get(("d", c0, c1)) if cache else None
            if tc is None:
                Gv: FloatArray = self._gvec(hid)
                Gabs: FloatArray = np.sqrt((Gv**2).sum(1))
                j: IntArray = np.clip(np.searchsorted(gline, Gabs, side="right") - 1, 0, len(gline) - 2)
                tc = (j.astype(np.int32), Gabs - gline[j], np.ascontiguousarray(_ylm(Gv, geo["Lmax"]).T))
                del Gv, Gabs
                if cache and _gridcache_ok(gc, sum(x.nbytes for x in tc)):
                    gc["tails_cache"][("d", c0, c1)] = tc
            dx: FloatArray
            YGt: FloatArray
            j, dx, YGt = tc
            sl: FloatArray
            p1d: list[ComplexArray]
            for a, IL, sl, lofL, p1d in per:
                ph: ComplexArray = p1d[0][i0] * p1d[1][i1] * p1d[2][i2]
                s_: int
                for s_ in range(nspins):
                    Ich: FloatArray = sl[s_][:, j] * dx + IL[s_][:, j]
                    fr: FloatArray = np.zeros(c1 - c0)
                    fi: FloatArray = np.zeros(c1 - c0)
                    L: int
                    for L, ell in enumerate(lofL):
                        c: complex = 4 * PI * (-1j) ** ell
                        if ell % 2 == 0:
                            fr += (c.real * YGt[L]) * Ich[L]
                        else:
                            fi += (c.imag * YGt[L]) * Ich[L]
                    out[s_, c0:c1] += ph * (fr + 1j * fi)
        return out

    def _kernel_dom(
        self,
        e_x: FloatArray,
        n_x: FloatArray,
        v_x: FloatArray,
        sig_x: FloatArray,
        dsig_x: FloatArray,
        tau_x: FloatArray,
        dtau_x: FloatArray,
        *xargs: FloatArray,
    ) -> None:
        """self.kernel.calculate on the X slab of the distributed layout in chunks of KCHUNK points (all arguments are slab arrays (..., Pl))"""
        Pl: int = e_x.shape[-1]
        c0: int
        for c0 in range(0, Pl, KCHUNK):
            c1: int = min(c0 + KCHUNK, Pl)
            self.kernel.calculate(
                e_x[c0:c1],
                n_x[:, c0:c1],
                v_x[:, c0:c1],
                sig_x[:, c0:c1],
                dsig_x[:, c0:c1],
                tau_x[:, c0:c1],
                dtau_x[:, c0:c1],
                *[a[..., c0:c1] for a in xargs],
            )

    def _paw_layout_dom(self, par: _KPar) -> dict[str, Any]:
        """the one-centre layout of the distributed grid: as _paw_layout, the columns this rank's items read and the
        points they deposit to, both owned by the X-slab ranks; the column requests are exchanged once (serve: the local
        indices each rank sends to each requester)"""
        geo: dict[str, Any] = cast("dict[str, Any]", self._geom)
        dom: DomLayout = self._domL
        items: list[PawItem]
        owner: list[int]
        items, owner = self._paw_items(par)
        key: tuple[str | int, ...] = ("laydom",) + tuple(self._owner[0])
        if key in geo["pts"]:
            return cast("dict[str, Any]", geo["pts"][key])
        mine: list[PawItem] = [(a, r, d) for (a, r, d), rk in zip(items, owner) if rk == par.rank]
        cols: list[IntArray] | IntArray = [
            np.unique(self._readout_op(a, d, force=True, cache=False).indices) for a, r, d in mine
        ]
        cols = np.unique(np.concatenate(cols)).astype(np.int64) if cols else np.zeros(0, np.int64)
        U: list[IntArray] | IntArray = [self._splat_ops(a, d)[0] for a, r, d in mine]
        U = np.unique(np.concatenate(U)).astype(np.int64) if U else np.zeros(0, np.int64)
        segc: IntArray = np.searchsorted(cols, dom.bounds)
        serve: list[IntArray] = [
            x - dom.p0 for x in par.alltoallv_list([cols[segc[q] : segc[q + 1]] for q in range(par.size)])
        ]
        seg: IntArray = np.searchsorted(U, dom.bounds)
        recv: list[IntArray] = [
            x - dom.p0 for x in par.alltoallv_list([U[seg[q] : seg[q + 1]] for q in range(par.size)])
        ]
        pos: dict[tuple[int, tuple[int, ...] | None], IntArray] = {
            (a, None if d is None else tuple(d)): np.searchsorted(U, self._splat_ops(a, d)[0]) for a, r, d in mine
        }
        geo["pts"][key] = dict(mine=mine, cols=cols, segc=segc, serve=serve, U=U, seg=seg, recv=recv, pos=pos)
        return cast("dict[str, Any]", geo["pts"][key])

    def _field_fetch_dom(self, par: _KPar, lay: dict[str, Any], fields: FloatArray | None = None) -> FloatArray:
        """the prefiltered convolved fields (default: this update's _pref_dom) at this rank's column list, from the X-slab owners (one
        alltoallv) -> (ncols, items)"""
        if fields is None:
            fields = cast(FloatArray, self._pref_dom)
        nI: int = len(fields)
        parts: list[FloatArray] = [
            np.concatenate([fields[:, idx]]).ravel() if len(idx) else np.zeros(0) for idx in lay["serve"]
        ]
        got: list[FloatArray] = par.alltoallv_list(parts)
        out: FloatArray = np.empty((len(lay["cols"]), nI))
        q: int
        for q in range(par.size):
            a0: int
            a1: int
            a0, a1 = lay["segc"][q], lay["segc"][q + 1]
            if a1 > a0:
                out[a0:a1] = got[q].reshape(nI, a1 - a0).T
        return np.ascontiguousarray(out)

    def _tails_adjoint_dom_pass(self) -> None:
        """the tails adjoint of every atom with genuine tails (distributed grids): atoms round-robin over the ranks, the
        node-summed feedback potential's spline coefficients fetched at the stencil columns of their tails points, the
        cubic read-out as a CSR operator (map_coordinates' weights), the results summed over the ranks -> self._tadj,
        read by GPAW's per-atom calls"""
        geo: dict[str, Any] = cast("dict[str, Any]", self._geom)
        par: _KPar = self._dom_par
        nat: int = len(self.wfs.setups)
        atoms: list[int] = [a for a in range(nat) if not self._species_tables(a)["negl"]]
        self._tadj = {}
        if not atoms or getattr(self, "_vfb_pref_dom", None) is None:
            return
        import scipy.sparse as sps

        key: tuple[str, int, tuple[int, ...]] = ("taillay", par.size, tuple(atoms))
        lay: dict[str, Any] | None = geo["pts"].get(key)
        if lay is None:
            mine: list[int] = atoms[par.rank :: par.size]
            gops: dict[int, Any] = {a: self._cubic_csr(self._tail_coords(a)) for a in mine}
            cols: IntArray = (
                np.unique(np.concatenate([np.unique(o.indices) for o in gops.values()])).astype(np.int64)
                if mine
                else np.zeros(0, np.int64)
            )
            ops: dict[int, Any] = {
                a: sps.csr_matrix(
                    (o.data, np.searchsorted(cols, o.indices).astype(np.int32), o.indptr), shape=(o.shape[0], len(cols))
                )
                for a, o in gops.items()
            }
            del gops
            dom: DomLayout = self._domL
            segc: IntArray = np.searchsorted(cols, dom.bounds)
            serve: list[IntArray] = [
                x - dom.p0 for x in par.alltoallv_list([cols[segc[q] : segc[q + 1]] for q in range(par.size)])
            ]
            lay = geo["pts"][key] = cast("dict[str, Any]", dict(mine=mine, cols=cols, segc=segc, serve=serve, ops=ops))
        vC: FloatArray = self._field_fetch_dom(par, lay, fields=self._vfb_pref_dom)
        ns: int = vC.shape[1]
        sz: list[int] = [ns * len(self.wfs.setups[a].xc_correction.B_pqL) for a in range(nat)]
        off: IntArray = np.r_[0, np.cumsum(sz)]
        buf: FloatArray = np.zeros(int(off[-1]))
        nru: dict[int, int] = {a: len(self._species_tables(a)["rq"]) for a in lay["mine"]}
        a: int
        for a in lay["mine"]:
            v: FloatArray = lay["ops"][a] @ vC
            buf[off[a] : off[a + 1]] = np.array(
                [self._tails_adjoint_from(a, v[:, s_].reshape(len(weight_n), nru[a])) for s_ in range(ns)]
            ).ravel()
        self._domL.allsum(buf)
        self._tadj = {a: buf[off[a] : off[a + 1]].reshape(ns, -1).copy() for a in atoms}

    def _process_mgga_dom(
        self, e_g: FloatArray, nt_sg: FloatArray, v_sg: FloatArray, sigma_xg: FloatArray, dedsigma_xg: FloatArray
    ) -> None:
        """process_mgga on a distributed fine grid: GPAW's XC-grid blocks are redistributed to X slabs over the world
        communicator (dconv_dom.DomLayout; every rank, k-point / band replicas included, takes P / ranks points), the
        convolutions are distributed 3D FFTs, the spline prefilters reciprocal-space divisions, the pointwise work runs
        on the slab, the one-centre pass reads the fields through the column service and deposits to the slab owners,
        and the results go back to GPAW's blocks.  DCONV_SPLAT=cubic is not implemented here."""
        import time

        from .dconv_dom import DomLayout

        t0: float = time.time()
        self._t_tick = time.perf_counter()
        nspins: int = len(nt_sg)
        if os.environ.get("DCONV_SPLAT", "linear") == "cubic":
            raise NotImplementedError("DCONV on a distributed grid: DCONV_SPLAT=cubic is not implemented")
        if self.fixed_ke:
            taut_sG: FloatArray | None = self._fixed_taut_sG.copy()
            if not self._taut_gradv_init:
                self._taut_gradv_init = True
                self.wfs.calculate_kinetic_energy_density()
        else:
            taut_sG = self.wfs.calculate_kinetic_energy_density()
        if taut_sG is None:  # LCAO start: von Weizsaecker tau (as process_mgga)
            nt0_sg: FloatArray = nt_sg.copy()
            nt0_sg[nt0_sg < 1e-10] = np.inf
            tw_sg: FloatArray = sigma_xg[::2] / 8 / nt0_sg
            taut_sG = self.wfs.gd.empty(len(nt_sg))
            tw_g: FloatArray
            taut_G: FloatArray
            for tw_g, taut_G in zip(tw_sg, taut_sG):
                self.restrict_and_collect(tw_g, taut_G)
        taut_sg: FloatArray = np.empty_like(nt_sg)
        taut_g: FloatArray
        for taut_G, taut_g in zip(taut_sG, taut_sg):
            taut_G += 1.0 / self.wfs.nspins * self.tauct_G
            self.distribute_and_interpolate(taut_G, taut_g)
        gdx: Any = self.gd
        geo: dict[str, Any] = self._geometry(gdx, full=False)
        N: tuple[int, ...] = geo["N"]
        P: int = int(np.prod(N))
        world: Any = self.wfs.world
        lk: tuple[tuple[int, ...], int, int] = (N, id(gdx), world.size)
        if getattr(self, "_domL_key", None) != lk:
            self._domL = DomLayout(world, gdx, N)
            self._domL_key = lk
            self._dom_par = _KPar(world)
        dom: DomLayout
        par: _KPar
        dom, par = self._domL, self._dom_par
        Pl: int = dom.Pl
        self._slab = (dom.p0, dom.p1, True)
        # GPAW blocks -> X slabs (the first replica of each block: one input on every rank, as the k-parallel bcast of process_mgga)
        nt_x: FloatArray = dom.to_slab(nt_sg.reshape(nspins, *gdx.n_c))
        sig_x: FloatArray = dom.to_slab(sigma_xg.reshape(len(sigma_xg), *gdx.n_c))
        tau_x: FloatArray = dom.to_slab(taut_sg.reshape(nspins, *gdx.n_c))
        del taut_sg
        self._D_asp = self._gather_dasp_dom(dom)
        n_x: FloatArray = np.clip(nt_x, 0.0, None)
        self._tick("tau")
        self._tails = {a: self._onecentre_tails(a) for a in range(len(geo["pos"]))}
        nw: int = len(self.nodes)
        xoff: bool
        xoff = self._xoff = not self._x_on()  # no term reads the convolved descriptors
        # spin-polarised states of models that do not read the channel descriptors use only the total-density
        # descriptors, i.e. the sum of the two spins' node convolutions: the convolution of the total density (+ the
        # tails of the spin-summed D_sLq) is formed once (~1e-16 relative).  The adjoint (feedback, deposits, tails
        # adjoint) is per spin (both spins identical).  DCONV_SPINSUM=0: per-spin convolutions.
        spinsum: bool
        spinsum = self._spinsum = nspins == 2 and not xoff and self._spinsum_on()
        nch: int = 1 if spinsum else nspins
        Fsum: ComplexArray | None = None
        if not xoff and any(not self._species_tables(a)["negl"] for a in range(len(geo["pos"]))):
            saved: dict[int, dict[str, Any]] = self._tails
            if spinsum:
                self._tails = {a: dict(t, D_sLq=t["D_sLq"].sum(0, keepdims=True), nspins=1) for a, t in saved.items()}
            try:
                Fsum = self._tails_F_idx(dom.gidx())
            finally:
                self._tails = saved
            Fsum *= P / geo["vol"]
        items: list[tuple[int, int]] = [(s_, w) for s_ in range(nch) for w in range(nw)]
        S_x: FloatArray = np.empty((0 if xoff else len(items), Pl))
        pref_x: FloatArray = np.empty((0 if xoff else len(items), Pl))
        s_: int
        for s_ in range(0 if xoff else nch):
            mk: ComplexArray = dom.fft(n_x.sum(0) if spinsum else n_x[s_])
            if Fsum is not None:
                mk += Fsum[s_]
            w: int
            for w in range(nw):
                spec: ComplexArray = mk * self._dom_damp(w)
                S_x[s_ * nw + w] = np.real(dom.ifft(spec))
                pref_x[s_ * nw + w] = dom.prefilter(spec)
                del spec
            del mk
        del Fsum
        self._pref_dom = pref_x
        self._S_prefT = self._S_pref = None
        self._S_wg = self._S_swg = None
        del pref_x
        self._tick("tails+S")
        D: dict[str, Any] = (
            self._descriptors_off(n_x) if xoff else self._descriptors(n_x, S_x.reshape(nch, nw, Pl), keep_P=False)
        )
        del S_x
        K: int = len(self.cs)
        live: BoolArray = D["live"]
        dtau_x: FloatArray = np.zeros((nspins, Pl))
        dedx: FloatArray = np.zeros((K, Pl))
        e_x: FloatArray = np.zeros(Pl)
        v_x: FloatArray = np.zeros((nspins, Pl))
        dsig_x: FloatArray = np.zeros((len(sig_x), Pl))
        if nspins == 1:
            self._kernel_dom(e_x, nt_x, v_x, sig_x, dsig_x, tau_x, dtau_x, D["x_tot"], dedx)
            dedx_s: FloatArray | None = None
        else:
            dedx_s = np.zeros((K, 2, Pl))
            self._kernel_dom(e_x, nt_x, v_x, sig_x, dsig_x, tau_x, dtau_x, D["x_tot"], dedx, D["x_s"], dedx_s)
        self._check_skips(xoff, dedx, dedx_s, D)
        self._tick("kernel")
        if self.vv10:
            from .rvv10 import RVV10

            if self._rvv10 is None:
                self._rvv10 = RVV10(b=float(self.vv10.get("b", 10.0)))
            n_tot: FloatArray = np.clip(nt_x.sum(0), 0.0, None)
            sig_tot: FloatArray = sig_x[0] if nspins == 1 else sig_x[0] + 2.0 * sig_x[1] + sig_x[2]
            E_nl: float
            out_v: FloatArray
            E_nl, out_v = self._rvv10.calculate_dom(n_tot, np.clip(sig_tot, 0.0, None), gdx.cell_cv, dom)
            e_x += out_v[0]
            for s_ in range(nspins):
                v_x[s_] += out_v[1]
            if nspins == 1:
                dsig_x[0] += out_v[2]
            else:
                dsig_x[0] += out_v[2]
                dsig_x[1] += 2.0 * out_v[2]
                dsig_x[2] += out_v[2]
            self.timings["E_nl_vv10"] = E_nl
            del out_v
        self._tick("vv10")
        first: bool = self._deposit_wg is None and os.environ.get("DCONV_FIRST_SYNC", "1") != "0"
        sync: bool = (
            (bool(os.environ.get("DCONV_SYNC")) or first)
            and not os.environ.get("DCONV_NO_DEPOSIT")
            and getattr(self, "density", None) is not None
        )
        self._paw_cache = None
        if sync:
            self._deposit_new = None if xoff else np.zeros((nspins, nw, Pl))
            cache: PawCache = self._paw_pass(par)
            self._deposit_wg = self._deposit_new
            self._deposit_new = None
            self._paw_cache = cache
        self._tick("sync")
        if xoff:  # no descriptor feedback, no deposits, no tails adjoint
            vfb_x: FloatArray = np.zeros((nspins, Pl))
            self._vfb_swg = vfb_x[:, None, :]
            self._vfb_wg = self._vfb_swg[0]
            self._vfb_pref = None
            self._vfb_pref_dom = None
            self._deposit_new = None
            self._deposit_wg = _DEP_USED
        else:
            vx_s: FloatArray
            F_s: FloatArray
            vx_s, F_s = self._feedback_terms(n_x, dedx, dedx_s, D)
            F_s[:, :, ~live] = 0.0
            F_shape: tuple[int, ...] = F_s.shape
            if (
                getattr(self, "_deposit_new", None) is not None
                and cast(FloatArray, self._deposit_new).shape == F_s.shape
            ):
                self._deposit_wg = self._deposit_new
            if (
                getattr(self, "_deposit_wg", None) is not None
                and cast(FloatArray, self._deposit_wg).shape == F_s.shape
                and not os.environ.get("DCONV_NO_DEPOSIT")
            ):
                F_s += cast(FloatArray, self._deposit_wg)
            if (
                not sync
                and isinstance(getattr(self, "_deposit_wg", None), np.ndarray)
                and cast(FloatArray, self._deposit_wg).size
            ):
                self._deposit_wg = _DEP_USED
            vfb_x = np.empty((nspins, Pl))
            self._vfb_pref_dom = np.empty((nspins, Pl))
            # spin runs without channel descriptors have the same feedback sources in both channels (the grid sources of
            # the total descriptors and identical sphere deposits): the second channel's convolutions are a copy
            # (DCONV_SPIN_DEDUP=0: computed)
            same: bool = (
                nspins == 2
                and D.get("dx_s") is None
                and os.environ.get("DCONV_SPIN_DEDUP", "1") != "0"
                and np.array_equal(F_s[0], F_s[1])
            )
            for s_ in range(nspins):  # node-summed feedback spectrum on the Y slab (sum in node order, as process_mgga)
                if same and s_ == 1:
                    vfb_x[1] = vfb_x[0]
                    self._vfb_pref_dom[1] = self._vfb_pref_dom[0]
                    continue
                spec = np.zeros(dom.PlG, complex)
                for w in range(nw):
                    spec += dom.fft(F_s[s_, w]) * self._dom_damp(w)
                vfb_x[s_] = np.real(dom.ifft(spec))
                self._vfb_pref_dom[s_] = dom.prefilter(spec)
                del spec
            del F_s
            self._vfb_swg = vfb_x[:, None, :]
            self._vfb_wg = self._vfb_swg[0]
            self._vfb_pref = None
            self._deposit_new = None if sync else np.zeros(F_shape)
            for s_ in range(nspins):
                v_x[s_] += vx_s[s_] + vfb_x[s_]
        self._tick("feedback")
        # X slabs -> GPAW's blocks (every replica)
        e_g[...] = dom.to_blocks(e_x[None])[0].reshape(e_g.shape)
        v_sg += dom.to_blocks(v_x).reshape(v_sg.shape)
        dedsigma_xg[...] = dom.to_blocks(dsig_x).reshape(dedsigma_xg.shape)
        dedtaut_sg: FloatArray = dom.to_blocks(dtau_x)
        del e_x, v_x, dsig_x, dtau_x
        self.dedtaut_sG = self.wfs.gd.empty(self.wfs.nspins)
        self.ekin = 0.0
        for s_ in range(self.wfs.nspins):
            self.restrict_and_collect(dedtaut_sg[s_], self.dedtaut_sG[s_])
            self.ekin -= self.wfs.gd.integrate(self.dedtaut_sG[s_] * (taut_sG[s_] - self.tauct_G / self.wfs.nspins))
        del dedtaut_sg
        self._tick("restrict")
        self.timings["grid"] = time.time() - t0
        live_n: FloatArray = dom.allsum([float(live.sum()), float((vfb_x.sum(0)[live] ** 2).sum())])
        self._x_stats = dict(x_med=None, vfb_rms=float(np.sqrt(live_n[1] / max(live_n[0], 1.0))))
        del D, vfb_x
        if not sync and getattr(self, "density", None) is not None and not os.environ.get("DCONV_NO_DEPOSIT"):
            t1: float = time.time()
            self._paw_cache = self._paw_pass(par)
            self.timings["paw_pass"] = time.time() - t1
            self._tick("paw_pass")
        self._pref_dom = None
        self._tails_adjoint_dom_pass()
        self._tick("tails_adj")

    # ------------------------------------------------------------------ PAW part (one-centre, sphere -> grid feedback)
    def _setup_tables(self, a: int) -> tuple[FloatArray, FloatArray, FloatArray, float]:
        """T[w, l, q, g] = (g_{l, w} * Delta_q)(r_g) for the AE route (Delta = n_q - nt_q) and the PS route, per l,
        cached per setup symbol (independent of D); see _species_tables"""
        st: dict[str, Any] = self._species_kprod(a)
        return st["T_ae"], st["T_ps"], st["ru"], st["rcut"]

    def _points(self, a: int, dirs: Dirs = None) -> tuple[FloatArray, tuple[int, ...]]:
        """fractional grid coordinates (3, n_dir ng) of atom a's one-centre points (direction n, radial g) and their
        shape; cached per geometry.  dirs = (n0, n1): the points of the Lebedev directions n0 <= n < n1 only
        (direction-chunk work items)"""
        geo: dict[str, Any] = cast("dict[str, Any]", self._geom)
        if a not in geo["pts"]:
            r_g: FloatArray = self.wfs.setups[a].xc_correction.rgd.r_g
            pts: FloatArray = geo["pos"][a][None, None, :] + r_g[None, :, None] * R_nv[:, None, :]  # (n, g, 3)
            geo["pts"][a] = ((pts.reshape(-1, 3) @ geo["inv"] * np.array(geo["N"])).T, pts.shape[:2])
        if dirs is None or tuple(dirs) == (0, geo["pts"][a][1][0]):
            return cast("tuple[FloatArray, tuple[int, ...]]", geo["pts"][a])
        c: FloatArray
        nd: int
        ng: int
        c, (nd, ng) = geo["pts"][a]
        return c[:, dirs[0] * ng : dirs[1] * ng], (dirs[1] - dirs[0], ng)

    def _readout_budget(self) -> float:
        par: _KPar | None = getattr(self, "_par", None)
        return (
            float(os.environ.get("DCONV_READOUT_MB", "2048")) * 2**20 / _local_ranks(par.size if par is not None else 1)
        )

    def _readout_used(self) -> int:
        return cast(
            int,
            sum(
                v.data.nbytes + v.indices.nbytes
                for k, v in cast("dict[str, Any]", self._geom)["pts"].items()
                if isinstance(k, tuple) and k[0] in ("rd", "rdl") and v is not None
            ),
        )

    def _readout_op_local(self, a: int, dirs: Dirs, lay: dict[str, Any]) -> Any:
        """the read-out operator of _readout_op with its columns renumbered to the rank's column list lay["cols"]
        (distributed layout); cached within the per-node budget, else rebuilt per call"""
        geo: dict[str, Any] = cast("dict[str, Any]", self._geom)
        ck: tuple[str, int, tuple[int, ...] | None] = ("rdl", a, None if dirs is None else tuple(dirs))
        if ck in geo["pts"]:
            return geo["pts"][ck]
        import scipy.sparse as sps

        op: Any = self._readout_op(a, dirs, force=True, cache=False)
        li: IntArray = (
            lay["g2l"][op.indices] if "g2l" in lay else np.searchsorted(lay["cols"], op.indices).astype(np.int32)
        )  # (no grid-sized map)
        loc: Any = sps.csr_matrix((op.data, li, op.indptr), shape=(op.shape[0], len(lay["cols"])))
        if self._readout_used() + loc.data.nbytes + loc.indices.nbytes <= self._readout_budget():
            geo["pts"][ck] = loc
        return loc

    def _readout_op(self, a: int, dirs: Dirs = None, force: bool = False, cache: bool = True) -> Any:
        """sparse (CSR) operator of the cubic B-spline read-out at atom a's one-centre points: row = point, the 64
        weights w_x w_y w_z of map_coordinates(order=3, prefilter=False, mode="grid-wrap") on the periodic grid; one
        sparse x dense product reads every prefiltered field (same values up to the summation order, ~1e-16).  Cached
        per geometry while the cached operators stay within DCONV_READOUT_MB (default 2048) per node, divided among the
        node's ranks; None beyond that or with DCONV_READOUT=0 (the callers then use map_coordinates).  force=True:
        built in any case (cached only within the budget)."""
        if os.environ.get("DCONV_READOUT", "1") == "0" and not force:
            return None
        geo: dict[str, Any] = cast("dict[str, Any]", self._geom)
        ck: tuple[str, int, tuple[int, ...] | None] = ("rd", a, None if dirs is None else tuple(dirs))
        if ck in geo["pts"]:
            return geo["pts"][ck]
        coords: FloatArray
        coords, _ = self._points(a, dirs)
        n: int = coords.shape[1]
        fits: bool = self._readout_used() + 12 * 64 * n <= self._readout_budget()
        if not fits and not force:
            return None
        op: Any = self._cubic_csr(coords)
        if cache and fits:
            geo["pts"][ck] = op
        return op

    @staticmethod
    def _cubic_wi(coords: FloatArray, N: Sequence[int]) -> tuple[list[FloatArray], list[IntArray]]:
        """per-axis weights W_c (n, 4) and periodic grid indices I_c (n, 4) of the cubic B-spline read-out at fractional
        grid coordinates (3, n): the factors of _cubic_csr's operator"""
        W: list[FloatArray]
        ii: list[IntArray]
        W, ii = [], []
        d: int
        for d in range(3):
            fl: FloatArray = np.floor(coords[d])
            t: FloatArray = coords[d] - fl
            t2: FloatArray
            t3: FloatArray
            t2, t3 = t * t, t * t * t
            W.append(
                np.stack(
                    [
                        (1.0 - t) ** 3 / 6.0,
                        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
                        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
                        t3 / 6.0,
                    ],
                    1,
                )
            )
            ii.append(
                (fl.astype(np.int64)[:, None] + np.arange(-1, 3)[None, :]) % N[d]
            )  # nodes floor(x) - 1 .. floor(x) + 2
        return W, ii

    def _cubic_csr(self, coords: FloatArray) -> Any:
        """CSR operator (n, P) of the cubic B-spline read-out (map_coordinates order 3, prefilter False, grid-wrap) at fractional grid
        coordinates (3, n)"""
        import scipy.sparse as sps

        geo: dict[str, Any] = cast("dict[str, Any]", self._geom)
        n: int = coords.shape[1]
        N: tuple[int, ...] = geo["N"]
        W: list[FloatArray]
        ii: list[IntArray]
        W, ii = self._cubic_wi(coords, N)
        val: FloatArray = (W[0][:, :, None, None] * W[1][:, None, :, None] * W[2][:, None, None, :]).reshape(-1)
        idx: IntArray = (
            ((ii[0][:, :, None, None] * N[1] + ii[1][:, None, :, None]) * N[2] + ii[2][:, None, None, :])
            .reshape(-1)
            .astype(np.int32)
        )
        return sps.csr_matrix((val, idx, np.arange(0, 64 * n + 1, 64, dtype=np.int64)), shape=(n, int(np.prod(N))))

    def _read(self, a: int, F: str | FloatArray | Sequence[FloatArray], dirs: Dirs = None) -> FloatArray:
        """prefiltered periodic grid fields F (a list of N-shaped arrays, a (P, k) stack, or _PREF: the convolved fields
        of all (spin, node) items) at atom a's one-centre points -> (k, n_pts): the sparse read-out operator
        (_readout_op) or, without it, map_coordinates per field.  Distributed layout (inside the k-parallel one-centre
        pass): the local-column operator and the fields at the rank's column list (_PREF: fetched by _pref_fetch)."""
        lay: dict[str, Any] | None = getattr(self, "_lay", None)
        if lay is not None:
            op: Any = self._readout_op_local(a, dirs, lay)
            Fl: FloatArray | None = (
                self._prefC if F is _PREF else np.stack([np.asarray(f).reshape(-1)[lay["cols"]] for f in F], 1)
            )
            return np.ascontiguousarray((op @ Fl).T)
        if F is _PREF:
            if self._S_prefT is None:
                raise RuntimeError(
                    "DCONV (distributed layout): one-centre evaluation outside this update's one-centre pass (GPAW called "
                    "calculate_paw_correction with a D_sp that differs from the update's); k-point parallel runs: set DCONV_DIST=0 "
                    "for the replicated layout (domain-decomposed runs have no replicated layout)"
                )
            F = self._S_prefT
        op = self._readout_op(a, dirs)
        if op is None:
            coords: FloatArray
            coords, _ = self._points(a, dirs)
            if isinstance(F, np.ndarray) and F.ndim == 2:
                F = [F[:, i].reshape(cast("dict[str, Any]", self._geom)["N"]) for i in range(F.shape[1])]
            return np.array([map_coordinates(f, coords, order=3, mode="grid-wrap", prefilter=False) for f in F])
        if not (isinstance(F, np.ndarray) and F.ndim == 2):
            F = np.stack([np.asarray(f).reshape(-1) for f in F], 1)
        return np.ascontiguousarray((op @ F).T)

    def _onecentre_fields(self, a: int, ae: bool = True, dirs: Dirs = None) -> tuple[FloatArray | None, FloatArray]:
        """convolved fields at the one-centre points (direction n, radial g) per spin: AE and PS, (nspins, K, n_dir, ng).
        AE = PS + sum_L Y_L(n) (g_{l,w} * (Delta n_L - pseudized Delta n_L))(r_g), the radial kernel products from the species table T_d (linear in D)
        """
        tails: dict[str, Any] = self._tails[a]
        nspins: int = tails["nspins"]
        if (
            getattr(self, "_spinsum", False) and nspins == 2
        ):  # node fields of the total density only -> (1, K, n_dir, ng)
            nspins = 1
        shp: tuple[int, ...]
        _, shp = self._points(a, dirs)
        C_ps: FloatArray = self._read(a, _PREF, dirs).reshape(
            (nspins, len(self.nodes)) + tuple(shp)
        )  # all spins and nodes in one product
        if not ae:
            return None, C_ps
        st: dict[str, Any] = self._species_kprod(a)
        D_sLq: FloatArray = tails["D_sLq"] if nspins == tails["nspins"] else tails["D_sLq"].sum(0, keepdims=True)
        fL: FloatArray = np.empty((nspins, len(self.nodes), tails["Lmax"], st["T_ae"].shape[-1]))
        ell: int
        for ell in range(tails["lmax"] + 1):
            Ls: slice = slice(ell * ell, (ell + 1) ** 2)
            fL[:, :, Ls] = np.matmul(
                D_sLq[:, None, Ls], (st["T_ae"][:, ell] - st["T_ps"][:, ell])[None]
            )  # (s, w, L, g)
        fL[:, :, 0] += st["Tc"][None] / nspins
        C_ae: FloatArray = C_ps + np.matmul(Y_nL[slice(*dirs) if dirs else slice(None), : tails["Lmax"]], fL)
        return C_ae, C_ps

    def calculate_paw_correction(
        self,
        setup: Any,
        D_sp: FloatArray,
        dEdD_sp: FloatArray | None = None,
        addcoredensity: bool = True,
        a: int | None = None,
        _core_only: bool = False,
        _routes: tuple[bool, ...] = (True, False),
        _dirs: Dirs = None,
    ) -> float:
        """one-centre (PAW) correction of atom a; _routes (k-parallel one-centre pass): evaluate only the AE (True)
        and / or PS (False) route -- energy, dE/dD and deposits are additive over the routes (e_xc0 is subtracted with
        the AE route); _dirs = (n0, n1): only the Lebedev directions n0 <= n < n1 (additive over a partition of the
        directions; e_xc0 with the chunk n0 = 0)"""
        import time

        t0: float = time.time()
        xcc: Any = setup.xc_correction
        if xcc is None:
            return 0.0
        cache: PawCache | None = getattr(self, "_paw_cache", None)
        miss: bool = False
        if (
            not _core_only and addcoredensity and cache and a in cache
        ):  # computed in this update's synchronous pre-pass (process_mgga)
            D0: FloatArray
            E0: float
            dEdD0: FloatArray
            D0, E0, dEdD0 = cache[a]
            same: bool = D0.shape == np.shape(D_sp) and np.array_equal(D0, D_sp)
            if not same and getattr(self, "_D_asp", None) is not None and D0.shape == np.shape(D_sp):
                # k-parallel: GPAW's per-atom D_sp may be another rank's copy, equal up to the last bits (see process_mgga)
                same = float(np.abs(D0 - np.asarray(D_sp)).max()) <= 1e-12 * max(1.0, float(np.abs(D0).max()))
            miss = not same
            if same:
                if dEdD_sp is not None:
                    dEdD_sp += dEdD0
                    s_: int
                    for s_ in range(len(D_sp)):
                        if not getattr(self, "_xoff", False):  # no feedback potential without descriptors
                            dEdD_sp[s_] += self._tails_adjoint(a, s_)
                    if not hasattr(self, "dEdD_asp"):
                        self.dEdD_asp = {}
                    self.dEdD_asp[a] = np.zeros_like(dEdD_sp)
                return E0
        if miss and getattr(self, "_D_asp", None) is not None:
            # k-parallel: D_sp changed within the update -> evaluate afresh but leave this update's (complete, rank-summed) deposits alone
            saved: tuple[PawCache | None, FloatArray | None] = (self._paw_cache, getattr(self, "_deposit_new", None))
            self._paw_cache = self._deposit_new = None
            try:
                return self.calculate_paw_correction(setup, D_sp, dEdD_sp, addcoredensity, a, _core_only)
            finally:
                self._paw_cache, self._deposit_new = saved
        nspins: int = len(D_sp)
        rgd: Any = xcc.rgd
        r_g: FloatArray
        dv_g: FloatArray
        r_g, dv_g = rgd.r_g, rgd.dv_g
        ng: int = len(r_g)
        Lmax: int = xcc.Lmax
        dsl: slice = slice(*_dirs) if _dirs is not None else slice(None)  # direction chunk
        Y: FloatArray = Y_nL[dsl, :Lmax]
        rnY: FloatArray = rnablaY_nLv[dsl, :Lmax]
        wn: FloatArray = weight_n[dsl]
        nd: int = len(wn)
        kin: dict[str, Any] = self._kinetic_tables(
            cast(int, a)
        )  # separable one-centre kinetic tables (GPAW's tau_npg / taut_npg are not built)
        D_sLq: FloatArray = np.inner(D_sp, xcc.B_pqL.T)  # (nspins, Lmax, nq)
        xoff: bool = getattr(self, "_xoff", False)  # no descriptor fields / feedback when no term reads x
        C_ae: FloatArray | None
        C_ps: FloatArray | None
        C_ae, C_ps = (
            (None, None) if xoff else self._onecentre_fields(cast(int, a), ae=True in _routes, dirs=_dirs)
        )  # (nspins, K, nd, ng)
        T_ae: FloatArray | None
        T_ps: FloatArray | None
        ru: FloatArray | None
        rcut: float | None
        T_ae, T_ps, ru, rcut = (None, None, None, None) if xoff else self._setup_tables(cast(int, a))
        K: int = len(self.nodes)
        fac: float = math.sqrt(4 * PI)
        E: float = 0.0
        dEdD_sqL: FloatArray = np.zeros((nspins, xcc.n_qg.shape[0], Lmax))
        dEdD_sp_tau: FloatArray = np.zeros((nspins, len(D_sp[0])))
        dep_acc: dict[int, FloatArray] = (
            {}
        )  # AE + PS point weights of the sphere -> grid deposit (same points): one splat per spin
        ae: bool
        for ae in _routes:
            n_qg: FloatArray = xcc.n_qg if ae else xcc.nt_qg
            nc: FloatArray | float = (xcc.nc_g if ae else xcc.nct_g) if addcoredensity else 0.0
            kC: FloatArray
            kA: FloatArray
            kRD: FloatArray
            kRP: FloatArray
            kC, kA, (kRD, kRP) = kin["C"][dsl], kin["A"][dsl], kin["R"][ae]
            tauc: FloatArray = (xcc.tauc_g if ae else xcc.tauct_g) / (fac * nspins)
            sign: float = 1.0 if ae else -1.0
            C: FloatArray | None = C_ae if ae else C_ps  # (nspins, K, nd, ng)
            T: FloatArray | None = T_ae if ae else T_ps
            n_sLg: FloatArray = np.matmul(D_sLq, n_qg)
            n_sLg[:, 0] += fac * nc / nspins
            dndr_sLg: FloatArray = np.array(
                [[rgd.derivative(n_sLg[s_, L]) for L in range(Lmax)] for s_ in range(nspins)]
            )
            n_sng: FloatArray = np.matmul(Y, n_sLg)
            a_sng: FloatArray = np.matmul(Y, dndr_sLg)
            b_snvg: FloatArray = np.matmul(rnY.transpose(0, 2, 1)[None], n_sLg[:, None])
            sig_xng: FloatArray = np.empty((2 * nspins - 1, nd, ng))
            for s_ in range(nspins):
                sg: FloatArray = (b_snvg[s_] ** 2).sum(1)
                sg[:, 1:] /= r_g[1:] ** 2
                sg[:, 0] = sg[:, 1]
                sig_xng[2 * s_] = sg + a_sng[s_] ** 2
            if nspins == 2:
                sg = (b_snvg[0] * b_snvg[1]).sum(1)
                sg[:, 1:] /= r_g[1:] ** 2
                sg[:, 0] = sg[:, 1]
                sig_xng[1] = sg + a_sng[0] * a_sng[1]
            tau_sng: FloatArray = self._kinetic_tau(D_sp, kC, kA, kRD, kRP) + tauc[None, None, :]
            n_s: FloatArray = np.clip(n_sng.reshape(nspins, -1), 0.0, None)
            Dd: dict[str, Any] = (
                self._descriptors_off(n_s)
                if xoff
                else self._descriptors(
                    n_s, cast(FloatArray, C).reshape(cast(FloatArray, C).shape[0], K, -1), keep_P=True
                )
            )
            ns: FloatArray = np.maximum(n_s.sum(0), 1e-10)
            if nspins == 1:
                t: dict[str, FloatArray] = self.kernel_terms(
                    n_s[0], sig_xng[0].ravel(), tau_sng[0].ravel(), Dd["x_tot"]
                )
                dedn_s: FloatArray = t["dedn"][None, :] + _einsum("cp,cp->p", t["dedx"], Dd["dx_tot"])[None, :]
                dedsig_x: FloatArray = t["dedsigma"][None, :]
                dedtau_s: FloatArray = t["dedtau"][None, :]
                self._check_skips(xoff, t["dedx"], None, Dd)
                Phi_s: FloatArray | None = (
                    None if xoff else self._phi_adj(n_s.sum(0), t["dedx"] / ns[None, :], Dd["P_tot"])[None]
                )  # (nspins, K, P): dE/dC_s,w per point
            else:
                t = self.kernel_terms_spin(n_s, sig_xng.reshape(3, -1), tau_sng.reshape(2, -1), Dd["x_s"], Dd["x_tot"])
                vtot: FloatArray = _einsum("cp,cp->p", t["dedx_tot"], Dd["dx_tot"])
                dedn_s = t["dedn"] + vtot[None, :]
                dedsig_x = t["dedsigma"]
                dedtau_s = t["dedtau"]
                self._check_skips(xoff, t["dedx_tot"], t["dedx_s"], Dd)
                # the kernel reads the total-density descriptors only (no channel descriptors): the same adjoint for both spins
                Phi_tot: FloatArray | None = (
                    None if xoff else self._phi_adj(n_s.sum(0), t["dedx_tot"] / ns[None, :], Dd["P_tot"])
                )
                Phi_s = None if xoff else np.array([Phi_tot, Phi_tot])
            e_ng: FloatArray = t["e"].reshape(nd, ng)
            E += sign * float(np.sum(wn[:, None] * dv_g[None, :] * e_ng))
            if dEdD_sp is not None:
                dedn_sng: FloatArray = dedn_s.reshape(nspins, nd, ng)
                dedsig_xng: FloatArray = dedsig_x.reshape(2 * nspins - 1, nd, ng)
                dedtau_sng: FloatArray = dedtau_s.reshape(nspins, nd, ng)
                # dE/dD_p through tau for both spins: sum_ng tau_npg (w_n dv_g dE/dtau_sng), with the separable tables
                Xt: FloatArray = np.matmul(
                    wn[None, :, None] * dv_g[None, None, :] * dedtau_sng, np.ascontiguousarray(kRD.T)
                )  # (s, n, p)
                Xp: FloatArray = np.matmul(
                    wn[None, :, None] * dv_g[None, None, :] * dedtau_sng, np.ascontiguousarray(kRP.T)
                )
                dtau_ps: FloatArray = (0.5 * ((Xt * kC[None]).sum(1) + (Xp * kA[None]).sum(1))).T
                del Xt, Xp
                for s_ in range(nspins):
                    # radial part of the gradient correction (as gga.add_radial_gradient_correction), per direction
                    vv: FloatArray = np.empty((nd, ng))
                    n: int
                    for n in range(nd):
                        tmp: FloatArray = np.empty(ng)
                        rgd.derivative2(-2 * dv_g * dedsig_xng[2 * s_, n] * a_sng[s_, n], tmp)
                        if nspins == 2:
                            tmp2: FloatArray = np.empty(ng)
                            rgd.derivative2(dv_g * dedsig_xng[1, n] * a_sng[1 - s_, n], tmp2)
                            tmp -= tmp2
                        tmp[1:] /= dv_g[1:]
                        tmp[0] = tmp[1]
                        vv[n] = tmp
                    dEdD_sqL[s_] += sign * _einsum("n,nL,ng,qg->qL", wn, Y, dv_g[None, :] * (dedn_sng[s_] + vv), n_qg)
                    B_nvg: FloatArray = dedsig_xng[2 * s_][:, None, :] * b_snvg[s_]
                    if nspins == 2:
                        B_nvg = B_nvg + 0.5 * dedsig_xng[1][:, None, :] * b_snvg[1 - s_]
                    B_nvq: FloatArray = np.matmul(B_nvg * rgd.dr_g[None, None, :], n_qg.T)
                    dEdD_sqL[s_] += sign * 8 * PI * _einsum("n,nLv,nvq->qL", wn, rnY, B_nvq)
                    dEdD_sp_tau[s_] += sign * dtau_ps[:, s_]
                    if Phi_s is None:  # no descriptors read (dE/dx = 0): no own-atom non-local route, no deposit
                        continue
                    # own-atom non-local route: dE/dD_s through C_s (AE: full Delta, PS: pseudized Delta); T per l
                    Phi: FloatArray = Phi_s[s_].reshape(K, nd, ng)
                    Phi_wLg: FloatArray = _einsum("n,nL,kng->kLg", wn, Y, Phi) * dv_g
                    ell: int
                    for ell in (
                        range(cast(FloatArray, T).shape[1]) if self._own_route() else ()
                    ):  # off by default: the deposit route already contains it
                        Ls: slice = slice(ell * ell, (ell + 1) ** 2)
                        dEdD_sqL[s_][:, Ls] += sign * np.tensordot(
                            cast(FloatArray, T)[:, ell], Phi_wLg[:, Ls], axes=([0, 2], [0, 2])
                        )
                    # sphere -> grid feedback through S~_s at the points: deposit the point weights (AE + PS summed, splatted below)
                    Wd: FloatArray = sign * wn[:, None] * dv_g[None, :] * Phi
                    dep_acc[s_] = Wd if s_ not in dep_acc else dep_acc[s_] + Wd
        for s_, Wd in dep_acc.items():
            self._deposit_points(cast(int, a), s_, Wd, _dirs)
        if dEdD_sp is not None:
            for s_ in range(nspins):
                dEdD_sp[s_] += np.inner(dEdD_sqL[s_].reshape(-1), xcc.B_pqL.reshape((len(xcc.B_pqL), -1)))
                dEdD_sp[s_] += dEdD_sp_tau[s_]
                # grid energy's dependence on D_s through the tails: dE_grid/dD_s = sum_w <vfb_s,w, d(Delta m~_s)/dD_s>
                # (not in the synchronous pre-pass: the feedback potential vfb of this update does not exist yet; added
                # on the cache hit)
                if not _core_only and not xoff:
                    dEdD_sp[s_] += self._tails_adjoint(cast(int, a), s_)
            if not hasattr(self, "dEdD_asp"):
                self.dEdD_asp = {}
            self.dEdD_asp[cast(int, a)] = np.zeros_like(dEdD_sp)
        self.timings["paw"] = self.timings.get("paw", 0.0) + time.time() - t0
        return E - (xcc.e_xc0 if (addcoredensity and True in _routes and (_dirs is None or _dirs[0] == 0)) else 0.0)

    def _kinetic_tables(self, a: int) -> dict[str, Any]:
        """separable form of GPAW's one-centre kinetic table (mgga.create_kinetic) of atom a's setup, per pair
        p = (i1 <= i2) of partial waves (j, L): tau_npg = 1/2 [C_np dphi_j1 dphi_j2 (r_g) + A_np phi_j1 phi_j2 / r_g^2]
        with C_np = Y_L1 Y_L2 (n), A_np = sum_v rnablaY_L1v rnablaY_L2v (n) (r_0 -> r_1 as GPAW): (n_dir x n_p) angular
        and (n_p x ng) radial tables per route, equal to GPAW's tau_npg / taut_npg up to the association of the factors
        (~1e-16).  Cached per setup."""
        setup: Any = self.wfs.setups[a]
        xcc: Any = setup.xc_correction
        key: TabKey = ("kin", setup.symbol, len(xcc.rgd.r_g))
        if key in self._tabs:
            return self._tabs[key]
        jlL: list[tuple[int, int, int]] = xcc.jlL
        pr: list[tuple[int, int]] = [(i1, i2) for i1 in range(len(jlL)) for i2 in range(i1, len(jlL))]
        L1: IntArray = np.array([jlL[i1][2] for i1, _ in pr])
        L2: IntArray = np.array([jlL[i2][2] for _, i2 in pr])
        j1: IntArray = np.array([jlL[i1][0] for i1, _ in pr])
        j2: IntArray = np.array([jlL[i2][0] for _, i2 in pr])
        Y: FloatArray = xcc.Y_nL
        rY: FloatArray = xcc.rnablaY_nLv[:, : xcc.Lmax]
        C: FloatArray = Y[:, L1] * Y[:, L2]
        A: FloatArray = cast(
            FloatArray, rY[:, L1, 0] * rY[:, L2, 0] + rY[:, L1, 1] * rY[:, L2, 1] + rY[:, L1, 2] * rY[:, L2, 2]
        )
        r_g: FloatArray = xcc.rgd.r_g
        R: dict[bool, tuple[FloatArray, FloatArray]] = {}
        ae: bool
        phi_jg: FloatArray
        for ae, phi_jg in ((True, xcc.phi_jg), (False, xcc.phit_jg)):
            phi: FloatArray = np.asarray(phi_jg)
            dphi: FloatArray = np.zeros(phi.shape)
            j: int
            for j in range(len(phi)):
                xcc.rgd.derivative(phi[j], dphi[j])
            RD: FloatArray = dphi[j1] * dphi[j2]
            RP: FloatArray = phi[j1] * phi[j2]
            RP[:, 1:] /= r_g[1:] ** 2
            RP[:, 0] = RP[:, 1]
            R[ae] = (RD, RP)
        self._tabs[key] = dict(C=C, A=A, R=R)
        return self._tabs[key]

    @staticmethod
    def _kinetic_tau(D_sp: FloatArray, C: FloatArray, A: FloatArray, RD: FloatArray, RP: FloatArray) -> FloatArray:
        """sum_p D_sp tau_npg -> (nspins, n_dir, ng) from the separable tables (_kinetic_tables)"""
        D: FloatArray = np.asarray(D_sp)
        return cast(FloatArray, 0.5 * (np.matmul(D[:, None, :] * C[None], RD) + np.matmul(D[:, None, :] * A[None], RP)))

    def kernel_terms(self, n: FloatArray, sigma: FloatArray, tau: FloatArray, x: FloatArray) -> dict[str, FloatArray]:
        st: ModuleType
        _, st = _kernel_modules()
        k: DConvKernel = self.kernel
        return cast(
            "dict[str, FloatArray]", st.constrained_terms(k.model, k.base, n, sigma, tau, x, base_eval=k.base_eval())
        )

    def kernel_terms_spin(
        self, n_s: FloatArray, sig_s: FloatArray, tau_s: FloatArray, x_s: FloatArray, x_tot: FloatArray
    ) -> dict[str, FloatArray]:
        st: ModuleType
        _, st = _kernel_modules()
        k: DConvKernel = self.kernel
        return cast(
            "dict[str, FloatArray]",
            st.constrained_terms_spin(k.model, k.base, n_s, sig_s, tau_s, x_s, x_tot, base_eval=k.base_eval_spin()),
        )

    def _tick(self, label: str) -> None:
        """phase timer of process_mgga: self.timings['phases'][label] += seconds since the previous tick (cumulative over updates)"""
        import time

        t: float = time.perf_counter()
        ph: dict[str, float] = self.timings.setdefault("phases", {})
        ph[label] = ph.get(label, 0.0) + t - getattr(self, "_t_tick", t)
        self._t_tick = t

    def _dasp(self, a: int) -> FloatArray:
        """D_sp of atom a for this update: rank 0's copy in k-parallel runs (see process_mgga), GPAW's density.D_asp otherwise"""
        d: dict[int, FloatArray] | None = getattr(self, "_D_asp", None)
        return d[a] if d is not None else self.density.D_asp[a]

    def _kpar(self) -> _KPar | None:
        """None (serial) or the k-point parallel work split (_KPar); the grid part needs the whole grid on every rank,
        i.e. parallel={'domain': 1, 'band': 1}"""
        w: Any = self.wfs.world
        if w.size == 1:
            return None
        if self.wfs.gd.comm.size != 1 or self.wfs.bd.comm.size != 1:
            raise RuntimeError(
                "DensityConvMGGA in parallel needs parallel={'domain': 1, 'band': 1} (k-point parallelisation only)"
            )
        if getattr(self, "_par", None) is None or self._par.comm is not w:
            self._par = _KPar(w)
        return self._par

    def _kernel_calculate(self, par: _KPar | None, *args: FloatArray, xoff: int = 0) -> None:
        """self.kernel.calculate (pointwise) on this rank's contiguous point range (all points in serial runs) in chunks
        of KCHUNK points; k-parallel: the outputs (positional 0, 2, 4, 6, and 8, 10 when the descriptor arguments cover
        the whole grid) all-gathered.  The descriptor arguments (positional 7-10: x, dE/dx, x_s, dE/dx_s) are (..., n)
        arrays of the points xoff ... xoff + n (the rank's slab in the distributed layout)."""
        N: tuple[int, ...] = args[0].shape
        P: int = args[0].size
        p0: int
        p1: int
        p0, p1 = (0, P) if par is None else par.rng(P)

        def flat(a: FloatArray) -> FloatArray:
            return a.reshape(a.shape[: a.ndim - len(N)] + (P,))

        xfull: bool = len(args) > 7 and args[7].shape[-1] == P
        outs: list[FloatArray] = [args[i] for i in (0, 2, 4, 6) if i < len(args)]
        outs_x: list[FloatArray] = [
            args[i] for i in (8, 10) if xfull and i < len(args)
        ]  # (..., P) descriptor gradients when they cover the whole grid
        fargs: list[FloatArray] = [flat(a) if i < 7 else a for i, a in enumerate(args)]
        c0: int
        for c0 in range(p0, p1, KCHUNK):
            c1: int = min(c0 + KCHUNK, p1)
            self.kernel.calculate(
                *[a[..., c0:c1] if i < 7 else a[..., c0 - xoff : c1 - xoff] for i, a in enumerate(fargs)]
            )
        if par is None:
            return
        a: FloatArray
        for a in outs:
            par.fill_last(flat(a))
        for a in outs_x:
            par.fill_last(a)

    @staticmethod
    def _lpt(cost: Sequence[float], size: int) -> tuple[list[int], float]:
        """longest processing time first onto the least loaded rank (ties: lowest index / rank) -> (owner per item, makespan)"""
        import heapq

        load: list[tuple[float, int]] = [(0.0, rk) for rk in range(size)]
        owner: list[int] = [0] * len(cost)
        i: int
        for i in sorted(range(len(cost)), key=lambda i: (-cost[i], i)):
            l_: float
            rk: int
            l_, rk = heapq.heappop(load)
            owner[i] = rk
            heapq.heappush(load, (l_ + cost[i], rk))
        return owner, max(l_ for l_, _ in load)

    def _paw_items(self, par: _KPar) -> tuple[list[PawItem], list[int]]:
        """work items (atom, routes, directions) of the distributed one-centre pass and their ranks (LPT, cost ~ radial
        points x directions).  Default: each atom as nch chunks of contiguous Lebedev directions, both routes in one
        item (the grid fields are read once per point), nch <= 8 ranks / atoms the smallest count whose LPT makespan is
        within 3% of the best one (per-item overhead ~1.5 directions).  DCONV_PAW_SPLIT=routes: (atom, route) items with
        more ranks than atoms.  Cached."""
        nat: int = len(self.wfs.setups)
        nd: int = len(weight_n)
        mode: str = os.environ.get("DCONV_PAW_SPLIT", "dirs")
        key: tuple[int, int, str] = (par.size, nat, mode)
        if getattr(self, "_owner", None) is None or self._owner[0] != key:
            ng: list[int] = [len(self.wfs.setups[a].xc_correction.rgd.r_g) for a in range(nat)]
            items: list[PawItem]
            cost: list[float]
            if par.size == 1:
                items = [(a, (True, False), None) for a in range(nat)]
                cost = [float(ng[a]) for a in range(nat)]
            elif mode == "routes":
                split: bool = par.size > nat
                items = (
                    [(a, r, None) for a in range(nat) for r in ((True,), (False,))]
                    if split
                    else [(a, (True, False), None) for a in range(nat)]
                )
                cost = [ng[a] * (0.6 if len(r) == 1 else 1.0) for a, r, _ in items]
            else:
                cand: dict[int, tuple[float, list[PawItem], list[float]]] = {}
                nch: int
                for nch in range(1, min(nd, -(-8 * par.size // nat)) + 1):
                    b: IntArray = np.linspace(0, nd, nch + 1).round().astype(int)
                    dl: list[Dirs] = [None] if nch == 1 else [(int(b[i]), int(b[i + 1])) for i in range(nch)]
                    it_: list[PawItem] = [(a, (True, False), d) for a in range(nat) for d in dl]
                    c_: list[float] = [ng[a] * ((d[1] - d[0] if d else nd) + (1.5 if d else 0.0)) for a, _, d in it_]
                    cand[nch] = (self._lpt(c_, par.size)[1], it_, c_)
                best: float = min(v[0] for v in cand.values())
                items, cost = next(v[1:] for k, v in sorted(cand.items()) if v[0] <= 1.03 * best)
            owner: list[int]
            owner, _ = self._lpt(cost, par.size)
            self._owner = (key, items, owner)
        return self._owner[1], self._owner[2]

    def _paw_pass(self, par: _KPar | None) -> PawCache:
        """one-centre pass of every atom on the current D_asp and fields (_core_only: energy, dE/dD without the tails
        adjoint, sphere -> grid deposits into the *_new arrays) -> {a: (D_sp, E_a, dE/dD)}, the cache of GPAW's per-atom
        calls.  k-parallel: atoms distributed as (atom, routes, directions) items (_paw_items), the per-atom results
        summed over the items, the deposits summed over the ranks so that every rank holds the complete arrays"""
        nat: int = len(self.wfs.setups)
        if par is None:
            cache: PawCache = {}
            a_: int
            for a_ in range(nat):
                D_sp_: FloatArray = np.asarray(self._dasp(a_))
                dEdD_: FloatArray = np.zeros_like(D_sp_)
                E_a: float = self.calculate_paw_correction(self.wfs.setups[a_], D_sp_, dEdD_, a=a_, _core_only=True)
                cache[a_] = (D_sp_.copy(), E_a, dEdD_)
            return cache
        sizes: list[int] = [np.asarray(self._dasp(a)).size for a in range(nat)]
        off: IntArray = nat + np.r_[0, np.cumsum(sizes)]
        buf: FloatArray = np.zeros(off[-1])
        items: list[PawItem]
        owner: list[int]
        items, owner = self._paw_items(par)
        dist: bool = getattr(self, "_slab", (0, 0, False))[2]
        dom: bool = getattr(self, "_dom_on", False)
        xoff: bool = getattr(self, "_xoff", False)
        if xoff and not dom:
            dist = False  # no convolved fields to fetch and no deposits (dE/dx = 0)
        lay: dict[str, Any] | None = (self._paw_layout_dom(par) if dom else self._paw_layout(par)) if dist else None
        if dist:  # the prefiltered fields at this rank's columns; deposits onto its touched points
            if dom and xoff:  # no convolved fields are read
                self._prefC = self._field_fetch_dom(
                    par, cast("dict[str, Any]", lay), fields=np.zeros((0, self._domL.Pl))
                )
            elif dom:  # all convolved fields, from the X-slab owners
                self._prefC = self._field_fetch_dom(par, cast("dict[str, Any]", lay))
            else:
                self._prefC = self._pref_fetch(par, cast("dict[str, Any]", lay))
            dep: FloatArray | None = getattr(self, "_deposit_new", None)
            self._dep_acc = (
                None if dep is None else np.zeros((dep.shape[0], dep.shape[1], len(cast("dict[str, Any]", lay)["U"])))
            )
            self._lay = lay
        try:
            routes: tuple[bool, ...]
            dirs: Dirs
            rk: int
            for (a_, routes, dirs), rk in zip(items, owner):
                if rk != par.rank:
                    continue
                D_sp_ = np.asarray(self._dasp(a_))
                dEdD_ = np.zeros_like(D_sp_)
                buf[a_] += self.calculate_paw_correction(
                    self.wfs.setups[a_], D_sp_, dEdD_, a=a_, _core_only=True, _routes=routes, _dirs=dirs
                )
                buf[off[a_] : off[a_ + 1]] += dEdD_.ravel()
        finally:
            self._lay = self._prefC = None
        if dist:
            if self._dep_acc is not None:
                self._deposit_scatter(par, cast("dict[str, Any]", lay))
            self._dep_acc = None
            self._pref_own = None  # the fields' owner copies are not read again in this update
            self._pref_dom = None
        par.sum(buf)
        cache = {}
        a: int
        for a in range(nat):
            D0: FloatArray = np.asarray(self._dasp(a))
            cache[a] = (D0.copy(), float(buf[a]), buf[off[a] : off[a + 1]].reshape(D0.shape).copy())
        dep = getattr(self, "_deposit_new", None)
        if not dist and isinstance(dep, np.ndarray):
            par.sum(dep)
        return cache

    def _paw_layout(self, par: _KPar) -> dict[str, Any]:
        """distributed layout of the k-parallel one-centre pass, per geometry and item assignment: this rank's items,
        the grid columns they read (sorted union of the 64-point stencils of their sphere points) and every rank's
        column list (field service), the grid points they deposit to (U, sorted union of the trilinear corners) with
        each item's positions in U, U split by slab owner and the received index lists (local to this rank's slab) of
        the deposit scatter"""
        geo: dict[str, Any] = cast("dict[str, Any]", self._geom)
        items: list[PawItem]
        owner: list[int]
        items, owner = self._paw_items(par)
        key: tuple[str | int, ...] = ("lay",) + tuple(self._owner[0])
        if key in geo["pts"]:
            return cast("dict[str, Any]", geo["pts"][key])
        mine: list[PawItem] = [(a, r, d) for (a, r, d), rk in zip(items, owner) if rk == par.rank]
        cols: list[IntArray] | IntArray = [
            np.unique(self._readout_op(a, d, force=True, cache=False).indices) for a, r, d in mine
        ]
        cols = np.unique(np.concatenate(cols)).astype(np.int64) if cols else np.zeros(0, np.int64)
        U: list[IntArray] | IntArray = [self._splat_ops(a, d)[0] for a, r, d in mine]
        U = np.unique(np.concatenate(U)).astype(np.int64) if U else np.zeros(0, np.int64)
        P: int = int(np.prod(geo["N"]))
        all_cols: list[IntArray] = par.alltoallv_list([cols] * par.size)
        seg: IntArray = np.searchsorted(U, [par.rng(P, q)[0] for q in range(par.size)] + [P])
        p0: int = par.rng(P)[0]
        recv: list[IntArray] = [x - p0 for x in par.alltoallv_list([U[seg[q] : seg[q + 1]] for q in range(par.size)])]
        pos: dict[tuple[int, tuple[int, ...] | None], IntArray] = {
            (a, None if d is None else tuple(d)): np.searchsorted(U, self._splat_ops(a, d)[0]) for a, r, d in mine
        }
        g2l: IntArray = np.full(P, -1, np.int32)  # grid point -> position in cols (-1: not read by this rank's items)
        g2l[cols] = np.arange(len(cols), dtype=np.int32)
        geo["pts"][key] = dict(mine=mine, cols=cols, all_cols=all_cols, U=U, seg=seg, recv=recv, pos=pos, g2l=g2l)
        return cast("dict[str, Any]", geo["pts"][key])

    def _pref_fetch(self, par: _KPar, lay: dict[str, Any]) -> FloatArray:
        """the prefiltered convolved fields of every (spin, node) item at this rank's column list (ncols, items): each rank sends its own
        items' values at the other ranks' columns (one alltoallv; the fields themselves stay on their owner rank)"""
        own: FloatArray = cast(FloatArray, self._pref_own)
        i0: int
        i1: int
        i0, i1 = self._pref_rng
        nI: int = len(self.nodes) * self._tails[0]["nspins"]
        parts: list[FloatArray] = [np.ascontiguousarray(own[:, c]).ravel() for c in lay["all_cols"]]
        got: list[FloatArray] = par.alltoallv_list(parts)
        nc: int = len(lay["cols"])
        out: FloatArray = np.empty((nc, nI))
        r: int
        for r in range(par.size):
            r0: int
            r1: int
            r0, r1 = par.rng(nI, r)
            out[:, r0:r1] = got[r].reshape(r1 - r0, nc).T
        return out

    def _deposit_scatter(self, par: _KPar, lay: dict[str, Any]) -> None:
        """add this rank's compact deposit accumulation (spin, node, U) to the slab owners' _deposit_new (source ranks in order)"""
        acc: FloatArray = cast(FloatArray, self._dep_acc)
        ns_: int
        nw_: int
        ns_, nw_ = acc.shape[:2]
        parts: list[FloatArray] = [
            np.ascontiguousarray(acc[:, :, lay["seg"][q] : lay["seg"][q + 1]]).ravel() for q in range(par.size)
        ]
        got: list[FloatArray] = par.alltoallv_list(parts)
        dep: FloatArray = cast(FloatArray, self._deposit_new)
        r: int
        for r in range(par.size):
            idx: IntArray = lay["recv"][r]
            if len(idx):
                dep[:, :, idx] += got[r].reshape(ns_, nw_, len(idx))

    def _deposit_points(self, a: int, s: int, Wpts: FloatArray, dirs: Dirs = None) -> None:
        """accumulate sum_pt W_w(pt) delta(r - r_pt) on the grid (trilinear splat, per spin and node) for the next grid evaluation"""
        if getattr(self, "_deposit_new", None) is None:
            return
        lay: dict[str, Any] | None
        acc: FloatArray | None
        lay, acc = getattr(self, "_lay", None), getattr(self, "_dep_acc", None)
        if (
            lay is None
        ):  # the point weights dE/dC of this update, kept for the explicit position derivative (add_forces)
            if getattr(self, "_fdep_owner", None) is not self._deposit_new:
                self._fdep, self._fdep_owner = {}, self._deposit_new
            ent: dict[int, FloatArray] = self._fdep.setdefault(
                (a, None if dirs is None else (int(dirs[0]), int(dirs[1]))), {}
            )
            W: FloatArray = np.asarray(Wpts, float)
            ent[s] = ent[s] + W if s in ent else W.copy()
        if (
            lay is not None and acc is not None
        ):  # distributed layout: onto the rank's touched points, scattered to the slab owners after the pass
            uidx: IntArray
            Mt: Any
            uidx, Mt = self._splat_ops(a, dirs)
            N: tuple[int, ...] = cast("dict[str, Any]", self._geom)["N"]
            dvg: float = cast("dict[str, Any]", self._geom)["vol"] / np.prod(N)
            Wf: FloatArray = np.ascontiguousarray(np.asarray(Wpts).reshape(acc.shape[1], -1).T)
            acc[s][:, lay["pos"][(a, None if dirs is None else tuple(dirs))]] += (Mt @ Wf).T / dvg
            return
        self._splat(cast(FloatArray, self._deposit_new)[s], a, Wpts, dirs)

    # ------------------------------------------------------------------ forces
    # GPAW's forces contain the derivative through D_asp (projectors), the compensation charges and the pseudo core
    # densities.  The shell term's one-centre evaluation also depends on R_a explicitly: (1) its points R_a + r read the
    # convolved grid fields, (2) the tails of every atom are placed at R_a in those fields.  dE/dR_a at fixed pseudo
    # density and D_asp:
    #   (1) sum_{s,w,pt} W_s,w(pt) grad F_s,w(R_a + r_pt)     (W = dE/dC, the sphere -> grid deposit weights of this update)
    #   (2) the derivative of the reciprocal-space tails sum_a e^{-iG.R_a} F_a(G) against the node-summed feedback potential
    # Exact on the replicated layouts (serial; k-point MPI with DCONV_SPLAT=cubic); the distributed layouts (default under MPI) do not keep
    # the fields needed here, and their forces lack this term.
    @staticmethod
    def _cubic_dw(coords: FloatArray) -> list[FloatArray]:
        """per-axis derivative weights d/dt of the cubic B-spline read-out (order as _cubic_wi)"""
        dW: list[FloatArray] = []
        d: int
        for d in range(3):
            t: FloatArray = coords[d] - np.floor(coords[d])
            t2: FloatArray = t * t
            dW.append(np.stack([-0.5 * (1.0 - t) ** 2, 1.5 * t2 - 2.0 * t, -1.5 * t2 + t + 0.5, 0.5 * t2], 1))
        return dW

    def _grad_read(self, fields: FloatArray, coords: FloatArray, chunk: int = 4096) -> FloatArray:
        """Cartesian gradient (3, n, k) of the cubic B-spline read-out of prefiltered fields (P, k) at fractional grid coordinates (3, n)"""
        geo: dict[str, Any] = cast("dict[str, Any]", self._geom)
        N: tuple[int, ...] = geo["N"]
        n: int
        k: int
        n, k = coords.shape[1], fields.shape[1]
        out: FloatArray = np.zeros((3, n, k))
        c0: int
        for c0 in range(0, n, chunk):
            c1: int = min(n, c0 + chunk)
            cc: FloatArray = coords[:, c0:c1]
            W: list[FloatArray]
            ii: list[IntArray]
            W, ii = self._cubic_wi(cc, N)
            dW: list[FloatArray] = self._cubic_dw(cc)
            idx: IntArray = (
                (ii[0][:, :, None, None] * N[1] + ii[1][:, None, :, None]) * N[2] + ii[2][:, None, None, :]
            ).reshape(c1 - c0, 64)
            Fv: FloatArray = fields[idx]
            d: int
            for d in range(3):
                w: list[FloatArray] = [dW[e] if e == d else W[e] for e in range(3)]
                val: FloatArray = (w[0][:, :, None, None] * w[1][:, None, :, None] * w[2][:, None, None, :]).reshape(
                    c1 - c0, 64
                )
                out[d, c0:c1] = np.einsum("pm,pmk->pk", val, Fv)
        J: FloatArray = geo["inv"] * np.array(N, float)[None, :]  # du_d / dr_v
        return cast(FloatArray, np.einsum("vd,dnk->vnk", J, out))

    def explicit_dEdR(self) -> tuple[FloatArray, bool]:
        """dE/dR_a (natoms, 3; Hartree / Bohr) of the explicit position dependence, summed over the ranks; (array, available)"""
        nat: int = len(self.wfs.setups)
        g: FloatArray = np.zeros((nat, 3))
        if getattr(self, "_xoff", False) or getattr(self, "_geom", None) is None:
            return g, True
        if (
            getattr(self, "_dom_on", False)
            or (getattr(self, "_slab", None) is not None and self._slab[2])
            or self._S_prefT is None
        ):
            return g, False
        K: int = len(self.nodes)
        a: int
        dk: Dirs
        ent: dict[int, FloatArray]
        for (a, dk), ent in (getattr(self, "_fdep", None) or {}).items():
            coords: FloatArray
            coords, _ = self._points(a, dk)
            grad: FloatArray = self._grad_read(self._S_prefT, coords)
            s_: int
            Wd: FloatArray
            for s_, Wd in ent.items():
                g[a] += np.einsum("wn,vnw->v", Wd.reshape(K, -1), grad[:, :, s_ * K : (s_ + 1) * K])
        par: _KPar | None = self._kpar()
        if getattr(self, "_vfb_swg", None) is not None:
            # (2) in reciprocal space, as the tails enter the fields: t(r) = IFFT[(Ng / V) sum_a e^{-iG.R_a} F_a(G)], dE = dv sum_r vfb dt
            # -> dE/dR_a = (1 / Ng) sum_G Re[-i G e^{-iG.R_a} F_a(G) conj(FFT vfb(G))]
            geo: dict[str, Any] = cast("dict[str, Any]", self._geom)
            N: tuple[int, ...] = geo["N"]
            Ng: int = int(np.prod(N))
            hs: dict[str, Any] = self._tails_half()
            nh: int = len(hs["hidx"])
            ns: int = cast(FloatArray, self._vfb_swg).shape[0]
            Vc: list[ComplexArray] = [
                np.conj(_fftn(cast(FloatArray, self._vfb_swg)[s_].sum(0).reshape(N)).reshape(-1)) for s_ in range(ns)
            ]
            atoms: list[int] = [a for a in range(nat) if not self._species_tables(a)["negl"]]
            for a in atoms if par is None else atoms[par.rank :: par.size]:
                Fh: ComplexArray = self._tails_fsum(0, nh, atoms=[a])
                A: ComplexArray = np.empty((ns, Ng), complex)
                A[:, hs["hidx"]] = Fh
                A[:, hs["cdst"]] = np.conj(A[:, hs["csrc"]])
                c0: int
                for c0 in range(0, Ng, 1 << 20):
                    c1: int = min(Ng, c0 + (1 << 20))
                    Gv: FloatArray = self._gvec(np.arange(c0, c1))
                    for s_ in range(ns):
                        g[a] += np.real(-1j * (A[s_, c0:c1] * Vc[s_][c0:c1]) @ Gv) / Ng
        if par is not None:
            par.sum(g)
        return g, True

    def add_forces(self, F_av: FloatArray) -> None:
        MGGA.add_forces(self, F_av)
        g: FloatArray
        ok: bool
        g, ok = self.explicit_dEdR()
        if not ok and not getattr(self, "_fwarned", False):
            self._fwarned = True
            if self.wfs.world.rank == 0:
                import warnings

                warnings.warn(
                    "LOCUS forces: the explicit position derivative of the shell term needs a replicated layout (serial, or "
                    "DCONV_SPLAT=cubic with k-point parallelisation); omitted in this run (error ~1e-3 eV/A)"
                )
        F_av -= g

    def _splat(self, target_wg: FloatArray, a: int, Wpts: FloatArray, dirs: Dirs = None) -> None:
        """target_wg (K, P) += sum_pt W_w(pt) delta(r - r_pt) / dv_g at the sphere points of atom a, per node.  "cubic"
        (_splat_mode): the exact transpose of the sphere-point read-out map_coordinates(order 3, prefilter=False) of the
        prefiltered fields, i.e. cubic B-spline weights here and the (symmetric, periodic) spline prefilter applied once
        to the accumulated deposit at use (`_deposit_filter`); "linear": trilinear weights and no filter, an approximate
        adjoint.  dirs: the points of a direction chunk (see _points)."""
        coords: FloatArray
        shp: tuple[int, ...]
        coords, shp = self._points(a, dirs)
        N: tuple[int, ...] = cast("dict[str, Any]", self._geom)["N"]
        dvg: float = cast("dict[str, Any]", self._geom)["vol"] / np.prod(N)
        if self._splat_mode() == "cubic":
            base: IntArray = np.floor(coords).astype(int) - 1
            wts: list[FloatArray] = []
            d: int
            for d in range(3):
                t: FloatArray = coords[d][None, :] - (base[d][None, :] + np.arange(4)[:, None])  # (4, P) in (-2, 2)
                at_: FloatArray = np.abs(t)
                wts.append(
                    np.where(
                        at_ < 1.0, 2.0 / 3.0 - at_**2 + 0.5 * at_**3, np.where(at_ < 2.0, (2.0 - at_) ** 3 / 6.0, 0.0)
                    )
                )
            Wf: list[FloatArray] | FloatArray = [np.asarray(Wpts[w]).ravel() / dvg for w in range(target_wg.shape[0])]
            ox: int
            for ox in range(4):
                oy: int
                for oy in range(4):
                    wxy: FloatArray = wts[0][ox] * wts[1][oy]
                    oz: int
                    for oz in range(4):
                        wgt: FloatArray = wxy * wts[2][oz]
                        idx: IntArray = np.ravel_multi_index(
                            ((base[0] + ox) % N[0], (base[1] + oy) % N[1], (base[2] + oz) % N[2]), N
                        )
                        w: int
                        for w in range(target_wg.shape[0]):
                            target_wg[w] += np.bincount(idx, weights=wgt * Wf[w], minlength=target_wg.shape[1])
            return
        # trilinear splat through the cached operator (distinct grid points touched + sparse (touched point x sphere
        # point) weights): one sparse x dense product for all nodes onto those points only
        uidx: IntArray
        Mt: Any
        uidx, Mt = self._splat_ops(a, dirs)
        K: int = target_wg.shape[0]
        Wf = np.ascontiguousarray(np.asarray(Wpts).reshape(K, -1).T)  # (point, node)
        target_wg[:, uidx] += (Mt @ Wf).T / dvg

    def _splat_ops(self, a: int, dirs: Dirs = None) -> tuple[IntArray, Any]:
        """trilinear splat operator of atom a's one-centre points (of a direction chunk), cached per geometry: the distinct grid points touched
        (uidx) and the CSR matrix (len(uidx), n_pts) of the trilinear corner weights"""
        geo: dict[str, Any] = cast("dict[str, Any]", self._geom)
        ck: tuple[object, ...] = ("splat", a) if dirs is None else ("splat", a, tuple(dirs))
        if ck not in geo["pts"]:
            coords: FloatArray
            coords, _ = self._points(a, dirs)
            N: tuple[int, ...] = geo["N"]
            c0: IntArray = np.floor(coords).astype(int)
            fr: FloatArray = coords - c0
            idx: list[IntArray]
            wgt: list[FloatArray]
            idx, wgt = [], []
            dx: int
            for dx in (0, 1):
                dy: int
                for dy in (0, 1):
                    dz: int
                    for dz in (0, 1):
                        wgt.append(
                            (fr[0] if dx else 1 - fr[0]) * (fr[1] if dy else 1 - fr[1]) * (fr[2] if dz else 1 - fr[2])
                        )
                        idx.append(
                            np.ravel_multi_index(((c0[0] + dx) % N[0], (c0[1] + dy) % N[1], (c0[2] + dz) % N[2]), N)
                        )
            import scipy.sparse as sps

            uidx: IntArray
            inv: IntArray
            uidx, inv = np.unique(np.concatenate(idx), return_inverse=True)
            npt: int = coords.shape[1]
            Mt: Any = sps.csr_matrix(
                (np.concatenate(wgt), (inv.astype(np.int32), np.tile(np.arange(npt, dtype=np.int32), 8))),
                shape=(len(uidx), npt),
            )
            geo["pts"][ck] = (uidx, Mt)
        return cast("tuple[IntArray, Any]", geo["pts"][ck])

    def _splat_mode(self) -> str:
        """sphere -> grid splat: DCONV_SPLAT if set; otherwise the exact transpose of the cubic read-out ("cubic")
        except on the distributed layout, which implements the trilinear approximate adjoint only ("linear")"""
        m: str | None = os.environ.get("DCONV_SPLAT")
        if m:
            return m
        if getattr(self, "_dom_on", False) or (getattr(self, "_slab", None) is not None and self._slab[2]):
            return "linear"
        return "cubic"

    def _own_route(self) -> bool:
        """the explicit own-atom route dE/dD = sum Phi T: off by default, as the sphere -> grid deposit fed through the
        tails adjoint already contains every atom's sphere points (own and neighbours) and, with exact tails, the
        explicit route would count the own-atom part twice.  On when the deposit is switched off (DCONV_NO_DEPOSIT) or
        with DCONV_OWN_ROUTE=1 (diagnostics)."""
        v: str | None = os.environ.get("DCONV_OWN_ROUTE")
        if v is not None:
            return v != "0"
        return bool(os.environ.get("DCONV_NO_DEPOSIT"))

    def _deposit_filter(self, dep: FloatArray, N: tuple[int, ...]) -> FloatArray:
        """the transpose of the spline prefilter (periodic, symmetric: the filter itself) on every (..., P) deposit
        array (cubic splat)"""
        if self._splat_mode() != "cubic":
            return dep
        from scipy.ndimage import spline_filter

        out: FloatArray = np.empty_like(dep)
        ix: tuple[int, ...]
        for ix in np.ndindex(dep.shape[:-1]):
            out[ix] = spline_filter(dep[ix].reshape(N), order=3, mode="grid-wrap").reshape(-1)
        return out

    def _tails_adjoint(self, a: int, s: int = 0) -> FloatArray | float:
        """dE_grid/dD_s,p = sum_w int vfb_s,w(r) Y_L(r^) [P Delta_qL](|r - R_a|) dr B_pqL, by quadrature on the
        (direction, ru) points.  The tails do not depend on the node w, so only the node-summed feedback potential is
        read."""
        if getattr(self, "_dom_on", False):  # computed for every atom in the distributed pass (_tails_adjoint_dom_pass)
            t: dict[int, FloatArray] = getattr(self, "_tadj", None) or {}
            return t[a][s] if a in t else 0.0
        if getattr(self, "_vfb_swg", None) is None:
            return 0.0
        st: dict[str, Any] = self._species_tables(a)
        if st["negl"]:  # tails of this species are rounding noise (see _species_tables)
            return 0.0
        geo: dict[str, Any] = cast("dict[str, Any]", self._geom)
        N: tuple[int, ...] = geo["N"]
        if self._vfb_pref is None:
            from scipy.ndimage import spline_filter

            self._vfb_pref = [
                spline_filter(cast(FloatArray, self._vfb_swg)[s_].sum(0).reshape(N), order=3, mode="grid-wrap")
                for s_ in range(cast(FloatArray, self._vfb_swg).shape[0])
            ]
        v: FloatArray = map_coordinates(
            self._vfb_pref[s], self._tail_coords(a), order=3, mode="grid-wrap", prefilter=False
        ).reshape(len(weight_n), len(st["rq"]))
        return self._tails_adjoint_from(a, v)

    def _tail_coords(self, a: int) -> FloatArray:
        """fractional grid coordinates (3, n_dir x n_ru) of atom a's tails quadrature points (cached per geometry)"""
        geo: dict[str, Any] = cast("dict[str, Any]", self._geom)
        ck: tuple[str, int] = ("tail", a)
        if ck not in geo["pts"]:
            ru: FloatArray = self._species_tables(a)["rq"]  # log-grid tails quadrature points
            pts: FloatArray = geo["pos"][a][None, None, :] + ru[None, :, None] * R_nv[:, None, :]
            geo["pts"][ck] = (pts.reshape(-1, 3) @ geo["inv"] * np.array(geo["N"])).T
        return cast(FloatArray, geo["pts"][ck])

    def _tails_adjoint_from(self, a: int, v: FloatArray) -> FloatArray:
        """the tails adjoint of atom a from the node-summed feedback potential v (n_dir, n_ru) at its tails points"""
        xcc: Any = self.wfs.setups[a].xc_correction
        st: dict[str, Any] = self._species_tables(a)
        vL: FloatArray = 4 * PI * _einsum("n,nL,nu->Lu", weight_n, Y_nL[:, : xcc.Lmax], v)  # angular projection
        out: FloatArray = np.einsum("Lu,qLu->qL", vL, st["PUrq"][:, st["lofL"]])
        return cast(FloatArray, np.inner(out.reshape(-1), xcc.B_pqL.reshape((len(xcc.B_pqL), -1))))
