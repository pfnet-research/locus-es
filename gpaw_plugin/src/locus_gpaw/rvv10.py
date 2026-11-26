"""rVV10 non-local correlation (Sabatini, Gorni, de Gironcoli, PRB 87, 041108 (2013)) on a uniform periodic grid by FFT,
the O(N log N) form of VV10.

    E_c^nl = int n(r) [ beta + 1/2 int n(r') Phi(r, r') dr' ] dr,   beta = (3 / b^2)^(3/4) / 32
    Phi = -3/2 / [ (k k')^(3/2) (q R^2 + 1)(q' R^2 + 1)(q R^2 + q' R^2 + 2) ],   q = w0 / k
    w0 = sqrt( C |grad n / n|^4 + 4 pi n / 3 ),   k = b (3 pi / 2) (n / (9 pi))^(1/6)
Interpolation (Roman-Perez-Soler): q is saturated at q_cut, Phi(q, q', R) ~ sum_ab p_a(q) p_b(q') Phi_ab(R) with cubic-spline cardinal
functions p_a on a q-mesh; theta_a = n p_a(q) / k^(3/2); E = 1/2 sum_ab int theta_a (Phi_ab * theta_b) + beta N.
Returns the energy and dE/dn, dE/dsigma (sigma = |grad n|^2) on the grid (the GGA-type potential)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast, overload

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline, PPoly

from ._types import BoolArray, ComplexArray, FloatArray, IntArray

if TYPE_CHECKING:
    from .dconv import _KPar
    from .dconv_dom import DomLayout

C_VV: float = 0.0093


def q_mesh(n_q: int = 24, q_min: float = 1e-4, q_cut: float = 0.5) -> FloatArray:
    """logarithmic-like mesh (denser at small q), as QE's rVV10"""
    t: FloatArray = np.linspace(0, 1, n_q)
    return cast(FloatArray, q_min + (q_cut - q_min) * (np.expm1(4.0 * t) / np.expm1(4.0)))


def saturate(q: FloatArray, q_cut: float, m: int = 12) -> tuple[FloatArray, FloatArray]:
    """q_s = q_cut (1 - exp(-sum_{j=1..m} (q/q_cut)^j / j)) and dq_s/dq"""
    x: FloatArray = q / q_cut
    s: FloatArray = np.zeros_like(q)
    ds: FloatArray = np.zeros_like(q)
    xp: FloatArray = np.ones_like(q)
    j: int
    for j in range(1, m + 1):
        ds += xp
        xp = xp * x
        s += xp / j
    e: FloatArray = np.exp(-s)
    return q_cut * (1.0 - e), e * ds


def cardinal_splines(qm: FloatArray) -> list[CubicSpline]:
    """p_a(q) and p_a'(q) evaluators: natural cubic splines through unit vectors"""
    sp: list[CubicSpline] = [CubicSpline(qm, np.eye(len(qm))[a], bc_type="natural") for a in range(len(qm))]
    return sp


_KT_CACHE: dict[tuple[tuple[float, ...], float, int, int, float], tuple[FloatArray, FloatArray]] = {}


def kernel_table(
    qm: FloatArray, Gmax: float, nG: int = 1500, nR: int = 20000, Rmax: float = 400.0
) -> tuple[FloatArray, FloatArray]:
    """cached wrapper (process-wide; the table depends only on qm and Gmax)"""
    key: tuple[tuple[float, ...], float, int, int, float]
    key = (tuple(np.round(np.asarray(qm, float), 12)), round(float(Gmax), 6), nG, nR, Rmax)
    if key not in _KT_CACHE:
        from .dconv import _disk_cache  # on-disk cache of the table

        tb: dict[str, FloatArray] = _disk_cache(
            "rvv10_kt_v1",
            [np.asarray(qm, float), float(Gmax), nG, nR, Rmax],
            lambda: dict(zip(("G", "T"), _kernel_table(qm, Gmax, nG, nR, Rmax))),
        )
        _KT_CACHE[key] = (tb["G"], tb["T"])
    return _KT_CACHE[key]


def _kernel_table(
    qm: FloatArray, Gmax: float, nG: int = 1500, nR: int = 20000, Rmax: float = 400.0
) -> tuple[FloatArray, FloatArray]:
    """Phi_ab(G) = 4 pi int R^2 Phi_ab(R) sin(GR)/(GR) dR on a uniform |G| table (0..Gmax), direct quadrature on a fine
    R mesh"""
    R: FloatArray = (np.arange(nR) + 0.5) * (Rmax / nR)
    dR: float = Rmax / nR
    G: FloatArray = np.linspace(0, Gmax, nG)
    sinc: FloatArray = np.sinc(np.outer(G, R) / np.pi)
    nq: int = len(qm)
    out: FloatArray = np.zeros((nq, nq, nG))
    pairs: list[tuple[int, int]] = [(a, b) for a in range(nq) for b in range(a, nq)]
    W: FloatArray = np.empty((nR, len(pairs)))
    i: int
    a: int
    b: int
    for i, (a, b) in enumerate(pairs):
        x: FloatArray
        y: FloatArray
        x, y = qm[a] * R**2, qm[b] * R**2
        W[:, i] = R**2 * (-1.5 / ((x + 1) * (y + 1) * (x + y + 2)))
    T: FloatArray = 4 * np.pi * (sinc @ W) * dR
    for i, (a, b) in enumerate(pairs):
        out[a, b] = out[b, a] = T[:, i]
    return G, out


class RVV10:
    def __init__(
        self, b: float = 6.3, n_q: int = 24, q_cut: float = 0.5, q_min: float = 1e-4, n_thr: float = 1e-12
    ) -> None:
        self.b: float
        self.n_thr: float
        self.b, self.n_thr = b, n_thr
        self.qm: FloatArray = q_mesh(n_q, q_min, q_cut)
        self.q_cut: float = q_cut
        self.sp: list[CubicSpline] = cardinal_splines(self.qm)
        # all cardinal functions as one vector-valued spline
        self.sp2: CubicSpline = CubicSpline(self.qm, np.eye(len(self.qm)), bc_type="natural")
        self.beta: float = (3.0 / b**2) ** 0.75 / 32.0
        self.kvv: float = b * 1.5 * np.pi * (9 * np.pi) ** (-1.0 / 6.0)
        self._kern: Optional[tuple[FloatArray, FloatArray]] = None
        self._grid: Optional[dict[str, Any]] = None

    @overload
    def _card(self, q: FloatArray, rows: slice = ..., deriv: Literal[False] = ...) -> FloatArray: ...

    @overload
    def _card(self, q: FloatArray, rows: slice = ..., *, deriv: Literal[True]) -> tuple[FloatArray, FloatArray]: ...

    def _card(
        self, q: FloatArray, rows: slice = slice(None), deriv: bool = False
    ) -> Union[FloatArray, tuple[FloatArray, FloatArray]]:
        """cardinal functions p_a(q) of the rows a as (n_rows, len(q)) and optionally p_a'(q) (scipy PPoly of sp2 restricted to the rows)"""
        sp: PPoly = (
            self.sp2
            if rows == slice(None)
            else PPoly.construct_fast(np.ascontiguousarray(self.sp2.c[:, :, rows]), self.sp2.x)
        )
        p: FloatArray = np.ascontiguousarray(sp(q).T)
        return (p, np.ascontiguousarray(sp(q, 1).T)) if deriv else p

    def _rgrid(self, N: tuple[int, ...], cell: FloatArray) -> dict[str, Any]:
        """reciprocal half grid (rfftn layout), kernel-table interpolation (j, f) and the G grouped by table interval j;
        cached per grid and cell"""
        key: tuple[tuple[int, ...], bytes] = (tuple(N), np.asarray(cell).tobytes())
        if self._grid is not None and self._grid["key"] == key:
            return self._grid
        B: FloatArray = 2 * np.pi * np.linalg.inv(cell).T
        idx: list[npt.NDArray[Any]] = [np.fft.fftfreq(N[i], 1.0 / N[i]) for i in range(2)] + [np.arange(N[2] // 2 + 1)]
        I: npt.NDArray[Any]
        J: npt.NDArray[Any]
        K: npt.NDArray[Any]
        I, J, K = np.meshgrid(*idx, indexing="ij")
        Gv: FloatArray = I[..., None] * B[0] + J[..., None] * B[1] + K[..., None] * B[2]
        Gabs: FloatArray = np.sqrt((Gv**2).sum(-1))
        if self._kern is None or self._kern[0][-1] < Gabs.max():
            self._kern = kernel_table(self.qm, 1.01 * Gabs.max())
        Gt: FloatArray
        Kt: FloatArray
        Gt, Kt = self._kern
        jf: FloatArray = Gabs / (Gt[1] - Gt[0])
        j: IntArray = np.minimum(jf.astype(int), len(Gt) - 2)
        f: FloatArray = jf - j
        jr: IntArray = j.ravel()
        order: IntArray = np.argsort(jr, kind="stable")
        js: IntArray = jr[order]
        starts: IntArray = np.flatnonzero(np.r_[True, js[1:] != js[:-1]])
        bounds: IntArray = np.r_[starts, len(js)]
        groups: list[tuple[int, int, int]] = [
            (int(js[bounds[i]]), int(bounds[i]), int(bounds[i + 1])) for i in range(len(starts))
        ]
        fs: FloatArray = f.ravel()[order]
        self._grid = dict(
            key=key,
            shape=Gabs.shape,
            order=order.astype(np.int32 if len(order) < 2**31 else np.int64),
            groups=groups,
            fs=fs,
            KtT=np.ascontiguousarray(Kt.transpose(2, 0, 1)),
        )
        return self._grid

    def _products(self, T: FloatArray, c0: int, c1: int, gr: Optional[dict[str, Any]] = None) -> None:
        """in place on T (nq, c1 - c0): the float columns [c0, c1) of the G-sorted spectra ((re, im) interleaved) of every channel b ->
        sum_b Phi_ab(G) Theta_b(G) for every channel a, the kernel interpolated per |G| table interval j as (1 - f) Phi(G_j) @ Theta +
        f Phi(G_j+1) @ Theta (dense products over b)"""
        gr = cast("dict[str, Any]", self._grid) if gr is None else gr
        KtT: FloatArray
        fs: FloatArray
        KtT, fs = gr["KtT"], gr["fs"]
        jj: int
        s0: int
        s1: int
        for jj, s0, s1 in gr["groups"]:
            lo: int
            hi: int
            lo, hi = max(2 * s0, c0), min(2 * s1, c1)
            if lo >= hi:
                continue
            A: FloatArray = T[:, lo - c0 : hi - c0]
            w: FloatArray = np.repeat(fs[lo // 2 : (hi + 1) // 2], 2)[lo % 2 : lo % 2 + hi - lo]
            T[:, lo - c0 : hi - c0] = (KtT[jj] @ A) * (1 - w) + (KtT[jj + 1] @ A) * w

    def _convolve(self, T: FloatArray, N: tuple[int, ...], par: Optional[_KPar] = None) -> FloatArray:
        """u_a = sum_b Phi_ab * theta_b (per volume), in place.  T (rows: this rank's channels b; >= 2 M floats per
        row) holds theta_b on the whole grid in its first P floats, on return u of the same channels.  k-parallel
        (par): the products run on G slabs of all channels between two alltoallv transposes."""
        import scipy.fft as sfft

        from .dconv import _fft_workers

        wk: int = _fft_workers()
        nq: int = len(self.qm)
        gr: dict[str, Any] = cast("dict[str, Any]", self._grid)
        M: int = int(np.prod(gr["shape"]))
        P: int = int(np.prod(N))
        order: IntArray = gr["order"]
        b: int
        for b in range(len(T)):  # spectra, G sorted by table interval, over the channel's own input
            T[b] = sfft.rfftn(T[b, :P].reshape(N), workers=wk).reshape(-1)[order].view(float)
        if par is None:
            self._products(T, 0, 2 * M)
        else:
            c0: int
            c1: int
            c0, c1 = par.rng(2 * M)
            Ts: FloatArray = par.transpose(T, nq, 2 * M)  # (nq, c1 - c0): all channels, this rank's G slab
            self._products(Ts, c0, c1)
            T[:] = par.transpose_back(Ts, nq, 2 * M)
            del Ts
        uG: ComplexArray = np.empty(M, complex)
        a: int
        for a in range(len(T)):
            # (Phi_ab * theta_b)(r) = (1/V) sum_G Phi_ab(G) [dv Theta_b(G)] e^{iGr} = irfftn(Phi_ab Theta_b) since dv N_tot = V
            uG[order] = T[a].view(complex)
            T[a, :P] = sfft.irfftn(uG.reshape(gr["shape"]), s=N, workers=wk).reshape(-1)
        return T

    def _qfields(self, n: FloatArray, sigma: FloatArray, sl: slice) -> dict[str, npt.NDArray[Any]]:
        """the pointwise q-chain on the points sl: dict of n, m, nn, ss, wg2, w0, k, qs, dqs, pref"""
        n, sigma = n[sl], sigma[sl]
        m: BoolArray = n > self.n_thr
        nn: FloatArray = np.where(m, n, self.n_thr)
        ss: FloatArray = np.where(m, sigma, 0.0)
        wg2: FloatArray = C_VV * (ss / nn**2) ** 2
        wp: FloatArray = 4 * np.pi * nn / 3
        w0: FloatArray = np.sqrt(wg2 + wp)
        k: FloatArray = self.kvv * nn ** (1.0 / 6.0)
        q: FloatArray = w0 / k
        qs: FloatArray
        dqs: FloatArray
        qs, dqs = saturate(q, self.q_cut)
        pref: FloatArray = np.where(m, nn * k**-1.5, 0.0)
        return dict(n=n, m=m, nn=nn, ss=ss, wg2=wg2, w0=w0, k=k, qs=qs, dqs=dqs, pref=pref)

    def calculate(
        self, n: FloatArray, sigma: FloatArray, cell: FloatArray, par: Optional[_KPar] = None
    ) -> tuple[float, FloatArray, FloatArray]:
        """n, sigma: (N1, N2, N3) arrays on a periodic grid spanning cell (3x3, bohr).  Returns E, dE/dn, dE/dsigma.
        par (dconv._KPar, optional): the channels, their G slabs and the pointwise part are distributed over the ranks
        (every rank holds the full input grid, the outputs are all-gathered; point by point the serial expressions)."""
        from .dconv import KCHUNK

        N: tuple[int, ...] = n.shape
        vol: float = abs(np.linalg.det(cell))
        dv: float = vol / n.size
        P: int
        nq: int
        P, nq = n.size, len(self.qm)
        p0: int
        p1: int
        p0, p1 = (0, P) if par is None else par.rng(P)
        b0: int
        b1: int
        b0, b1 = (0, nq) if par is None else par.rng(nq)
        n, sigma = n.reshape(-1), sigma.reshape(-1)
        gr: dict[str, Any] = self._rgrid(N, cell)
        M: int = int(np.prod(gr["shape"]))
        CH: int = max(KCHUNK // 8, 1024)
        # theta of this rank's channels b on the whole grid, written into the convolution buffer (the first P floats of each row)
        T: FloatArray = np.empty((b1 - b0, 2 * M))
        c0: int
        for c0 in range(0, P, CH):
            sl: slice = slice(c0, min(c0 + CH, P))
            f: dict[str, npt.NDArray[Any]] = self._qfields(n, sigma, sl)
            T[:, sl] = f["pref"][None] * self._card(f["qs"], slice(b0, b1))
        self._convolve(T, N, par)
        if par is not None:  # u of this rank's points, all channels
            u_all: FloatArray = par.transpose(T[:, :P], nq, P)
            del T
        else:
            u_all = T[:, :P]
        # pointwise part on this rank's points, every channel a, in chunks
        out: FloatArray = np.zeros((3, P))
        for c0 in range(p0, p1, CH):
            c1: int = min(c0 + CH, p1)
            sl = slice(c0, c1)
            f = self._qfields(n, sigma, sl)
            n_: FloatArray
            m: BoolArray
            nn: FloatArray
            ss: FloatArray
            w0: FloatArray
            k: FloatArray
            dqs: FloatArray
            pref: FloatArray
            wg2: FloatArray
            n_, m, nn, ss, w0, k, dqs, pref, wg2 = (
                f[x] for x in ("n", "m", "nn", "ss", "w0", "k", "dqs", "pref", "wg2")
            )
            u: FloatArray = u_all[:, c0 - p0 : c1 - p0]
            # derivatives of q w.r.t. n and sigma
            dw0_dn: FloatArray = (-4 * wg2 / nn + 4 * np.pi / 3) / (2 * w0)
            dw0_ds: FloatArray = (2 * C_VV * ss / nn**4) / (2 * w0)
            dk_dn: FloatArray = k / (6 * nn)
            dq_dn: FloatArray = (dw0_dn * k - w0 * dk_dn) / k**2 * dqs
            dq_ds: FloatArray = dw0_ds / k * dqs
            dpref_dn: FloatArray = np.where(m, k**-1.5 - 1.5 * nn * k**-2.5 * dk_dn, 0.0)
            p: FloatArray
            dp: FloatArray
            p, dp = self._card(f["qs"], deriv=True)
            theta: FloatArray = pref[None] * p
            e_r: FloatArray = np.where(m, self.beta * n_, 0.0) + 0.5 * (theta * u).sum(
                0
            )  # energy density per volume (E = sum e_r dv)
            du: FloatArray = u  # dE/dtheta_a(r) (per volume)
            out[0, sl] = e_r
            out[1, sl] = np.where(
                m, self.beta + (du * (dpref_dn[None] * p + pref[None] * dp * dq_dn[None])).sum(0), 0.0
            )
            out[2, sl] = np.where(m, (du * (pref[None] * dp * dq_ds[None])).sum(0), 0.0)
        del u_all
        if par is not None:
            par.fill_last(out)
        self.last_e: Optional[FloatArray] = out[0].reshape(N)
        E: float = dv * float(out[0].sum())
        return E, out[1].reshape(N), out[2].reshape(N)

    def calculate_dom(
        self, n: FloatArray, sigma: FloatArray, cell: FloatArray, dom: DomLayout
    ) -> tuple[float, FloatArray]:
        """calculate on the distributed layout dconv_dom.DomLayout: n, sigma are this rank's X slab (Pl,), the
        convolutions use complex distributed 3D FFTs on the Y slab.  Returns (E, out) with out (3, Pl): energy density
        per volume, dE/dn, dE/dsigma; E summed over the ranks."""
        from .dconv import KCHUNK

        N: tuple[int, ...] = dom.N
        P: int
        nq: int
        P, nq = int(np.prod(N)), len(self.qm)
        vol: float = abs(np.linalg.det(cell))
        dv: float = vol / P
        Pl: int = dom.Pl
        key: tuple[tuple[int, ...], bytes, int, int] = (N, np.asarray(cell).tobytes(), dom.R, dom.rank)
        gr: Optional[dict[str, Any]] = getattr(self, "_gridd", None)
        if gr is None or gr["key"] != key:
            B: FloatArray = 2 * np.pi * np.linalg.inv(cell).T
            # |G| with calculate's real-FFT conventions: the half grid kz = 0 .. N2 / 2 carries its own G, a point
            # of the other half the G of its Hermitian partner (-i0, -i1, -i2 mod N)
            f0: FloatArray
            f1: FloatArray
            f0, f1 = (np.fft.fftfreq(N[i], 1.0 / N[i]) for i in range(2))
            i0: IntArray = np.arange(N[0])[:, None, None]
            i1: IntArray = np.arange(dom.y0, dom.y1)[None, :, None]
            i2: IntArray = np.arange(N[2])[None, None, :]
            upper: BoolArray = i2 <= N[2] // 2
            K0: FloatArray = np.where(upper, f0[i0], f0[(-i0) % N[0]])
            K1: FloatArray = np.where(upper, f1[i1], f1[(-i1) % N[1]])
            K2: IntArray = np.where(upper, i2, N[2] - i2)
            K0, K1, K2 = np.broadcast_arrays(K0, K1, K2)
            Gv: FloatArray = (
                K0.reshape(-1)[:, None] * B[0] + K1.reshape(-1)[:, None] * B[1] + K2.reshape(-1)[:, None] * B[2]
            )
            Gabs: FloatArray = np.sqrt((Gv**2).sum(-1))
            del Gv, K0, K1, K2
            gmax: float = dom.allmax(float(Gabs.max()) if len(Gabs) else 0.0)
            if self._kern is None or self._kern[0][-1] < gmax:
                self._kern = kernel_table(self.qm, 1.01 * gmax)
            Gt: FloatArray
            Kt: FloatArray
            Gt, Kt = self._kern
            jf: FloatArray = Gabs / (Gt[1] - Gt[0])
            j: IntArray = np.minimum(jf.astype(int), len(Gt) - 2)
            f: Union[FloatArray, dict[str, npt.NDArray[Any]]]
            f = jf - j
            order: IntArray = np.argsort(j, kind="stable")
            js: IntArray = j[order]
            starts: IntArray = np.flatnonzero(np.r_[True, js[1:] != js[:-1]]) if len(js) else np.zeros(0, int)
            bounds: IntArray = np.r_[starts, len(js)]
            groups: list[tuple[int, int, int]] = [
                (int(js[bounds[i]]), int(bounds[i]), int(bounds[i + 1])) for i in range(len(starts))
            ]
            self._gridd: dict[str, Any]
            gr = self._gridd = dict(
                key=key,
                order=order.astype(np.int32 if len(order) < 2**31 else np.int64),
                groups=groups,
                fs=f[order],
                KtT=np.ascontiguousarray(Kt.transpose(2, 0, 1)),
            )
        CH: int = max(KCHUNK // 8, 1024)
        n, sigma = np.asarray(n).reshape(-1), np.asarray(sigma).reshape(-1)
        T: FloatArray = np.empty((nq, Pl))
        c0: int
        for c0 in range(0, Pl, CH):
            sl: slice = slice(c0, min(c0 + CH, Pl))
            f = self._qfields(n, sigma, sl)
            T[:, sl] = f["pref"][None] * self._card(f["qs"])

        def conv(T: FloatArray) -> FloatArray:  # in place: theta_b (rows) -> u_a = sum_b Phi_ab * theta_b
            order: IntArray = gr["order"]
            TG: ComplexArray = np.empty((nq, dom.PlG), complex)
            b: int
            for b in range(nq):
                TG[b] = dom.fft(T[b])[order]
            self._products(TG.view(float), 0, 2 * dom.PlG, gr)
            uG: ComplexArray = np.empty(dom.PlG, complex)
            a: int
            for a in range(nq):
                uG[order] = TG[a]
                T[a] = np.real(dom.ifft(uG))
            return T

        u_all: FloatArray = conv(T)
        out: FloatArray = np.zeros((3, Pl))
        for c0 in range(0, Pl, CH):
            c1: int = min(c0 + CH, Pl)
            sl = slice(c0, c1)
            f = self._qfields(n, sigma, sl)
            n_: FloatArray
            m: BoolArray
            nn: FloatArray
            ss: FloatArray
            w0: FloatArray
            k: FloatArray
            dqs: FloatArray
            pref: FloatArray
            wg2: FloatArray
            n_, m, nn, ss, w0, k, dqs, pref, wg2 = (
                f[x] for x in ("n", "m", "nn", "ss", "w0", "k", "dqs", "pref", "wg2")
            )
            u: FloatArray = u_all[:, sl]
            dw0_dn: FloatArray = (-4 * wg2 / nn + 4 * np.pi / 3) / (2 * w0)
            dw0_ds: FloatArray = (2 * C_VV * ss / nn**4) / (2 * w0)
            dk_dn: FloatArray = k / (6 * nn)
            dq_dn: FloatArray = (dw0_dn * k - w0 * dk_dn) / k**2 * dqs
            dq_ds: FloatArray = dw0_ds / k * dqs
            dpref_dn: FloatArray = np.where(m, k**-1.5 - 1.5 * nn * k**-2.5 * dk_dn, 0.0)
            p: FloatArray
            dp: FloatArray
            p, dp = self._card(f["qs"], deriv=True)
            theta: FloatArray = pref[None] * p
            e_r: FloatArray = np.where(m, self.beta * n_, 0.0) + 0.5 * (theta * u).sum(0)
            du: FloatArray = u
            out[0, sl] = e_r
            out[1, sl] = np.where(
                m, self.beta + (du * (dpref_dn[None] * p + pref[None] * dp * dq_dn[None])).sum(0), 0.0
            )
            out[2, sl] = np.where(m, (du * (pref[None] * dp * dq_ds[None])).sum(0), 0.0)
        E: float = dv * float(dom.allsum([out[0].sum()])[0])
        return E, out
