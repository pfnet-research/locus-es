"""LOCUS for PySCF (molecules, all-electron Gaussian basis): r2SCAN + VV10 + the shell term, self-consistent (RKS / UKS).

    from locus_shell.pyscf_locus import run_locus
    e, mf, info = run_locus(mol)                       # PBE start -> r2SCAN + VV10 -> LOCUS on the same grids
    mf = make_ks(mol); mf.kernel()                     # LOCUS directly

The shell descriptor x = sum_a p_a(n) (G_a * n) / n uses the density fitted in the overlap metric to the even-tempered auxiliary basis
aug_etb(1.6) of the orbital basis (ridge 1e-11 tr S / M); each auxiliary shell convolved with a node Gaussian G_a (width r_a, 10 nodes
from 0.3 to 6 bohr) is again a Gaussian shell (exponent alpha a / (alpha + a), coefficient times (a / (alpha + a))^(l + 3/2), a = 1 / 2 r_a^2),
and p_a are the natural cubic-spline cardinal weights in ln r0 of r0 = clip(4 / k_F(n), 0.3, 6).  The potential is the exact derivative of
the energy of the fitted descriptor: a local part on the grid and the adjoint of fit and convolution in the auxiliary coefficient space,
V_ij += sum_Q (ij|Q) [(S + ridge)^-1 b]_Q.  Dispersion: PySCF's VV10 non-local correlation with b = 12, C = 0.0093 (the molecular
counterpart of the rVV10 of the solid-state implementation).  NumPy and SciPy only.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Sequence
from types import ModuleType
from typing import Any, Callable, NoReturn, Optional, Union, cast

import numpy as np
import scipy.linalg
from pyscf import df, dft, gto, lib
from pyscf.dft import numint
from pyscf.gto.mole import ANG_OF, ATOM_OF, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_EXP
from scipy.interpolate import CubicSpline

from . import kernel as K
from ._types import BoolArray, FloatArray, IntArray

DENSITY_NODES: tuple[float, ...] = (0.3, 0.4243, 0.6, 0.8485, 1.2, 1.6971, 2.4, 3.3941, 4.8, 6.0)  # bohr
CS: tuple[float, ...] = (4.0,)  # the shell descriptor x_c(4)
N_MIN: float = 1.0e-10  # dead points of the kernel
VV10: tuple[float, float] = (12.0, 0.0093)


# ------------------------------------------------------------------------------------------------------------------- node weights
class NodeWeights:
    """natural cubic-spline cardinal weights p_a of r0 = clip(c / k_F(m), nodes[0], nodes[-1]) in ln r0 (dp/dm = 0 where clamped)"""

    def __init__(self, nodes: Sequence[float] = DENSITY_NODES) -> None:
        self.nodes: FloatArray = np.asarray(nodes, float)
        self.x: FloatArray = np.log(self.nodes)
        sp: Any = CubicSpline(self.x, np.eye(len(self.nodes)), bc_type="natural")
        self.c: FloatArray = np.ascontiguousarray(sp.c.transpose(1, 0, 2))  # (nint, 4, A)
        nint: int
        A: int
        nint, A = self.c.shape[0], self.c.shape[2]
        C: FloatArray = np.zeros((nint + 1, 4, A))
        C[:nint] = self.c
        C[nint, 3, -1] = 1.0  # the exact unit vector of the upper clamp
        self.C: FloatArray = np.ascontiguousarray(C.reshape(-1, A).T)  # (A, 4 (nint + 1))

    def channel(
        self, m: FloatArray, S: FloatArray, cs: Sequence[float]
    ) -> tuple[FloatArray, FloatArray, IntArray, FloatArray]:
        """v (K, P) = sum_a p_a S_a, dv (K, P) = sum_a dp_a/dm S_a, and the interval data for adjoint()"""
        nint: int = self.c.shape[0]
        M: FloatArray = (S @ self.C).reshape(-1, nint + 1, 4)
        m = np.maximum(m, 1e-30)
        kf: FloatArray = np.cbrt(3.0 * math.pi**2 * m)
        P: int = len(m)
        ar: IntArray = np.arange(P)
        v: FloatArray
        dv: FloatArray
        v, dv = np.empty((len(cs), P)), np.empty((len(cs), P))
        I: IntArray
        Tt: FloatArray
        I, Tt = np.empty((len(cs), P), np.intp), np.empty((len(cs), P))
        ic: int
        c: float
        for ic, c in enumerate(cs):
            r0: FloatArray = c / kf
            hi: BoolArray = r0 >= self.nodes[-1]
            inside: BoolArray = (r0 > self.nodes[0]) & ~hi
            u: FloatArray = np.log(np.clip(r0, self.nodes[0], self.nodes[-1]))
            i: IntArray = np.clip(np.searchsorted(self.x, u, side="right") - 1, 0, nint - 1)
            i[hi] = nint
            t: FloatArray = u - self.x[np.minimum(i, nint - 1)]
            t[~inside] = 0.0
            Mi: FloatArray = M[ar, i]
            v[ic] = ((Mi[:, 0] * t + Mi[:, 1]) * t + Mi[:, 2]) * t + Mi[:, 3]
            dv[ic] = np.where(inside, ((3.0 * Mi[:, 0] * t + 2.0 * Mi[:, 1]) * t + Mi[:, 2]) * (-1.0 / (3.0 * m)), 0.0)
            I[ic], Tt[ic] = i, t
        return v, dv, I, Tt

    def adjoint(self, w: FloatArray, I: IntArray, Tt: FloatArray) -> FloatArray:  # noqa: E741
        """F (P, A) = sum_c w_c p_a(u_c)"""
        nint: int = self.c.shape[0]
        K_: int
        P: int
        K_, P = w.shape
        Z: FloatArray = np.zeros((P, nint + 1, 4))
        ar: IntArray = np.arange(P)
        ic: int
        for ic in range(K_):
            t: FloatArray
            wc: FloatArray
            t, wc = Tt[ic], w[ic]
            Z[ar, I[ic]] += np.stack([wc * t**3, wc * t * t, wc * t, wc], 1)
        return Z.reshape(P, -1) @ self.C.T


# ---------------------------------------------------------------------------------------------------------- density fit and convolution
class AuxConv:
    """per-geometry density fit (aug_etb(beta), S + ridge, (ij|Q)) and the node-convolved auxiliary shells per atom"""

    def __init__(
        self, mol: Any, nodes: Sequence[float] = DENSITY_NODES, beta: float = 1.6, max_memory: float = 4000
    ) -> None:
        self.mol: Any = mol
        aux: Any
        self.aux: Any
        aux = self.aux = df.addons.make_auxmol(mol, df.addons.aug_etb(mol, beta=beta))
        assert (aux._bas[:, NPRIM_OF] == 1).all() and (aux._bas[:, NCTR_OF] == 1).all() and not aux.cart
        S: FloatArray = aux.intor("int1e_ovlp")
        S += 1.0e-11 * np.trace(S) / len(S) * np.eye(len(S))
        self.cho: tuple[FloatArray, bool] = scipy.linalg.cho_factor(S)
        self.naux: int = aux.nao
        self.pmol: Any = gto.mole.conc_mol(mol, aux)
        npair: int = mol.nao * (mol.nao + 1) // 2
        self.I3: Optional[FloatArray] = None
        if npair * self.naux * 8 / 1e6 < max_memory:
            self.I3 = self.pmol.intor(
                "int3c1e", aosym="s2ij", shls_slice=(0, mol.nbas, 0, mol.nbas, mol.nbas, self.pmol.nbas)
            )
        self._i3_blk: int = max(1, int(max_memory * 1e6 / 4 / (8 * npair)))
        lmax: IntArray = np.full(aux.natm, -1)
        i: int
        for i in range(aux.nbas):
            A: int = aux._bas[i, ATOM_OF]
            lmax[A] = max(lmax[A], aux._bas[i, ANG_OF])
        env: FloatArray = np.append(aux._env, [0.0, 1.0])
        p0: int = len(env) - 2
        bas: list[list[int]]
        yoff: dict[tuple[int, int], int]
        o: int
        bas, yoff, o = [], {}, 0
        for A in range(aux.natm):
            ell: int
            for ell in range(lmax[A] + 1):
                bas.append([A, ell, 1, 1, 0, p0, p0 + 1, 0])
                yoff[A, ell] = o
                o += 2 * ell + 1
        self.ang: Any = aux.copy()
        self.ang._bas = np.asarray(bas, np.int32)
        self.ang._env = env
        self.atom_coords: FloatArray = aux.atom_coords()
        anode: FloatArray = 1.0 / (2.0 * np.asarray(nodes) ** 2)
        loc: IntArray = aux.ao_loc_nr()
        self.atoms: list[dict[str, Any]] = []
        for A in range(aux.natm):
            sh: IntArray = np.where(aux._bas[:, ATOM_OF] == A)[0]
            if len(sh) == 0:
                continue
            ells: IntArray = aux._bas[sh, ANG_OF]
            al: FloatArray = aux._env[aux._bas[sh, PTR_EXP]]
            cf: FloatArray = aux._env[aux._bas[sh, PTR_COEFF]]
            assert (cf > 0).all()
            f: FloatArray = anode[:, None] / (al[None, :] + anode[:, None])
            ys: tuple[int, int] = (yoff[A, 0], yoff[A, lmax[A]] + 2 * lmax[A] + 1)
            aidx: list[int]
            ridx: list[int]
            kidx: list[int]
            aidx, ridx, kidx = [], [], []
            k: int
            for k, i in enumerate(sh):
                mm: int
                for mm in range(2 * ells[k] + 1):
                    aidx.append(loc[i] + mm)
                    ridx.append(yoff[A, ells[k]] - ys[0] + mm)
                    kidx.append(k)
            self.atoms.append(
                dict(
                    A=A,
                    K=len(sh),
                    ys=ys,
                    NY=ys[1] - ys[0],
                    aidx=np.array(aidx),
                    ridx=np.array(ridx),
                    kidx=np.array(kidx),
                    nalp=-(al[None] * f),
                    lnb=np.log(cf)[None] + (ells + 1.5)[None] * np.log(f),
                )
            )

    def _i3_blocks(self) -> Iterator[tuple[int, int, FloatArray]]:
        mol: Any
        aux: Any
        mol, aux = self.mol, self.aux
        if self.I3 is not None:
            yield 0, self.naux, self.I3
            return
        loc: IntArray = aux.ao_loc_nr()
        k0: int = 0
        while k0 < aux.nbas:
            k1: int = k0
            while k1 < aux.nbas and loc[k1 + 1] - loc[k0] <= self._i3_blk:
                k1 += 1
            k1 = max(k1, k0 + 1)
            yield loc[k0], loc[k1], self.pmol.intor(
                "int3c1e", aosym="s2ij", shls_slice=(0, mol.nbas, 0, mol.nbas, mol.nbas + k0, mol.nbas + k1)
            )
            k0 = k1

    def fit(self, dms: FloatArray) -> FloatArray:
        dms = np.asarray(dms)
        dmt: FloatArray = lib.pack_tril(dms + dms.transpose(0, 2, 1))
        diag: IntArray = np.cumsum(np.arange(1, self.mol.nao + 1)) - 1
        dmt[:, diag] *= 0.5
        t: FloatArray = np.zeros((len(dms), self.naux))
        q0: int
        q1: int
        I3: FloatArray
        for q0, q1, I3 in self._i3_blocks():
            t[:, q0:q1] = dmt @ I3
        return cast(FloatArray, scipy.linalg.cho_solve(self.cho, t.T).T)

    def fock(self, b: FloatArray) -> FloatArray:
        u: FloatArray = scipy.linalg.cho_solve(self.cho, b.T)
        vt: Union[float, FloatArray] = 0.0
        q0: int
        q1: int
        I3: FloatArray
        for q0, q1, I3 in self._i3_blocks():
            vt = vt + I3 @ u[q0:q1]
        return np.array([lib.unpack_tril(cast(FloatArray, vt)[:, s]) for s in range(b.shape[0])])

    def set_coeff(self, coeff: FloatArray) -> None:
        self.S: int = coeff.shape[0]
        d: dict[str, Any]
        for d in self.atoms:
            Cm: FloatArray = np.zeros((d["NY"], d["K"], self.S))
            Cm[d["ridx"], d["kidx"]] = coeff[:, d["aidx"]].T
            d["Cm"] = Cm.reshape(d["NY"], -1)
            d["dC"] = np.zeros((d["NY"], d["K"] * self.S))

    def _geom(self, coords: FloatArray) -> tuple[FloatArray, FloatArray]:
        Y: FloatArray = np.asarray(numint.eval_ao(self.ang, coords))
        r2: FloatArray = ((coords[:, None, :] - self.atom_coords[None]) ** 2).sum(-1)
        return Y, np.ascontiguousarray(r2.T)

    @staticmethod
    def _exps(d: dict[str, Any], r2A: FloatArray) -> FloatArray:
        """X (P, A, K) = N (a / (alpha + a))^(l + 3/2) exp(-alpha a / (alpha + a) r_A^2) of the atom's shells at every node"""
        A_: int
        Kd: int
        A_, Kd = d["nalp"].shape
        X: FloatArray = np.exp(np.outer(r2A, d["nalp"].reshape(-1)) + d["lnb"].reshape(-1)[None, :])
        return X.reshape(len(r2A), A_, Kd)

    def forward(self, coords: FloatArray) -> FloatArray:
        """(P, A, S) = (G_a * n_s)(coords)"""
        Y: FloatArray
        r2: FloatArray
        Y, r2 = self._geom(coords)
        acc: Optional[FloatArray] = None
        d: dict[str, Any]
        for d in self.atoms:
            G: FloatArray = (Y[:, d["ys"][0] : d["ys"][1]] @ d["Cm"]).reshape(len(coords), d["K"], self.S)
            part: FloatArray = np.matmul(self._exps(d, r2[d["A"]]), G)
            acc = part if acc is None else acc + part
        return cast(FloatArray, acc)

    def adjoint(self, coords: FloatArray, wF: FloatArray) -> None:
        """accumulate sum_a <chi^(a) | wF_s,a> on these points; wF (P, S, A)"""
        Y: FloatArray
        r2: FloatArray
        Y, r2 = self._geom(coords)
        wFt: FloatArray = np.ascontiguousarray(wF.transpose(0, 2, 1))  # (P, A, S)
        d: dict[str, Any]
        for d in self.atoms:
            H: FloatArray = np.matmul(self._exps(d, r2[d["A"]]).transpose(0, 2, 1), wFt)  # (P, K, S)
            d["dC"] += Y[:, d["ys"][0] : d["ys"][1]].T @ H.reshape(len(coords), -1)

    def gather_b(self) -> FloatArray:
        b: FloatArray = np.zeros((self.S, self.naux))
        d: dict[str, Any]
        for d in self.atoms:
            dC: FloatArray = d["dC"].reshape(d["NY"], d["K"], self.S)
            b[:, d["aidx"]] = dC[d["ridx"], d["kidx"]].T
        return b


# ------------------------------------------------------------------------------------------------------------------- r2SCAN base
def _grad_from_sigma(sig_s: FloatArray) -> tuple[FloatArray, FloatArray]:
    """gradient vectors with the given contractions (sigma_aa, sigma_ab, sigma_bb) (Cauchy-Schwarz respected by the caller)"""
    saa: FloatArray
    sab: FloatArray
    sbb: FloatArray
    saa, sab, sbb = np.maximum(sig_s[0], 0.0), sig_s[1], np.maximum(sig_s[2], 0.0)
    a: FloatArray = np.sqrt(saa)
    with np.errstate(divide="ignore", invalid="ignore"):
        bx: FloatArray = np.where(a > 0, sab / np.where(a > 0, a, 1.0), 0.0)
    by: FloatArray = np.sqrt(np.maximum(sbb - bx * bx, 0.0))
    z: FloatArray = np.zeros_like(a)
    return np.stack([a, z, z]), np.stack([bx, by, z])


def base_eval_spin(
    ns_s: FloatArray, sig_s: FloatArray, tau_s: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """libxc r2SCAN through PySCF in the layout of the kernel: exchange per particle of the channel densities 2 n_s, polarized
    correlation -> ex2, vrx, vsx, vtx (2, P), ec (P), vrc (2, P), vsc (3, P), vtc (2, P)"""
    P: int = ns_s.shape[1]
    ex2: FloatArray
    vrx: FloatArray
    vsx: FloatArray
    vtx: FloatArray
    ex2, vrx, vsx, vtx = (np.empty((2, P)) for _ in range(4))
    s: int
    for s in range(2):
        m: FloatArray = 2.0 * ns_s[s]
        rho: FloatArray = np.zeros((6, P))
        rho[0], rho[1], rho[5] = m, 2.0 * np.sqrt(np.maximum(sig_s[2 * s], 0.0)), 2.0 * tau_s[s]
        exc: FloatArray
        vxc: Any
        exc, vxc = dft.libxc.eval_xc("MGGA_X_R2SCAN", rho, spin=0, deriv=1)[:2]
        ex2[s], vrx[s], vsx[s], vtx[s] = exc, vxc[0], vxc[1], vxc[3]
    ga: FloatArray
    gb: FloatArray
    ga, gb = _grad_from_sigma(sig_s)
    ra: FloatArray
    rb: FloatArray
    ra, rb = np.zeros((6, P)), np.zeros((6, P))
    ra[0], ra[1:4], ra[5] = ns_s[0], ga, tau_s[0]
    rb[0], rb[1:4], rb[5] = ns_s[1], gb, tau_s[1]
    exc, vxc = dft.libxc.eval_xc("MGGA_C_R2SCAN", (ra, rb), spin=1, deriv=1)[:2]
    return ex2, vrx, vsx, vtx, exc, np.asarray(vxc[0]).T, np.asarray(vxc[1]).T, np.asarray(vxc[3]).T


# -------------------------------------------------------------------------------------------------------------------- the NumInt
def _point_terms(
    model: K.LocusModel, nw: NodeWeights, rho: FloatArray, conv: FloatArray, rks: bool
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """rho (S, 6, P) [n, grad n, lapl, tau] of the density sets (RKS: total; UKS: up, dn), conv (P, A) node convolutions of the total
    density -> e (P), wv (S, 6, P) = dE/d[n, grad n, lapl, tau] of the sets, F (P, A) = dE/d conv"""
    P: int = rho.shape[-1]
    rs: FloatArray = np.stack([0.5 * rho[0], 0.5 * rho[0]]) if rks else rho
    n_s: FloatArray = np.clip(rs[:, 0], 0.0, None)
    gr: FloatArray = rs[:, 1:4]
    tau: FloatArray = np.clip(rs[:, 5], 0.0, None)
    sig: FloatArray = np.array([(gr[0] ** 2).sum(0), (gr[0] * gr[1]).sum(0), (gr[1] ** 2).sum(0)])
    n: FloatArray = n_s.sum(0)
    nsafe: FloatArray = np.maximum(n, 1e-30)
    v: FloatArray
    dv: FloatArray
    I_t: IntArray
    T_t: FloatArray
    v, dv, I_t, T_t = nw.channel(n, conv, CS)
    x_t: FloatArray = v / nsafe
    dx_t: FloatArray = (dv - x_t) / nsafe
    x_s: FloatArray = np.stack([x_t, x_t], 1)  # descriptors of the total density in both channels
    o: dict[str, FloatArray] = K.constrained_terms_spin(
        model, "r2SCAN", n_s, sig, tau, x_s, x_t, base_eval=base_eval_spin
    )
    g_t: FloatArray = o["dedx_tot"]
    live: BoolArray = n > N_MIN
    g_t[:, ~live] = 0.0
    vloc: FloatArray = (g_t * dx_t).sum(0)
    F: FloatArray = nw.adjoint(g_t / nsafe, I_t, T_t)
    F[~live] = 0.0
    wv: FloatArray = np.zeros((2, 6, P))
    s: int
    for s in range(2):
        wv[s, 0] = o["dedn"][s] + vloc
        wv[s, 1:4] = 2.0 * o["dedsigma"][2 * s] * gr[s] + o["dedsigma"][1] * gr[1 - s]
        wv[s, 5] = o["dedtau"][s]
    if rks:
        wv = 0.5 * (wv[0] + wv[1])[None]
    return o["e"], wv, F


class LocusNumInt(numint.NumInt):  # type: ignore[misc]
    """PySCF NumInt for LOCUS (E_xc without the VV10 part, which PySCF adds from mf.nlc = 'VV10' with nlc_coeff (vv10, 1))"""

    def __init__(
        self,
        shell: Optional[dict[str, float]] = None,
        vv10: Optional[Sequence[float]] = VV10,
        nodes: Sequence[float] = DENSITY_NODES,
        beta: float = 1.6,
        aux_blk: int = 4096,
        i3_memory: float = 4000,
    ) -> None:
        super().__init__()
        self.model: K.LocusModel = K.LocusModel(shell)
        self.vv10: Optional[tuple[float, ...]] = tuple(vv10) if vv10 else None
        self.nodes: tuple[float, ...]
        self.beta: float
        self.aux_blk: int
        self.i3_memory: float
        self.nodes, self.beta, self.aux_blk, self.i3_memory = tuple(nodes), beta, aux_blk, i3_memory
        self.nw: NodeWeights = NodeWeights(nodes)
        self._aux: Optional[AuxConv]
        self._aux_key: Optional[tuple[bytes, str, int]]
        self._aux, self._aux_key = None, None
        self.timings: dict[str, float] = {}

    def nlc_coeff(self, xc_code: str) -> tuple[tuple[tuple[float, ...], float], ...]:
        return ((self.vv10, 1.0),) if self.vv10 else ()

    def hybrid_coeff(self, xc_code: str, spin: int = 0) -> float:
        return 0.0

    def rsh_coeff(self, xc_code: str) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def _xc_type(self, xc_code: str) -> str:
        return "MGGA"

    def eval_xc(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("LocusNumInt: LOCUS is evaluated in nr_rks / nr_uks only")

    eval_xc_eff: Callable[..., NoReturn] = eval_xc

    def aux_cache(self, mol: Any) -> AuxConv:
        key: tuple[bytes, str, int] = (mol.atom_coords().tobytes(), str(mol._basis), mol.nao)
        if self._aux_key != key:
            self._aux = AuxConv(mol, self.nodes, self.beta, max_memory=self.i3_memory)
            self._aux_key = key
        return cast(AuxConv, self._aux)

    def nr_rks(
        self,
        mol: Any,
        grids: Any,
        xc_code: str,
        dms: FloatArray,
        relativity: int = 0,
        hermi: int = 1,
        max_memory: float = 2000,
        verbose: Optional[int] = None,
    ) -> tuple[float, float, FloatArray]:
        dm: FloatArray = np.asarray(dms)
        n: float
        exc: float
        v: FloatArray
        n, exc, v = cast("tuple[float, float, FloatArray]", self._xc(mol, grids, dm[None], True, max_memory))
        return n, exc, v[0]

    def nr_uks(
        self,
        mol: Any,
        grids: Any,
        xc_code: str,
        dms: FloatArray,
        relativity: int = 0,
        hermi: int = 1,
        max_memory: float = 2000,
        verbose: Optional[int] = None,
    ) -> tuple[FloatArray, float, FloatArray]:
        return cast("tuple[FloatArray, float, FloatArray]", self._xc(mol, grids, np.asarray(dms), False, max_memory))

    def _xc(
        self, mol: Any, grids: Any, dms: FloatArray, rks: bool, max_memory: float = 2000
    ) -> tuple[Union[float, FloatArray], float, FloatArray]:
        t0: float = time.time()
        dms = 0.5 * (dms + dms.transpose(0, 2, 1))
        Sn: int
        nao: int
        Sn, nao = len(dms), mol.nao
        if grids.coords is None:
            grids.build(with_non0tab=True)
        ng: int = len(grids.weights)
        aux: AuxConv = self.aux_cache(mol)
        aux.set_coeff(aux.fit(dms.sum(0)[None]))
        rho: FloatArray = np.zeros((Sn, 6, ng))
        p0: int = 0
        ao: FloatArray
        mask: Any
        weight: FloatArray
        coords: FloatArray
        for ao, mask, weight, coords in self.block_loop(mol, grids, nao, 1, max_memory):
            p1: int = p0 + weight.size
            s: int
            for s in range(Sn):
                r: FloatArray = numint.eval_rho(mol, ao, dms[s], mask, xctype="MGGA", with_lapl=False)
                rho[s, 0:4, p0:p1] = r[0:4]
                rho[s, 5, p0:p1] = r[-1]
            p0 = p1
        n_tot: FloatArray = np.clip(rho[:, 0], 0.0, None).sum(0)
        live: IntArray = np.where(n_tot > N_MIN)[0]
        coords = grids.coords[live]
        conv: FloatArray = (
            np.concatenate(
                [aux.forward(coords[q0 : q0 + self.aux_blk]) for q0 in range(0, len(live), self.aux_blk)], 0
            )[:, :, 0]
            if len(live)
            else np.zeros((0, len(self.nodes)))
        )
        w: FloatArray = grids.weights
        nelec: FloatArray = rho[:, 0] @ w
        e_l: FloatArray
        wv_l: FloatArray
        F_l: FloatArray
        e_l, wv_l, F_l = _point_terms(self.model, self.nw, rho[:, :, live], conv, rks)
        excsum: float = float(w[live] @ e_l)
        wv: FloatArray = np.zeros((Sn, 6, ng))
        wv[:, :, live] = wv_l * w[live]
        F: FloatArray = F_l * w[live][:, None]
        vmat: FloatArray = np.zeros((Sn, nao, nao))
        p0 = 0
        coords_b: FloatArray
        for ao, mask, weight, coords_b in self.block_loop(mol, grids, nao, 1, max_memory):
            p1 = p0 + weight.size
            for s in range(Sn):
                v: FloatArray = wv[s, :, p0:p1]
                aow: FloatArray = (
                    ao[0] * (0.5 * v[0])[:, None]
                    + ao[1] * v[1][:, None]
                    + ao[2] * v[2][:, None]
                    + ao[3] * v[3][:, None]
                )
                Vh: FloatArray = ao[0].T @ aow
                wt: FloatArray = 0.25 * v[5]
                c: int
                for c in range(1, 4):
                    Vh += ao[c].T @ (ao[c] * wt[:, None])
                vmat[s] += Vh + Vh.T
            p0 = p1
        q0: int
        for q0 in range(0, len(live), self.aux_blk):
            aux.adjoint(coords[q0 : q0 + self.aux_blk], F[q0 : q0 + self.aux_blk][:, None, :])
        vmat += aux.fock(aux.gather_b())
        self.timings["xc"] = time.time() - t0
        self.last: dict[str, Union[float, FloatArray]] = dict(E_xc=excsum, nelec=nelec.copy())
        if rks:
            return float(nelec[0]), excsum, vmat
        return nelec, excsum, vmat


# --------------------------------------------------------------------------------------------------------------------- SCF protocol
class _VV10NumInt(numint.NumInt):  # type: ignore[misc]
    """plain PySCF NumInt whose VV10 nlc uses (b, C) = vv10 (the r2SCAN + VV10 reference)"""

    def __init__(self, vv10: Sequence[float] = VV10) -> None:
        super().__init__()
        self.vv10: tuple[float, ...] = tuple(vv10)

    def nlc_coeff(self, xc_code: str) -> tuple[tuple[tuple[float, ...], float]]:
        return ((self.vv10, 1.0),)


class ADIISDIIS(lib.diis.DIIS):  # type: ignore[misc]
    """ADIIS while the DIIS error ||FDS - SDF|| > 0.1, CDIIS below (both histories kept every cycle)"""

    switch: float = 0.1

    def __init__(self, mf: Any = None, filename: Optional[str] = None, Corth: Optional[FloatArray] = None) -> None:
        from pyscf.scf import diis as sdiis

        super().__init__(mf, filename)
        self.sdiis: ModuleType
        self.cd: Any
        self.ad: Any
        self._synced: bool
        self.sdiis, self.cd, self.ad, self._synced = sdiis, sdiis.CDIIS(mf), sdiis.ADIIS(), False

    def update(
        self, s: FloatArray, d: FloatArray, f: FloatArray, mf: Any, h1e: FloatArray, vhf: FloatArray, **kw: Any
    ) -> FloatArray:  # kw: f_prev of newer PySCF versions
        if not self._synced:
            self.cd.space = self.ad.space = self.space
            self._synced = True
        err: float = float(np.linalg.norm(self.sdiis.get_err_vec(s, d, f)))
        fc: FloatArray = self.cd.update(s, d, f, mf, h1e, vhf, **kw)
        fa: FloatArray = self.ad.update(s, d, f, mf, h1e, vhf, **kw)
        return fa if err > self.switch else fc


def robust_diis(mf: Any, space: int = 12) -> Any:
    mf.DIIS = ADIISDIIS
    mf.diis = True
    mf.diis_space = space
    return mf


def _ks(mol: Any, rks: Optional[bool], grid_level: int, nlc_grid_level: int, vv10: Optional[Sequence[float]]) -> Any:
    rks = (mol.spin == 0) if rks is None else rks
    mf: Any = dft.RKS(mol) if rks else dft.UKS(mol)
    mf.xc = "r2scan"
    mf.grids.level = grid_level
    if vv10:
        mf.nlc = "vv10"  # lower case: PySCF >= 2.4 compares with "vv10"; the coefficients come from nlc_coeff
        mf.nlcgrids.level = nlc_grid_level
    return mf


def make_ks(
    mol: Any,
    rks: Optional[bool] = None,
    grid_level: int = 4,
    nlc_grid_level: int = 1,
    vv10: Optional[Sequence[float]] = VV10,
    shell: Optional[dict[str, float]] = None,
) -> Any:
    """LOCUS mean field (RKS for closed shells, UKS otherwise), ADIIS -> CDIIS"""
    mf: Any = robust_diis(_ks(mol, rks, grid_level, nlc_grid_level, vv10))
    mf._numint = LocusNumInt(shell=shell, vv10=vv10)
    return mf


def run_locus(
    mol: Any,
    rks: Optional[bool] = None,
    grid_level: int = 4,
    nlc_grid_level: int = 1,
    conv_tol: float = 1e-8,
    max_cycle: int = 100,
    start: Optional[str] = "pbe",
    start_grid_level: int = 2,
    start_conv_tol: float = 1e-6,
    start_max_cycle: int = 60,
    retry_shift: float = 0.25,
    diis_space: int = 12,
    init_guess: str = "minao",
    verbose: int = 0,
    log: bool = True,
) -> tuple[float, Any, dict[str, Any]]:
    """the protocol of the paper's molecular calculations: PBE on a coarse grid -> r2SCAN + VV10 -> LOCUS on the same grids, ADIIS -> CDIIS,
    one level-shifted retry of a stage that does not converge.  Returns (e_tot, mf, info)."""
    rks = (mol.spin == 0) if rks is None else rks
    info: dict[str, Any] = dict(stages=[])

    def run(name: str, mf: Any, dm: Optional[FloatArray]) -> Any:
        t0: float = time.time()
        mf.conv_tol, mf.max_cycle, mf.verbose, mf.diis_space, mf.init_guess = (
            conv_tol,
            max_cycle,
            verbose,
            diis_space,
            init_guess,
        )
        mf.kernel(dm0=dm)
        if not mf.converged and retry_shift:
            from pyscf.scf import diis as sdiis

            mf.DIIS = sdiis.CDIIS
            mf.level_shift = retry_shift
            mf.kernel(dm0=mf.make_rdm1())
            mf.level_shift = 0.0
            if not mf.converged:
                mf.kernel(dm0=mf.make_rdm1())
            robust_diis(mf, diis_space)
        rec: dict[str, Any] = dict(
            stage=name, converged=bool(mf.converged), e_tot=float(mf.e_tot), sec=time.time() - t0
        )
        info["stages"].append(rec)
        if log:
            print(
                f"[run_locus] {name:<12s} {'converged' if rec['converged'] else 'NOT CONVERGED'}: E = {rec['e_tot']:.10f} ({rec['sec']:.0f} s)",
                flush=True,
            )
        return mf

    dm: Optional[FloatArray] = None
    if start:
        mf0: Any = robust_diis(dft.RKS(mol) if rks else dft.UKS(mol), diis_space)
        mf0.xc, mf0.grids.level, mf0.conv_tol, mf0.conv_tol_grad, mf0.max_cycle = (
            start,
            start_grid_level,
            start_conv_tol,
            3e-3,
            start_max_cycle,
        )
        mf0.verbose, mf0.init_guess = verbose, init_guess
        mf0.kernel()
        info["stages"].append(dict(stage="start", converged=bool(mf0.converged), e_tot=float(mf0.e_tot)))
        dm = mf0.make_rdm1()
    mf1: Any = _ks(mol, rks, grid_level, nlc_grid_level, VV10)
    mf1._numint = _VV10NumInt(VV10)
    robust_diis(mf1, diis_space)
    mf1 = run("r2scan+vv10", mf1, dm)
    info["e_r2scan_vv10"] = float(mf1.e_tot)
    mf: Any = make_ks(mol, rks, grid_level, nlc_grid_level)
    mf.grids, mf.nlcgrids = mf1.grids, mf1.nlcgrids
    mf = run("locus", mf, mf1.make_rdm1())
    info.update(converged=bool(mf.converged), e_locus=float(mf.e_tot))
    return float(mf.e_tot), mf, info
