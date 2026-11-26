"""MPI layout of the exporter: k-points and, within a k-point, real-space domains (GPAW parallel={'band': 1}).

Per-k data are computed on the ranks of the k-point and gathered in the serial order u = k * nspins + s; integrals over
the grid or over plane waves and sums over atoms are first summed over the domain communicator.  Ranks of one k-point
hold identical per-k scalars (eigenvalues, occupations).  GPL-3.0-or-later (uses GPAW).
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable
from typing import Any, TypeVar, cast

import numpy as np

from ._types import FloatArray, IntArray

ItemT = TypeVar("ItemT")

LAYOUT: dict[str, int] = {"band": 1}


def check_layout(calc: Any) -> None:
    wfs: Any = calc.wfs
    if wfs.bd.comm.size != 1:
        raise NotImplementedError("the exporter does not use band parallelisation: run with GPAW parallel={'band': 1}")


def gcomm(calc: Any) -> Any:
    """the domain communicator: the ranks that share one k-point (size 1 without domain decomposition)"""
    return calc.wfs.gd.comm


def domain_sum(calc: Any, a: Any) -> None:
    """in-place sum over the domain ranks (a numpy array)"""
    comm: Any = gcomm(calc)
    if comm.size > 1:
        comm.sum(a)


def gather_root(comm: Any, obj: ItemT) -> list[ItemT] | None:
    """the objects of all ranks of comm (pickled) on its rank 0, None on the others"""
    if comm.size == 1:
        return [obj]
    data: IntArray = np.frombuffer(pickle.dumps(obj), dtype=np.uint8).copy()
    size: IntArray = np.array([data.size], np.int64)
    if comm.rank != 0:
        comm.send(size, 0, tag=11)
        comm.send(data, 0, tag=12)
        return None
    out: list[ItemT] = [obj]
    r: int
    for r in range(1, comm.size):
        n: IntArray = np.empty(1, np.int64)
        comm.receive(n, r, tag=11)
        buf: IntArray = np.empty(int(n[0]), np.uint8)
        comm.receive(buf, r, tag=12)
        out.append(cast(ItemT, pickle.loads(buf.tobytes())))
    return out


def kcomm(calc: Any) -> Any:
    return calc.wfs.kd.comm


def ks0(calc: Any) -> int:
    """global index u of this rank's first (k-point, spin) pair: GPAW distributes the k-points (u = k * nspins + s, spins not split)"""
    return int(calc.wfs.kd.k0) * int(calc.wfs.nspins)


def nks(calc: Any) -> int:
    """number of (spin, k-point) pairs over all ranks"""
    return int(calc.wfs.kd.nibzkpts * calc.wfs.nspins)


def local(calc: Any, u: int) -> int | None:
    """the rank-local index of global k-point u, or None"""
    q: int = int(u) - ks0(calc)
    return q if 0 <= q < len(calc.wfs.kpt_u) else None


def is_master(calc: Any) -> bool:
    return cast(bool, calc.wfs.world.rank == 0)


def gather(calc: Any, items: Iterable[ItemT]) -> list[ItemT]:
    """the lists of all ranks concatenated in rank order (= global k order for per-k lists), on every rank"""
    from gpaw.mpi import broadcast

    comm: Any = kcomm(calc)
    if comm.size == 1:
        return list(items)
    out: list[ItemT] = []
    r: int
    for r in range(comm.size):
        out += broadcast(list(items) if comm.rank == r else None, root=r, comm=comm)
    return out


def gather_array(calc: Any, rows: Iterable[Any]) -> FloatArray:
    """per-k rows (a list over this rank's k-points) -> array over all k-points"""
    return np.array(gather(calc, [np.asarray(x) for x in rows]))


def owner_value(calc: Any, value: ItemT | None, has: bool) -> ItemT:
    """a value known on the rank that owns it (has=True) -> on every rank"""
    from gpaw.mpi import broadcast

    comm: Any = kcomm(calc)
    if comm.size == 1:
        return cast(ItemT, value)
    who: IntArray = np.array([comm.rank if has else -1])
    comm.max(who)
    return cast(ItemT, broadcast(value if comm.rank == int(who[0]) else None, root=int(who[0]), comm=comm))
