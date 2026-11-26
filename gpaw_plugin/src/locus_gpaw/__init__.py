"""LOCUS for GPAW: r2SCAN + rVV10 (b = 12) + the shell term, self-consistent in the PAW method.  GPL-3.0-or-later.

    from locus_gpaw import LOCUS
    calc = GPAW(mode=PW(600), xc=LOCUS(), ...)

Importing the package registers the names with GPAW's xc factory, so that ground states written with LOCUS can be read back:
{'name': 'LOCUS'} (and {'name': 'LOCUS', 'shell': False} for the reference r2SCAN + rVV10), and the {'name': 'DCONV', 'model': ...} form
whose model is LOCUS or r2SCAN + rVV10.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union, cast

if TYPE_CHECKING:
    from .dconv import DensityConvMGGA

__version__ = "0.1.0"


def LOCUS(shell: bool = True, vv10_b: float = 12.0, lowdens: bool = True) -> DensityConvMGGA:
    """the LOCUS functional (shell=False: r2SCAN + rVV10 alone, the reference functional of the d-state subtraction of E_S)"""
    from .dconv import DensityConvMGGA
    from .kernel import LocusModel

    model: LocusModel = LocusModel(None if shell else dict(c=0.0))
    xc: DensityConvMGGA = DensityConvMGGA(model, [4.0], "r2SCAN", vv10={"b": float(vv10_b)})
    xc.kernel.lowdens = bool(lowdens)
    xc.name = "LOCUS" if shell else "r2SCAN+rVV10"
    spec: dict[str, Union[str, bool, float]] = {"name": "LOCUS"}
    if not shell:
        spec["shell"] = False
    if float(vv10_b) != 12.0:
        spec["vv10_b"] = float(vv10_b)
    if not lowdens:
        spec["lowdens"] = False
    xc._run_spec = spec
    return xc


def _from_spec(kernel: dict[str, Any]) -> Optional[DensityConvMGGA]:
    """LOCUS from an xc dict, or None"""
    from pathlib import Path

    name: Any = kernel.get("name")
    if name == "LOCUS":
        return LOCUS(
            shell=kernel.get("shell", True), vv10_b=kernel.get("vv10_b", 12.0), lowdens=kernel.get("lowdens", True)
        )
    if name == "DCONV" and kernel.get("model"):
        stem: str = Path(str(kernel["model"])).parent.name
        if stem in ("locus",):
            return LOCUS()
        if stem in ("r2scan_vv10",):
            return LOCUS(shell=False)
        return None  # another DCONV model: left to the xc factory
    return None


def register() -> None:
    import gpaw.calculator
    import gpaw.xc

    if getattr(gpaw.xc.XC, "_locus", False):
        return
    orig: Any = gpaw.xc.XC

    def XC(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(kernel, dict):
            xc: Optional[DensityConvMGGA] = _from_spec(kernel)
            if xc is not None:
                return xc
        return orig(kernel, *args, **kwargs)

    cast(Any, XC)._locus = True
    mod: Any
    for mod in (gpaw.xc, gpaw.calculator):
        mod.XC = XC
    name: str
    for name in ("gpaw.xc.tools", "gpaw.setup"):
        import importlib

        try:
            m: Any = importlib.import_module(name)
            if getattr(m, "XC", None) is orig:
                m.XC = XC
        except ImportError:
            pass


register()
