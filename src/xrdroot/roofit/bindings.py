"""``RooFit::bindFunction`` and ``bindPdf``: a plain function, or a ``TF1``, as a RooFit node.

A function of one to four numbers becomes a ``RooCFunctionNBinding`` - a
``RooCFunctionNPdfBinding`` as a density - whose variables are its proxies
``x``, ``y``, ``z`` and ``w``, and whose value is the function called at
their values, point by point; a ``TF1`` (``TF2``, ``TF3``) becomes a
``RooTFnBinding`` of its observables and, if given, its parameters. RooFit
integrates and generates them as any node with no closed forms, numerically.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from .collections import as_list
from .pdf import RooAbsPdf
from .real import Context, RooAbsReal

__all__ = ["bind_function", "bind_pdf"]

#: The proxies' names of the bound function's arguments, in order.
ARGUMENTS = ("x", "y", "z", "w")


#: The functions RooFit knows by name, and their arguments' - which name the proxies - as
#: ROOT 6.40 prints them when a macro binds them; any other is shown by its address. A
#: Python script hands ROOT a wrapper of its own, which ROOT knows by its address only.
REGISTERED = {
    ("TMath", "Erf"): ("x",), ("TMath", "Erfc"): ("x",), ("TMath", "Gamma"): ("z",),
    ("TMath", "Freq"): ("x",), ("TMath", "LnGamma"): ("z",), ("TMath", "BesselJ0"): ("x",),
    ("TMath", "BesselI0"): ("x",), ("TMath", "DiLog"): ("x",), ("TMath", "ErfInverse"): ("x",),
    ("TMath", "StruveH0"): ("x",), ("TMath", "Beta"): ("p", "q"),
    ("TMath", "Poisson"): ("x", "par"), ("Math", "erf"): ("x",), ("Math", "lgamma"): ("x",),
}


def _from_macro() -> bool:
    """Whether the call came from a translated macro - its module ``__cint__`` - rather than
    from Python."""
    frame = sys._getframe(1)
    while frame is not None and str(frame.f_globals.get("__name__", "")).startswith("xrdroot"):
        frame = frame.f_back  # type: ignore[assignment]
    return frame is not None and frame.f_globals.get("__name__") == "__cint__"


def _registered(func: Any) -> tuple[str, tuple[str, ...]] | None:
    """The C++ name and argument names RooFit has for ``func``, if a macro bound it."""
    if not _from_macro():
        return None
    import xrdroot.pyroot as ROOT

    for (space, name), arguments in REGISTERED.items():
        if getattr(getattr(ROOT, space), name, None) is func:
            return f"{'ROOT::Math' if space == 'Math' else space}::{name}", arguments
    return None


class _CFunction:
    """The binding of a function of numbers: its proxies and its value."""

    func: Any
    variables: list[Any]

    def _bind(self, func: Any, variables: list[Any]) -> None:
        if not 1 <= len(variables) <= len(ARGUMENTS):
            raise TypeError(f"bindFunction binds a function of one to four variables, not of "
                            f"{len(variables)}")  # fmt: skip
        known = _registered(func)
        self.func = func
        self._shown = known[0] if known else f"({hex(id(func))})"
        self._arguments = known[1] if known else ARGUMENTS
        pairs = zip(self._arguments, variables, strict=False)
        self.variables = [self._proxy(name, var) for name, var in pairs]  # type: ignore[attr-defined]

    def compute(self, ctx: Context) -> Any:
        values = np.broadcast_arrays(*(np.asarray(v.compute(ctx), dtype=np.float64)
                                       for v in self.variables))  # fmt: skip
        flat = [np.ravel(one) for one in values]
        found = np.array([self.func(*point) for point in zip(*flat, strict=True)], dtype=np.float64)
        return float(found[0]) if values[0].ndim == 0 else found.reshape(values[0].shape)

    def ClassName(self) -> str:
        """``RooCFunction3PdfBinding<double,double,double,double>``: ROOT's template's name."""
        kind = "PdfBinding" if isinstance(self, RooAbsPdf) else "Binding"
        types = ",".join(["double"] * (len(self.variables) + 1))
        return f"RooCFunction{len(self.variables)}{kind}<{types}>"

    def printArgs(self) -> str:
        pairs = zip(self._arguments, self.variables, strict=False)
        shown = "".join(f"{name}={var.GetName()} " for name, var in pairs)
        return f"[ function={self._shown} {shown}]"


class RooCFunctionBinding(_CFunction, RooAbsReal):
    """``RooCFunctionNBinding``: a function of up to four numbers, as a RooFit function."""

    def __init__(self, name: Any, title: Any, func: Any, variables: list[Any]) -> None:
        RooAbsReal.__init__(self, name, title)
        self._bind(func, variables)


class RooCFunctionPdfBinding(_CFunction, RooAbsPdf):
    """``RooCFunctionNPdfBinding``: a function of up to four numbers, as a RooFit density."""

    def __init__(self, name: Any, title: Any, func: Any, variables: list[Any]) -> None:
        RooAbsPdf.__init__(self, name, title)
        self._bind(func, variables)


class _TFn:
    """The binding of a ``TF1``: its observables, its parameters, and its value."""

    tf: Any
    obs: Any
    params: Any

    def _bind(self, tf: Any, obs: Any, params: Any) -> None:
        self.tf = tf
        self.obs = self._list_proxy("obs", as_list(obs))  # type: ignore[attr-defined]
        self.params = self._list_proxy("params", as_list(params or []))  # type: ignore[attr-defined]

    def compute(self, ctx: Context) -> Any:
        values = np.broadcast_arrays(*(np.asarray(v.compute(ctx), dtype=np.float64)
                                       for v in self.obs))  # fmt: skip
        columns = [np.ravel(one) for one in values]
        given = np.array([float(p.compute(ctx)) for p in self.params]) if len(self.params) else None
        column = columns[0] if len(columns) == 1 else np.stack(columns, axis=-1)
        found = np.asarray(self.tf._xrd.evaluate(column, given), dtype=np.float64)
        return float(found[0]) if values[0].ndim == 0 else found.reshape(values[0].shape)

    def printArgs(self) -> str:
        names = ",".join(one.GetName() for one in self.obs)
        params = ",".join(one.GetName() for one in self.params)
        return (f"[ TFn={{{self.tf.GetName()}={self.tf.GetTitle()}}} obs=({names}) "
                f"params=({params}) ]")  # fmt: skip


class RooTFnBinding(_TFn, RooAbsReal):
    """``RooTFnBinding``: a ``TF1``, ``TF2`` or ``TF3`` as a RooFit function."""

    def __init__(self, name: Any, title: Any, tf: Any, obs: Any, params: Any = None) -> None:
        RooAbsReal.__init__(self, name, title)
        self._bind(tf, obs, params)


class RooTFnPdfBinding(_TFn, RooAbsPdf):
    """``RooTFnPdfBinding``: a ``TF1``, ``TF2`` or ``TF3`` as a RooFit density."""

    def __init__(self, name: Any, title: Any, tf: Any, obs: Any, params: Any = None) -> None:
        RooAbsPdf.__init__(self, name, title)
        self._bind(tf, obs, params)


def _is_tf(first: Any) -> bool:
    return hasattr(first, "_xrd") and hasattr(first, "GetNdim")


def bind_function(*args: Any) -> Any:
    """``bindFunction(name, func, x, ...)`` or ``bindFunction(tf, obs[, params])``."""
    if _is_tf(args[0]):
        return RooTFnBinding(args[0].GetName(), args[0].GetTitle(), *args)
    name, func, *variables = args
    return RooCFunctionBinding(name, name, func, variables)


def bind_pdf(*args: Any) -> Any:
    """``bindPdf(name, func, x, ...)`` or ``bindPdf(tf, obs[, params])``."""
    if _is_tf(args[0]):
        return RooTFnPdfBinding(args[0].GetName(), args[0].GetTitle(), *args)
    name, func, *variables = args
    return RooCFunctionPdfBinding(name, name, func, variables)
