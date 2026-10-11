"""``TGraphSmooth``: a graph's points interpolated or smoothed, each method a new ``TGraph``.

``Approx`` is R's ``approx`` - linear, or a step - through the points with
those of one x merged as ``ties`` says; ``SmoothKern`` R's ``ksmooth``,
``SmoothLowess`` R's ``lowess`` and ``SmoothSuper`` Friedman's super
smoother (:mod:`xrdroot.graphsmooth`, :mod:`xrdroot.supsmu`). As ROOT's do,
the smoothers sort the input graph's points by x where it is, and each call
hands back a graph the smoother keeps, replaced by the next call's.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ... import graphsmooth as gs
from ...supsmu import supsmu
from .objects import TNamed

__all__ = ["TGraphSmooth"]


def _error(text: str) -> None:
    """ROOT's complaint, which it writes to ``std::cout``, and the null graph it returns."""
    print(f"Error: {text}")


def _none(given: Any) -> bool:
    """A pointer a macro passed as ``nullptr`` - or ``0``."""
    return given is None or (np.ndim(given) == 0 and given == 0)


def _graph(x: Any, y: Any) -> Any:
    from .graphs import TGraph

    made = TGraph(len(x), np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return made


class TGraphSmooth(TNamed):
    """The smoother, and the graph it made last."""

    CLASS_TITLE = "Graph Smoother"

    def __init__(self, name: str = "", title: str = "") -> None:
        super().__init__(str(name), str(title))
        self._out: Any = None

    def _sorted_input(self, grin: Any) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """``Smoothin``: the input graph's points put in order of x, where they are."""
        order = np.argsort(np.asarray(grin.GetX(), dtype=np.float64), kind="stable")
        x, y = np.asarray(grin.GetX())[order].copy(), np.asarray(grin.GetY())[order].copy()
        for i, (xi, yi) in enumerate(zip(x, y, strict=True)):
            grin.SetPoint(i, float(xi), float(yi))
        return x, y

    def _kept(self, x: Any, y: Any) -> Any:
        self._out = _graph(x, y)
        return self._out

    def Approx(self, grin: Any, option: str = "linear", nout: int = 50, xout: Any = None,
               yleft: float = 0.0, yright: float = 0.0, rule: int = 0, f: float = 0.0,
               ties: str = "mean") -> Any:  # fmt: skip
        """``Approx``: the points interpolated at ``nout`` x - spread evenly, or ``xout``."""
        text = str(option).lower()
        kind = 1 if "linear" in text else 2 if "constant" in text else 0
        tie = gs.TIES.get(str(ties).lower())
        if not 0 <= f <= 1:
            return _error("Invalid f value")
        if tie is None:
            return _error(f"Method not known: {ties}")
        x, y = gs.collapse_ties(grin.GetX(), grin.GetY(), tie)
        low, high = (y[0], y[-1]) if rule == 2 else (yleft, yright)
        if _none(xout):  # as many points as asked for, or as the input has, if more
            points = list(gs.spread(x[0], x[-1], max(int(nout), len(x))))
        else:
            points = [float(xout[i]) for i in range(int(nout))]
        values = [gs.approx1(v, f, x, y, kind, float(low), float(high)) for v in points]
        return self._kept(points, values)

    def SmoothKern(self, grin: Any, option: str = "normal", bandwidth: float = 0.5,
                   nout: int = 100, xout: Any = None) -> Any:  # fmt: skip
        """``SmoothKern``: the kernel smoother at ``nout`` x, or at ``xout`` in order."""
        kernel = 2 if "normal" in str(option).lower() else 1
        x, y = self._sorted_input(grin)
        if _none(xout):
            points = gs.spread(x[0], x[-1], max(int(nout), len(x)))
        else:
            points = np.sort(np.asarray(xout, dtype=np.float64)[: int(nout)], kind="stable")
        return self._kept(points, gs.ksmooth(x, y, points, kernel, float(bandwidth)))

    def SmoothLowess(self, grin: Any, option: str = "", span: float = 0.67, iter: int = 3,
                     delta: float = 0.0) -> Any:  # fmt: skip
        """``SmoothLowess``: R's ``lowess`` at each point, ``delta`` a hundredth of the range."""
        x, y = self._sorted_input(grin)
        delta = delta if delta != 0 else 0.01 * abs(x[-1] - x[0])
        return self._kept(x, gs.lowess(x, y, float(span), int(iter), float(delta)))

    def SmoothSuper(self, grin: Any, option: str = "", bass: float = 0.0, span: float = 0.0,
                    isPeriodic: bool = False, w: Any = None) -> Any:  # fmt: skip
        """``SmoothSuper``: Friedman's super smoother at each point, ``bass`` its tone."""
        if span < 0 or span > 1:
            return _error("Span must be between 0 and 1")
        x, y = self._sorted_input(grin)
        if isPeriodic and (x[0] < 0 or x[-1] > 1):
            return _error("x must be between 0 and 1 for periodic smooth")
        weights = np.ones(len(x)) if w is None else np.asarray(w, dtype=np.float64)[: len(x)]
        fit = supsmu(x, y, weights, 2 if isPeriodic else 1, float(span), float(bass))
        return self._kept(x, fit)
