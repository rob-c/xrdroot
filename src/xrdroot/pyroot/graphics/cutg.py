"""``TCutG``: a closed polygon of a 2-D plane, and a histogram drawn only within one.

A ``TCutG`` is a graph whose points go round a region; ``IsInside`` says
whether a point is in it, by ``TMath::IsInside``'s crossing count. It is
kept by name among ``gROOT``'s specials, so a histogram drawn with the
cut's name in square brackets - ``"col [cut]"``, or ``"[-cut]"`` for what
is outside it - finds it, and is drawn with only the cells whose centres
the cut holds, as ``THistPainter`` draws it.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ..core.graphs import TGraph
from ..core.messages import Warning as root_warning

__all__ = ["TCutG"]

#: A cut named in a draw option: ``[name]``, or ``[-name]`` for its outside.
CUT_NAME = re.compile(r"\[(-?)([^\]]*)\]")


def is_inside(xs: Any, ys: Any, x: Any, y: Any) -> Any:
    """``TMath::IsInside``: whether each ``(x, y)`` is inside the polygon, by crossings."""
    px, py = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    odd = np.zeros(np.broadcast(x, y).shape, dtype=bool)
    for i in range(len(px)):
        j = i - 1
        spans = ((py[i] < y) & (py[j] >= y)) | ((py[j] < y) & (py[i] >= y))
        with np.errstate(divide="ignore", invalid="ignore"):
            cross = px[i] + (y - py[i]) / (py[j] - py[i]) * (px[j] - px[i]) < x
        odd ^= spans & cross
    return odd


class TCutG(TGraph):
    """``TCutG(name, n[, x, y])``: a polygon of ``n`` points, kept by name in ``gROOT``."""

    CLASS_TITLE = "A Graphical cut."

    def __init__(self, name: Any = "", n: int = 0, *points: Any) -> None:
        from ..core.troot import gROOT

        super().__init__(int(n), *points)
        self.SetName(str(name))
        self._variables = ["x", "y"]
        specials = gROOT.GetListOfSpecials()
        old = specials.FindObject(str(name))
        if old is not None:
            root_warning("TCutG", f"Replacing existing TCutG: {name} (Potential memory leak).")
            specials.Remove(old)
        specials.Add(self)

    def IsInside(self, x: float, y: float) -> int:
        return int(is_inside(self.GetX(), self.GetY(), x, y))

    def SetVarX(self, name: str) -> None:
        self._variables[0] = str(name)

    def SetVarY(self, name: str) -> None:
        self._variables[1] = str(name)

    def GetVarX(self) -> str:
        return self._variables[0]

    def GetVarY(self) -> str:
        return self._variables[1]

    def Area(self) -> float:
        """The polygon's signed area, by ROOT's sum over its sides."""
        x, y = self.GetX(), self.GetY()
        return 0.5 * float(np.sum((x[:-1] - x[1:]) * (y[:-1] + y[1:])))

    def Center(self, cx: Any = None, cy: Any = None) -> tuple[float, float]:
        """The polygon's centre, by Stokes' theorem round its border, into ``cx`` and ``cy``."""
        from ..core.refs import store

        x, y = self.GetX(), self.GetY()
        t = 2 * x[:-1] * y[:-1] + y[:-1] * x[1:] + x[:-1] * y[1:] + 2 * x[1:] * y[1:]
        area = 6 * self.Area()
        found = (float(np.sum((x[:-1] - x[1:]) * t)) / area,
                 float(np.sum((y[1:] - y[:-1]) * t)) / area)  # fmt: skip
        store(cx, found[0])
        store(cy, found[1])
        return found

    def IntegralHist(self, h: Any, option: str = "") -> float:
        """The sum of ``h``'s cells whose centres are inside, times their areas with ``width``."""
        nx, ny = h.GetNbinsX(), h.GetNbinsY()
        xaxis, yaxis = h.GetXaxis(), h.GetYaxis()
        total = 0.0
        for i in range(1, nx + 1):
            for j in range(1, ny + 1):
                if self.IsInside(xaxis.GetBinCenter(i), yaxis.GetBinCenter(j)):
                    width = xaxis.GetBinWidth(i) * yaxis.GetBinWidth(j) if "width" in option else 1
                    total += h.GetBinContent(i, j) * width
        return total


def _kept(histogram: Any, cut: TCutG, outside: bool) -> Any:
    """A copy of a 2-D histogram with the cells the cut excludes emptied."""
    made = histogram.copy()
    xs, ys = (np.concatenate(([0.0], axis.centers(), [0.0])) for axis in made.axes[:2])
    cx, cy = np.meshgrid(xs, ys)  # rows y, columns x: ROOT's global bins in order
    inside = is_inside(cut.GetX(), cut.GetY(), cx, cy)
    inner = np.zeros(inside.shape, dtype=bool)
    inner[1:-1, 1:-1] = True
    cells = made._cells()
    cells[(inner & (inside == outside)).ravel()] = 0
    return made


def cut_drawn(obj: Any, option: str) -> tuple[Any, str]:
    """What is drawn of ``obj`` with ``option``: a 2-D histogram's cells within the cuts its
    option names - ``[cut]``, or ``[-cut]`` for outside - and the option without them."""
    from ..core.troot import gROOT

    named = CUT_NAME.findall(option)
    if not named:
        return obj, option
    stripped = CUT_NAME.sub("", option)
    xrd = getattr(obj, "_xrd", obj)
    if len(getattr(xrd, "axes", ())) != 2:
        return obj, stripped
    for sign, name in named:
        cut = gROOT.GetListOfSpecials().FindObject(name.strip())
        xrd = xrd if cut is None else _kept(xrd, cut, sign == "-")
    return xrd, stripped
