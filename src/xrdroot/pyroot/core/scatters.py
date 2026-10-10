"""``TScatter`` and ``TScatter2D``: points with a colour and a size each, by ROOT's methods.

The points live in the :class:`xrdroot.ScatterPlot` this stands for; the
frame is its ``GetHistogram``, whose axes ``GetXaxis`` and the rest hand
back, so ``GetZaxis()->SetRangeUser`` sets the colour scale's ends as it
does in ROOT. ``Draw("logc")`` and ``Draw("logs")`` colour and size the
points by the logarithms of their values.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np

from ...scatterplot import ScatterPlot
from .objects import TAttFill, TAttLine, TAttMarker, TNamed
from .wrapping import adopt, register, unwrap, wrap

__all__ = ["TScatter", "TScatter2D"]


def _columns(args: tuple[Any, ...], columns: int) -> list[Any]:
    """``(n, x, y, ...)``: the first ``n`` of each array given, zeros for a position array
    left out; ``(n)`` alone or nothing is ``n`` points at the origin."""
    if not args:
        return [np.zeros(0)] * columns
    count = int(args[0]) if np.ndim(args[0]) == 0 else len(args[0])
    given = [np.asarray(a, dtype=np.float64)[:count] for a in args[1:] if a is not None]
    while len(given) < columns:
        given.append(np.zeros(count))
    return given


class TScatter(TNamed, TAttLine, TAttFill, TAttMarker):
    """``TScatter(n, x, y[, colors[, sizes]])``: points in a plane, coloured and sized."""

    CLASS_TITLE = "A scatter plot"
    IN_SPACE = False

    def __init__(self, *args: Any) -> None:
        TNamed.__init__(self, "Scatter", "Scatter")
        columns = 3 if self.IN_SPACE else 2
        given = _columns(args, columns)
        z = given[2] if self.IN_SPACE else None
        extra = given[columns:]
        self._xrd = ScatterPlot.new(
            "Scatter", given[0], given[1], z, colors=extra[0] if extra else None,
            sizes=extra[1] if len(extra) > 1 else None, title="Scatter",
        )  # fmt: skip

    def _adopted(self, xrd: Any) -> None:
        self._xrd = xrd
        TNamed.__init__(self, xrd.name, xrd.title)

    def _attribute_holder(self) -> Any:
        return self._xrd.members

    def ClassName(self) -> str:
        return self._xrd.classname

    def SetName(self, name: Any) -> None:
        TNamed.SetName(self, name)
        self._xrd.members["TNamed"]["fName"] = str(name)

    def SetTitle(self, title: Any = "") -> None:
        """``SetTitle("title;x;y;z")``: the frame, if made already, is made again from it."""
        TNamed.SetTitle(self, title)
        self._xrd.members["TNamed"]["fTitle"] = str(title)
        self._xrd.members["fHistogram"] = None

    def GetN(self) -> int:
        return len(self._xrd)

    def GetColor(self) -> Any:
        return self._xrd.colors

    def GetSize(self) -> Any:
        return self._xrd.sizes

    def GetGraph(self) -> Any:
        return wrap(self._xrd.graph)

    def GetHistogram(self) -> Any:
        return wrap(self._xrd.frame())

    def SetHistogram(self, h: Any) -> None:
        self._xrd.members["fHistogram"] = unwrap(h)

    def GetXaxis(self) -> Any:
        return self.GetHistogram().GetXaxis()

    def GetYaxis(self) -> Any:
        return self.GetHistogram().GetYaxis()

    def GetZaxis(self) -> Any:
        return self.GetHistogram().GetZaxis()

    def GetMargin(self) -> float:
        return float(self._xrd.members["fMargin"])

    def SetMargin(self, margin: float) -> None:
        """``SetMargin``: how much wider than the points the frame is, and the frame made again."""
        self._xrd.members.update(fMargin=float(margin), fHistogram=None)

    def GetMaxMarkerSize(self) -> float:
        return float(self._xrd.members["fMaxMarkerSize"])

    def GetMinMarkerSize(self) -> float:
        return float(self._xrd.members["fMinMarkerSize"])

    def SetMaxMarkerSize(self, size: float) -> None:
        self._xrd.members["fMaxMarkerSize"] = float(size)

    def SetMinMarkerSize(self, size: float) -> None:
        self._xrd.members["fMinMarkerSize"] = float(size)

    def Print(self, option: str = "") -> None:
        """``Print``: each point's position, colour value and size value."""
        colors, sizes = self.GetColor(), self.GetSize()
        for at, (x, y) in enumerate(zip(self._xrd.x, self._xrd.y, strict=True)):
            extra = "" if colors is None else f", color[{at}]={colors[at]:g}"
            extra += "" if sizes is None else f", size[{at}]={sizes[at]:g}"
            print(f"x[{at}]={x:g}, y[{at}]={y:g}{extra}")

    def Draw(self, option: str = "") -> None:
        """``Draw``: ``A`` for axes, ``SAME`` over what is drawn, and ``LOGC`` and ``LOGS``
        for colours and sizes by the logarithms of their values."""
        upper = str(option).upper()
        self._xrd.members["fLogC"] = "LOGC" in upper
        self._xrd.members["fLogS"] = "LOGS" in upper
        rest = upper.replace("LOGC", "").replace("LOGS", "")
        TNamed.Draw(self, rest)


class TScatter2D(TScatter):
    """``TScatter2D(n, x, y, z[, colors[, sizes]])``: points in space, coloured and sized."""

    CLASS_TITLE = "A scatter plot in space"
    IN_SPACE = True


for _cls in (TScatter, TScatter2D):
    register(_cls.__name__, factory=partial(adopt, _cls))
