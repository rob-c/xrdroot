"""``RooMinimizer::contour``: Minuit's n-sigma contours of two parameters, on a frame of them.

Each contour is MnContours run at ``n²`` times the error level - the
likelihood risen by ``n²/2`` - closed into a ``TGraph`` of line style ``n``'s
place among those asked for, width 2, in blue, with a marker at the
minimum; the error level and every parameter are put back afterwards, as
``RooMinimizer`` puts them back from its snapshot.
"""

from __future__ import annotations

from typing import Any

from ...fit.minuit import contour as minuit_contour
from ...plugins import require
from ..messages import ERROR, WARNING, log
from ..plot.frame import RooPlot

__all__ = ["contour", "two_variable_frame"]

#: The colour of every contour: ``kBlue``.
BLUE = 600


def two_variable_frame(var1: Any, var2: Any) -> RooPlot:
    """``RooPlot(var1, var2)``: an empty frame over both ranges, 100 bins of ``var1`` along
    and ``var2``'s range up, its axes the variables' titles."""
    frame = RooPlot(var1, var1.getMin(), var1.getMax(), 100)
    frame.SetTitle("A RooPlot")
    frame.hist.members["TH1"]["fYaxis"]["TNamed"]["fTitle"] = var2.getTitle(True)
    frame.SetMinimum(var2.getMin())
    frame.SetMaximum(var2.getMax())
    return frame


def _index(minimizer: Any, var: Any) -> int:
    names = [p.GetName() for p in minimizer.params]
    if var.GetName() in names:
        return names.index(var.GetName())
    text = f"{var.GetName()} is not a floating parameter of {minimizer.function.GetName()}"
    log(minimizer, ERROR, "Minimization", f"RooMinimizer::contour({minimizer.GetName()}) ERROR: "
        f"{text}")  # fmt: skip
    return -1


def _graph(minimizer: Any, points: list[Any], n: float, place: int) -> Any:
    return require("roofit.contour_graph")(minimizer, points, n, place)


def _add_contour(minimizer: Any, frame: Any, indices: tuple[int, int], level: tuple[float, int],
                 npoints: int) -> None:  # fmt: skip
    """The ``n``-sigma contour, the ``place``-th asked for, onto the frame - or why not."""
    n, place = level
    points = minuit_contour(minimizer.minuit, *indices, int(npoints))
    if points is None:
        log(minimizer, ERROR, "Minimization", f"RooMinimizer::contour({minimizer.GetName()}) "
            f"ERROR: MINUIT did not return a contour graph for n={n:g}")  # fmt: skip
        return
    frame.addObject(_graph(minimizer, points, n, place), "L")


def contour(minimizer: Any, var1: Any, var2: Any, levels: list[float], npoints: int) -> Any:
    """The frame of ``var1`` and ``var2`` with the minimum marked and a contour at each
    positive ``n`` of ``levels`` - ``None`` if either is not a free parameter."""
    TMarker = require("root.class", "TMarker")

    ix, iy = _index(minimizer, var1), _index(minimizer, var2)
    if ix < 0 or iy < 0:
        return None
    frame = two_variable_frame(var1, var2)
    frame.addObject(TMarker(var1.getVal(), var2.getVal(), 8))
    if minimizer.minuit is None:
        log(minimizer, WARNING, "Minimization", "RooMinimizer::contour: Error, run Migrad before "
            "contours!")  # fmt: skip
        return frame
    saved = [p.clone(p.GetName()) for p in minimizer.params]  # the snapshot ROOT takes
    errordef = minimizer.minuit.errordef
    for place, n in enumerate(levels):
        if n > 0:
            minimizer.minuit.errordef = n * n * errordef
            _add_contour(minimizer, frame, (ix, iy), (n, place), npoints)
    minimizer.minuit.errordef = errordef
    for par, copy in zip(minimizer.params, saved, strict=True):
        par.copy_value_from(copy)
    return frame
