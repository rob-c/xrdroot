"""Public ROOT graphics made from the minimizer's contour points."""

from __future__ import annotations

from typing import Any

from ...roofit.fitting.contour import BLUE


def make_graph(minimizer: Any, points: list[Any], n: float, place: int) -> Any:
    from ..core.graphs import TGraph

    xs = [float(p[0]) for p in points] + [float(points[0][0])]
    ys = [float(p[1]) for p in points] + [float(points[0][1])]
    graph: Any = TGraph(len(xs), xs, ys)
    graph.SetName(f"contour_{minimizer.function.GetName()}_n{n:f}")
    graph.SetLineStyle(place + 1)
    graph.SetLineWidth(2)
    graph.SetLineColor(BLUE)
    return graph
