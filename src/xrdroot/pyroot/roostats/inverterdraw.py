"""``HypoTestInverterPlot::Draw``: the observed and expected curves, the size, and a legend.

``OBS`` or ``EXP`` draws only that; ``SAME`` draws on what the pad has;
``CLB`` adds ``CLb``, blue, and ``2CL`` the other of ``CLs`` and ``CLs+b``,
dotted. The observed curve goes on top, then a legend in the upper right.
"""

from __future__ import annotations

from typing import Any

from .inverterplot import BLUE, RED

__all__ = ["draw"]


def _observed(plot: Any, axis: bool) -> tuple[Any, Any]:
    graph = plot.MakePlot()
    if axis:
        graph.Draw("APL")
        graph.GetHistogram().SetTitle(plot.GetTitle())
        return graph, graph
    graph.Draw("PL")
    return graph, None


def _expected(plot: Any, axis: bool, frame: Any) -> tuple[Any, Any]:
    graphs = plot.MakeExpectedPlot()
    if axis and frame is None:
        graphs.Draw("A")
        graphs.GetHistogram().SetTitle(plot.GetTitle())  # a multigraph's frame, always here
        return graphs, graphs.GetListOfGraphs().First()
    graphs.Draw()
    return graphs, frame


def _size_line(plot: Any, frame: Any) -> Any:
    """The red line at the test size across the frame, and the axes' titles."""
    from ..graphics.shapes import TLine

    alpha = 1.0 - plot._results.ConfidenceLevel()
    line = TLine(frame.GetXaxis().GetXmin(), alpha, frame.GetXaxis().GetXmax(), alpha)
    line.SetLineColor(RED)
    line.Draw()
    param = next(iter(plot._results._parameters), None)
    if param is not None:
        frame.GetXaxis().SetTitle(param.GetName())
    frame.GetYaxis().SetTitle("p value")
    return line


def _extra(plot: Any, option: str, observed: Any) -> list[Any]:
    """``CLb`` and the second CL, as asked."""
    drawn: list[Any] = []
    if "CLB" in option:
        clb = plot.MakePlot("CLb")
        clb.SetMarkerColor(BLUE + 4)
        clb.Draw("PL")
        if observed is not None:
            observed.SetMarkerColor(RED)
        drawn.append(clb)
    if "2CL" in option:
        other = plot.MakePlot("CLs+b" if plot._results._use_cls else "CLs")
        other.SetMarkerColor(BLUE)
        other.Draw("PL")
        other.SetLineStyle(3)
        drawn.insert(0, other)
    return drawn


def draw(plot: Any, option: str) -> list[Any]:
    from ..graphics.legend import TLegend
    from ..graphics.pads import current

    axis = "SAME" not in option
    observed = frame = expected = None
    if "OBS" in option or "EXP" not in option:
        observed, frame = _observed(plot, axis)
    if "EXP" in option or "OBS" not in option:
        expected, frame = _expected(plot, axis, frame)
    kept = [one for one in (observed, expected) if one is not None]
    if frame is not None:
        kept.append(_size_line(plot, frame))
    extra = _extra(plot, option, observed)
    if observed is not None:
        observed.Draw("PL")
    tall = expected is not None or "2CL" in option or "CLB" in option
    legend = TLegend(0.6, 0.6, 0.9, 0.6 + (0.3 if tall else 0.15))
    for one in ([observed] if observed is not None else []) + extra:
        legend.AddEntry(one, "", "PEL")
    if expected is not None:
        graphs = list(expected.GetListOfGraphs())
        for i in range(len(graphs) - 1, -1, -1):
            legend.AddEntry(graphs[i], "", "L" if i == len(graphs) - 1 else "F")
    legend.Draw()
    pad = current()
    if pad is not None:
        pad.RedrawAxis()
    return [*kept, *extra, legend]
