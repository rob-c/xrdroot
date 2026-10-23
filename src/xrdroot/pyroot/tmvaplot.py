"""ROOT graph and canvas construction for the TMVA engine's ROC arrays.

Numerical ROC rates and sample selection belong to TMVA. These internal
adapters retain its public TGraph return types, titles, axes, line colors and
canvas/legend behavior without exposing new ROOT namespace members.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__: list[str] = []


def result_graph(title: Any, sensitivity: Any, specificity: Any) -> Any:
    """A ClassificationResult names and titles its public graph after the method."""
    from .core.graphs import TGraph

    graph = TGraph(len(sensitivity), np.asarray(sensitivity), np.asarray(specificity))
    graph.SetName(title)
    graph.SetTitle(title)
    return graph


def factory_graph(title: Any, sensitivity: Any, specificity: Any, titled: Any) -> Any:
    """Factory graphs optionally title their axes, retaining ROOT's default when suppressed."""
    from .core.graphs import TGraph

    graph = TGraph(len(sensitivity), np.asarray(sensitivity), np.asarray(specificity))
    if titled:
        graph.SetTitle("Signal efficiency vs. Background rejection")
        graph.GetXaxis().SetTitle("Signal efficiency")
        graph.GetYaxis().SetTitle("Background rejection")
    graph.SetName(str(title))
    return graph


def roc_canvas(factory: Any, dataset: Any) -> Any:
    """``GetROCCurve(dataloader)``: a canvas of every method's ROC curve, with a legend."""
    from .graphics.canvas import TCanvas
    from .graphics.legend import TLegend

    name = dataset.GetName() if hasattr(dataset, "GetName") else str(dataset)
    canvas = TCanvas(name, "ROC Curve", 800, 600)
    legend = TLegend(0.15, 0.15, 0.35, 0.3, "MVA Method")
    colour = 1
    for position, item in enumerate(factory.booked.get(name, [])):
        graph = factory.GetROCCurve(name, item.method.name)
        graph.SetLineWidth(2)
        graph.SetLineColor(colour)
        colour += 1
        if colour == 5 or colour == 10 or colour == 11:
            colour += 1
        graph.Draw("AL" if position == 0 else "L")
        legend.AddEntry(graph, item.method.name, "l")
    legend.Draw()
    canvas._tmva_legend = legend  # type: ignore[attr-defined]
    return canvas
