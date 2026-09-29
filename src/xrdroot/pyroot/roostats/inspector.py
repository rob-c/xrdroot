"""``ProfileInspector``: each nuisance parameter's profiled value along the parameter of interest.

At a hundred points across the parameter of interest's range the profile
likelihood is minimised, and each free nuisance parameter's value there is
kept: a ``TGraph`` ``<nuisance>_<poi>_profile`` of each, in a ``TList``.
"""

from __future__ import annotations

from typing import Any

from ...roofit.messages import ERROR, log

__all__ = ["ProfileInspector"]

#: The points along the parameter of interest.
POINTS = 100


class ProfileInspector:
    """The nuisance parameters' profiles along the parameter of interest."""

    def GetListOfProfilePlots(self, data: Any, config: Any) -> Any:
        from ..core.collections import TList
        from ..core.graphs import TGraph

        poi_set, nuisance, pdf = (config.GetParametersOfInterest(),
                                  config.GetNuisanceParameters(), config.GetPdf())  # fmt: skip
        refusal = _refusal(poi_set, nuisance, pdf)
        if refusal:
            log(None, ERROR, "InputArguments", refusal)
            return None
        poi = next(iter(poi_set))
        profile = pdf.createNLL(data).createProfile(poi)
        low, high = poi.getMin(), poi.getMax()
        step = (high - low) / (POINTS - 1)
        xs = [low + step * i for i in range(POINTS)]
        values: dict[str, list[float]] = {}
        for x in xs:
            poi.setVal(x)
            profile.getVal()
            for par in nuisance:
                if not par.isConstant():
                    values.setdefault(par.GetName(), []).append(par.getVal())
        found = TList()
        for name, ys in values.items():  # added as the last point is, in the set's order
            graph: Any = TGraph(POINTS, xs, ys)
            graph.SetName(f"{name}_{poi.GetName()}_profile")
            graph.GetXaxis().SetTitle(poi.GetName())
            graph.GetYaxis().SetTitle(name)
            graph.SetTitle("")
            found.Add(graph)
        return found


def _refusal(poi_set: Any, nuisance: Any, pdf: Any) -> str:
    """What the configuration lacks for the plots, as ROOT says it - nothing if it is whole."""
    if poi_set is None:
        return "no parameters of interest"
    if len(poi_set) != 1:
        return "only one parameter of interest is supported currently"
    if nuisance is None:
        return "no nuisance parameters"
    return "pdf not set" if pdf is None else ""
