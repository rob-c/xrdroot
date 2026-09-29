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
        from ..core import TGraph
        from ..core.collections import TList

        poi_set, nuisance, pdf = (config.GetParametersOfInterest(),
                                  config.GetNuisanceParameters(), config.GetPdf())  # fmt: skip
        for missing, text in ((poi_set is None, "no parameters of interest"),
                              (poi_set is not None and len(poi_set) != 1,
                               "only one parameter of interest is supported currently"),
                              (nuisance is None, "no nuisance parameters"),
                              (pdf is None, "pdf not set")):  # fmt: skip
            if missing:
                log(None, ERROR, "InputArguments", text)
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
            graph = TGraph(POINTS, xs, ys)
            graph.SetName(f"{name}_{poi.GetName()}_profile")
            graph.GetXaxis().SetTitle(poi.GetName())
            graph.GetYaxis().SetTitle(name)
            graph.SetTitle("")
            found.Add(graph)
        return found
