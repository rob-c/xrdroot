"""``ProfileInspector`` and the ``ROOT.RooStats`` namespace it lives in.

The inspector profiles the counting model's likelihood at a hundred points
of the signal strength, keeping the constrained background's value at each;
a configuration without what it needs is refused as ROOT refuses it.
"""

from __future__ import annotations

from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from roostatsmodels import counting


def test_each_free_nuisance_parameter_is_a_graph_along_the_poi() -> None:
    w, data, sb, _ = counting()
    w.var("nom").setConstant(True)
    sb.SetNuisanceParameters("nu,b")  # b is constant: no graph of it
    plots = ROOT.RooStats.ProfileInspector().GetListOfProfilePlots(data, sb)
    assert [g.GetName() for g in plots] == ["nu_mu_profile"]
    graph = plots.At(0)
    assert (graph.GetN(), graph.GetX()[0], graph.GetX()[99]) == (100, 0.0, 10.0)
    assert (graph.GetXaxis().GetTitle(), graph.GetYaxis().GetTitle()) == ("mu", "nu")
    assert graph.GetY()[99] < graph.GetY()[0]  # more signal, less background


class _Config:
    def __init__(self, poi: Any, nuisance: Any, pdf: Any) -> None:
        self.poi, self.nuisance, self.pdf = poi, nuisance, pdf

    def GetParametersOfInterest(self) -> Any:
        return self.poi

    def GetNuisanceParameters(self) -> Any:
        return self.nuisance

    def GetPdf(self) -> Any:
        return self.pdf


@pytest.mark.parametrize(("poi", "nuisance", "pdf", "said"), [
    (None, [], "p", "no parameters of interest"),
    ([1, 2], [], "p", "only one parameter of interest is supported currently"),
    ([1], None, "p", "no nuisance parameters"),
    ([1], [], None, "pdf not set"),
])  # fmt: skip
def test_a_configuration_short_of_something_is_refused(poi: Any, nuisance: Any, pdf: Any,
                                                       said: str, capsys: Any) -> None:
    inspector = ROOT.RooStats.ProfileInspector()
    assert inspector.GetListOfProfilePlots(None, _Config(poi, nuisance, pdf)) is None
    assert said in capsys.readouterr().out


def test_the_namespace_refuses_a_name_it_has_not_by_that_name() -> None:
    refused = r"ROOT has RooStats::Nothing; xrdroot\.pyroot does not"
    with pytest.raises(AttributeError, match=refused):
        ROOT.RooStats.Nothing  # noqa: B018
    with pytest.raises(AttributeError):
        ROOT.RooStats.__wrapped__  # noqa: B018
    assert repr(ROOT.RooStats.HistFactory) == "<namespace RooStats::HistFactory>"
