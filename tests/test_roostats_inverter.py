"""HypoTestInverter on the counting model: a fixed scan and an automatic one, against ROOT.

The asymptotic calculator's CLs at each point, the upper limit where the
straight lines between the points cross five percent, the expected limits
from the asymptotic bands, and the automatic search's limit - ROOT 6.40's
numbers, to the last bit.
"""

from __future__ import annotations

from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from roostatsmodels import counting
from xrdroot.roofit.messages import service


@pytest.fixture(autouse=True)
def _quiet() -> Any:
    service().reset()
    ROOT.RooStats.AsymptoticCalculator.SetPrintLevel(-1)
    yield
    service().reset()


def _calculator() -> Any:
    _, data, sb, b = counting()
    calc = ROOT.RooStats.AsymptoticCalculator(data, b, sb)
    calc.SetOneSided(True)
    return calc


@pytest.fixture(scope="module")
def scan() -> Any:
    ROOT.RooStats.AsymptoticCalculator.SetPrintLevel(-1)
    inverter = ROOT.RooStats.HypoTestInverter(_calculator())
    inverter.UseCLs(True)
    inverter.SetFixedScan(5, 0, 4)
    return inverter.GetInterval()


def test_a_fixed_scan_has_roots_cls_at_each_point(scan: Any) -> None:
    ys = [scan.GetYValue(i) for i in range(scan.ArraySize())]
    assert [scan.GetXValue(i) for i in range(5)] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert ys == [1.0, 0.5862507359129521, 0.214844268311849, 0.05447033376117904,
                  0.010564800823040144]  # fmt: skip
    assert scan.CLb(1) == 0.8528773942114782


def test_the_upper_limit_is_where_the_lines_cross_the_size(scan: Any) -> None:
    assert scan.UpperLimit() == 3.1018172346381125
    assert scan.UpperLimitEstimatedError() == 0.0
    assert scan.LowerLimit() == 0.0
    expected = [scan.GetExpectedUpperLimit(s) for s in (-2, -1, 0, 1, 2)]
    assert expected == [1.010049463222704, 1.6682609061353406, 2.134872073299358,
                        3.209090776143463, 10.0]  # fmt: skip


def test_an_automatic_scan_finds_the_limit_itself(capsys: Any) -> None:
    inverter = ROOT.RooStats.HypoTestInverter(_calculator())
    inverter.UseCLs(True)
    result = inverter.GetInterval()
    assert (result.ArraySize(), result.UpperLimit()) == (19, 3.0560402662897936)
    assert result.UpperLimitEstimatedError() == pytest.approx(4.956035581926699e-13, rel=1e-3)
    assert "\tLimit: mu < 3.05604 +/- 4.956" in capsys.readouterr().out


def test_the_plot_has_the_observed_curve_and_the_bands(scan: Any) -> None:
    plot = ROOT.RooStats.HypoTestInverterPlot("p", "t", scan)
    graph = plot.MakePlot()
    assert (graph.GetName(), graph.GetTitle(), graph.GetN()) == ("CLs_observed", "Observed CLs", 5)
    bands = plot.MakeExpectedPlot()
    assert [one.GetTitle() for one in bands.GetListOfGraphs()] == [
        "Expected CLs #pm 2 #sigma", "Expected CLs #pm 1 #sigma", "Expected CLs - Median"]
    assert bands.GetListOfGraphs().At(0).GetErrorYlow(1) == 0.24375849080917983
