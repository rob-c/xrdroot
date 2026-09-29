"""RooStats' SamplingDistribution: sampled values with weights, their integrals and quantiles.

Every integral (with its error), CDF value, quantile and variation below is
what ROOT 6.40 gave for the same ten values and weights through PyROOT.
"""

from __future__ import annotations

import ctypes
import math
from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.roostats.sampling import SamplingDistribution

VALUES = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0, 5.5, 3.5]
WEIGHTS = [1.0, 2.0, 1.0, 0.5, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0]
INF = math.inf


def weighted() -> SamplingDistribution:
    return ROOT.RooStats.SamplingDistribution("sd", "a title", VALUES, WEIGHTS, "ts")


def plain() -> SamplingDistribution:
    return ROOT.RooStats.SamplingDistribution("p", "plain", VALUES, "q")


def test_a_distribution_keeps_its_values_weights_and_names() -> None:
    """Its name, title, variable, size, values and weights as it was given them."""
    sd = weighted()
    assert (sd.GetName(), sd.GetTitle(), sd.GetVarName(), sd.GetSize()) == (
        "sd", "a title", "ts", 10)  # fmt: skip
    assert plain().GetVarName() == "q"
    assert plain().GetSampleWeights() == [1.0] * 10
    assert sd.GetSamplingDistribution() == VALUES
    assert sd.GetSampleWeights() == WEIGHTS
    assert sd.ClassName() == "RooStats::SamplingDistribution"


#: ROOT 6.40: (low, high, normalize, lowClosed, highClosed) -> integral and error.
INTEGRALS = [
    ((2.0, 5.0, True, True, False), 0.5185185185185185, 0.1767791900530811),
    ((2.0, 5.0, True, False, True), 0.37037037037037035, 0.1657028133941454),
    ((2.0, 5.0, False, True, True), 8.0, 4.0),
    ((-10, 0.5, True, True, False), 0.0, 0.0),
    ((0.0, 100.0, False, True, False), 13.5, 4.8218253804964775),
    ((3.0, 3.0, True, True, True), 0.07407407407407407, 0.07330803410637798),
]


@pytest.mark.parametrize(("ends", "integral", "error"), INTEGRALS)
def test_an_integral_counts_the_weights_between_its_ends(
    ends: Any, integral: float, error: float
) -> None:
    """``IntegralAndError`` into a ``ctypes`` cell, and ``Integral``, as ROOT 6.40's."""
    sd = weighted()
    cell = ctypes.c_double(0)
    assert sd.IntegralAndError(cell, *ends) == pytest.approx(integral, rel=1e-14)
    assert cell.value == pytest.approx(error, rel=1e-14)
    assert sd.Integral(*ends) == pytest.approx(integral, rel=1e-14)
    assert sd.IntegralAndError(None, *ends) == pytest.approx(integral, rel=1e-14)


def test_the_cdf_includes_its_point() -> None:
    """ROOT 6.40: 0, 13/27 at 3 and at 3.2, and 1 at the last value."""
    sd = weighted()
    assert [sd.CDF(x) for x in (0.0, 3.0, 3.2, 9.0)] == pytest.approx(
        [0.0, 0.48148148148148145, 0.48148148148148145, 1.0], rel=1e-14
    )


#: ROOT 6.40: p -> InverseCDF(p), InverseCDF(p, 1, var), var, InverseCDFInterpolate(p).
QUANTILES = [
    (0.0, -INF, -INF, -INF, -INF),
    (0.05, -INF, -INF, -INF, -INF),
    (0.2, 2.0, 2.0, 3.0, 2.0),
    (0.3, 3.0, 3.0, 3.5, 3.0),
    (0.5, 5.0, 5.0, 6.0, 4.0),
    (0.6, 5.5, 5.5, 9.0, 5.0),
    (0.8, 9.0, 9.0, INF, 6.0),
    (0.85, 9.0, 9.0, INF, 7.499999999999998),
    (0.95, INF, INF, INF, INF),
    (1.0, INF, INF, INF, INF),
]


@pytest.mark.parametrize(("p", "nominal", "again", "varied", "interpolated"), QUANTILES)
def test_a_quantile_is_the_p_n_th_value_with_roostats_edges(
    p: float, nominal: float, again: float, varied: float, interpolated: float
) -> None:
    """``InverseCDF`` with and without a variation, and ``InverseCDFInterpolate``."""
    sd = plain()
    cell = ctypes.c_double(0)
    assert sd.InverseCDF(p) == nominal
    assert sd.InverseCDF(p, 1.0, cell) == again
    assert cell.value == varied
    assert sd.InverseCDFInterpolate(p) == pytest.approx(interpolated, rel=1e-14)


@pytest.mark.parametrize(
    ("p", "sigma", "nominal", "varied"),
    [(0.2, -1.0, 2.0, 1.5), (0.3, -2.0, 3.0, -INF), (0.6, 3.0, 5.5, INF), (0.4, 5.0, 3.5, INF),
     (0.6, -3.0, 5.5, -INF)],
)
def test_a_variation_past_either_end_is_infinite(
    p: float, sigma: float, nominal: float, varied: float
) -> None:
    """ROOT 6.40: the varied quantile, or an infinity once it runs off the values."""
    sd = plain()
    assert sd.quantile(p, sigma) == (nominal, varied)
    assert sd.InverseCDF(p, sigma) == nominal


def test_adding_puts_the_others_values_after_these() -> None:
    """ROOT 6.40: the sorted ten, then 7 and 8; a nameless one takes the other's names."""
    sd = plain()
    sd.InverseCDF(0.5)  # sorted now
    other = SamplingDistribution("o", "o", [7.0, 8.0], "")
    sd.Add(other)
    assert (sd.GetSize(), sd.GetVarName()) == (12, "q")
    assert sd.GetSamplingDistribution() == [1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 5.0, 5.5, 6.0, 9.0,
                                            7.0, 8.0]  # fmt: skip
    assert sd.InverseCDF(0.5) == 5.5  # sorted again: the seventh of twelve
    sd.Add(None)
    assert sd.GetSize() == 12
    empty = SamplingDistribution()
    assert (empty.GetName(), empty.GetTitle(), empty.GetSize()) == (
        "SamplingDistribution_DefaultName", "SamplingDistribution", 0)  # fmt: skip
    assert empty.Integral(0, 1) == 0.0
    cell = ctypes.c_double(0)
    assert empty.IntegralAndError(cell, 0, 1) == 0.0 and cell.value == INF
    empty.Add(other)
    assert (empty.GetName(), empty.GetVarName(), empty.GetSize()) == (
        "SamplingDistribution_DefaultName", "", 2)  # fmt: skip
    nameless = SamplingDistribution("", "", [1.0])
    nameless.Add(SamplingDistribution("n", "t", [2.0], "v"))
    assert (nameless.GetName(), nameless.GetTitle(), nameless.GetVarName()) == ("n", "t", "v")


def toys() -> Any:
    """A dataset of three toys of ``x`` (titled ``the x``)."""
    x = ROOT.RooRealVar("x", "the x", 0.0)
    data = ROOT.RooDataSet("ds", "ds", ROOT.RooArgSet(x))
    for value in (1.0, 2.0, 3.0):
        x.setVal(value)
        data.add(ROOT.RooArgSet(x))
    return data


def test_a_distribution_from_toys_takes_a_column_and_its_title() -> None:
    """ROOT 6.40: the first column when ``<name>_TS0`` is not there, titled ``the x``; or the
    column and variable named; nothing from no toys."""
    data = toys()
    found = SamplingDistribution("fd", "fd", data)
    assert found.GetVarName() == "the x"
    assert (found.GetSamplingDistribution(), found.GetSampleWeights()) == (
        [1.0, 2.0, 3.0], [1.0, 1.0, 1.0])  # fmt: skip
    named = SamplingDistribution("fd", "fd", data, "x", "named")
    assert (named.GetVarName(), named.GetSamplingDistribution()) == ("named", [1.0, 2.0, 3.0])
    untitled = SamplingDistribution("fd", "fd", data, "x", "")
    assert untitled.GetVarName() == "the x"
    empty = ROOT.RooDataSet("ed", "ed", ROOT.RooArgSet(ROOT.RooRealVar("x", "x", 0.0)))
    none = SamplingDistribution("fe", "fe", empty, "", "v")
    assert (none.GetVarName(), none.GetSize()) == ("v", 0)
    assert SamplingDistribution("fe", "fe", empty).GetVarName() == ""
