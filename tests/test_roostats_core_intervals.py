"""RooStats' intervals: SimpleInterval, and LikelihoodInterval where it cannot find its ends.

A SimpleInterval's membership and messages, and a LikelihoodInterval over a
flat likelihood - whose minimum Minuit2 calls invalid, so MINOS gives no
errors and the ends are the parameter's range - are what ROOT 6.40 gave for
the same calls through PyROOT.
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterator
from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.roofit.messages import service
from xrdroot.roostats.intervals import ConfInterval, Named, SimpleInterval

VALUES = [0.3, 1.2, -0.4, 2.1, 0.8, 1.5, 0.1, 1.9, 0.6, 1.1]


@pytest.fixture(autouse=True)
def _messages() -> Iterator[None]:
    """Every test starts from the message streams RooFit starts with."""
    service().reset()
    yield
    service().reset()


def test_a_simple_interval_holds_a_point_between_its_ends(capsys: Any) -> None:
    """ROOT 6.40: ``[-1, 2]`` at 90% holds 1, not 3; a point of other parameters is refused
    with the reason."""
    mu = ROOT.RooRealVar("mu", "mu", 1.0, -5, 5)
    nu = ROOT.RooRealVar("nu", "nu", 1.0, -5, 5)
    si = ROOT.RooStats.SimpleInterval("si", mu, -1.0, 2.0, 0.9)
    assert (si.GetName(), si.GetTitle(), si.ClassName()) == ("si", "si", "RooStats::SimpleInterval")
    assert (si.ConfidenceLevel(), si.LowerLimit(), si.UpperLimit()) == (0.9, -1.0, 2.0)
    kinds = ("RooStats::ConfInterval", "TNamed", "RooAbsArg")
    assert [si.InheritsFrom(kind) for kind in kinds] == [True, True, False]
    assert si.IsInInterval(ROOT.RooArgSet(mu))
    mu.setVal(3.0)
    points = (ROOT.RooArgSet(mu), ROOT.RooArgSet(nu), ROOT.RooArgSet(mu, nu))
    assert [si.IsInInterval(point) for point in points] == [False, False, False]
    assert capsys.readouterr().out == (
        "[#0] ERROR:InputArguments -- size is ok, but parameters don't match\n"
        "[#0] ERROR:InputArguments -- size is wrong, parameters don't match\n"
    )
    assert [p.GetName() for p in si.GetParameters()] == ["mu"]


def test_an_empty_interval_has_no_parameters_and_no_level() -> None:
    """ROOT 6.40: ``SimpleInterval()`` is nameless, at level 0, of no parameters - and holds
    no point, not even one of none."""
    empty = SimpleInterval()
    assert (empty.GetName(), empty.ConfidenceLevel(), len(empty.GetParameters())) == ("", 0.0, 0)
    assert not empty.IsInInterval(ROOT.RooArgSet())
    base = ConfInterval("base")
    assert (base.ConfidenceLevel(), base.CheckParameters([])) == (0.95, True)
    named = Named("n", "a title")
    named.SetName("m")
    named.SetTitle("t")
    assert (named.GetName(), named.GetTitle(), named.ClassName()) == ("m", "t", "RooStats::Named")
    assert Named(None).GetName() == "" and named.InheritsFrom("TObject")


def flat() -> tuple[Any, Any]:
    """``1 + 0 a x``: a likelihood that does not depend on ``a`` at all."""
    w = ROOT.RooWorkspace("w")
    w.factory("EXPR::flat('1 + 0*a*x', x[-10,10], a[1,0,3])")
    x = w.var("x")
    data = ROOT.RooDataSet("d", "d", ROOT.RooArgSet(x))
    for value in VALUES:
        x.setVal(value)
        data.add(ROOT.RooArgSet(x))
    return w, data


@pytest.mark.filterwarnings("ignore::UserWarning")  # the fit's zero error, which iminuit replaces
def test_a_flat_likelihood_has_the_range_for_its_ends(capsys: Any) -> None:
    """ROOT 6.40: the minimum is not valid, so ``FindLimits`` fails and leaves its cells; then
    MINOS gives nothing and the ends are ``[0, 3]``; there is no contour."""
    w, data = flat()
    a = w.var("a")
    plc = ROOT.RooStats.ProfileLikelihoodCalculator(data, w.pdf("flat"), [a], 0.3173)
    interval = plc.GetInterval()
    assert "Global fit failed - status = " in capsys.readouterr().out
    lower, upper = ctypes.c_double(7.0), ctypes.c_double(8.0)
    assert not interval.FindLimits(a, lower, upper)
    assert (lower.value, upper.value) == (7.0, 8.0)
    assert (interval.LowerLimit(a), interval.UpperLimit(a)) == (0.0, 3.0)
    assert interval.GetContourPoints(a, a, [0.0] * 3, [0.0] * 3, 3) == 0
    said = capsys.readouterr()
    assert said.out == (
        "Error: Minimization failed  \n"
        "Error returned from minimization of likelihood function - cannot find interval "
        "limits \n"
        "Warning: lower value for a is at limit 0\n"
        "Warning: upper value for a is at limit 0\n"
        "[#1] INFO:Minimization -- LikelihoodInterval - Finding the contour of a ( 0 ) and "
        "a ( 0 ) \n"
        "[#0] ERROR:Minimization -- LikelihoodInterval - Error finding contour for parameters "
        "a and a\n"
    )
    assert said.err == (
        "Error in <Minuit2>: Minuit2Minimizer::GetMinosError Failed - invalid function minimum\n"
        "Error in <Minuit2>: Minuit2Minimizer::Contour Invalid function minimum\n"
    )


def test_an_upper_end_at_the_range_names_the_cell_it_was_given(capsys: Any) -> None:
    """ROOT's warning names ``upper`` as it was when handed in - 8 here - not the range."""
    w, data = flat()
    a = w.var("a")
    interval = ROOT.RooStats.ProfileLikelihoodCalculator(data, w.pdf("flat"), [a]).GetInterval()
    interval.LowerLimit(a)  # the minimizer made, and failed, once
    capsys.readouterr()
    upper = ctypes.c_double(8.0)
    assert interval.FindLimits(a, None, upper)
    assert upper.value == 3.0
    assert capsys.readouterr().out.endswith("Warning: upper value for a is at limit 8\n")
