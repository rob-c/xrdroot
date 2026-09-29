"""RooStats' NuisanceParametersSampler: nuisance parameters drawn from their prior, toy by toy.

The draws and weights below - random ones at seed 7, and the expected,
binned ones with those of no weight skipped - are what ROOT 6.40 gave for
the same prior through PyROOT (until ROOT's own sampler crashed refreshing
its draws, where these carry on).
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterator
from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.roofit.messages import service
from xrdroot.roostats.nuisance import NuisanceParametersSampler

EXPECTED = "[#1] INFO:InputArguments -- Using expected nuisance parameters.\n"
RANDOM = "[#1] INFO:InputArguments -- Using randomized nuisance parameters.\n"


@pytest.fixture(autouse=True)
def _messages() -> Iterator[None]:
    """Every test starts from the message streams RooFit starts with."""
    service().reset()
    yield
    service().reset()


def draws(sampler: Any, target: Any, var: Any, n: int) -> list[tuple[float, float]]:
    cell = ctypes.c_double(0)
    found = []
    for _ in range(n):
        sampler.NextPoint(target, cell)
        found.append((var.getVal(), cell.value))
    return found


def test_random_draws_come_ntoys_at_a_time_with_weight_one(capsys: Any) -> None:
    """ROOT 6.40 at seed 7: 1.2826705876890085, 1.111967518273741, 0.9753636899404228."""
    ROOT.RooRandom.randomGenerator().SetSeed(7)
    b = ROOT.RooRealVar("b", "b", 1.0, 0.0, 2.0)
    prior = ROOT.RooGaussian("g", "g", b, ROOT.RooFit.RooConst(1.0), ROOT.RooFit.RooConst(0.2))
    sampler = ROOT.RooStats.NuisanceParametersSampler(prior, ROOT.RooArgSet(b), 3, False)
    found = draws(sampler, ROOT.RooArgSet(b), b, 4)
    assert found[:3] == [(1.2826705876890085, 1.0), (1.111967518273741, 1.0),
                         (0.9753636899404228, 1.0)]  # fmt: skip
    assert found[3][1] == 1.0
    assert capsys.readouterr().out == RANDOM * 2


def test_expected_draws_are_the_bins_with_weights_and_skip_those_of_none(capsys: Any) -> None:
    """ROOT 6.40: five bins of ``1 + b`` below 1.2 and none above; the empty two skipped."""
    b = ROOT.RooRealVar("b", "b", 1.0, 0.0, 2.0)
    prior = ROOT.RooGenericPdf("prior", "prior", "b > 1.2 ? 0 : 1 + b", ROOT.RooArgList(b))
    sampler = NuisanceParametersSampler(prior, [b], 5, True)
    found = draws(sampler, [b], b, 5)
    assert found[:3] == [
        (0.2, pytest.approx(0.25000862273903807, rel=1e-9)),
        (pytest.approx(0.6), pytest.approx(0.3333448303187175, rel=1e-9)),
        (1.0, pytest.approx(0.4166810378983968, rel=1e-9)),
    ]
    assert found[3:] == found[:2]
    assert capsys.readouterr().out == EXPECTED * 2
    assert sampler.next_point([b]) == pytest.approx(0.4166810378983968, rel=1e-9)


def test_expected_draws_of_two_parameters_adjust_the_number_of_toys(capsys: Any) -> None:
    """Three bins each of two parameters are nine points: the number of toys follows."""
    a = ROOT.RooRealVar("a", "a", 0.5, 0.0, 1.0)
    b = ROOT.RooRealVar("b", "b", 0.5, 0.0, 1.0)
    prior = ROOT.RooGenericPdf("prior", "prior", "1 + a + b", ROOT.RooArgList(a, b))
    sampler = NuisanceParametersSampler(prior, [a, b], 3, True)
    weight = sampler.NextPoint([a, b])
    assert weight > 0.0 and sampler._ntoys == 9
    assert capsys.readouterr().out == EXPECTED + (
        "[#1] INFO:InputArguments -- Adjusted number of toys to number of bins of nuisance "
        "parameters: 9\n"
    )


def test_without_a_prior_a_refresh_draws_nothing() -> None:
    """No prior, or no parameters: ``Refresh`` leaves the sampler as it was."""
    for sampler in (NuisanceParametersSampler(), NuisanceParametersSampler(object())):
        sampler.Refresh()
        assert sampler._points is None
