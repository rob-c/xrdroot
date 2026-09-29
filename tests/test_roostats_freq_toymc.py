"""``ToyMCSampler``: toys of a model at a point and the test statistic of each, as ROOT 6.40's.

An extended Gaussian of ten expected events was sampled in ROOT through
PyROOT after ``RooRandom::randomGenerator()->SetSeed(4357)``: the number of
events of each toy, the events of a toy of a fixed size and of one made on
prototype data, and the profile likelihood ratio of toys generated until
enough fall in a tail are ROOT's. The sampler's refusals are said as
ROOT says them.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.roostats.moretests import NumEventsTestStat
from xrdroot.roostats.teststats import ProfileLikelihoodTestStat
from xrdroot.roostats.toymc import ToyMCSampler


@pytest.fixture(autouse=True)
def _fresh(tmp_path: Any) -> Iterator[None]:
    yield from fresh(tmp_path)


def _extended() -> tuple[Any, ...]:
    """``Gauss(y; m, s)`` extended to ``n`` = 10 events, seeded with 4357."""
    ROOT.RooRandom.randomGenerator().SetSeed(4357)
    y = ROOT.RooRealVar("y", "", -5, 5)
    m = ROOT.RooRealVar("m", "", 0.5, -1, 2)
    s = ROOT.RooRealVar("s", "", 1, 0.5, 2)
    n = ROOT.RooRealVar("n", "", 10, 0, 100)
    gs = ROOT.RooGaussian("gs", "", y, m, s)
    ext = ROOT.RooExtendPdf("ext", "", gs, n)
    return y, m, ext, ROOT.RooArgSet(y), ROOT.RooArgSet(m)


def _sampler(statistic: Any, toys: int, ext: Any, obs: Any, poi: Any) -> ToyMCSampler:
    sampler = ToyMCSampler(statistic, toys)
    sampler.SetPdf(ext)
    sampler.SetObservables(obs)
    sampler.SetParametersForTestStat(poi)
    return sampler


def _values(toy: Any) -> list[float]:
    return [toy.get(i).getRealValue("y") for i in range(toy.numEntries())]


def test_toys_of_an_extended_model_and_of_a_fixed_size_are_roots() -> None:
    """Poisson-varied toys' sizes; then four events; then events on three prototype events."""
    y, _m, ext, obs, poi = _extended()
    sampler = _sampler(NumEventsTestStat(ext), 5, ext, obs, poi)
    found = sampler.GetSamplingDistribution(poi)
    assert found.GetSamplingDistribution() == [19.0, 9.0, 8.0, 17.0, 9.0]
    assert (found.GetVarName(), found.GetName()) == ("Number of events", "Number of events")
    sampler.SetNEventsPerToy(4)
    assert sampler.nEventsPerToy() == 4
    assert _values(sampler.GenerateToyData(poi)) == pytest.approx(
        [0.7651375956857009, 0.42125454222514236, 0.6534603604040967, -0.07162723874898802],
        rel=1e-12)  # fmt: skip
    proto = ROOT.RooDataSet("proto", "", obs)
    for value in (0.1, 0.2, 0.3):
        y.setVal(value)
        proto.add(obs)
    sampler.SetProtoData(proto)
    assert _values(sampler.GenerateToyData(poi)) == pytest.approx(
        [2.3533644938896714, 0.16191567299188137, -0.006902372630861464], rel=1e-12)


def test_toys_go_on_until_enough_fall_in_the_tail() -> None:
    """Three toys at most 0.05 take ten toys; one of at least 0.5 takes four - ROOT's toys."""
    _y, _m, ext, obs, poi = _extended()
    sampler = _sampler(ProfileLikelihoodTestStat(ext), 4, ext, obs, poi)
    sampler.SetToysLeftTail(3, 0.05)
    left = sampler.GetSamplingDistribution(poi).GetSamplingDistribution()
    assert left == pytest.approx([
        0.023396419477576202, 0.78819195754734, 0.2744580921921429, 0.022143981455284845,
        0.22290790245994474, 1.2971992667221308, 0.8785389295066164, 2.8087543397888215,
        0.11126849498316549, 0.002168664568971135], rel=1e-9)  # fmt: skip
    sampler.SetToysRightTail(1, 0.5)
    right = sampler.GetSamplingDistribution(poi).GetSamplingDistribution()
    assert right == pytest.approx([0.0019879633362276383, 2.7138840660836583,
                                   0.6422362425817846, 0.0675264216637963], rel=1e-9)
    sampler.SetToysBothTails(0, 0, 0)
    sampler.SetMaxToys(2)
    assert sampler.GetSamplingDistribution(poi).GetSize() == 2


def test_every_five_hundred_toys_are_said(capfd: Any) -> None:
    """ROOT's progress line at the 500th toy, with the tail's count - each toy the same data
    here, for speed."""
    _y, _m, ext, obs, poi = _extended()
    sampler = _sampler(NumEventsTestStat(ext), 501, ext, obs, poi)
    sampler.SetToysRightTail(1, 1)
    toy = ext.generate(obs, 3)
    sampler.GenerateToyData = lambda *args, **kwargs: (toy, 1.0)  # type: ignore[method-assign]
    capfd.readouterr()
    found = sampler.GetSamplingDistribution(poi)
    assert capfd.readouterr().out == (
        "[#0] PROGRESS:Generation -- generated toys: 500 / 501 (tails: 500 / 1)\n")
    assert (found.GetSize(), sum(found.GetSamplingDistribution())) == (501, 1503.0)
