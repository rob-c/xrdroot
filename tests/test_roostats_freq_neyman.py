"""``FeldmanCousins``, ``NeymanConstruction`` and the ``ConfidenceBelt`` they fill, as ROOT 6.40's.

The constructions below were run in ROOT through PyROOT after
``RooRandom::randomGenerator()->SetSeed(4357)``, with a handful of toys per
point: the lines of progress each point prints, the acceptance regions and
the points kept are ROOT's to the printed digit. The corners - a sampler
that cannot sample, the belt file ROOT writes, points the belt never saw -
are refused or said as ROOT says them.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.roostats.belt import AcceptanceRegion, ConfidenceBelt, PointSetInterval
from xrdroot.roostats.neyman import FeldmanCousins, NeymanConstruction


@pytest.fixture(autouse=True)
def _fresh(tmp_path: Any) -> Iterator[None]:
    """A clean ROOT session, seeded as the reference runs were."""
    yield from fresh(tmp_path)


def _counting(bins: int = 5) -> tuple[Any, ...]:
    """rs401c's number counting: ``x ~ Poisson(mu + 3)``, one event drawn after seed 4357."""
    ROOT.RooRandom.randomGenerator().SetSeed(4357)
    x = ROOT.RooRealVar("x", "", 1, 0, 50)
    mu = ROOT.RooRealVar("mu", "", 2.5, 0, 15)
    b = ROOT.RooConstVar("b", "", 3.0)
    mean = ROOT.RooAddition("mean", "", ROOT.RooArgList(mu, b))
    pois = ROOT.RooPoisson("pois", "", x, mean)
    data = pois.generate(ROOT.RooArgSet(x), 1)
    mc = ROOT.RooStats.ModelConfig("poissonProblem", ROOT.RooWorkspace())
    mc.SetPdf(pois)
    mc.SetParametersOfInterest(ROOT.RooArgSet(mu))
    mc.SetObservables(ROOT.RooArgSet(x))
    fc = FeldmanCousins(data, mc)
    fc.SetNBins(bins)
    fc.FluctuateNumDataEntries(False)
    return x, mu, b, mean, pois, data, mc, fc


#: ROOT's lines for the five points of the counting experiment, 20 toys each.
COUNTING = """\
FeldmanCousins: Model has no nuisance parameters
FeldmanCousins: # points to test = 5
[#1] INFO:Eval -- lookup index = 0
NeymanConstruction: Prog: 1/5 total MC = 20 this test stat = 0.592829
 mu=1.5 [-inf, 1.09453]  in interval = 1

NeymanConstruction: Prog: 2/5 total MC = 20 this test stat = 0.017044
 mu=4.5 [-inf, 2.66724]  in interval = 1

NeymanConstruction: Prog: 3/5 total MC = 20 this test stat = 0.661744
 mu=7.5 [-inf, 1.23941]  in interval = 1

NeymanConstruction: Prog: 4/5 total MC = 20 this test stat = 1.90254
 mu=10.5 [-inf, 0.677129]  in interval = 0

NeymanConstruction: Prog: 5/5 total MC = 20 this test stat = 3.49785
 mu=13.5 [-inf, 1.03987]  in interval = 0

[#1] INFO:Eval -- 3 points in interval
"""


def test_feldman_cousins_on_a_counting_experiment_keeps_roots_points(capfd: Any) -> None:
    """Five points, twenty toys each: ROOT's statistics, regions and three points kept."""
    _x, mu, _b, _mean, _pois, _data, _mc, fc = _counting()
    fc.SetTestSize(0.1)
    fc.AdditionalNToysFactor(0.04)
    fc.CreateConfBelt(True)
    capfd.readouterr()
    interval = fc.GetInterval()
    out = capfd.readouterr().out
    assert out.endswith(COUNTING)
    assert "FeldmanCousins: ntoys per point = 20\n" in out
    assert "nEvents per toy will not fluctuate, will always be 1\n" in out
    assert (interval.LowerLimit(mu), interval.UpperLimit(mu)) == (1.5, 7.5)
    belt = fc.GetConfidenceBelt()
    scan = fc.GetPointsToScan()
    found = []
    for i in range(scan.numEntries()):
        point = scan.get(i)
        found.append((interval.IsInInterval(point), belt.GetAcceptanceRegionMin(point),
                      round(belt.GetAcceptanceRegionMax(point), 6)))  # fmt: skip
    assert found == [(True, -float("inf"), 1.094535), (True, -float("inf"), 2.667242),
                     (True, -float("inf"), 1.239413), (False, -float("inf"), 0.677129),
                     (False, -float("inf"), 1.039872)]  # fmt: skip
    assert (fc.Size(), fc.ConfidenceLevel()) == (0.1, 0.9)
    assert belt.ConfidenceLevels() == []


def test_feldman_cousins_says_its_sampler_before_the_interval(capfd: Any) -> None:
    """``GetTestStatSampler`` makes the sampler once: adaptive and fluctuating are said so."""
    _x, _mu, _b, _mean, _pois, _data, _mc, fc = _counting()
    fc.UseAdaptiveSampling(True)
    fc.FluctuateNumDataEntries(True)
    capfd.readouterr()
    sampler = fc.GetTestStatSampler()
    assert fc.GetTestStatSampler() is sampler
    assert capfd.readouterr().out == (
        "FeldmanCousins: ntoys per point: adaptive\n"
        "FeldmanCousins: nEvents per toy will fluctuate about  expectation\n"
    )
    assert sampler.GetNToys() == 1000
    assert sampler.GetTestStatistic().GetVarName() == "Profile Likelihood Ratio"
    assert fc.GetConfidenceBelt() is None


def _gauss() -> tuple[Any, ...]:
    """Ten events of ``Gauss(y; m, s)``: ``m`` of interest, ``s`` a nuisance parameter."""
    ROOT.RooRandom.randomGenerator().SetSeed(4357)
    y = ROOT.RooRealVar("y", "", -5, 5)
    m = ROOT.RooRealVar("m", "", 0.5, -1, 2)
    s = ROOT.RooRealVar("s", "", 1, 0.5, 2)
    gs = ROOT.RooGaussian("gs", "", y, m, s)
    data = gs.generate(ROOT.RooArgSet(y), 10)
    mc = ROOT.RooStats.ModelConfig("gaussProblem", ROOT.RooWorkspace())
    mc.SetPdf(gs)
    mc.SetParametersOfInterest(ROOT.RooArgSet(m))
    mc.SetNuisanceParameters(ROOT.RooArgSet(s))
    mc.SetObservables(ROOT.RooArgSet(y))
    return y, m, s, gs, data, mc


def _few_toys(fc: Any) -> Any:
    fc.SetTestSize(0.2)
    fc.AdditionalNToysFactor(0.04)
    fc.FluctuateNumDataEntries(False)
    return fc


def test_a_nuisance_parameter_makes_a_profile_construction(capfd: Any) -> None:
    """Each point of ``m`` takes ``s`` at its conditional best fit, as ROOT's profile does."""
    _y, m, _s, _gs, data, mc = _gauss()
    fc = _few_toys(FeldmanCousins(data, mc))
    fc.SetNBins(3)
    fc.CreateConfBelt(True)
    capfd.readouterr()
    interval = fc.GetInterval()
    out = capfd.readouterr().out
    assert "Model has nuisance parameters, will do profile construction\n" in out
    assert "FeldmanCousins: # points to test = 3\n" in out
    assert (interval.LowerLimit(m), interval.UpperLimit(m)) == (0.5, 0.5)
    scan, belt = fc.GetPointsToScan(), fc.GetConfidenceBelt()
    assert scan.ClassName() == "RooDataSet"
    found = []
    for i in range(scan.numEntries()):
        point = scan.get(i)
        found.append((point.getRealValue("m"), point.getRealValue("s"),
                      interval.IsInInterval(point), belt.GetAcceptanceRegionMax(point)))
    wanted = [(-0.5, 1.2236516456764148, False, 1.0637234439204448),
              (0.5, 0.5889344216702703, True, 1.5997818729010413),
              (1.5, 1.1025932983314566, False, 1.428287033693989)]  # fmt: skip
    assert found == [pytest.approx(one, rel=1e-9) for one in wanted]


def test_given_points_of_interest_are_the_profile_constructions_scan(capfd: Any) -> None:
    """``SetPOIPointsToTest``: two points of ``m``, none kept - and the empty limits said."""
    _y, m, _s, _gs, data, mc = _gauss()
    points = ROOT.RooDataSet("pts", "", ROOT.RooArgSet(m))
    for value in (0.25, 1.75):
        m.setVal(value)
        points.add(ROOT.RooArgSet(m))
    fc = _few_toys(FeldmanCousins(data, mc))
    fc.SetPOIPointsToTest(points)
    capfd.readouterr()
    interval = fc.GetInterval()
    assert capfd.readouterr().out.endswith(
        "FeldmanCousins: # points to test = 2\n"
        "NeymanConstruction: Prog: 1/2 total MC = 10 this test stat = 1.31404\n"
        " m=0.25 s=0.669406 [-inf, 1.19135]  in interval = 0\n\n"
        "NeymanConstruction: Prog: 2/2 total MC = 10 this test stat = 8.04265\n"
        " m=1.75 s=1.35612 [-inf, 1.48604]  in interval = 0\n\n"
        "[#1] INFO:Eval -- 0 points in interval\n"
    )
    assert (interval.LowerLimit(m), interval.UpperLimit(m)) == (0.0, 0.0)
    empty = "[#0] ERROR:InputArguments -- RooDataSet::getRange(pointsInInterval) WARNING: empty "
    assert capfd.readouterr().out == (empty + "dataset\n") * 2


def test_without_the_profile_every_parameter_is_scanned(capfd: Any) -> None:
    """``DoProfileConstruction(False)``: a grid of ``m`` and ``s``, two bins each."""
    _y, m, s, _gs, data, mc = _gauss()
    fc = _few_toys(FeldmanCousins(data, mc))
    fc.SetNBins(2)
    fc.DoProfileConstruction(False)
    capfd.readouterr()
    interval = fc.GetInterval()
    out = capfd.readouterr().out
    assert "FeldmanCousins: Model has no nuisance parameters\n" in out
    assert "FeldmanCousins: # points to test = 4\n" in out
    assert " m=1.25 s=1.625 [-inf, 0.786166]  in interval = 0\n" in out
    assert fc.GetPointsToScan().ClassName() == "RooDataHist"
    assert (interval.LowerLimit(m), interval.UpperLimit(s)) == (0.0, 0.0)


def _neyman(capfd: Any, adaptive: bool) -> tuple[Any, Any, Any]:
    """A Neyman construction of the profile likelihood ratio at ``m`` = 0 and 1, both tails."""
    y, m, s, gs, data, mc = _gauss()
    s.setConstant(True)
    toys = ROOT.RooStats.ToyMCSampler(ROOT.RooStats.ProfileLikelihoodTestStat(gs), 10)
    toys.SetPdf(gs)
    toys.SetObservables(ROOT.RooArgSet(y))
    toys.SetNEventsPerToy(10)
    points = ROOT.RooDataSet("pts", "", ROOT.RooArgSet(m))
    for value in (0.0, 1.0):
        m.setVal(value)
        points.add(ROOT.RooArgSet(m))
    nc = NeymanConstruction(data, mc)
    nc.SetTestStatSampler(toys)
    nc.SetLeftSideTailFraction(0.5)
    nc.SetParameterPointsToTest(points)
    if adaptive:
        nc.SetConfidenceLevel(0.5)
        nc.UseAdaptiveSampling(True)
    else:
        nc.SetTestSize(0.3)
        nc.CreateConfBelt(True)
        nc.AdditionalNToysFactor(0.5)
    capfd.readouterr()
    return nc, points, m


def test_a_neyman_construction_with_both_tails_keeps_roots_regions(capfd: Any) -> None:
    """Half the size in each tail, ten toys a point: ROOT's regions, both points kept."""
    nc, points, m = _neyman(capfd, adaptive=False)
    interval = nc.GetInterval()
    assert capfd.readouterr().out == (
        "[#1] INFO:Eval -- lookup index = 0\n"
        "NeymanConstruction: Prog: 1/2 total MC = 10 this test stat = 1.62741\n"
        " m=0 [0.0535383, 1.75172]  in interval = 1\n\n"
        "NeymanConstruction: Prog: 2/2 total MC = 10 this test stat = 0.92207\n"
        " m=1 [0.0459519, 1.84603]  in interval = 1\n\n"
        "[#1] INFO:Eval -- 2 points in interval\n"
    )
    assert (interval.LowerLimit(m), interval.UpperLimit(m)) == (0.0, 1.0)
    belt = nc.GetConfidenceBelt()
    regions = [(belt.GetAcceptanceRegionMin(points.get(i)), belt.GetAcceptanceRegionMax(
        points.get(i))) for i in range(2)]  # fmt: skip
    assert regions == [pytest.approx((0.05353825581583571, 1.7517235751503364), rel=1e-9),
                       pytest.approx((0.04595194347029263, 1.8460268006143572), rel=1e-9)]
    assert (nc.Size(), nc.ConfidenceLevel()) == (0.3, 0.7)
    assert nc.GetTestStatSampler().GetNToys() == 10


def test_adaptive_sampling_doubles_the_toys_while_the_data_is_near_an_edge(capfd: Any) -> None:
    """ROOT's rounds: 48 toys at the first point, 144 at the second - neither kept."""
    nc, _points, _m = _neyman(capfd, adaptive=True)
    interval = nc.GetInterval()
    assert capfd.readouterr().out == (
        "NeymanConstruction: Prog: 1/2 total MC = 48 this test stat = 1.62741\n"
        " m=0 [0.036991, 0.715826]  in interval = 0\n\n"
        "NeymanConstruction: Prog: 2/2 total MC = 144 this test stat = 0.92207\n"
        " m=1 [0.031796, 0.694988]  in interval = 0\n\n"
        "[#1] INFO:Eval -- 0 points in interval\n"
    )
    assert interval.GetParameterPoints().numEntries() == 0


@pytest.mark.parametrize("adaptive", [False, True])
def test_a_sampler_that_cannot_sample_leaves_no_interval(capfd: Any, adaptive: bool) -> None:
    """Without observables the sampler refuses, and the construction says so and gives none."""
    nc, _points, _m = _neyman(capfd, adaptive)
    statistic = nc.GetTestStatSampler().GetTestStatistic()
    toys = ROOT.RooStats.ToyMCSampler(statistic, 10)
    toys.SetPdf(statistic.GetPdf())
    nc.SetTestStatSampler(toys)
    assert nc.GetInterval() is None
    out = capfd.readouterr().out
    assert "Observables not set.\n" in out
    assert out.endswith("[#0] ERROR:Eval -- Neyman Construction: error generating sampling "
                        "distribution\n")  # fmt: skip


def test_the_constructions_setters_and_the_belt_file_they_do_not_write() -> None:
    """The data and model are replaced; ROOT's ``SamplingDistributions.root`` is refused."""
    _y, _m, _s, _gs, data, mc = _gauss()
    for made in (NeymanConstruction(data, mc), FeldmanCousins(data, mc)):
        made.SetData(None)
        made.SetModel(None)
        made.SaveBeltToFile(False)
        with pytest.raises(UnsupportedFeatureError, match=r"SamplingDistributions\.root"):
            made.SaveBeltToFile(True)
        assert (made._data, made._model) == (None, None)


def test_an_extended_model_fluctuates_each_toys_number_of_events(capfd: Any) -> None:
    """A model without observables gets its sampler first, then its observables from the data."""
    ROOT.RooRandom.randomGenerator().SetSeed(4357)
    y = ROOT.RooRealVar("y", "", -5, 5)
    m = ROOT.RooRealVar("m", "", 0.5, -1, 2)
    s = ROOT.RooRealVar("s", "", 1)
    n = ROOT.RooRealVar("n", "", 10)
    gs = ROOT.RooGaussian("gs", "", y, m, s)
    ext = ROOT.RooExtendPdf("ext", "", gs, n)
    data = ext.generate(ROOT.RooArgSet(y))
    mc = ROOT.RooStats.ModelConfig("extProblem", ROOT.RooWorkspace())
    mc.SetPdf(ext)
    mc.SetParametersOfInterest(ROOT.RooArgSet(m))
    fc = FeldmanCousins(data, mc)
    fc.SetConfidenceLevel(0.8)
    fc.AdditionalNToysFactor(0.04)
    fc.SetNBins(2)
    fc.SetParameterPointsToTest(None)  # remade by the construction, as ROOT remakes it
    capfd.readouterr()
    fc.GetTestStatSampler()
    interval = fc.GetInterval()
    assert capfd.readouterr().out == (
        "FeldmanCousins: ntoys per point = 10\n"
        "FeldmanCousins: nEvents per toy will fluctuate about  expectation\n"
        "\n=== Using the following for extProblem ===\n"
        "Observables:             RooArgSet:: = (y)\n"
        "Parameters of Interest:  RooArgSet:: = (m)\n"
        "PDF:                     RooExtendPdf::ext[ pdf=gs n=n ] = 0.882497\n\n"
        "FeldmanCousins: Model has no nuisance parameters\n"
        "FeldmanCousins: # points to test = 2\n"
        "NeymanConstruction: Prog: 1/2 total MC = 10 this test stat = 0.0459115\n"
        " m=-0.25 [-inf, 1.16683]  in interval = 1\n\n"
        "NeymanConstruction: Prog: 2/2 total MC = 10 this test stat = 12.7352\n"
        " m=1.25 [-inf, 1.75012]  in interval = 0\n\n"
        "[#1] INFO:Eval -- 1 points in interval\n"
    )
    assert (interval.LowerLimit(m), interval.UpperLimit(m)) == (-0.25, -0.25)
    assert fc.GetTestStatSampler().nEventsPerToy() == 0


def _hist_of_points() -> tuple[Any, Any, Any]:
    """Four bins of ``a`` in [0, 4], the second and third weighing something."""
    a = ROOT.RooRealVar("a", "", 0, 4)
    a.setBins(4)
    events = ROOT.RooDataSet("events", "", ROOT.RooArgSet(a))
    for value in (1.5, 2.5):
        a.setVal(value)
        events.add(ROOT.RooArgSet(a))
    hist = ROOT.RooDataHist("h", "", ROOT.RooArgSet(a), events)
    return a, hist, ROOT.RooRealVar("other", "", 0)


def test_a_histogram_of_points_is_an_interval_of_its_bins_that_weigh(capfd: Any) -> None:
    """A bin with weight is in; the limits are the parameter's own; other parameters are not."""
    a, hist, other = _hist_of_points()
    interval = PointSetInterval("hs", hist)
    assert interval.GetParameterPoints() is hist
    assert interval.GetParameters().names() == ["a"]
    a.setVal(1.2)
    inside = interval.IsInInterval(ROOT.RooArgSet(a))
    a.setVal(3.5)
    outside = interval.IsInInterval(ROOT.RooArgSet(a))
    assert (inside, outside) == (True, False)
    assert (interval.LowerLimit(a), interval.UpperLimit(a)) == (0.0, 4.0)
    capfd.readouterr()
    assert not interval.IsInInterval(ROOT.RooArgSet(other))
    assert capfd.readouterr().out == (
        "[#0] ERROR:InputArguments -- size is ok, but parameters don't match\n")


def test_a_belt_of_a_histogram_finds_each_points_bin(capfd: Any) -> None:
    """Regions by bin: a point anywhere in the bin finds it; an unfilled bin is refused."""
    a, hist, _other = _hist_of_points()
    belt = ConfidenceBelt("belt", "a belt", hist)
    assert (belt.GetName(), belt.GetTitle(), belt.GetParameters().names()) == (
        "belt", "a belt", ["a"])
    capfd.readouterr()
    a.setVal(1.5)
    belt.AddAcceptanceRegion(ROOT.RooArgSet(a), 0, 0.5, 2.5, 0.9)
    a.setVal(1.9)
    assert belt.GetAcceptanceRegionMin(ROOT.RooArgSet(a), 0.9) == 0.5
    assert belt.GetAcceptanceRegion(ROOT.RooArgSet(a), 0.9).GetLookupIndex() == 0
    assert capfd.readouterr().out == (
        "[#1] INFO:Eval -- using default cl, leftside for now\n"
        "[#1] INFO:Eval -- lookup index = 0\n" + "[#1] INFO:Eval -- using default cl, leftside "
        "for now\n" * 2)  # fmt: skip
    a.setVal(3.5)
    with pytest.raises(RuntimeError, match="Sampling summaries are not filled yet"):
        belt.GetAcceptanceRegionMax(ROOT.RooArgSet(a))


def test_a_belt_refuses_points_of_other_parameters(capfd: Any) -> None:
    """A region for them is said to be a problem; asked for, there is none - its limits NaN."""
    a, _hist, other = _hist_of_points()
    points = ROOT.RooDataSet("pts", "", ROOT.RooArgSet(a))
    points.add(ROOT.RooArgSet(a))
    belt = ConfidenceBelt("belt", points)
    capfd.readouterr()
    belt.AddAcceptanceRegion(ROOT.RooArgSet(other), 0, 1.0, 2.0)
    minimum = belt.GetAcceptanceRegionMin(ROOT.RooArgSet(other))
    maximum = belt.GetAcceptanceRegionMax(ROOT.RooArgSet(other))
    assert minimum != minimum and maximum != maximum
    mismatch = "[#0] ERROR:InputArguments -- size is ok, but parameters don't match\n"
    problem = "[#0] ERROR:InputArguments -- problem with parameters\n"
    assert capfd.readouterr().out == (
        mismatch + problem + "[#1] INFO:Eval -- lookup index = 0\n" + (mismatch + problem) * 2)
    a.setVal(3.0)
    with pytest.raises(RuntimeError, match="CreateConfBelt"):
        belt.GetAcceptanceRegion(ROOT.RooArgSet(a))


def test_an_acceptance_region_and_a_belt_without_points() -> None:
    """A region is its limits and its lookup index; a belt may be made before its points."""
    region = AcceptanceRegion(2, -1.0, 3.0)
    assert (region.GetLookupIndex(), region.GetLowerLimit(), region.GetUpperLimit()) == (2, -1, 3)
    empty = AcceptanceRegion()
    assert empty.GetLookupIndex() == 0 and empty.GetLowerLimit() != empty.GetLowerLimit()
    assert ConfidenceBelt("bare")._points is None
