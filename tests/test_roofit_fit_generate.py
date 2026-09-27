"""Generated datasets, bit for bit as ROOT 6.40.04 draws them.

Each reference was printed by ROOT through PyROOT with ``repr`` after
``RooRandom::randomGenerator()->SetSeed(4357)``, for the same densities
built here from the engine's classes. The events, the counts, and the
generator's next number afterwards - which says every draw was taken, and
in ROOT's order - must all be ROOT's.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from xrdroot.roofit.categories import RooCategory
from xrdroot.roofit.pdfs.addpdf import RooAddPdf
from xrdroot.roofit.pdfs.basic import RooExponential, RooGaussian, RooPolynomial
from xrdroot.roofit.pdfs.extend import RooExtendPdf
from xrdroot.roofit.pdfs.prodpdf import RooProdPdf
from xrdroot.roofit.rng import generator
from xrdroot.roofit.variables import RooConstVar, RooRealVar

#: ROOT's first five Gaussian events.
FIRST = [
    1.9978654352187961,
    -0.8695287788286805,
    1.5635925123910157,
    -0.06010554265230894,
    1.648527370025855,
]


def _gauss() -> tuple[Any, Any, Any, Any]:
    """A Gaussian of mean 0 and width 2 in x on [-10, 10], and ROOT's seed."""
    generator().SetSeed(4357)
    x = RooRealVar("x", "x", -10, 10)
    y = RooRealVar("y", "y", 0, 5)
    m = RooRealVar("m", "m", 0, -5, 5)
    s = RooRealVar("s", "s", 2, 0.1, 10)
    return RooGaussian("g", "g", x, m, s), x, y, s


def _column(data: Any, name: str) -> list[float]:
    return [float(v) for v in data.column(name)]


def test_a_count_by_position_or_option_draws_that_many_named_after_the_density() -> None:
    """``generate(x, n)``, ``NumEvents(n)`` and ``Name``: the dataset is ``gData``, "Generated
    From g", unless named; a count of 2.5 draws 3, as ROOT's loop does."""
    g, x, _, _ = _gauss()
    data = g.generate([x], NumEvents=3)
    assert (data.GetName(), data.GetTitle(), _column(data, "x")) == (
        "gData",
        "Generated From g",
        FIRST[:3],
    )
    generator().SetSeed(4357)
    named = g.generate([x], 2, Name="foo")
    assert (named.GetName(), _column(named, "x")) == ("foo", FIRST[:2])
    assert generator().Rndm() == 0.9472010820172727
    generator().SetSeed(4357)
    assert _column(g.generate([x], 2.5), "x") == FIRST[:3]


@pytest.mark.xfail(strict=True, reason="no count for a density without a yield is refused")
def test_no_count_for_a_density_without_a_yield_is_an_empty_dataset() -> None:
    """ROOT hands back an empty dataset, draws nothing, and says nothing."""
    g, x, _, _ = _gauss()
    assert g.generate([x]).numEntries() == 0
    assert _column(g.generate([x], 2), "x") == FIRST[:2]


def test_no_count_for_a_density_without_a_yield_is_refused_by_name(capsys: Any) -> None:
    """Asked for events and told nowhere how many, xrdroot names the density and returns
    nothing, drawing no number."""
    g, x, _, _ = _gauss()
    capsys.readouterr()
    assert g.generate([x]) is None
    assert capsys.readouterr().out == (
        "[#0] ERROR:Generation -- RooGenContext::g:generate: PDF not extendable: cannot "
        "calculate expected number of events\n"
    )
    assert _column(g.generate([x], 2), "x") == FIRST[:2]


def _sum(second: float) -> tuple[Any, Any]:
    """A Gaussian of 2.2 events plus an exponential of ``second``, in x, and ROOT's seed."""
    g, x, _, _ = _gauss()
    e = RooExponential("e", "e", x, RooRealVar("a", "a", -0.2, -1, 0))
    n1 = RooRealVar("n1", "n1", 2.2, 0, 100)
    n2 = RooRealVar("n2", "n2", second, 0, 100)
    return RooAddPdf("model", "model", [g, e], [n1, n2]), x


#: ROOT's events of that sum, picking a component by a uniform draw for each.
SUMMED = [0.6098192620720511, -9.73501519241836, 1.1120699475215172, -1.73628337788473]


def test_a_sum_picks_a_component_for_each_event_and_draws_as_many_as_it_expects() -> None:
    """With no count, a sum of yields draws its expected number - 4.6, so five - each from the
    component a uniform draw picks, as ``RooAddGenContext`` does."""
    model, x = _sum(2.4)
    data = model.generate([x])
    assert _column(data, "x") == [*SUMMED, 3.060153005644679]
    assert generator().Rndm() == 0.9189364027697593


@pytest.mark.xfail(strict=True, reason="an expected count is rounded up, not to the nearest")
def test_an_expected_count_is_rounded_to_the_nearest_event() -> None:
    """ROOT draws four events where 4.4 are expected, and five for 4.6."""
    model, x = _sum(2.2)
    assert _column(model.generate([x]), "x") == SUMMED
    assert generator().Rndm() == 0.40251527237705886


def test_an_extended_generation_draws_a_poisson_number_of_events_first() -> None:
    """``Extended()`` draws the count from a Poisson of the expected number, then the events."""
    model, x = _sum(2.2)
    data = model.generate([x], Extended=True)
    assert _column(data, "x") == [
        7.775137291137071,
        -0.1906637832325524,
        -5.713806628959901,
        -8.42627061138046,
        -2.50252241268754,
        -6.012172393848232,
        3.9028039643528865,
        -7.648191008320282,
        -0.9797168622844765,
        -0.061780205853060544,
        -2.1555617451667786,
    ]
    assert generator().Rndm() == 0.4889833992347121


def test_a_variable_the_density_does_not_depend_on_is_drawn_uniformly_after_it() -> None:
    """A real variable is uniform over its range, a category takes a state uniformly by the
    order they were defined; each after the density's own, in reverse order of name."""
    g, x, y, _ = _gauss()
    c = RooCategory("c", "c")
    for label, index in (("B", 3), ("A", 1), ("C", 7)):
        c.defineType(label, index)
    data = g.generate([x, y, c], 4)
    rows = list(zip(_column(data, "x"), _column(data, "y"), _column(data, "c")))
    assert rows == [
        (1.9978654352187961, 0.8145493769552559, 3.0),
        (1.5635925123910157, 1.1582827137317508, 1.0),
        (1.648527370025855, 3.7215267156716436, 1.0),
        (-0.14940893396964228, 3.799718990921974, 1.0),
    ]
    assert generator().Rndm() == 0.31563762156292796


def test_sums_products_and_conditional_products_draw_roots_events_in_turn() -> None:
    """A sum of a Gaussian and an exponential by a fraction, then a product of independent
    Gaussians, then ``gy(y) * gc(x|y)`` - which draws y first - from one run of the generator."""
    from xrdroot.roofit.cmdargs import RooCmdArg

    g, x, y, s = _gauss()
    e = RooExponential("e", "e", x, RooRealVar("a", "a", -0.2, -1, 0))
    f = RooRealVar("f", "f", 0.3, 0, 1)
    model = RooAddPdf("model", "model", [g, e], [f])
    assert _column(model.generate([x], 5), "x") == [
        0.6098192620720511,
        -9.73501519241836,
        1.1120699475215172,
        -5.787699592081026,
        -3.3675587509014804,
    ]
    gy = RooGaussian("gy", "gy", y, RooConstVar("two", "", 2), RooConstVar("one", "", 1))
    prod = RooProdPdf("prod", "prod", [g, gy])
    data = prod.generate([x, y], 3)
    assert list(zip(_column(data, "x"), _column(data, "y"))) == [
        (-3.846468667580453, 2.149059345813937),
        (-3.8431369418445582, 0.4737658347003162),
        (-2.5118401564207, 0.74873879365623),
    ]
    gc = RooGaussian("gc", "gc", x, y, s)
    cprod = RooProdPdf("cprod", "cprod", [gy], RooCmdArg("Conditional", [gc], [x]))
    data = cprod.generate([x, y], 3)
    assert list(zip(_column(data, "x"), _column(data, "y"))) == [
        (1.307724561127713, 2.9837936905312676),
        (-0.5422426671529577, 1.6089083882980049),
        (8.413123339388592, 3.951401982176443),
    ]
    assert generator().Rndm() == 0.5015338051598519


def _proto() -> tuple[Any, Any, Any, Any]:
    """Three values of y from a Gaussian - ROOT's - to generate x with, and x's Gaussian."""
    _, x, y, s = _gauss()
    gy = RooGaussian("gy", "gy", y, RooConstVar("two", "", 2), RooConstVar("one", "", 1))
    proto = gy.generate([y], 3)
    return RooGaussian("gc", "gc", x, y, s), proto, x, y


#: The prototype's values of y, in ROOT's order.
PROTO_Y = [2.998932717609398, 1.5652356105856597, 2.781796256195508]


def test_prototype_data_gives_each_event_its_conditional_values_round_again_if_needed() -> None:
    """With ``ProtoData`` the mean of each event's Gaussian is the prototype's y: as many
    events as it has, or more, taking its events round again."""
    gc, proto, x, y = _proto()
    assert _column(proto, "y") == PROTO_Y
    data = gc.generate([x], ProtoData=proto)
    assert (_column(data, "x"), _column(data, "y")) == (
        [2.938827174957089, 3.2137629806115147, 2.6683616052374446],
        PROTO_Y,
    )
    data = gc.generate([x], 5, ProtoData=proto)
    assert _column(data, "x") == [
        1.1971807283442435,
        1.4158266766160175,
        2.7976206722645007,
        2.1774063844482483,
        4.3476235517300665,
    ]
    assert _column(data, "y") == PROTO_Y + PROTO_Y[:2]
    assert generator().Rndm() == 0.5196721151005477
    assert y.getVal() == 2.5


def test_a_density_that_cannot_draw_itself_with_prototype_data_is_accepted_or_rejected() -> None:
    """A line in xs, with prototype data of a variable it does not use: RooFit finds its
    largest value by trial points and then accepts uniform draws below it."""
    _, proto, _, _ = _proto()
    xs = RooRealVar("xs", "xs", -1, 1)
    p = RooPolynomial("p", "p", xs, [RooRealVar("b", "b", 0.5, -1, 1)])
    data = p.generate([xs], ProtoData=proto)
    assert _column(data, "xs") == [0.8403609124943614, -0.09022854268550873, 0.019178228452801704]
    assert _column(data, "y") == PROTO_Y
    assert generator().Rndm() == 0.4069547592662275


def test_a_conditional_density_is_accepted_or_rejected_over_both_its_variables(
    capsys: Any,
) -> None:
    """``1 + b xs`` with b from the prototype: two dimensions, so a hundred thousand trial
    points and RooFit's warning that sharp peaks may be missed - and ROOT's events."""
    from xrdroot.roofit.data.dataset import RooDataSet

    generator().SetSeed(4357)
    xs = RooRealVar("xs", "xs", -1, 1)
    b = RooRealVar("b", "b", 0, 1)
    proto = RooDataSet("proto", "proto", [b])
    proto.add_columns({"b": np.array([0.2, 0.9])})
    p = RooPolynomial("p", "p", xs, [b])
    capsys.readouterr()
    data = p.generate([xs], ProtoData=proto)
    assert _column(data, "xs") == [0.797475416213274, 0.7720118770375848]
    assert generator().Rndm() == 0.35202217381447554
    out = capsys.readouterr().out
    assert "WARNING: performing accept/reject sampling on a p.d.f in 2 dimensions" in out
    assert "Determining maximum value by taking 100000 trial samples." in out


def _binned() -> tuple[Any, Any, Any]:
    """A Gaussian of mean 0.5 and width 2 in six bins of [-5, 5], with a yield of 20.3."""
    generator().SetSeed(4357)
    x = RooRealVar("x", "x", -5, 5)
    x.setBins(6)
    g = RooGaussian("g", "g", x, RooRealVar("m", "m", 0.5, -5, 5), RooRealVar("s", "s", 2, 0.1, 10))
    return g, RooExtendPdf("e", "e", g, RooRealVar("n", "n", 20.3, 0, 100)), x


def _counts(hist: Any) -> list[float]:
    return [float(v) for v in hist.weights()]


def test_a_binned_generation_draws_poisson_bins_and_moves_events_to_the_total() -> None:
    """Each bin is a Poisson draw; then bins picked by accept-reject are raised or lowered until
    the total is the one asked for - ROOT's counts, and ROOT's generator after."""
    g, _, x = _binned()
    first = g.generateBinned([x], 20)
    assert (first.GetName(), first.numEntries(), first.sumEntries()) == ("genData", 6, 20.0)
    assert _counts(first) == [1.0, 0.0, 10.0, 4.0, 4.0, 1.0]
    assert _counts(g.generateBinned([x], 3)) == [0.0, 0.0, 1.0, 0.0, 2.0, 0.0]
    big = g.generateBinned([x], 60, Name="big")
    assert (big.GetName(), _counts(big)) == ("big", [1.0, 7.0, 20.0, 18.0, 10.0, 4.0])
    assert generator().Rndm() == 0.4456630467902869


def test_an_extended_binned_generation_leaves_each_bin_its_poisson_draw() -> None:
    """``Extended()`` keeps the Poisson draws as they fall; with no count, a density with a
    yield draws its expected number rounded - 20 of 20.3."""
    g, e, x = _binned()
    assert _counts(g.generateBinned([x], 20, Extended=True)) == [1, 2, 10, 4, 6, 1]
    assert _counts(e.generateBinned([x], Extended=True)) == [0, 1, 6, 4, 2, 1]
    plain = e.generateBinned([x])
    assert (_counts(plain), plain.sumEntries()) == ([0, 1, 6, 7, 5, 1], 20.0)
    assert generator().Rndm() == 0.37043020129203796


@pytest.mark.xfail(strict=True, reason="each bin's sum of squares is not sqrt(n) squared")
def test_a_binned_generation_keeps_each_bins_error_as_the_square_of_its_root() -> None:
    """ROOT stores ``sqrt(n)`` as each bin's error, so the sum of squares is its square: 10
    events carry 10.000000000000002."""
    g, _, x = _binned()
    hist = g.generateBinned([x], 20)
    assert [float(v) for v in hist.weights_squared()] == [1, 0, 10.000000000000002, 4, 4, 1]


@pytest.mark.xfail(strict=True, reason="expected bins do not add up to the count asked for")
def test_expected_data_is_the_count_shared_out_by_each_bins_share() -> None:
    """``ExpectedData`` - or ``Asimov`` - fills each bin with its share of the events, the
    shares normalised over the bins: 20 events are 20.0 in all, and 20.3 expected 20.3."""
    g, e, x = _binned()
    hist = g.generateBinned([x], 20, ExpectedData=True)
    assert hist.sumEntries() == 20.0
    assert _counts(hist) == pytest.approx(
        [
            0.442597482225,
            2.186118686593,
            5.391941154226,
            6.640842203689,
            4.084207398695,
            1.2542930746,
        ],
        rel=1e-12,
    )
    assert e.generateBinned([x], Asimov=True).sumEntries() == pytest.approx(20.3, rel=1e-14)


def test_expected_data_draws_no_random_numbers() -> None:
    """An Asimov dataset is the expectation itself: the generator is where it was."""
    g, e, x = _binned()
    first = _counts(g.generateBinned([x], 20, ExpectedData=True))
    assert _counts(e.generateBinned([x], ExpectedData=True)) == pytest.approx(
        [v * 20.3 / 20 for v in first], rel=1e-12
    )
    assert generator().Rndm() == 0.999741748906672


def test_a_binned_generation_without_a_count_needs_a_yield(capsys: Any) -> None:
    """Told nowhere how many events, a density without a yield cannot be generated binned."""
    g, _, x = _binned()
    capsys.readouterr()
    assert g.generateBinned([x]) is None
    assert capsys.readouterr().out == (
        "[#0] ERROR:InputArguments -- RooAbsPdf::generateBinned(g) ERROR: No event count provided "
        "and p.d.f does not provide expected number of events\n"
    )
    assert generator().Rndm() == 0.999741748906672
