"""RooFit's binned data, its tables and its histograms, held to what ROOT 6.40 printed.

Every expected number and line below was printed by ROOT itself through
PyROOT for the same data: ``RooDataHist`` from histograms and datasets,
``Roo1DTable`` counts of categories, and ``createHistogram``'s bins and
errors. Where the engine parts from ROOT the test says so and pins only what
the two share.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from xrdroot.hist import Histogram
from xrdroot.roofit import histograms
from xrdroot.roofit.binning import RooUniformBinning
from xrdroot.roofit.categories import RooCategory, RooThresholdCategory
from xrdroot.roofit.cmdargs import RooCmdArg
from xrdroot.roofit.collections import RooArgList, RooArgSet
from xrdroot.roofit.data.datahist import RooDataHist
from xrdroot.roofit.data.dataset import RooDataSet
from xrdroot.roofit.data.table import Roo1DTable
from xrdroot.roofit.functions import RooFormulaVar
from xrdroot.roofit.messages import service
from xrdroot.roofit.pdfs.basic import RooGaussian, ref
from xrdroot.roofit.variables import RooRealVar


@pytest.fixture(autouse=True)
def _plain() -> Iterator[None]:
    """RooFit's first message streams, and histograms handed back as the engine made them.

    ``import ROOT`` installs the ``TH1`` wrapper for good; these tests read the
    engine's own :class:`xrdroot.Histogram`, whichever tests ran before.
    """
    service().reset()
    wrapper = histograms.WRAP[0]
    histograms.set_wrapper(lambda made: made)
    yield
    histograms.set_wrapper(wrapper)
    service().reset()


def th1() -> Histogram:
    """Ten bins over ``[0, 10]``, bin ``i`` holding ``1.5 i + 1`` with error ``(i + 1) / 2``."""
    h = Histogram.book("h", list(np.linspace(0, 10, 11)), title="h")
    centres = np.arange(10) + 0.5
    h.fill(centres, weight=1.5 * np.arange(10) + 1)
    squares = h._sumw2()
    assert squares is not None
    squares[1:11] = (0.5 * (np.arange(10) + 1)) ** 2
    return h


def test_a_histogram_imported_widens_the_range_to_its_bins_saying_so(capsys: Any) -> None:
    """``Import(h)`` moves ``[1.3, 7.6]`` out to ``[1, 8]`` with ROOT's words, and keeps 7 bins."""
    x = RooRealVar("x", "x", 1.3, 7.6)
    dh = RooDataHist("dh", "dh", RooArgList([x]), Import=th1())
    dh.Print()
    assert capsys.readouterr().out == (
        "[#1] INFO:DataHandling -- RooDataHist::adjustBinning(dh): fit range of variable x "
        "expanded to nearest bin boundaries: [1.3,7.6] --> [1,8]\n"
        "RooDataHist::dh[x] = 7 bins (49 weights)\n"
    )
    assert (x.getMin(), x.getMax(), dh.numEntries(), dh.sumEntries()) == (1.0, 8.0, 7, 49.0)
    assert (dh.sum(False), dh.sum(True), dh.sum(True, True)) == (49.0, 49.0, 49.0)
    found = [(dh.get(i).find("x").getVal(), dh.weight(), dh.weightSquared()) for i in range(7)]
    assert found == [
        (1.5, 2.5, 1.0),
        (2.5, 4.0, 2.25),
        (3.5, 5.5, 4.0),
        (4.5, 7.0, 6.25),
        (5.5, 8.5, 9.0),
        (6.5, 10.0, 12.25),
        (7.5, 11.5, 16.0),
    ]
    assert (dh.isWeighted(), dh.isNonPoissonWeighted(), dh.printArgs()) == (True, True, "[x]")


def test_a_histogram_on_its_own_bins_is_imported_without_a_word(capsys: Any) -> None:
    """A range already on the bin edges needs no widening, so ROOT says nothing."""
    z = RooRealVar("z", "z", 0, 10)
    dz = RooDataHist("dz", "dz", z, RooCmdArg("Import", th1()))
    dz.Print()
    assert capsys.readouterr().out == "RooDataHist::dz[z] = 10 bins (77.5 weights)\n"


def test_the_weight_at_a_point_is_the_weight_of_the_bin_it_falls_in() -> None:
    """``weight(row, 0)`` looks the bin up: ROOT's 7 for ``x = 4.2``.

    ROOT's ``weight(row)`` interpolates linearly by default (6.55 here); the
    engine does not interpolate - see the report of this suite.
    """
    x = RooRealVar("x", "x", 1.3, 7.6)
    dh = RooDataHist("dh", "dh", x, Import=th1())
    x.setVal(4.2)
    assert dh.weight(RooArgSet([x]), 0) == 7.0
    y = RooRealVar("x", "x", -50, 50)
    y.setVal(-20)
    assert dh.weight(RooArgSet([y])) == 0.0


def test_a_histogram_of_uneven_bins_imported_as_a_density_weighs_by_the_bin_widths(
    capsys: Any,
) -> None:
    """``Import(h, True)``: a density times each bin's width: ROOT's 2, 8, 18, 32."""
    edges = [0.0, 1.0, 3.0, 6.0, 10.0]
    hv = Histogram.book("hv", edges, title="hv")
    hv.fill([0.5, 2, 4.5, 8], weight=[2.0, 4.0, 6.0, 8.0])
    y = RooRealVar("y", "y", 0.5, 7)
    dv = RooDataHist("dv", "dv", y, RooCmdArg("Import", hv, True))
    assert capsys.readouterr().out == (
        "[#1] INFO:DataHandling -- RooDataHist::adjustBinning(dv): fit range of variable y "
        "expanded to nearest bin boundaries: [0.5,7] --> [0,10]\n"
    )
    assert (y.getMin(), y.getMax(), y.getBins(), dv.numEntries()) == (0.0, 10.0, 4, 4)
    assert (dv.sumEntries(), dv.sum(True), dv.sum(True, True)) == (60.0, 200.0, 20.0)
    found = [(dv.get(i).find("y").getVal(), dv.weight()) for i in range(4)]
    assert found == [(0.5, 2.0), (2.0, 8.0), (4.5, 18.0), (8.0, 32.0)]
    assert list(dv.binVolumes()) == [1.0, 2.0, 3.0, 4.0]


def test_a_two_dimensional_histogram_is_imported_bin_by_bin_x_slowest() -> None:
    """A ``TH2``'s bins come in ROOT's order: the first variable's bins slowest."""
    h2 = Histogram.book("h2", [0, 1, 2, 3, 4], [0, 1, 2, 3], title="h2")
    for i in range(4):
        for j in range(3):
            h2.fill([i + 0.5], [j + 0.5], weight=[10.0 * i + j])
    x = RooRealVar("x", "x", 0, 4)
    y = RooRealVar("y", "y", 0, 3)
    d2 = RooDataHist("d2", "d2", RooArgList([x, y]), Import=h2)
    assert d2.printValue() == "12 bins (192 weights)"
    found = [(d2.get(i).find("x").getVal(), d2.get(i).find("y").getVal(), d2.weight())
             for i in (0, 1, 2, 3, 11)]  # fmt: skip
    assert found == [(0.5, 0.5, 0.0), (0.5, 1.5, 1.0), (0.5, 2.5, 2.0), (1.5, 0.5, 10.0),
                     (3.5, 2.5, 32.0)]  # fmt: skip


class Wrapped:
    """What PyROOT's ``TH1`` is to the engine: something holding a histogram as ``_xrd``."""

    def __init__(self, held: Histogram) -> None:
        self._xrd = held


def weighted_xz() -> tuple[RooDataSet, RooRealVar, RooRealVar]:
    """Four weighted events in ``x`` of 4 bins and ``z`` of 2, one on ``x``'s upper edge."""
    x = RooRealVar("x", "x", 0, 4)
    z = RooRealVar("z", "z", 0, 2)
    w = RooRealVar("w", "w", 0, 10)
    x.setBins(4)
    z.setBins(2)
    d = RooDataSet("d", "d", RooArgSet([x, z, w]), WeightVar="w")
    d.add_columns({"x": [0.5, 4.0, 1.2, 1.7], "z": [0.5, 2.0, 1.5, 1.5]}, [1.0, 2.5, 0.5, 3.0])
    return d, x, z


def test_a_dataset_binned_sums_its_weights_and_their_squares_per_bin() -> None:
    """A ``RooDataHist`` of weighted events: ROOT's weights and squared weights, bin by bin."""
    d, x, z = weighted_xz()
    dh = RooDataHist("dh", "dh", RooArgSet([x, z]), d)
    assert dh.printValue() == "8 bins (7 weights)"
    found = [
        (
            dh.get(i).find("x").getVal(),
            dh.get(i).find("z").getVal(),
            dh.weight(),
            dh.weightSquared(),
        )
        for i in range(8)
    ]
    assert found == [
        (0.5, 0.5, 1.0, 1.0),
        (0.5, 1.5, 0.0, 0.0),
        (1.5, 0.5, 0.0, 0.0),
        (1.5, 1.5, 3.5, 9.25),
        (2.5, 0.5, 0.0, 0.0),
        (2.5, 1.5, 0.0, 0.0),
        (3.5, 0.5, 0.0, 0.0),
        (3.5, 1.5, 2.5, 6.25),
    ]
    assert dh.isNonPoissonWeighted()
    again = RooDataHist("again", "again", RooArgSet([x, z]), Import=d)
    assert list(again.weights()) == list(dh.weights())


def test_an_empty_binned_dataset_has_every_bin_and_no_weight(capsys: Any) -> None:
    """``RooDataHist(name, title, vars)`` alone: ROOT's four empty bins."""
    x = RooRealVar("x", "x", 0, 4)
    x.setBins(4)
    e = RooDataHist("e", "e", RooArgSet([x]))
    e.Print()
    assert capsys.readouterr().out == "RooDataHist::e[x] = 4 bins (0 weights)\n"
    assert (e.sumEntries(), e.isNonPoissonWeighted()) == (0.0, False)
    none = RooDataHist("none", "none")
    assert (none.numEntries(), list(none.binVolumes())) == (1, [1.0])


def test_a_pyroot_histogram_is_imported_through_what_it_holds() -> None:
    """A ``TH1`` from ``import ROOT`` is read through its engine histogram."""
    z = RooRealVar("z", "z", 0, 10)
    dz = RooDataHist("dz", "dz", z, Import=Wrapped(th1()))
    assert dz.sumEntries() == 77.5


def test_binned_data_is_its_own_binned_clone_and_plots_in_its_own_bins() -> None:
    """A frame shows binned data in the data's bins: ROOT's 1, 3.5, 0, 2.5 at the centres."""
    d, x, z = weighted_xz()
    dh = RooDataHist("dh", "dh", RooArgSet([x, z]), d)
    assert dh.binnedClone() is dh
    frame = RooRealVar("x", "x", 0, 4).frame(Bins=8)
    dh.plotOn(frame)
    points = frame.getObject(0)
    assert (list(points.x), list(points.y)) == ([0.5, 1.5, 2.5, 3.5], [1.0, 3.5, 0.0, 2.5])
    assert dh.default_binning(RooRealVar("q", "q", 0, 1)) is None


def categorised() -> tuple[RooDataSet, RooRealVar, RooCategory, RooCategory]:
    """Twelve events: ``c`` cycling ``A, B, C``, ``k`` taking ``up`` twice then ``down`` twice."""
    x = RooRealVar("x", "x", 0, 10)
    c = RooCategory("c", "c")
    for label, index in (("A", 0), ("B", 1), ("C", 2)):
        c.defineType(label, index)
    k = RooCategory("k", "k")
    k.defineType("up", 0)
    k.defineType("down", 5)
    d = RooDataSet("d", "d", RooArgSet([x, c, k]))
    d.add_columns(
        {
            "x": [i * 0.8 + 0.1 for i in range(12)],
            "c": [i % 3 for i in range(12)],
            "k": [0 if i % 4 < 2 else 5 for i in range(12)],
        }
    )
    return d, x, c, k


def test_a_table_counts_the_events_in_each_state_as_root_prints_it(capsys: Any) -> None:
    """``data.table(c)`` and its verbose box, with a cut in the title: ROOT's text."""
    d, _, c, _ = categorised()
    d.table(c).Print()
    d.table(c, "x>3").Print("v")
    assert capsys.readouterr().out == (
        "Roo1DTable::c = (A=4,B=4,C=4)\n"
        "\n"
        "  Table c : d(x>3)\n"
        "  +---+---+\n"
        "  | A | 2 |\n"
        "  | B | 3 |\n"
        "  | C | 3 |\n"
        "  +---+---+\n"
        "\n"
    )
    table = d.table(c)
    assert (table.get("B"), table.getFrac("B"), table.getOverflow()) == (4.0, 1 / 3, 0.0)
    assert (table.GetName(), table.GetTitle(), table.printClassName()) == ("c", "d", "Roo1DTable")


def test_a_table_of_several_categories_counts_every_combination(capsys: Any) -> None:
    """A set of categories is tabled as their product, the first fastest: ROOT's text."""
    d, _, c, k = categorised()
    d.table(RooArgSet([c, k])).Print()
    d.table(RooArgSet([c, k]), "x>5").Print("v")
    assert capsys.readouterr().out == (
        "Roo1DTable::(c x k) = ({A;up}=2,{B;up}=2,{C;up}=2,{A;down}=2,{B;down}=2,{C;down}=2)\n"
        "\n"
        "  Table (c x k) : d(x>5)\n"
        "  +----------+---+\n"
        "  |   {A;up} | 1 |\n"
        "  |   {B;up} | 0 |\n"
        "  |   {C;up} | 1 |\n"
        "  | {A;down} | 0 |\n"
        "  | {B;down} | 2 |\n"
        "  | {C;down} | 1 |\n"
        "  +----------+---+\n"
        "\n"
    )


def test_a_table_of_a_derived_category_computes_its_state_per_event() -> None:
    """A category the dataset has no column of is worked out event by event: four of each."""
    d, x, *_ = categorised()
    th = RooThresholdCategory("th", "th", x, "hi", 2)
    th.addThreshold(3.0, "lo", 0)
    th.addThreshold(6.0, "mid", 1)
    table = d.table(th)
    assert [table.get(label) for label in ("lo", "mid", "hi")] == [4.0, 4.0, 4.0]


def test_an_empty_table_prints_every_state_and_a_fraction_of_nothing(capsys: Any) -> None:
    """With no events the verbose box still lists each state: ROOT's zeros."""
    x = RooRealVar("x", "x", 0, 10)
    c = RooCategory("c", "c")
    for label, index in (("A", 0), ("B", 1), ("C", 2)):
        c.defineType(label, index)
    e = RooDataSet("e", "e", RooArgSet([x, c]))
    table = e.table(c)
    table.Print("v")
    assert capsys.readouterr().out == (
        "\n  Table c : e\n  +---+---+\n  | A | 0 |\n  | B | 0 |\n  | C | 0 |\n  +---+---+\n\n"
    )
    assert table.getFrac("A") == 0.0


def test_a_table_of_weights_widens_its_column_to_the_biggest_count(capsys: Any) -> None:
    """Counts are summed weights, printed ``%g`` in a column as wide as the largest: ROOT's box."""
    x = RooRealVar("x", "x", 0, 10)
    c = RooCategory("c", "c")
    for label, index in (("A", 0), ("B", 1), ("C", 2)):
        c.defineType(label, index)
    w = RooRealVar("w", "w", 0, 1e6)
    big = RooDataSet("big", "big", RooArgSet([x, c, w]), WeightVar="w")
    big.add_columns({"x": [1.0, 2.0], "c": [0, 1]}, [12345.0, 0.25])
    big.table(c).Print("v")
    big.table(c).Print()
    assert capsys.readouterr().out == (
        "\n"
        "  Table c : big\n"
        "  +---+-------+\n"
        "  | A | 12345 |\n"
        "  | B |  0.25 |\n"
        "  | C |     0 |\n"
        "  +---+-------+\n"
        "\n"
        "Roo1DTable::c = (A=12345,B=0.25)\n"
    )


def test_a_table_made_by_hand_prints_only_its_nonzero_states_unless_verbose() -> None:
    """``printMultiline`` leaves out empty states in its standard form, as ROOT's does."""
    table = Roo1DTable("t", "title", ["long", "s"], [3.0, 0.0])
    assert table.printStream(table.defaultPrintContents(""), 3) == (
        "\n  Table t : title\n  +------+---+\n  | long | 3 |\n  +------+---+\n\n"
    )


def ten_weighted() -> tuple[RooDataSet, RooRealVar, RooRealVar]:
    """Ten events on ``x`` of 5 bins and ``y`` of 2, event ``i`` weighing ``(i + 1) / 2``."""
    x = RooRealVar("x", "x", 0, 10)
    y = RooRealVar("y", "y", -5, 5)
    w = RooRealVar("w", "w", 0, 10)
    x.setBins(5)
    y.setBins(2)
    d = RooDataSet("d", "d", RooArgSet([x, y, w]), WeightVar="w")
    d.add_columns(
        {"x": [i + 0.5 for i in range(10)], "y": [i - 4.5 for i in range(10)]},
        [0.5 * (i + 1) for i in range(10)],
    )
    return d, x, y


def contents(h: Histogram) -> tuple[list[Any], list[float]]:
    """The bins' contents, and their errors to six places, as the ROOT run printed them."""
    return h.values().tolist(), [round(float(e), 6) for e in h.errors().reshape(-1)]


def test_a_histogram_of_data_sums_the_weights_in_the_variables_own_bins() -> None:
    """``createHistogram("h1", x)``: ROOT's five bins of summed weights and their errors."""
    d, x, _ = ten_weighted()
    h = d.createHistogram("h1", x)
    assert (h.name.startswith("h1"), list(h.edges())) == (True, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    assert contents(h) == (
        [1.5, 3.5, 5.5, 7.5, 9.5],
        [1.118034, 2.5, 3.905125, 5.315073, 6.726812],
    )


def test_a_binning_command_sets_the_bins_by_number_by_range_or_by_name() -> None:
    """``Binning(4)``, ``Binning(4, 2, 6)`` and ``Binning("coarse")``: ROOT's contents for each."""
    d, x, _ = ten_weighted()
    four = d.createHistogram("h2", x, RooCmdArg("Binning", 4))
    window = d.createHistogram("h3", x, RooCmdArg("Binning", 4, 2, 6))
    x.setBinning(RooUniformBinning(0, 10, 2), "coarse")
    coarse = d.createHistogram("h5", x, RooCmdArg("Binning", "coarse"))
    given = d.createHistogram("h8", x, RooCmdArg("Binning", RooUniformBinning(0, 10, 2)))
    assert contents(four) == ([1.5, 6.0, 6.5, 13.5], [1.118034, 3.535534, 4.609772, 7.826238])
    assert contents(window) == ([1.5, 2.0, 2.5, 3.0], [1.5, 2.0, 2.5, 3.0])
    assert list(window.edges()) == [2.0, 3.0, 4.0, 5.0, 6.0]
    assert contents(coarse) == ([7.5, 20.0], [3.708099, 9.082951])
    assert contents(given) == contents(coarse)


def test_a_cut_keeps_only_its_events_out_of_the_histogram() -> None:
    """``Cut("y>0")``: ROOT's bins hold only the last five events."""
    d, x, _ = ten_weighted()
    h = d.createHistogram("h6", x, Cut="y>0")
    assert contents(h) == ([0.0, 0.0, 3.0, 7.5, 9.5], [0.0, 0.0, 3.0, 5.315073, 6.726812])


def test_a_second_variable_makes_a_two_dimensional_histogram() -> None:
    """``YVar(y, Binning(2))``, ``YVar(y)`` and ``"x,y"`` all bin both: ROOT's contents.

    A named histogram of two variables fails in the engine's own hands (the
    name is written where only a ``TH1`` keeps it), so ``YVar`` is given to
    the histogram maker directly - see the report of this suite.
    """
    d, x, y = ten_weighted()
    four = histograms.data_histogram(
        d, x, (RooCmdArg("Binning", 4),), {"YVar": (y, RooCmdArg("Binning", 2))}
    )
    own = histograms.data_histogram(d, x, (RooCmdArg("YVar", y),), {})
    named = d.createHistogram("x,y", RooCmdArg("Binning", 5), RooCmdArg("Binning", 2))
    by_name = d.createHistogram("x,y")
    assert four.values().tolist() == [[1.5, 0.0], [6.0, 0.0], [0.0, 6.5], [0.0, 13.5]]
    expected = [[1.5, 0.0], [3.5, 0.0], [2.5, 3.0], [0.0, 7.5], [0.0, 9.5]]
    assert own.values().tolist() == named.values().tolist() == by_name.values().tolist() == expected


def test_a_histogram_named_by_its_variable_alone_has_the_variables_bins() -> None:
    """``createHistogram("x")``: the variable's own five bins, ROOT's contents."""
    d, *_ = ten_weighted()
    h = d.createHistogram("x")
    assert contents(h)[0] == [1.5, 3.5, 5.5, 7.5, 9.5]
    assert histograms.names_of([d.variable("x"), d.variable("y")]) == ["x", "y"]


def test_a_third_variable_makes_a_three_dimensional_histogram() -> None:
    """``ZVar`` adds the third axis; every event lands in its own cell of it."""
    _, x, y = ten_weighted()
    z = RooRealVar("z", "z", 0, 1)
    z.setBins(1)
    d3 = RooDataSet("d3", "d3", RooArgSet([x, y, z]))
    d3.add_columns({"x": [0.5, 9.5], "y": [-4.5, 4.5], "z": [0.5, 0.5]})
    h = histograms.data_histogram(d3, x, (), {"YVar": y, "ZVar": (z, RooCmdArg("Binning", 2))})
    assert (h.values().shape, float(h.values().sum())) == ((5, 2, 2), 2.0)
    assert (h.values()[0, 0, 1], h.values()[4, 1, 1]) == (1.0, 1.0)


def test_a_histogram_of_a_density_is_its_probability_in_each_bin() -> None:
    """``pdf.createHistogram``: the normalised density times each bin's width, ROOT's values."""
    x = RooRealVar("x", "x", 0, 10)
    g = RooGaussian("g", "g", x, ref(5.0), ref(2.0))
    h = g.createHistogram("gh", x, RooCmdArg("Binning", 5))
    expected = [0.054669931530952454, 0.24501362442970276, 0.403959184885025]
    assert h.values().tolist() == pytest.approx([*expected, expected[1], expected[0]], rel=1e-7)
    f = RooFormulaVar("f", "f", "x*x", [x])
    assert f.createHistogram("fh", x, RooCmdArg("Binning", 5)).values().shape == (5,)
