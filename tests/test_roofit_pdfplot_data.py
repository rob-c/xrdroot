"""RooFit's frames and the data drawn on them, held to what ROOT 6.40 drew and printed.

Every point, error bar and message below is ROOT's, printed through PyROOT
for the same events - generated after ``SetSeed(4357)``, which the engine
reproduces bit for bit - binned and drawn with the same options.
"""

from __future__ import annotations

from typing import Any

import pytest

from xrdroot.roofit.binning import RooBinning
from xrdroot.roofit.collections import RooArgSet
from xrdroot.roofit.data.dataset import RooDataSet
from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.rng import generator
from xrdroot.roofit.variables import RooRealVar

#: ``h_gData`` in the frame's 20 bins: (x, y, -error, +error) per bin, as ROOT drew it.
DEFAULT = [
    (-9.5, 0, 0, 1.1478745),
    (-8.5, 0, 0, 1.1478745),
    (-7.5, 0, 0, 1.1478745),
    (-6.5, 0, 0, 1.1478745),
    (-5.5, 0, 0, 1.1478745),
    (-4.5, 1, 0.82724622, 2.2995266),
    (-3.5, 7, 2.5814705, 3.7702807),
    (-2.5, 8, 2.7683861, 3.9451415),
    (-1.5, 20, 4.434448, 5.5465192),
    (-0.5, 26, 5.0660149, 6.1643241),
    (0.5, 60, 7.7243171, 8.7890231),
    (1.5, 26, 5.0660149, 6.1643241),
    (2.5, 22, 4.6545024, 5.7613664),
    (3.5, 16, 3.957801, 5.0830656),
    (4.5, 10, 3.1086944, 4.2669498),
    (5.5, 4, 1.9143392, 3.1627532),
    (6.5, 0, 0, 1.1478745),
    (7.5, 0, 0, 1.1478745),
    (8.5, 0, 0, 1.1478745),
    (9.5, 0, 0, 1.1478745),
]


def points(hist: Any) -> list[tuple[float, ...]]:
    low, high = hist.errors()
    return [
        (float(x), float(y), float(lo), float(hi))
        for x, y, lo, hi in zip(hist.x, hist.y, low, high)
    ]


def flat(rows: Any) -> list[float]:
    return [float(v) for row in rows for v in row]


def x_errors(hist: Any) -> list[tuple[float, float]]:
    return list(zip(hist.members["fEXlow"], hist.members["fEXhigh"]))


def generated() -> tuple[RooRealVar, Any]:
    x = RooRealVar("x", "x", 0, -10, 10)
    x.setBins(20)
    x.setRange("win", -1.5, 2.5)
    g = RooGaussian("g", "g", x, RooRealVar("m", "m", 1, -5, 5), RooRealVar("s", "s", 2, 0.1, 10))
    generator().SetSeed(4357)
    return x, g.generate([x], 200)


def test_a_frame_is_the_variables_range_in_its_bins_titled_after_it() -> None:
    """``x.frame()``: the variable's bins over its range, ``A RooPlot of "x"``."""
    x, _ = generated()
    frame = x.frame()
    assert frame.GetTitle() == 'A RooPlot of "x"'
    assert (frame.GetNbinsX(), frame.GetXmin(), frame.GetXmax()) == (20, -10.0, 10.0)
    assert frame.ClassName() == "RooPlot"
    assert frame.getPlotVar() is x
    assert frame.GetName().startswith("frame_x_")


def test_a_frame_takes_bins_a_range_a_title_and_a_name() -> None:
    """``frame(Bins(10), Range("win"), Title("T"), Name("fr"))``, ``frame(5)``, ``frame(-4, 4)``."""
    x, _ = generated()
    named = x.frame(Bins=10, Range="win", Title="T", Name="fr")
    assert (named.GetName(), named.GetTitle(), named.GetNbinsX()) == ("fr", "T", 10)
    assert (named.GetXmin(), named.GetXmax()) == (-1.5, 2.5)
    assert x.frame(5).GetNbinsX() == 5
    ranged = x.frame(-4, 4, 8)
    assert (ranged.GetNbinsX(), ranged.GetXmin(), ranged.GetXmax()) == (8, -4.0, 4.0)
    assert x.frame(-4, 4).GetNbinsX() == 20
    numbers = x.frame(Range=(-2.0, 2.0))
    assert (numbers.GetXmin(), numbers.GetXmax()) == (-2.0, 2.0)


def test_data_are_drawn_as_poisson_intervals_in_the_frames_bins_as_root_draws_them() -> None:
    """Each bin a point at its centre, its bar the central 68% Poisson interval."""
    x, data = generated()
    frame = x.frame()
    data.plotOn(frame)
    hist = frame.getObject(0)
    assert hist.GetName() == "h_gData"
    assert hist.ClassName() == "RooHist"
    assert flat(points(hist)) == pytest.approx(flat(DEFAULT), rel=1e-7)
    assert x_errors(hist) == [(0.5, 0.5)] * 20
    assert frame.GetMaximum() == pytest.approx(72.228474, rel=1e-7)
    assert frame.GetMinimum() == 0.0
    assert frame.GetYaxis().GetTitle() == "Events / ( 1 )"


def test_data_are_binned_as_binning_says_named_and_coloured() -> None:
    """``Binning(8)`` over the variable's range, and the name and colour given."""
    x, data = generated()
    frame = x.frame(Bins=10, Range="win")
    data.plotOn(frame, Binning=8, MarkerColor=2, Name="mine")
    hist = frame.getObject(0)
    assert hist.GetName() == "mine"
    assert [p[1] for p in points(hist)] == [0, 0, 12, 50, 90, 44, 4, 0]
    assert points(hist)[4] == pytest.approx((1.25, 90, 9.4691773, 10.521997), rel=1e-7)
    assert x_errors(hist)[0] == (1.25, 1.25)
    assert hist._core["TAttMarker"]["fMarkerColor"] == 2


def test_data_binned_over_a_range_take_sum_of_squares_errors_when_asked() -> None:
    """``Binning(4, -2, 2)`` with ``DataError(SumW2)``: the bars are square roots."""
    x, data = generated()
    frame = x.frame(5)
    data.plotOn(frame, Binning=(4, -2.0, 2.0), DataError="SumW2")
    assert flat(points(frame.getObject(0))) == pytest.approx(
        flat(
            [
                (-1.5, 20, 4.472136, 4.472136),
                (-0.5, 26, 5.0990195, 5.0990195),
                (0.5, 60, 7.7459667, 7.7459667),
                (1.5, 26, 5.0990195, 5.0990195),
            ]
        ),
        rel=1e-7,
    )


def test_data_cut_by_a_formula_or_a_range_say_how_many_events_are_drawn(capsys: Any) -> None:
    """``Cut`` and ``CutRange`` name the histogram after them and say what they kept."""
    x, data = generated()
    frame = x.frame(-4, 4, 8)
    data.plotOn(frame, Cut="x>0", DataError=None)
    data.plotOn(frame, CutRange="win", XErrorSize=0, Rescale=2.0)
    out = capsys.readouterr().out
    assert (
        "[#1] INFO:Plotting -- RooTreeData::plotOn: plotting 124 events out of 200 total events"
        in out
    )
    assert "RooTreeData::plotOn: plotting 127 events out of 200 total events" in out
    cut, ranged = frame.getObject(0), frame.getObject(1)
    assert (cut.GetName(), ranged.GetName()) == ("h_gData_Cut[x>0]", "h_gData_CutRange[win]")
    assert points(cut) == [
        (-3.5, 0, 0, 0),
        (-2.5, 0, 0, 0),
        (-1.5, 0, 0, 0),
        (-0.5, 0, 0, 0),
        (0.5, 60, 0, 0),
        (1.5, 26, 0, 0),
        (2.5, 22, 0, 0),
        (3.5, 16, 0, 0),
    ]
    assert flat(points(ranged)[1:5]) == pytest.approx(
        flat(
            [
                (-2.5, 0, 0, 2.2957489),
                (-1.5, 22, 6.5311588, 8.833041),
                (-0.5, 52, 10.13203, 12.328648),
                (0.5, 120, 15.448634, 17.578046),
            ]
        ),
        rel=1e-7,
    )
    assert x_errors(ranged)[0] == (0.0, 0.0)
    assert frame.GetMaximum() == pytest.approx(144.45695, rel=1e-7)


#: Bins ``[-10, -2, 0, 3, 10]``: each point scaled to the nominal one-unit bin, as ROOT scales it.
VARIABLE = [
    (-6, 10, 2.4736256, 3.176916),
    (-1, 115, 16.893952, 19.578722),
    (1.5, 180, 17.293658, 19.040678),
    (6.5, 21.428571, 3.8903732, 4.6700331),
]


def test_data_in_bins_of_different_widths_are_scaled_to_the_nominal_bin() -> None:
    """A ``RooBinning`` given, or named on the variable: a wide bin is scaled down to one unit."""
    x, data = generated()
    edges = RooBinning(-10, 10)
    for boundary in (-2, 0, 3):
        edges.addBoundary(boundary)
    given = x.frame()
    data.plotOn(given, Binning=edges)
    x.setBinning(edges, "coarse")
    named = x.frame()
    data.plotOn(named, Binning="coarse")
    for frame in (given, named):
        hist = frame.getObject(0)
        assert flat(points(hist)) == pytest.approx(flat(VARIABLE), rel=1e-7)
        assert x_errors(hist) == [(4, 4), (1, 1), (1.5, 1.5), (3.5, 3.5)]


def test_data_on_a_variable_of_uneven_bins_are_drawn_in_them() -> None:
    """A frame of a variable whose own binning is uneven draws the data in that binning."""
    x, data = generated()
    edges = RooBinning(-10, 10)
    for boundary in (-2, 0, 3):
        edges.addBoundary(boundary)
    x.setBinning(edges)
    frame = x.frame()
    data.plotOn(frame)
    assert frame.getObject(0).GetN() == 4


def test_new_data_on_a_frame_supersede_the_event_count_curves_are_scaled_to(capsys: Any) -> None:
    """Data drawn again in wider bins say so, as ``RooPlot::updateFitRangeNorm`` does."""
    x, data = generated()
    edges = RooBinning(-10, 10)
    for boundary in (-2, 0, 3):
        edges.addBoundary(boundary)
    x.setBinning(edges, "coarse")
    frame = x.frame()
    data.plotOn(frame, Binning="coarse")
    capsys.readouterr()
    data.plotOn(frame, Invisible=True, DrawOption="B")
    assert capsys.readouterr().out == (
        "[#1] INFO:Plotting -- RooPlot::updateFitRangeNorm: New event count of 40 will supersede "
        "previous event count of 200 for normalization of PDF projections\n"
    )
    assert frame.getDrawOptions("h_gData") == "P"
    assert frame.items[1][1:] == ("B", True)
    data.plotOn(frame, RefreshNorm=False)
    assert frame.getFitRangeNEvt() == 40.0


def weighted(values: list[tuple[float, float]]) -> tuple[RooRealVar, RooDataSet]:
    x = RooRealVar("x", "x", 0, -3, 3)
    x.setBins(3)
    w = RooRealVar("w", "w", 1)
    data = RooDataSet("d", "d", [x, w], WeightVar="w")
    for value, weight in values:
        x.setVal(value)
        data.add(RooArgSet(x), weight)
    return x, data


FRACTIONAL = [(-2.5, 1.5), (-2.2, 0.5), (0.3, 2.25), (1.1, -0.5), (2.9, 3.0)]


def test_weighted_data_take_sum_of_squares_errors_and_say_so(capsys: Any) -> None:
    """Weights not whole numbers: ROOT's automatic choice is ``SumW2``, and it says so."""
    x, data = weighted(FRACTIONAL)
    frame = x.frame()
    data.plotOn(frame)
    assert (
        "[#1] INFO:InputArguments -- RooAbsData::plotOn(d) INFO: dataset has non-integer weights, "
        "auto-selecting SumW2 errors instead of Poisson errors" in capsys.readouterr().out
    )
    assert flat(points(frame.getObject(0))) == pytest.approx(
        flat(
            [(-2, 2, 1.5811388, 1.5811388), (0, 2.25, 2.25, 2.25), (2, 2.5, 3.0413813, 3.0413813)]
        ),
        rel=1e-7,
    )


def test_fractional_counts_with_poisson_errors_interpolate_between_integers() -> None:
    """``DataError(Poisson)`` for 2.25 events: the bars between those of 2 and 3 events."""
    x, data = weighted(FRACTIONAL)
    frame = x.frame()
    data.plotOn(frame)
    data.plotOn(frame, DataError="Poisson")
    assert flat(points(frame.getObject(1))) == pytest.approx(
        flat(
            [
                (-2, 2, 1.2918146, 2.6378596),
                (0, 2.25, 1.3770371, 2.7079412),
                (2, 2.5, 1.4622596, 2.7780227),
            ]
        ),
        rel=1e-7,
    )
    assert frame.GetMaximum() == pytest.approx(5.8455194, rel=1e-7)
    assert frame.GetMinimum() == pytest.approx(-0.84551939, rel=1e-7)


@pytest.mark.xfail(
    strict=True,
    reason="data.py:88-90, hist.py:156: ROOT's warning names the TH1 filled, d_plot__x",
)
def test_fractional_counts_with_poisson_errors_are_warned_of_by_the_histograms_name(
    capsys: Any,
) -> None:
    """ROOT warns of ``RooHist::addBin(d_plot__x)``: the histogram the data were binned in."""
    x, data = weighted(FRACTIONAL)
    data.plotOn(x.frame(), DataError="Poisson")
    assert (
        "[#0] WARNING:Plotting -- RooHist::addBin(d_plot__x) WARNING: non-integer bin entry 2.25 "
        "with Poisson errors, interpolating between Poisson errors of adjacent integer"
        in capsys.readouterr().out
    )


@pytest.mark.xfail(
    strict=True,
    reason="hist.py:28, cmdargs.py ERROR_TYPES: ROOT's ErrorType is Poisson 0, SumW2 1, None 2, "
    "Auto 3, Expected 4",
)
def test_data_error_three_is_roots_automatic_choice() -> None:
    """``RooAbsData::Auto`` is 3 in ROOT: for these weights it chooses sums of squares."""
    x, data = weighted(FRACTIONAL)
    frame = x.frame()
    data.plotOn(frame, DataError=3)
    assert frame.getObject(0).errors()[0][0] == pytest.approx(1.5811388, rel=1e-7)


@pytest.mark.xfail(
    strict=True,
    reason="hist.py:163-185: ROOT drops a negative bin with Poisson errors, saying it cannot",
)
def test_a_negative_count_with_poisson_errors_is_left_out_as_root_does(capsys: Any) -> None:
    """ROOT warns, cannot make an interval for -3 events, and adds no point for that bin."""
    x, data = weighted([(-2.5, 1.0), (-2.2, 2.0), (0.3, -3.0), (2.9, 3.0)])
    frame = x.frame()
    data.plotOn(frame, DataError="Poisson")
    out = capsys.readouterr().out
    assert "WARNING: negative entry set to zero when Poisson error bars are requested" in out
    assert "RooHistError::getPoissonInterval: cannot calculate interval for n = -3" in out
    assert "RooHist::addBin: unable to add bin with -3 events" in out
    assert frame.getObject(0).GetN() == 2


def test_a_frame_finds_names_and_restyles_what_is_on_it() -> None:
    """``findObject``, ``getObject``, ``nameOf``, ``getDrawOptions`` and ``setDrawOptions``."""
    from xrdroot.roofit.plot.hist import RooHist

    x, data = generated()
    frame = x.frame()
    data.plotOn(frame)
    assert frame.findObject("h_gData").GetName() == "h_gData"
    assert frame.findObject("h_gData", RooHist) is frame.getObject(0)
    assert frame.findObject("h_gData", "RooCurve") is None
    assert frame.findObject("nothing") is None
    assert (frame.numItems(), frame.nameOf(0)) == (1, "h_gData")
    assert frame.getDrawOptions("h_gData") == "P"
    assert frame.getDrawOptions("nothing") == ""
    assert frame.setDrawOptions("h_gData", "E") is True
    assert frame.setDrawOptions("nothing", "E") is False
    assert frame.getDrawOptions("h_gData") == "E"
    frame.setInvisible("h_gData")
    assert frame.items[0][2] is True
    frame.setPadFactor(0.2)
    assert frame.getPadFactor() == 0.2
    frame.remove("h_gData")
    assert frame.numItems() == 0


def test_a_frame_keeps_the_maximum_and_minimum_it_is_given() -> None:
    """``SetMaximum`` and ``SetMinimum`` hold until changed; unset, an empty frame's are zero."""
    x2 = RooRealVar("x2", "x2", 0, 0, 1)
    frame = x2.frame()
    assert (frame.GetMaximum(), frame.GetMinimum()) == (0.0, 0.0)
    frame.SetMaximum(3.0)
    frame.SetMinimum(-1.0)
    assert (frame.GetMaximum(), frame.GetMinimum()) == (3.0, -1.0)


def test_a_frames_axes_are_titled_after_the_variable_and_the_data_and_can_be_retitled() -> None:
    """``x (GeV)`` along, ``Events / ( 1 GeV )`` up; a macro may set either, or any attribute."""
    x, data = generated()
    x.setUnit("GeV")
    frame = x.frame()
    assert frame.GetXaxis().GetTitle() == "x (GeV)"
    assert frame.GetYaxis().GetTitle() == ""
    data.plotOn(frame)
    assert frame.GetYaxis().GetTitle() == "Events / ( 1 GeV )"
    frame.GetYaxis().SetTitle("counts")
    frame.SetXTitle("mass")
    assert (frame.GetXaxis().GetTitle(), frame.GetYaxis().GetTitle()) == ("mass", "counts")
    frame.SetYTitle("n")
    assert frame.GetYaxis().GetTitle() == "n"
    axis = frame.GetYaxis()
    axis.SetTitleOffset(1.4)
    assert axis.GetTitleOffset() == pytest.approx(1.4)
    with pytest.raises(AttributeError, match="a RooPlot's axis has no SetNothing here"):
        axis.SetNothing(1)


def test_drawing_a_frame_draws_its_axes_then_what_is_visible_on_it_then_its_axes_again() -> None:
    """``RooPlot::Draw``: ``FUNC``, each visible item with its option, then ``AXISSAME``."""
    from xrdroot.roofit.plot import frame as frames
    from xrdroot.roofit.plot.params import Pave

    x, data = generated()
    frame = x.frame()
    data.plotOn(frame)
    data.plotOn(frame, Invisible=True)
    frame.addObject(Pave(0.1, 0.2, 0.3, 0.4, "NDC"))
    frame.addTH1(Pave(0.1, 0.2, 0.3, 0.4, "NDC"), "", True)
    frame.setDrawOptions("h_gData", "")
    drawn: list[tuple[Any, str]] = []
    frames.set_drawer(lambda obj, option: drawn.append((obj, option)))
    try:
        frame.Draw("same")
    finally:
        frames.set_drawer(lambda obj, option: frames.DRAWN.append((obj, option)))
    assert [option for _, option in drawn] == ["FUNCSAME", "LP", "", "AXISSAME"]
    assert drawn[0][0] is frame.hist
    frame.Draw()
    assert frames.DRAWN[-4][1] == "FUNC"


@pytest.mark.xfail(
    strict=True,
    reason="frame.py:272-282: ROOT prints a frame as 'frame_x_...[x] = (::,::,TPaveText::box)'",
)
def test_a_frame_prints_its_variable_and_its_items_as_root_does(capsys: Any) -> None:
    """ROOT's inline form names the variable, and - slicing its printables - prints ``::``."""
    x, data = generated()
    frame = x.frame()
    data.plotOn(frame)
    capsys.readouterr()
    frame.Print()
    out = capsys.readouterr().out
    assert out.startswith("frame_x_") and out.endswith("[x] = (::)\n")


def test_a_hist_made_by_hand_counts_the_events_of_its_points_in_a_range() -> None:
    """Without the events behind it, a ``RooHist`` counts the heights of its points inside."""
    from xrdroot.roofit.plot.hist import RooHist

    made = RooHist(
        "h", "", [0.5, 1.5, 2.5], [3.0, 4.0, 5.0], [0.5] * 3, [0.5] * 3, [1] * 3, [1] * 3
    )
    assert (made.GetN(), made.ClassName(), made.GetName()) == (3, "RooHist", "h")
    made.SetName("renamed")
    assert made.GetName() == "renamed"
    assert made.events_between(1.0, 3.0) == 9.0
    assert made.fit_range_events() == 0.0
    made.raw_entries = 12.0
    assert made.fit_range_events() == 12.0


def test_a_binned_dataset_is_drawn_in_its_own_bins_as_root_draws_it() -> None:
    """A ``RooDataHist`` of 120 events brings its binning: the points are ROOT's."""
    from xrdroot.roofit.data.datahist import RooDataHist

    x = RooRealVar("x", "x", 0, -10, 10)
    x.setBins(20)
    g1 = RooGaussian("g1", "g1", x, RooRealVar("m1", "m1", 1), RooRealVar("s1", "s1", 1.5))
    generator().SetSeed(4357)
    binned = RooDataHist("dh", "dh", [x], g1.generate([x], 120))
    frame = x.frame()
    binned.plotOn(frame)
    hist = frame.getObject(0)
    assert hist.GetName() == "h_dh"
    counts = [0, 0, 0, 0, 0, 0, 0, 3, 7, 19, 42, 20, 15, 13, 1, 0, 0, 0, 0, 0]
    assert [p[1] for p in points(hist)] == counts
    assert flat(points(hist)[9:11]) == pytest.approx(
        [-0.5, 19, 4.3202194, 5.4351962, 0.5, 42, 6.4548307, 7.5321802], rel=1e-7
    )


def test_the_graphics_layer_may_install_its_own_axes_and_paves() -> None:
    """``set_axis`` and ``set_pave`` are the hooks the pyroot layer puts its classes in by."""
    from xrdroot.roofit.plot import frame as frames
    from xrdroot.roofit.plot import params

    x, _ = generated()
    made: list[Any] = []

    def axis(row: Any, owner: Any) -> str:
        made.append(row)
        return "axis"

    def pave(*args: Any) -> Any:
        made.append(args)
        return params.Pave(*args)

    frames.set_axis(axis)
    params.set_pave(pave)
    try:
        frame = x.frame()
        assert frame.GetXaxis() == "axis"
        RooGaussian("g", "g", x, 0.0, 1.0).paramOn(frame)
    finally:
        frames.set_axis(frames.Axis)
        params.set_pave(params.Pave)
    assert made[0] is frame.axis("x")
    assert made[1][4] == "BRNDC"


def test_a_hist_with_no_nominal_width_leaves_the_frames_bin_width_as_it_was() -> None:
    """The first data to set the scale set the events; a width of zero changes nothing."""
    from xrdroot.roofit.plot.hist import RooHist

    x, _ = generated()
    frame = x.frame()
    made = RooHist("h", "", [0.5], [4.0], [0.5], [0.5], [2.0], [2.0])
    made.entries = 4.0
    frame.add_plotable(made, "P")
    assert (frame.getFitRangeNEvt(), frame.getFitRangeBinW()) == (4.0, 1.0)
    assert frame.getFitRangeNEvt(0.0, 1.0) == 4.0
    frame.addObject(RooHist("other", ""), "P")
    frame.setInvisible("other")
    assert [one[2] for one in frame.items] == [False, True]


def test_a_frame_names_its_class_and_title_as_a_printable() -> None:
    """``RooPrintable``'s class name and title of a frame: ``RooPlot``, ``A RooPlot of "x"``."""
    from xrdroot.roofit.printing import kClassName, kInline, kTitle

    x, _ = generated()
    frame = x.frame(Name="fr")
    assert frame.printStream(kClassName | kTitle, kInline) == 'RooPlot:: "A RooPlot of "x""'
