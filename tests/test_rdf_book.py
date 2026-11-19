"""``RDataFrame.Book``: a RooFit helper filled by the frame's event loop.

ROOT books a helper - ``RooDataSetHelper``, ``RooDataHistHelper`` - and the
loop hands it the named columns, matched to the variables by position. The
booked result is what the helper holds once the loop is done, so every
``Book`` of one frame costs one loop between them.
"""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT

#: The columns and the entries they hold, in the order the frame serves them.
XS = (0.5, -1.5, 2.5, 0.25, -3.75, 4.5)
YS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


def _frame():
    """A frame of ``XS`` and ``YS``, defined by callables of no columns, counted per entry."""
    xs, ys = list(XS), list(YS)
    return ROOT.RDataFrame(len(xs)).Define("x", lambda: xs.pop(0)).Define("y", lambda: ys.pop(0))


def _variables():
    """Two variables whose bins hold every one of ``XS`` and ``YS``."""
    x = ROOT.RooRealVar("x", "x", -5.0, 5.0)
    y = ROOT.RooRealVar("y", "y", 0.0, 7.0)
    x.setBins(4)
    y.setBins(7)
    return x, y


def test_a_booked_dataset_holds_every_entry_in_the_order_it_was_read():
    x, y = _variables()
    booked = _frame().Book(
        ROOT.std.move(ROOT.RooDataSetHelper("dataset", "a dataset", ROOT.RooArgSet(x, y))),
        ("x", "y"),
    )
    data = booked.GetValue()
    assert data.numEntries() == len(XS)
    assert data.column("x").tolist() == pytest.approx(XS)
    assert data.column("y").tolist() == pytest.approx(YS)
    assert data.mean(x) == pytest.approx(float(np.mean(XS)))


def test_a_booked_histogram_holds_every_entry_in_its_bin():
    x, y = _variables()
    booked = _frame().Book(
        ROOT.std.move(ROOT.RooDataHistHelper("datahist", "a histogram", ROOT.RooArgSet(x, y))),
        ("x", "y"),
    )
    data = booked.GetValue()
    assert data.numEntries() == 4 * 7  # one bin for each combination of the variables' bins
    assert data.sum() == pytest.approx(float(len(XS)))
    # each entry in the bin its own values fall in, and nowhere else
    bins = data._bin_of({"x": np.asarray(XS), "y": np.asarray(YS)})
    assert len(set(bins.tolist())) == len(XS)
    for at, value in zip(bins, YS, strict=True):
        centre = float(data.get(int(at)).find("y").getVal())
        # the centre of the bin its value falls in, a value on an edge taking the upper bin
        assert abs(centre - value) <= 0.5


def test_two_books_of_one_frame_cost_one_loop_between_them():
    x, y = _variables()
    frame = _frame()
    first = frame.Book(
        ROOT.std.move(ROOT.RooDataSetHelper("dataset", "a dataset", ROOT.RooArgSet(x, y))),
        ("x", "y"),
    )
    second = frame.Book(
        ROOT.std.move(ROOT.RooDataHistHelper("datahist", "a histogram", ROOT.RooArgSet(x, y))),
        ("x", "y"),
    )
    assert ROOT.RDF.RunGraphs([first, second]) == 1
    assert first.GetValue().numEntries() == len(XS)
    assert second.GetValue().sum() == pytest.approx(float(len(XS)))


def test_a_helper_filled_one_entry_at_a_time_holds_the_same_events():
    """``Fill`` is what ROOT's helper is given by a loop that calls it entry by entry."""
    x, y = _variables()
    dataset = ROOT.RooDataSetHelper("dataset", "a dataset", ROOT.RooArgSet(x, y))
    histogram = ROOT.RooDataHistHelper("datahist", "a histogram", ROOT.RooArgSet(x, y))
    for one, two in zip(XS, YS, strict=True):
        dataset.Fill(one, two)
        histogram.Fill(one, two)
    assert dataset.result().numEntries() == len(XS)
    assert dataset.result().column("y").tolist() == pytest.approx(YS)
    assert histogram.result().sum() == pytest.approx(float(len(XS)))


def test_an_event_outside_every_bin_is_left_out_of_the_histogram():
    x, y = _variables()
    histogram = ROOT.RooDataHistHelper("datahist", "a histogram", ROOT.RooArgSet(x, y))
    histogram.fill_columns({"x": np.asarray([0.5, 99.0]), "y": np.asarray([1.0, 2.0])})
    assert histogram.result().sum() == pytest.approx(1.0)


def test_add_arrays_fills_a_histogram_from_columns_and_their_weights():
    x, y = _variables()
    data = ROOT.RooDataHist("datahist", "a histogram", ROOT.RooArgSet(x, y))
    data.add_arrays({"x": np.asarray([0.5]), "y": np.asarray([1.0])}, weights=np.asarray([3.0]))
    assert data.sum() == pytest.approx(3.0)
    assert data.sum(correctForBinSize=False) == pytest.approx(3.0)
