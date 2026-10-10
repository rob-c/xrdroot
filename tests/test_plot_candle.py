"""Candle and violin plots: the option's digits, one candle's numbers, and the layers drawn.

The numbers are ``TCandle::Calculate``'s on a slice: the quantiles the box
and whiskers stand at, the whiskers pulled in to the farthest bins within
1.5 IQR when asked, the mean, the notch, and the points beyond.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot import Histogram
from xrdroot.plot import candle
from xrdroot.plot.candle import SETTINGS, candle_of, parse_option, part
from xrdroot.plot.candleplot import candle_layers
from xrdroot.plot.model import Area, Boxes, Curve, Points
from xrdroot.plot.options import choose
from xrdroot.plot.request import Request


@pytest.fixture(autouse=True)
def _default_settings():
    kept = dict(SETTINGS)
    yield
    SETTINGS.update(kept)


def test_the_option_is_read_as_a_preset_its_digits_or_the_first_preset():
    assert parse_option("CANDLEX2") == 112321 and parse_option("candle") == 112311
    assert parse_option("CANDLEY") == 112311 + candle.HORIZONTAL
    assert parse_option("VIOLINX(112000000)") == 112000000
    assert parse_option("candlex(0)") == 0 and parse_option("candle()") == 0
    assert parse_option("VIOLIN2") == 13300330 and parse_option("candlex9") == 112311
    assert (part(112321, candle.MEDIAN), part(112321, candle.POINTS)) == (2, 1)
    chosen = choose("candley2 scat", "two-dimensional histogram")
    assert chosen.candle == "CANDLEY2" and chosen.has("CANDLE") and not chosen.has("SCAT")
    assert choose("violinx(1)", "two-dimensional histogram").candle == "VIOLINX(1)"


COUNTS = [0, 0, 1, 4, 10, 20, 30, 20, 10, 4, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]


def _slice():
    h = Histogram.book("s", (20, 0.0, 20.0))
    h.fill(np.repeat(np.arange(20) + 0.5, COUNTS))
    return h


def test_a_candle_stands_at_the_quartiles_with_whiskers_to_the_ends_or_within_the_iqr():
    rng = np.random.default_rng(0)
    whole = candle_of(_slice(), 3.0, 1.0, parse_option("candlex3"), rng)
    assert whole is not None and (whole.pos, whole.width, whole.entries) == (3.0, 1.0, 101.0)
    assert whole.box_down < whole.median < whole.box_up
    assert whole.whisker_up == pytest.approx(19.0) and whole.whisker_down == pytest.approx(2.0)
    pulled = candle_of(_slice(), 3.0, 1.0, parse_option("candlex1"), rng)
    assert pulled.whisker_up == 10.5 and pulled.whisker_down == 2.5  # the farthest within 1.5 IQR
    assert pulled.points[1].tolist() == [18.5] and pulled.points[0].tolist() == [3.0]  # beyond
    assert pulled.mean == pytest.approx(whole.mean) and pulled.median_err > 0
    assert candle_of(Histogram.book("e", (4, 0.0, 1.0)), 0.0, 1.0, 1, rng) is None
    SETTINGS["box_range"] = 0.9
    assert candle_of(_slice(), 0.0, 1.0, 1, rng).box_up > whole.box_up


def test_a_slice_whose_quantiles_come_out_of_order_is_dismissed_as_root_dismisses_it(monkeypatch):
    monkeypatch.setattr(Histogram, "quantiles", lambda self, p: [1.0, 3.0, 2.0, 1.0, 4.0])
    assert candle_of(_slice(), 0.0, 1.0, 1, np.random.default_rng(0)) is None


def test_every_entry_is_a_point_at_its_bin_or_scattered_over_it_up_to_roots_most():
    rng = np.random.default_rng(1)
    at_bins = candle_of(_slice(), 3.0, 1.0, parse_option("candlex5"), rng)
    assert len(at_bins.points[0]) == 101 and set(at_bins.points[0].tolist()) == {3.0}
    scattered = candle_of(_slice(), 3.0, 1.0, parse_option("candlex6"), rng)
    assert np.all((scattered.points[0] >= 2.5) & (scattered.points[0] <= 3.5))
    assert np.all((scattered.points[1] >= 2.0) & (scattered.points[1] <= 19.0))
    big = Histogram.book("big", (3, 0.0, 3.0))
    big.fill(np.array([0.5, 1.5, 2.5]), weight=np.array([5000.0, 1.0, 1.0]))
    capped = candle_of(big, 0.0, 1.0, parse_option("candlex5"), rng)
    assert len(capped.points[0]) == candle.MOST_POINTS


def _sin():
    h = Histogram.book("h2", (6, 0.0, 6.0), (30, -2.0, 2.0))
    rng = np.random.default_rng(2)
    xs = np.repeat(np.arange(6) + 0.5, 200)
    h.fill(xs, rng.normal(np.sin(xs), 0.3))
    h._core["fBarWidth"], h._core["fBarOffset"] = 400, 250
    h._core["TAttFill"]["fFillColor"] = 5
    return h


def _layers(option, h=None):
    request = Request(choose(option, "two-dimensional histogram"), {})
    return candle_layers(h if h is not None else _sin(), request)


def _of(layers, kind):
    return [layer for layer in layers if isinstance(layer, kind)]


def test_the_layers_are_what_each_digit_asks_for_in_the_histograms_look():
    kinds = [type(layer) for layer in _layers("candlex2")]
    assert kinds.count(Area) == 6 and Boxes not in kinds  # six notched boxes, filled
    assert kinds.count(Points) >= 6  # a mean circle each, and outliers
    plain = _layers("candlex(1111)")
    box = _of(plain, Boxes)[0]
    assert len(_of(plain, Boxes)) == 6 and len(_of(plain, Points)) == 0
    assert box.x1[0] - box.x0[0] == pytest.approx(0.4) and box.x0[0] == pytest.approx(0.55)
    assert all(isinstance(layer, Curve) for layer in _layers("violinx(10000000)"))  # zero lines
    assert len(_of(_layers("candlex(100)"), Curve)) == 6  # a dashed mean line each
    assert len(_of(_layers("candlex(30)"), Points)) == 6  # a median circle each


def test_horizontal_candles_swap_the_axes():
    sideways = _layers("candley(1)")
    assert all(isinstance(layer, Boxes | Curve) for layer in sideways)
    box = _of(sideways, Boxes)[0]
    assert box.y1[0] - box.y0[0] == pytest.approx(0.4 * 4 / 30)  # of the y bins' width
    assert _layers("candle", Histogram.book("e", (2, 0.0, 1.0), (2, 0.0, 1.0))) == []


def test_scaled_candles_shrink_with_their_entries_and_unscaled_violins_fill_their_width():
    SETTINGS["scaled_candle"] = True
    h = _sin()
    h.fill(np.full(300, 0.5), np.zeros(300))  # the first bin now holds the most
    scaled = _of(_layers("candlex(1)", h), Boxes)
    assert scaled[0].x1[0] - scaled[0].x0[0] > scaled[1].x1[0] - scaled[1].x0[0]
    SETTINGS["scaled_violin"] = False
    violins = _of(_layers("violinx1", h), Area)
    assert len(violins) == 6 and max(violins[1].x) - min(violins[1].x) == pytest.approx(0.4)
    SETTINGS["scaled_violin"] = True
    thinner = _of(_layers("violinx1", h), Area)
    assert max(thinner[1].x) - min(thinner[1].x) < 0.4
