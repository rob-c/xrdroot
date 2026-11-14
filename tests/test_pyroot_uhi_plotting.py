"""UHI's plotting protocol on ROOT's histograms: ``kind``, ``axes``, ``values()`` and the rest."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot.pyroot._pythonization._uhi.plotting import (
    Kind,
    PlottableAxisContinuous,
    _get_sum_of_weights_squared,
    _shape,
)


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_a_histogram_is_a_plottable_histogram():
    plottable = pytest.importorskip("uhi.typing.plottable")
    for h in (ROOT.TH1D("a", "", 2, 0, 1), ROOT.TProfile2D("b", "", 2, 0, 1, 2, 0, 1)):
        assert isinstance(h, plottable.PlottableHistogram)


def test_values_are_the_bins_themselves_read_only_unless_asked():
    h = ROOT.TH2F("v", "", 2, 0, 2, 3, 0, 3)
    h.Fill(0.5, 2.5, 4.0)
    values = h.values()
    with pytest.raises(ValueError, match="read-only"):
        values[0, 0] = 1
    h.values(writable=True)[1, 0] = 7.0
    expect(
        (values.shape, (2, 3)), (values[0, 2], 4.0), (h.GetBinContent(2, 1), 7.0),
        (h.values(flow=True).shape, (4, 5)), (_shape(h), (4, 5)), (_shape(h, flow=False), (2, 3)),
    )  # fmt: skip


def test_bytes_and_profiles_give_copies_and_refuse_to_be_written():
    c = ROOT.TH1C("c", "", 2, 0, 2)
    c.Fill(0.5)
    p = ROOT.TProfile("p", "", 2, 0, 2)
    p.Fill(0.5, 2.0)
    p.Fill(0.5, 4.0)
    for h in (c, p):
        with pytest.raises(TypeError, match=r"values\(writable=True\) is not supported"):
            h.values(writable=True)
    expect(
        (c.values().tolist(), [1.0, 0.0]),
        (p.values(flow=True).tolist(), [0.0, 3.0, 0.0, 0.0]),
        (p.counts().tolist(), [2.0, 0.0]),
        (p.variances().tolist(), [pytest.approx(0.5), 0.0]),
    )


def test_kind_is_count_but_for_a_profile():
    expect(
        (ROOT.TH1D("k1", "", 1, 0, 1).kind, "COUNT"),
        (ROOT.TProfile("k2", "", 1, 0, 1).kind, Kind.MEAN),
        (ROOT.TProfile3D("k3", "", 1, 0, 1, 1, 0, 1, 1, 0, 1).kind, "MEAN"),
    )


def test_variances_and_counts_follow_the_weights():
    h = ROOT.TH1D("w", "", 3, 0, 3)
    h.Fill(0.5)
    expect((h.variances().tolist(), [1.0, 0.0, 0.0]), (h.counts().tolist(), [1.0, 0.0, 0.0]))
    h.Fill(0.5, 2.0)
    expect(
        (h.variances().tolist(), [5.0, 0.0, 0.0]),
        (h.counts().tolist(), [pytest.approx(9 / 5), 0.0, 0.0]),
        (_get_sum_of_weights_squared(h, flow=True).tolist(), [0.0, 5.0, 0.0, 0.0, 0.0]),
    )


def test_an_axis_is_its_bins_edges_or_its_labels():
    h = ROOT.TH2D("ax", "", 2, 0, 1, 2, 0, 4)
    h.GetYaxis().SetBinLabel(1, "a")
    h.GetYaxis().SetBinLabel(2, "b")
    x, y = h.axes
    expect(
        (len(x), 2), (x[1], (0.5, 1.0)), (list(x), [(0.0, 0.5), (0.5, 1.0)]),
        (x.traits.discrete, False), (x.traits.circular, False),
        (x.traits.underflow, True), (x.traits.overflow, True),
        (list(y), ["a", "b"]), (y.traits.discrete, True),
        (x == ROOT.TH1D("ax2", "", 2, 0, 1).axes[0], True), (x == y, False), (x == "x", False),
        (isinstance(x, PlottableAxisContinuous), True),
    )  # fmt: skip


def test_plotting_libraries_draw_it():
    mplhep = pytest.importorskip("mplhep")
    pytest.importorskip("matplotlib").use("Agg")
    h = ROOT.TH1D("plotted", "", 4, 0, 4)
    h[...] = np.arange(4)
    assert mplhep.histplot(h)
