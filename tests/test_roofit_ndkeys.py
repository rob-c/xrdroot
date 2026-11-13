"""``RooNDKeysPdf``: the adaptive kernel estimate in more than one dimension."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT


@pytest.fixture
def cloud():
    rng = np.random.default_rng(11)
    x = ROOT.RooRealVar("x", "x", -4.0, 4.0)
    y = ROOT.RooRealVar("y", "y", -4.0, 4.0)
    data = ROOT.RooDataSet("d", "d", {x, y})
    data.add_columns({"x": rng.normal(0, 1, 200), "y": rng.normal(0, 1, 200)})
    return x, y, data


def test_the_estimate_is_largest_where_the_events_are(cloud):
    x, y, data = cloud
    estimate = ROOT.RooNDKeysPdf("k", "k", [x, y], data)
    x.setVal(0.0)
    y.setVal(0.0)
    centre = estimate.getVal()
    x.setVal(3.9)
    y.setVal(3.9)
    edge = estimate.getVal()
    assert centre > 0.0
    assert edge < centre / 100.0
    assert [one.GetName() for one in estimate.observables()] == ["x", "y"]
    assert "2 dimension" in repr(estimate)


def test_every_kernel_is_as_wide_as_the_data_is_spread(cloud):
    x, y, data = cloud
    estimate = ROOT.RooNDKeysPdf("k", "k", [x, y], data)
    width = np.asarray(estimate._widths)
    assert width.shape == (2,)
    assert np.all(width > 0.0) and np.all(width < 2.0)


def test_the_adaptive_option_widens_each_event_for_itself(cloud):
    x, y, data = cloud
    estimate = ROOT.RooNDKeysPdf("k", "k", [x, y], data, "a")
    widths = np.asarray(estimate._widths)
    assert widths.shape == (len(estimate._points), 2)
    assert np.all(widths > 0.0)
    # the factor stays within the documented half-to-two of the fixed widths
    fixed = np.asarray(ROOT.RooNDKeysPdf("k2", "k2", [x, y], data)._widths)
    assert np.all(widths >= fixed * 0.5 - 1e-12)
    assert np.all(widths <= fixed * 2.0 + 1e-12)


def test_mirroring_adds_a_reflected_copy_of_every_event(cloud):
    x, y, data = cloud
    plain = ROOT.RooNDKeysPdf("k", "k", [x, y], data)
    mirrored = ROOT.RooNDKeysPdf("km", "km", [x, y], data, "m")
    assert len(mirrored._points) == 5 * len(plain._points)   # itself and four reflections
    x.setVal(3.9)
    y.setVal(0.0)
    # mirroring is what keeps an estimate from falling away at the edge of the range
    assert mirrored.getVal() > plain.getVal()


def test_rho_scales_every_width(cloud):
    x, y, data = cloud
    narrow = ROOT.RooNDKeysPdf("k", "k", [x, y], data, "", 0.5)
    wide = ROOT.RooNDKeysPdf("kw", "kw", [x, y], data, "", 2.0)
    assert np.all(np.asarray(wide._widths) == pytest.approx(4.0 * np.asarray(narrow._widths)))
    # a kernel set may be given one width per dimension or one per event and dimension
    points = np.asarray(narrow._points)
    per_dimension = narrow._kernels(points[:3], points, narrow._widths)
    per_event = narrow._kernels(points[:3], points, np.broadcast_to(narrow._widths, points.shape))
    assert per_dimension == pytest.approx(per_event)


def test_an_estimate_with_no_observables_says_so(cloud):
    _, _, data = cloud
    with pytest.raises(ValueError, match="at least one observable"):
        ROOT.RooNDKeysPdf("k", "k", [], data)
