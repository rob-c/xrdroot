"""UHI's indexing of ROOT's histograms: ``h[i]``, ``h[a:b:rebin(n)]``, ``h[..., ::sum]``.

The numbers are ROOT 6.40's own, from its PyROOT on the same histograms.
"""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot.pyroot._pythonization._uhi.indexing import (
    _apply_actions,
    _get_processed_slices,
    _temporarily_disable_add_directory,
)
from xrdroot.pyroot.uhi import loc, overflow, rebin, underflow
from xrdroot.pyroot.uhi import sum as uhi_sum


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def _one_d() -> ROOT.TH1D:
    """UHI's own 1-D test histogram: ten bins of 0, 2, ... 18, the underflow 3, overflow 1."""
    h = ROOT.TH1D("h1", "h1", 10, 0, 1)
    h[...] = np.array([3.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 1.0])
    return h


def _two_d() -> ROOT.TH2D:
    x, y = np.mgrid[0:2, 0:5]
    h = ROOT.TH2D("h2", "h2", 2, 0, 2, 5, 0, 5)
    h[...] = np.pad(x + 2 * y, 1, mode="constant")
    return h


def _three_d() -> ROOT.TH3D:
    x, y, z = np.mgrid[0:2, 0:5, 0:10]
    h = ROOT.TH3D("h3", "h3", 2, 0, 2, 5, 0, 5, 10, 0, 10)
    h[...] = np.pad(x + 2 * y + 3 * z, 1, mode="constant") + 1
    return h


def test_a_bin_is_found_by_number_coordinate_flow_tag_or_len():
    h = _one_d()
    expect(
        (h[0], 0.0), (h[9], 18.0), (h[-1], 18.0), (h[-10], 0.0),
        (h[loc(0.15)], 2.0), (h[loc(0.55) + 2], 14.0), (h[loc(0.55) - 2], 6.0),
        (h[loc(-1)], 3.0), (h[loc(2)], 1.0),
        (h[underflow], 3.0), (h[overflow], 1.0), (h[len], 1.0),
        (h[np.int64(2)], 4.0),
    )  # fmt: skip


def test_a_bin_past_either_end_is_an_index_error():
    h = _one_d()
    for index in (10, -11):
        with pytest.raises(IndexError, match="out of range"):
            h[index]
    with pytest.raises(IndexError, match="Slice indices"):
        h[5:2]
    with pytest.raises(IndexError, match="Expected 1 indices, got 2"):
        h[1, 2]
    with pytest.raises(IndexError, match="Only one ellipsis"):
        _two_d()[..., ...]


def test_an_index_of_another_kind_is_a_type_error():
    with pytest.raises(TypeError, match="Unsupported index type: str"):
        _one_d()["a"]
    with pytest.raises(TypeError, match="Unsupported index type: str"):
        _get_processed_slices(_one_d(), ["a"])
    with pytest.raises(IndexError, match="Expected 1 indices, got 2"):
        _get_processed_slices(_one_d(), [1, 2])
    with pytest.raises(TypeError, match="Expected an integer"):
        loc(0.5) + 1.5
    with pytest.raises(TypeError, match="Expected an integer"):
        loc(0.5) - 1.5


def test_a_tag_says_what_it_is():
    expect((repr(loc(0.5)), "loc(0.5)"), (repr(loc(0.5) + 2), "loc(0.5) + 2"),
           (repr(rebin(3)), "rebin(3)"))  # fmt: skip


def test_a_slice_gathers_what_it_leaves_out_into_the_flow():
    h = _one_d()
    cut = h[2:4]
    expect(
        (list(cut), [5.0, 4.0, 6.0, 79.0]),
        (cut.GetEntries(), 94.0),
        (cut.GetName(), "h1"),
        (cut.GetXaxis().GetXmin(), pytest.approx(0.2)),
        (list(h[5:]), [23.0, 10.0, 12.0, 14.0, 16.0, 18.0, 1.0]),
        (list(h[: loc(0.5)]), [3.0, 0.0, 2.0, 4.0, 6.0, 8.0, 71.0]),
        (list(h[{0: slice(2, 4)}]), [5.0, 4.0, 6.0, 79.0]),
    )


def test_a_slice_of_more_axes_gathers_its_leftovers_into_the_first_and_last_bins():
    """As ``ROOT::Internal::Slice`` does: everything before the cut into global bin 0,
    everything after it into the last - not each axis's own flow."""
    h = _two_d()
    h.SetBinContent(0, 3, 7.0)
    h.SetBinContent(3, 0, 5.0)
    cut = h[0:1, 1:3]
    three = _three_d()[0:1, 1:3, 2:5]
    expect(
        (cut.values(flow=True).tolist(),
         [[13.0, 0.0, 0.0, 0.0], [0.0, 2.0, 4.0, 0.0], [0.0, 0.0, 0.0, 38.0]]),
        (cut.GetEntries(), 57.0),
        (three.values(flow=True).sum(), 1932.0),
        (three.GetBinContent(0), 231.0),
        (three.GetBinContent(three.GetNcells() - 1), 1623.0),
    )  # fmt: skip


def test_a_sum_step_integrates_or_projects():
    h, h2, h3 = _one_d(), _two_d(), _three_d()
    expect(
        (h[::sum], 94.0), (h[0:len:sum], 90.0), (h[2:5:sum], 18.0), (h[:4:sum], 15.0),
        (h[4::sum], 79.0), (h[uhi_sum], 94.0), (h[(slice(None, 4, sum),)], 15.0),
        (h2[::sum, ::sum], 45.0), (h2[1, ::sum], 25.0),
        (list(h2[:2:sum, 1:3]), [1.0, 5.0, 9.0, 30.0]),
        (list(h2[..., ::sum]), [0.0, 20.0, 25.0, 0.0]),
        (h2[:, uhi_sum].GetNbinsX(), 2),
        (h3[::sum, ::sum, ::sum], 1900.0),
        (list(h3[::sum, :2:sum, 1:3]), [109.0, 22.0, 34.0, 1894.0]),
        (h3[:, :, ::sum].GetDimension(), 2),
        (h3[::sum, :, :].GetNbinsX(), 5),
    )  # fmt: skip
    with pytest.raises(ValueError, match="Invalid axis"):
        ROOT.uhi.sum(h2, (2,))


def test_a_rebin_step_merges_bins():
    h = _one_d()
    merged = h[:: rebin(2)]
    three = _three_d()[:: rebin(2), :: rebin(2), :: rebin(2)]
    expect(
        (list(merged), [3.0, 2.0, 10.0, 18.0, 26.0, 34.0, 1.0]),
        (list(h[rebin(2)]), [3.0, 2.0, 10.0, 18.0, 26.0, 34.0, 1.0]),
        (list(h[: loc(0.55) : rebin(2)]), [3.0, 2.0, 10.0, 79.0]),
        (list(_two_d()[:: rebin(2), 1:3:uhi_sum]), [1.0, 14.0, 30.0]),
        (list(_two_d()[:: rebin(2), 1:3:sum]), [1.0, 14.0, 30.0]),
        (three.values().ravel()[:6].tolist(), [32.0, 80.0, 128.0, 176.0, 224.0, 64.0]),
    )


def test_a_step_that_is_neither_rebin_nor_sum_is_refused():
    h = _one_d()
    with pytest.raises(ValueError, match="Unsupported action"):
        h[::2]
    with pytest.raises(ValueError, match="Unsupported action"):
        _apply_actions(h, ["x"], [slice(0, 1)], (slice(None),), h)


def test_setting_takes_a_bin_a_scalar_or_an_array():
    h = _one_d()
    h[0] = 42
    h[loc(0.15)] = 43
    h[underflow] = 7
    expect((h.GetBinContent(1), 42.0), (h.GetBinContent(2), 43.0), (h.GetBinContent(0), 7.0))
    h[1:3] = 5
    expect((list(h)[1:5], [42.0, 5.0, 5.0, 6.0]))
    h[:3] = [41, 42, 43, 44]
    expect((list(h)[:5], [41.0, 42.0, 43.0, 44.0, 6.0]))
    h[...] = np.arange(10)
    expect((next(iter(h)), 41.0), (list(h)[-1], 1.0), (h.GetEntries(), 45 + 42.0))
    two = _two_d()
    two[0:2, 0:2] = np.array([[42], [3]])
    two[{0: 1, 1: slice(2, 4)}] = [8, 9]
    expect(
        (two[0, 1], 42.0), (two[1, 0], 3.0), (two[1, 2], 8.0), (two[1, 3], 9.0),
        (two[underflow, 0], 0.0),
    )  # fmt: skip


def test_setting_too_many_or_too_few_values_is_a_value_error():
    h = _one_d()
    for values in (range(9), range(11), range(13)):
        with pytest.raises(ValueError, match="bin values"):
            h[:] = values
    with pytest.raises(ValueError, match="Expected 3 bin values, got 4"):
        h[1:4] = range(4)


def test_iterating_gives_every_bin_with_the_flow():
    h = ROOT.TH2I("i2", "", 2, 0, 2, 1, 0, 1)
    h[...] = np.arange(12).reshape(4, 3)
    expect((list(h), [float(v) for v in range(12)]))


def test_slicing_keeps_a_profile_s_entries_with_its_bins():
    p = ROOT.TProfile("p", "", 4, 0, 4)
    for x, y in ((0.5, 2.0), (1.5, 4.0), (1.5, 6.0), (3.5, 1.0)):
        p.Fill(x, y)
    cut = p[1:3]
    expect((cut.GetBinContent(1), 5.0), (cut.GetBinEntries(1), 2.0), (cut.GetNbinsX(), 2))
    # ROOT's own slice keeps its profile's entries unsliced: 10.0 and 1.0 here.


def test_a_histogram_made_inside_the_switch_goes_into_no_directory():
    with _temporarily_disable_add_directory():
        assert not ROOT.TH1.AddDirectoryStatus()
        ROOT.TH1D("kept_out", "", 1, 0, 1)
    assert ROOT.TH1.AddDirectoryStatus()
    assert ROOT.gDirectory.FindObject("kept_out") is None
