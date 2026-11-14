"""ROOT 7's histograms: their statistics, their bin contents, slices and conversions, the
fillers, and the namespaces they are reached through."""

from __future__ import annotations

import math
import threading

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect
from xrdroot.errors import UnsupportedFeatureError

E = ROOT.Experimental
I = E.Internal


def test_statistics_disable_a_dimension_and_refuse_to_be_read_once_tainted() -> None:
    stats = E.RHistStats(2)
    stats.DisableDimension(1)
    stats.Fill(1, "b")
    stats.Fill((3.0, "c"), E.RWeight(2.0))
    expect((stats.GetNEntries(), 2), (stats.GetSumW(), 3.0), (stats.IsEnabled(1), False),
           (stats.GetDimensionStats().fSumWX, 7.0), (stats.ComputeVariance(), pytest.approx(8 / 9)),
           (stats.ComputeStdDev(), pytest.approx(math.sqrt(8 / 9))), (stats.GetNDimensions(), 2))
    for call, why in ((lambda: stats.GetDimensionStats(1), "dimension is disabled"),
                      (lambda: stats.Fill("a", 1), "invalid type of argument in RHistStats"),
                      (lambda: stats.Fill(1), "invalid number of arguments to Fill"),
                      (lambda: E.RHistStats(0), "nDimensions must be > 0"),
                      (lambda: stats.Add(E.RHistStats(1)), "number of dimensions not identical"),
                      (lambda: stats.AddAtomic(E.RHistStats(2)), "the same dimensions must be "
                       "enabled to combine statistics with AddAtomic")):
        with pytest.raises(ValueError, match=why):
            call()
    with pytest.raises(IndexError):
        stats.IsEnabled(2)
    stats.Taint()
    with pytest.raises(RuntimeError, match="statistics are tainted"):
        stats.ComputeNEffectiveEntries()
    with pytest.raises(RuntimeError, match="unable to store RHistStats"):
        stats.Streamer()


def _moments_of_one_two_two_seven() -> tuple[float, float]:
    """The skewness and excess kurtosis of 1, 2, 2 and 7, from the central moments."""
    spread = np.array([1.0, 2.0, 2.0, 7.0]) - 3.0
    variance = float(np.mean(spread ** 2))
    skew = float(np.mean(spread ** 3)) / variance ** 1.5
    return skew, float(np.mean(spread ** 4)) / variance ** 2 - 3


def test_skewness_and_kurtosis_are_roots_standardized_moments() -> None:
    stats = E.RHistStats(1)
    for x in (1.0, 2.0, 2.0, 7.0):
        stats.Fill(x)
    skew, kurt = _moments_of_one_two_two_seven()
    assert stats.ComputeSkewness() == pytest.approx(skew)
    assert stats.ComputeKurtosis() == pytest.approx(kurt)


def test_moments_with_no_spread_or_no_entries_are_nan() -> None:
    flat, empty = E.RHistStats(1), E.RHistStats(1)
    flat.Fill(3.0)
    flat.Fill(3.0)
    expect((math.isnan(flat.ComputeSkewness()), True),
           (math.isnan(empty.ComputeKurtosis()), True), (math.isnan(empty.ComputeVariance()), True),
           (math.isnan(empty.ComputeStdDev()), True))


def test_statistics_scale_skip_a_disabled_dimension_and_round_to_nan() -> None:
    stats = E.RHistStats(1)
    for x in (1.0, 2.0, 2.0, 7.0):
        stats.Fill(x)
    stats.Scale(2.0)
    disabled = E.RHistStats(2)
    disabled.DisableDimension(0)
    disabled.Fill("a", 2.0)
    disabled.Scale(3.0)
    disabled.GetDimensionStats(1).AddAtomic(disabled.GetDimensionStats(1))
    expect((disabled.GetSumW(), 3.0), (disabled.GetDimensionStats(1).fSumWX, 12.0))
    rounded = E.RHistStats(1)
    rounded.fSumW, rounded.GetDimensionStats().fSumWX2 = 3.0, 0.0
    rounded.GetDimensionStats().fSumWX, rounded.GetDimensionStats().fSumWX3 = 1e-9, 1.0
    expect((stats.GetSumW2(), 16.0), (rounded.ComputeStdDev(), pytest.approx(math.nan,
           nan_ok=True)), (math.isnan(rounded.ComputeSkewness()), True))


def test_bin_contents_count_as_their_cpp_types_do() -> None:
    error = E.RBinWithError(1.0, 2.0)
    error.AtomicAdd(1.5)
    error.AtomicInc()
    error += E.RBinWithError(1.0, 1.0)
    single = E.RHistEngine["float"](1, (0, 1))
    single.Fill(0.5, E.RWeight(0.1))
    whole = E.RHistEngine["short"](1, (0, 1))
    whole.SetBinContent(0, 2.9)
    expect(((error.fSum, error.fSum2), (4.5, 6.25)), (single.GetBinContent(0),
           float(__import__("numpy").float32(0.1))), (whole.GetBinContent(0), 2),
           (E.RHistEngine["ROOT::Experimental::RBinWithError"] is E.RHistEngine[E.RBinWithError],
           True), (single.SupportsWeightedFilling, True), (whole.SupportsWeightedFilling, False))
    with pytest.raises(UnsupportedFeatureError, match="bins of type std::string is not supported"):
        E.RHist["std::string"]
    with pytest.raises(ValueError, match="It is not possible to convert to RBinWithError"):
        E.RHistEngine["double"](1, (0, 1)).Convert[E.RBinWithError]()


def test_an_engine_clones_converts_slices_and_refuses_what_root_refuses() -> None:
    engine = E.RHistEngine[E.RBinWithError](E.RRegularAxis(4, (0, 4)),
                                            E.RCategoricalAxis(["a", "b"]))  # fmt: skip
    for x, name in ((0.5, "a"), (1.5, "b"), (2.5, "a"), (9.0, "z")):
        engine.Fill(x, name, E.RWeight(2.0))
    engine.FillAtomic((0.5, "a"))
    clone = engine.Clone()
    engine.Scale(0.5)
    summed = engine.Slice(E.RSliceSpec.ROperationRebin(2), E.RSliceSpec.ROperationSum())
    as_double = engine.Convert["double"]()
    expect((clone.GetBinContent(0, 0).fSum, 3.0), (engine.GetBinContent(0, 0).fSum, 1.5),
           (summed.GetBinContent(0).fSum2, 2.25), (summed.GetBinContent(1).fSum, 1.0),
           (as_double.GetBinContent([E.RBinIndex.Overflow(), E.RBinIndex.Overflow()]), 1.0),
           (len(list(engine.GetFullMultiDimRange())), 18), (engine.GetAxes().size(), 2))
    for call, why in ((lambda: engine.Slice(E.RSliceSpec()), "invalid number of specifications"),
                      (lambda: engine.Slice([E.RSliceSpec.ROperationSum()] * 2),
                       "summing across all dimensions is not supported"),
                      (lambda: engine.SetBinContent(0, E.RBinWithError()),
                       "invalid number of indices passed to SetBinContent"),
                      (lambda: engine.AddAtomic(E.RHistEngine[E.RBinWithError](4, (0, 4))),
                       "axes configurations not identical in AddAtomic")):
        with pytest.raises(ValueError, match=why):
            call()
    with pytest.raises(RuntimeError, match="unable to store RHistEngine"):
        engine.Streamer()


def test_a_histogram_slice_keeps_its_statistics_unless_bins_are_dropped() -> None:
    hist = E.RHist["int"](E.RRegularAxis(4, (0, 4)), E.RRegularAxis(2, (0, 2)))
    for x, y in ((0.5, 0.5), (1.5, 1.5), (3.5, 0.5)):
        hist.Fill(x, y)
    kept = hist.Slice(E.RSliceSpec(), E.RSliceSpec.ROperationSum())
    dropped = hist.Slice(E.RSliceSpec(), E.RSliceSpec(I.CreateBinIndexRange(0, 1, 0),
                                                      E.RSliceSpec.ROperationSum()))  # fmt: skip
    hist.SetBinContent(0, 0, 5)
    tainted = hist.Slice([hist.GetAxes()[0].GetNormalRange(1, 3), E.RSliceSpec()])
    expect((kept.GetNEntries(), 3), (kept.ComputeMean(), pytest.approx(5.5 / 3)),
           (kept.GetNDimensions(), 1), (dropped.GetStats().IsTainted(), True),
           (dropped.GetBinContent(3), 1), (tainted.GetStats().IsTainted(), True),
           (tainted.GetBinContent(E.RBinIndex.Underflow(), 0), 5),
           (hist.GetFullMultiDimRange() == E.RBinIndexMultiDimRange(), False))
    with pytest.raises(RuntimeError, match="unable to store RHist"):
        hist.Streamer()


def test_a_histogram_clones_converts_adds_atomically_and_moves() -> None:
    hist = E.RHist["int"](E.RCategoricalAxis(["a", "b"]))
    hist.Fill("a")
    clone, converted = hist.Clone(), hist.Convert["float"]()
    clone.AddAtomic(hist)
    moved = E.RHist["int"](clone)
    expect((clone.GetNEntries(), 2), (hist.GetNEntries(), 1), (converted.GetBinContent(0), 1.0),
           (type(converted).__name__, "RHist<float>"), (moved.GetBinContent(0), 2),
           (hist.GetStats().IsEnabled(0), False), (hist.SupportsWeightedFilling, False),
           (moved.GetTotalNBins(), 3), (hist.ComputeNEffectiveEntries(), 1.0))


def test_the_auto_axis_filler_makes_its_axis_from_what_it_has_kept() -> None:
    filler = E.RHistAutoAxisFiller["double"](4, 3, 0.25)
    filler.Fill(0.0, E.RWeight(2.0))
    filler.Fill(2.0)
    filler.Flush()
    filler.Fill(math.nan)
    full = E.RHistAutoAxisFiller["int"](2, 2)
    full.Fill(1.0)
    full.Fill(3.0)  # the second fill fills the buffer, which flushes it
    later = full.GetHist()
    full.Fill(2.0)
    axis = filler.GetHist().GetAxes()[0].GetRegularAxis()
    expect(((axis.GetLow(), axis.GetHigh()), (-0.5, 2.5)), (filler.GetHist().GetNEntries(), 3),
           (filler.GetHist().GetBinContent(0), 2.0), (later.GetNEntries(), 3),
           ((filler.GetNNormalBins(), filler.GetMaxBufferSize(), filler.GetMarginFraction()),
            (4, 3, 0.25)))  # fmt: skip
    for args, why in (((0,), "nNormalBins must be > 0"), ((1, 0), "maxBufferSize must be > 0"),
                      ((1, 1, 0), "marginFraction must be > 0")):
        with pytest.raises(ValueError, match=why):
            E.RHistAutoAxisFiller["int"](*args)
    for values, why in (((), "buffer is empty"), ((math.inf,), "could not determine axis"),
                        ((1.0, 1.0), "axis interval is empty")):
        empty = E.RHistAutoAxisFiller["int"](2)
        for value in values:
            empty.Fill(value)
        with pytest.raises(RuntimeError, match=why):
            empty.Flush()


def test_threads_fill_one_histogram_through_their_own_contexts() -> None:
    hist = E.RHist["int"](E.RRegularAxis(1, (0, 1)))
    filler = E.RHistConcurrentFiller["int"](hist)

    def fill() -> None:
        context = E.RHistFillContext["int"].__class_getitem__("int")(hist)
        for _ in range(200):
            context.Fill(0.5)
        context.Flush()

    threads = [threading.Thread(target=fill) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    expect((hist.GetNEntries(), 800), (hist.GetBinContent(0), 800), (filler.GetHist() is hist,
           True))  # fmt: skip
    with pytest.raises(ValueError, match="hist must not be nullptr"):
        E.RHistConcurrentFiller(None)


def test_the_atomic_additions_add_into_a_cell_or_a_bin() -> None:
    class Cell:
        value = 1

    cell, bin = Cell(), E.RBinWithError()
    I.AtomicAdd(cell, 2)
    I.AtomicInc(cell)
    I.AtomicAdd(bin, 3.0)
    I.AtomicInc(bin)
    expect((cell.value, 4), ((bin.fSum, bin.fSum2), (4.0, 10.0)))


def test_a_histogram_converts_to_a_th1_with_its_labels_and_without_flow_bins() -> None:
    names = E.RHist["int"](E.RCategoricalAxis(["a", "b"]))
    names.Fill("b")
    names.Fill("z")
    regular = E.RHistEngine["long"](E.RRegularAxis(2, (0, 2), False))
    regular.Fill(1.5)
    tainted = E.RHist["double"](2, (0, 2))
    tainted.Fill(0.5)
    tainted.SetBinContent(1, 4.0)
    th1, th1l, th1d = (E.Hist.ConvertToTH1I(names), E.Hist.ConvertToTH1L(regular),
                       E.Hist.ConvertToTH1D(tainted))  # fmt: skip
    axis = ROOT.TAxis()
    E.Hist.Internal.ConvertAxis(axis, E.RVariableBinAxis([0, 1, 3]))
    expect((th1.GetXaxis().GetBinLabel(2), "b"), (th1.GetBinContent(2), 1.0),
           (th1.GetBinContent(3), 1.0), (th1.GetStats()[:4], [2.0, 2.0, 0.0, 0.0]),
           (th1l.GetBinContent(2), 1.0), (th1l.GetBinContent(3), 0.0), (th1l.GetEntries(), 0.0),
           (th1d.GetEntries(), 0.0), (th1d.GetBinContent(2), 4.0), (axis.GetNbins(), 2),
           (axis.GetXmax(), 3.0), (E.Hist.ConvertToTH1I.__name__, "ConvertToTH1I"),
           (repr(E.Hist), "<namespace ROOT::Experimental::Hist>"),
           (E.RCanvas is ROOT.rcanvas.RCanvas, True))
