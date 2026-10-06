"""``ROOT.Experimental.RHist`` and its axes, and the libc++ random numbers ROOT's macros draw."""

from __future__ import annotations

import math

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect
from xrdroot.errors import UnsupportedFeatureError

E = ROOT.Experimental


def test_a_regular_axis_finds_bins_and_its_flow_bins_as_roots_does() -> None:
    axis, bare = E.RRegularAxis(4, (0.0, 2.0)), E.RRegularAxis(4, (0.0, 2.0), False)
    expect((axis.value_index(-1.0), (0, True)), (axis.value_index(2.0), (5, True)),
           (axis.value_index(float("nan")), (5, True)), (axis.value_index(0.6), (2, True)),
           (bare.value_index(-1.0), (0, False)), (bare.value_index(0.6), (1, True)),
           ((axis.GetTotalNBins(), bare.GetTotalNBins()), (6, 4)),
           ((axis.GetLow(), axis.GetHigh(), axis.ComputeHighEdge(1)), (0.0, 2.0, 1.0)),
           ([b.GetIndex() for b in axis.GetNormalRange(1, 3)], [1, 2]),
           (len(axis.GetFullRange()), 6), (len(bare.GetFullRange()), 4),
           (axis.bin_index(E.RBinIndex()), (0, False)), (axis.bin_index(E.RBinIndex(9)),
           (0, False)),
           (bare.bin_index(E.RBinIndex.Overflow()), (5, False)))  # fmt: skip
    for bad in ((0, (0, 1)), (3, (1, 1)), (3, (0, math.inf))):
        with pytest.raises(ValueError, match="needs at least one bin"):
            E.RRegularAxis(*bad)
    with pytest.raises(ValueError, match="not a range"):
        axis.GetNormalRange(3, 1)


def test_a_variable_axis_finds_bins_between_its_edges() -> None:
    axis = E.RVariableBinAxis([1, 10, 100, 1000])
    expect((axis.value_index(5.0), (1, True)), (axis.value_index(500.0), (3, True)),
           (axis.value_index(0.5), (0, True)), (axis.GetBinEdges(), [1.0, 10.0, 100.0, 1000.0]),
           (axis.GetNNormalBins(), 3), (axis.HasFlowBins(), True))  # fmt: skip
    with pytest.raises(ValueError, match="two or more edges"):
        E.RVariableBinAxis([1, 1])


def test_bin_indices_move_compare_and_name_themselves() -> None:
    first, under = E.RBinIndex(0), E.RBinIndex.Underflow()
    expect(((first + 2).GetIndex(), 2), ((first - 1).IsInvalid(), True),
           ((under + 1).IsInvalid(), True), (first < E.RBinIndex(1), True),
           (E.RBinIndex(1) >= first, True), (under >= first, False),
           (len({first, E.RBinIndex(0)}), 1), (repr(under), "RBinIndex(Underflow)"),
           (repr(first), "RBinIndex(0)"), (E.RBinIndex.Overflow().IsOverflow(), True),
           (first.IsNormal(), True), (under.IsUnderflow(), True))  # fmt: skip


def test_a_histogram_fills_its_bins_and_its_statistics() -> None:
    hist = E.RHist["int"](E.RRegularAxis(4, (0.0, 2.0)))
    for x in (0.1, 0.6, 0.6, 5.0, -1.0):
        hist.Fill(x)
    hist.Fill(1.1, E.RWeight(2.7))
    expect((hist.GetBinContent(1), 2), (hist.GetBinContent(E.RBinIndex.Overflow()), 1),
           (hist.GetBinContent([E.RBinIndex.Underflow()]), 1), (hist.GetBinContent(2), 2),
           (hist.GetNEntries(), 6), (hist.GetNDimensions(), 1), (hist.GetTotalNBins(), 6),
           (type(hist).__name__, "RHist<int>"), (hist.GetEngine() is hist, True))  # fmt: skip
    stats = hist.GetStats()
    mean = (0.1 + 0.6 + 0.6 + 5.0 - 1.0 + 2.7 * 1.1) / 7.7
    assert hist.ComputeMean() == pytest.approx(mean, rel=1e-15)
    assert stats.GetDimensionStats(0).fSumWX2 == pytest.approx(0.01 + 0.72 + 25 + 1 + 2.7 * 1.21)
    assert hist.ComputeStdDev() == pytest.approx(math.sqrt(stats.ComputeVariance()), rel=1e-15)
    assert stats.ComputeNEffectiveEntries() == pytest.approx(7.7 ** 2 / (5 + 2.7 ** 2))
    with pytest.raises(ValueError, match="filled with 2 values"):
        hist.Fill(1.0, 2.0)
    with pytest.raises(ValueError, match="There is no bin"):
        hist.GetBinContent(9)


def test_histograms_add_scale_clear_and_set_their_bins() -> None:
    one, two = E.RHist["double"](4, (0.0, 2.0)), E.RHist["double"](4, (0.0, 2.0))
    one.Fill(0.1)
    two.Fill(0.1, E.RWeight(3.0))
    one.Add(two)
    one.Scale(2.0)
    one.SetBinContent(E.RBinIndex(3), 7.5)
    expect((one.GetBinContent(0), 8.0), (one.GetBinContent(3), 7.5), (one.GetNEntries(), 2),
           (one.GetStats().GetSumW(), 8.0), (one.GetStats().GetSumW2(), 40.0))  # fmt: skip
    one.Clear()
    empty = one.GetStats()
    expect((one.GetBinContent(0), 0.0), (one.GetNEntries(), 0),
           (math.isnan(one.ComputeMean()), True), (math.isnan(one.ComputeStdDev()), True),
           (math.isnan(empty.ComputeNEffectiveEntries()), True), (empty.GetNDimensions(), 1))


def test_a_bin_with_error_keeps_its_sum_of_squares() -> None:
    hist = E.RHist[E.RBinWithError](E.RVariableBinAxis([0, 1, 3]), E.RRegularAxis(2, (0, 2)))
    hist.Fill(0.5, 0.5, E.RWeight(2.0))
    hist.Fill(0.5, 0.5)
    engine = E.RHistEngine["double"]([E.RRegularAxis(2, (0, 1), False)])
    engine.Fill(0.25)
    bin = hist.GetBinContent(0, 0)
    hist.SetBinContent(1, 1, E.RBinWithError(4.0, 5.0))
    other = E.RHist[E.RBinWithError](E.RVariableBinAxis([0, 1, 3]), E.RRegularAxis(2, (0, 2)))
    other.Add(hist)
    other.Scale(2.0)
    engine.Clear()
    expect(((bin.fSum, bin.fSum2, float(bin)), (3.0, 5.0, 3.0)), (repr(bin),
           "RBinWithError(3.0, 5.0)"),
           (other.GetBinContent(1, 1).fSum2, 20.0), (engine.GetTotalNBins(), 2),
           (engine.GetBinContent(0), 0.0), (hist.ComputeMean(1), 0.5))  # fmt: skip


def test_a_histogram_converts_to_the_th1_of_its_bins_and_statistics() -> None:
    hist = E.RHist["int"](E.RRegularAxis(4, (0.0, 2.0)))
    for x in (0.1, 0.6, 0.6, 5.0):
        hist.Fill(x)
    h1 = E.Hist.ConvertToTH1I(hist)
    errors = E.RHist[E.RBinWithError](E.RRegularAxis(2, (0.0, 1.0), False))
    errors.Fill(0.2, E.RWeight(3.0))
    h2 = E.Hist.ConvertToTH1D(errors)
    engine = E.RHistEngine["float"](E.RVariableBinAxis([0, 1, 4]))
    engine.Fill(2.0)
    h3 = E.Hist.ConvertToTH1F(engine)
    expect((h1.ClassName(), "TH1I"), (h1.GetBinContent(2), 2.0), (h1.GetBinContent(5), 1.0),
           (h1.GetEntries(), 4.0), (h1.GetMean(), pytest.approx(1.575)),
           (h2.GetBinContent(1), 3.0), (h2.GetBinError(1), 3.0), (h2.GetBinContent(0), 0.0),
           (h3.GetXaxis().GetBinUpEdge(2), 4.0), (h3.GetBinContent(2), 1.0),
           (E.Hist.ConvertToTH1S(hist).ClassName(), "TH1S"),
           (E.Hist.ConvertToTH1C(hist).ClassName(), "TH1C"))  # fmt: skip
    with pytest.raises(ValueError, match="one dimension"):
        E.Hist.ConvertToTH1D(E.RHist["double"](E.RRegularAxis(1, (0, 1)),
               E.RRegularAxis(1, (0, 1))))


def test_fill_contexts_fill_the_histogram_their_filler_shares() -> None:
    hist = E.RHist["int"](4, (0.0, 2.0))
    filler = E.RHistConcurrentFiller(hist)
    context = filler.CreateFillContext()
    context.Fill(0.5)
    context.Flush()
    filler.Flush()
    assert (hist.GetNEntries(), hist.GetBinContent(1)) == (1, 1)


def test_a_frame_books_a_histogram_of_its_columns() -> None:
    frame = ROOT.RDataFrame(4).Define("x", "rdfentry_ * 0.5").Define("w", "2.0")
    one = frame.Hist(4, (0.0, 2.0), "x")
    weighted = frame.Hist(4, (0.0, 2.0), "x", "w")
    axes = frame.Hist([E.RRegularAxis(2, (0.0, 2.0))], ["x"])
    given = frame.Hist(E.RHist["int"](2, (0.0, 2.0)), "x", "w")
    expect((one.IsReady(), False), (one.GetNEntries(), 4), (one.IsReady(), True),
           (weighted.GetStats().GetSumW(), 8.0), (axes.GetBinContent(1), 2.0),
           (given.GetBinContent(0), 4), (E.Hist.ConvertToTH1D(one).GetEntries(), 4.0))
    with pytest.raises(ValueError, match="expected 1, got 2"):
        frame.Hist([E.RRegularAxis(2, (0.0, 2.0))], ["x", "w"])
    with pytest.raises(AttributeError):
        one._missing  # noqa: B018


def test_the_experimental_namespace_has_the_histograms_and_refuses_the_rest() -> None:
    assert E.ML.RDataLoader is not None and repr(E) == "<namespace ROOT::Experimental>"
    with pytest.raises(UnsupportedFeatureError, match="RNTupleImporter is not supported"):
        E.RNTupleImporter  # noqa: B018
    with pytest.raises(AttributeError):
        E.__wrapped__  # noqa: B018


def test_a_fill_past_an_axis_without_flow_bins_changes_no_bin() -> None:
    hist = E.RHist[E.RBinWithError](E.RRegularAxis(2, (0.0, 1.0), False))
    hist.Fill(5.0)
    hist.Fill(0.2)
    engine = E.RHistEngine[E.RBinWithError](E.RRegularAxis(2, (0.0, 1.0), False))
    engine.Fill(0.2)
    hist.Add(engine)
    expect((hist.GetNEntries(), 2), (hist.GetBinContent(0).fSum, 2.0),
           (hist.ComputeNEffectiveEntries(), 2.0))
    hist.Clear()
    assert (hist.GetBinContent(0).fSum2, hist.GetNEntries()) == (0.0, 0)


def test_a_booked_histogram_runs_with_the_frames_other_results() -> None:
    frame = ROOT.RDataFrame(4).Define("x", "rdfentry_ * 0.5")
    hist, count = frame.Hist(4, (0.0, 2.0), "x"), frame.Count()
    assert ROOT.RDF.RunGraphs([hist, count]) == 1
    assert hist.GetNEntries() == 4 and count.GetValue() == 4
