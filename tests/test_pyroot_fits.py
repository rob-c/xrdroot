"""``TFitResult``, ``TFitResultPtr``, ``TEfficiency``, and how xrdroot objects are wrapped."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot import Efficiency, Histogram
from xrdroot.pyroot.core import fits, treelinks, wrapping


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def fitted():
    ROOT.gRandom.SetSeed(7)
    h = ROOT.TH1D("h", "", 40, -4, 4)
    h.FillRandom("gaus", 4000)
    return h, h.Fit("gaus", "QS")


def test_a_fit_result_answers_by_roots_accessors(capsys):
    _h, r = fitted()
    expect(
        (int(r), 0),
        (r, 0),
        (r.__index__(), 0),
        (bool(bool(r)), True),
        (hash(r), hash(0)),
        (bool(r != 1), True),
        (bool(r != "x"), True),
        (bool("status 0" in repr(r)), True),
        (bool(r.Get() is not None), True),
        (bool(r.Parameter(1) == r.Parameters()[1] == r.GetParams()[1]), True),
        (r.ParError(2), r.Error(2)),
        (list(r.Errors()), list(r.GetErrors())),
        (bool(r.LowerError(0) == r.UpperError(0) == r.ParError(0)), True),
        (r.ParName(2), "Sigma"),
        (r.Index("Mean"), 1),
        (r.Index("zz"), -1),
        (bool(r.NPar() == r.NTotalParameters() == 3), True),
        (r.NFreeParameters(), 3),
        (bool(not r.IsParameterFixed(0)), True),
        (bool(r.IsParameterBound(2)), True),
        (r.Ndf(), r.Ndf()),
        (bool(r.Chi2() > 0), True),
        (bool(0 <= r.Prob() <= 1), True),
        (bool(r.MinFcnValue() > 0), True),
        (bool(r.Edm() >= 0), True),
        (bool(r.NCalls() > 0), True),
        (r.Status(), 0),
        (bool(r.IsValid()), True),
        (bool(not r.IsEmpty()), True),
        (r.CovMatrixStatus(), 3),
        (bool(r.MinimizerType().startswith("Minuit")), True),
        (r.CovMatrix(1, 1), pytest.approx(r.ParError(1) ** 2)),
        (r.Correlation(0, 0), pytest.approx(1)),
    )
    matrix = r.GetCovarianceMatrix()
    expect(
        (bool(matrix.GetNrows() == matrix.GetNcols() == 3), True),
        (matrix(1, 1), r.CovMatrix(1, 1)),
        (matrix[1][1], r.CovMatrix(1, 1)),
        (len(matrix.GetMatrixArray()), 9),
        (r.GetCorrelationMatrix()(2, 2), pytest.approx(1)),
        (r.FittedFunction().GetName(), "gaus"),
    )
    r.Print("V")
    matrix.Print()
    out = capsys.readouterr().out
    expect(
        (bool("Minimizer is Minuit2" in out), True),
        (bool("3x3 matrix is as follows" in out), True),
        (r.GetName(), "TFitResult-h-gaus"),
    )


def test_an_empty_fit_is_roots_warning_and_status(capsys):
    empty = ROOT.TH1D("e", "", 10, 0, 1)
    r = empty.Fit("gaus", "Q")
    expect(
        (int(r), -1),
        (bool(not r), True),
        (bool(r.Get() is None), True),
        (bool("Fit data is empty" in capsys.readouterr().err), True),
    )
    with pytest.raises(AttributeError):
        r.Parameter  # noqa: B018
    with pytest.raises(AttributeError):
        r._hidden  # noqa: B018
    assert fits.TFitResult().IsEmpty()


def test_fitting_a_user_function_by_name_and_by_object():
    h, _ = fitted()
    mine = ROOT.TF1("mine", "[0]*exp(-0.5*((x-[1])/[2])**2)", -4, 4)
    mine.SetParameters(100, 0, 1)
    r = h.Fit("mine", "QS")
    expect(
        (r.Parameter(2), pytest.approx(1.0, rel=0.1)),
        (mine.GetParameter(2), pytest.approx(r.Parameter(2))),
    )
    r2 = h.Fit(mine, "QS")
    expect(
        (r2.Chi2(), pytest.approx(r.Chi2(), rel=1e-6)),
        (fits._model("gaus"), "gaus"),
        (fits._model("nothing_named_so"), "nothing_named_so"),
    )


def test_an_efficiency_is_made_filled_and_answered_bin_by_bin():
    e = ROOT.TEfficiency("eff", "t;x;eff", 4, 0, 4)
    for x, passed in ((0.5, True), (0.5, False), (1.5, True), (2.5, 1)):
        e.Fill(passed, x)
    e.FillWeighted(True, 2.0, 3.5)
    expect(
        (e.GetName(), "eff"),
        (e.GetTitle(), "t"),
        (e.GetDimension(), 1),
        (e.GetTotalHistogram().GetXaxis().GetTitle(), "x"),
        (e.GetGlobalBin(2), 2),
        (e.FindFixBin(1.5), 2),
        (e.GetPassedHistogram().GetEntries(), 4),
        (bool(e.GetCopyPassedHisto() is not e.GetPassedHistogram()), True),
        (e.GetCopyTotalHisto().GetEntries(), 5),
    )
    e.SetName("renamed")
    assert e.GetName() == "renamed"
    two = ROOT.TEfficiency("e2", "", 2, 0, 2, 2, 0, 2)
    two.Fill(True, 0.5, 0.5)
    expect(
        (two.GetDimension(), 2),
        (two.GetEfficiency(two.GetGlobalBin(1, 1)), 1),
        (ROOT.TEfficiency().GetName(), "eff"),
        (ROOT.TEfficiency("v", "", 2, np.array([0.0, 1, 3])).GetDimension(), 1),
    )


def test_an_efficiencys_intervals_follow_its_statistic_option():
    e = ROOT.TEfficiency("eff", "", 1, 0, 1)
    for passed in (True, True, False, False):
        e.Fill(passed, 0.5)
    expect(
        (e.GetEfficiency(1), 0.5),
        (bool(e.GetStatisticOption() == e.kFCP == 0), True),
    )
    low, up = e.GetEfficiencyErrorLow(1), e.GetEfficiencyErrorUp(1)
    expect(
        (bool(low > 0), True),
        (bool(up > 0), True),
        (bool(not e.UsesBayesianStat()), True),
    )
    e.SetStatisticOption(ROOT.TEfficiency.kFWilson)
    assert e.GetEfficiencyErrorUp(1) != up
    e.SetStatisticOption(ROOT.TEfficiency.kBJeffrey)
    assert e.UsesBayesianStat()
    e.SetConfidenceLevel(0.95)
    e.SetBetaAlpha(2.0)
    e.SetBetaBeta(3.0)
    assert (e.GetConfidenceLevel(), e.GetBetaAlpha(), e.GetBetaBeta()) == (0.95, 2.0, 3.0)
    graph = e.CreateGraph()
    expect(
        (graph.ClassName(), "TGraphAsymmErrors"),
        (graph.GetN(), 1),
    )
    shown = e.CreateHistogram()
    assert shown.GetBinContent(1) == pytest.approx(e.GetEfficiency(1))
    other = ROOT.TEfficiency("o", "", 1, 0, 1)
    other.Fill(True, 0.5)
    e.Add(other)
    e += other
    assert e.GetTotalHistogram().GetEntries() == 6


def test_an_efficiency_from_two_histograms_is_checked_first():
    passed, total = ROOT.TH1D("p", "", 2, 0, 2), ROOT.TH1D("t", "", 2, 0, 2)
    total.Fill(0.5)
    passed.Fill(0.5)
    expect(
        (bool(ROOT.TEfficiency.CheckConsistency(passed, total)), True),
        (bool(not ROOT.TEfficiency.CheckConsistency(total, ROOT.TH1D("z", "", 3, 0, 1))), True),
        (ROOT.TEfficiency(passed, total).GetEfficiency(1), 1),
    )
    read = wrapping.wrap(Efficiency.book("r", (2, 0, 2)))
    expect(
        (read.ClassName(), "TEfficiency"),
        (bool(read.GetLineColor() >= 0), True),
    )


def test_the_wrapper_registry_hands_back_one_wrapper_per_object():
    made = Histogram.book("h", (2, 0, 1), kind="F")
    first = wrapping.wrap(made)
    expect(
        (first.ClassName(), "TH1F"),
        (bool(wrapping.wrap(made) is first), True),
        (bool(wrapping.wrap(first) is first), True),
        (bool(wrapping.wrap(None) is None), True),
        (wrapping.wrap(3), 3),
        (bool(wrapping.unwrap(first) is made), True),
        (wrapping.unwrap(5), 5),
    )

    class Thing:
        classname = "TThing"

    class Stand:
        def __init__(self, xrd):
            self._xrd = xrd

    wrapping.register(kind=Thing, factory=Stand)
    wrapping.register("TOtherThing", factory=Stand)
    try:
        thing = Thing()
        assert isinstance(wrapping.wrap(thing), Stand)
    finally:
        wrapping.BY_TYPE.pop()
        wrapping.BY_CLASSNAME.pop("TOtherThing")


def test_what_the_trees_are_given_is_the_cores(tmp_path):
    from xrdroot.pyroot.trees import _base

    expect(
        (bool(treelinks.connect()), True),
        (bool(_base.hooks.draw is ROOT.draw_hook), True),
        (bool(_base.hooks.directory() is None), True),
    )
    out = ROOT.TFile("t.root", "RECREATE")
    expect(
        (bool(_base.hooks.directory() is out), True),
        (bool(out._xrd is not None), True),
    )
    registry = _base.hooks.registry()
    booked = ROOT.TH1D("h", "", 2, 0, 1)
    expect(
        (bool(registry["h"] is booked._xrd), True),
        (bool("h" in list(registry)), True),
        (len(registry), 1),
    )
    registry["h"] = booked._xrd
    registry["made"] = Histogram.book("tmp", (1, 0, 1))
    assert out.Get("made").GetName() == "made"
    marker = ROOT.TNamed("marker", "")
    registry["marker"] = marker
    assert registry["marker"] is marker
    del registry["marker"]
    with pytest.raises(KeyError):
        registry["marker"]
    with pytest.raises(KeyError):
        del registry["marker"]
    out.Close()
    ROOT.TFile(str(tmp_path / "t.root"))
    assert _base.hooks.directory() is None


def test_the_trees_are_connected_whichever_is_imported_first(monkeypatch):
    monkeypatch.setattr(treelinks.importlib.util, "find_spec", lambda name: None)
    monkeypatch.delitem(treelinks.sys.modules, treelinks.BASE)
    assert not treelinks.connect()
