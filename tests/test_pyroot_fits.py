"""``TFitResult``, ``TFitResultPtr``, ``TEfficiency``, and how xrdroot objects are wrapped."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
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
    assert int(r) == 0
    assert r == 0
    assert r.__index__() == 0
    assert bool(r)
    assert hash(r) == hash(0)
    assert r != 1
    assert r != "x"
    assert "status 0" in repr(r)
    assert r.Get() is not None
    assert r.Parameter(1) == r.Parameters()[1] == r.GetParams()[1]
    assert r.ParError(2) == r.Error(2)
    assert list(r.Errors()) == list(r.GetErrors())
    assert r.LowerError(0) == r.UpperError(0) == r.ParError(0)
    assert r.ParName(2) == "Sigma"
    assert r.Index("Mean") == 1
    assert r.Index("zz") == -1
    assert r.NPar() == r.NTotalParameters() == 3
    assert r.NFreeParameters() == 3
    assert not r.IsParameterFixed(0)
    assert r.IsParameterBound(2)
    assert r.Ndf() == r.Ndf()
    assert r.Chi2() > 0
    assert 0 <= r.Prob() <= 1
    assert r.MinFcnValue() > 0
    assert r.Edm() >= 0
    assert r.NCalls() > 0
    assert r.Status() == 0
    assert r.IsValid()
    assert not r.IsEmpty()
    assert r.CovMatrixStatus() == 3
    assert r.MinimizerType().startswith("Minuit")
    assert r.CovMatrix(1, 1) == pytest.approx(r.ParError(1) ** 2)
    assert r.Correlation(0, 0) == pytest.approx(1)
    matrix = r.GetCovarianceMatrix()
    assert matrix.GetNrows() == matrix.GetNcols() == 3
    assert matrix(1, 1) == r.CovMatrix(1, 1)
    assert matrix[1][1] == r.CovMatrix(1, 1)
    assert len(matrix.GetMatrixArray()) == 9
    assert r.GetCorrelationMatrix()(2, 2) == pytest.approx(1)
    assert r.FittedFunction().GetName() == "gaus"
    r.Print("V")
    matrix.Print()
    out = capsys.readouterr().out
    assert "Minimizer is Minuit2" in out
    assert "3x3 matrix is as follows" in out
    assert r.GetName() == "TFitResult-h-gaus"


def test_an_empty_fit_is_roots_warning_and_status(capsys):
    empty = ROOT.TH1D("e", "", 10, 0, 1)
    r = empty.Fit("gaus", "Q")
    assert int(r) == -1
    assert not r
    assert r.Get() is None
    assert "Fit data is empty" in capsys.readouterr().err
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
    assert r.Parameter(2) == pytest.approx(1.0, rel=0.1)
    assert mine.GetParameter(2) == pytest.approx(r.Parameter(2))
    r2 = h.Fit(mine, "QS")
    assert r2.Chi2() == pytest.approx(r.Chi2(), rel=1e-6)
    assert fits._model("gaus") == "gaus"
    assert fits._model("nothing_named_so") == "nothing_named_so"


def test_an_efficiency_is_made_filled_and_answered_bin_by_bin():
    e = ROOT.TEfficiency("eff", "t;x;eff", 4, 0, 4)
    for x, passed in ((0.5, True), (0.5, False), (1.5, True), (2.5, 1)):
        e.Fill(passed, x)
    e.FillWeighted(True, 2.0, 3.5)
    assert e.GetName() == "eff"
    assert e.GetTitle() == "t"
    assert e.GetDimension() == 1
    assert e.GetTotalHistogram().GetXaxis().GetTitle() == "x"
    assert e.GetGlobalBin(2) == 2
    assert e.FindFixBin(1.5) == 2
    assert e.GetPassedHistogram().GetEntries() == 4
    assert e.GetCopyPassedHisto() is not e.GetPassedHistogram()
    assert e.GetCopyTotalHisto().GetEntries() == 5
    e.SetName("renamed")
    assert e.GetName() == "renamed"
    two = ROOT.TEfficiency("e2", "", 2, 0, 2, 2, 0, 2)
    two.Fill(True, 0.5, 0.5)
    assert two.GetDimension() == 2
    assert two.GetEfficiency(two.GetGlobalBin(1, 1)) == 1
    assert ROOT.TEfficiency().GetName() == "eff"
    assert ROOT.TEfficiency("v", "", 2, np.array([0.0, 1, 3])).GetDimension() == 1


def test_an_efficiencys_intervals_follow_its_statistic_option():
    e = ROOT.TEfficiency("eff", "", 1, 0, 1)
    for passed in (True, True, False, False):
        e.Fill(passed, 0.5)
    assert e.GetEfficiency(1) == 0.5
    assert e.GetStatisticOption() == e.kFCP == 0
    low, up = e.GetEfficiencyErrorLow(1), e.GetEfficiencyErrorUp(1)
    assert low > 0
    assert up > 0
    assert not e.UsesBayesianStat()
    e.SetStatisticOption(ROOT.TEfficiency.kFWilson)
    assert e.GetEfficiencyErrorUp(1) != up
    e.SetStatisticOption(ROOT.TEfficiency.kBJeffrey)
    assert e.UsesBayesianStat()
    e.SetConfidenceLevel(0.95)
    e.SetBetaAlpha(2.0)
    e.SetBetaBeta(3.0)
    assert (e.GetConfidenceLevel(), e.GetBetaAlpha(), e.GetBetaBeta()) == (0.95, 2.0, 3.0)
    graph = e.CreateGraph()
    assert graph.ClassName() == "TGraphAsymmErrors"
    assert graph.GetN() == 1
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
    assert ROOT.TEfficiency.CheckConsistency(passed, total)
    assert not ROOT.TEfficiency.CheckConsistency(total, ROOT.TH1D("z", "", 3, 0, 1))
    assert ROOT.TEfficiency(passed, total).GetEfficiency(1) == 1
    read = wrapping.wrap(Efficiency.book("r", (2, 0, 2)))
    assert read.ClassName() == "TEfficiency"
    assert read.GetLineColor() >= 0


def test_the_wrapper_registry_hands_back_one_wrapper_per_object():
    made = Histogram.book("h", (2, 0, 1), kind="F")
    first = wrapping.wrap(made)
    assert first.ClassName() == "TH1F"
    assert wrapping.wrap(made) is first
    assert wrapping.wrap(first) is first
    assert wrapping.wrap(None) is None
    assert wrapping.wrap(3) == 3
    assert wrapping.unwrap(first) is made
    assert wrapping.unwrap(5) == 5

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

    assert treelinks.connect()
    assert _base.hooks.draw is ROOT.draw_hook
    assert _base.hooks.directory() is None
    out = ROOT.TFile("t.root", "RECREATE")
    assert _base.hooks.directory() is out
    assert out._xrd is not None
    registry = _base.hooks.registry()
    booked = ROOT.TH1D("h", "", 2, 0, 1)
    assert registry["h"] is booked._xrd
    assert "h" in list(registry)
    assert len(registry) == 1
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
