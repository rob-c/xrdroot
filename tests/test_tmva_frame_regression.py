"""The Factory over a regression: its training, its tables and the histograms of its deviations."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from tmvasupport import regression_tree, session
from xrdroot.tmva import regeval

__all__ = ["session"]


def regression_loader(name: str = "dataset") -> object:
    """A loader of two variables and one target over :func:`regression_tree`."""
    made = ROOT.TMVA.DataLoader(name)
    made.AddVariable("var1", "Variable 1", "units", "F")
    made.AddVariable("var2", "Variable 2", "units", "F")
    made.AddTarget("fvalue")
    made.AddRegressionTree(regression_tree(), 1.0)
    made.PrepareTrainingAndTestTree("", "SplitMode=Random:NormMode=NumEvents:!V")
    return made


def regress(methods: list[tuple[str, str, str]], output: str = "reg.root") -> object:
    """A regression Factory that has booked, trained, tested and evaluated ``methods``."""
    target = ROOT.TFile.Open(output, "RECREATE")
    factory = ROOT.TMVA.Factory("job", target, "!V:!Silent:AnalysisType=Regression")
    data = regression_loader()
    for kind, title, text in methods:
        factory.BookMethod(data, kind, title, text)
    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()
    target.Close()
    return factory


def test_a_regression_is_ranked_by_its_deviation_and_its_histograms_written(session, capsys):
    regress([("LD", "LD", "!H:!V:VarTransform=N"), ("BDT", "BDT", "!H:!V:NTrees=5:BoostType=Grad")])
    printed = capsys.readouterr().out
    assert "Evaluation results ranked by smallest RMS on test sample:" in printed
    assert "Ranking input variables (method unspecific)..." in printed
    assert "Dataset[dataset] : Create results for training" in printed
    with xrdroot.open_root("reg.root") as found:
        method = found["dataset/Method_LD/LD"].keys()
        assert any("Quadr_Deviation_target_0" in name for name in method)
        assert any("MVA_LDtrain_reg_var0_rtgt0" in name for name in method)
        assert "fvalue" in found["dataset/TestTree"].keys()


def test_a_regression_read_back_by_a_reader_gives_every_target(session):
    regress([("LD", "LD", "")])
    reader = ROOT.TMVA.Reader("!Color:Silent")
    var1, var2 = np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32)
    reader.AddVariable("var1", var1)
    reader.AddVariable("var2", var2)
    reader.BookMVA("LD", "dataset/weights/job_LD.weights.xml")
    var1[0], var2[0] = 2.0, 3.0
    every = reader.EvaluateRegression("LD")
    assert len(every) == 1
    assert reader.EvaluateRegression(0, "LD") == every[0]
    assert reader.EvaluateRegression([2.0, 3.0], "LD")[0] == every[0]


def test_the_mutual_information_of_an_empty_histogram_is_minus_one():
    assert regeval.mutual_information(np.zeros((4, 4))) == -1.0


def test_a_regression_of_no_events_takes_the_unit_range():
    empty = np.zeros(0)
    assert regeval._sum(empty) == 0.0


def test_a_quantile_of_an_empty_histogram_is_its_low_edge():
    from xrdroot.tmva import hists

    made = hists.book("q", "q", 4, 0.0, 4.0)
    assert regeval.quantile(made, 0.5) == 0.0


def test_a_quantile_on_a_bin_edge_is_the_middle_of_the_empty_bins_after_it():
    from xrdroot.tmva import hists

    made = hists.book("q", "q", 6, 0.0, 6.0)
    made.fill(np.array([0.5, 4.5]))
    # ROOT's integral is 0, .5, .5, .5, .5, 1, 1: .5 is reached at 1, and 1 to 4 is empty.
    assert regeval.quantile(made, 0.5) == 2.5
    assert regeval.quantile(made, 0.75) == 4.5


def test_a_quantile_of_the_whole_histogram_is_its_high_edge():
    from xrdroot.tmva import hists

    made = hists.book("q", "q", 4, 0.0, 4.0)
    made.fill(np.array([0.5, 3.5]))
    assert regeval.quantile(made, 1.0) == 4.0


def test_a_regression_that_hits_every_target_is_histogrammed_on_rounded_axes(session):
    data = regression_loader()
    events = data.dataset().train
    made = regeval.deviation_histograms("MVA_x", data.info, events, events.targets)
    deviation = next(h for h in made if h.name == "MVA_x_reg_var0_rtgt0")
    assert deviation.axes[1].low < 0.0 < deviation.axes[1].high
    squared = next(h for h in made if h.name == "MVA_x_Quadr_Deviation_target_0_")
    assert squared.axes[0].low < 0.0 < squared.axes[0].high
    assert float(np.sum(regeval.hists.bins(squared))) == pytest.approx(
        float(np.sum(events.weights))
    )


def test_an_axis_of_no_width_and_nothing_in_it_is_widened_about_its_one_end():
    nbins, low, high = regeval.hists.auto_axis(10, 0.0, 0.0, np.zeros(0))
    assert nbins == 10 and low < -0.9 and high > 0.9
    assert regeval.hists.auto_axis(10, 0.0, 1.0, np.zeros(0)) == (10, 0.0, 1.0)


def test_a_constant_variable_is_histogrammed_on_the_range_roots_buffer_would_find():
    made = regeval.hists.filled("c", "c", (40, 2.0, 2.0), np.full(5, 2.0))
    edges = made.axes[0].edges()
    assert edges[0] < 2.0 < edges[-1]
    assert float(np.sum(regeval.hists.bins(made))) == 5.0
