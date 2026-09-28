"""RooFit under ``import ROOT``: the ``RooFit`` namespace and the kit's hooks into the engine.

The levels, topics and constants below are ROOT 6.40's, printed by PyROOT;
the commands are the engine's :class:`RooCmdArg`, named as ROOT names them.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.pyroot import roofit as kit
from xrdroot.pyroot.roofit.commands import COMMANDS
from xrdroot.roofit import histograms
from xrdroot.roofit.cmdargs import RooCmdArg
from xrdroot.roofit.plot import frame


@pytest.fixture(autouse=True)
def _fresh(tmp_path: Any) -> Iterator[None]:
    """A clean ROOT session, and the kit's histogram wrapper in place, as ``import ROOT`` sets."""
    wrapper = histograms.WRAP[0]
    histograms.set_wrapper(kit._histogram_wrapper())
    yield from fresh(tmp_path)
    histograms.set_wrapper(wrapper)


def test_the_roofit_namespace_holds_roots_message_levels_and_topics() -> None:
    """``RooFit.INFO``, ``RooFit.Fitting``...: a macro's message settings are these numbers."""
    rf = ROOT.RooFit
    levels = [rf.DEBUG, rf.INFO, rf.PROGRESS, rf.WARNING, rf.ERROR, rf.FATAL]
    topics = [rf.Generation, rf.Minimization, rf.Plotting, rf.Fitting, rf.Integration,
              rf.LinkStateMgmt, rf.Eval, rf.Caching, rf.Optimization, rf.ObjectHandling,
              rf.InputArguments, rf.Tracing, rf.Contents, rf.DataHandling, rf.NumIntegration,
              rf.FastEvaluations, rf.HistFactory]  # fmt: skip
    assert levels == [0, 1, 2, 3, 4, 5]
    assert topics == [1 << bit for bit in range(17)]
    assert repr(rf) == "<namespace RooFit>"


def test_each_command_of_the_namespace_makes_its_named_argument() -> None:
    """``RooFit.Binning(10, 0, 5)`` is the engine's command of that name and those values."""
    rf = ROOT.RooFit
    binning = rf.Binning(10, 0, 5)
    assert (isinstance(binning, RooCmdArg), binning.GetName(), binning.args) == (
        True,
        "Binning",
        (10, 0, 5),
    )
    assert (rf.LineColor(2).GetName(), rf.LineColor(2).value(0)) == ("LineColor", 2)
    assert all(getattr(rf, name)().GetName() == name for name in COMMANDS)
    assert (rf.Save.__name__, rf.Save.__doc__) == (
        "Save",
        "``RooFit::Save(...)``: the ``Save`` command argument.",
    )


def test_pyroots_keyword_spellings_of_commands_are_read_as_arguments() -> None:
    """``YVar(var=y, Binning=5)``, ``Format(what="NE", AutoPrecision=1)``: PyROOT's spellings."""
    rf = ROOT.RooFit
    y = ROOT.RooRealVar("y", "y", 0, 1)
    made = rf.YVar(var=y, Binning=5)
    assert (made.value(0) is y, made.value(1).GetName(), made.value(1).value(0)) == (
        True,
        "Binning",
        5,
    )
    shown = rf.Format(what="NE", AutoPrecision=1)
    assert (shown.value(0), shown.value(1).GetName(), shown.value(1).value(0)) == (
        "NE",
        "AutoPrecision",
        1,
    )


def test_roofit_const_is_a_constant_named_after_its_value(capsys: Any) -> None:
    """``RooFit.RooConst(2.5)`` prints as ROOT's ``RooConstVar::2.5 = 2.5``."""
    c = ROOT.RooFit.RooConst(2.5)
    c.Print()
    assert capsys.readouterr().out == "RooConstVar::2.5 = 2.5\n"
    assert (c.GetName(), c.ClassName()) == ("2.5", "RooConstVar")


def test_roots_namespace_gives_roofits_classes_and_not_the_core_parts() -> None:
    """``ROOT.RooRealVar`` is the engine's class; ``TMatrixDSym`` is left to the core part."""
    from xrdroot.roofit.variables import RooRealVar

    assert ROOT.RooRealVar is RooRealVar and vars(kit)["RooRealVar"] is RooRealVar
    assert "RooWorkspace" in kit.__all__ and "TMatrixDSym" not in kit.__all__


def test_a_histogram_of_data_under_import_root_is_a_th1_renamed(capsys: Any) -> None:
    """``createHistogram("h", x)`` hands back a ``TH1`` with ROOT's contents, named as asked."""
    x = ROOT.RooRealVar("x", "x", 0, 10)
    x.setBins(5)
    d = ROOT.RooDataSet("d", "d", ROOT.RooArgSet(x))
    for value in (0.5, 1.5, 4.5, 9.5):
        x.setVal(value)
        d.add(ROOT.RooArgSet(x))
    h = d.createHistogram("h1", x)
    assert (h.ClassName()[:3], h.GetName().startswith("h1"), h.GetNbinsX()) == ("TH1", True, 5)
    assert [h.GetBinContent(i) for i in range(1, 6)] == [2.0, 0.0, 1.0, 0.0, 1.0]
    y = ROOT.RooRealVar("y", "y", 0, 1)
    both = ROOT.RooDataSet("b", "b", ROOT.RooArgSet(x, y))
    both.add(ROOT.RooArgSet(x, y))
    two = both.createHistogram("h2", x, ROOT.RooFit.YVar(y, ROOT.RooFit.Binning(2)))
    assert (two.GetName().startswith("h2"), two.GetNbinsY()) == (True, 2)


def test_the_kits_hooks_are_the_core_parts_when_it_has_them() -> None:
    """A frame's axis is the core part's ``TAxis``; a ``paramOn`` box the graphics part's pave."""
    from xrdroot.pyroot.core.axes import TAxis
    from xrdroot.pyroot.core.wrapping import wrap
    from xrdroot.pyroot.graphics.paves import TPaveText

    assert (kit._axis() == TAxis._of, kit._histogram_wrapper() is wrap) == (True, True)
    assert kit._pave() is TPaveText


def test_without_the_core_parts_the_engines_stand_ins_serve(monkeypatch: Any) -> None:
    """Were the core part's axis or wrapper missing, the engine's own would be used."""
    monkeypatch.setitem(sys.modules, "xrdroot.pyroot.core.axes", None)
    monkeypatch.setitem(sys.modules, "xrdroot.pyroot.core.wrapping", None)
    assert kit._axis() is frame.Axis
    made = object()
    assert kit._histogram_wrapper()(made) is made


def _weighted() -> tuple[Any, ...]:
    """rf403's model: a flat density's 200 events weighted ``x*x+10``, and a parabola to fit."""
    ROOT.RooRandom.randomGenerator().SetSeed(4357)
    x = ROOT.RooRealVar("x", "x", -10, 10)
    data = ROOT.RooPolynomial("px", "px", x).generate({x}, 200)
    data.addColumn(ROOT.RooFormulaVar("w", "event weight", "(x*x+10)", [x]))
    wdata = ROOT.RooDataSet(data.GetName(), data.GetTitle(), data.get(), Import=data, WeightVar="w")
    a1, a2 = ROOT.RooRealVar("a1", "a1", 0, -1, 1), ROOT.RooRealVar("a2", "a2", 1, 0, 10)
    p2 = ROOT.RooPolynomial("p2", "p2", x, [ROOT.RooRealVar("a0", "a0", 1), a1, a2], 0)
    return x, wdata, a1, a2, p2


def test_a_weighted_fit_warns_and_sumw2_corrects_its_errors_as_root_does(capsys: Any) -> None:
    """ROOT's warning, then with ``SumW2Error`` its second HESSE and ``V C^-1 V``'s errors."""
    from refmachine import roots

    _, wdata, a1, a2, p2 = _weighted()
    capsys.readouterr()
    p2.fitTo(wdata, PrintLevel=-1)
    warned = [line for line in capsys.readouterr().out.splitlines() if "WARNING" in line]
    assert warned == [
        "[#0] WARNING:InputArguments -- RooAbsPdf::fitTo(p2): WARNING: a likelihood fit is "
        "requested of what appears to be weighted data."
    ]
    assert (a1.getError(), a2.getError()) == roots((0.008124401125600644, 0.004725590174225074),
                                                   rel=1e-7)  # fmt: skip
    a1.setVal(0), a2.setVal(1), a1.setError(0), a2.setError(0)
    r = p2.fitTo(wdata, Save=True, SumW2Error=True, PrintLevel=-1)
    said = capsys.readouterr().out
    assert "WARNING" not in said and said.splitlines()[-1] == (
        "[#1] INFO:Fitting -- RooAbsPdf::fitTo(p2) Calculating sum-of-weights-squared "
        "correction matrix for covariance matrix"
    )
    errors = (a1.getError(), a2.getError(), r.covarianceMatrix()(0, 1))
    assert errors == pytest.approx((0.06099054077538379, 0.05810222152516743,
                                    0.0007353183966308126), rel=1e-7)  # fmt: skip
    assert (r.edm(), r.minNll(), r.covQual()) == pytest.approx(
        (16314.401342266516, 23656.91371382899, 3), rel=1e-7
    )
    r.Print()
    assert "Status : MINIMIZE=0 HESSE=0 HESSE=0" in capsys.readouterr().out


def test_a_sumw2_correction_is_refused_when_the_squared_weights_hessian_is_singular(
    capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RooFit's error when the weight-squared covariance has no Cholesky decomposition."""
    import numpy as np

    _, wdata, _, _, p2 = _weighted()

    def singular(matrix: Any) -> Any:
        raise np.linalg.LinAlgError("not positive definite")

    monkeypatch.setattr(np.linalg, "cholesky", singular)
    capsys.readouterr()
    r = p2.fitTo(wdata, Save=True, SumW2Error=True, PrintLevel=-1)
    assert capsys.readouterr().out.splitlines()[-1] == (
        "[#0] ERROR:Fitting -- RooAbsPdf::fitTo(p2) ERROR: Cannot apply sum-of-weights correction "
        "to covariance matrix: correction matrix calculated with weight-squared is singular"
    )
    assert r.covQual() == -1


def test_a_binned_clone_prints_its_bins_with_its_variables_at_the_last_bin(capsys: Any) -> None:
    """``binnedClone()->Print("v")``, as rf403 prints it: RooFit leaves ``x`` at the last bin."""
    _, wdata, *_ = _weighted()
    binned = wdata.binnedClone()
    binned.Print("v")
    observable = '1)  x = 9.9  L(-10 - 10)  "x"'
    assert capsys.readouterr().out.splitlines() == [
        "DataStore pxData_binned (Generated From px_binned)",
        "  Contains 100 entries",
        "  Observables: ",
        f"    {observable}",
        "Binned Dataset pxData_binned (Generated From px_binned)",
        "  Contains 100 bins with a total weight of 8464.53",
        f"  Observables:     {observable}",
    ]
    assert binned.printMultiline(0, False, "").splitlines()[-1] == "  Observables (x)"
