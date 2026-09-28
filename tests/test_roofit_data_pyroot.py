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

    assert (kit._axis() == TAxis._of, kit._histogram_wrapper().wraps is wrap) == (True, True)
    assert kit._pave() is TPaveText


def test_without_the_core_parts_the_engines_stand_ins_serve(monkeypatch: Any) -> None:
    """Were the core part's axis or wrapper missing, the engine's own would be used."""
    monkeypatch.setitem(sys.modules, "xrdroot.pyroot.core.axes", None)
    monkeypatch.setitem(sys.modules, "xrdroot.pyroot.core.wrapping", None)
    assert kit._axis() is frame.Axis
    made = object()
    assert kit._histogram_wrapper()(made) is made
