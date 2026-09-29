"""The ``Classification`` envelope, and the ``ROOT.TMVA`` namespace it is reached through."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from tmvasupport import loader, session
from xrdroot.tmva import TMVAError

__all__ = ["session"]


def _envelope(options: str) -> object:
    made = ROOT.TMVA.Experimental.Classification(loader(), options)
    made.BookMethod(ROOT.TMVA.Types.kLD, "LD", "")
    made.BookMethod("Fisher", "Fisher", "VarTransform=N")
    made.BookMethod("Cuts", "Cuts", "FitMethod=MC:SampleSize=2000")
    made.Evaluate()
    return made


def test_the_envelope_trains_each_method_silently_and_tables_its_roc_integral(session, capsys):
    envelope = _envelope("!V:!Silent")
    printed = capsys.readouterr().out
    assert "Loading booked method: LD LD" in printed
    assert "Jobs = 1 Real Time =" in printed
    assert "Elapsed time for training" not in printed
    results = envelope.GetResults()
    assert [r.GetMethodTitle() for r in results] == ["LD", "Fisher", "Cuts"]
    first = results[0]
    assert first.GetDataLoaderName() == "dataset" and not first.IsCutsMethod()
    assert 0.5 < first.GetROCIntegral() < 1
    graph = first.GetROCGraph()
    assert graph.GetName() == "LD" and graph.GetTitle() == "LD"
    cuts = envelope.GetResult("Cuts", "Cuts")
    assert cuts.IsCutsMethod() and cuts.GetROCIntegral() == cuts.area


def test_the_envelope_with_several_jobs_lists_the_fastest_first(session):
    envelope = _envelope("!V:Silent:Jobs=2")
    assert envelope.jobs == 2 and len(envelope.GetResults()) == 3


def test_the_envelope_refuses_a_result_it_does_not_have(session):
    envelope = ROOT.TMVA.Experimental.Classification(loader(), "Silent")
    with pytest.raises(TMVAError, match="Method BDT/BDT not found in the results"):
        envelope.GetResult("BDT", "BDT")


def test_the_tmva_namespace_names_what_it_lacks_and_shows_what_it_is():
    assert repr(ROOT.TMVA) == "<namespace TMVA>"
    with pytest.raises(AttributeError, match=r"ROOT has TMVA::MethodPyKeras; xrdroot\.pyroot"):
        _ = ROOT.TMVA.MethodPyKeras
    with pytest.raises(AttributeError):
        _ = ROOT.TMVA.__wrapped__
    assert ROOT.TMVA.TMVAGui("file.root") is None
    assert ROOT.TMVA.Experimental.Classification is not None
