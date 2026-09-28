"""The Factory end to end: booking, training, testing, evaluating, and what it writes."""

from __future__ import annotations

import os

import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from xrdroot.tmva import TMVAError

from tmvasupport import classify, loader, session, weights

__all__ = ["session"]


def test_a_linear_discriminant_is_trained_tested_and_written_with_its_trees(session, capsys):
    classify([(ROOT.TMVA.Types.kLD, "LD", "!H:!V")])
    printed = capsys.readouterr().out
    assert "Evaluation results ranked by best signal efficiency" in printed
    assert os.path.exists(weights("LD"))
    with xrdroot.open_root("out.root") as found:
        test = found["dataset/TestTree"]
        assert "LD" in test.keys() and test.num_entries == 200


def test_the_factory_greets_only_when_it_writes_no_file(session, capsys):
    ROOT.TMVA.Factory("job", "!V:AnalysisType=Classification")
    assert "TMVA Version" in capsys.readouterr().out
    ROOT.TMVA.Factory("job", ROOT.TFile.Open("f.root", "RECREATE"), "!V")
    assert "TMVA Version" not in capsys.readouterr().out


def test_a_method_booked_twice_under_one_title_is_refused(session):
    factory = ROOT.TMVA.Factory("job", "!V:Silent:AnalysisType=Classification")
    data = loader()
    factory.BookMethod(data, "LD", "LD", "")
    with pytest.raises(TMVAError, match="already exists"):
        factory.BookMethod(data, "LD", "LD", "")


def test_a_method_xrdroot_does_not_have_is_refused_by_name(session):
    factory = ROOT.TMVA.Factory("job", "!V:Silent:AnalysisType=Classification")
    with pytest.raises(xrdroot.UnsupportedFeatureError, match="PyKeras"):
        factory.BookMethod(loader(), "PyKeras", "keras", "")
