"""The data side of TMVA end to end: a loader's trees split, transformed, ranked and written."""

from __future__ import annotations

import os

import xrdroot
import xrdroot.pyroot as ROOT
from tmvasupport import classify, session, weights

__all__ = ["session"]


def test_a_likelihood_over_every_transformation_writes_its_weights_and_its_plots(session, capsys):
    classify(
        [
            (ROOT.TMVA.Types.kLikelihood, "Likelihood", "!H:!V:VarTransform=N,D,P,G,U"),
            (ROOT.TMVA.Types.kLD, "LD", "!H:!V:VarTransform=D"),
        ],
        options="!V:!Silent:AnalysisType=Classification:Transformations=I;D;P;G,D",
    )
    printed = capsys.readouterr().out
    assert "Ranking input variables" in printed
    assert os.path.exists(weights("Likelihood"))
    with xrdroot.open_root("out.root") as found:
        assert found["dataset/TestTree"].num_entries == 200
