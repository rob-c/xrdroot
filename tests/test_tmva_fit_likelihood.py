"""The projective likelihood: its reference histograms, its densities, and its weight file."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from tmvafitsupport import SPLIT, reader
from tmvasupport import classify, gaussian, session
from xrdroot.tmva.methods.likelihood import EPSILON, transform_output

__all__ = ["session"]

LIKELIHOOD = ROOT.TMVA.Types.kLikelihood


def test_a_likelihood_writes_its_histograms_and_is_read_back_by_a_reader(session):
    factory = classify([(LIKELIHOOD, "Likelihood", "!H:!V:NAvEvtPerBin=20")])
    with xrdroot.open_root("out.root") as found:
        names = found["dataset/Method_Likelihood/Likelihood"].keys()
    assert any(name.endswith("_additional_check") for name in names)
    assert any(name.endswith("_nice") for name in names)
    made = reader("Likelihood", silent=True)
    high = made.EvaluateMVA([2.0, 2.0, 2.0, 2.0], "Likelihood")
    low = made.EvaluateMVA([-2.0, -2.0, -2.0, -2.0], "Likelihood")
    assert 0.0 <= low < high <= 1.0
    trained = factory.GetMethod("dataset", "Likelihood")
    assert 0.0 < trained.evaluate(np.zeros((1, 4)))[0] < 1.0


def test_a_transformed_likelihood_of_decorrelated_variables_without_negative_weights(session):
    options = "!H:!V:TransformOutput:VarTransform=D:IgnoreNegWeightsInTraining"
    classify([(LIKELIHOOD, "LikelihoodD", options)])
    found = reader("LikelihoodD", silent=True).EvaluateMVA([2.0, 2.0, 2.0, 2.0], "LikelihoodD")
    assert found > 0


def test_integer_variables_have_a_bin_per_value(session):
    made = ROOT.TMVA.DataLoader("dataset")
    for variable in ("var1", "var2"):
        made.AddVariable(f"int({variable}*3)", "I")
    made.AddSignalTree(gaussian("TreeS", 1.0, 1))
    made.AddBackgroundTree(gaussian("TreeB", -1.0, 2))
    made.PrepareTrainingAndTestTree("", SPLIT)
    classify([(LIKELIHOOD, "LikelihoodI", "!H:!V")], data=made)
    with xrdroot.open_root("out.root") as found:
        folder = found["dataset/Method_Likelihood/LikelihoodI"]
        histogram = folder[next(k for k in folder.keys() if k.endswith("_sig"))]
        axis = histogram.axes[0]
        assert (axis.high - axis.low) / axis.nbins == pytest.approx(1.0)


def test_the_likelihood_ratio_is_kept_inside_its_range_and_transformed_through_a_sigmoid():
    ratio = transform_output(np.array([1.0, 0.0, 1e300]), np.array([1.0, 1.0, 0.0]), False)
    assert ratio[0] == 0.5 and ratio[1] == pytest.approx(EPSILON) and ratio[2] < 1.0
    found = transform_output(np.array([1.0, 0.0]), np.array([1.0, 1.0]), True)
    assert found[0] == 0.0 and np.isfinite(found[1])
