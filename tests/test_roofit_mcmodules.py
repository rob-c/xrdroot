"""The MC-study modules and the hooks the study gives them."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT


@pytest.fixture
def model():
    x = ROOT.RooRealVar("x", "x", -5.0, 5.0)
    mean = ROOT.RooRealVar("mean", "mean", 0.0, -2.0, 2.0)
    sigma = ROOT.RooRealVar("sigma", "sigma", 1.0, 0.5, 2.0)
    return x, mean, sigma, ROOT.RooGaussian("gauss", "gauss", x, mean, sigma)


def test_a_binned_study_keeps_the_chi_square_of_every_fit(model):
    x, _, _, gauss = model
    study = ROOT.RooMCStudy(gauss, {x}, Silence=True, Binned=True)
    module = study.addModule(ROOT.RooChi2MCSModule())
    assert module.study is study
    study.generateAndFit(3, 200)
    data = study.fitParDataSet()
    names = set(data.columns())
    assert {"chi2", "chi2red", "prob"} <= names
    chi2 = data.column("chi2")
    assert len(chi2) == len(study.rows)
    assert np.all(chi2 >= 0.0)
    assert np.all(data.column("prob") >= 0.0)


def test_a_fit_that_failed_contributes_no_chi_square(model):
    x, _, _, gauss = model
    study = ROOT.RooMCStudy(gauss, {x}, Silence=True, Binned=True)
    study.addModule(ROOT.RooChi2MCSModule())
    study.generateAndFit(1, 100)
    before = len(study.rows)
    study._fit_sample(study.last_sample.__class__("empty", "empty", list(study.observables)))
    assert len(study.rows) == before           # nothing was fitted, so no row was added


def test_the_randomisation_module_draws_the_generator_parameters(model):
    x, mean, sigma, gauss = model
    study = ROOT.RooMCStudy(gauss, {x}, Silence=True)
    module = ROOT.RooRandomizeParamMCSModule()
    module.sampleUniform(mean, -1.0, 1.0)
    module.sampleGaussian(sigma, 1.0, 0.1)
    module.sampleSumUniform(ROOT.RooArgSet(mean), 0.0, 0.5)
    module.sampleSumGauss(ROOT.RooArgSet(sigma), 0.0, 0.2)
    study.addModule(module)
    study.generateAndFit(2, 100)
    data = study.fitParDataSet()
    assert "mean_gen" in set(data.columns())
    assert len(data.column("mean_gen")) == 2
    assert -1.5 <= float(np.mean(data.column("mean_gen"))) <= 1.5


def test_the_dll_module_measures_the_distance_from_the_null(model):
    x, _, _, gauss = model
    study = ROOT.RooMCStudy(gauss, {x}, Silence=True, Binned=True)
    module = study.addModule(ROOT.RooDLLSignificanceMCSModule("sigma", 1.0))
    assert module.initializeInstance() is True
    study.generateAndFit(2, 150)
    data = study.fitParDataSet()
    assert {"dll_nullhypo_sigma", "significance_nullhypo_sigma"} <= set(data.columns())
    assert len(data.column("dll_nullhypo_sigma")) == 2
    assert np.all(data.column("significance_nullhypo_sigma") >= 0.0)


def test_a_module_without_a_result_contributes_nothing(model):
    x, _, _, gauss = model
    study = ROOT.RooMCStudy(gauss, {x}, Silence=True)
    empty = ROOT.RooChi2MCSModule()
    empty.study = study
    study.last_sample = None
    assert empty.processAfterFit(True) is True
    assert empty.finalizeRun()["chi2"] == [0.0]


def test_a_chi_square_module_asked_only_for_a_bad_fit_measures_nothing():
    module = ROOT.RooChi2MCSModule()
    module.initializeInstance()
    module.initializeRun(1)
    assert module.processAfterFit(False) is True
    assert module.finalizeRun()["chi2"] == []


def test_the_dll_module_measures_nothing_when_no_fit_has_happened(model):
    x, _, _, gauss = model
    study = ROOT.RooMCStudy(gauss, {x}, Silence=True)
    module = ROOT.RooDLLSignificanceMCSModule("sigma", 1.0)
    module.study = study
    assert module.processAfterFit(True) is True
    assert module.finalizeRun()["dll_nullhypo_sigma"] == []


def test_a_constant_parameter_gets_no_pull_column(model):
    x, mean, _, gauss = model
    mean.setConstant(True)
    study = ROOT.RooMCStudy(gauss, {x}, Silence=True)
    study.generateAndFit(2, 80)
    data = study.fitParDataSet()
    assert "sigmaerr" in set(data.columns())
    assert "sigmapull" in set(data.columns())
    assert "meanpull" not in set(data.columns())


def test_a_module_with_nothing_to_finalise_and_a_short_column_are_both_handled(model):
    x, _, _, gauss = model
    study = ROOT.RooMCStudy(gauss, {x}, Silence=True, Binned=True)

    class Silent:
        def initializeInstance(self):
            return True

        def initializeRun(self, numSamples):
            return True

        def processBeforeGen(self, sampleNum):
            return True

        def processAfterFit(self, fitOk):
            return True

    class Short:
        def initializeInstance(self):
            return True

        def initializeRun(self, numSamples):
            return True

        def finalizeRun(self):
            return {"partial": []}

    study.addModule(Silent())   # nothing to finalise: it contributes no column
    study.addModule(Short())    # a column shorter than the rows keeps one value per row
    study.generateAndFit(1, 100)
    assert set(study.rows[0]) >= {"partial"}
    assert study.rows[0]["partial"] == 0.0
