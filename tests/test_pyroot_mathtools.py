"""``ROOT.Math``'s functors and tools, ``ROOT.Fit.Fitter``, and names given by keyword."""

from __future__ import annotations

import types

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot.pyroot.core import mathtools


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_functors_wrap_python_functions():
    one = ROOT.Math.Functor1D(lambda x: x * x - 1)
    many = ROOT.Math.Functor(lambda xx: xx[0] + xx[1], 2)
    expect(
        (one(3), 8),
        (one.NDim(), 1),
        (many([1, 2]), 3),
        (many.NDim(), 2),
    )
    grad = ROOT.Math.GradFunctor1D(lambda x: x * x, lambda x: 2 * x)
    assert grad.Derivative(3) == 6
    grad2 = ROOT.Math.GradFunctor(lambda xx: xx[0] * xx[1], lambda xx, i: xx[1 - i], 2)
    assert grad2.Derivative([2, 3], 0) == 3


def test_the_integrator_is_gsls_rule_first():
    integrator = ROOT.Math.Integrator()
    integrator.SetFunction(ROOT.Math.Functor1D(lambda x: x * x - 1))
    expect(
        (integrator.Integral(0, 3), 5.999999999999999),
        (integrator.Status(), 0),
        (bool(integrator.Error() < 1e-12), True),
    )
    wiggly = ROOT.Math.IntegratorOneDim(ROOT.Math.Functor1D(lambda x: np.sqrt(abs(x))))
    expect(
        (wiggly.Integral(-1, 1), pytest.approx(4 / 3, rel=1e-8)),
        (mathtools.kronrod(lambda x: 1.0, 0.0, 2.0)[0], pytest.approx(2.0)),
    )


def test_the_root_finder_brackets_or_follows_a_derivative():
    finder = ROOT.Math.RootFinder(ROOT.Math.RootFinder.kGSL_NEWTON)
    finder.SetFunction(ROOT.Math.GradFunctor1D(lambda x: x * x - 1, lambda x: 2 * x), 3)
    expect(
        (bool(finder.Solve()), True),
        (finder.Root(), 1.0),
    )
    bracket = ROOT.Math.RootFinder()
    bracket.SetFunction(ROOT.Math.Functor1D(lambda x: x - 0.25), 0, 1)
    expect(
        (bool(bracket.Solve()), True),
        (bracket.Root(), pytest.approx(0.25)),
    )


def test_the_fitter_minimises_and_prints_as_root_does(capsys):
    def rosenbrock(xx):
        return 100 * (xx[1] - xx[0] ** 2) ** 2 + (1 - xx[0]) ** 2

    fitter = ROOT.Fit.Fitter()
    assert fitter.FitFCN(ROOT.Math.Functor(rosenbrock, 2), np.zeros(2))
    fitter.Result().Print(ROOT.std.cout)
    out = capsys.readouterr().out
    expect(
        (bool("MinFCN" in out), True),
        (bool("Chi2 " not in out), True),
        (bool("Par_1" in out), True),
        (fitter.Result().Parameter(0), pytest.approx(1.0, abs=1e-3)),
    )

    def gradient(xx, i):
        if i == 0:
            return 2 * (200 * xx[0] ** 3 - 200 * xx[0] * xx[1] + xx[0] - 1)
        return 200 * (xx[1] - xx[0] ** 2)

    graded = ROOT.Fit.Fitter()
    graded.FitFCN(ROOT.Math.GradFunctor(rosenbrock, gradient, 2))
    assert graded.Result().Parameter(1) == pytest.approx(1.0, abs=1e-3)


def test_a_minimizer_from_the_factory_prints_minuit2s_lines(capsys):
    minimizer = ROOT.Math.Factory.CreateMinimizer("Minuit2", "")
    minimizer.SetMaxFunctionCalls(100000)
    minimizer.SetMaxIterations(1000)
    minimizer.SetTolerance(0.001)
    minimizer.SetPrintLevel(1)
    minimizer.SetStrategy(1)
    minimizer.SetFunction(ROOT.Math.Functor(lambda v: (v[1] - v[0] ** 2) ** 2 + (1 - v[0]) ** 2, 2))
    minimizer.SetVariable(0, "x", -1.0, 0.01)
    minimizer.SetLimitedVariable(1, "y", 1.2, 0.01, -10, 10)
    expect(
        (bool(minimizer.Minimize()), True),
        (minimizer.Status(), 0),
        (minimizer.NDim(), 2),
    )
    lines = capsys.readouterr().out.splitlines()
    expect(
        (
            lines[0],
            "Minuit2Minimizer: Minimize with max-calls 100000 "
            "convergence for edm < 0.001 strategy 1",
        ),
        (lines[1], "Minuit2Minimizer : Valid minimum - status = 0"),
        (bool(lines[2].startswith("FVAL  = ")), True),
        (bool(lines[5].startswith("x\t  = ")), True),
        (bool("\t +/-  " in lines[5]), True),
        (minimizer.X()[0], pytest.approx(1.0, abs=0.01)),
        (bool(minimizer.MinValue() < 1e-4), True),
        (len(minimizer.Errors()), 2),
        (bool(minimizer.Edm() >= 0), True),
        (bool(minimizer.NCalls() > 0), True),
    )
    graded = ROOT.Math.Minimizer()
    graded.SetFunction(
        ROOT.Math.GradFunctor(lambda v: (v[0] - 2) ** 2, lambda v, i: 2 * (v[0] - 2), 1)
    )
    graded.SetVariable(0, "a", 0.0, 0.1)
    expect(
        (bool(graded.Minimize()), True),
        (graded.X()[0], pytest.approx(2.0, abs=1e-4)),
    )
    graded._found = types.SimpleNamespace(
        valid=False,
        fval=0.0,
        fmin=types.SimpleNamespace(edm=0.0),
        nfcn=1,
        values=[0.0],
        errors=[0.0],
    )
    assert graded.Status() == 3
    graded._report(["a"])
    assert "Invalid minimum - status = 3" in capsys.readouterr().out


def test_a_histogram_fills_from_arrays_as_pyroot_lets_it():
    h = ROOT.TH1D("h", "", 4, 0, 4)
    expect(
        (h.Fill(np.array([0.5, 1.5, 1.5])), -1),
        (h.GetBinContent(2), 2),
    )
    h.Fill(np.array([2.5]), np.array([3.0]))
    expect(
        (h.GetBinContent(3), 3),
        (h.GetEntries(), 4),
    )
    h2 = ROOT.TH2D("h2", "", 2, 0, 2, 2, 0, 2)
    h2.Fill(np.array([0.5, 1.5]), np.array([0.5, 0.5]))
    expect(
        (h2.GetBinContent(1, 1), 1),
        (h2.GetBinContent(2, 1), 1),
    )


def test_a_file_that_would_not_open_lists_nothing(capsys):
    ROOT.TFile("missing.root").ls()
    assert capsys.readouterr().out == ""


def test_std_streams_write_what_they_are_given(capsys):
    ROOT.std.cout << "a" << 1 << ROOT.std.endl
    ROOT.std.cerr << "b"
    ROOT.std.cout.flush()
    out = capsys.readouterr()
    expect(
        (out.out, "a1\n"),
        (out.err, "b"),
        (repr(ROOT.std.cout), "<std::cout>"),
    )


def test_histograms_take_roots_arguments_by_name():
    h = ROOT.TH1D("h", "t", nbinsx=20, xlow=0.0, xup=10.0)
    expect(
        (h.GetNbinsX(), 20),
        (h.GetXaxis().GetXmax(), 10),
    )
    v = ROOT.TH1D(name="v", title="t", nbinsx=3, xbins=[0.0, 1.0, 2.0, 5.0])
    assert v.GetXaxis().GetXmax() == 5
    two = ROOT.TH2F("two", "t", 10, 0, 1, nbinsy=4, ylow=-1, yup=1)
    expect(
        (two.GetNbinsY(), 4),
        (two.GetYaxis().GetXmin(), -1),
    )
    with pytest.raises(TypeError, match="no argument 'nbins'"):
        ROOT.TH1D("bad", "t", nbins=3)


def test_a_formula_may_use_a_formula_made_before_it():
    ROOT.TFormula("shape", "x*x")
    ROOT.TFormula("scaled", "[0]*x")
    uses = ROOT.TF1("uses", "[0] + [1]*shape + scaled", 0, 1)
    uses.SetParameters(1, 2, 3)
    expect(
        (uses.Eval(0.5), pytest.approx(1 + 2 * 0.25 + 3 * 0.5)),
        (uses.GetNpar(), 3),
        (ROOT.TF1("plain", "gaus", -1, 1).GetNpar(), 3),
    )


def test_a_file_hands_out_objects_by_subscript(tmp_path):
    with ROOT.TFile("s.root", "RECREATE") as out:
        ROOT.TH1D("h", "", 1, 0, 1)
        out.Write()
    with ROOT.TFile("s.root") as back:
        assert back["h"].GetNbinsX() == 1
        with pytest.raises(KeyError, match="has no object 'nope'"):
            back["nope"]


def test_a_fitter_set_up_first_minimises_from_its_parameters_settings():
    pytest.importorskip("iminuit")

    def bowl(xx):
        return (xx[0] - 3) ** 2 + 4 * (xx[1] + 1) ** 2 + 1

    fitter = ROOT.Fit.Fitter()
    with pytest.raises(ValueError, match="none was given"):
        fitter.FitFCN()
    fitter.SetFCN(ROOT.Math.Functor(bowl, 2), np.array([1.0, 1.0]))
    config = fitter.Config()
    settings = config.ParSettings(0)
    settings.SetStepSize(0.01)
    settings.SetName("a")
    settings.SetValue(2.0)
    config.ParSettings(1).SetLimits(-5, 5)
    assert (settings.StepSize(), settings.Name(), settings.Value(), config.NPar()) == (
        0.01, "a", 2.0, 2)
    assert fitter.FitFCN()
    assert fitter.Result().Parameter(0) == pytest.approx(3.0, abs=1e-2)  # MIGRAD's edm 0.01
    assert fitter.Result().ParName(0) == "a"
    config.ParSettings(0).Fix()
    assert config.ParamsSettings()[0].IsFixed()
    fitter.FitFCN()
    assert fitter.Result().Parameter(0) == pytest.approx(2.0)
    config.ParSettings(0).Release()
    assert not config.ParSettings(0).IsFixed()
    config.SetParamsSettings(2)
    assert config.ParSettings(1).Value() == 0.0
