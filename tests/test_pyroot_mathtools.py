"""``ROOT.Math``'s functors and tools, ``ROOT.Fit.Fitter``, and names given by keyword."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.pyroot.core import mathtools


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_functors_wrap_python_functions():
    one = ROOT.Math.Functor1D(lambda x: x * x - 1)
    many = ROOT.Math.Functor(lambda xx: xx[0] + xx[1], 2)
    assert one(3) == 8 and one.NDim() == 1 and many([1, 2]) == 3 and many.NDim() == 2
    grad = ROOT.Math.GradFunctor1D(lambda x: x * x, lambda x: 2 * x)
    assert grad.Derivative(3) == 6
    grad2 = ROOT.Math.GradFunctor(lambda xx: xx[0] * xx[1], lambda xx, i: xx[1 - i], 2)
    assert grad2.Derivative([2, 3], 0) == 3


def test_the_integrator_is_gsls_rule_first():
    integrator = ROOT.Math.Integrator()
    integrator.SetFunction(ROOT.Math.Functor1D(lambda x: x * x - 1))
    assert integrator.Integral(0, 3) == 5.999999999999999 and integrator.Status() == 0
    assert integrator.Error() < 1e-12
    wiggly = ROOT.Math.IntegratorOneDim(ROOT.Math.Functor1D(lambda x: np.sqrt(abs(x))))
    assert wiggly.Integral(-1, 1) == pytest.approx(4 / 3, rel=1e-8)
    assert mathtools.kronrod(lambda x: 1.0, 0.0, 2.0)[0] == pytest.approx(2.0)


def test_the_root_finder_brackets_or_follows_a_derivative():
    finder = ROOT.Math.RootFinder(ROOT.Math.RootFinder.kGSL_NEWTON)
    finder.SetFunction(ROOT.Math.GradFunctor1D(lambda x: x * x - 1, lambda x: 2 * x), 3)
    assert finder.Solve() and finder.Root() == 1.0
    bracket = ROOT.Math.RootFinder()
    bracket.SetFunction(ROOT.Math.Functor1D(lambda x: x - 0.25), 0, 1)
    assert bracket.Solve() and bracket.Root() == pytest.approx(0.25)


def test_the_fitter_minimises_and_prints_as_root_does(capsys):
    def rosenbrock(xx):
        return 100 * (xx[1] - xx[0] ** 2) ** 2 + (1 - xx[0]) ** 2

    fitter = ROOT.Fit.Fitter()
    assert fitter.FitFCN(ROOT.Math.Functor(rosenbrock, 2), np.zeros(2))
    fitter.Result().Print(ROOT.std.cout)
    out = capsys.readouterr().out
    assert "MinFCN" in out and "Chi2 " not in out and "Par_1" in out
    assert fitter.Result().Parameter(0) == pytest.approx(1.0, abs=1e-3)

    def gradient(xx, i):
        if i == 0:
            return 2 * (200 * xx[0] ** 3 - 200 * xx[0] * xx[1] + xx[0] - 1)
        return 200 * (xx[1] - xx[0] ** 2)

    graded = ROOT.Fit.Fitter()
    graded.FitFCN(ROOT.Math.GradFunctor(rosenbrock, gradient, 2))
    assert graded.Result().Parameter(1) == pytest.approx(1.0, abs=1e-3)


def test_std_streams_write_what_they_are_given(capsys):
    ROOT.std.cout << "a" << 1 << ROOT.std.endl
    ROOT.std.cerr << "b"
    ROOT.std.cout.flush()
    out = capsys.readouterr()
    assert out.out == "a1\n" and out.err == "b" and repr(ROOT.std.cout) == "<std::cout>"


def test_histograms_take_roots_arguments_by_name():
    h = ROOT.TH1D("h", "t", nbinsx=20, xlow=0.0, xup=10.0)
    assert h.GetNbinsX() == 20 and h.GetXaxis().GetXmax() == 10
    v = ROOT.TH1D(name="v", title="t", nbinsx=3, xbins=[0.0, 1.0, 2.0, 5.0])
    assert v.GetXaxis().GetXmax() == 5
    two = ROOT.TH2F("two", "t", 10, 0, 1, nbinsy=4, ylow=-1, yup=1)
    assert two.GetNbinsY() == 4 and two.GetYaxis().GetXmin() == -1
    with pytest.raises(TypeError, match="no argument 'nbins'"):
        ROOT.TH1D("bad", "t", nbins=3)


def test_a_formula_may_use_a_formula_made_before_it():
    ROOT.TFormula("shape", "x*x")
    ROOT.TFormula("scaled", "[0]*x")
    uses = ROOT.TF1("uses", "[0] + [1]*shape + scaled", 0, 1)
    uses.SetParameters(1, 2, 3)
    assert uses.Eval(0.5) == pytest.approx(1 + 2 * 0.25 + 3 * 0.5) and uses.GetNpar() == 3
    assert ROOT.TF1("plain", "gaus", -1, 1).GetNpar() == 3


def test_a_file_hands_out_objects_by_subscript(tmp_path):
    with ROOT.TFile("s.root", "RECREATE") as out:
        ROOT.TH1D("h", "", 1, 0, 1)
        out.Write()
    with ROOT.TFile("s.root") as back:
        assert back["h"].GetNbinsX() == 1
        with pytest.raises(KeyError, match="has no object 'nope'"):
            back["nope"]
