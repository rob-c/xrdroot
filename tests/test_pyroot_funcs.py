"""``TF1``, ``TF2``, ``TF3``, ``TFormula`` and what a fit hands back."""

from __future__ import annotations

import array
import ctypes
import math

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.pyroot.core import funcs


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_a_formula_function_is_put_in_groots_list_replacing_its_name():
    first = ROOT.TF1("f", "[0]*x + [1]", 0, 10)
    second = ROOT.TF1("f", "x*x", 0, 10)
    assert ROOT.gROOT.GetFunction("f") is second
    assert ROOT.gROOT.GetListOfFunctions().GetSize() == 1
    assert first.GetName() == "f"
    assert ROOT.TF1().GetNpar() == 0
    assert ROOT.TF1("g").GetXmax() == 1
    copy = ROOT.TF1(second)
    assert copy is not second
    assert copy.Eval(3) == 9
    assert ROOT.gROOT.GetFunction("f") is copy


def test_parameters_are_set_named_limited_and_fixed():
    f = ROOT.TF1("f", "[0] + [1]*x + [2]*x*x", 0, 1)
    f.SetParameters(1, 2, 3)
    assert list(f.GetParameters()) == [1, 2, 3]
    assert f.GetParameter(1) == 2
    f.SetParameters(np.array([4.0, 5.0]))
    f.SetParameter(2, 6)
    held = np.zeros(3)
    assert list(f.GetParameters(held)) == [4, 5, 6]
    assert list(held) == [4, 5, 6]
    f.SetParNames("a", "b", "c")
    f.SetParName(0, "offset")
    assert f.GetParName(0) == "offset"
    assert f.GetParNumber("b") == 1
    assert f.GetParNumber("zz") == -1
    assert f.GetParameter("b") == 5
    f.SetParameter("c", 7.0)
    assert f.GetParameter(2) == 7
    f.SetParError(0, 0.5)
    f.SetParErrors([0.1, 0.2, 0.3])
    assert f.GetParError(1) == 0.2
    assert list(f.GetParErrors()) == [0.1, 0.2, 0.3]
    f.SetParLimits(1, 0, 10)
    low, high = ctypes.c_double(0), ctypes.c_double(0)
    assert f.GetParLimits(1, low, high) == (0, 10)
    assert (low.value, high.value) == (0, 10)
    f.FixParameter(2, 1.5)
    assert f.IsFixed(2)
    assert f.GetParameter(2) == 1.5
    assert f.GetNumberFreeParameters() == 2
    f.ReleaseParameter(2)
    assert not f.IsFixed(2)


def test_a_function_is_evaluated_integrated_and_searched_as_roots_is():
    f = ROOT.TF1("f", "x*x", 0, 2)
    assert f.Eval(3) == 9
    assert f(2) == 4
    assert list(f.Eval(np.array([1.0, 2.0]))) == [1, 4]
    assert f.EvalPar(array.array("d", [3.0])) == 9
    assert f.EvalPar([2.0], [0.0]) == 4
    assert f.Integral(0, 1) == pytest.approx(1 / 3)
    assert f.Derivative(1.0) == pytest.approx(2)
    assert f.GetMaximum() == pytest.approx(4)
    assert f.GetMinimum() == pytest.approx(0, abs=1e-8)
    assert f.GetMaximumX() == pytest.approx(2)
    assert f.GetMinimumX() == pytest.approx(0, abs=1e-4)
    assert f.GetX(1.0) == pytest.approx(1.0, rel=1e-6)
    assert f.Mean(0, 2) == pytest.approx(1.5)
    assert f.Variance(0, 2) == pytest.approx(0.15)
    assert f.Moment(2, 0, 2) == pytest.approx(2.4)
    assert f.CentralMoment(1, 0, 2) == pytest.approx(0)
    zero = ROOT.TF1("z", "0*x", 0, 1)
    assert zero.Moment(1, 0, 1) == 0
    assert zero.CentralMoment(2, 0, 1) == 0
    g = ROOT.TF1("g", "[0]*x", 0, 1)
    assert g.Derivative(0.5, [3.0]) == pytest.approx(3)


def test_the_range_the_grid_and_the_fit_record():
    f = ROOT.TF1("f", "x", -1, 1)
    f.SetRange(0, 5)
    lo, hi = ctypes.c_double(0), ctypes.c_double(0)
    assert f.GetRange(lo, hi) == (0, 5)
    assert (lo.value, hi.value) == (0, 5)
    assert (f.GetXmin(), f.GetXmax()) == (0, 5)
    f.SetNpx(500)
    assert f.GetNpx() == 500
    assert f.GetChisquare() == 0
    assert f.GetNDF() == 0
    assert f.GetNumberFitPoints() == 0
    f.SetChisquare(4.0)
    f.SetNDF(2)
    assert f.GetChisquare() == 4
    assert f.GetNDF() == 2
    assert f.GetProb() == pytest.approx(math.exp(-2))
    f.SetNormalized(True)
    f.Update()
    assert f.EvalPar([2.0]) == pytest.approx(2 / 12.5)
    assert f.Eval(2) == 2
    assert f.IsValid()


def test_a_python_function_is_called_as_pyroot_calls_it(capsys):
    def line(x, p):
        return p[0] + p[1] * x[0]

    f = ROOT.TF1("line", line, 0, 10, 2)
    f.SetParameters(1, 2)
    assert f.Eval(3) == 7
    assert f(3) == 7
    assert f.GetNpar() == 2
    assert f.Integral(0, 1) == pytest.approx(2)
    alone = ROOT.TF1("sq", lambda x: x[0] ** 2, 0, 3)
    assert alone.Eval(3) == 9
    assert alone.GetNpar() == 0

    class Shape:
        def __call__(self, x, p):
            return 2 * x[0]

    assert ROOT.TF1("c", Shape(), 0, 1, 0).Eval(0.5) == 1
    f.Print("V")
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "Compiled based function: line  based on a functor object.  Ndim = 1, Npar = 2"
    assert out[1] == "List of  Parameters: "
    assert out[2].strip().startswith("p0 =")


def test_formulas_print_as_roots_do(capsys):
    ROOT.TF1("g", "gaus", -1, 1).Print("V")
    lines = capsys.readouterr().out.splitlines()
    assert lines[:4] == [
        "Formula based function:     g ",
        "                    g : gaus Ndim= 1, Npar= 3, Number= 100 ",
        " Formula expression: ",
        "\t[Constant]*exp(-0.5*((x-[Mean])/[Sigma])*((x-[Mean])/[Sigma])) ",
    ]
    assert lines[5].startswith("Par   0             Constant =  ")


def test_a_tformula_stands_alone():
    formula = ROOT.TFormula("form", "x+[0]")
    formula.SetParameter(0, 2)
    assert formula.Eval(1) == 3
    assert formula.GetNdim() == 1
    assert formula.GetFormula() is formula
    formula.SetName("renamed")
    formula.SetTitle("t")
    assert (formula.GetName(), formula.GetTitle()) == ("renamed", "t")
    formula.SetBit(4)
    assert formula.TestBit(4)
    assert ROOT.TFormula().Eval(1) == 0
    assert formula.GetExpFormula() == "x+[p0]"


def test_copies_are_not_listed_and_draw_copies_draw(monkeypatch):
    f = ROOT.TF1("f", "x", 0, 1)
    copy = f.Clone("g")
    assert copy.GetName() == "g"
    assert ROOT.gROOT.GetFunction("g") is None
    assert f.Clone().GetName() == "f"
    other = ROOT.TF1("h", "2*x", 0, 1)
    f.Copy(other)
    assert other.Eval(1) == 1
    assert f.DrawCopy("same").Eval(1) == 1
    shown = f.GetHistogram()
    assert shown.GetNbinsX() == 100
    assert shown.GetBinContent(1) == pytest.approx(0.005)
    f.SetLineColor(ROOT.kBlue)
    assert f._xrd._f1["TAttLine"]["fLineColor"] == ROOT.kBlue


def test_functions_of_two_and_three_variables():
    f2 = ROOT.TF2("f2", "x*y", 0, 1, 0, 2)
    assert f2.Eval(0.5, 2) == 1
    assert f2(np.array([0.5, 2.0])) == 1
    assert f2.GetNdim() == 2
    assert f2.Integral(0, 1, 0, 2) == pytest.approx(1.0)
    assert (f2.GetYmin(), f2.GetYmax()) == (0, 2)
    f2.SetRange(0, 0, 1, 1)
    assert f2.GetRange() == (0, 0, 1, 1)
    f2.SetNpx(4)
    f2.SetNpy(4)
    assert f2.GetNpy() == 4
    x, y = ctypes.c_double(0), ctypes.c_double(0)
    px, py = f2.GetRandom2(x, y, ROOT.TRandom3(1))
    assert 0 <= px <= 1
    assert 0 <= py <= 1
    assert x.value == px
    f3 = ROOT.TF3("f3", "x+y+z", 0, 1, 0, 1, 0, 1)
    f3.SetNpz(5)
    assert f3.Eval(1, 1, 1) == 3
    assert f3.Integral(0, 1, 0, 1, 0, 1) == pytest.approx(1.5)
    assert f3.GetNpz() == 5


def test_random_numbers_from_a_function_are_groots():
    f = ROOT.TF1("f", "x", 0, 1)
    ROOT.gRandom.SetSeed(3)
    first = f.GetRandom()
    ROOT.gRandom.SetSeed(3)
    assert f.GetRandom() == first
    assert 0 <= first <= 1
    assert 0.5 <= f.GetRandom(0.5, 1.0) <= 1
    assert 0 <= f.GetRandom(ROOT.TRandom3(1)) <= 1


def test_standard_functions_are_made_when_first_asked():
    assert funcs.standard_function("nothing") is None
    assert ROOT.gROOT.GetFunction("expo").GetNpar() == 2
    assert funcs._arguments(print) == 2
    assert funcs._arguments(lambda *a: 0) == 2
