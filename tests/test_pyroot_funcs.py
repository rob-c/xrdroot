"""``TF1``, ``TF2``, ``TF3``, ``TFormula`` and what a fit hands back."""

from __future__ import annotations

import array
import ctypes
import math

import numpy as np
import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot.pyroot.core import funcs


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_a_formula_function_is_put_in_groots_list_replacing_its_name():
    first = ROOT.TF1("f", "[0]*x + [1]", 0, 10)
    second = ROOT.TF1("f", "x*x", 0, 10)
    expect(
        (bool(ROOT.gROOT.GetFunction("f") is second), True),
        (ROOT.gROOT.GetListOfFunctions().GetSize(), 1),
        (first.GetName(), "f"),
        (ROOT.TF1().GetNpar(), 0),
        (ROOT.TF1("g").GetXmax(), 1),
    )
    copy = ROOT.TF1(second)
    expect(
        (bool(copy is not second), True),
        (copy.Eval(3), 9),
        (bool(ROOT.gROOT.GetFunction("f") is copy), True),
    )


def test_parameters_are_set_named_limited_and_fixed():
    f = ROOT.TF1("f", "[0] + [1]*x + [2]*x*x", 0, 1)
    f.SetParameters(1, 2, 3)
    expect(
        (list(f.GetParameters()), [1, 2, 3]),
        (f.GetParameter(1), 2),
    )
    f.SetParameters(np.array([4.0, 5.0]))
    f.SetParameter(2, 6)
    held = np.zeros(3)
    expect(
        (list(f.GetParameters(held)), [4, 5, 6]),
        (list(held), [4, 5, 6]),
    )
    f.SetParNames("a", "b", "c")
    f.SetParName(0, "offset")
    expect(
        (f.GetParName(0), "offset"),
        (f.GetParNumber("b"), 1),
        (f.GetParNumber("zz"), -1),
        (f.GetParameter("b"), 5),
    )
    f.SetParameter("c", 7.0)
    assert f.GetParameter(2) == 7
    f.SetParError(0, 0.5)
    f.SetParErrors([0.1, 0.2, 0.3])
    expect(
        (f.GetParError(1), 0.2),
        (list(f.GetParErrors()), [0.1, 0.2, 0.3]),
    )
    f.SetParLimits(1, 0, 10)
    low, high = ctypes.c_double(0), ctypes.c_double(0)
    expect(
        (f.GetParLimits(1, low, high), (0, 10)),
        ((low.value, high.value), (0, 10)),
    )
    f.FixParameter(2, 1.5)
    expect(
        (bool(f.IsFixed(2)), True),
        (f.GetParameter(2), 1.5),
        (f.GetNumberFreeParameters(), 2),
    )
    f.ReleaseParameter(2)
    assert not f.IsFixed(2)


def test_a_function_is_evaluated_integrated_and_searched_as_roots_is():
    f = ROOT.TF1("f", "x*x", 0, 2)
    expect(
        (f.Eval(3), 9),
        (f(2), 4),
        (list(f.Eval(np.array([1.0, 2.0]))), [1, 4]),
        (f.EvalPar(array.array("d", [3.0])), 9),
        (f.EvalPar([2.0], [0.0]), 4),
        (f.Integral(0, 1), pytest.approx(1 / 3)),
        (f.Derivative(1.0), pytest.approx(2)),
        (f.GetMaximum(), pytest.approx(4)),
        (f.GetMinimum(), pytest.approx(0, abs=1e-8)),
        (f.GetMaximumX(), pytest.approx(2)),
        (f.GetMinimumX(), pytest.approx(0, abs=1e-4)),
        (f.GetX(1.0), pytest.approx(1.0, rel=1e-6)),
        (f.Mean(0, 2), pytest.approx(1.5)),
        (f.Variance(0, 2), pytest.approx(0.15)),
        (f.Moment(2, 0, 2), pytest.approx(2.4)),
        (f.CentralMoment(1, 0, 2), pytest.approx(0)),
    )
    zero = ROOT.TF1("z", "0*x", 0, 1)
    expect(
        (zero.Moment(1, 0, 1), 0),
        (zero.CentralMoment(2, 0, 1), 0),
    )
    g = ROOT.TF1("g", "[0]*x", 0, 1)
    assert g.Derivative(0.5, [3.0]) == pytest.approx(3)


def test_the_range_the_grid_and_the_fit_record():
    f = ROOT.TF1("f", "x", -1, 1)
    f.SetRange(0, 5)
    lo, hi = ctypes.c_double(0), ctypes.c_double(0)
    expect(
        (f.GetRange(lo, hi), (0, 5)),
        ((lo.value, hi.value), (0, 5)),
        ((f.GetXmin(), f.GetXmax()), (0, 5)),
    )
    f.SetNpx(500)
    expect(
        (f.GetNpx(), 500),
        (f.GetChisquare(), 0),
        (f.GetNDF(), 0),
        (f.GetNumberFitPoints(), 0),
    )
    f.SetChisquare(4.0)
    f.SetNDF(2)
    expect(
        (f.GetChisquare(), 4),
        (f.GetNDF(), 2),
        (f.GetProb(), pytest.approx(math.exp(-2))),
    )
    f.SetNormalized(True)
    f.Update()
    expect(
        (f.EvalPar([2.0]), pytest.approx(2 / 12.5)),
        (f.Eval(2), 2),
        (bool(f.IsValid()), True),
    )


def test_a_python_function_is_called_as_pyroot_calls_it(capsys):
    def line(x, p):
        return p[0] + p[1] * x[0]

    f = ROOT.TF1("line", line, 0, 10, 2)
    f.SetParameters(1, 2)
    expect(
        (f.Eval(3), 7),
        (f(3), 7),
        (f.GetNpar(), 2),
        (f.Integral(0, 1), pytest.approx(2)),
    )
    alone = ROOT.TF1("sq", lambda x: x[0] ** 2, 0, 3)
    expect(
        (alone.Eval(3), 9),
        (alone.GetNpar(), 0),
    )

    class Shape:
        def __call__(self, x, p):
            return 2 * x[0]

    assert ROOT.TF1("c", Shape(), 0, 1, 0).Eval(0.5) == 1
    f.Print("V")
    out = capsys.readouterr().out.splitlines()
    expect(
        (out[0], "Compiled based function: line  based on a functor object.  Ndim = 1, Npar = 2"),
        (out[1], "List of  Parameters: "),
        (bool(out[2].strip().startswith("p0 =")), True),
    )


def test_formulas_print_as_roots_do(capsys):
    ROOT.TF1("g", "gaus", -1, 1).Print("V")
    lines = capsys.readouterr().out.splitlines()
    expect(
        (
            lines[:4],
            [
                "Formula based function:     g ",
                "                    g : gaus Ndim= 1, Npar= 3, Number= 100 ",
                " Formula expression: ",
                "\t[Constant]*exp(-0.5*((x-[Mean])/[Sigma])*((x-[Mean])/[Sigma])) ",
            ],
        ),
        (bool(lines[5].startswith("Par   0             Constant =  ")), True),
    )


def test_a_tformula_stands_alone():
    formula = ROOT.TFormula("form", "x+[0]")
    formula.SetParameter(0, 2)
    expect(
        (formula.Eval(1), 3),
        (formula.GetNdim(), 1),
        (bool(formula.GetFormula() is formula), True),
    )
    formula.SetName("renamed")
    formula.SetTitle("t")
    assert (formula.GetName(), formula.GetTitle()) == ("renamed", "t")
    formula.SetBit(4)
    expect(
        (bool(formula.TestBit(4)), True),
        (ROOT.TFormula().Eval(1), 0),
        (formula.GetExpFormula(), "x+[p0]"),
    )


def test_copies_are_not_listed_and_draw_copies_draw(monkeypatch):
    f = ROOT.TF1("f", "x", 0, 1)
    copy = f.Clone("g")
    expect(
        (copy.GetName(), "g"),
        (bool(ROOT.gROOT.GetFunction("g") is None), True),
        (f.Clone().GetName(), "f"),
    )
    other = ROOT.TF1("h", "2*x", 0, 1)
    f.Copy(other)
    expect(
        (other.Eval(1), 1),
        (f.DrawCopy("same").Eval(1), 1),
    )
    shown = f.GetHistogram()
    expect(
        (shown.GetNbinsX(), 100),
        (shown.GetBinContent(1), pytest.approx(0.005)),
    )
    f.SetLineColor(ROOT.kBlue)
    assert f._xrd._f1["TAttLine"]["fLineColor"] == ROOT.kBlue


def test_functions_of_two_and_three_variables():
    f2 = ROOT.TF2("f2", "x*y", 0, 1, 0, 2)
    expect(
        (f2.Eval(0.5, 2), 1),
        (f2(np.array([0.5, 2.0])), 1),
        (f2.GetNdim(), 2),
        (f2.Integral(0, 1, 0, 2), pytest.approx(1.0)),
        ((f2.GetYmin(), f2.GetYmax()), (0, 2)),
    )
    f2.SetRange(0, 0, 1, 1)
    assert f2.GetRange() == (0, 0, 1, 1)
    f2.SetNpx(4)
    f2.SetNpy(4)
    assert f2.GetNpy() == 4
    x, y = ctypes.c_double(0), ctypes.c_double(0)
    px, py = f2.GetRandom2(x, y, ROOT.TRandom3(1))
    expect(
        (bool(0 <= px <= 1), True),
        (bool(0 <= py <= 1), True),
        (x.value, px),
    )
    f3 = ROOT.TF3("f3", "x+y+z", 0, 1, 0, 1, 0, 1)
    f3.SetNpz(5)
    expect(
        (f3.Eval(1, 1, 1), 3),
        (f3.Integral(0, 1, 0, 1, 0, 1), pytest.approx(1.5)),
        (f3.GetNpz(), 5),
    )


def test_random_numbers_from_a_function_are_groots():
    f = ROOT.TF1("f", "x", 0, 1)
    ROOT.gRandom.SetSeed(3)
    first = f.GetRandom()
    ROOT.gRandom.SetSeed(3)
    expect(
        (f.GetRandom(), first),
        (bool(0 <= first <= 1), True),
        (bool(0.5 <= f.GetRandom(0.5, 1.0) <= 1), True),
        (bool(0 <= f.GetRandom(ROOT.TRandom3(1)) <= 1), True),
    )


def test_standard_functions_are_made_when_first_asked():
    expect(
        (bool(funcs.standard_function("nothing") is None), True),
        (ROOT.gROOT.GetFunction("expo").GetNpar(), 2),
        (funcs._arguments(print), 2),
        (funcs._arguments(lambda *a: 0), 2),
    )


def test_a_formula_is_written_as_a_tformula_and_a_function_using_it_as_root_expands_it(tmp_path):
    # hist002_TH1_fillrandom_userfunc: ROOT 6.40's file holds a TFormula and this TF1's text.
    form1 = ROOT.TFormula("form1", "abs(sin(x)/x)")
    sqroot = ROOT.TF1("sqroot", "x*gaus(0) + [3]*form1", 0.0, 10.0)
    assert form1.ClassName() == "TFormula"
    path = str(tmp_path / "formula.root")
    out = ROOT.TFile.Open(path, "RECREATE")
    out.WriteObject(form1, form1.GetName())
    out.WriteObject(sqroot, sqroot.GetName())
    out.Close()
    with xrdroot.open_root(path) as back:
        assert back.key("form1").classname == "TFormula"
        assert back["form1"].formula == "abs(sin(x)/x)"
        assert back["sqroot"].formula == (
            "x*[p0]*exp(-0.5*((x-[p1])/[p2])*((x-[p1])/[p2]))+[p3]*(abs(sin(x)/x))"
        )
