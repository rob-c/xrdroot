"""PyROOT's ``TF1(name, f, xmin, xmax, npar=..., ndim=...)``: C++'s trailing arguments by name."""

from __future__ import annotations

import xrdroot.pyroot as ROOT


def line(x, p):
    return p[0] + p[1] * x[0]


def test_a_python_function_takes_its_parameter_count_by_keyword():
    made = ROOT.TF1("keyword_line", line, 0, 1, npar=2, ndim=1)
    made.SetParameters(1.0, 2.0)
    assert made.GetNpar() == 2 and made.Eval(0.5) == 2.0
    assert ROOT.TF1("keyword_none", line, 0, 1, ndim=1).GetNpar() == 0
    assert ROOT.TF1("keyword_only", line, 0, 1, npar=2).GetNpar() == 2


def spread(x, p):
    return p[0] * x[0] * x[2] + x[1] * p[1]


def test_a_python_function_is_handed_every_coordinate_as_roots_pointer_hands_them():
    made = ROOT.TF1("every_coordinate", spread, -10, 10, 2)
    made.SetParameters(2.0, 3.0)
    assert made.Eval(1.0, 2.0, 3.0) == 2.0 * 3.0 + 2.0 * 3.0
    rows = [[2.0, 2.0, 1.0], [1.0, 2.0, 3.0]]
    assert made.EvalPar(rows, [2.0, 3.0]).tolist() == [10.0, 12.0]
    assert made.EvalPar([1.0, 2.0, 3.0]) == 12.0


def test_a_function_of_x_alone_or_normalised_is_evaluated_as_before():
    alone = ROOT.TF1("x_alone", lambda x: x[0] * 2, 0, 1)
    assert alone.Eval(0.25) == 0.5 and list(alone.Eval([0.25, 0.5])) == [0.5, 1.0]
    normalised = ROOT.TF1("normalised_line", line, 0, 1, 2)
    normalised.SetParameters(1.0, 0.0)
    normalised.SetNormalized(True)
    assert abs(normalised.EvalPar([0.5]) - 1.0) < 1e-9
