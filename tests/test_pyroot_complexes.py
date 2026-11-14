"""``TComplex``: ROOT's complex numbers, their arithmetic either way round, and their print."""

from __future__ import annotations

import math

import pytest

import xrdroot.pyroot as ROOT


def test_a_complex_number_adds_subtracts_multiplies_and_divides_with_a_real_either_side():
    c, d = ROOT.TComplex(4.0, 0), 2.0
    assert (d + c).Re() == 6.0 and (c + d).Re() == 6.0
    assert (d - c).Re() == -2.0 and (c - d).Re() == 2.0
    assert (d * c).Re() == 8.0 and (c * d).Re() == 8.0
    assert (d / c).Re() == 0.5 and (c / d).Re() == 2.0


def test_a_complex_number_is_no_operand_for_a_string():
    with pytest.raises(TypeError):
        "string" / ROOT.TComplex(4.0, 0)


def test_a_complex_number_prints_and_takes_apart_as_roots_does(capsys):
    d = ROOT.TComplex(1, 2)
    assert str(d) == "(1,2i)" and str(-d) == "(-1,-2i)"
    assert (d.Im(), d.Rho(), d.Rho2()) == (2.0, math.sqrt(5), pytest.approx(5.0))
    assert d.Theta() == math.atan2(2, 1)
    assert ROOT.TComplex().Theta() == 0.0
    assert complex(d) == 1 + 2j and d == ROOT.TComplex(d) and hash(d) == hash(1 + 2j)
    assert d != "d" and ROOT.TComplex(3) == 3
    d.Print()
    assert capsys.readouterr().out == "(1,2i)\n"


def test_complex_functions_are_roots_static_ones():
    d = ROOT.TComplex(1, 2)
    assert str(ROOT.TComplex.Conjugate(d)) == "(1,-2i)"
    assert str(ROOT.TComplex.Sqrt(d)) == "(1.27202,0.786151i)"
    assert ROOT.TComplex.Abs(d) == math.sqrt(5)
    assert str(ROOT.TComplex.I()) == "(0,1i)" and str(ROOT.TComplex.One()) == "(1,0i)"
    assert ROOT.TComplex.Log(ROOT.TComplex.Exp(d)) == d
    assert ROOT.TComplex.Power(d, 2) == ROOT.TComplex(-3, 4)
    assert str(ROOT.TComplex(1, math.pi / 2, True)) == "(6.12323e-17,1i)"
