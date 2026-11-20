"""``ROOT.Math``'s ``Polynomial``, its ``IGenFunction`` interface and the 1-D minimiser."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT

Math = ROOT.Math


def test_a_polynomial_is_its_coefficients_constant_first():
    poly = Math.Polynomial(2)
    poly.SetParameters([1, -4, 1])
    assert (poly.Order(), poly.Parameters()) == (2, [1.0, -4.0, 1.0])
    # 1 - 4*2 + 2**2, and the derivative -4 + 2x there
    assert (poly(2), poly.DoEval(2), poly.Derivative(2)) == (-3.0, -3.0, 0.0)
    assert poly.Derivative(3) == 2.0


def test_the_coefficients_may_be_given_in_turn_or_to_the_constructor():
    in_turn = Math.Polynomial(1)
    in_turn.SetParameters(2.0, 3.0)
    built = Math.Polynomial(1, 2.0, 3.0)
    assert (in_turn(5), built(5)) == (17.0, 17.0)


def test_a_polynomial_refuses_a_wrong_number_of_coefficients():
    with pytest.raises(ValueError, match="order 2 takes 3 coefficients and was given 2"):
        Math.Polynomial(2).SetParameters([1, 2])


def test_the_interface_is_what_a_subclass_overrides():
    class Square(Math.IGenFunction):
        def DoEval(self, x):
            return x * x

    square = Square()
    assert (square(3), square.NDim()) == (9.0, 1)
    with pytest.raises(NotImplementedError, match="derive from it and override DoEval"):
        Math.IGenFunction().DoEval(1)


@pytest.mark.parametrize(
    ("kind", "name"),
    [(Math.Minim1D.kBRENT, "brent"), (Math.Minim1D.kGOLDENSECTION, "goldensection")],
)
def test_both_kinds_of_search_reach_the_same_minimum(kind, name):
    poly = Math.Polynomial(2, 1.0, -4.0, 1.0)
    minimizer = Math.GSLMinimizer1D(kind)
    assert minimizer.SetFunction(poly, 1, -10, 10) is True
    assert minimizer.Minimize(100, 0.01, 0.01) == 0
    assert minimizer.Status() == 0
    assert (minimizer.Type(), minimizer.Name()) == (kind, name)
    assert minimizer.XMinimum() == pytest.approx(2.0, abs=0.01)
    assert minimizer.FValMinimum() == pytest.approx(-3.0, abs=1e-4)
    assert minimizer.Iterations() > 0


def test_the_default_kind_is_brents():
    minimizer = Math.GSLMinimizer1D()
    assert (minimizer.Type(), minimizer.Name()) == (Math.Minim1D.kBRENT, "brent")


def test_a_search_out_of_iterations_says_so_and_stops_at_its_first_point():
    minimizer = Math.GSLMinimizer1D()
    minimizer.SetFunction(Math.Functor1D(lambda x: (x - 3) ** 2), 0, 10)
    assert minimizer.Minimize(0, 0.01, 0.01) == 1
    assert (minimizer.Status(), minimizer.Iterations()) == (1, 0)
    assert minimizer.XMinimum() == pytest.approx(3.819660112501052)


def test_the_search_moves_the_side_it_finds_the_higher_value_on():
    # a least point left of the first point tried: the right end comes in
    left = Math.GSLMinimizer1D()
    left.SetFunction(Math.Functor1D(lambda x: (x + 5) ** 2), -10, 10)
    assert left.Minimize(100, 1e-8, 1e-8) == 0
    assert left.XMinimum() == pytest.approx(-5.0, abs=1e-6)
    # a least point at the first point tried: both ends come in
    middle = Math.GSLMinimizer1D()
    middle.SetFunction(Math.Functor1D(lambda x: (x + 2.360679774997897) ** 2), -10, 10)
    assert middle.Minimize(100, 1e-8, 1e-8) == 0
    assert middle.XMinimum() == pytest.approx(-2.360679774997897, abs=1e-6)
    assert middle.FValMinimum() == 0.0


def test_a_bracket_of_fewer_than_two_numbers_is_refused():
    with pytest.raises(ValueError, match="give SetFunction\\(f, low, high\\)"):
        Math.GSLMinimizer1D().SetFunction(Math.Functor1D(lambda x: x * x), 1.0)
