"""``ROOT::Math``'s function interfaces, which a macro's own function derives from.

What a translated ``class F : public ROOT::Math::IMultiGenFunction`` becomes is a
subclass of one of these, and the numerical tools - Minuit, the genetic search,
the integrators - call ``operator()``, ask for ``NDim`` and ``Gradient``, and read
and write the parameters through them. Here each of ROOT's aliases is held to the
class behind it, and every default ROOT defines - the zero derivative, the
``Par_i`` names, the gradient written in terms of the derivative - with it.
"""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from xrdroot.pyroot.core import funcbase

Math = ROOT.Math


def test_roots_names_are_the_classes_behind_its_typedefs():
    expect = {
        "IMultiGenFunction": funcbase.IBaseFunctionMultiDim,
        "IMultiGenFunctionTempl": funcbase.IBaseFunctionMultiDim,
        "IBaseFunctionMultiDim": funcbase.IBaseFunctionMultiDim,
        "IGenFunction": funcbase.IBaseFunctionOneDim,
        "IGradFunction": funcbase.IGradientFunctionOneDim,
        "IMultiGradFunction": funcbase.IGradientFunctionMultiDim,
        "IParamFunction": funcbase.IParametricFunctionOneDim,
        "IParamMultiFunction": funcbase.IParametricFunctionMultiDimTempl,
        "IParametricFunctionMultiDim": funcbase.IParametricFunctionMultiDimTempl,
        "IParamGradFunction": funcbase.IParametricGradFunctionOneDim,
        "IParamMultiGradFunction": funcbase.IParametricGradFunctionMultiDimTempl,
        "IParametricGradFunctionMultiDim": funcbase.IParametricGradFunctionMultiDimTempl,
    }
    for name, made in expect.items():
        assert getattr(Math, name) is made, name


def test_a_backend_type_in_a_template_names_the_same_class():
    # a translated ``IParametricFunctionMultiDimTempl<double>`` asks for the class with
    # the backend type subscripted, which is a double here
    asked = Math.IParametricFunctionMultiDimTempl["double"]
    assert asked is funcbase.IParametricFunctionMultiDimTempl
    assert Math.IMultiGenFunction["double"] is funcbase.IBaseFunctionMultiDim


class Rosenbrock(Math.IMultiGenFunction):
    """A macro's own function of two coordinates, as the test programs write one."""

    def NDim(self):
        return 2

    def Clone(self):
        return Rosenbrock()

    def DoEval(self, x):
        return 100 * (x[1] - x[0] * x[0]) ** 2 + (1 - x[0]) ** 2


class Parabola(Math.IGenFunction):
    """And its one-dimensional counterpart."""

    def Clone(self):
        return Parabola()

    def DoEval(self, x):
        return (x - 3) ** 2


def test_a_multi_dimensional_function_is_called_through_operator():
    rosen = Rosenbrock()
    expect = {
        "at the minimum": rosen([1.0, 1.0]),
        "and away from it": rosen(np.array([0.0, 0.0])),
        "ndim": rosen.NDim(),
        "clone": type(rosen.Clone()),
        "has no gradient": rosen.HasGradient(),
    }
    assert expect == {
        "at the minimum": 0.0,
        "and away from it": 1.0,
        "ndim": 2,
        "clone": Rosenbrock,
        "has no gradient": False,
    }


def test_a_one_dimensional_function_takes_a_number_or_a_point():
    parabola = Parabola()
    point = np.array([5.0])
    assert (parabola(3.0), parabola(point), parabola.NDim()) == (0.0, 4.0, 1)
    # ROOT's own derivative is zero unless the function says otherwise
    assert (parabola.Derivative(3.0), parabola.DoDerivative(3.0)) == (0.0, 0.0)


def test_a_plain_derivative_fills_the_gradient_in_place():
    one = [0.0]
    Parabola().Gradient(4.0, one)
    many = [0.0, 0.0]
    ParabolaGradient().Gradient(np.array([1.0, 2.0]), many)
    assert (one, many) == ([0.0], [2.0, 4.0])


class ParabolaGradient(Math.IMultiGradFunction):
    """A gradient function, which the tools ask for a derivative at a point through."""

    def NDim(self):
        return 2

    def Clone(self):
        return ParabolaGradient()

    def DoEval(self, x):
        return x[0] ** 2 + x[1] ** 2

    def Derivative(self, x, icoord=0):
        return 2.0 * x[icoord]

    def DoDerivative(self, x, icoord):
        return 2.0 * x[icoord]


def test_a_gradient_function_says_it_has_one_and_gives_its_derivatives():
    found = ParabolaGradient()
    grad = [0.0, 0.0]
    found.Gradient(np.array([3.0, 4.0]), grad)
    assert (found.HasGradient(), grad, found.Derivative([3.0, 4.0], 1)) == (True, [6.0, 8.0], 8.0)


def test_a_cell_takes_the_value_and_the_derivative_at_once():
    class Cell:
        value = 0.0

    one = [0.0]
    value, derivative = Cell(), Cell()
    Parabola().FdF(5.0, value, derivative)
    many_value, many = Cell(), [0.0, 0.0]
    ParabolaGradient().FdF(np.array([1.0, 2.0]), many_value, many)
    assert (value.value, derivative.value) == (4.0, 0.0)
    assert (many_value.value, many) == (5.0, [2.0, 4.0])
    assert one == [0.0]


class Line(Math.IParamFunction):
    """A parametric function of one number, as ``testFit.cxx``'s ``Func1D`` is."""

    def __init__(self):
        self._p = [0.0, 0.0, 0.0]

    def SetParameters(self, p):
        self._p = [float(p[0]), float(p[1]), float(p[2])]

    def Parameters(self):
        return self._p

    def NPar(self):
        return 3

    def Clone(self):
        made = Line()
        made.SetParameters(self._p)
        return made

    def DoEvalPar(self, x, p):
        return p[0] * x * x + p[1] * x + p[2]


def test_a_parametric_function_is_evaluated_at_the_parameters_it_was_given():
    line = Line()
    line.SetParameters([1.0, 2.0, 3.0])
    assert (line(2.0), line(2.0, [0.0, 0.0, 1.0]), line.DoEval(2.0)) == (11.0, 1.0, 11.0)
    assert (line.NPar(), line.ParameterName(1), line.Parameters()) == (3, "Par_1", [1.0, 2.0, 3.0])


class Plane(Math.IParamMultiFunction):
    """A parametric function of several coordinates: a plane over a grid, as ``GradFunc2D``."""

    def __init__(self):
        self._p = [0.0] * 5

    def SetParameters(self, p):
        self._p = [float(one) for one in p]

    def Parameters(self):
        return self._p

    def NPar(self):
        return 5

    def NDim(self):
        return 2

    def Clone(self):
        return Plane()

    def DoEvalPar(self, x, p):
        return p[0] * x[0] ** 2 + p[1] * x[0] + p[2] * x[1] ** 2 + p[3] * x[1] + p[4]


def test_a_parametric_function_of_several_coordinates():
    plane = Plane()
    plane.SetParameters([1.0, 2.0, 3.0, 4.0, 5.0])
    assert (plane([1.0, 1.0]), plane([1.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0])) == (15.0, 1.0)


class ParabolaPar(Math.IParamGradFunction):
    """A parametric function of one number that also says its parameter derivatives."""

    def __init__(self):
        self._p = [0.0, 0.0]

    def Parameters(self):
        return self._p

    def SetParameters(self, p):
        self._p = [float(p[0]), float(p[1])]

    def NPar(self):
        return 2

    def Clone(self):
        return ParabolaPar()

    def DoEvalPar(self, x, p):
        return p[0] * x + p[1]

    def DoParameterDerivative(self, x, p, ipar):
        return float(x) if ipar == 0 else 1.0


def test_the_parameter_gradient_is_what_the_derivative_says():
    parabola = ParabolaPar()
    parabola.SetParameters([2.0, 1.0])
    cached = [0.0, 0.0]
    given = [0.0, 0.0]
    parabola.ParameterGradient(3.0, cached)
    parabola.ParameterGradient(3.0, [4.0, 5.0], given)
    assert (cached, given) == ([3.0, 1.0], [3.0, 1.0])
    cached_derivative = parabola.ParameterDerivative(3.0, 0)
    given_derivative = parabola.ParameterDerivative(3.0, [1.0, 1.0], 1)
    assert (cached_derivative, given_derivative) == (3.0, 1.0)
    assert (parabola.HasParameterHessian(), parabola.ParameterHessian(3.0, [1.0, 1.0], [0.0]),
            parabola.ParameterG2(3.0, [1.0, 1.0], [0.0])) == (False, False, False)  # fmt: skip


def test_a_point_is_a_number_or_the_first_of_a_point():
    assert (funcbase._number(3), funcbase._number([3.0]), funcbase._number(np.array([3.0]))) == (
        3.0,
        3.0,
        3.0,
    )


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: Math.IMultiGenFunction().NDim(), "override it to return how many"),
        (lambda: Math.IMultiGenFunction().Clone(), "derive from IMultiGenFunction"),
        (lambda: Math.IMultiGenFunction().DoEval([1.0]), "derive from IMultiGenFunction"),
        (lambda: Math.IGenFunction().Clone(), "derive from IGenFunction"),
        (lambda: Math.IParamFunction().Parameters(), "derive from IParamFunction"),
        (lambda: Math.IParamFunction().SetParameters([1.0]), "derive from IParamFunction"),
        (lambda: Math.IParamFunction().NPar(), "derive from IParamFunction"),
        (lambda: Math.IParamFunction().DoEval(1.0), "derive from IParamFunction"),
        (lambda: Math.IParamFunction().DoEvalPar(1.0, [1.0]), "derive from IParamFunction"),
        (lambda: Math.IParamMultiFunction().DoEvalPar([1.0], [1.0]), "derive from IParametric"),
        (lambda: Math.IParamGradFunction().DoParameterDerivative(1.0, [1.0], 0), "one parameter"),
        (lambda: Math.IParamMultiGradFunction().DoParameterDerivative(
            [1.0], [1.0], 0), "one parameter"),
    ],
)
def test_an_interface_says_in_a_sentence_what_its_subclass_must_write(make, message):
    with pytest.raises(NotImplementedError, match=message):
        make()


def test_a_parametric_function_without_an_evaluation_says_which_method_it_needs():
    # a subclass that overrides nothing cannot be evaluated at its parameters, and ROOT's
    # own DoEval - which calls DoEvalPar - is the method to name
    empty_many = Math.IParamMultiFunction()
    empty_one = Math.IParamFunction()
    with pytest.raises(NotImplementedError, match="derive from IParametricFunctionMultiDim"):
        empty_many.DoEval([1.0])
    with pytest.raises(NotImplementedError, match="derive from IParamFunction"):
        empty_one.DoEval(1.0)


def test_the_functor_is_the_interface_a_python_function_is_handed_to_root_through():
    functor = Math.Functor(lambda xx: xx[0] + xx[1], 2)
    graded = Math.GradFunctor(lambda xx: xx[0] * xx[1], lambda xx, i: xx[1 - i], 2)
    one = Math.Functor1D(lambda x: x * x)
    graded_one = Math.GradFunctor1D(lambda x: x * x, lambda x: 2 * x)
    assert (functor.HasGradient(), functor.DoEval([1.0, 2.0]), functor.NDim()) == (False, 3.0, 2)
    assert (graded.HasGradient(), graded.DoDerivative([2.0, 3.0], 0), graded.HasGradient()) == (
        True,
        3.0,
        True,
    )
    cloned, derivative = one.Clone()(4.0), graded_one.DoDerivative(4.0)
    assert (cloned, derivative, graded_one.HasGradient()) == (16.0, 8.0, True)
    assert (graded.Clone().Derivative([2.0, 3.0], 1), functor.Clone().NDim()) == (2.0, 2)


def test_the_minimiser_uses_the_gradient_only_of_a_function_that_has_one():
    from xrdroot.pyroot.core.mathtools import gradient

    def plain(x):
        return x[0]

    graded = Math.GradFunctor(lambda xx: xx[0] ** 2, lambda xx, i: 2 * xx[0], 1)
    assert gradient(plain) is None
    assert gradient(graded)(np.array([3.0])) == [6.0]


def test_a_plain_interface_says_it_has_no_gradient_and_no_derivative():
    # ROOT's own answers: no gradient unless the class derives from a gradient interface,
    # and a derivative of zero unless the class says otherwise
    assert (Math.IGenFunction().HasGradient(), Math.IGenFunction().DoDerivative(2.0)) == (
        False, 0.0)  # fmt: skip
    assert (Math.IMultiGenFunction().HasGradient(), Math.IMultiGenFunction().DoDerivative(
        [1.0], 0)) == (False, 0.0)  # fmt: skip
    assert Math.IGradFunction().HasGradient() is True
    assert Math.IMultiGenFunction().Derivative([1.0], 0) == 0.0


def test_a_functor_says_whether_it_has_a_gradient_and_copies_itself_with_the_same_ones():
    assert (Math.Functor1D(lambda x: x).HasGradient(), Math.GradFunctor1D(
        lambda x: x, lambda x: 1.0).HasGradient()) == (False, True)  # fmt: skip
    assert Math.GradFunctor1D(lambda x: x * x, lambda x: 2 * x).Clone().Derivative(3.0) == 6.0
