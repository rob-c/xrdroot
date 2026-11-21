"""``ROOT.Math``'s interfaces, its integration kinds and ``Util``, as a macro derives from them."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_the_integration_kinds_have_root_s_values():
    expect(
        (ROOT.Math.IntegrationOneDim.kGAUSS, 0),
        (ROOT.Math.IntegrationOneDim.kLEGENDRE, 1),
        (ROOT.Math.IntegrationOneDim.kADAPTIVE, 2),
        (ROOT.Math.IntegrationOneDim.kADAPTIVESINGULAR, 3),
        (ROOT.Math.IntegrationOneDim.kNONADAPTIVE, 4),
        (ROOT.Math.IntegrationMultiDim.kADAPTIVE, 0),
        (ROOT.Math.IntegrationMultiDim.kVEGAS, 1),
        (ROOT.Math.IntegrationMultiDim.kMISER, 2),
        (ROOT.Math.IntegrationMultiDim.kPLAIN, 3),
    )


def test_an_integrator_takes_a_known_rule_and_integrates_the_function_it_was_given():
    integrator = ROOT.Math.Integrator(ROOT.Math.IntegrationOneDim.kADAPTIVESINGULAR)
    integrator.SetFunction(ROOT.Math.Functor1D(lambda x: x))
    expect((integrator.Integral(0.0, 1.0), 0.5))


def test_an_integrator_integrates_a_function_handed_to_the_call():
    """ROOT's ``Integral(f, a, b)`` overload, which a macro wrapping a TF1 per integral uses."""
    integrator = ROOT.Math.Integrator()
    assert integrator.Integral(ROOT.Math.Functor1D(lambda x: x * x), 0.0, 3.0) == pytest.approx(9.0)


def test_a_wrapped_tf1_is_what_the_one_dimensional_tools_call():
    function = ROOT.TF1("f", "[0] * x + [1]", 0.0, 1.0)
    function.SetParameters(2.0, 1.0)
    wrapped = ROOT.Math.WrappedTF1(function)
    expect(
        (wrapped(3.0), 7.0),
        (wrapped.DoEval(4.0), 9.0),
        (wrapped.NDim(), 1),
    )
    assert wrapped.Derivative(4.0) == pytest.approx(2.0)
    integrator = ROOT.Math.Integrator()
    assert integrator.Integral(wrapped, 0.0, 1.0) == pytest.approx(2.0)


def test_a_multidimensional_integral_is_the_volume_times_the_lattice_s_mean():
    ig = ROOT.Math.IntegratorMultiDim(ROOT.Math.IntegrationMultiDim.kVEGAS)
    ig.SetFunction(ROOT.Math.Functor(lambda xx: xx[0] + xx[1], 2))
    found = ig.Integral([0.0, 0.0], [1.0, 1.0])
    expect(
        (found, 1.0),
        (ig.Result(), 1.0),
        (ig.Status(), 0),
    )
    assert ig.NEval() == 65536
    assert ig.Error() >= 0.0


def test_a_multidimensional_integrator_takes_its_function_with_the_rule():
    adaptive = ROOT.Math.AdaptiveIntegratorMultiDim(
        ROOT.Math.Functor(lambda xx: xx[0], 1), 1e-5, 1e-5, 10000
    )
    expect((adaptive.Integral(np.zeros(1), np.ones(1)), 0.5))


def test_a_box_with_a_different_number_of_corners_is_refused():
    ig = ROOT.Math.IntegratorMultiDim()
    ig.SetFunction(ROOT.Math.Functor(lambda xx: xx[0], 1))
    with pytest.raises(ValueError, match="as many lower corners as upper ones"):
        ig.Integral([0.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="as many lower corners as upper ones"):
        ig.Integral([], [])


def test_the_tolerances_are_accepted_and_change_nothing():
    ig = ROOT.Math.IntegratorMultiDim()
    ig.SetFunction(ROOT.Math.Functor(lambda xx: xx[0], 1))
    before = ig.Integral([0.0], [2.0])
    ig.SetRelTolerance(1e-3)
    ig.SetAbsTolerance(1e-3)
    ig.SetMaxPts(10)
    assert ig.Integral([0.0], [2.0]) == before == 2.0


def test_util_to_string_writes_what_a_cpp_stream_would():
    expect(
        (ROOT.Math.Util.ToString("x0"), "x0"),
        (ROOT.Math.Util.ToString(3), "3"),
        (ROOT.Math.Util.ToString(0.5), "0.5"),
        (ROOT.Math.Util.ToString(1e-7), "1e-07"),
    )


def test_a_multidim_function_interface_calls_its_subclass_s_do_eval():
    class Squared(ROOT.Math.IMultiGenFunction):
        def DoEval(self, x):
            return x[0] * x[0] + x[1]

        def NDim(self):
            return 2

    expect((Squared()([3.0, 1.0]), 10.0))


def test_the_multidim_interface_says_what_to_override():
    plain = ROOT.Math.IMultiGenFunction()
    with pytest.raises(NotImplementedError, match="override DoEval and NDim"):
        plain([1.0])
    with pytest.raises(NotImplementedError, match="override DoEval and NDim"):
        plain.DoEval([1.0])
    with pytest.raises(NotImplementedError, match="return the dimension from NDim"):
        plain.NDim()


def test_a_gradient_interface_fills_a_gradient_array_from_derivative():
    class Weighted(ROOT.Math.IMultiGradFunction):
        def DoEval(self, x):
            return 2 * x[0] + 3 * x[1]

        def Derivative(self, x, icoord=0):
            return (2.0, 3.0)[icoord]

        def NDim(self):
            return 2

    grad = [0.0, 0.0]
    Weighted().Gradient([1.0, 1.0], grad)
    expect((grad, [2.0, 3.0]))
    with pytest.raises(NotImplementedError, match="override Derivative"):
        ROOT.Math.IMultiGradFunction().Derivative([1.0])


def test_a_param_function_interface_evaluates_at_the_parameters_it_holds():
    class Scaled(ROOT.Math.IParamFunction):
        def __init__(self):
            self._p = [1.0, 0.0]

        def DoEvalPar(self, x, p):
            return p[0] * x + p[1]

        def Parameters(self):
            return self._p

        def SetParameters(self, p):
            self._p = list(p)

        def NPar(self):
            return 2

    function = Scaled()
    function.SetParameters([2.0, 1.0])
    expect(
        (function(3.0), 7.0),
        (function.DoEval(3.0), 7.0),
        (function.NPar(), 2),
    )


def test_a_param_multifunction_evaluates_at_the_parameters_it_holds():
    class Scaled(ROOT.Math.IParamMultiFunction):
        def __init__(self):
            self._p = [1.0, 0.0]

        def DoEvalPar(self, x, p):
            return p[0] * x[0] + p[1]

        def Parameters(self):
            return self._p

        def SetParameters(self, p):
            self._p = list(p)

        def NPar(self):
            return 2

        def NDim(self):
            return 1

    function = Scaled()
    function.SetParameters([2.0, 1.0])
    expect((function([3.0]), 7.0), (function.NPar(), 2))


def test_the_param_multifunction_interfaces_say_what_to_override():
    plain = ROOT.Math.IParamMultiFunction()
    with pytest.raises(NotImplementedError, match="override DoEvalPar"):
        plain.DoEvalPar([1.0], [0.0])
    with pytest.raises(NotImplementedError, match="from Parameters"):
        plain.Parameters()
    with pytest.raises(NotImplementedError, match="in SetParameters"):
        plain.SetParameters([0.0])
    with pytest.raises(NotImplementedError, match="count from NPar"):
        plain.NPar()
    with pytest.raises(NotImplementedError, match="override ParameterGradient"):
        ROOT.Math.IParamMultiGradFunction().ParameterGradient([1.0], [0.0], [0.0])


def test_the_param_function_interface_says_what_to_override():
    plain = ROOT.Math.IParamFunction()
    with pytest.raises(NotImplementedError, match="override DoEvalPar"):
        plain.DoEvalPar(1.0, [0.0])
    with pytest.raises(NotImplementedError, match="Parameters"):
        plain.Parameters()
    with pytest.raises(NotImplementedError, match="SetParameters"):
        plain.SetParameters([0.0])
    with pytest.raises(NotImplementedError, match="NPar"):
        plain.NPar()
