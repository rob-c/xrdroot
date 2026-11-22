"""``ROOT::Math``'s function interfaces: what a macro's own function derives from.

ROOT's numerical tools - the minimisers, the integrators, the fitters - are
handed one of these interfaces rather than a function, so that they can ask a
function how many dimensions it has, clone it, and, for the parametric ones,
read and write its parameters. A macro says what its function is at a point by
overriding ``DoEval`` - or ``DoEvalPar`` when the function carries parameters -
and everything else here is ROOT's own: ``operator()`` in terms of those, the
gradient in terms of ``DoDerivative``, the parameter gradient in terms of
``DoParameterDerivative``.

The names are ROOT's, alias for alias, because a translated ``class F : public
ROOT::Math::IMultiGenFunction`` names the alias and expects the class behind
it. Two conventions differ from C++, both forced by Python: ``operator()``
takes a coordinate that is a number or a one-element array, since C++ overloads
those two, and the arguments ROOT writes through are written into in place.
"""

from __future__ import annotations

from typing import Any


def _no_evaluation_par(instead: str) -> NotImplementedError:
    """The refusal of an evaluation a derived class has to write: an interface's own body
    cannot know what the function is at a point, and saying which method to fill in beats
    a NaN coming out of the tools that call it."""
    return NotImplementedError(
        "DoEvalPar is what a parametric function is at a point for given parameters; derive "
        f"from {instead} and override it."
    )


def _number(x: Any) -> float:
    """The coordinate of a one-dimensional function: C++ overloads
    ``operator()(double)`` and ``operator()(const double *)``, so a caller may
    pass either the number or a one-element array holding it."""
    if isinstance(x, (int, float)):
        return float(x)
    return float(x[0])


class IBaseParam:
    """``ROOT::Math::IBaseParam``: the parameter API both parametric interfaces share."""

    def Parameters(self) -> Any:
        """``Parameters()``: the parameter values, as the derived class holds them."""
        raise NotImplementedError(
            "Parameters() is what a parametric function's own class returns its parameters "
            "from; derive from IParamFunction and override it."
        )

    def SetParameters(self, p: Any) -> None:
        """``SetParameters(p)``: the values a ``const double *`` points at, into the function."""
        raise NotImplementedError(
            "SetParameters() is what a parametric function's own class takes its parameters "
            "through; derive from IParamFunction and override it."
        )

    def NPar(self) -> int:
        """``NPar()``: how many parameters the function has."""
        raise NotImplementedError(
            "NPar() is what a parametric function's own class says how many parameters it has "
            "from; derive from IParamFunction and override it."
        )

    def ParameterName(self, i: int) -> str:
        """``ParameterName(i)``: ``Par_i`` unless the derived class names its own, as ROOT's."""
        return f"Par_{int(i)}"


class IBaseFunctionOneDim:
    """``ROOT::Math::IBaseFunctionOneDim``, which ``IGenFunction`` names.

    A macro's own class overrides ``DoEval``; ``__call__`` is what the tools
    call, as they call ``operator()``, which ROOT defines in terms of ``DoEval``.
    """

    def __call__(self, x: Any) -> float:
        return float(self.DoEval(_number(x)))

    def DoEval(self, x: float) -> float:
        """``DoEval(x)``: what the function is at ``x``, which a subclass says."""
        raise NotImplementedError(
            "IGenFunction is the interface ROOT's tools call a function of one number "
            "through; derive from it and override DoEval, or pass a plain Python function "
            "to Functor1D."
        )

    def NDim(self) -> int:
        """``NDim()``: one, this being the one-dimensional interface."""
        return 1

    def Clone(self) -> Any:
        """``Clone()``: a copy, which ROOT's tools make before changing a function."""
        raise NotImplementedError(
            "Clone() is how a function hands ROOT's tools a copy of itself; derive from "
            "IGenFunction and override it."
        )

    def HasGradient(self) -> bool:
        """``HasGradient()``: whether the class also derives from a gradient interface."""
        return False

    def Derivative(self, x: Any) -> float:
        """``Derivative(x)``: the function's own derivative there."""
        return float(self.DoDerivative(_number(x)))

    def DoDerivative(self, x: float) -> float:
        """``DoDerivative(x)``: the derivative, which ROOT answers zero for unless told."""
        return 0.0

    def Gradient(self, x: Any, g: Any) -> None:
        """``Gradient(x, g)``: the one-component gradient, into the array given."""
        g[0] = self.DoDerivative(_number(x))

    def FdF(self, x: Any, f: Any, df: Any) -> None:
        """``FdF(x, f, df)``: the value and the derivative at once, into the two references."""
        at = _number(x)
        f.value = float(self.DoEval(at))
        df.value = float(self.DoDerivative(at))


class IBaseFunctionMultiDim:
    """``ROOT::Math::IBaseFunctionMultiDim``, which ``IMultiGenFunction`` names.

    A macro's own class overrides ``NDim``, ``Clone`` and ``DoEval``; what the
    minimisers call is ``__call__(x)``, ROOT's ``operator()(const double *)``.
    """

    def __class_getitem__(cls, kind: Any) -> type:
        """``IMultiGenFunction<double>``: the backend type, which is a double here."""
        return cls

    def __call__(self, x: Any) -> float:
        return float(self.DoEval(x))

    def DoEval(self, x: Any) -> float:
        """``DoEval(x)``: what the function is at the point ``x``, which a subclass says."""
        raise NotImplementedError(
            "DoEval is what a function of several coordinates says it is at a point; derive "
            "from IMultiGenFunction and override it, or pass a plain Python function to "
            "Functor."
        )

    def NDim(self) -> int:
        """``NDim()``: how many coordinates a point has."""
        raise NotImplementedError(
            "NDim() is the coordinates a function of several takes; derive from "
            "IMultiGenFunction and override it to return how many there are."
        )

    def Clone(self) -> Any:
        """``Clone()``: a copy, which ROOT's tools make before changing a function."""
        raise NotImplementedError(
            "Clone() is how a function hands ROOT's tools a copy of itself; derive from "
            "IMultiGenFunction and override it."
        )

    def HasGradient(self) -> bool:
        """``HasGradient()``: whether the class also derives from a gradient interface."""
        return False

    def Derivative(self, x: Any, icoord: int = 0) -> float:
        """``Derivative(x, icoord)``: the partial derivative along one coordinate."""
        return float(self.DoDerivative(x, int(icoord)))

    def DoDerivative(self, x: Any, icoord: int) -> float:
        """``DoDerivative(x, icoord)``: the partial derivative, zero unless told."""
        return 0.0

    def Gradient(self, x: Any, grad: Any) -> None:
        """``Gradient(x, grad)``: every partial derivative, into the array given.

        Through ``Derivative``, which is ROOT's own chain - ``Gradient`` asks ``Derivative``
        for each coordinate and ``Derivative`` asks ``DoDerivative`` - so a class that writes
        either one is asked, and one that writes neither is answered zeros.
        """
        for icoord in range(self.NDim()):
            grad[icoord] = self.Derivative(x, icoord)

    def FdF(self, x: Any, f: Any, grad: Any) -> None:
        """``FdF(x, f, grad)``: the value and the gradient at once, into what it is given."""
        f.value = float(self.DoEval(x))
        self.Gradient(x, grad)


class IGradientFunctionOneDim(IBaseFunctionOneDim):
    """``ROOT::Math::IGradientFunctionOneDim``, which ``IGradFunction`` names: one that
    can say its derivative, so the tools need not difference it."""

    def HasGradient(self) -> bool:
        """``HasGradient()``: yes, this being the gradient interface."""
        return True


class IGradientFunctionMultiDim(IBaseFunctionMultiDim):
    """``ROOT::Math::IGradientFunctionMultiDim``, which ``IMultiGradFunction`` names:
    a function of several coordinates that can say its partial derivatives."""

    def HasGradient(self) -> bool:
        """``HasGradient()``: yes, this being the gradient interface."""
        return True


class IParametricFunctionOneDim(IBaseFunctionOneDim, IBaseParam):
    """``ROOT::Math::IParametricFunctionOneDim``, which ``IParamFunction`` names.

    A macro's own class overrides ``DoEvalPar``, ``Parameters``, ``SetParameters``,
    ``NPar`` and ``Clone``; ``DoEval(x)`` is then the function at the parameters
    last set, as ROOT defines it.
    """

    def __call__(self, x: Any, parameters: Any = None) -> float:
        """``operator()(x)`` at the parameters last set, or ``operator()(x, p)`` at ``p``."""
        at = _number(x)
        if parameters is None:
            return float(self.DoEval(at))
        return float(self.DoEvalPar(at, parameters))

    def DoEval(self, x: float) -> float:
        """``DoEval(x)``: ``DoEvalPar`` at the parameters last set, as ROOT's own does."""
        if type(self).DoEvalPar is IParametricFunctionOneDim.DoEvalPar:
            raise _no_evaluation_par("IParamFunction")
        return float(self.DoEvalPar(x, self.Parameters()))

    def DoEvalPar(self, x: float, p: Any) -> float:
        """``DoEvalPar(x, p)``: what the function is at ``x`` for the parameters ``p``."""
        raise _no_evaluation_par("IParamFunction")


class IParametricFunctionMultiDimTempl(IBaseFunctionMultiDim, IBaseParam):
    """``ROOT::Math::IParametricFunctionMultiDim``, spelled ``…Templ<T>`` in ROOT's own
    header and named ``IParamMultiFunction`` too: several coordinates, and parameters.

    ``T`` is the backend type, ``double`` in everything ROOT's scalar tools use, so a
    translated ``IParametricFunctionMultiDimTempl<double>`` is this class.
    """

    def __class_getitem__(cls, kind: Any) -> type:
        """``IParametricFunctionMultiDimTempl<double>``: the backend type, a double here."""
        return cls

    def __call__(self, x: Any, parameters: Any = None) -> float:
        """``operator()(x)`` at the parameters last set, or ``operator()(x, p)`` at ``p``."""
        if parameters is None:
            return float(self.DoEval(x))
        return float(self.DoEvalPar(x, parameters))

    def DoEval(self, x: Any) -> float:
        """``DoEval(x)``: ``DoEvalPar`` at the parameters last set, as ROOT's own does."""
        if type(self).DoEvalPar is IParametricFunctionMultiDimTempl.DoEvalPar:
            raise _no_evaluation_par("IParametricFunctionMultiDim")
        return float(self.DoEvalPar(x, self.Parameters()))

    def DoEvalPar(self, x: Any, p: Any) -> float:
        """``DoEvalPar(x, p)``: what the function is at the point ``x`` for ``p``."""
        raise _no_evaluation_par("IParametricFunctionMultiDim")


class _ParameterGradient(IBaseParam):
    """The parameter gradient of a parametric function, which both gradient interfaces
    say the same way: the Hessian is ROOT's to ask for and no function here gives it."""

    def ParameterGradient(self, x: Any, *rest: Any) -> None:
        """``ParameterGradient(x, grad)`` at the parameters last set, or ``(x, p, grad)``."""
        at, p, grad = _gradient_arguments(self, x, rest)
        for ipar in range(self.NPar()):
            grad[ipar] = self.DoParameterDerivative(at, p, ipar)

    def ParameterDerivative(self, x: Any, *rest: Any) -> float:
        """``ParameterDerivative(x, ipar)``, or ``(x, p, ipar)`` for parameters given."""
        at, p, ipar = (x, *(_derivative_arguments(self, rest)))
        return float(self.DoParameterDerivative(at, p, ipar))

    def DoParameterDerivative(self, x: Any, p: Any, ipar: int) -> float:
        """``DoParameterDerivative``: one parameter's partial derivative, to be written."""
        raise NotImplementedError(
            "DoParameterDerivative is how a parametric function's own class says the "
            "derivative with respect to one parameter; override it."
        )

    def HasParameterHessian(self) -> bool:
        """``HasParameterHessian()``: whether the class can say the parameter Hessian."""
        return False

    def ParameterHessian(self, x: Any, p: Any, h: Any) -> bool:
        """``ParameterHessian(x, p, h)``: no, unless a subclass fills the triangle given."""
        return False

    def ParameterG2(self, x: Any, p: Any, g2: Any) -> bool:
        """``ParameterG2(x, p, g2)``: the diagonal of the Hessian, which this cannot say."""
        return False


class IParametricGradFunctionOneDim(_ParameterGradient, IParametricFunctionOneDim):
    """``ROOT::Math::IParametricGradFunctionOneDim``, which ``IParamGradFunction``
    names: a parametric function of one number that can say its parameter gradient."""


class IParametricGradFunctionMultiDimTempl(_ParameterGradient, IParametricFunctionMultiDimTempl):
    """``ROOT::Math::IParametricGradFunctionMultiDim``, spelled ``…Templ<T>`` in ROOT's
    own header and named ``IParamMultiGradFunction`` too: several coordinates,
    parameters, and the gradient with respect to those parameters."""


def _gradient_arguments(fn: Any, x: Any, rest: tuple[Any, ...]) -> tuple[Any, Any, Any]:
    """``ParameterGradient``'s two forms: the parameters are the values last set unless
    they were given, as ROOT overloads ``(x, grad)`` and ``(x, p, grad)``."""
    return (x, rest[0], rest[1]) if len(rest) == 2 else (x, fn.Parameters(), rest[0])


def _derivative_arguments(fn: Any, rest: tuple[Any, ...]) -> tuple[Any, int]:
    """``ParameterDerivative``'s two forms: ``(p, ipar)``, or ``(ipar)`` at the
    parameters last set."""
    return (rest[0], rest[1]) if len(rest) == 2 else (fn.Parameters(), rest[0])


#: ROOT's names for the classes above: one per typedef or alias in ``IFunctionfwd.h``
#: and ``IParamFunctionfwd.h``, so a translated macro names whichever ROOT does.
IMultiGenFunction = IMultiGenFunctionTempl = IBaseFunctionMultiDim
IGenFunction = IBaseFunctionOneDim
IGradFunction = IGradientFunctionOneDim
IMultiGradFunction = IGradientFunctionMultiDim
IParamFunction = IParametricFunctionOneDim
IParamMultiFunction = IParamMultiFunctionTempl = IParametricFunctionMultiDim = (
    IParametricFunctionMultiDimTempl
)
IParamGradFunction = IParametricGradFunctionOneDim
IParamMultiGradFunction = IParamMultiGradFunctionTempl = IParametricGradFunctionMultiDim = (
    IParametricGradFunctionMultiDimTempl
)

__all__ = [
    "IBaseFunctionMultiDim",
    "IBaseFunctionOneDim",
    "IBaseParam",
    "IGenFunction",
    "IGradFunction",
    "IGradientFunctionMultiDim",
    "IGradientFunctionOneDim",
    "IMultiGenFunction",
    "IMultiGenFunctionTempl",
    "IMultiGradFunction",
    "IParamFunction",
    "IParamGradFunction",
    "IParamMultiFunction",
    "IParamMultiFunctionTempl",
    "IParamMultiGradFunction",
    "IParamMultiGradFunctionTempl",
    "IParametricFunctionMultiDim",
    "IParametricFunctionMultiDimTempl",
    "IParametricFunctionOneDim",
    "IParametricGradFunctionMultiDim",
    "IParametricGradFunctionMultiDimTempl",
    "IParametricGradFunctionOneDim",
]
