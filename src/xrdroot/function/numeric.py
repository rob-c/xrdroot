"""Integrals, extrema, roots and derivatives of a function of one variable.

These are the numerical methods ``TF1`` calls, rebuilt over functions that
take a whole array of points at once, so that each step of each method is
one evaluation of the formula rather than a Python loop over points:

* :func:`integrate` is adaptive Gauss-Kronrod quadrature, 21 points to a
  panel, bisecting whichever panel's error estimate is largest until the
  total meets ``TF1::Integral``'s tolerances - with QUADPACK's error
  estimate, and its change of variable for an infinite end;
* :func:`minimise` is ``ROOT::Math::BrentMinimizer1D``: a scan of ``npx``
  points to bracket the least value, then Brent's golden-section and
  parabolic search inside the bracket, repeated if it does not converge;
  maximising and root-finding are the same search on ``-f`` and ``|f - y|``,
  as ``TF1::GetMaximum`` and ``TF1::GetX`` make them;
* :func:`derivative` is ``ROOT::Math::RichardsonDerivator``'s first
  derivative, two central differences combined to cancel the error of each.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["integrate", "minimise", "derivative", "richardson"]

Array = Any
Vectorised = Callable[[Array], Array]

#: The Kronrod nodes of the 21-point rule on [-1, 1], the outermost first,
#: the centre last; every other one is also a node of the 10-point Gauss rule.
KRONROD_NODES = (
    0.995657163025808080735527280689003,
    0.973906528517171720077964012084452,
    0.930157491355708226001207180059508,
    0.865063366688984510732096688423493,
    0.780817726586416897063717578345042,
    0.679409568299024406234327365114874,
    0.562757134668604683339000099272694,
    0.433395394129247190799265943165784,
    0.294392862701460198131126603103866,
    0.148874338981631210884826001129720,
    0.0,
)
KRONROD_WEIGHTS = (
    0.011694638867371874278064396062192,
    0.032558162307964727478818972459390,
    0.054755896574351996031381300244580,
    0.075039674810919952767043140916190,
    0.093125454583697605535065465083366,
    0.109387158802297641899210590325805,
    0.123491976262065851077808771084993,
    0.134709217311473325928054001771707,
    0.142775938577060080797094273138717,
    0.147739104901338491374841515972068,
    0.149445554002916905664936468389821,
)
#: The 10-point Gauss weights, for Kronrod nodes 1, 3, 5, 7 and 9.
GAUSS_WEIGHTS = (
    0.066671344308688137593568809893332,
    0.149451349150580593145776339657697,
    0.219086362515982043995534934228163,
    0.269266719309996355091226921569469,
    0.295524224714752870173892994651338,
)
#: The 21 points on [-1, 1], and each one's weight in each rule.
_NODES = np.concatenate([-np.asarray(KRONROD_NODES[:-1]), [0.0], np.asarray(KRONROD_NODES[-2::-1])])
_KRONROD = np.concatenate(
    [np.asarray(KRONROD_WEIGHTS[:-1]), [KRONROD_WEIGHTS[-1]], np.asarray(KRONROD_WEIGHTS[-2::-1])]
)
_GAUSS = np.zeros(21)
_GAUSS[[1, 3, 5, 7, 9]] = GAUSS_WEIGHTS
_GAUSS[[19, 17, 15, 13, 11]] = GAUSS_WEIGHTS

#: The most panels an integral is cut into before it is taken as it stands.
PANELS = 1000
#: Machine epsilon and the smallest normal double, which QUADPACK's error
#: estimate is floored by.
EPSILON = float(np.finfo(np.float64).eps)
TINY = float(np.finfo(np.float64).tiny)
#: ``ROOT::Math::MinimizerOptions``' default number of searches a Brent
#: minimisation repeats if its bracket turns out not to hold the minimum.
SEARCHES = 10
#: ``(3 - sqrt(5))/2``, the golden section Brent's method steps by.
GOLDEN = 0.381966011250105097


def _panel(f: Vectorised, low: float, high: float) -> tuple[float, float]:
    """One panel's 21-point estimate and QUADPACK's estimate of its error."""
    centre, half = 0.5 * (low + high), 0.5 * (high - low)
    values = np.asarray(f(centre + half * _NODES), dtype=np.float64)
    kronrod = float(np.dot(_KRONROD, values))
    gauss = float(np.dot(_GAUSS, values))
    mean = 0.5 * kronrod
    absolute = abs(half) * float(np.dot(_KRONROD, np.abs(values)))
    spread = abs(half) * float(np.dot(_KRONROD, np.abs(values - mean)))
    error = abs((kronrod - gauss) * half)
    if spread != 0 and error != 0:
        error = spread * min(1.0, (200 * error / spread) ** 1.5)
    if absolute > TINY / (50 * EPSILON):
        error = max(50 * EPSILON * absolute, error)
    return kronrod * half, error


def _finite(f: Vectorised, low: float, high: float, epsabs: float, epsrel: float) -> float:
    """Adaptive quadrature over a finite range: bisect the worst panel until the sum is good."""
    value, error = _panel(f, low, high)
    heap = [(-error, low, high, value)]
    total, spent = value, error
    while spent > max(epsabs, epsrel * abs(total)) and len(heap) < PANELS:
        worst, a, b, part = heapq.heappop(heap)
        middle = 0.5 * (a + b)
        if not a < middle < b:
            heapq.heappush(heap, (worst, a, b, part))
            break  # a panel as narrow as the doubles allow: nothing more to gain
        left, left_error = _panel(f, a, middle)
        right, right_error = _panel(f, middle, b)
        total += left + right - part
        spent += left_error + right_error + worst
        heapq.heappush(heap, (-left_error, a, middle, left))
        heapq.heappush(heap, (-right_error, middle, b, right))
    return math.fsum(entry[3] for entry in heap)


def _mapped(f: Vectorised, low: float, high: float) -> Vectorised:
    """``f`` over an infinite range, as a function on ``(0, 1]`` by ``x = (1 - t)/t``."""
    if math.isinf(low) and math.isinf(high):
        return lambda t: (f((1 - t) / t) + f(-(1 - t) / t)) / (t * t)
    if math.isinf(high):
        return lambda t: f(low + (1 - t) / t) / (t * t)
    return lambda t: f(high - (1 - t) / t) / (t * t)


def integrate(
    f: Vectorised, low: float, high: float, epsabs: float = 1e-12, epsrel: float = 1e-12
) -> float:
    """``TF1::Integral``: the integral of ``f`` from ``low`` to ``high``, either end infinite.

    Reversed ends give the negative, and equal ones nothing, as they should.
    """
    if low == high:
        return 0.0
    if low > high:
        return -integrate(f, high, low, epsabs, epsrel)
    if math.isinf(low) or math.isinf(high):
        return _finite(_mapped(f, low, high), 0.0, 1.0, epsabs, epsrel)
    return _finite(f, low, high, epsabs, epsrel)


def _scan(f: Vectorised, low: float, high: float, npx: int) -> tuple[float, float, float]:
    """``BrentMethods::MinimStep``: the least of ``npx`` evenly spaced points, bracketed."""
    npx = max(npx, 2)
    step = (high - low) / (npx - 1)
    points = low + step * np.arange(npx)
    values = np.asarray(f(points), dtype=np.float64)
    best = int(np.argmin(np.where(np.isnan(values), np.inf, values)))
    found = float(points[best])
    left, right = max(low, found - step), min(high, found + step)
    return min(found, right), left, right


def _parabola(state: dict[str, float]) -> float | None:
    """The step a parabola through the three best points proposes, or ``None`` to bisect."""
    x, w, v = state["x"], state["w"], state["v"]
    fx, fw, fv = state["fx"], state["fw"], state["fv"]
    r = (x - w) * (fx - fv)
    q = (x - v) * (fx - fw)
    p = (x - v) * q - (x - w) * r
    q = 2 * (q - r)
    p = -p if q > 0 else p
    q = abs(q)
    previous = state["e"]
    state["e"] = state["d"]
    inside = q * (state["a"] - x) < p < q * (state["b"] - x)
    if abs(p) < abs(0.5 * q * previous) and inside:
        return p / q
    return None


def _step(state: dict[str, float], tol: float) -> float:
    """Brent's next step from the best point: a parabola's, or a golden section's."""
    x, middle = state["x"], 0.5 * (state["a"] + state["b"])
    proposed = _parabola(state) if abs(state["e"]) > tol else None
    if proposed is not None:
        u = x + proposed
        if u - state["a"] < 2 * tol or state["b"] - u < 2 * tol:
            proposed = math.copysign(tol, middle - x)
        return proposed
    state["e"] = (state["a"] - x) if x >= middle else (state["b"] - x)
    return GOLDEN * state["e"]


def _update(state: dict[str, float], u: float, fu: float) -> None:
    """Keep the bracket, and the three best points, in Brent's bookkeeping."""
    if fu <= state["fx"]:
        state["b" if u < state["x"] else "a"] = state["x"]
        state["v"], state["fv"] = state["w"], state["fw"]
        state["w"], state["fw"] = state["x"], state["fx"]
        state["x"], state["fx"] = u, fu
        return
    state["a" if u < state["x"] else "b"] = u
    if fu <= state["fw"] or state["w"] == state["x"]:
        state["v"], state["fv"] = state["w"], state["fw"]
        state["w"], state["fw"] = u, fu
    elif fu <= state["fv"] or state["v"] in (state["x"], state["w"]):
        state["v"], state["fv"] = u, fu


def _brent(
    one: Callable[[float], float], bracket: tuple[float, float, float], eps: float, iterations: int
) -> tuple[float, bool, float, float]:
    """``BrentMethods::MinimBrent``: the minimum's place, whether it converged, the bracket."""
    start, a, b = bracket
    first = one(start)
    state = {"a": a, "b": b, "x": start, "w": start, "v": start}
    state.update(fx=first, fw=first, fv=first, e=0.0, d=0.0)
    for _ in range(iterations):
        x, middle = state["x"], 0.5 * (state["a"] + state["b"])
        tol = eps + abs(x) * eps
        if abs(x - middle) <= 2 * tol - 0.5 * (state["b"] - state["a"]):
            return x, True, state["a"], state["b"]
        state["d"] = _step(state, tol)
        u = x + (state["d"] if abs(state["d"]) >= tol else math.copysign(tol, state["d"]))
        _update(state, u, one(u))
    return state["x"], False, state["a"], state["b"]


def minimise(
    f: Vectorised,
    low: float,
    high: float,
    npx: int = 100,
    eps: float = 1e-10,
    iterations: int = 100,
) -> float:
    """Where ``f`` is least between ``low`` and ``high``, as ``BrentMinimizer1D`` finds it."""

    def one(point: float) -> float:
        return float(np.asarray(f(np.asarray([point], dtype=np.float64)))[0])

    x = 0.5 * (low + high)
    for _ in range(SEARCHES):
        bracket = _scan(f, low, high, npx)
        x, converged, low, high = _brent(one, bracket, eps, iterations)
        if converged:
            break
    return x


def richardson(values: Callable[[float], Array], h: float) -> Array:
    """Two central differences, of steps ``h`` and ``h/2``, combined to cancel their errors.

    This is ``RichardsonDerivator::Derivative1`` and ``TF1::GradientPar``
    alike: ``(8 (f(+h/2) - f(-h/2)) - (f(+h) - f(-h))) / (6 h)``, where
    ``values(s)`` is the function stepped by ``s``.
    """
    d0 = values(h) - values(-h)
    d2 = values(h / 2) - values(-h / 2)
    return (1 / (2.0 * h)) * (8 * d2 - d0) / 3.0


def derivative(f: Vectorised, x: Array, h: float) -> Array:
    """``TF1::Derivative``: Richardson's first derivative of ``f`` at every ``x``."""
    points = np.asarray(x, dtype=np.float64)
    return richardson(lambda step: np.asarray(f(points + step), dtype=np.float64), h)
