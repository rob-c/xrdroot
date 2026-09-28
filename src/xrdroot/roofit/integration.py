"""Integrals over a model's variables: in closed form where the class has one, else RooFit's way.

``RooRealIntegral`` integrates what a class can do in closed form
analytically and the rest numerically, and so does :func:`integral`. The
numerical part is RooFit's own algorithms, step for step, because a fit
normalises its model with them at every call and a different integrator,
however accurate, would move the minimum Minuit finds by more than ROOT
prints: :func:`romberg` is ``RooRombergIntegrator`` - ``RooIntegrator1D``,
the trapezoid rule refined until the Wynn-epsilon extrapolation of the last
five agrees to 1e-7 - and :func:`improper` is ``RooImproperIntegrator1D``,
which maps an open end onto a finite one with ``x -> 1/x``.

The integrand is evaluated at every point of a refinement at once, as an
array along a new last axis, so a context whose variables are columns of
events integrates for every event at the same time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .messages import INFO, WARNING, log

__all__ = ["EPS", "announce", "integral", "improper", "integrate_1d", "numeric_names", "romberg"]

#: ``RooNumIntConfig``'s default absolute and relative tolerances.
EPS = 1e-7
#: ``RooIntegrator1D``'s most refinements, and how many points it extrapolates from.
MAX_STEPS = 20
N_POINTS = 5

Integrand = Callable[[np.ndarray[Any, Any]], Any]


def _trapezoids(func: Integrand, saved: Any, n: int, low: float, high: float) -> Any:
    width = high - low
    if n == 1:
        values = func(np.array([low, high]))
        return 0.5 * width * (values[..., 0] + values[..., 1])
    count = 1 << (n - 2)
    step = width / count
    points = low + (0.5 + np.arange(count)) * step
    return 0.5 * (saved + width * np.sum(func(points), axis=-1) / count)


def _midpoints(func: Integrand, saved: Any, n: int, low: float, high: float) -> Any:
    width = high - low
    if n == 1:
        return width * func(np.array([0.5 * (low + high)]))[..., 0]
    it = 3 ** (n - 2)
    step = width / (3.0 * it)
    first = low + 0.5 * step + 3 * step * np.arange(it)
    points = np.stack([first, first + 2 * step], axis=-1).reshape(-1)
    return (saved + width * np.sum(func(points), axis=-1) / it) / 3.0


def _extrapolate(h: list[float], s: list[Any]) -> tuple[Any, Any]:
    """Neville's extrapolation of the last five refinements to a step of zero."""
    xa = h[-N_POINTS:]
    c = [np.asarray(one, dtype=np.float64) for one in s[-N_POINTS:]]
    d = list(c)
    ns = int(np.argmin(np.abs(xa)))
    value = c[ns]
    ns -= 1
    error: Any = 0.0
    for m in range(1, N_POINTS):
        for i in range(N_POINTS - m):
            w = c[i + 1] - d[i]
            den = w / (xa[i] - xa[i + m])
            d[i] = xa[i + m] * den
            c[i] = xa[i] * den
        error = d[ns]  # the steps shrink, so the tableau is always walked up from its last row
        ns -= 1
        value = value + error
    return value, error


def romberg(
    func: Integrand,
    low: float,
    high: float,
    trapezoid: bool = True,
    eps_abs: float = EPS,
    eps_rel: float = EPS,
    name: str = "",
) -> Any:
    """``RooFit::Detail::integrate1d``: the integral of ``func`` from ``low`` to ``high``."""
    if high - low == 0.0:
        return 0.0 * func(np.array([low]))[..., 0]
    h = [1.0]
    s: list[Any] = []
    saved: Any = 0.0
    for j in range(1, MAX_STEPS + 1):
        saved = (_trapezoids if trapezoid else _midpoints)(func, saved, j, low, high)
        s.append(saved)
        if j >= N_POINTS:
            value, error = _extrapolate(h, s)
            bound = np.maximum(eps_rel * np.abs(value), eps_abs)
            if np.all(np.abs(error) <= bound):
                return value
        h.append(h[-1] / 4.0 if trapezoid else h[-1] / 9.0)
    log(
        None,
        WARNING,
        "Integration",
        f"RooRombergIntegrator::integral: integral of {name} over "
        f"range ({low:g},{high:g}) did not converge after {MAX_STEPS} steps",
    )
    return s[-1]


def _inverted(func: Integrand) -> Integrand:
    """``RooInvTransform``: ``f(1/x) / x^2``, what an open end is integrated as."""
    return lambda x: func(1.0 / x) / (x * x)


def improper(func: Integrand, low: float, high: float, name: str = "") -> Any:
    """``RooImproperIntegrator1D``: an integral with one end or both at infinity."""
    inv = _inverted(func)
    open_low, open_high = np.isinf(low), np.isinf(high)
    if open_low and open_high:
        return (
            romberg(func, -1.0, 1.0, name=name)
            + romberg(inv, -1.0, 0.0, False, name=name)
            + romberg(inv, 0.0, 1.0, False, name=name)
        )
    if open_low:
        if high >= 0:
            return romberg(inv, -1.0, 0.0, False, name=name) + romberg(func, -1.0, high, name=name)
        return romberg(inv, 1.0 / high, 0.0, False, name=name)
    if low <= 0:
        return romberg(inv, 0.0, 1.0, False, name=name) + romberg(func, low, 1.0, name=name)
    return romberg(inv, 0.0, 1.0 / low, False, name=name)


def integrate_1d(func: Integrand, low: float, high: float, name: str = "") -> Any:
    """``RooIntegrator1D`` over a closed range, ``RooImproperIntegrator1D`` over an open one."""
    if np.isinf(low) or np.isinf(high):
        return improper(func, low, high, name)
    return romberg(func, low, high, name=name)


# -- integrals of a model's functions ----------------------------------------------


def _expanded(ctx: dict[str, Any]) -> dict[str, Any]:
    """``ctx`` with a new last axis on every array, for the points to integrate at."""
    return {
        k: (v[..., None] if isinstance(v, np.ndarray) and v.ndim else v) for k, v in ctx.items()
    }


def integral_name(func: Any, names: frozenset[str], rng: Any) -> str:
    """``g_Int[x]``, or ``g_Int[x|signal]`` over a named range: what ROOT calls an integral."""
    order = [one.GetName() for one in func.leaves() if one.GetName() in names]
    suffix = f"|{rng}" if rng else ""
    return f"{func.GetName()}_Int[{','.join(order)}{suffix}]"


def numeric_names(func: Any, names: frozenset[str], rng: Any = None) -> list[str]:
    """The variables of ``names`` that ``func`` has no closed form for, in its order."""
    names = frozenset(names) & func.dependents()
    closed = func.analytic_names(names, rng) & names
    return [one.GetName() for one in func.leaves() if one.GetName() in names - closed]


def announce(func: Any, names: frozenset[str], rng: Any = None, label: str | None = None,
             normalising: bool = False) -> None:  # fmt: skip
    """``RooRealIntegral::init``'s line for a numerical integral, as RooFit prints it on making one.

    RooFit makes - and so announces - its integral objects at moments of its
    own: twice when it sets up to generate, once when it plots a curve, once
    when a fit first evaluates its likelihood. The callers here say when;
    this says what. A density that normalises itself through a cache of its
    own (``normalised_by_cache``) makes no integral to normalise a copy for
    generating or plotting - ``normalising`` - and says nothing then.
    """
    if normalising and getattr(func, "normalised_by_cache", False):
        return
    inner = getattr(func, "announce_inner", None)
    if inner is not None:  # integrals a closed form is made of, which may themselves be numeric
        inner(frozenset(names), rng)
    numeric = numeric_names(func, names, rng)
    if not numeric:
        return
    method = "RooIntegrator1D" if len(numeric) == 1 else "RooAdaptiveIntegratorND"
    if len(numeric) == 1 and any(np.isinf(func.bounds(numeric[0], rng))):
        method = "RooImproperIntegrator1D"
    log(
        func,
        INFO,
        "NumericIntegration",
        f"RooRealIntegral::init({label or integral_name(func, names, rng)}) "
        f"using numeric integrator {method} to calculate Int({','.join(numeric)})",
    )


def integral(func: Any, names: frozenset[str], ctx: dict[str, Any], rng: Any = None) -> Any:
    """The integral of ``func`` over the variables ``names``, over each part of a range ``a,b``."""
    names = frozenset(names) & func.dependents()
    if not names:
        return func.compute(ctx)
    parts = [one for one in str(rng).split(",") if one] if rng else [None]
    total: Any = 0.0
    for part in parts:
        total = total + _over(func, names, ctx, part)
    return total


def _over(func: Any, names: frozenset[str], ctx: dict[str, Any], rng: Any) -> Any:
    """Closed form over what the class can do, numerically over the rest."""
    closed = func.analytic_names(names, rng) & names
    rest = [one.GetName() for one in func.leaves() if one.GetName() in names - closed]
    if not rest:
        return func.analytic(closed, ctx, rng)

    def inner(c: dict[str, Any]) -> Any:
        return func.analytic(closed, c, rng) if closed else func.compute(c)

    return numeric(func, rest, inner, ctx, rng)


def numeric(
    func: Any,
    rest: list[str],
    inner: Callable[[dict[str, Any]], Any],
    ctx: dict[str, Any],
    rng: Any,
) -> Any:
    """``inner`` integrated over the variables ``rest``: in one dimension by Romberg, in more
    by ROOT's adaptive cubature, as ``RooRealIntegral`` picks them."""
    bounds = [func.bounds(name, rng) for name in rest]
    if len(rest) > 1 and not any(np.isinf(b).any() for b in (np.array(bounds),)):
        return _cubature(func, rest, inner, ctx, bounds)
    return _nested(func, rest, inner, ctx, rng)


def _nested(
    func: Any,
    rest: list[str],
    inner: Callable[[dict[str, Any]], Any],
    ctx: dict[str, Any],
    rng: Any,
) -> Any:
    name, *others = rest
    low, high = func.bounds(name, rng)

    def along(points: np.ndarray[Any, Any]) -> Any:
        c = _expanded(ctx)
        c[name] = points
        values = _nested(func, others, inner, c, rng) if others else inner(c)
        return np.broadcast_to(values, np.broadcast(values, points).shape)

    return integrate_1d(along, low, high, func.GetName())


def _cubature(
    func: Any,
    rest: list[str],
    inner: Callable[[dict[str, Any]], Any],
    ctx: dict[str, Any],
    bounds: list[tuple[float, float]],
) -> Any:
    """``RooAdaptiveIntegratorND``, once for each event if the context has events."""
    from .cubature import integrate_nd

    shape, events = _events(ctx)
    lows, highs = [b[0] for b in bounds], [b[1] for b in bounds]
    found = np.empty(shape)
    for index, c in events:
        found[index] = integrate_nd(_integrand(inner, c, rest), lows, highs, vectorized=True).value
    return found if shape else float(found)


def _events(ctx: dict[str, Any]) -> tuple[Any, list[tuple[Any, dict[str, Any]]]]:
    """The shape of the context's columns, and each event's own context - one, if it has none."""
    arrays = {k: v for k, v in ctx.items() if isinstance(v, np.ndarray) and v.ndim}
    if not arrays:
        return (), [((), dict(ctx))]
    shape = np.broadcast_shapes(*(v.shape for v in arrays.values()))
    events = []
    for index in np.ndindex(*shape):
        c = dict(ctx)
        c.update({k: np.broadcast_to(v, shape)[index] for k, v in arrays.items()})
        events.append((index, c))
    return shape, events


def _integrand(inner: Callable[[dict[str, Any]], Any], ctx: dict[str, Any], rest: list[str]) -> Any:
    """``inner`` at points of the variables ``rest``, the rest of one event's values as ``ctx``."""

    def at(points: np.ndarray[Any, Any]) -> Any:
        cc = dict(ctx)
        cc.update({name: points[:, i] for i, name in enumerate(rest)})
        return np.broadcast_to(inner(cc), (len(points),))

    return at
