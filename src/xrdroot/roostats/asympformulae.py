"""The asymptotic formulae: p-values from the ratio on the data and on the Asimov data.

With ``sqrt(q)`` and ``sqrt(q_A)``: one-sided, the null's p-value is the
normal tail beyond ``sqrt(q)`` and the alternate's the normal
distribution at ``sqrt(q_A) - sqrt(q)``; two-sided, both tails. The
``tilde`` forms, for a parameter bounded at the alternate, take over where
``q`` is beyond ``q_A``. ``GetExpectedPValues`` turns an observed pair into
the pair expected at ``nsigma`` from the median - for the bands of a limit.
"""

from __future__ import annotations

import math
from collections.abc import Callable

__all__ = ["expected_p_values", "p_values"]


def _cdf_c(x: float) -> float:
    from ..function.ndtri import normal_cdf_c

    return normal_cdf_c(x)


def _cdf(x: float) -> float:
    from ..function.analytic import gaussian_cdf

    return gaussian_cdf(x)


def p_values(qmu: float, qmu_a: float, one_sided: bool, discovery: bool, qtilde: bool,
             tol: float) -> tuple[float, float]:  # fmt: skip
    """The null's and the alternate's p-values - ``CLs+b`` and ``CLb`` - from ``q`` and ``q_A``."""
    root = math.sqrt(qmu) if qmu > 0 else 0.0
    root_a = math.sqrt(qmu_a) if qmu_a > 0 else 0.0
    if one_sided or discovery:
        pnull, palt = _cdf_c(root), _cdf(root_a - root)
    else:
        pnull = 2.0 * _cdf_c(root)
        palt = _cdf_c(root + root_a) + _cdf_c(root - root_a)
    if qtilde and qmu > qmu_a and (qmu_a > 0 or qmu > tol):  # to avoid the case 0/0
        tail, shift = (qmu + qmu_a) / (2 * root_a), (qmu - qmu_a) / (2 * root_a)
        if one_sided:
            pnull, palt = _cdf_c(tail), _cdf_c(shift)
        else:
            pnull, palt = _cdf_c(root) + _cdf_c(tail), _cdf_c(root_a + root) + _cdf_c(shift)
    return pnull, palt


def _bracket(fn: Callable[[float], float], xmin: float, xmax: float,
             npx: int = 100) -> tuple[float, float, float]:  # fmt: skip
    """``MinimStep`` for a root: the first grid step the sign changes over, and its middle - or,
    none found, said on the error stream, the empty range ``(1, 0)``."""
    import sys

    dx = (xmax - xmin) / (npx - 1)
    last, value = xmin, fn(xmin)
    if value == 0:
        return xmin, xmin, xmin
    for i in range(1, npx):
        x = xmin + i * dx
        y = fn(x)
        if y == 0:
            return x, x, x
        if math.copysign(1.0, y) * math.copysign(1.0, value) < 0:
            return 0.5 * (last + x), last, x
        last, value = x, y
    sys.stderr.write("Info in <BrentMethods::MinimStep>: Grid search failed to find a root in the "
                     " interval \nInfo in <BrentMethods::MinimStep>: xmin = "
                     f"{xmin:.6g} xmax = {xmax:.6g} npts = {npx}\n")  # fmt: skip
    return 0.0, 1.0, 0.0


def _root(fn: Callable[[float], float], low: float, high: float) -> tuple[bool, float]:
    """``BrentRootFinder::Solve``: a bracket on a grid of a hundred, then Brent's method on
    ``|f|`` in it - ten times over, narrowing, until it converges."""
    from ..numerics.minimize1d import minim_brent

    xmin, xmax, x = low, high, 0.0
    for _ in range(11):
        x, xmin, xmax = _bracket(fn, xmin, xmax)
        if xmin > xmax:
            return False, 0.0
        x, ok, xmin, xmax = minim_brent(lambda t: abs(fn(t)), xmin, xmax, x, 1e-8, 1e-10, 100)
        if ok:
            return True, x
    return False, x


def expected_p_values(pnull: float, palt: float, nsigma: float, use_cls: bool,
                      one_sided: bool = True) -> float:  # fmt: skip
    """``GetExpectedPValues``: the p-value - ``CLs``, or ``CLs+b`` - expected ``nsigma`` from the
    median, for an observed ``pnull`` and ``palt``; -1 where it cannot be found."""
    from ..roofit.messages import ERROR, log

    if one_sided:
        root = -_quantile(pnull)
        root_a = _quantile(palt) + root
        clsplusb = _cdf_c(root_a - nsigma)
        if not use_cls:
            return clsplusb
        clb = _cdf(nsigma)
        return -1.0 if clb == 0 else clsplusb / clb
    root_t = -_quantile(0.5 * pnull)
    if root_t == 0:
        return -1.0
    found, root_ta = _root(_palt(root_t, palt, -1), 0.0, 20.0)
    if found:
        found, value = _root(_palt(root_ta, _cdf(nsigma), 1), 0.0, 20.0)
        if found:
            return 2 * _cdf_c(value)
    log(None, ERROR, "Eval", "Error finding expected p-values - return -1")
    return -1.0


def _quantile(p: float) -> float:
    from ..function.ndtri import ndtri

    return ndtri(p)


def _palt(offset: float, pval: float, case: int) -> Callable[[float], float]:
    """``PaltFunction``: the two tails about ``offset``, less ``pval``."""
    return lambda x: _cdf_c(x + offset) + _cdf_c(case * (x - offset)) - pval
