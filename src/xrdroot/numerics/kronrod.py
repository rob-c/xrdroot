"""GSL's Gauss-Kronrod rules - ``gsl_integration_qk`` - and its error rescaling.

One rule over ``[a, b]``: the Kronrod estimate, its error from the embedded
Gauss rule scaled as QUADPACK scales it, the integral of ``|f|`` and of
``|f - mean|``, evaluated in QUADPACK's order so the rounding is GSL's.
"""

from __future__ import annotations

import math
from collections.abc import Callable

__all__ = ["DBL_EPSILON", "DBL_MIN", "qk21", "rescale_error"]

DBL_EPSILON = 2.2204460492503131e-16
DBL_MIN = 2.2250738585072014e-308

#: The 21-point Kronrod abscissae, the 10-point Gauss weights, the Kronrod weights.
XGK21 = (0.995657163025808080735527280689003, 0.973906528517171720077964012084452,
         0.930157491355708226001207180059508, 0.865063366688984510732096688423493,
         0.780817726586416897063717578345042, 0.679409568299024406234327365114874,
         0.562757134668604683339000099272694, 0.433395394129247190799265943165784,
         0.294392862701460198131126603103866, 0.148874338981631210884826001129720, 0.0)  # fmt: skip
WG21 = (0.066671344308688137593568809893332, 0.149451349150580593145776339657697,
        0.219086362515982043995534934228163, 0.269266719309996355091226921569469,
        0.295524224714752870173892994651338)  # fmt: skip
WGK21 = (0.011694638867371874278064396062192, 0.032558162307964727478818972459390,
         0.054755896574351996031381300244580, 0.075039674810919952767043140916190,
         0.093125454583697605535065465083366, 0.109387158802297641899210590325805,
         0.123491976262065851077958109831074, 0.134709217311473325928054001771707,
         0.142775938577060080797094273138717, 0.147739104901338491374841515972068,
         0.149445554002916905664936468389821)  # fmt: skip

Rule = tuple[float, float, float, float]


def rescale_error(err: float, result_abs: float, result_asc: float) -> float:
    """QUADPACK's error estimate from the Gauss-Kronrod difference."""
    err = abs(err)
    if result_asc != 0 and err != 0:
        scale = math.pow(200 * err / result_asc, 1.5)
        err = result_asc * scale if scale < 1 else result_asc
    if result_abs > DBL_MIN / (50 * DBL_EPSILON):
        err = max(err, 50 * DBL_EPSILON * result_abs)
    return err


def qk(xgk: tuple[float, ...], wg: tuple[float, ...], wgk: tuple[float, ...],
       f: Callable[[float], float], a: float, b: float) -> Rule:  # fmt: skip
    """``gsl_integration_qk``: result, error, integral of ``|f|``, of ``|f - mean|``."""
    n = len(xgk)
    center, half = 0.5 * (a + b), 0.5 * (b - a)
    f_center = f(center)
    result_gauss = f_center * wg[n // 2 - 1] if n % 2 == 0 else 0.0
    result_kronrod = f_center * wgk[n - 1]
    result_abs = abs(result_kronrod)
    fv1, fv2 = [0.0] * n, [0.0] * n
    for j in range((n - 1) // 2):
        jtw = j * 2 + 1
        abscissa = half * xgk[jtw]
        fval1, fval2 = f(center - abscissa), f(center + abscissa)
        fv1[jtw], fv2[jtw] = fval1, fval2
        result_gauss += wg[j] * (fval1 + fval2)
        result_kronrod += wgk[jtw] * (fval1 + fval2)
        result_abs += wgk[jtw] * (abs(fval1) + abs(fval2))
    for j in range(n // 2):
        jtwm1 = j * 2
        abscissa = half * xgk[jtwm1]
        fval1, fval2 = f(center - abscissa), f(center + abscissa)
        fv1[jtwm1], fv2[jtwm1] = fval1, fval2
        result_kronrod += wgk[jtwm1] * (fval1 + fval2)
        result_abs += wgk[jtwm1] * (abs(fval1) + abs(fval2))
    mean = result_kronrod * 0.5
    result_asc = wgk[n - 1] * abs(f_center - mean)
    for j in range(n - 1):
        result_asc += wgk[j] * (abs(fv1[j] - mean) + abs(fv2[j] - mean))
    err = (result_kronrod - result_gauss) * half
    result_kronrod *= half
    result_abs *= abs(half)
    result_asc *= abs(half)
    return result_kronrod, rescale_error(err, result_abs, result_asc), result_abs, result_asc


def qk21(f: Callable[[float], float], a: float, b: float) -> Rule:
    return qk(XGK21, WG21, WGK21, f, a, b)
