"""``TF1::Integral`` for the predefined shapes: in closed form, as ROOT integrates them.

ROOT does not integrate ``gaus``, ``expo``, ``landau`` or a ``polN``
numerically. A function whose formula is one of those shapes alone carries
a number - 100 for a Gaussian, 200 for an exponential, 300 plus the degree
for a polynomial, 400 for a Landau - and ``TF1::Integral`` hands such a
function to ``AnalyticalIntegral``, which writes the integral down: the
difference of two Gaussian cumulative distributions, ``exp(a) expm1(b - a)``
over the slope, the polynomial's antiderivative term by term, the
difference of two of CERNLIB's ``DISLAN`` values. ``TH1::FillRandom`` and
``TF1::GetRandom`` build their tables from those integrals, so a histogram
filled from ``gaus`` with ROOT's seed is ROOT's histogram only if the table
is these numbers.

So each is ROOT's operation for operation, in Python's own doubles - the
error function is the Cephes one ``ROOT::Math::erf`` is, coefficients and
branch points and all, not the C library's - and each is one number at a
time, because the C library's ``exp`` is what ROOT calls and NumPy's
vectorised one may round the last bit differently.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

__all__ = ["erf", "erfc", "gaussian_cdf", "landau_cdf", "integral"]

#: Cephes's ``MAXLOG``, past which ``exp(-x*x)`` underflows and ``erfc`` is its limit.
MAXLOG = 709.782712893383973096206318587
#: ``erfc``'s rational approximation for ``1 <= |x| < 8``, numerator then denominator.
ERF_P = (
    2.46196981473530512524e-10,
    5.64189564831068821977e-1,
    7.46321056442269912687e0,
    4.86371970985681366614e1,
    1.96520832956077098242e2,
    5.26445194995477358631e2,
    9.34528527171957607540e2,
    1.02755188689515710272e3,
    5.57535335369399327526e2,
)
ERF_Q = (
    1.32281951154744992508e1,
    8.67072140885989742329e1,
    3.54937778887819891062e2,
    9.75708501743205489753e2,
    1.82390916687909736289e3,
    2.24633760818710981792e3,
    1.65666309194161350182e3,
    5.57535340817727675546e2,
)
#: ``erfc``'s approximation from eight widths out.
ERF_R = (
    5.64189583547755073984e-1,
    1.27536670759978104416e0,
    5.01905042251180477414e0,
    6.16021097993053585195e0,
    7.40974269950448939160e0,
    2.97886665372100240670e0,
)
ERF_S = (
    2.26052863220117276590e0,
    9.39603524938001434673e0,
    1.20489539808096656605e1,
    1.70814450747565897222e1,
    9.60896809063285878198e0,
    3.36907645100081516050e0,
)
#: ``erf``'s approximation inside one width, in the square of the argument.
ERF_T = (
    9.60497373987051638749e0,
    9.00260197203842689217e1,
    2.23200534594684319226e3,
    7.00332514112805075473e3,
    5.55923013010394962768e4,
)
ERF_U = (
    3.35617141647503099647e1,
    5.21357949780152679795e2,
    4.59432382970980127987e3,
    2.26290000613890934246e4,
    4.92673942608635921086e4,
)
#: ``sqrt(2)``, which ``normal_cdf`` scales the width by.
SQRT2 = 1.41421356237309504880
#: The number ``TFormula`` gives each predefined shape, which ``AnalyticalIntegral`` reads.
GAUSSIAN, EXPONENTIAL, POLYNOMIAL, LANDAU = 100, 200, 300, 400

#: ``DISLAN``: for each region of ``v``, where it ends and the numerator and
#: denominator of its rational function, lowest power first. The first two
#: regions and the last are not rational functions of ``v`` and are below.
DISLAN = (
    (
        1.0,
        (0.2868328584e0, 0.3564363231e0, 0.1523518695e0, 0.2251304883e-1),
        (1.0, 0.6191136137e0, 0.1720721448e0, 0.2278594771e-1),
    ),
    (
        4.0,
        (0.2868329066e0, 0.3003828436e0, 0.9950951941e-1, 0.8733827185e-2),
        (1.0, 0.4237190502e0, 0.1095631512e0, 0.8693851567e-2),
    ),
)
#: The regions ``DISLAN`` takes in ``u = 1/v``, the same way.
DISLAN_INVERSE = (
    (
        12.0,
        (0.1000351630e1, 0.4503592498e1, 0.1085883880e2, 0.7536052269e1),
        (1.0, 0.5539969678e1, 0.1933581111e2, 0.2721321508e2),
    ),
    (
        50.0,
        (0.1000006517e1, 0.4909414111e2, 0.8505544753e2, 0.1532153455e3),
        (1.0, 0.5009928881e2, 0.1399819104e3, 0.4200002909e3),
    ),
    (
        300.0,
        (0.1000000983e1, 0.1329868456e3, 0.9162149244e3, -0.9605054274e3),
        (1.0, 0.1339887843e3, 0.1055990413e4, 0.5532224619e3),
    ),
)
#: The left tail's rational function, and the two tail series.
DISLAN_LEFT = (
    (0.2514091491e0, -0.6250580444e-1, 0.1458381230e-1, -0.2108817737e-2, 0.7411247290e-3),
    (1.0, -0.5571175625e-2, 0.6225310236e-1, -0.3137378427e-2, 0.1931496439e-2),
)
DISLAN_A1 = (-0.4583333333e0, 0.6675347222e0, -0.1641741416e1)
DISLAN_A2 = (1.0, -0.4227843351e0, -0.2043403138e1)


def _polevl(x: float, coefficients: Sequence[float]) -> float:
    """Cephes's ``polevl``: highest power first."""
    total = coefficients[0]
    for coefficient in coefficients[1:]:
        total = total * x + coefficient
    return total


def _p1evl(x: float, coefficients: Sequence[float]) -> float:
    """Cephes's ``p1evl``: the same, with a leading coefficient of one unwritten."""
    total = x + coefficients[0]
    for coefficient in coefficients[1:]:
        total = total * x + coefficient
    return total


def erfc(a: float) -> float:
    """``ROOT::Math::erfc``, which is Cephes's: ``1 - erf`` inside one, rational past it."""
    x = abs(a)
    if x < 1.0:
        return 1.0 - erf(a)
    z = -a * a
    if z < -MAXLOG:
        return 2.0 if a < 0 else 0.0
    z = math.exp(z)
    if x < 8.0:
        y = (z * _polevl(x, ERF_P)) / _p1evl(x, ERF_Q)
    else:
        y = (z * _polevl(x, ERF_R)) / _p1evl(x, ERF_S)
    # Cephes also checks for y underflowing to zero here, which a double's
    # subnormals keep from happening before exp(-x*x) itself does, above.
    return 2.0 - y if a < 0 else y


def erf(x: float) -> float:
    """``ROOT::Math::erf``, which is Cephes's: ``1 - erfc`` past one, a rational function inside."""
    if abs(x) > 1.0:
        return 1.0 - erfc(x)
    z = x * x
    return x * _polevl(z, ERF_T) / _p1evl(z, ERF_U)


def gaussian_cdf(x: float, sigma: float = 1.0, x0: float = 0.0) -> float:
    """``ROOT::Math::normal_cdf``: from ``erfc`` in the far left tail, else from ``erf``."""
    z = (x - x0) / (sigma * SQRT2)
    if z < -1.0:
        return 0.5 * erfc(-z)
    return 0.5 * (1.0 + erf(z))


def _nested(coefficients: Sequence[float], v: float) -> float:
    """``c0 + (c1 + (c2 + ...)*v)*v``, nested as ``DISLAN`` nests it."""
    total = coefficients[-1]
    for c in reversed(coefficients[:-1]):
        total = c + total * v
    return total


def _far_left(v: float) -> float:
    """``DISLAN`` below ``-5.5``: the asymptotic series in ``exp(v + 1)``."""
    u = math.exp(v + 1)
    series = 1 + (DISLAN_A1[0] + (DISLAN_A1[1] + DISLAN_A1[2] * u) * u) * u
    return 0.3989422803 * math.exp(-1.0 / u) * math.sqrt(u) * series


def _left(v: float) -> float:
    """``DISLAN`` from ``-5.5`` to ``-1``: a ratio in ``v`` times ``exp(-u)/sqrt(u)``."""
    u = math.exp(-v - 1)
    return (math.exp(-u) / math.sqrt(u)) * (_nested(DISLAN_LEFT[0], v) / _nested(DISLAN_LEFT[1], v))


def _far_right(v: float) -> float:
    """``DISLAN`` from 300 on: one minus the asymptotic series."""
    u = 1.0 / (v - v * math.log(v) / (v + 1))
    return 1 - (DISLAN_A2[0] + (DISLAN_A2[1] + DISLAN_A2[2] * u) * u) * u


def _dislan_tails(v: float) -> float | None:
    """``DISLAN`` left of ``-1`` and past 300, which are not a ratio in ``v`` or ``1/v``."""
    if v < -5.5:
        return _far_left(v)
    if v < -1:
        return _left(v)
    return _far_right(v) if v >= 300 else None


def landau_cdf(x: float, xi: float = 1.0, x0: float = 0.0) -> float:
    """``ROOT::Math::landau_cdf``: CERNLIB's ``DISLAN``, region by region."""
    v = (x - x0) / xi
    tail = _dislan_tails(v)
    if tail is not None:
        return tail
    for end, top, bottom in DISLAN:
        if v < end:
            return _nested(top, v) / _nested(bottom, v)
    u = 1.0 / v
    for end, top, bottom in DISLAN_INVERSE[:-1]:
        if v < end:
            return _nested(top, u) / _nested(bottom, u)
    _end, top, bottom = DISLAN_INVERSE[-1]  # below 300: past it is a tail, above
    return _nested(top, u) / _nested(bottom, u)


def _gaussian(p: Sequence[float], a: float, b: float, normalized: bool) -> float:
    amp, mean, sigma = p[0], p[1], p[2]
    if sigma <= 0:
        return math.nan
    difference = gaussian_cdf(b, sigma, mean) - gaussian_cdf(a, sigma, mean)
    if normalized:
        return amp * difference
    return amp * math.sqrt(2 * math.pi) * sigma * difference


def _exponential(p: Sequence[float], a: float, b: float, normalized: bool) -> float:
    if p[1] == 0:
        return math.exp(p[0]) * (b - a)
    ea, eb = p[0] + p[1] * a, p[0] + p[1] * b
    return math.exp(ea) * math.expm1(eb - ea) / p[1]


def _landau(p: Sequence[float], a: float, b: float, normalized: bool) -> float:
    amp, mean, sigma = p[0], p[1], p[2]
    if sigma <= 0:
        return math.nan
    difference = landau_cdf(b, sigma, mean) - landau_cdf(a, sigma, mean)
    return amp * difference if normalized else amp * sigma * difference


#: ``AnalyticalIntegral``'s closed forms, by the shape's number.
_FORMS: dict[int, Callable[[Sequence[float], float, float, bool], float]] = {
    GAUSSIAN: _gaussian,
    EXPONENTIAL: _exponential,
    LANDAU: _landau,
}


def integral(number: int, p: Sequence[float], a: float, b: float, normalized: bool) -> float:
    """``AnalyticalIntegral``: a predefined shape's integral from ``a`` to ``b``, or NaN.

    NaN is ROOT's answer for a shape it has no closed form for, and for a
    width that is not positive; ``TF1::Integral`` then integrates numerically.
    """
    if POLYNOMIAL <= number < LANDAU:
        return _polynomial(p, number - POLYNOMIAL, float(a), float(b))
    form = _FORMS.get(number)
    return math.nan if form is None else form(p, float(a), float(b), normalized)


def _polynomial(p: Sequence[float], degree: int, a: float, b: float) -> float:
    """``sum p_i/(i+1) (b^(i+1) - a^(i+1))``, added up in ROOT's order."""
    result = 0.0
    for i in range(degree + 1):
        result += p[i] / (i + 1) * (math.pow(b, i + 1) - math.pow(a, i + 1))
    return result
