"""The probabilities ROOT's tests hand back: ``TMath::Prob`` and ``TMath::KolmogorovProb``.

A chi-square test and a Kolmogorov test each end in one number, the chance of
a disagreement at least as large as the one seen if the two things compared
were really the same. Both numbers come from a function ROOT keeps in
``TMath``, and a p-value that differs from ROOT's in its tenth digit is a
p-value nobody can check against the paper it was quoted in. So these are
ROOT's own: the incomplete gamma function is the Cephes routine ROOT's
MathCore carries, its logarithm of the gamma function is Cephes's too rather
than the C library's, and the Kolmogorov sum is CERNLIB's ``PROBKL`` as Rene
Brun translated it - with its constants, its ranges and its rounding. SciPy
would give the same answers to a few parts in 10^15 and is not a dependency.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = [
    "incomplete_gamma",
    "incomplete_gamma_c",
    "kolmogorov_prob",
    "log_gamma",
    "nint",
    "prob",
    "smooth_array",
]

#: Cephes's ``MACHEP``: the relative size of the last term that still
#: changes a series or a continued fraction.
MACHEP = 1.11022302462515654042363166809e-16

#: Cephes's ``MAXLOG``: the largest logarithm whose exponential is a double.
MAXLOG = 709.782712893383973096206318587

#: Cephes's ``MAXLGM``: past this the logarithm of the gamma function is infinite.
MAXLGM = 2.556348e305

#: Where the continued fraction rescales its terms so they never overflow.
BIG = 4.503599627370496e15
BIGINV = 2.22044604925031308085e-16

#: ``log(sqrt(2 pi))``, as Cephes spells it.
LS2PI = 0.91893853320467274178

#: Stirling's series for the logarithm of the gamma function, and the two
#: rational functions Cephes uses for it between two and three.
STIRLING = (
    8.11614167470508450300e-4,
    -5.95061904284301438324e-4,
    7.93650340457716943945e-4,
    -2.77777777730099687205e-3,
    8.33333333333331927722e-2,
)
NUMERATOR = (
    -1.37825152569120859100e3,
    -3.88016315134637840924e4,
    -3.31612992738871184744e5,
    -1.16237097492762307383e6,
    -1.72173700820839662146e6,
    -8.53555664245765465627e5,
)
DENOMINATOR = (
    -3.51815701436523470549e2,
    -1.70642106651881159223e4,
    -2.20528590553854454839e5,
    -1.13933444367982507207e6,
    -2.53252307177582951285e6,
    -2.01889141433532773231e6,
)

#: ``PROBKL``'s constants: ``sqrt(2 pi)`` to the digits it was written with,
#: and ``-pi**2 / 8`` times one, nine and twenty-five.
ROOT_TWO_PI = 2.50662827
THETA = (-1.2337005501361697, -11.103304951225528, -30.842513753404244)

#: The exponents of the first four terms of the Kolmogorov sum, over ``z**2``.
TERMS = (-2.0, -8.0, -18.0, -32.0)


def nint(x: float) -> int:
    """``TMath::Nint``: the nearest whole number, a half going to the even one."""
    if x >= 0:
        found = int(x + 0.5)
        return found - 1 if found & 1 and x + 0.5 == found else found
    found = int(x - 0.5)
    return found + 1 if found & 1 and x - 0.5 == found else found


def _polynomial(x: float, coefficients: tuple[float, ...]) -> float:
    """Cephes's ``polevl``: the polynomial with these coefficients, highest first."""
    total = coefficients[0]
    for coefficient in coefficients[1:]:
        total = total * x + coefficient
    return total


def _monic(x: float, coefficients: tuple[float, ...]) -> float:
    """Cephes's ``p1evl``: the same with a leading coefficient of one left unwritten."""
    total = x + coefficients[0]
    for coefficient in coefficients[1:]:
        total = total * x + coefficient
    return total


def _log_gamma_small(x: float) -> float:
    """``lgam`` below thirteen: shifted into [2, 3) and a rational function there."""
    z, p, u = 1.0, 0.0, x
    while u >= 3.0:
        p -= 1.0
        u = x + p
        z *= u
    while u < 2.0:
        z /= u
        p += 1.0
        u = x + p
    if u == 2.0:
        return math.log(z)
    x = x + (p - 2.0)
    return math.log(z) + x * _polynomial(x, NUMERATOR) / _monic(x, DENOMINATOR)


def log_gamma(x: float) -> float:
    """Cephes's ``lgam`` for a positive ``x``, the logarithm of the gamma function.

    It is what ROOT's incomplete gamma function divides by, and it is not
    the C library's ``lgamma``, which can differ from it in the last bit - so
    it is here too, for the numbers to be ROOT's. Only the positive half is
    needed by anything here, and a number that is not positive is refused.
    """
    if not x > 0:
        raise ValueError(
            f"the logarithm of the gamma function is taken here of a positive number, not {x}"
        )
    if x < 13.0:
        return _log_gamma_small(x)
    if x > MAXLGM:
        return math.inf
    q = (x - 0.5) * math.log(x) - x + LS2PI
    if x > 1.0e8:
        return q
    p = 1.0 / (x * x)
    if x >= 1000.0:
        return q + ((7.9365079365079365079365e-4 * p - 2.7777777777777777777778e-3) * p
                    + 0.0833333333333333333333) / x  # fmt: skip
    return q + _polynomial(p, STIRLING) / x


def _prefactor(a: float, x: float) -> float:
    """``x**a * exp(-x) / Gamma(a)``, or zero where its logarithm is below a double's."""
    found = a * math.log(x) - x - log_gamma(a)
    return 0.0 if found < -MAXLOG else math.exp(found)


def incomplete_gamma(a: float, x: float) -> float:
    """``ROOT::Math::inc_gamma``, Cephes's ``igam``: the regularised lower incomplete gamma.

    ``P(a, x)``, the chance that a gamma variable of shape ``a`` is below
    ``x``: a power series when ``x`` is small against ``a``, one minus the
    continued fraction of :func:`incomplete_gamma_c` when it is not.
    """
    if a <= 0:
        return 1.0
    if x <= 0:
        return 0.0
    if x > 1.0 and x > a:
        return 1.0 - incomplete_gamma_c(a, x)
    scale = _prefactor(a, x)
    if scale == 0.0:
        return 0.0
    r, c, total = a, 1.0, 1.0
    while True:
        r += 1.0
        c *= x / r
        total += c
        if not c / total > MACHEP:
            return total * scale / a


def incomplete_gamma_c(a: float, x: float) -> float:
    """``ROOT::Math::inc_gamma_c``, Cephes's ``igamc``: the upper one, ``1 - P(a, x)``.

    A continued fraction, evaluated as Cephes evaluates it - rescaled
    whenever its terms grow past ``2**52`` - and handed to the series of
    :func:`incomplete_gamma` where that converges faster.
    """
    if a <= 0:
        return 0.0
    if x <= 0:
        return 1.0
    if x < 1.0 or x < a:
        return 1.0 - incomplete_gamma(a, x)
    scale = _prefactor(a, x)
    if scale == 0.0:
        return 0.0
    return _continued_fraction(a, x) * scale


def _continued_fraction(a: float, x: float) -> float:
    """The continued fraction of ``igamc``, to Cephes's ``MACHEP``."""
    y, z, c = 1.0 - a, x + (1.0 - a) + 1.0, 0.0
    pkm2, qkm2, pkm1 = 1.0, x, x + 1.0
    qkm1 = z * x
    answer, change = pkm1 / qkm1, 1.0
    while change > MACHEP:
        c += 1.0
        y += 1.0
        z += 2.0
        yc = y * c
        pk, qk = pkm1 * z - pkm2 * yc, qkm1 * z - qkm2 * yc
        # A denominator of zero keeps the answer and carries on, as Cephes does.
        r = pk / qk if qk else answer
        change = abs((answer - r) / r) if qk else 1.0
        answer = r
        pkm2, pkm1, qkm2, qkm1 = pkm1, pk, qkm1, qk
        if abs(pk) > BIG:
            pkm2, pkm1, qkm2, qkm1 = pkm2 * BIGINV, pkm1 * BIGINV, qkm2 * BIGINV, qkm1 * BIGINV
    return answer


def prob(chi2: float, ndf: int) -> float:
    """``TMath::Prob``: the chance of a chi-square at least ``chi2`` on ``ndf`` degrees of freedom.

        >>> prob(0.0, 3), prob(2.0, 2)
        (1.0, 0.36787944117144233)

    That is ``1 - P(ndf/2, chi2/2)``, the upper incomplete gamma function. As
    ROOT has it, no degrees of freedom - or fewer - gives zero, and so does a
    chi-square below zero; ``ndf`` is a whole number, cut to one as C++
    converts a double to an ``Int_t``.
    """
    freedom = int(ndf)
    if freedom <= 0 or chi2 < 0:
        return 0.0
    if chi2 == 0:
        return 1.0
    return incomplete_gamma_c(0.5 * freedom, 0.5 * chi2)


def kolmogorov_prob(z: float) -> float:
    """``TMath::KolmogorovProb``: the chance of a Kolmogorov distance of at least ``z``.

        >>> kolmogorov_prob(0.1), round(kolmogorov_prob(1.36), 4)
        (1.0, 0.0494)

    ``2 * sum((-1)**(j-1) * exp(-2 j**2 z**2))``, ``z`` being the distance
    times the root of the number of entries. Below 0.2 it is one; below 0.755
    the theta-function inversion is used, which needs three terms there; up
    to 6.8116 the sum itself, of as many terms as ``Nint(3 / z)`` says; and
    past that it is zero, as CERNLIB's ``PROBKL`` has it.
    """
    u = abs(z)
    if u < 0.2:
        return 1.0
    if u < 0.755:
        v = 1.0 / (u * u)
        return (
            1
            - ROOT_TWO_PI
            * (math.exp(THETA[0] * v) + math.exp(THETA[1] * v) + math.exp(THETA[2] * v))
            / u
        )
    if u < 6.8116:
        v = u * u
        count = max(1, nint(3.0 / u))
        r = [math.exp(TERMS[j] * v) if j < count else 0.0 for j in range(4)]
        return 2 * (r[0] - r[1] + r[2] - r[3])
    return 0.0


# -- TH1::SmoothArray ---------------------------------------------------------


def _median(values: list[float]) -> float:
    """``TMath::Median`` of three or five numbers: the middle one, once they are in order."""
    return sorted(values)[len(values) // 2]


def _ends_of_three(zz: list[float]) -> None:
    """The ends a running median of three does not reach, from the line through the next two."""
    last = len(zz) - 1
    zz[0] = _median([zz[1], zz[0], 3 * zz[1] - 2 * zz[2]])
    zz[last] = _median([zz[last - 1], zz[last], 3 * zz[last - 1] - 2 * zz[last - 2]])


def _running_medians(zz: list[float]) -> None:
    """353: running medians of three, then five, then three, in place.

    The median of five cannot reach the second point or the second to last,
    and takes a median of three there; after the last median of three the
    two ends stay as they are, as Friedman's paper says.
    """
    count = len(zz)
    for turn in range(3):
        yy = list(zz)
        width, first = (5, 2) if turn == 1 else (3, 1)
        for at in range(first, count - first):
            zz[at] = _median(yy[at - first : at - first + width])
        if turn == 0:
            _ends_of_three(zz)
        if turn == 1:
            zz[1] = _median(yy[0:3])
            zz[count - 2] = _median(yy[count - 3 : count])


def _flat_tops(zz: list[float]) -> list[float]:
    """Q: a quadratic put through the plateaus of three a median leaves at a peak or a dip."""
    yy = list(zz)
    for at in range(2, len(zz) - 2):
        if zz[at - 1] != zz[at] or zz[at] != zz[at + 1]:
            continue
        before, after = zz[at - 2] - zz[at], zz[at + 2] - zz[at]
        if before * after > 0:
            _rounded_off(zz, yy, at, -1 if abs(after) > abs(before) else 1)
    return yy


def _rounded_off(zz: list[float], yy: list[float], at: int, side: int) -> None:
    """The plateau at ``at`` given its shape back, leaning to the nearer ``side``."""
    far, near = zz[at - 2 * side], zz[at + 2 * side]
    yy[at] = -0.5 * far + zz[at] / 0.75 + near / 6.0
    yy[at + side] = 0.5 * (near - far) + zz[at]


def _hanned(yy: list[float]) -> list[float]:
    """H: a running mean weighted a quarter, a half and a quarter, the ends left as they are."""
    middle = [0.25 * yy[at - 1] + 0.5 * yy[at] + 0.25 * yy[at + 1] for at in range(1, len(yy) - 1)]
    return [yy[0], *middle, yy[-1]]


def _smoothed_once(xx: list[float]) -> list[float]:
    """One pass of 353QH twice: smooth, then smooth what the smoothing left, and add them."""
    zz = list(xx)
    _running_medians(zz)
    smooth = _hanned(_flat_tops(zz))
    rest = [x - z for x, z in zip(xx, smooth)]
    _running_medians(rest)
    rest = _hanned(_flat_tops(rest))
    if min(xx) < 0:
        return [r + z for r, z in zip(smooth, rest)]
    return [max(r + z, 0.0) for r, z in zip(smooth, rest)]


def smooth_array(values: Any, ntimes: int = 1) -> np.ndarray[Any, Any]:
    """``TH1::SmoothArray``: HBOOK's ``hsmoof``, J. Friedman's 353QH twice, ``ntimes`` over.

        >>> smooth_array([0.0, 1.0, 2.0, 3.0, 4.0]).tolist()
        [0.0, 1.0, 2.0, 3.0, 4.0]

    Running medians of three, five and three step over a single wild value
    rather than smearing it, a quadratic gives back the shape of a peak the
    medians flattened, and a running mean of a quarter, a half and a quarter
    finishes it; then the same again on what that left over, added back. If
    nothing given is negative, nothing that comes back is. Fewer than three
    values have no neighbours to smooth by and are refused, as ROOT does; a
    number of times below one smooths nothing, as ROOT's loop does not run.
    """
    xx = [float(value) for value in np.asarray(values, dtype=np.float64).ravel()]
    if len(xx) < 3:
        raise ValueError(f"smoothing needs at least 3 values, and {len(xx)} were given")
    for _ in range(int(ntimes)):
        xx = _smoothed_once(xx)
    return np.array(xx)
