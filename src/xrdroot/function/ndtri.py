"""``ROOT::Math::Cephes::ndtri``: the inverse of the normal distribution, as MathCore has it.

``normal_quantile`` and ``normal_quantile_c`` are this, and the normal upper
tail ``normal_cdf_c`` is Cephes's ``erfc`` past one and ``1 - erf`` inside:
MathCore's own, to the last bit, where the standard library rounds
differently.
"""

from __future__ import annotations

import math

from .analytic import SQRT2, _p1evl, _polevl, erf, erfc

__all__ = ["ndtri", "normal_cdf_c", "normal_quantile", "normal_quantile_c"]

#: ``sqrt(2 pi)``.
S2PI = 2.50662827463100050242e0
#: ``exp(-2)``: below it, and above one less it, the tails' expansions.
TAIL = 0.13533528323661269189

P0 = (-5.99633501014107895267e1, 9.80010754185999661536e1, -5.66762857469070293439e1,
      1.39312609387279679503e1, -1.23916583867381258016e0)  # fmt: skip
Q0 = (1.95448858338141759834e0, 4.67627912898881538453e0, 8.63602421390890590575e1,
      -2.25462687854119370527e2, 2.00260212380060660359e2, -8.20372256168333339912e1,
      1.59056225126211695515e1, -1.18331621121330003142e0)  # fmt: skip
P1 = (4.05544892305962419923e0, 3.15251094599893866154e1, 5.71628192246421288162e1,
      4.40805073893200834700e1, 1.46849561928858024014e1, 2.18663306850790267539e0,
      -1.40256079171354495875e-1, -3.50424626827848203418e-2,
      -8.57456785154685413611e-4)  # fmt: skip
Q1 = (1.57799883256466749731e1, 4.53907635128879210584e1, 4.13172038254672030440e1,
      1.50425385692907503408e1, 2.50464946208309415979e0, -1.42182922854787788574e-1,
      -3.80806407691578277194e-2, -9.33259480895457427372e-4)  # fmt: skip
P2 = (3.23774891776946035970e0, 6.91522889068984211695e0, 3.93881025292474443415e0,
      1.33303460815807542389e0, 2.01485389549179081538e-1, 1.23716634817820021358e-2,
      3.01581553508235416007e-4, 2.65806974686737550832e-6, 6.23974539184983293730e-9)  # fmt: skip
Q2 = (6.02427039364742014255e0, 3.67983563856160859403e0, 1.37702099489081330271e0,
      2.16236993594496635890e-1, 1.34204006088543189037e-2, 3.28014464682127739104e-4,
      2.89247864745380683936e-6, 6.79019408009981274425e-9)  # fmt: skip


def ndtri(y0: float) -> float:
    """The ``x`` below which a standard normal falls with chance ``y0``."""
    if y0 <= 0.0:
        return -math.inf
    if y0 >= 1.0:
        return math.inf
    upper = y0 > 1.0 - TAIL
    y = 1.0 - y0 if upper else y0
    if y > TAIL:
        y -= 0.5
        y2 = y * y
        return (y + y * (y2 * _polevl(y2, P0) / _p1evl(y2, Q0))) * S2PI
    x = math.sqrt(-2.0 * math.log(y))
    x0 = x - math.log(x) / x
    z = 1.0 / x
    far = x >= 8.0
    x1 = z * _polevl(z, P2 if far else P1) / _p1evl(z, Q2 if far else Q1)
    x = x0 - x1
    return x if upper else -x


def normal_quantile(z: float, sigma: float = 1.0) -> float:
    return sigma * ndtri(z)


def normal_quantile_c(z: float, sigma: float = 1.0) -> float:
    return -sigma * ndtri(z)


def normal_cdf_c(x: float, sigma: float = 1.0, x0: float = 0.0) -> float:
    z = (x - x0) / (sigma * SQRT2)
    return 0.5 * erfc(z) if z > 1.0 else 0.5 * (1.0 - erf(z))
