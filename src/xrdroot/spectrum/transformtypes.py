"""The kinds of transform ``TSpectrumTransform`` knows, by ROOT's numbers.

The numbers are ``TSpectrumTransform``'s enum: they are compared by order in
ROOT's code - every kind from ``FOURIER_WALSH`` on is a mixed transform, from
``COS_WALSH`` on one made of a cosine or sine, past ``WALSH_HAAR`` one whose
coefficients are gathered two blocks at a time - so the order is part of
what they mean.
"""

from __future__ import annotations

import math

#: ``kTransformHaar`` and the rest, in ROOT's order.
HAAR, WALSH, COS, SIN, FOURIER, HARTLEY = 0, 1, 2, 3, 4, 5
FOURIER_WALSH, FOURIER_HAAR, WALSH_HAAR = 6, 7, 8
COS_WALSH, COS_HAAR, SIN_WALSH, SIN_HAAR = 9, 10, 11, 12

#: ``kTransformForward`` and ``kTransformInverse``.
FORWARD, INVERSE = 0, 1

#: The mixed transforms whose second half is Haar's, which widen their butterflies stage by stage.
HAAR_MIXED = frozenset({FOURIER_HAAR, WALSH_HAAR, COS_HAAR, SIN_HAAR})

#: The mixed transforms made from Fourier's, which keep an imaginary part.
FOURIER_MIXED = frozenset({FOURIER_WALSH, FOURIER_HAAR})

#: The mixed transforms of a real spectrum folded into Fourier's, one number a coefficient.
PLAIN_MIXED = frozenset({FOURIER_WALSH, FOURIER_HAAR, WALSH_HAAR})

#: The mixed transforms of a cosine's mirrored spectrum.
COS_MIXED = frozenset({COS_WALSH, COS_HAAR})

#: The mixed transforms of a sine's mirrored spectrum.
SIN_MIXED = frozenset({SIN_WALSH, SIN_HAAR})

#: ROOT's ``pi = 3.14159265358979323846``, which is the double nearest pi.
PI = 3.14159265358979323846

#: ``TMath::Sqrt(2.0)``, and the ``1 / TMath::Sqrt(2.0)`` of the butterflies.
SQRT2 = math.sqrt(2.0)
HALF_SQRT2 = 1 / math.sqrt(2.0)


def stages(num: int) -> int:
    """How many times ``num`` halves to one: ROOT's ``iter``, the log2 of a power of two."""
    count = 0
    while num > 1:
        count += 1
        num //= 2
    return count


def power2(exponent: int) -> int:
    """``(Int_t) TMath::Power(2, exponent)``: a power of two, a fraction truncated to 0."""
    return int(2.0**exponent)
