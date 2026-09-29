"""References ROOT printed on one machine: to the bit there, to a stated tolerance elsewhere.

ROOT 6.40.04 printed the RooFit references on macOS x86-64, so its
``std::exp``, ``std::cos``, ``std::erfc`` were Apple's x86-64 libm. xrdroot
calls the same C library functions ROOT calls (:mod:`xrdroot.random.libm`),
so on such a machine its numbers are ROOT's to the last bit, and the tests
say so. Other C libraries round some values differently in the last place -
Apple's x86-64 ``cos(0.8999999999999999)`` is ``0.6216099682706646``, where
glibc's, correctly rounded, is ``...645`` - and ROOT itself prints those
other bits there. Minuit2 is compiled code too: built for arm64, its
compiler fuses multiplies and adds that x86-64 rounds apart, so iminuit's
MIGRAD takes other steps from the same likelihood. So off that machine a
reference is held to a tolerance, each justified where it is used by what
lies between the library call and the number: a sum that cancels, an
adaptive rule that cuts a region more or fewer, a minimiser that stops a
step earlier or later.
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Any

import pytest

__all__ = ["FIT_REL", "PROFILE_REL", "ROOTS_MACHINE", "roots"]

#: Whether this is the kind of machine the references were printed on - which
#: ``XRDROOT_TEST_OTHER_MACHINE=1`` denies, to try the tolerances where it is.
ROOTS_MACHINE = (
    sys.platform == "darwin"
    and platform.machine() == "x86_64"
    and not os.environ.get("XRDROOT_TEST_OTHER_MACHINE")
)


#: How close a fitted value or HESSE error is to ROOT's: 1e-8 on ROOT's machine, and 1e-7
#: elsewhere. HESSE's errors are second differences over steps Minuit2 sizes from the
#: likelihood's precision, so each carries a relative error of about sqrt(eps) = 1.5e-8, and
#: a likelihood an ulp away - or Minuit2's own arithmetic fused differently - moves it by
#: that: the fits seen off ROOT's machine differ by up to 1.8e-8.
FIT_REL = 1e-8 if ROOTS_MACHINE else 1e-7

#: How close a number made from a minimised likelihood - the minimum itself, a profile's
#: value, ``q = 2 ΔNLL`` and the p-values, CLs and limits made from it - is to ROOT's off
#: ROOT's machine. There the likelihood is a few ulps away (glibc's ``lgamma(9)`` and ``erf``
#: round otherwise than Apple's), so MIGRAD walks another path and stops at another point
#: within its EDM of the minimum - 3e-13 on the counting model - and the minimum's value, flat
#: to first order, moves by about that: the p-values and limits seen move by up to 1e-11 of
#: themselves, the profile's value by 1e-12. 1e-9 leaves a hundredfold margin, and is still
#: a millionth of the smallest change a formula or a fit gone wrong would make.
PROFILE_REL = 1e-9


def roots(expected: Any, **tolerance: float) -> Any:
    """``expected`` itself on ROOT's machine, else ``pytest.approx(expected, **tolerance)``."""
    return expected if ROOTS_MACHINE else pytest.approx(expected, **tolerance)
